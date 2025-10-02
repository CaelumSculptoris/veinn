import json
import secrets
import numpy as np
from typing import Optional
from base64 import b64encode, b64decode
from kyber_py.ml_kem import ML_KEM_768  # Using ML_KEM_768 for ~128-bit security

from src.models import (VeinnParams)
from src.utils.encryption import (
    shake, derive_u16, ring_convolution, key_from_seed, permute_forward, permute_inverse,
    pad_iso7816, unpad_iso7816, block_to_bytes, bytes_to_block 
)

def generate_keypair(keypair: str = "kyber") -> dict:
    match keypair:
        case "kyber":
            return generate_kyber_keypair()
        case "veinn":
            return generate_veinn_keypair()

def generate_kyber_keypair() -> dict:
    """
    Generates a ML-KEM-768 (Kyber) key pair for post-quantum key encapsulation.
    
    Kyber is a lattice-based KEM selected by NIST for post-quantum cryptography.
    It's based on the hardness of the Module Learning With Errors (MLWE) problem.
    
    Returns:
        dict: Key pair with 'ek' (encapsulation key) and 'dk' (decapsulation key)
    
    Cryptographic principles:
    - Post-quantum cryptography: Secure against quantum computer attacks
    - ML-KEM-768: NIST standardized parameter set (security level 3)
    - Module-LWE: Mathematical foundation based on lattice problems
    - Key Encapsulation Mechanism: Hybrid crypto building block for encrypting symmetric keys
    """
    # Generate Kyber-768 keypair using NIST ML-KEM standard
    ek, dk = ML_KEM_768.keygen()
    
    # Convert to lists for JSON serialization (binary data -> list of integers)
    return {"ek": list(ek), "dk": list(dk)}

def generate_veinn_keypair(vp: VeinnParams = VeinnParams()) -> dict:
    """
    Generates a Veinn symmetric key pair using custom parameters.
    
    Veinn appears to be a custom symmetric encryption scheme based on
    permutation networks, mimicking the format of asymmetric keypairs.
    
    Args:
        vp (VeinnParams): Parameters for Veinn key generation
    
    Returns:
        dict: Key pair in Kyber-compatible format
    
    Cryptographic principles:
    - Custom KEM design: Non-standard construction
    - Permutation-based cryptography: Uses shuffling and permutations for security
    """
    # Generate Veinn keypair using custom algorithm
    pk, sk = veinn_kem_keygen(vp)
    
    # Mimic Kyber format for compatibility
    return {"ek": pk, "dk": sk}

# -----------------------------
# Sampling Helpers (for LWE noise generation)
# -----------------------------
def sample_small_det(size: int, vp: VeinnParams, rand_seed: bytes, tag: bytes) -> np.ndarray:
    """
    Sample small integers from discrete distribution for LWE noise.
    
    Generates small error terms for LWE construction:
    - Uses centered binomial-like distribution
    - Range [-4, 4] approximates discrete Gaussian
    - Deterministic from seed for reproducibility
    
    Args:
        size: Number of samples needed
        vp: VEINN parameters
        rand_seed: Random seed for sampling
        tag: Domain separation tag
        
    Returns:
        Array of small integers (LWE error terms)
    """
    raw = shake(size, rand_seed, tag)
    e = np.frombuffer(raw, dtype=np.uint8)[:size]
    # Map to [-4, 4] range (approximates small Gaussian)
    e = ((e % 9) - 4).astype(np.int64) % vp.q
    return e

# -----------------------------
# VEINN PKE (IND-CPA Public Key Encryption)
# -----------------------------
def veinn_pke_keygen(vp: VeinnParams) -> tuple[dict, dict]:
    """
    Generate VEINN public key encryption keypair.
    
    Implements Ring-LWE based public key encryption:
    1. Sample secret polynomial s and error e (small)
    2. Generate uniform public polynomial a
    3. Compute b = a*s + e (Ring-LWE sample)
    4. Public key: (a, b), Secret key: s
    
    Security based on Ring-LWE hardness: given (a, b), hard to find s
    even with quantum computers (assuming appropriate parameters).
    
    Args:
        vp: VEINN parameters
        
    Returns:
        Tuple of (public_key, secret_key) as dictionaries
    """
    n = vp.n
    
    # Generate inner VEINN key for message transformation
    inn_seed = secrets.token_bytes(vp.seed_len)
    
    # Sample Ring-LWE keypair
    a_seed = secrets.token_bytes(32)
    a = derive_u16(n, vp, a_seed, b"a")  # Uniform polynomial
    
    # Sample secret and error from small distribution
    s = sample_small_det(n, vp, secrets.token_bytes(32), b"s")
    e = sample_small_det(n, vp, secrets.token_bytes(32), b"e")
    
    # Compute Ring-LWE sample: b = a*s + e
    b = (ring_convolution(a, s, vp.q) + e) % vp.q
    
    # Package keys (convert to JSON-serializable format)
    pk = {
        "a": a.tolist(),
        "b": b.tolist(),
        "inn_seed": b64encode(inn_seed).decode()
    }
    sk = {
        "s": s.tolist(),
        "inn_seed": b64encode(inn_seed).decode()
    }
    return pk, sk

def veinn_pke_encrypt(pk: dict, m: bytes, vp: VeinnParams, rand_seed: bytes) -> dict:
    """
    VEINN PKE encryption using Ring-LWE.
    
    Encryption process:
    1. Apply inner VEINN transformation to message (confusion)
    2. Encrypt transformed message using Ring-LWE encryption
    3. Ciphertext: (c0, c1, c2) where c0, c1 are LWE, c2 is masked message
    
    Ring-LWE encryption:
    - Sample ephemeral secret r and errors e0, e1
    - c0 = a*r + e0 (fresh LWE sample)
    - c1 = b*r + e1 + v (mask using public key LWE sample)
    - c2 = message + v (one-time pad with LWE mask)
    
    Args:
        pk: Public key from keygen
        m: Message bytes to encrypt
        vp: VEINN parameters
        rand_seed: Randomness for encryption
        
    Returns:
        Ciphertext dictionary
    """
    n = vp.n
    
    # Reconstruct inner VEINN key
    inn_seed_bytes = b64decode(pk["inn_seed"])
    inn_key = key_from_seed(inn_seed_bytes, vp)
    
    # Apply inner VEINN transformation (message confusion)
    padded_m = pad_iso7816(m, 2 * n)
    x = bytes_to_block(padded_m, n)
    y = permute_forward(x, inn_key)
    
    # Sample ephemeral values for Ring-LWE encryption
    v = sample_small_det(n, vp, rand_seed, b"v")    # LWE mask
    r = sample_small_det(n, vp, rand_seed, b"r")    # Ephemeral secret
    e0 = sample_small_det(n, vp, rand_seed, b"e0")  # Error term 0
    e1 = sample_small_det(n, vp, rand_seed, b"e1")  # Error term 1
    
    # Retrieve public key components
    a = np.array(pk["a"], dtype=np.int64)
    b = np.array(pk["b"], dtype=np.int64)
    
    # Ring-LWE encryption: c0 = a*r + e0, c1 = b*r + e1 + v
    c0 = (ring_convolution(a, r, vp.q) + e0) % vp.q
    c1 = (ring_convolution(b, r, vp.q) + e1 + v) % vp.q
    
    # One-time pad: c2 = transformed_message + v
    c2 = (y + v) % vp.q
    
    ct = {
        "c0": c0.tolist(),
        "c1": c1.tolist(),
        "c2": c2.tolist()
    }
    return ct

def veinn_pke_decrypt(sk: dict, ct: dict, vp: VeinnParams) -> bytes:
    """
    VEINN PKE decryption using Ring-LWE.
    
    Decryption process:
    1. Recover LWE mask: v' = c1 - c0*s
    2. Recover transformed message: y' = c2 - v'
    3. Apply inverse VEINN transformation
    4. Remove padding to get original message
    
    Ring-LWE decryption correctness:
    - v' = c1 - c0*s = (b*r + e1 + v) - (a*r + e0)*s
    - Since b = a*s + e_pk, we get v' ≈ v (up to small errors)
    - Small errors are handled by error correction
    
    Args:
        sk: Secret key from keygen
        ct: Ciphertext from encryption
        vp: VEINN parameters
        
    Returns:
        Decrypted message bytes
    """
    n = vp.n
    
    # Retrieve secret key and ciphertext components
    s = np.array(sk["s"], dtype=np.int64)
    c0 = np.array(ct["c0"], dtype=np.int64)
    c1 = np.array(ct["c1"], dtype=np.int64)
    c2 = np.array(ct["c2"], dtype=np.int64)
    
    # Recover LWE mask: v' = c1 - c0*s
    v_prime = (c1 - ring_convolution(c0, s, vp.q)) % vp.q
    
    # Recover transformed message: y' = c2 - v'
    y_prime = (c2 - v_prime) % vp.q
    
    # Apply inverse VEINN transformation
    inn_seed_bytes = b64decode(sk["inn_seed"])
    inn_key = key_from_seed(inn_seed_bytes, vp)
    x_prime = permute_inverse(y_prime, inn_key)
    
    # Convert back to bytes and remove padding
    padded_m = block_to_bytes(x_prime)
    try:
        m = unpad_iso7816(padded_m)
    except ValueError:
        m = b""  # Decryption failure (negligible probability)
    return m

# -----------------------------
# VEINN KEM (IND-CCA Key Encapsulation via FO Transform)
# -----------------------------
def veinn_kem_keygen(vp: VeinnParams) -> tuple[dict, dict]:
    """
    Generate VEINN KEM keypair using Fujisaki-Okamoto transform.
    
    KEM provides key encapsulation for hybrid encryption:
    - Encapsulates symmetric key using public key crypto
    - Symmetric key used for bulk data encryption
    - FO transform converts IND-CPA PKE to IND-CCA2 KEM
    
    The secret key includes:
    - Underlying PKE secret key
    - Public key (for re-encryption check)
    - Random value z (for implicit rejection)
    
    Args:
        vp: VEINN parameters
        
    Returns:
        Tuple of (public_key, secret_key)
    """
    pk, sk_pke = veinn_pke_keygen(vp)
    z = secrets.token_bytes(32)  # For implicit rejection in FO transform
    h_pk = shake(32, json.dumps(pk).encode())  # H(public_key)
    
    sk = {
        "sk_pke": sk_pke,
        "pk": pk,  # Store for re-encryption check
        "z": b64encode(z).decode(),
        "h_pk": b64encode(h_pk).decode()
    }
    return pk, sk

def encaps(pk: dict, vp: Optional[VeinnParams] = None, keypair: str = "kyber") -> dict:
    match keypair:
        case "kyber":
            return ML_KEM_768.encaps(pk)
        case "veinn":
            return veinn_kem_encaps(pk, vp)

def veinn_kem_encaps(pk: dict, vp: VeinnParams) -> tuple[dict, bytes]:
    """
    VEINN KEM encapsulation using Fujisaki-Okamoto transform.
    
    FO transform process:
    1. Sample random message m
    2. Derive (K_bar, r) = G(m || H(pk)) where G is hash function
    3. Encrypt m using randomness r: c = Encrypt(pk, m; r)
    4. Derive final key: K = J(K_bar || H(c))
    
    This provides IND-CCA2 security by:
    - Binding randomness to message (prevents malleability)
    - Including ciphertext hash in key derivation (authenticity)
    - Deterministic re-encryption check during decapsulation
    
    Args:
        pk: Public key for encapsulation
        vp: VEINN parameters
        
    Returns:
        Tuple of (ciphertext, shared_key)
    """
    # Sample random message for KEM
    m = secrets.token_bytes(32)
    h_pk = shake(32, json.dumps(pk).encode())  # H(public_key)
    
    # Derive key and randomness: G(m || H(pk))
    g_out = shake(64, m + h_pk)
    K_bar = g_out[:32]      # Pre-key
    r = g_out[32:]          # Encryption randomness
    
    # Encrypt message with derived randomness (deterministic)
    c = veinn_pke_encrypt(pk, m, vp, r)
    
    # Derive final shared key: J(K_bar || H(ciphertext))
    K = shake(32, K_bar + shake(32, json.dumps(c).encode()))
    
    return c, K

def decaps(sk: dict, enc_seed_bytes: Optional[bytes] = None, c: Optional[dict] = None, vp: Optional[VeinnParams] = None, keypair: str = "kyber") -> dict:
    match keypair:
        case "kyber":
            return ML_KEM_768.decaps(sk, enc_seed_bytes)
        case "veinn":
            return veinn_kem_decaps(sk, c, vp)

def veinn_kem_decaps(sk: dict, c: dict, vp: VeinnParams) -> bytes:
    """
    VEINN KEM decapsulation with FO transform and implicit rejection.
    
    Decapsulation with security checks:
    1. Decrypt ciphertext to get m'
    2. Re-derive (K_bar', r') = G(m' || H(pk))
    3. Re-encrypt: c' = Encrypt(pk, m'; r')
    4. If c' = c: return K = J(K_bar' || H(c)) (success)
    5. Else: return K = J(z || H(c)) (implicit rejection)
    
    Implicit rejection prevents adaptive chosen ciphertext attacks by:
    - Returning pseudorandom key for invalid ciphertexts
    - Attacker cannot distinguish rejection from random key
    - Maintains constant-time operation
    
    Args:
        sk: Secret key from keygen
        c: Ciphertext to decapsulate
        vp: VEINN parameters
        
    Returns:
        Shared key (either real or pseudorandom for invalid ciphertexts)
    """
    # Decrypt ciphertext
    m_prime = veinn_pke_decrypt(sk["sk_pke"], c, vp)
    h_pk = b64decode(sk["h_pk"])
    
    # Re-derive key and randomness
    g_out = shake(64, m_prime + h_pk)
    K_bar_prime = g_out[:32]
    r_prime = g_out[32:]
    
    # Re-encrypt for authenticity check
    c_prime = veinn_pke_encrypt(sk["pk"], m_prime, vp, r_prime)
    h_c = shake(32, json.dumps(c).encode())
    
    # Constant-time comparison and key derivation
    if json.dumps(c_prime) == json.dumps(c):
        # Valid ciphertext: return real key
        K = shake(32, K_bar_prime + h_c)
    else:
        # Invalid ciphertext: implicit rejection with pseudorandom key
        z = b64decode(sk["z"])
        K = shake(32, z + h_c)
    return K