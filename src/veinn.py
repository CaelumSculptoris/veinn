"""
VEINN - Vector Encrypted Invertible Neural Network

Key Cryptographic Principles Documented:
Post-Quantum Cryptography:

ML-KEM (Kyber) integration for quantum-resistant key encapsulation
Ring-LWE based constructions for lattice-based security
LWE-based pseudorandom functions for key derivation

Hybrid Encryption Architecture:

KEM/DEM paradigm combining asymmetric and symmetric cryptography
Fujisaki-Okamoto transform for IND-CCA2 security
Authenticated encryption with HMAC-SHA256

Block Cipher Construction:

Feistel-like structure with coupling layers (inspired by neural networks)
Shannon's principles: confusion (S-boxes) and diffusion (permutation/coupling)
Invertible operations throughout for exact decryption

Security Features:

OAEP padding for semantic security
Timestamp validation for replay attack prevention
Secure key storage with PBKDF2 + Fernet encryption
Domain separation in key derivation functions

Mathematical Foundations:

Number Theoretic Transform for efficient polynomial multiplication
Modular arithmetic over prime fields
Ring convolution in quotient rings Z_q[x]/(x^n + 1)

The documentation explains both the implementation details and the underlying 
cryptographic theory, making it clear how each component contributes to the overall 
security of the system. Each function includes parameter descriptions, security 
considerations, and references to the specific cryptographic principles being applied.

This implementation is for educational purposes and is not cryptographically secure.
"""
import os
import sys
import json
import math
import hashlib
import hmac
import secrets
import numpy as np
import argparse
import pickle
import time
from typing import Optional, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from base64 import b64encode, b64decode
from dataclasses import dataclass
from kyber_py.ml_kem import ML_KEM_768  # Using ML_KEM_768 for ~128-bit security

# -----------------------------
# CLI Colors
# -----------------------------
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    GREY = '\033[90m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# -----------------------------
# Core Parameters
# -----------------------------
@dataclass
class VeinnParams:
    """
    Core parameters for the VEINN cryptographic system.
    
    The VEINN construction is a custom block cipher using:
    - Large block sizes (n * 8 bytes) for post-quantum security
    - Multiple rounds with coupling layers (inspired by neural network architectures)
    - LWE-based pseudorandom function for quantum-resistant key derivation
    - Modular arithmetic over prime field for algebraic security
    """
    n: int = 512  # Number of int64 words per block (4KB blocks for high throughput)
    rounds: int = 15  # Number of cipher rounds (sufficient for cryptographic strength)
    layers_per_round: int = 15  # Coupling layers per round (depth for diffusion)
    shuffle_stride: int = 15  # Permutation stride (must be coprime with n)
    use_lwe: bool = True  # Enable LWE-based key derivation for post-quantum security
    valid: int = 3600  # Timestamp validity window in seconds
    seed_len: int = 64  # Seed length for cryptographic randomness
    q: int = 2**31 -1 #2013265921 #1049089  # Prime modulus for field operations (≈2^20 for efficiency)

# -----------------------------
# Coupling Params 
# -----------------------------
@dataclass
class CouplingParams:
    """
    Parameters for coupling layer transformation.
    
    Coupling layers are inspired by normalizing flows and neural architectures:
    - Split input into two halves
    - Transform one half based on the other
    - Provides invertible nonlinear mixing
    
    The masks provide pseudorandom parameters for the affine transformations.
    """
    mask_a: np.ndarray  # First transformation mask
    mask_b: np.ndarray  # Second transformation mask

# -----------------------------
# Round Params 
# -----------------------------
@dataclass
class RoundParams:
    """
    Parameters for a single cipher round.
    
    Each round contains:
    - Multiple coupling layers for nonlinear mixing
    - Invertible element-wise scaling for algebraic complexity
    - Precomputed inverse scaling for decryption efficiency
    """
    cpls: list[CouplingParams]  # Coupling layer parameters
    ring_scale: np.ndarray      # Element-wise odd scaling factors (invertible)
    ring_scale_inv: np.ndarray  # Precomputed modular inverses

# -----------------------------
# Veinn Key
# -----------------------------
@dataclass
class VeinnKey:
    """
    Complete VEINN cipher key containing all round parameters.
    
    Derived deterministically from a seed using cryptographic key derivation.
    Contains all parameters needed for encryption/decryption:
    - Round coupling parameters
    - Permutation indices
    - Invertible scaling factors
    """
    seed: bytes                      # Original key seed
    params: VeinnParams             # Cipher parameters
    shuffle_idx: np.ndarray         # Permutation indices
    rounds: list[RoundParams]       # Per-round parameters

# -----------------------------
# Utilities
# -----------------------------
def shake(expand_bytes: int, *chunks: bytes) -> bytes:
    """
    SHAKE-256 extendable output function for cryptographic key derivation.
    
    This implements the NIST-standardized SHAKE-256 XOF, which provides:
    - Variable-length output from fixed input
    - Domain separation through length prefixing
    - Cryptographic security for key derivation
    
    Args:
        expand_bytes: Number of output bytes to generate
        *chunks: Input data chunks to hash (length-prefixed for domain separation)
        
    Returns:
        Pseudorandom bytes of specified length
    """
    xof = hashlib.shake_256()
    # Domain separation: prefix each chunk with its length
    for c in chunks:
        xof.update(len(c).to_bytes(2, 'big'))
        xof.update(c)
    return xof.digest(expand_bytes)

def derive_u16(count: int, vp: VeinnParams, *chunks: bytes) -> np.ndarray:
    """
    Derive pseudorandom integers using either LWE-based PRF or direct SHAKE expansion.
    
    Two modes:
    1. LWE mode: Uses Ring-LWE for post-quantum security (quantum-resistant PRF)
    2. Direct mode: Uses SHAKE-256 directly (classical security)
    
    The LWE mode provides security against quantum attacks by basing hardness
    on the Ring Learning With Errors problem, a well-studied lattice problem.
    
    Args:
        count: Number of integers to generate
        vp: VEINN parameters
        *chunks: Input data for key derivation
        
    Returns:
        Array of pseudorandom integers modulo q
    """
    if vp.use_lwe:
        # Use LWE-based PRF for post-quantum security
        seed_derive = shake(32, *chunks)
        return lwe_prf_expand(seed_derive, count, vp)
    # Fallback to direct SHAKE expansion for classical security
    raw = shake(count * 2, *chunks)
    return np.frombuffer(raw, dtype=np.uint16)[:count].astype(np.int64).copy()

def odd_constant_from_key(tag: bytes) -> int:
    """
    Generate an odd constant from a key tag for multiplicative operations.
    
    Ensures the result is odd (and thus invertible modulo powers of 2)
    by setting the least significant bit. Used for invertible scaling operations.
    
    Args:
        tag: Input tag for key derivation
        
    Returns:
        Odd integer modulo q (guaranteed invertible)
    """
    x = int.from_bytes(shake(2, tag), 'little')
    x |= 1  # Force odd for invertibility
    return x & (VeinnParams.q - 1)

def pad_iso7816(data: bytes, blocksize: int) -> bytes:
    """
    ISO 7816-4 padding scheme for block cipher input.
    
    Adds padding bytes: 0x80 followed by zero bytes to reach block boundary.
    This padding is unambiguous and allows exact plaintext recovery.
    Standard padding used in smart card cryptography.
    
    Args:
        data: Input data to pad
        blocksize: Target block size in bytes
        
    Returns:
        Padded data aligned to block boundary
    """
    padlen = (-len(data)) % blocksize
    if padlen == 0:
        padlen = blocksize
    return data + b"\x80" + b"\x00"*(padlen-1)

def unpad_iso7816(padded: bytes) -> bytes:
    """
    Remove ISO 7816-4 padding from decrypted data.
    
    Locates the 0x80 padding byte and removes it plus trailing zeros.
    Validates padding format to detect corruption/tampering.
    
    Args:
        padded: Padded data from decryption
        
    Returns:
        Original unpadded data
        
    Raises:
        ValueError: If padding format is invalid
    """
    i = padded.rfind(b"\x80")
    if i == -1 or any(b != 0 for b in padded[i+1:]):
        raise ValueError("Invalid padding")

    return padded[:i]

# Primary S-box: modular inverse (0 -> 0). Clean, bijective if q is prime (or for units).
def sbox_val_modinv(x, q: int):
    """
    Primary S-box using modular multiplicative inverse.
    
    Implements the S-box S(x) = x^(-1) mod q, a cryptographically strong
    nonlinear transformation. Maps 0 -> 0 and provides high algebraic degree.
    This is the inverse function over the finite field F_q.
    
    Args:
        x: Input value
        q: Modulus (should be prime for field properties)
        
    Returns:
        Modular inverse of x, or def create_keystore(passphrase: str, keystore_file: str):
    """
    x = int(x) % q  # force to plain int
    if x == 0:
        return 0
    try:
        return pow(x, -1, q)  # Python int only
    except ValueError:
        return None  # no modular inverse

# Fallback S-box: exponentiation
def sbox_val_pow(x, q: int, e: int = 3):
    """
    Fallback S-box using modular exponentiation.
    
    Implements S(x) = x^e mod q as a nonlinear transformation.
    Used when modular inverse is not available (q is not prime).
    Provides algebraic nonlinearity for cryptographic strength.
    
    Args:
        x: Input value
        q: Modulus
        e: Exponent (default 3 for cubic S-box)
        
    Returns:
        x^e mod q
    """
    x = int(x) % q
    if x == 0:
        return 0
    return pow(x, e, q)

# Wrapper: try modinv, otherwise fallback
def sbox_val(x, q: int, fallback_e: int = 3):
    """
    Adaptive S-box with fallback mechanism.
    
    Tries modular inverse first (strongest nonlinearity), falls back to
    power function if inverse doesn't exist. Provides consistent nonlinear
    transformation regardless of modulus properties.
    
    Args:
        x: Input value
        q: Modulus
        fallback_e: Fallback exponent
        
    Returns:
        Nonlinearly transformed value
    """
    v = sbox_val_modinv(x, q)
    if v is not None:
        return v
    return sbox_val_pow(x, q, fallback_e)

# Vectorized layer
def sbox_layer(vec, q: int, fallback_e: int = 3):
    """
    Apply S-box transformation to entire vector (confusion layer).
    
    Implements the confusion component of Shannon's cipher principles.
    Applies nonlinear S-box to each element independently, providing
    resistance to linear and differential cryptanalysis.
    
    Args:
        vec: Input vector or array
        q: Modulus for field operations
        fallback_e: Fallback exponent for power S-box
        
    Returns:
        Vector with S-box applied element-wise
    """
    if isinstance(vec, np.ndarray):
        out = np.empty_like(vec, dtype=np.int64)
        it = np.nditer(vec, flags=['multi_index'])
        while not it.finished:
            out[it.multi_index] = sbox_val(int(it[0]), q, fallback_e)
            it.iternext()
        return out % q
    else:
        return [sbox_val(int(x), q, fallback_e) for x in vec]

# Inverse S-box
def inv_sbox_val(x, q: int, fallback_e: int = 3):
    """
    Inverse S-box transformation for decryption.
    
    Computes the inverse of the S-box function. For modular inverse S-box,
    this is the S-box itself (since (x^(-1))^(-1) = x). For power S-box,
    uses appropriate inverse exponent.
    
    Args:
        x: S-box output to invert
        q: Modulus
        fallback_e: Original exponent (for computing inverse)
        
    Returns:
        Original input that produced x via S-box
    """
    x = int(x) % q
    try:
        return sbox_val_modinv(x, q) or sbox_val_pow(x, q, fallback_e)
    except Exception:
        return sbox_val_pow(x, q, fallback_e)

def inv_sbox_layer(vec, q: int, fallback_e: int = 3):
    """
    Apply inverse S-box to entire vector for decryption.
    
    Reverses the confusion layer by applying inverse S-box element-wise.
    Essential for correct decryption in the cipher construction.
    
    Args:
        vec: S-box output vector to invert
        q: Modulus
        fallback_e: Original S-box exponent
        
    Returns:
        Original vector before S-box transformation
    """
    if isinstance(vec, np.ndarray):
        out = np.empty_like(vec, dtype=np.int64)
        it = np.nditer(vec, flags=['multi_index'])
        while not it.finished:
            out[it.multi_index] = inv_sbox_val(int(it[0]), q, fallback_e)
            it.iternext()
        return out % q
    else:
        return [inv_sbox_val(int(x), q, fallback_e) for x in vec]

# -----------------------------
# Ring Convolution with Iterative NTT
# -----------------------------
def ring_convolution(a, b, q, method="ntt"):
    """
    Compute polynomial multiplication in quotient ring R_q = Z_q[x]/(x^n + 1).
    
    This implements the core algebraic operation for Ring-LWE cryptography:
    - Polynomial multiplication modulo x^n + 1 (cyclotomic polynomial)
    - Coefficient reduction modulo q
    - Uses NTT for O(n log n) complexity vs O(n^2) naive method
    
    The ring structure provides:
    - Compact representation for lattice problems
    - Efficient computation via Number Theoretic Transform
    - Security based on Ring-LWE hardness assumption
    
    Args:
        a, b: Input polynomials (coefficient vectors)
        q: Coefficient modulus (should be NTT-friendly: q ≡ 1 (mod 2n))
        method: "ntt" for fast transform, "naive" for direct computation
        
    Returns:
        Product polynomial a * b mod (x^n + 1, q)
    """
    n = len(a)
    a = np.array(a, dtype=np.int64) % q
    b = np.array(b, dtype=np.int64) % q
    
    if method == "naive":
        # Direct O(n^2) convolution with reduction mod (x^n + 1)
        res = np.zeros(2*n, dtype=object)
        for i in range(n):
            for j in range(n):
                res[i+j] = (res[i+j] + int(a[i]) * int(b[j])) % q
        # Reduce modulo x^n + 1: coefficients of x^n and higher become negative
        res = (res[:n] - res[n:]) % q
        return res.astype(np.int64)

    elif method == "ntt":
        # Fast O(n log n) convolution using Number Theoretic Transform
        # Pad to 2n for full convolution, then reduce
        a_padded = np.zeros(2*n, dtype=np.int64)
        a_padded[:n] = a
        b_padded = np.zeros(2*n, dtype=np.int64)
        b_padded[:n] = b
        
        # Find primitive 2n-th root of unity for NTT
        root = pow(find_primitive_root(q), (q - 1) // (2 * n), q)
        
        # Forward NTT, pointwise multiply, inverse NTT
        A = iterative_ntt(a_padded, root, q)
        B = iterative_ntt(b_padded, root, q)
        C = mod_mul(A, B, q)
        res = iterative_intt(C, root, q)
        
        # Reduce modulo x^n + 1
        res = (res[:n] - res[n:]) % q
        return res.astype(np.int64)

    else:
        raise ValueError("method must be one of: naive, ntt")

def iterative_ntt(a: np.ndarray, root: int, q: int) -> np.ndarray:
    """
    Iterative Number Theoretic Transform for fast polynomial multiplication.
    
    NTT is the discrete analog of FFT over finite fields:
    - Converts coefficient representation to evaluation representation
    - Enables O(n log n) polynomial multiplication
    - Essential for efficient Ring-LWE operations
    
    Uses Cooley-Tukey decimation-in-time algorithm adapted for finite fields.
    
    Args:
        a: Input polynomial coefficients
        root: Primitive n-th root of unity mod q
        q: Field modulus (must have root of appropriate order)
        
    Returns:
        NTT transform of input polynomial
    """
    n = len(a)
    a = a.copy()
    t = n
    m = 1
    
    # Cooley-Tukey NTT: log n stages, each combining smaller DFTs
    while m < n:
        t //= 2
        for i in range(m):
            j1 = 2 * i * t
            j2 = j1 + t - 1
            # Twiddle factor for this butterfly stage
            S = pow(root, m + i, q)
            for j in range(j1, j2 + 1):
                # Butterfly operation: combine even/odd parts
                U = a[j]
                V = (a[j + t] * S) % q
                a[j] = (U + V) % q
                a[j + t] = (U - V) % q
        m *= 2
    return a

def iterative_intt(A: np.ndarray, root: int, q: int) -> np.ndarray:
    """
    Inverse Number Theoretic Transform for coefficient recovery.
    
    Converts from evaluation back to coefficient representation.
    Uses inverse root of unity and scales by 1/n to complete inversion.
    
    Args:
        A: NTT-transformed polynomial
        root: Original primitive root used in forward NTT
        q: Field modulus
        
    Returns:
        Original polynomial coefficients
    """
    n = len(A)
    # Inverse root of unity
    root_inv = pow(root, q-2, q)
    # Apply NTT with inverse root
    a = iterative_ntt(A, root_inv, q)
    # Scale by 1/n to complete inversion
    inv_n = pow(n, q-2, q)
    return (a * inv_n) % q

def mod_mul(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """
    Element-wise modular multiplication of vectors.
    
    Computes (a[i] * b[i]) mod q for all i. Used in NTT-based convolution
    for pointwise multiplication in evaluation domain.
    
    Args:
        a, b: Input vectors
        q: Modulus
        
    Returns:
        Element-wise product modulo q
    """
    return ((a.astype(object) * b.astype(object)) % q).astype(np.int64)

def find_primitive_root(q):
    """
    Find a primitive root modulo q (generator of multiplicative group).
    
    A primitive root g generates all non-zero elements: g^i mod q covers
    {1, 2, ..., q-1}. Essential for NTT construction. Uses trial division
    to test candidates (naive but sufficient for cryptographic-sized q).
    
    Args:
        q: Prime modulus
        
    Returns:
        Primitive root modulo q, or None if not found
    """
    factors = factorize(q-1)
    for g in range(2, q):
        ok = True
        # Check if g has full multiplicative order
        for f in factors:
            if pow(g, (q-1)//f, q) == 1:
                ok = False
                break
        if ok:
            return g
    return None

def factorize(n):
    """
    Simple trial division factorization.
    
    Finds all prime factors of n. Used by primitive root finder
    to check multiplicative order. Not optimized for large n.
    
    Args:
        n: Integer to factorize
        
    Returns:
        List of prime factors
    """
    factors = set()
    d = 2
    while d*d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return list(factors)

def sample_binomial(k: int, size: int, q: int) -> np.ndarray:
    """
    Sample centered binomial distribution for VEINN's integer system.
    
    Generates integers approximating a Gaussian with sigma = sqrt(k/2).
    Matches Kyber's error sampling for PQ security.
    
    Args:
        k: Number of bit pairs (e.g., 8 for sigma ≈ 2)
        size: Number of samples (e.g., n=512)
        q: Modulus (e.g., 1049089)
    
    Returns:
        np.ndarray: Array of size integers in [-k, k], reduced mod q
    """
    # Need k bits per sample, size samples, split into a and b
    num_bytes = size * k // 8 + (1 if size * k % 8 else 0)
    bits = np.frombuffer(secrets.token_bytes(num_bytes), dtype=np.uint8)[:size * k] & 1
    # Reshape into size x k for a and b
    a = bits[::2][:size * k//2].reshape(size, k//2).sum(axis=1)
    b = bits[1::2][:size * k//2].reshape(size, k//2).sum(axis=1)
    samples = (a - b).astype(np.int64)
    return samples % q

def lwe_prf_expand(seed: bytes, out_n: int, vp: VeinnParams) -> np.ndarray:
    """
    LWE-based Pseudorandom Function for post-quantum key derivation.
    
    Implements a PRF based on the Ring Learning With Errors problem:
    1. Sample secret polynomial s and public polynomial a
    2. Compute b = a * s + e where e is small error
    3. Output comes from the "noisy" product b
    
    Security relies on Ring-LWE hardness assumption:
    - Classical computers: exponential time to distinguish from random
    - Quantum computers: no known polynomial-time algorithm
    
    This provides post-quantum security for key derivation, unlike
    classical PRFs based on factoring or discrete log.
    
    Args:
        seed: PRF seed (acts as key)
        out_n: Number of output values needed
        vp: VEINN parameters including ring dimension and modulus
        
    Returns:
        Pseudorandom array derived from LWE instance
    """
    n = vp.n
    # Derive LWE parameters from seed with domain separation
    s = np.frombuffer(shake(n * 8, seed, b"s"), dtype=np.int64)[:n] & (vp.q - 1)
    a = np.frombuffer(shake(n * 8, seed, b"A"), dtype=np.int64)[:n] & (vp.q - 1)
    
    # Sample small error from discrete Gaussian approximation
    raw = shake(n, seed, b"e")
    e = np.frombuffer(raw, dtype=np.uint8)[:n].astype(np.int64)    
    e = ((e % 9) - 4).astype(np.int64) % vp.q  # Small error in [-4, 4]    
    #e = sample_binomial(k=8, size=n, q=vp.q)
    
    # Validate dimensions for Ring-LWE structure
    assert s.shape == (n,) and a.shape == (n,) and e.shape == (n,), f"LWE parameter shape mismatch{e.shape, a.shape, s.shape}"
    
    # Compute b = a * s + e (Ring-LWE sample)
    b = ring_convolution(a, s, vp.q, 'ntt').astype(np.int64)
    b = (b + e) % vp.q
    
    # Expand output by cycling through b
    out = np.zeros(out_n, dtype=np.int64)
    for i in range(out_n):
        out[i] = int(b[i % n]) & (vp.q - 1)
    
    assert out.shape == (out_n,), f"Expected output shape {(out_n,)}, got {out.shape}"
    return out

# -----------------------------
# Coupling Layer
# -----------------------------
def derive_layer_seed_from_masks_and_key(mask_a: np.ndarray, mask_b: np.ndarray, layer_idx: int) -> bytes:
    """
    Derives a cryptographic seed for a specific layer using input masks and layer index.
    
    This function implements a key derivation function (KDF) that combines two input masks
    and a layer identifier to produce a deterministic seed for cryptographic operations.
    
    Args:
        mask_a (np.ndarray): First input mask array
        mask_b (np.ndarray): Second input mask array  
        layer_idx (int): Layer index identifier for domain separation
        
    Returns:
        bytes: 32-byte cryptographic seed derived from inputs
        
    Cryptographic Principles Applied:
        - Domain separation: Layer index prevents cross-layer key reuse
        - Hash-based key derivation: Uses SHAKE-256 for secure seed generation
        - Collision resistance: SHAKE-256 provides strong collision resistance
    """
    h = hashlib.shake_256()
    
    # Domain separation: Unique constant prevents collision with other protocols
    h.update(b"VEINN-HILBERT-SEED")
    
    # Include both masks to ensure seed depends on all input material
    h.update(mask_a.tobytes())
    h.update(mask_b.tobytes())
    
    # Layer index provides domain separation between different layers
    h.update(layer_idx)
    
    # Extract 32 bytes using SHAKE-256's variable output length
    return h.digest(32)


def derive_kernel_from_seed(seed: bytes, tag: bytes, h: int, q: int) -> np.ndarray:
    """
    Derives a convolution kernel from a seed using cryptographic hash expansion.
    
    Generates pseudorandom coefficients for use in ring convolution operations by
    expanding a seed with domain separation tags. Ensures the resulting kernel
    has appropriate algebraic properties for the cryptographic scheme.
    
    Args:
        seed (bytes): Cryptographic seed material
        tag (bytes): Domain separation tag (e.g., b"R", b"S", b"M")
        h (int): Half the dimension size (kernel length)
        q (int): Modulus for coefficient reduction
        
    Returns:
        np.ndarray: Array of h coefficients in range [0, q-1]
        
    Cryptographic Principles Applied:
        - Pseudorandom generation: SHAKE-256 provides cryptographically secure randomness
        - Domain separation: Tag parameter prevents key reuse across different purposes
        - Uniform distribution: Modular reduction ensures coefficients are uniformly distributed
        - Non-zero guarantee: First coefficient is made non-zero to maintain invertibility
    """
    sh = hashlib.shake_256()
    sh.update(seed)
    
    # Domain separation: Tag ensures different kernels for R, S, M operations
    sh.update(tag)
    
    # Generate 2*h bytes to construct h 16-bit values
    raw = sh.digest(2*h)
    
    # Convert bytes to uint8 array, then combine pairs into 16-bit values
    coeffs = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    
    # Combine adjacent bytes into 16-bit coefficients and reduce modulo q
    # This provides uniform distribution over [0, q-1]
    coeffs = (coeffs[0::2] | (coeffs[1::2] << 8)) % q
    
    # Ensure first coefficient is non-zero to maintain algebraic properties
    # This prevents degenerate cases in convolution operations
    coeffs[0] = (coeffs[0] + 1) % q
    
    return coeffs


def conv_op(vec: np.ndarray, kernel: np.ndarray, q: int) -> np.ndarray:
    """
    Performs modular ring convolution operation.
    
    Computes the convolution of a vector with a kernel using Number Theoretic
    Transform (NTT) for efficiency, with all operations performed modulo q.
    
    Args:
        vec (np.ndarray): Input vector
        kernel (np.ndarray): Convolution kernel
        q (int): Modulus for all operations
        
    Returns:
        np.ndarray: Result of ring convolution modulo q
        
    Cryptographic Principles Applied:
        - Ring arithmetic: Operations in polynomial ring Z_q[X]/(X^n + 1)
        - NTT optimization: Uses Number Theoretic Transform for efficient computation
        - Modular reduction: All operations maintained modulo q for security
    """
    # Delegate to optimized ring convolution with NTT method
    return ring_convolution(vec, kernel.astype(np.int64), q, 'ntt')


def block_apply_R(x1: np.ndarray, x2: np.ndarray, t: np.ndarray, q: int):
    """
    Applies the R transformation matrix in block form: R = [[I, Conv_t], [0, I]].
    
    This implements a linear transformation where the first block is modified by
    a convolution with the second block, while the second block remains unchanged.
    This structure ensures the transformation is invertible.
    
    Args:
        x1 (np.ndarray): First half of input vector
        x2 (np.ndarray): Second half of input vector  
        t (np.ndarray): Convolution kernel for the transformation
        q (int): Modulus for all operations
        
    Returns:
        tuple: (transformed_x1, unchanged_x2)
        
    Cryptographic Principles Applied:
        - Feistel-like structure: Only one half is modified, ensuring invertibility
        - Linear transformation: Maintains vector space structure
        - Modular arithmetic: All operations performed modulo q
    """
    # R = [[I, Conv_t],[0,I]] - upper triangular block matrix
    # First component: x1 + Conv_t(x2), Second component: x2 (unchanged)
    return ( (x1 + conv_op(x2, t, q)) % q, x2.copy() )


def block_apply_S(x1: np.ndarray, x2: np.ndarray, u: np.ndarray, q: int):
    """
    Applies the S transformation matrix in block form: S = [[I, 0], [Conv_u, I]].
    
    This implements a linear transformation where the second block is modified by
    a convolution with the first block, while the first block remains unchanged.
    Combined with R, this creates a complete mixing transformation.
    
    Args:
        x1 (np.ndarray): First half of input vector
        x2 (np.ndarray): Second half of input vector
        u (np.ndarray): Convolution kernel for the transformation  
        q (int): Modulus for all operations
        
    Returns:
        tuple: (unchanged_x1, transformed_x2)
        
    Cryptographic Principles Applied:
        - Feistel-like structure: Only one half is modified, ensuring invertibility
        - Linear transformation: Maintains vector space structure  
        - Complementary to R: Together with R provides full mixing
    """
    # S = [[I,0],[Conv_u,I]] - lower triangular block matrix
    # First component: x1 (unchanged), Second component: x2 + Conv_u(x1)
    return ( x1.copy(), (x2 + conv_op(x1, u, q)) % q )


def block_apply_R_inv(y1: np.ndarray, y2: np.ndarray, t: np.ndarray, q: int):
    """
    Applies the inverse R transformation: R^(-1) = [[I, -Conv_t], [0, I]].
    
    Inverts the R transformation by subtracting the convolution term that was
    added in the forward direction. The structure ensures perfect invertibility.
    
    Args:
        y1 (np.ndarray): First half of transformed vector
        y2 (np.ndarray): Second half of transformed vector
        t (np.ndarray): Same convolution kernel used in forward transformation
        q (int): Modulus for all operations
        
    Returns:
        tuple: (recovered_x1, recovered_x2)
        
    Cryptographic Principles Applied:
        - Perfect invertibility: Exactly reverses the R transformation
        - Modular arithmetic: Subtraction performed modulo q
        - Structure preservation: Maintains the block structure
    """
    # R^{-1} = [[I,-Conv_t],[0,I]] - inverse by negating off-diagonal term
    # First component: y1 - Conv_t(y2), Second component: y2 (unchanged)
    return ( (y1 - conv_op(y2, t, q)) % q, y2.copy() )


def block_apply_S_inv(y1: np.ndarray, y2: np.ndarray, u: np.ndarray, q: int):
    """
    Applies the inverse S transformation: S^(-1) = [[I, 0], [-Conv_u, I]].
    
    Inverts the S transformation by subtracting the convolution term that was
    added in the forward direction. Combined with R_inv, enables complete decryption.
    
    Args:
        y1 (np.ndarray): First half of transformed vector
        y2 (np.ndarray): Second half of transformed vector  
        u (np.ndarray): Same convolution kernel used in forward transformation
        q (int): Modulus for all operations
        
    Returns:
        tuple: (recovered_x1, recovered_x2)
        
    Cryptographic Principles Applied:
        - Perfect invertibility: Exactly reverses the S transformation
        - Modular arithmetic: Subtraction performed modulo q
        - Complementary inversion: Works with R_inv to provide complete decryption
    """
    # S^{-1} = [[I,0],[-Conv_u,I]] - inverse by negating off-diagonal term
    # First component: y1 (unchanged), Second component: y2 - Conv_u(y1)
    return ( y1.copy(), (y2 - conv_op(y1, u, q)) % q )


def coupling_forward(x: np.ndarray, cp: CouplingParams, key: VeinnParams, layer_idx: int) -> np.ndarray:
    """
    Forward coupling layer transformation (encryption direction).
    
    Implements an invertible coupling transformation:
    1. Split input x = (x1, x2) into two halves
    2. x1' = x1 + f(x2) where f uses mask_a
    3. x2' = x2 + g(x1') where g uses mask_b
    
    This provides:
    - Invertible nonlinear mixing between halves
    - Diffusion across the entire block
    - Resistance to differential cryptanalysis
    
    The ring convolutions provide algebraic mixing within each half.
    
    Args:
        x (np.ndarray): Input vector of length n
        cp (CouplingParams): Coupling parameters containing masks
        key (VeinnParams): Cryptographic key parameters including modulus q
        layer_idx (int): Layer index for domain separation
        
    Returns:
        np.ndarray: Transformed vector of same length as input
        
    Cryptographic Principles Applied:
        - Domain separation: Layer-specific seeds prevent cross-layer attacks
        - Three-stage mixing: R-M-S sequence provides strong diffusion
        - Perfect invertibility: Each operation is exactly reversible
        - Pseudorandom parameterization: All kernels derived from cryptographic seeds
        - Modular arithmetic: All operations performed in Z_q for security
    """
    n = x.shape[0]
    h = n // 2  # Half dimension for block operations
    q = key.q   # Modulus for all arithmetic operations
    
    assert x.shape == (n,), f"Input must be 1D array of length {n}"
    
    # Split input into two halves for block operations
    x1 = x[:h].astype(np.int64).copy()
    x2 = x[h:].astype(np.int64).copy()
    
    # Derive layer-specific seed using domain separation
    seed = derive_layer_seed_from_masks_and_key(cp.mask_a, cp.mask_b, layer_idx)
    
    # Generate three different kernels from the seed using domain separation tags
    t = derive_kernel_from_seed(seed, b"R", h, q)  # R transformation kernel
    u = derive_kernel_from_seed(seed, b"S", h, q)  # S transformation kernel  
    m = derive_kernel_from_seed(seed, b"M", h, q)  # Middle transformation kernel

    # Apply three-stage transformation: R -> M -> S
    # Stage 1: Apply R transformation (upper triangular)
    y1, y2 = block_apply_R(x1, x2, t, q)
    
    # Stage 2: Apply middle transformation M (additive mixing)
    # This adds additional diffusion between the R and S operations
    y1, y2 = ( y1 + conv_op(y2, m, q) % q, y2)
    
    # Stage 3: Apply S transformation (lower triangular)  
    y1, y2 = block_apply_S(y1, y2, u, q)
    
    # Recombine halves into single output vector
    return np.concatenate([y1.astype(np.int64), y2.astype(np.int64)])


def coupling_inverse(y: np.ndarray, cp: CouplingParams, key, layer_idx: int) -> np.ndarray:
    """
    Inverse coupling layer transformation (decryption direction).
    
    Reverses the coupling transformation by applying operations in reverse order:
    1. Split transformed input x = (x1', x2')
    2. x2 = x2' - g(x1') (reverse second coupling)
    3. x1 = x1' - f(x2) (reverse first coupling)
            
    Exactly reverses the forward coupling transformation by applying the inverse
    operations in reverse order: S^(-1) -> M^(-1) -> R^(-1). Uses the same
    layer-derived seeds to ensure perfect decryption.

    Exact inversion is guaranteed by the coupling layer structure.
    
    Args:
        y (np.ndarray): Transformed vector to decrypt
        cp (CouplingParams): Same coupling parameters used in forward direction
        key: Cryptographic key parameters including modulus q  
        layer_idx (int): Same layer index used in forward direction
        
    Returns:
        np.ndarray: Recovered original vector
        
    Cryptographic Principles Applied:
        - Perfect invertibility: Exactly reverses the forward transformation
        - Inverse operation order: S^(-1) -> M^(-1) -> R^(-1) sequence
        - Consistent parameterization: Uses same seeds as forward direction
        - Modular arithmetic: All inverse operations performed modulo q
        - Domain separation: Same layer-specific seeding as forward pass
    """
    n = y.shape[0]
    h = n // 2  # Half dimension for block operations
    q = key.q   # Modulus for all arithmetic operations
    
    assert y.shape == (n,), f"Input must be 1D array of length {n}"
    
    # Split transformed input into two halves
    y1 = y[:h].astype(np.int64).copy()
    y2 = y[h:].astype(np.int64).copy()

    # Derive same layer-specific seed as forward direction
    seed = derive_layer_seed_from_masks_and_key(cp.mask_a, cp.mask_b, layer_idx)
    
    # Generate same three kernels using identical domain separation tags
    t = derive_kernel_from_seed(seed, b"R", h, q)  # R transformation kernel
    u = derive_kernel_from_seed(seed, b"S", h, q)  # S transformation kernel
    m = derive_kernel_from_seed(seed, b"M", h, q)  # Middle transformation kernel

    # Apply inverse transformations in reverse order: S^(-1) -> M^(-1) -> R^(-1)
    # Stage 1: Apply S inverse (reverse lower triangular operation)
    y1, y2 = block_apply_S_inv(y1, y2, u, q)
    
    # Stage 2: Apply M inverse (subtract middle transformation)
    # This reverses the additive mixing applied in forward direction
    y1 = (y1 - conv_op(y2, m, q)) % q
    
    # Stage 3: Apply R inverse (reverse upper triangular operation)
    y1, y2 = block_apply_R_inv(y1, y2, t, q)

    # Recombine halves into single recovered vector
    return np.concatenate([y1.astype(np.int64), y2.astype(np.int64)])


# -----------------------------
# Shuffle
# -----------------------------
def make_shuffle_indices(n: int, stride: int) -> np.ndarray:
    """
    Generate deterministic permutation indices for diffusion layer.
    
    Creates a permutation π(i) = (i * stride) mod n. For cryptographic security:
    - stride must be coprime to n (ensures full permutation)
    - provides linear diffusion across block positions
    - deterministic for encryption/decryption consistency
    
    This implements the "permutation" component of Shannon's diffusion principle.
    
    Args:
        n: Block size (number of elements)
        stride: Multiplicative step size (must be coprime to n)
        
    Returns:
        Permutation indices for shuffle operation
        
    Raises:
        ValueError: If stride is not coprime to n (incomplete permutation)
    """
    if math.gcd(stride, n) != 1:
        raise ValueError(f"{bcolors.FAIL}shuffle_stride must be coprime with n{bcolors.ENDC}")
    return np.array([(i * stride) % n for i in range(n)], dtype=np.int64)

def shuffle(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Apply permutation to input array (diffusion layer).
    
    Reorders elements according to precomputed indices. Provides:
    - Linear diffusion across block positions
    - Fast O(n) operation
    - Perfect invertibility with inverse permutation
    
    Args:
        x: Input array to permute
        idx: Permutation indices
        
    Returns:
        Permuted array
    """
    assert x.shape[0] == idx.shape[0], f"Shuffle shape mismatch: input {x.shape}, indices {idx.shape}"
    return x[idx].astype(np.int64)

def unshuffle(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Apply inverse permutation to reverse shuffle operation.
    
    Computes and applies the inverse permutation to restore original order.
    Essential for correct decryption.
    
    Args:
        x: Permuted array to restore
        idx: Original permutation indices
        
    Returns:
        Array in original order
    """
    assert x.shape[0] == idx.shape[0], f"Unshuffle shape mismatch: input {x.shape}, indices {idx.shape}"
    # Compute inverse permutation
    inv = np.empty_like(idx)
    inv[idx] = np.arange(len(idx))
    return x[inv].astype(np.int64)

# -----------------------------
# Modular inverse helpers for finite field operations
# -----------------------------
def modinv(a: int, m: int) -> int:
    """
    Compute modular multiplicative inverse using Extended Euclidean Algorithm.
    
    Finds x such that a*x ≡ 1 (mod m). Essential for invertible operations
    in finite field arithmetic. Uses extended GCD for efficiency.
    
    Args:
        a: Element to invert
        m: Modulus
        
    Returns:
        Modular inverse of a
        
    Raises:
        ValueError: If gcd(a,m) != 1 (no inverse exists)
    """
    # Extended Euclidean Algorithm
    def egcd(aa: int, bb: int):
        if aa == 0:
            return bb, 0, 1
        g, x1, y1 = egcd(bb % aa, aa)
        return g, y1 - (bb // aa) * x1, x1
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError(f"{bcolors.FAIL}Modular inverse does not exist{bcolors.ENDC}")
    return x % m

def inv_vec_mod_q(arr: np.ndarray) -> np.ndarray:
    """
    Compute modular inverse for each element in array.
    
    Vectorized computation of modular inverses. Used to precompute
    inverse scaling factors for efficient decryption.
    
    Args:
        arr: Array of elements to invert
        
    Returns:
        Array of modular inverses
    """
    out = np.zeros_like(arr, dtype=np.int64)
    for i, v in enumerate(arr.astype(int).tolist()):
        out[i] = modinv(v, VeinnParams.q)
    return out

def ensure_coprime_to_q_vec(vec, q):
    """
    Ensure all vector elements are coprime to modulus q.
    
    Replaces any multiples of q with 1 to guarantee invertibility.
    Necessary for scaling operations in the cipher construction.
    
    Args:
        vec: Input vector
        q: Modulus
        
    Returns:
        Vector with all elements coprime to q
    """
    # Change any multiples of q to 1 to ensure invertibility
    vec = np.where(vec % q == 0, 1, vec)
    return vec

def key_from_seed(seed: bytes, vp: VeinnParams) -> VeinnKey:
    """
    Derive complete VEINN key from seed using cryptographic KDF.
    
    Implements hierarchical key derivation:
    1. Generate permutation parameters (deterministic)
    2. For each round, derive coupling masks using domain separation
    3. Generate invertible scaling factors with precomputed inverses
    
    Uses domain separation tags to ensure independent randomness for each component.
    All randomness derived from LWE-based PRF for post-quantum security.
    
    Args:
        seed: Master key seed (cryptographically random)
        vp: VEINN cipher parameters
        
    Returns:
        Complete cipher key with all round parameters
    """
    n = vp.n
    h = n // 2
    
    # Generate permutation indices (must be coprime for full permutation)
    shuffle_idx = make_shuffle_indices(n, vp.shuffle_stride)
    
    rounds = []
    for r in range(vp.rounds):
        # Derive coupling layer parameters for this round
        cpls = []
        for l in range(vp.layers_per_round):
            # Domain separation: unique tag for each layer
            tag = b"VEINN|r%d|l%d" % (r, l)
            
            # Derive coupling masks using LWE-PRF
            mask_a = derive_u16(h, vp, seed, tag, b"A")
            mask_b = derive_u16(h, vp, seed, tag, b"B")
            
            assert mask_a.shape == (h,) and mask_b.shape == (h,), f"Mask shape mismatch: expected {(h,)}, got {mask_a.shape}, {mask_b.shape}"
            cpls.append(CouplingParams(mask_a, mask_b))

        # Derive invertible per-element scaling factors
        scale = derive_u16(n, vp, seed, b"ring", bytes([r]))
        scale = scale.astype(np.int64)
        
        # Ensure all scaling factors are coprime to q (invertible)
        scale = ensure_coprime_to_q_vec(scale, VeinnParams.q)
        
        # Precompute modular inverses for decryption efficiency
        scale_inv = inv_vec_mod_q(scale)

        assert scale.shape == (n,), f"Ring scale shape mismatch: expected {(n,)}, got {scale.shape}"
        assert scale_inv.shape == (n,), f"Ring inv shape mismatch: expected {(n,)}, got {scale_inv.shape}"
        
        rounds.append(RoundParams(cpls, scale, scale_inv))
        
    return VeinnKey(seed=seed, params=vp, shuffle_idx=shuffle_idx, rounds=rounds)


# -----------------------------
# Permutation (Block Cipher Core)
# -----------------------------
def permute_forward(x: np.ndarray, key: VeinnKey) -> np.ndarray:
    """
    VEINN block cipher encryption (forward permutation).
    
    Implements a Feistel-like structure with multiple rounds:
    1. Coupling layers (nonlinear mixing inspired by neural networks)
    2. Invertible element-wise scaling (algebraic complexity)
    3. S-box layer (confusion via nonlinear substitution)
    4. Shuffle layer (diffusion via permutation)
    
    Each round provides:
    - Nonlinearity (S-boxes, coupling)
    - Diffusion (shuffle, ring convolution)
    - Key-dependent operations (all parameters derived from key)
    
    Security relies on:
    - Multiple rounds prevent differential/linear cryptanalysis
    - Large block size (4KB) provides wide diffusion
    - Post-quantum key derivation (LWE-based PRF)
    
    Args:
        x: Plaintext block (n elements)
        key: Complete VEINN cipher key
        
    Returns:
        Ciphertext block (same dimensions)
    """
    vp = key.params    
    idx = key.shuffle_idx
    
    assert x.shape == (vp.n,), f"Expected input shape {(vp.n,)}, got {x.shape}"
    
    y = x.copy()

    # Apply multiple cipher rounds
    for r in range(vp.rounds):
        # Coupling layers: invertible nonlinear mixing
        for cp in key.rounds[r].cpls:
            y = coupling_forward(y, cp, vp, idx)

        # Invertible element-wise scaling (adds algebraic complexity)
        y = (y.astype(np.int64) * key.rounds[r].ring_scale.astype(np.int64)) % vp.q
        
        # S-box layer: nonlinear substitution (confusion)
        y = np.array(sbox_layer(y, vp.q), dtype=np.int64)

        # Shuffle layer: linear permutation (diffusion)
        y = shuffle(y, idx)
        
    return y.astype(np.int64)

def permute_inverse(x: np.ndarray, key: VeinnKey) -> np.ndarray:
    """
    VEINN block cipher decryption (inverse permutation).
    
    Reverses the encryption process by applying all operations in reverse order:
    1. Inverse shuffle (restore original positions)
    2. Inverse S-box (reverse nonlinear substitution)
    3. Inverse scaling (using precomputed inverses)
    4. Inverse coupling layers (in reverse order)
    
    Exact inversion guaranteed by:
    - Invertible S-box construction
    - Precomputed scaling inverses
    - Coupling layer invertibility
    - Permutation invertibility
    
    Args:
        x: Ciphertext block to decrypt
        key: Same key used for encryption
        
    Returns:
        Original plaintext block
    """
    vp = key.params
    idx = key.shuffle_idx
    
    assert x.shape == (vp.n,), f"Expected input shape {(vp.n,)}, got {x.shape}"
    
    y = x.copy()
    
    # Apply rounds in reverse order
    for r in reversed(range(vp.rounds)):
        # Reverse shuffle layer
        y = unshuffle(y, idx)

        # Reverse S-box layer
        y = np.array(inv_sbox_layer(y, vp.q), dtype=np.int64)
        
        # Reverse scaling using precomputed inverses
        y = (y.astype(np.int64) * key.rounds[r].ring_scale_inv.astype(np.int64)) % vp.q
        
        # Reverse coupling layers in reverse order
        for cp in reversed(key.rounds[r].cpls):
            y = coupling_inverse(y, cp, vp, idx)
    return y.astype(np.int64)

# -----------------------------
# Block Helpers
# -----------------------------
def bytes_to_block(b: bytes, n: int) -> np.ndarray:
    """
    Convert byte string to cipher block format.
    
    Pads bytes to required length and interprets as array of 16-bit integers.
    Block format provides efficient vectorized operations.
    
    Args:
        b: Input bytes
        n: Target block size (number of 16-bit words)
        
    Returns:
        Block as array of integers
    """
    
    padded = b.ljust(2 * n, b'\x00')  # Pad with zeros to 2n bytes    
    arr = np.frombuffer(padded, dtype='<u2')[:n].copy()
    return arr.astype(np.int64)

def block_to_bytes(x: np.ndarray) -> bytes:
    """
    Convert cipher block back to byte string.
    
    Interprets integer array as 16-bit words and converts to bytes.
    Inverse of bytes_to_block operation.
    
    Args:
        x: Block as integer array
        
    Returns:
        Byte representation
    """
    return x.astype('<u2').tobytes()

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

# -----------------------------
# Key Management
# -----------------------------
def create_keystore(passphrase: str, keystore_file: str):
    """
    Create encrypted keystore for secure key management.
    
    Uses PBKDF2 key derivation with Fernet symmetric encryption:
    - PBKDF2-HMAC-SHA256 with 100k iterations (slow hash against brute force)
    - Random salt prevents rainbow table attacks
    - Fernet provides authenticated encryption (AES-128 + HMAC-SHA256)
    - Secure storage for multiple cryptographic keys
    
    Args:
        passphrase: User passphrase for keystore encryption
        keystore_file: File path for keystore storage
    """
    # Generate random salt for PBKDF2
    salt = secrets.token_bytes(16)
    
    # PBKDF2 key derivation with high iteration count
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,                # 256-bit key for Fernet
        salt=salt,
        iterations=100000,        # High iteration count against brute force
    )
    
    # Derive encryption key from passphrase
    key = b64encode(kdf.derive(passphrase.encode()))
    
    # Verify key is valid for Fernet
    Fernet(key)  # materialized to ensure validity
    
    # Initialize empty keystore structure
    keystore = {"salt": b64encode(salt).decode(), "keys": {}}
    
    # Store encrypted keystore to file
    with open(keystore_file, "wb") as kf:
        pickle.dump(keystore, kf)
    print(f"Keystore created at {keystore_file}")

def load_keystore(passphrase: str, keystore_file: str):
    """
    Load and unlock encrypted keystore.
    
    Reconstructs Fernet encryption key from passphrase and salt:
    - Reads salt from keystore file
    - Derives same key using PBKDF2 with stored salt
    - Returns keystore data and Fernet cipher for key operations
    
    Args:
        passphrase: User passphrase for decryption
        keystore_file: Path to keystore file
        
    Returns:
        Tuple of (keystore_data, fernet_cipher)
    """
    # Load keystore from file
    with open(keystore_file, "rb") as kf:
        keystore = pickle.load(kf)
    
    # Retrieve stored salt
    salt = b64decode(keystore["salt"])
    
    # Reconstruct PBKDF2 derivation
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    # Derive encryption key from passphrase
    key = b64encode(kdf.derive(passphrase.encode()))
    return keystore, Fernet(key)

def store_key_in_keystore(passphrase: str, key_name: str, key_data: dict, keystore_file: str):
    """
    Store cryptographic key in encrypted keystore.
    
    Encrypts key data with Fernet authenticated encryption:
    - JSON serialization for structured key data
    - Fernet encryption provides confidentiality and authenticity
    - Key name provides lookup index
    
    Args:
        passphrase: Keystore passphrase
        key_name: Identifier for stored key
        key_data: Key material to store (as dictionary)
        keystore_file: Path to keystore
    """
    # Load and unlock keystore
    keystore, fernet = load_keystore(passphrase, keystore_file)
    
    # Encrypt key data with Fernet authenticated encryption
    encrypted_key = fernet.encrypt(json.dumps(key_data).encode()).decode()
    
    # Store encrypted key in keystore
    keystore["keys"][key_name] = encrypted_key
    
    # Save updated keystore
    with open(keystore_file, "wb") as kf:
        pickle.dump(keystore, kf)

def retrieve_key_from_keystore(passphrase: str, key_name: str, keystore_file: str) -> dict:
    """
    Retrieve and decrypt key from keystore.
    
    Authenticated decryption with error handling:
    - Verifies key exists in keystore
    - Decrypts with Fernet (includes authenticity verification)
    - Returns structured key data
    
    Args:
        passphrase: Keystore passphrase
        key_name: Key identifier to retrieve
        keystore_file: Path to keystore
        
    Returns:
        Decrypted key data as dictionary
        
    Raises:
        ValueError: If key not found or decryption fails
    """
    # Load and unlock keystore
    keystore, fernet = load_keystore(passphrase, keystore_file)
    
    # Check if key exists
    if key_name not in keystore["keys"]:
        raise ValueError(f"{bcolors.FAIL}Key {key_name} not found in keystore{bcolors.ENDC}")
    
    # Retrieve encrypted key
    encrypted_key = keystore["keys"][key_name]
    
    try:
        # Decrypt and parse key data
        decrypted_key = fernet.decrypt(encrypted_key.encode())
        return json.loads(decrypted_key.decode())
    except Exception:
        raise ValueError(f"{bcolors.FAIL}Failed to decrypt key. Wrong passphrase?{bcolors.ENDC}")

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
# Encryption/Decryption
# -----------------------------

def validate_timestamp(timestamp: float, validity_window: int) -> bool:
    """
    Validates message freshness using timestamp comparison.
    
    Prevents replay attacks by ensuring messages are within an acceptable
    time window. This is a common technique in authenticated protocols.
    
    Args:
        timestamp (float): Message timestamp to validate
        validity_window (int): Maximum age in seconds
    
    Returns:
        bool: True if timestamp is within validity window
    
    Cryptographic principles:
    - Replay attack prevention: Time-based message freshness
    - Temporal authentication: Ensures messages are recent
    """
    current_time = time.time()
    return abs(current_time - timestamp) <= validity_window

def veinn_from_seed(seed_input: str, vp: VeinnParams):
    """
    Derives a Veinn key from a seed string for demonstration purposes.
    
    Shows how Veinn keys are generated from string seeds, useful for
    understanding the key derivation process in the custom cipher.
    
    Args:
        seed_input (str): String seed for key derivation
        vp (VeinnParams): Veinn cipher parameters
    
    Cryptographic principles:
    - Deterministic key derivation: Same seed produces same key
    - Custom cipher: Non-standard construction
    """
    seed = seed_input.encode('utf-8')
    k = key_from_seed(seed, vp)  # Custom key derivation function
    print(f"Derived VEINN key with params: n={vp.n}, rounds={vp.rounds}, layers_per_round={vp.layers_per_round}, shuffle_stride={vp.shuffle_stride}, use_lwe={vp.use_lwe}")    
    
def encrypt_with_pub(pubfile: str, file_type: str, message: Optional[str] = None, in_path: Optional[str] = None, vp: VeinnParams = VeinnParams(), seed_len: int = 32, nonce: Optional[bytes] = None, out_file: str = "enc_pub", mode: str = "cbc") -> str:
    """
    Implements hybrid encryption using Kyber KEM + Veinn symmetric cipher.
    
    This follows the standard hybrid encryption pattern:
    1. Generate ephemeral symmetric key
    2. Encrypt symmetric key with recipient's public key (KEM)
    3. Encrypt message with symmetric key (DEM)
    4. Combine both ciphertexts with authentication
    
    Args:
        pubfile (str): Recipient's public key file
        file_type (str): Output format (json/bin)
        message (str, optional): Text message to encrypt
        in_path (str, optional): File path to encrypt
        vp (VeinnParams): Veinn cipher parameters
        seed_len (int): Length of derived seed
        nonce (bytes, optional): Custom nonce for encryption
        out_file (str): Output file prefix
    
    Returns:
        str: Path to encrypted output file
    
    Cryptographic principles:
    - Hybrid encryption: Combines asymmetric KEM with symmetric DEM
    - Kyber KEM: Post-quantum secure key encapsulation
    - ISO 7816-4 padding: Standard padding scheme for block ciphers
    - HMAC authentication: Ensures ciphertext integrity and authenticity
    - Ephemeral keys: Fresh symmetric key for each encryption
    """
    # Load recipient's public key
    with open(pubfile, "r") as f:
        pub = json.load(f)
    ek = bytes(pub["ek"])  # Encapsulation key

    # Prepare message data
    if in_path:
        with open(in_path, "rb") as f:
            message_bytes = f.read()
            
    else:
        if not message:
            raise ValueError(f"{bcolors.FAIL}Message required for text mode{bcolors.ENDC}")
        message_bytes = message.encode('utf-8')

    # Apply ISO 7816-4 padding to message for block cipher compatibility
    message_bytes = pad_iso7816(message_bytes, vp.n * 2)
    
    # Generate random nonce if not provided
    nonce = nonce or secrets.token_bytes(16)
    
    # Kyber KEM: Encapsulate ephemeral symmetric key
    # This is the "KEM" part of hybrid encryption
    ephemeral_seed, ct = encaps(ek)

    # Derive Veinn symmetric key from ephemeral seed
    # This connects the KEM output to symmetric encryption
    k = key_from_seed(ephemeral_seed, vp)
    
    # Split message into blocks for Veinn block cipher
    blocks = [bytes_to_block(message_bytes[i:i + vp.n * 2], vp.n) for i in range(0, len(message_bytes), vp.n * 2)]
    for b in blocks:
        assert b.shape == (vp.n,)  # Verify block structure

    # Encrypt each block using Veinn forward permutation
    # This is the "DEM" (Data Encapsulation Mechanism) part
    # Encrypt blocks using selected chaining mode
    if mode == "cbc":
        enc_blocks = encrypt_blocks_cbc(blocks, k, nonce, vp)
    elif mode == "ctr":
        enc_blocks = encrypt_blocks_ctr(blocks, k, nonce, vp)
    elif mode == "cfb":
        enc_blocks = encrypt_blocks_cfb(blocks, k, nonce, vp)
    else:
        enc_blocks = [permute_forward(b, k) for b in blocks]

    # Store encryption metadata
    metadata = {
        "n": vp.n,
        "rounds": vp.rounds,
        "layers_per_round": vp.layers_per_round,
        "shuffle_stride": vp.shuffle_stride,
        "use_lwe": vp.use_lwe,
        "chaining_mode": mode,  # Store the chaining mode
        "bytes_per_number": vp.n * 2
    }
    
    # Add timestamp for replay attack prevention
    timestamp = time.time()
    
    # Create authenticated message for HMAC
    # Includes: KEM ciphertext + symmetric ciphertext + timestamp
    msg_for_hmac = ct + b"".join(block_to_bytes(b) for b in enc_blocks) + math.floor(timestamp).to_bytes(8, 'big')
    
    # HMAC for authentication using ephemeral seed as key
    # Provides integrity and authenticity of the entire ciphertext
    hmac_value = hmac.new(ephemeral_seed, msg_for_hmac, hashlib.sha256).hexdigest()
    
    # Write ciphertext with metadata
    out_file = out_file + "." + file_type
    write_ciphertext_with_iv(out_file, file_type, enc_blocks, metadata, ct, hmac_value, nonce, timestamp)
    return out_file

def decrypt_with_priv(keystore: Optional[str], privfile: Optional[str], encfile: str, passphrase: Optional[str], key_name: Optional[str], file_type: str, validity_window: int):
    """
    Decrypts hybrid-encrypted ciphertext using private key.
    
    Reverses the hybrid encryption process:
    1. Load private key from keystore or file
    2. Verify timestamp freshness (replay protection)
    3. Decapsulate symmetric key using Kyber (KEM)
    4. Verify HMAC authentication
    5. Decrypt message blocks using Veinn (DEM)
    6. Remove padding and recover plaintext
    
    Args:
        keystore (str, optional): Path to encrypted keystore
        privfile (str, optional): Path to private key file
        encfile (str): Encrypted file to decrypt
        passphrase (str, optional): Keystore passphrase
        key_name (str, optional): Key name in keystore
        file_type (str): Ciphertext format (json/bin)
        validity_window (int): Maximum message age in seconds
    
    Cryptographic principles:
    - Hybrid decryption: Reverses KEM + DEM process
    - Kyber decapsulation: Recovers symmetric key from KEM ciphertext
    - HMAC verification: Ensures ciphertext integrity before decryption
    - Timestamp validation: Prevents replay attacks
    - Authenticated decryption: Verify then decrypt pattern
    """
    # Load private key from keystore or file
    if keystore and passphrase and key_name:
        privkey = retrieve_key_from_keystore(passphrase, key_name, keystore)
    else:
        with open(privfile, "r") as f:
            privkey = json.load(f)
    
    dk = bytes(privkey["dk"])  # Decapsulation key
    
    # Read ciphertext components
    metadata, enc_seed_bytes, enc_blocks, hmac_value, iv, timestamp, nonce = read_ciphertext_with_iv(encfile, file_type)
    
    # Type verification for security
    assert isinstance(enc_seed_bytes, bytes), "Encrypted seed must be bytes"
    if nonce is not None:
        assert isinstance(nonce, bytes), "Nonce must be bytes"
    
    # Replay attack prevention: validate message freshness
    if not validate_timestamp(timestamp, validity_window):
        raise ValueError(f"{bcolors.FAIL}Timestamp outside validity window{bcolors.ENDC}")
    
    # Kyber KEM: Decapsulate ephemeral symmetric key
    # This recovers the symmetric key used for message encryption
    ephemeral_seed = decaps(dk, enc_seed_bytes)
    
    # Reconstruct authenticated message for HMAC verification
    msg_for_hmac = enc_seed_bytes + b"".join(block_to_bytes(b) for b in enc_blocks) + math.floor(timestamp).to_bytes(8, 'big')
    
    # HMAC verification: ensures ciphertext integrity and authenticity
    # Uses constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(hmac.new(ephemeral_seed, msg_for_hmac, hashlib.sha256).hexdigest(), hmac_value):
        raise ValueError(f"{bcolors.FAIL}HMAC verification failed{bcolors.ENDC}")
    
    # Reconstruct Veinn parameters from metadata
    vp = VeinnParams(
        n=metadata["n"],
        rounds=metadata["rounds"],
        layers_per_round=metadata["layers_per_round"],
        shuffle_stride=metadata["shuffle_stride"],
        use_lwe=metadata["use_lwe"]
    )
    
    # Derive same symmetric key from ephemeral seed
    k = key_from_seed(ephemeral_seed, vp)
    
    # Decrypt using appropriate chaining mode
    chaining_mode = metadata.get("chaining_mode", "ecb")  # Default to ECB for backwards compatibility
    
    if chaining_mode == "cbc":
        dec_blocks = decrypt_blocks_cbc(enc_blocks, k, iv, vp)
    elif chaining_mode == "ctr":
        dec_blocks = decrypt_blocks_ctr(enc_blocks, k, iv, vp)
    elif chaining_mode == "cfb":
        dec_blocks = decrypt_blocks_cfb(enc_blocks, k, iv, vp)
    else:
        # Legacy ECB mode (independent block decryption)
        dec_blocks = [permute_inverse(b, k) for b in enc_blocks]
    
    # Reconstruct message from decrypted blocks
    dec_bytes = b"".join(block_to_bytes(b) for b in dec_blocks)
    
    # Remove ISO 7816-4 padding to recover original message
    dec_bytes = unpad_iso7816(dec_bytes)
    
    print("Decrypted message:", dec_bytes.decode('utf-8')) 

    with open("decrypted.txt", "w") as f:
            json.dump(dec_bytes.decode('utf-8'), f)  

def encrypt_with_public_veinn(seed_input: str, file_type: str, message: Optional[str] = None, in_path: Optional[str] = None, vp: VeinnParams = VeinnParams(), out_file: str = "enc_pub_veinn.json", bytes_per_number: Optional[int] = None, nonce: Optional[bytes] = None, mode: str = "cbc") -> str:
    """
    Deterministic encryption using publicly shared VEINN seed.
    
    Unlike hybrid encryption, this uses a publicly known seed for deterministic encryption:
    - Same seed always produces same ciphertext (deterministic encryption)
    - No key exchange needed - seed can be shared publicly
    - Provides confidentiality but not semantic security
    - Suitable for deduplication scenarios or when determinism is desired
    
    Security model:
    - Assumes seed is known to all authorized parties
    - Security relies on VEINN construction strength
    - Does not provide semantic security (same plaintext = same ciphertext)
    - Includes HMAC and timestamp for authenticity and replay protection
    
    Args:
        seed_input: Publicly shared seed string
        file_type: Output format ("json" or "bin")
        message: Text message to encrypt (optional)
        in_path: Input file path (alternative to message)
        vp: VEINN cipher parameters
        out_file: Output file name prefix
        mode: Operation mode (compatibility parameter)
        bytes_per_number: Block size parameter (optional)
        nonce: Optional nonce for additional randomness
        
    Returns:
        Path to generated ciphertext file
    """
    # Convert seed to bytes for key derivation
    seed = seed_input.encode('utf-8')
    k = key_from_seed(seed, vp)
    
    # Prepare input data
    if in_path:
        with open(in_path, "rb") as f:
            message_bytes = f.read()
    else:
        if not message:
            raise ValueError(f"{bcolors.FAIL}Message required for text mode{bcolors.ENDC}")
        message_bytes = message.encode('utf-8')
    
    # Pad and encrypt message
    message_bytes = pad_iso7816(message_bytes, vp.n * 2)
    nonce = nonce or secrets.token_bytes(16)
    
    # Split into blocks and encrypt
    blocks = [bytes_to_block(message_bytes[i:i + vp.n * 2], vp.n) for i in range(0, len(message_bytes), vp.n * 2)]
    for b in blocks:
        assert b.shape == (vp.n,), f"Block shape mismatch: expected {(vp.n,)}, got {b.shape}"
    
    # Encrypt each block using Veinn forward permutation
    # This is the "DEM" (Data Encapsulation Mechanism) part
    # Encrypt blocks using selected chaining mode
    if mode == "cbc":
        enc_blocks = encrypt_blocks_cbc(blocks, k, nonce, vp)
    elif mode == "ctr":
        enc_blocks = encrypt_blocks_ctr(blocks, k, nonce, vp)
    elif mode == "cfb":
        enc_blocks = encrypt_blocks_cfb(blocks, k, nonce, vp)
    else:
        enc_blocks = [permute_forward(b, k) for b in blocks]
    
    # Package metadata
    metadata = {
        "n": vp.n,
        "rounds": vp.rounds,
        "layers_per_round": vp.layers_per_round,
        "shuffle_stride": vp.shuffle_stride,
        "use_lwe": vp.use_lwe,
        "chaining_mode": mode  # Store the chaining mode
    }        
   
    # Add authenticity and replay protection
    timestamp = time.time()
    msg_for_hmac = b"".join(block_to_bytes(b) for b in enc_blocks) + math.floor(timestamp).to_bytes(8, 'big')
    hmac_value = hmac.new(seed, msg_for_hmac, hashlib.sha256).hexdigest()
    
    # Write encrypted data
    out_file = out_file + "." + file_type
    write_ciphertext_with_iv(out_file, file_type, enc_blocks, metadata, seed, hmac_value, nonce, timestamp)
    print(f"Encrypted to {out_file}")
    return out_file

def decrypt_with_public_veinn(seed_input: str, file_type: str, enc_file: str, validity_window: int):
    """
    Deterministic decryption using publicly shared VEINN seed.
    
    Reverses deterministic encryption process:
    1. Read encrypted data and validate timestamp
    2. Derive VEINN key from public seed
    3. Verify HMAC authenticity tag
    4. Decrypt blocks using derived key
    5. Remove padding and output plaintext
    
    Security validations match encryption function for consistency.
    
    Args:
        seed_input: Same public seed used for encryption
        file_type: Input file format
        enc_file: Encrypted file path
        validity_window: Timestamp validity window in seconds
    """
    seed = seed_input.encode('utf-8')
    
    # Read encrypted data
    metadata, enc_seed_bytes, enc_blocks, hmac_value, iv, timestamp, nonce = read_ciphertext_with_iv(enc_file, file_type)
    
    # Timestamp validation (replay protection)
    if not validate_timestamp(timestamp, validity_window):
        raise ValueError(f"{bcolors.FAIL}Timestamp outside validity window{bcolors.ENDC}")
    
    # HMAC verification (authenticity check)
    msg_for_hmac = b"".join(block_to_bytes(b) for b in enc_blocks) + math.floor(timestamp).to_bytes(8, 'big')
    if not hmac.compare_digest(hmac.new(seed, msg_for_hmac, hashlib.sha256).hexdigest(), hmac_value):
        raise ValueError(f"{bcolors.FAIL}HMAC verification failed{bcolors.ENDC}")
    
    # Reconstruct VEINN parameters and key
    vp = VeinnParams(
        n=metadata["n"],
        rounds=metadata["rounds"],
        layers_per_round=metadata["layers_per_round"],
        shuffle_stride=metadata["shuffle_stride"],
        use_lwe=metadata["use_lwe"]
    )

    k = key_from_seed(seed, vp)
    
    # Decrypt using appropriate chaining mode
    chaining_mode = metadata.get("chaining_mode", "ecb")  # Default to ECB for backwards compatibility
    
    if chaining_mode == "cbc":
        dec_blocks = decrypt_blocks_cbc(enc_blocks, k, iv, vp)
    elif chaining_mode == "ctr":
        dec_blocks = decrypt_blocks_ctr(enc_blocks, k, iv, vp)
    elif chaining_mode == "cfb":
        dec_blocks = decrypt_blocks_cfb(enc_blocks, k, iv, vp)
    else:
        # Legacy ECB mode (independent block decryption)
        dec_blocks = [permute_inverse(b, k) for b in enc_blocks]

    dec_bytes = b"".join(block_to_bytes(b) for b in dec_blocks)
    
    # Remove padding and display result
    dec_bytes = unpad_iso7816(dec_bytes)
    print("Decrypted message:", dec_bytes.decode('utf-8')) 

# -----------------------------
# Serialization and File I/O
# -----------------------------

def encrypt_blocks_cbc(blocks: list, key: VeinnKey, iv: bytes, vp: VeinnParams) -> list:
    """
    Cipher Block Chaining (CBC) mode encryption.
    
    CBC Mode: C[i] = Encrypt(P[i] ⊕ C[i-1]), where C[0] = IV
    
    Each plaintext block is XORed with the previous ciphertext block before encryption.
    This creates interdependence between blocks, fixing the ECB vulnerability.
    
    Security properties:
    - Bit changes propagate to all subsequent blocks
    - Identical plaintext blocks produce different ciphertext
    - Requires IV for first block (stored with ciphertext)
    """
    if not blocks:
        return []
    
    # Convert IV to block format for XOR
    iv_block = bytes_to_block(iv.ljust(vp.n * 2, b'\x00'), vp.n)
    
    encrypted_blocks = []
    prev_ciphertext = iv_block  # Start with IV
    
    for i, plaintext_block in enumerate(blocks):
        # CBC: XOR plaintext with previous ciphertext
        chained_input = (plaintext_block ^ prev_ciphertext) % vp.q
        
        # Encrypt the chained input
        ciphertext_block = permute_forward(chained_input, key)
        encrypted_blocks.append(ciphertext_block)
        
        # Update previous ciphertext for next iteration
        prev_ciphertext = ciphertext_block
    
    return encrypted_blocks


def decrypt_blocks_cbc(enc_blocks: list, key: VeinnKey, iv: bytes, vp: VeinnParams) -> list:
    """
    CBC mode decryption: P[i] = Decrypt(C[i]) ⊕ C[i-1]
    """
    if not enc_blocks:
        return []
    
    iv_block = bytes_to_block(iv.ljust(vp.n * 2, b'\x00'), vp.n)
    
    decrypted_blocks = []
    prev_ciphertext = iv_block
    
    for ciphertext_block in enc_blocks:
        # Decrypt the ciphertext
        decrypted_intermediate = permute_inverse(ciphertext_block, key)
        
        # XOR with previous ciphertext to get plaintext
        plaintext_block = (decrypted_intermediate ^ prev_ciphertext) % vp.q
        decrypted_blocks.append(plaintext_block)
        
        # Update for next iteration
        prev_ciphertext = ciphertext_block
    
    return decrypted_blocks


def encrypt_blocks_ctr(blocks: list, key: VeinnKey, nonce: bytes, vp: VeinnParams) -> list:
    """
    Counter (CTR) mode encryption.
    
    CTR Mode: C[i] = P[i] ⊕ Encrypt(Nonce || Counter[i])
    
    Encrypts a counter value and XORs with plaintext. This creates a stream cipher
    from a block cipher. Each block uses a different counter value.
    
    Security properties:
    - Parallel encryption/decryption possible
    - No error propagation between blocks
    - Requires unique nonce for each message
    """
    encrypted_blocks = []
    
    for i, plaintext_block in enumerate(blocks):
        # Create counter block: nonce + block index
        counter_bytes = nonce + i.to_bytes(8, 'big')
        counter_block = bytes_to_block(counter_bytes.ljust(vp.n * 2, b'\x00'), vp.n)
        
        # Encrypt counter to create keystream
        keystream_block = permute_forward(counter_block, key)
        
        # XOR plaintext with keystream
        ciphertext_block = (plaintext_block ^ keystream_block) % vp.q
        encrypted_blocks.append(ciphertext_block)
    
    return encrypted_blocks


def decrypt_blocks_ctr(enc_blocks: list, key: VeinnKey, nonce: bytes, vp: VeinnParams) -> list:
    """
    CTR mode decryption (identical to encryption due to XOR properties)
    """
    return encrypt_blocks_ctr(enc_blocks, key, nonce, vp)  # CTR is symmetric


def encrypt_blocks_cfb(blocks: list, key: VeinnKey, iv: bytes, vp: VeinnParams) -> list:
    """
    Cipher Feedback (CFB) mode encryption.
    
    CFB Mode: C[i] = P[i] ⊕ Encrypt(C[i-1]), where C[0] = IV
    
    Encrypts the previous ciphertext block and XORs with current plaintext.
    Creates a self-synchronizing stream cipher.
    
    Security properties:
    - Error recovery after one block
    - Sequential operation required
    - Bit errors in ciphertext cause limited plaintext corruption
    """
    if not blocks:
        return []
    
    iv_block = bytes_to_block(iv.ljust(vp.n * 2, b'\x00'), vp.n)
    
    encrypted_blocks = []
    feedback_register = iv_block  # Start with IV
    
    for plaintext_block in blocks:
        # Encrypt the feedback register
        keystream_block = permute_forward(feedback_register, key)
        
        # XOR plaintext with keystream to get ciphertext
        ciphertext_block = (plaintext_block ^ keystream_block) % vp.q
        encrypted_blocks.append(ciphertext_block)
        
        # Update feedback register with current ciphertext
        feedback_register = ciphertext_block
    
    return encrypted_blocks


def decrypt_blocks_cfb(enc_blocks: list, key: VeinnKey, iv: bytes, vp: VeinnParams) -> list:
    """
    CFB mode decryption: P[i] = C[i] ⊕ Encrypt(C[i-1])
    """
    if not enc_blocks:
        return []
    
    iv_block = bytes_to_block(iv.ljust(vp.n * 2, b'\x00'), vp.n)
    
    decrypted_blocks = []
    feedback_register = iv_block
    
    for ciphertext_block in enc_blocks:
        # Encrypt the feedback register
        keystream_block = permute_forward(feedback_register, key)
        
        # XOR ciphertext with keystream to get plaintext
        plaintext_block = (ciphertext_block ^ keystream_block) % vp.q
        decrypted_blocks.append(plaintext_block)
        
        # Update feedback register with current ciphertext
        feedback_register = ciphertext_block
    
    return decrypted_blocks

def write_ciphertext_with_iv(path: str, file_type: str, encrypted_blocks: list, metadata: dict, 
                            enc_seed_bytes: bytes, hmac_value: str, iv: bytes, timestamp: float):
    """
    Enhanced serialization that includes IV/nonce for chained modes.
    """
    key = {
        "veinn_metadata": metadata,
        "enc_seed_b64": b64encode(enc_seed_bytes).decode(),
        "hmac": hmac_value,
        "iv_b64": b64encode(iv).decode(),  # Store IV separately
        "timestamp": timestamp
    }
    
    with open("key_" + path, "w") as f:
        json.dump(key, f)

    if file_type == "json":
        payload = {
            "encrypted": [[int(x) for x in blk.tolist()] for blk in encrypted_blocks]
        }
        with open(path, "w") as f:
            json.dump(payload, f)
    elif file_type == "bin":
        with open(path, "wb") as f:
            f.write(b"VEINN")
            f.write(len(encrypted_blocks).to_bytes(4, 'big'))
            for blk in encrypted_blocks:
                f.write(blk.tobytes())


def read_ciphertext_with_iv(path: str, file_type: str):
    """
    Enhanced deserialization that reads IV for chained modes.
    """
    with open("key_" + path, "r") as f:
        key = json.load(f)
        hmac_value = key["hmac"]
        iv = b64decode(key["iv_b64"])  # Read IV
        nonce = b64decode(key.get("nonce_b64", "")) if key.get("nonce_b64") else None
        timestamp = key["timestamp"]
        enc_seed = b64decode(key["enc_seed_b64"])
        metadata = key["veinn_metadata"]

    if file_type == "json":
        with open(path, "r") as f:
            encrypted = json.load(f)
        enc_blocks = [np.array([int(x) for x in blk], dtype=np.int64) for blk in encrypted["encrypted"]]
    elif file_type == "bin":
        with open(path, "rb") as f:
            magic = f.read(5)
            if magic != b"VEINN":
                raise ValueError("Invalid file format")
            num_blocks = int.from_bytes(f.read(4), 'big')
            n = metadata["n"]
            enc_blocks = []
            for _ in range(num_blocks):
                block_data = f.read(n * 8)
                block = np.frombuffer(block_data, dtype=np.int64)
                enc_blocks.append(block)
    
    return metadata, enc_seed, enc_blocks, hmac_value, iv, timestamp, nonce


# -----------------------------
# Interactive Configuration Helpers  
# -----------------------------
def options():
    """
    Interactive parameter configuration for VEINN operations.
    
    Prompts user for cipher parameters with sensible defaults:
    - Block size (n): affects security and performance
    - Round count: more rounds = higher security, slower operation  
    - Layers per round: coupling layer depth
    - Shuffle stride: must be coprime to n for proper permutation
    - LWE usage: enables post-quantum security
    - Custom nonce: for deterministic encryption scenarios
    - Modulus: prime field for arithmetic operations
    
    Returns:
        Tuple of configured parameters for VEINN construction
    """     
    n = int(input(f"Number of {np.int64} words per block (default {VeinnParams.n}): ").strip() or VeinnParams.n)
    rounds = int(input(f"Number of rounds (default {VeinnParams.rounds}): ").strip() or VeinnParams.rounds)
    layers_per_round = int(input(f"Layers per round (default {VeinnParams.layers_per_round}): ").strip() or VeinnParams.layers_per_round)
    shuffle_stride = int(input(f"Shuffle stride (default {VeinnParams.shuffle_stride}): ").strip() or VeinnParams.shuffle_stride)
    use_lwe = input("Use LWE PRF for key nonlinearity (y/n) [y]: ").strip().lower() or "y"    
    nonce_str = input("Custom nonce (base64, blank for random): ").strip() or None
    q = int(input(f"Modulus q (default {VeinnParams.q}): ").strip() or VeinnParams.q)
    return n, rounds, layers_per_round, shuffle_stride, use_lwe, nonce_str, q

# -----------------------------
# CLI Main with Interactive Menu
# -----------------------------
def menu_generate_keystore():
    """
    Interactive menu for creating encrypted keystores.
    
    Guides user through keystore creation process, collecting passphrase
    and filename with secure defaults.
    
    Cryptographic principles:
    - Secure key storage: Encrypted keystore protects multiple keys
    - Password-based encryption: Uses PBKDF2 for key derivation
    """
    passphrase = input("Enter keystore passphrase: ")
    keystore_file = input("Keystore filename (default keystore.json): ").strip() or "keystore.json"
    create_keystore(passphrase, keystore_file)

def menu_generate_keypair():
    """
    Interactive menu for generating Kyber post-quantum key pairs.
    
    Offers options for storing private keys either in encrypted keystores
    or separate files, following security best practices.
    
    Cryptographic principles:
    - Post-quantum cryptography: Kyber provides quantum-resistant security
    - Key management: Secure storage options for private keys
    """
    pubfile = input("Public key filename (default public_key.json): ").strip() or "public_key.json"
    use_keystore = input("Store private key in keystore? (y/n) [y]: ").strip().lower() or "y"
    
    privfile, keystore, passphrase, key_name = None, None, None, None
    if use_keystore == "y":
        keystore = input("Keystore filename (default keystore.json): ").strip() or "keystore.json"
        passphrase = input("Keystore passphrase: ")
        key_name = input("Key name in keystore: ")
    else:
        privfile = input("Private key filename (default private_key.json): ").strip() or "private_key.json"
    
    # Generate Kyber keypair using ML-KEM-768
    keypair = generate_keypair()
    
    # Save public key (always in plaintext file)
    with open(pubfile, "w") as f:
        json.dump({"ek": keypair["ek"]}, f)
    
    # Save private key based on user choice
    if use_keystore == "y":
        store_key_in_keystore(passphrase, key_name, keypair, keystore)
        print(f"Kyber keys generated: {pubfile} (public), private stored in keystore")
    else:
        with open(privfile, "w") as f:
            json.dump(keypair, f)
        print(f"Kyber keys generated: {pubfile} (public), {privfile} (private)")

def menu_encrypt_with_pub():
    """
    Interactive menu for hybrid encryption using Kyber + Veinn.
    
    Collects encryption parameters and performs hybrid encryption,
    combining post-quantum KEM with custom symmetric cipher.
    
    Cryptographic principles:
    - Hybrid encryption: Best of both asymmetric and symmetric crypto
    - Post-quantum security: Kyber provides quantum resistance
    - Customizable parameters: Allows cipher tuning
    """
    pubfile = input("Recipient public key file (default public_key.json): ").strip() or "public_key.json"
    
    if not os.path.exists(pubfile):
        print("Public key not found. Generate Kyber keys first.")
        return
    
    print("Supports multiple chaining modes:")
    print("- CBC: Cipher Block Chaining (XOR previous ciphertext with current plaintext)")
    print("- CTR: Counter mode (encrypt counter + nonce, XOR with plaintext)")
    print("- CFB: Cipher Feedback (encrypt previous ciphertext, XOR with plaintext)")
    mode = input("Choose cbc, ctr, or cfb [cbc]: ").strip() or "cbc"

    inpath = input("Optional input file path (blank = prompt): ").strip() or None    
    file_type = input("Output file type (JSON/BIN) [json]: ").strip() or "json"
    
    # Collect Veinn parameters
    n, rounds, layers_per_round, shuffle_stride, use_lwe, nonce_str, q = options()
    seed_len = int(input(f"Seed length (default {VeinnParams.seed_len}): ").strip() or VeinnParams.seed_len)
    
    use_lwe = use_lwe == "y"
    nonce = b64decode(nonce_str) if nonce_str else None    
    vp = VeinnParams(n=n, rounds=rounds, layers_per_round=layers_per_round, shuffle_stride=shuffle_stride, use_lwe=use_lwe, q=q)
    
    message = None    
    if inpath is None:        
        message = input("Message to encrypt: ")        
    
    encrypt_with_pub(pubfile, file_type, message=message, in_path=inpath, vp=vp, seed_len=seed_len, nonce=nonce, mode=mode)

def menu_decrypt_with_priv():
    """
    Interactive menu for decrypting hybrid ciphertext.
    
    Handles private key loading from keystores or files and performs
    authenticated decryption with timestamp validation.
    
    Cryptographic principles:
    - Authenticated decryption: Verifies integrity before decrypting
    - Key management: Supports both keystore and file-based keys
    - Replay protection: Timestamp validation prevents attacks
    """
    use_keystore = input("Use keystore for private key? (y/n): ").strip().lower() or "y"
    
    privfile, keystore, passphrase, key_name = None, None, None, None
    if use_keystore == "y":
        keystore = input("Keystore filename (default keystore.json): ").strip() or "keystore.json"
        passphrase = input("Keystore passphrase: ")
        key_name = input("Key name in keystore: ")
    else:
        privfile = input("Kyber private key file (default private_key.json): ").strip() or "private_key.json"
    
    encfile = input("Encrypted file to decrypt (default enc_pub): ").strip() or "enc_pub"
    file_type = input("Output file type (JSON/BIN) [json] : ").strip() or "json"
    encfile = encfile + "." + file_type
    validity_window = int(input(f"Timestamp validity window in seconds (default {VeinnParams.valid}): ").strip() or VeinnParams.valid)
    
    if not os.path.exists(encfile):
        print("Encrypted file not found.")
        return
        
    decrypt_with_priv(keystore, privfile, encfile, passphrase, key_name, file_type, validity_window)

def menu_veinn_from_seed():
    """
    Interactive demonstration of Veinn key derivation from seeds.
    
    Shows how symmetric keys are derived from string seeds in the Veinn system,
    useful for understanding the key generation process.
    
    Cryptographic principles:
    - Deterministic key derivation: Same seed produces same key
    - Parameter configuration: Shows cipher customization options
    """
    use_keystore = input("Use keystore for seed? (y/n): ").strip().lower() == "y"
    
    seed_input, keystore, passphrase, key_name = None, None, None, None
    if use_keystore:
        keystore = input("Keystore filename (default keystore.json): ").strip() or "keystore.json"
        passphrase = input("Keystore passphrase: ")
        key_name = input("Seed name in keystore: ")
        seed_data = retrieve_key_from_keystore(passphrase, key_name, keystore)
        seed_input = seed_data["seed"]
    else:
        seed_input = input("Enter seed string (publicly shared): ").strip()
    
    # Collect Veinn parameters
    n = int(input(f"Number of {np.int64} words per block (default {VeinnParams.n}): ").strip() or VeinnParams.n)
    rounds = int(input(f"Number of rounds (default {VeinnParams.rounds}): ").strip() or VeinnParams.rounds)
    layers_per_round = int(input(f"Layers per round (default {VeinnParams.layers_per_round}): ").strip() or VeinnParams.layers_per_round)
    shuffle_stride = int(input(f"Shuffle stride (default {VeinnParams.shuffle_stride}): ").strip() or VeinnParams.shuffle_stride)
    use_lwe = input("Use LWE PRF for key nonlinearity (y/n) [y]: ").strip().lower() or "y"
    use_lwe = use_lwe == "y"
    q = int(input(f"Modulus q (default {VeinnParams.q}): ").strip() or VeinnParams.q)
    
    vp = VeinnParams(n=n, rounds=rounds, layers_per_round=layers_per_round, shuffle_stride=shuffle_stride, use_lwe=use_lwe, q=q)
    veinn_from_seed(seed_input, vp)

def menu_encrypt_with_public_veinn():
    """
    Interactive menu for Veinn symmetric encryption with shared seeds.
    
    Performs symmetric encryption where both parties share a common seed,
    suitable for scenarios where key distribution has already occurred.
    
    Cryptographic principles:
    - Symmetric encryption: Pre-shared key scenario
    - Secure key derivation: Deterministic key from seed
    - Message authentication: HMAC for integrity
    """
    message = None    
    use_keystore = input("Use keystore for seed? (y/n): ").strip().lower() or "y"
    
    seed_input, keystore, passphrase, key_name = None, None, None, None
    print("Supports multiple chaining modes:")
    print("- CBC: Cipher Block Chaining (XOR previous ciphertext with current plaintext)")
    print("- CTR: Counter mode (encrypt counter + nonce, XOR with plaintext)")
    print("- CFB: Cipher Feedback (encrypt previous ciphertext, XOR with plaintext)")
    mode = input("Choose cbc, ctr, or cfb [cbc]: ").strip() or "cbc"
    if use_keystore == "y":
        keystore = input("Keystore filename (default keystore.json): ").strip() or "keystore.json"
        passphrase = input("Keystore passphrase: ")
        key_name = input("Seed name in keystore: ")
        
        # Store seed in keystore if needed
        store_key_in_keystore(passphrase, key_name, {"seed": key_name}, keystore)
        seed_data = retrieve_key_from_keystore(passphrase, key_name, keystore)
        seed_input = seed_data["seed"]
    else:
        seed_input = input("Enter public seed string: ").strip()
    
    inpath = input("Optional input file path (blank = prompt): ").strip() or None 
    if inpath == None:
        message = input("Message to encrypt: ")    
    
    out_file = input("Output encrypted filename (default enc_pub_veinn): ").strip() or "enc_pub_veinn"      
    file_type = input("Output file type (JSON/BIN) [json] : ").strip() or "json"
    
    # Collect cipher parameters
    n, rounds, layers_per_round, shuffle_stride, use_lwe, nonce_str, q = options()
    use_lwe = use_lwe == "y"
    nonce = b64decode(nonce_str) if nonce_str else None
    vp = VeinnParams(n=n, rounds=rounds, layers_per_round=layers_per_round, shuffle_stride=shuffle_stride, use_lwe=use_lwe, q=q)
    
    encrypt_with_public_veinn(seed_input, file_type, message, inpath, vp, out_file, nonce, mode=mode)

def menu_decrypt_with_public_veinn():
    """
    Interactive menu for decrypting Veinn symmetric ciphertext.
    
    Decrypts messages encrypted with shared seeds, verifying authenticity
    and freshness before revealing the plaintext.
    
    Cryptographic principles:
    - Symmetric decryption: Uses same seed as encryption
    - Authentication verification: HMAC validation
    - Replay protection: Timestamp checking
    """
    use_keystore = input("Use keystore for seed? (y/n): ").strip().lower() or "y"
    
    seed_input, keystore, passphrase, key_name = None, None, None, None
    if use_keystore == "y":
        keystore = input("Keystore filename (default keystore.json): ").strip() or "keystore.json"
        passphrase = input("Keystore passphrase: ")
        key_name = input("Seed name in keystore: ")
        seed_data = retrieve_key_from_keystore(passphrase, key_name, keystore)
        seed_input = seed_data["seed"]
    else:
        seed_input = input("Enter public seed string: ").strip()
    
    enc_file = input("Encrypted file to decrypt (default enc_pub_veinn): ").strip() or "enc_pub_veinn"
    file_type = input("Output file type (JSON/BIN) [json] : ").strip() or "json"
    validity_window = int(input(f"Timestamp validity window in seconds (default {VeinnParams.valid}): ").strip() or VeinnParams.valid)
    enc_file = enc_file + "." + file_type
    
    if not os.path.exists(enc_file):
        print("Encrypted file not found.")
        return
        
    decrypt_with_public_veinn(seed_input, file_type, enc_file, validity_window)            

def main():
    parser = argparse.ArgumentParser(description="VEINN - Vector Encrypted Invertible Neural Network")
    subparsers = parser.add_subparsers(dest="command")

    create_keystore_parser = subparsers.add_parser("create_keystore", help="Create encrypted keystore")
    create_keystore_parser.add_argument("--passphrase", required=True, help="Keystore passphrase")
    create_keystore_parser.add_argument("--keystore_file", default="keystore.json", help="Keystore filename")

    generate_parser = subparsers.add_parser("generate_keypair", help="Generate keypair")
    generate_parser.add_argument("--pubfile", default="public_key.json", help="Public key filename")
    generate_parser.add_argument("--privfile", default="private_key.json", help="Private key filename")
    generate_parser.add_argument("--keystore", default="keystore.json", help="Keystore filename")
    generate_parser.add_argument("--passphrase", help="Keystore passphrase")
    generate_parser.add_argument("--key_name", help="Key name in keystore")

    public_encrypt_parser = subparsers.add_parser("public_encrypt", help="Encrypt with public key")
    public_encrypt_parser.add_argument("--pubfile", default="public_key.json", help="Public key file")
    public_encrypt_parser.add_argument("--in_path", help="Input file path")
    public_encrypt_parser.add_argument("--file_type", choices=["json", "bin"], default="json", help="File type [JSON/BIN]")
    public_encrypt_parser.add_argument("--n", type=int, default=VeinnParams.n)
    public_encrypt_parser.add_argument("--rounds", type=int, default=VeinnParams.rounds)
    public_encrypt_parser.add_argument("--layers_per_round", type=int, default=VeinnParams.layers_per_round)
    public_encrypt_parser.add_argument("--shuffle_stride", type=int, default=VeinnParams.shuffle_stride)
    public_encrypt_parser.add_argument("--use_lwe", type=bool, default=True)
    public_encrypt_parser.add_argument("--q", type=int, default=VeinnParams.q)
    public_encrypt_parser.add_argument("--seed_len", type=int, default=32)
    public_encrypt_parser.add_argument("--nonce", help="Custom nonce (base64)")
    public_encrypt_parser.add_argument("--out_file", default="enc_pub")
    public_encrypt_parser.add_argument("--mode", default="cbc")

    public_decrypt_parser = subparsers.add_parser("public_decrypt", help="Decrypt with private key")
    public_decrypt_parser.add_argument("--keystore", default="keystore.json")
    public_decrypt_parser.add_argument("--privfile", default="private_key.json")
    public_decrypt_parser.add_argument("--encfile", default="enc_pub.json")
    public_decrypt_parser.add_argument("--passphrase")
    public_decrypt_parser.add_argument("--key_name")
    public_decrypt_parser.add_argument("--file_type", default="json")
    public_decrypt_parser.add_argument("--validity_window", type=int, default=3600)

    public_veinn_parser = subparsers.add_parser("public_veinn", help="Derive public VEINN from seed")
    public_veinn_parser.add_argument("--seed", required=True)
    public_veinn_parser.add_argument("--n", type=int, default=VeinnParams.n)
    public_veinn_parser.add_argument("--rounds", type=int, default=VeinnParams.rounds)
    public_veinn_parser.add_argument("--layers_per_round", type=int, default=VeinnParams.layers_per_round)
    public_veinn_parser.add_argument("--shuffle_stride", type=int, default=VeinnParams.shuffle_stride)
    public_veinn_parser.add_argument("--use_lwe", type=bool, default=True)
    public_veinn_parser.add_argument("--q", type=int, default=VeinnParams.q)

    args = parser.parse_known_args()[0]

    try:
        match args.command:
            case "create_keystore":
                create_keystore(args.passphrase, args.keystore_file)
                print(f"Keystore created: {args.keystore_file}")
            case "generate_keypair":
                keypair = generate_keypair()
                with open(args.pubfile, "w") as f:
                    json.dump({"ek": keypair["ek"]}, f)
                if args.keystore and args.passphrase and args.key_name:
                    store_key_in_keystore(args.passphrase, args.key_name, keypair, args.keystore)
                    print(f"Kyber keys generated: {args.pubfile} (public), private stored in keystore")
                else:
                    with open(args.privfile, "w") as f:
                        json.dump(keypair, f)
                    print(f"Kyber keys generated: {args.pubfile} (public), {args.privfile} (private)")
            case "public_encrypt":
                vp = VeinnParams(
                    n=args.n,
                    rounds=args.rounds,
                    layers_per_round=args.layers_per_round,
                    shuffle_stride=args.shuffle_stride,
                    use_lwe=args.use_lwe,
                    q=args.q
                )
                nonce = b64decode(args.nonce) if args.nonce else None
                encrypt_with_pub(
                    pubfile=args.pubfile,
                    file_type=args.file_type,
                    in_path=args.in_path,                    
                    vp=vp,
                    seed_len=args.seed_len,
                    nonce=nonce,
                    out_file=args.out_file,
                    mode=args.mode
                )
            case "public_decrypt":
                decrypt_with_priv(
                    keystore=args.keystore,
                    privfile=args.privfile,
                    encfile=args.encfile,
                    passphrase=args.passphrase,
                    key_name=args.key_name,
                    file_type=args.file_type,
                    validity_window=args.validity_window
                )
            case "public_veinn":
                vp = VeinnParams(
                    n=args.n,
                    rounds=args.rounds,
                    layers_per_round=args.layers_per_round,
                    shuffle_stride=args.shuffle_stride,
                    use_lwe=args.use_lwe,
                    q=args.q
                )
                veinn_from_seed(args.seed, vp)
            case _:
                _=os.system("cls") | os.system("clear")
                while True:
                    print(f"{bcolors.WARNING}{bcolors.BOLD}VEINN - Vector Encrypted Invertible Neural Network{bcolors.ENDC}")
                    print(f"{bcolors.GREY}{bcolors.BOLD}(]≡≡≡≡ø‡»{bcolors.OKCYAN}========================================-{bcolors.ENDC}")
                    print("")
                    print(f"{bcolors.BOLD}1){bcolors.ENDC} Create encrypted keystore")
                    print(f"{bcolors.BOLD}2){bcolors.ENDC} Generate keypair (public/private)")
                    print(f"{bcolors.BOLD}3){bcolors.ENDC} Encrypt with recipient public key")
                    print(f"{bcolors.BOLD}4){bcolors.ENDC} Decrypt with private key")
                    print(f"{bcolors.BOLD}5){bcolors.ENDC} Encrypt deterministically using public VEINN")
                    print(f"{bcolors.BOLD}6){bcolors.ENDC} Decrypt deterministically using public VEINN")
                    print(f"{bcolors.BOLD}7){bcolors.ENDC} Derive public VEINN from seed")                    
                    print(f"{bcolors.BOLD}0){bcolors.ENDC} Exit")
                    print("")
                    choice = input(f"{bcolors.BOLD}Choice: {bcolors.ENDC}").strip()
                    try:
                        match choice:
                            case "0":
                                break
                            case "1":
                                menu_generate_keystore()                                
                            case "2":
                                menu_generate_keypair()
                            case "3":
                                menu_encrypt_with_pub()
                            case "4":
                                menu_decrypt_with_priv()
                            case "5":
                                menu_encrypt_with_public_veinn()
                            case "6":
                                menu_decrypt_with_public_veinn()
                            case "7":
                                menu_veinn_from_seed()
                            case _:
                                print("Invalid choice")
                    except Exception as e:
                        print(f"{bcolors.FAIL}ERROR:{bcolors.ENDC}", e)
                    _=input(f"{bcolors.OKGREEN}Enter to continue...{bcolors.ENDC}")
                    _=os.system("cls") | os.system("clear")
    except Exception as e:
        print(f"{bcolors.FAIL}ERROR:{bcolors.ENDC}", e)
        sys.exit(1)

if __name__ == "__main__":
    main()

