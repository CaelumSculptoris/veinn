import json
import math
import hashlib
import hmac
import secrets
import numpy as np
import time
from typing import Optional
from base64 import b64encode, b64decode
from src.models import (VeinnParams, bcolors)
from src.utils.keygen import (encaps, decaps)
from src.utils.encryption import (
    key_from_seed, pad_iso7816, unpad_iso7816, bytes_to_block, block_to_bytes, permute_forward,
    permute_inverse, encrypt_blocks_cbc, encrypt_blocks_cfb, encrypt_blocks_ctr, 
    decrypt_blocks_cbc, decrypt_blocks_cfb, decrypt_blocks_ctr
)
from src.utils.keystore import (retrieve_key_from_keystore)

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
