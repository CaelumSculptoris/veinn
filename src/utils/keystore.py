import json
import secrets
import pickle
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from base64 import b64encode, b64decode
from src.models import (bcolors)
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
