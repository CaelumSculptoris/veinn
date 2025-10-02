import os
import json
import numpy as np
from base64 import b64encode, b64decode
from src.models import (VeinnParams)
from src.utils.keystore import(create_keystore, store_key_in_keystore, retrieve_key_from_keystore)
from src.utils.keygen import(generate_keypair)
from src.core import(
    encrypt_with_pub, decrypt_with_priv, veinn_from_seed, encrypt_with_public_veinn, 
    decrypt_with_public_veinn
)

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

