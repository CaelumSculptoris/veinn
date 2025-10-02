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

from src.utils.keygen import ( 
    generate_keypair, generate_kyber_keypair, generate_veinn_keypair, encaps, decaps
)

from src.utils.keystore import (
    create_keystore, retrieve_key_from_keystore, load_keystore, store_key_in_keystore
)

from src.utils.menu import (
    menu_generate_keypair, menu_generate_keystore, menu_encrypt_with_pub, menu_decrypt_with_priv,
    menu_decrypt_with_public_veinn, menu_encrypt_with_public_veinn, menu_veinn_from_seed
)

from src.models import (
    VeinnKey, VeinnParams, CouplingParams, bcolors
)
