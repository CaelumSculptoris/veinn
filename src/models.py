from dataclasses import dataclass
import numpy as np

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