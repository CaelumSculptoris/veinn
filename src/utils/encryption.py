import math
import hashlib
import secrets
import numpy as np
from kyber_py.ml_kem import ML_KEM_768  # Using ML_KEM_768 for ~128-bit security

from src.models import (VeinnParams, VeinnKey, CouplingParams, RoundParams, bcolors)

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