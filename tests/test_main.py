def test_chaining_modes_avalanche(vp: VeinnParams, message_sizes: list = [500, 1500, 3000], num_tests: int = 5):
    """
    Compare avalanche effects across different chaining modes and message sizes.
    
    This test verifies that the chaining modes fix the ECB vulnerability by
    ensuring bit changes propagate across block boundaries.
    """
    print("Chaining Modes Avalanche Comparison")
    print("=" * 60)
    
    modes = ["ecb", "cbc", "ctr", "cfb"]
    results = {}
    
    # Generate test keypair
    test_keypair = generate_keypair()
    ek = bytes(test_keypair["ek"])
    
    for mode in modes:
        print(f"\nTesting {mode.upper()} mode:")
        print("-" * 30)
        results[mode] = {}
        
        for msg_size in message_sizes:
            print(f"  {msg_size} bytes...", end=" ", flush=True)
            
            # Generate random test message
            test_message = secrets.token_bytes(msg_size)
            
            # Test bit flips
            bit_changes = []
            total_bits = msg_size * 8
            num_bit_tests = min(50, total_bits)  # Sample for performance
            bit_positions = np.random.choice(total_bits, size=num_bit_tests, replace=False)
            
            # Encrypt original message
            if mode == "ecb":
                original_encrypted = encrypt_message_ecb_test(test_message, ek, vp)
            else:
                original_encrypted = encrypt_message_chained_test(test_message, ek, vp, mode)
            
            for bit_pos in bit_positions:
                # Flip bit and encrypt - inline implementation to avoid function call issues
                if bit_pos >= len(test_message) * 8:
                    continue  # Skip if bit position is out of range
                
                byte_index = bit_pos // 8
                bit_index = bit_pos % 8
                
                # Convert to bytearray for mutability
                modified_message = bytearray(test_message)
                modified_message[byte_index] ^= (1 << bit_index)
                modified_message = bytes(modified_message)
                
                if mode == "ecb":
                    modified_encrypted = encrypt_message_ecb_test(modified_message, ek, vp)
                else:
                    modified_encrypted = encrypt_message_chained_test(modified_message, ek, vp, mode)
                
                # Calculate Hamming distance by converting bytes to numpy arrays
                orig_array = np.frombuffer(original_encrypted, dtype=np.uint8).astype(np.int64)
                mod_array = np.frombuffer(modified_encrypted, dtype=np.uint8).astype(np.int64)
                bit_diff = hamming_distance_bits(orig_array, mod_array)
                bit_changes.append(bit_diff)
            
            # Calculate statistics
            avg_bit_changes = np.mean(bit_changes)
            total_output_bits = len(original_encrypted) * 8
            avalanche_pct = (avg_bit_changes / total_output_bits) * 100
            
            results[mode][msg_size] = {
                'avalanche_percentage': avalanche_pct,
                'avg_bits_changed': avg_bit_changes,
                'total_output_bits': total_output_bits
            }
            
            print(f"{avalanche_pct:.1f}% avalanche")
    
    # Print comparison table
    print(f"\nAvalanche Comparison Table:")
    print("-" * 60)
    print(f"{'Mode':<6}", end="")
    for size in message_sizes:
        print(f"{size:>10}B", end="")
    print()
    print("-" * 60)
    
    for mode in modes:
        print(f"{mode.upper():<6}", end="")
        for size in message_sizes:
            pct = results[mode][size]['avalanche_percentage']
            if pct < 40:
                color = bcolors.FAIL
            elif pct > 45:
                color = bcolors.OKGREEN
            else:
                color = bcolors.WARNING
            print(f"{color}{pct:>9.1f}%{bcolors.ENDC}", end="")
        print()
    
    print("\nInterpretation:")
    print("- ECB mode should show poor avalanche for multi-block messages")
    print("- CBC, CTR, CFB should show ~50% avalanche across all sizes")
    print("- Values <40% indicate security concerns")
    print("- Values >45% indicate good diffusion properties")
    
    return results


def encrypt_message_ecb_test(message: bytes, ek: bytes, vp: VeinnParams) -> bytes:
    """Test helper: ECB mode encryption (original vulnerable method)"""
    padded_message = pad_iso7816(message, vp.n * 2)
    ephemeral_seed, ct_kem = encaps(ek)
    k = key_from_seed(ephemeral_seed, vp)
    
    encrypted_blocks = []
    for i in range(0, len(padded_message), vp.n * 2):
        block_bytes = padded_message[i:i + vp.n * 2]
        block_array = bytes_to_block(block_bytes, vp.n)
        encrypted_block = permute_forward(block_array, k)  # Independent encryption
        encrypted_blocks.append(block_to_bytes(encrypted_block))
    
    return ct_kem + b''.join(encrypted_blocks)


def encrypt_message_chained_test(message: bytes, ek: bytes, vp: VeinnParams, mode: str) -> bytes:
    """Test helper: Chained mode encryption"""
    padded_message = pad_iso7816(message, vp.n * 2)
    ephemeral_seed, ct_kem = encaps(ek)
    k = key_from_seed(ephemeral_seed, vp)
    iv = secrets.token_bytes(16)
    
    blocks = [bytes_to_block(padded_message[i:i + vp.n * 2], vp.n) 
              for i in range(0, len(padded_message), vp.n * 2)]
    
    if mode == "cbc":
        enc_blocks = encrypt_blocks_cbc(blocks, k, iv, vp)
    elif mode == "ctr":
        enc_blocks = encrypt_blocks_ctr(blocks, k, iv, vp)
    elif mode == "cfb":
        enc_blocks = encrypt_blocks_cfb(blocks, k, iv, vp)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    encrypted_data = b''.join(block_to_bytes(b) for b in enc_blocks)
    return ct_kem + iv + encrypted_data


#====================================================================
def flip_bit(data: np.ndarray, bit_position: int) -> np.ndarray:
    """
    Flip a single bit in the input array.


    Args:
        data: Input array of int64 values
        bit_position: Global bit position to flip (0 to n*64-1)

    Returns:
        Array with single bit flipped
    """
    result = data.copy()
    word_index = bit_position // 64
    bit_index = bit_position % 64

    if word_index < len(result):
        # Use numpy int64 to avoid C long overflow
        mask = np.int64(1) << np.int64(bit_index)
        result[word_index] = np.int64(result[word_index]) ^ mask

    return result


def hamming_distance_bits(a: np.ndarray, b: np.ndarray) -> int:
    """
    Calculate Hamming distance in bits between two arrays.


    Args:
        a, b: Arrays to compare

    Returns:
        Number of different bits
    """
    xor_result = a ^ b
    # Count set bits in each word and sum
    return sum(bin(int(word)).count('1') for word in xor_result)

def avalanche_test_full_encryption(message: bytes, vp: VeinnParams, num_tests: int = 100) -> dict:
    """
    Test avalanche effect through complete encryption pipeline including:
    1. ISO 7816-4 padding
    2. bytes_to_block conversion 
    3. VEINN block encryption
    4. All preprocessing steps from encrypt_with_pub
    
    Args:
        message: Original message bytes to test
        vp: VEINN parameters
        num_tests: Number of random bit flips to test
        
    Returns:
        Dictionary with comprehensive avalanche statistics
    """
    # Generate random key for testing
    seed = secrets.token_bytes(vp.seed_len)
    key = key_from_seed(seed, vp)
    
    # Apply full preprocessing pipeline (matching encrypt_with_pub)
    padded_message = pad_iso7816(message, vp.n * 2)
    
    # Split into blocks as done in encrypt_with_pub
    message_blocks = []
    for i in range(0, len(padded_message), vp.n * 2):
        block_data = padded_message[i:i + vp.n * 2]
        block = bytes_to_block(block_data, vp.n)
        message_blocks.append(block)
    
    # Encrypt all blocks to get baseline ciphertext
    original_encrypted_blocks = [permute_forward(block, key) for block in message_blocks]
    
    # Convert to flat bit array for bit manipulation
    original_message_bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
    total_message_bits = len(original_message_bits)
    
    # Convert encrypted blocks to flat bit array for comparison
    original_cipher_bytes = b"".join(block_to_bytes(block) for block in original_encrypted_blocks)
    original_cipher_bits = np.unpackbits(np.frombuffer(original_cipher_bytes, dtype=np.uint8))
    total_cipher_bits = len(original_cipher_bits)
    
    bit_changes = []
    block_changes = []
    
    print(f"Testing {num_tests} random bit flips in {total_message_bits}-bit message...", end="", flush=True)
    
    # Test random bit positions in original message
    test_positions = np.random.choice(total_message_bits, size=min(num_tests, total_message_bits), replace=False)
    
    for i, bit_pos in enumerate(test_positions):
        if i % 20 == 0:
            print(".", end="", flush=True)
        
        # Flip single bit in original message
        modified_bits = original_message_bits.copy()
        modified_bits[bit_pos] = 1 - modified_bits[bit_pos]  # Flip bit
        
        # Convert back to bytes
        # Pad to byte boundary if necessary
        if len(modified_bits) % 8 != 0:
            padding = 8 - (len(modified_bits) % 8)
            modified_bits = np.concatenate([modified_bits, np.zeros(padding, dtype=np.uint8)])
        
        modified_message = np.packbits(modified_bits).tobytes()[:len(message)]
        
        # Apply full encryption pipeline to modified message
        try:
            padded_modified = pad_iso7816(modified_message, vp.n * 2)
            
            # Process into blocks
            modified_blocks = []
            for j in range(0, len(padded_modified), vp.n * 2):
                block_data = padded_modified[j:j + vp.n * 2]
                block = bytes_to_block(block_data, vp.n)
                modified_blocks.append(block)
            
            # Encrypt modified blocks
            modified_encrypted_blocks = [permute_forward(block, key) for block in modified_blocks]
            
            # Convert to bits for comparison
            modified_cipher_bytes = b"".join(block_to_bytes(block) for block in modified_encrypted_blocks)
            modified_cipher_bits = np.unpackbits(np.frombuffer(modified_cipher_bytes, dtype=np.uint8))
            
            # Ensure same length for comparison (padding may cause differences)
            min_len = min(len(original_cipher_bits), len(modified_cipher_bits))
            
            # Calculate bit-level differences
            bit_diff = np.sum(original_cipher_bits[:min_len] != modified_cipher_bits[:min_len])
            
            # Calculate block-level differences
            block_diff = sum(1 for orig, mod in zip(original_encrypted_blocks, modified_encrypted_blocks) 
                           if not np.array_equal(orig, mod))
            
            bit_changes.append(bit_diff)
            block_changes.append(block_diff)
            
        except Exception as e:
            # Skip invalid modifications that break padding/structure
            continue
    
    print(" done!")
    
    if not bit_changes:
        return {"error": "No valid test cases completed"}
    
    bit_changes_array = np.array(bit_changes)
    block_changes_array = np.array(block_changes)
    
    results = {
        'message_info': {
            'original_length_bytes': len(message),
            'padded_length_bytes': len(padded_message),
            'total_blocks': len(message_blocks),
            'bits_per_block': vp.n * 64,
            'total_input_bits': total_message_bits,
            'total_output_bits': total_cipher_bits
        },
        'test_info': {
            'bits_tested': len(bit_changes),
            'successful_tests': len([x for x in bit_changes if x > 0])
        },
        'bit_avalanche': {
            'mean': float(np.mean(bit_changes_array)),
            'std': float(np.std(bit_changes_array)),
            'min': int(np.min(bit_changes_array)),
            'max': int(np.max(bit_changes_array)),
            'median': float(np.median(bit_changes_array))
        },
        'block_avalanche': {
            'mean': float(np.mean(block_changes_array)),
            'std': float(np.std(block_changes_array)),
            'min': int(np.min(block_changes_array)),
            'max': int(np.max(block_changes_array)),
            'total_blocks': len(message_blocks)
        },
        'avalanche_percentage': float(np.mean(bit_changes_array) / total_cipher_bits * 100),
        'block_avalanche_percentage': float(np.mean(block_changes_array) / len(message_blocks) * 100)
    }
    
    return results


def avalanche_test_message_sizes(vp: VeinnParams) -> dict:
    """
    Test avalanche effect across different message sizes to see how
    padding and block structure affects diffusion.
    """
    test_messages = [
        b"Hello",  # Very short
        b"Hello, World! This is a test message.",  # Medium
        b"A" * 100,  # Exactly 100 bytes
        b"B" * (vp.n * 2 - 10),  # Just under one block
        b"C" * (vp.n * 2),  # Exactly one block
        b"D" * (vp.n * 2 + 10),  # Just over one block
        b"E" * (vp.n * 4),  # Multiple blocks
    ]
    
    results = {}
    
    for i, msg in enumerate(test_messages):
        print(f"\nTesting message {i+1}/{len(test_messages)}: {len(msg)} bytes")
        result = avalanche_test_full_encryption(msg, vp, num_tests=50)
        
        if "error" not in result:
            results[f"message_{len(msg)}_bytes"] = result
            print(f"  Avalanche: {result['avalanche_percentage']:.2f}%, "
                  f"Blocks affected: {result['block_avalanche_percentage']:.2f}%")
    
    return results

def menu_full_avalanche_test():
    """
    Interactive menu for comprehensive avalanche testing.
    """
    print("Full Encryption Pipeline Avalanche Test")
    print("=" * 50)
    
    # Get VEINN parameters
    vp = VeinnParams()
    print(f"Using VEINN parameters: n={vp.n}, rounds={vp.rounds}, layers={vp.layers_per_round}")
    
    test_choice = input("\nChoose test type:\n1) Single message test\n2) Multiple message sizes\nChoice [1]: ").strip() or "1"
    
    if test_choice == "1":
        # Single message test
        message = input("Enter test message: ").encode('utf-8')
        num_tests = int(input("Number of bit flips to test [100]: ").strip() or "100")
        
        result = avalanche_test_full_encryption(message, vp, num_tests)
        
        if "error" not in result:
            print(f"\nResults for {len(message)}-byte message:")
            print(f"Message padded to {result['message_info']['padded_length_bytes']} bytes ({result['message_info']['total_blocks']} blocks)")
            print(f"Bit avalanche: {result['avalanche_percentage']:.2f}%")
            print(f"Block avalanche: {result['block_avalanche_percentage']:.2f}%")
            print(f"Average bits changed: {result['bit_avalanche']['mean']:.1f} / {result['message_info']['total_output_bits']}")
            print(f"Average blocks affected: {result['block_avalanche']['mean']:.1f} / {result['message_info']['total_blocks']}")
        else:
            print("Test failed:", result["error"])
    
    elif test_choice == "2":
        # Multiple message sizes
        results = avalanche_test_message_sizes(vp)
        
        print(f"\nSummary across message sizes:")
        print("-" * 60)
        for msg_key, result in results.items():
            size = msg_key.split('_')[1]
            print(f"{size:>8} bytes: {result['avalanche_percentage']:6.2f}% bit, "
                  f"{result['block_avalanche_percentage']:6.2f}% block avalanche")