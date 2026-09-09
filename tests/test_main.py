import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.core import validate_timestamp
from src.models import VeinnParams
from src.utils.encryption import (
    block_to_bytes,
    bytes_to_block,
    coupling_forward,
    coupling_inverse,
    decrypt_blocks_cbc,
    decrypt_blocks_cfb,
    decrypt_blocks_ctr,
    derive_kernel_from_seed,
    encrypt_blocks_cbc,
    encrypt_blocks_cfb,
    encrypt_blocks_ctr,
    inv_sbox_layer,
    key_from_seed,
    make_shuffle_indices,
    modinv,
    pad_iso7816,
    permute_forward,
    permute_inverse,
    ring_convolution,
    sbox_layer,
    shake,
    shuffle,
    unpad_iso7816,
    unshuffle,
)
from src.utils.keygen import decaps, encaps, generate_keypair, generate_kyber_keypair
from src.utils.keystore import create_keystore, retrieve_key_from_keystore, store_key_in_keystore
from src.utils.menu import options


def small_params():
    return VeinnParams(n=8, rounds=1, layers_per_round=1, shuffle_stride=3, use_lwe=False)


class TestUtilities(unittest.TestCase):
    def test_shake_is_sized_and_domain_separated(self):
        self.assertEqual(len(shake(17, b"a")), 17)
        self.assertNotEqual(shake(17, b"a", b"bc"), shake(17, b"ab", b"c"))

    def test_padding_round_trip_and_invalid_padding(self):
        for data in (b"", b"hello", b"\x80\x00"):
            padded = pad_iso7816(data, 8)
            self.assertEqual(len(padded) % 8, 0)
            self.assertEqual(unpad_iso7816(padded), data)
        with self.assertRaises(ValueError):
            unpad_iso7816(b"not padded")

    def test_block_conversion_for_16_bit_values(self):
        data = b"hello world"
        self.assertEqual(block_to_bytes(bytes_to_block(data, 8))[:len(data)], data)

    def test_shuffle_and_unshuffle(self):
        values = np.arange(8, dtype=np.int64)
        indices = make_shuffle_indices(8, 3)
        self.assertTrue(np.array_equal(unshuffle(shuffle(values, indices), indices), values))
        with self.assertRaises(ValueError):
            make_shuffle_indices(8, 2)

    def test_modinv_and_sbox(self):
        self.assertEqual((7 * modinv(7, 97)) % 97, 1)
        values = np.array([0, 1, 2, 7, 96], dtype=np.int64)
        self.assertTrue(np.array_equal(inv_sbox_layer(sbox_layer(values, 97), 97), values))

    def test_kernel_shape_and_range(self):
        kernel = derive_kernel_from_seed(b"seed", b"test", 4, 97)
        self.assertEqual(kernel.shape, (4,))
        self.assertTrue(np.all((kernel >= 0) & (kernel < 97)))

    def test_timestamp_window(self):
        with patch("src.core.time.time", return_value=100.0):
            self.assertTrue(validate_timestamp(95.0, 5))
            self.assertFalse(validate_timestamp(94.9, 5))


class TestTransforms(unittest.TestCase):
    def test_naive_ring_convolution_known_result(self):
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        b = np.array([5, 6, 7, 8], dtype=np.int64)
        expected = np.array([5 - 61, 16 - 52, 34 - 32, 60], dtype=np.int64) % 97
        self.assertTrue(np.array_equal(ring_convolution(a, b, 97, "naive"), expected))

    def test_ntt_matches_naive_ring_convolution(self):
        rng = np.random.default_rng(4)
        a = rng.integers(0, 97, 8, dtype=np.int64)
        b = rng.integers(0, 97, 8, dtype=np.int64)
        self.assertTrue(np.array_equal(
            ring_convolution(a, b, 97, "ntt"),
            ring_convolution(a, b, 97, "naive"),
        ))

    def test_coupling_and_permutation_inverse(self):
        vp = small_params()
        key = key_from_seed(b"test", vp)
        values = np.arange(vp.n, dtype=np.int64)
        cp = key.rounds[0].cpls[0]
        transformed = coupling_forward(values, cp, vp, 0)
        self.assertTrue(np.array_equal(coupling_inverse(transformed, cp, vp, 0), values))
        encrypted = permute_forward(values, key)
        self.assertTrue(np.array_equal(permute_inverse(encrypted, key), values))


class TestBlockModes(unittest.TestCase):
    def setUp(self):
        self.vp = small_params()
        self.key = key_from_seed(b"mode test", self.vp)
        self.blocks = [np.arange(8, dtype=np.int64), np.arange(8, 16, dtype=np.int64)]
        self.iv = b"0123456789abcdef"

    def test_cbc_round_trip(self):
        encrypted = encrypt_blocks_cbc(self.blocks, self.key, self.iv, self.vp)
        decrypted = decrypt_blocks_cbc(encrypted, self.key, self.iv, self.vp)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(self.blocks, decrypted)))

    def test_ctr_round_trip(self):
        encrypted = encrypt_blocks_ctr(self.blocks, self.key, self.iv, self.vp)
        decrypted = decrypt_blocks_ctr(encrypted, self.key, self.iv, self.vp)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(self.blocks, decrypted)))

    def test_cfb_round_trip(self):
        encrypted = encrypt_blocks_cfb(self.blocks, self.key, self.iv, self.vp)
        decrypted = decrypt_blocks_cfb(encrypted, self.key, self.iv, self.vp)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(self.blocks, decrypted)))


class TestKyber(unittest.TestCase):
    def test_standard_ml_kem_round_trip(self):
        keypair = generate_kyber_keypair()
        from kyber_py.ml_kem import ML_KEM_768
        shared_key, ciphertext = ML_KEM_768.encaps(bytes(keypair["ek"]))
        recovered = ML_KEM_768.decaps(bytes(keypair["dk"]), ciphertext)
        self.assertEqual(shared_key, recovered)

    def test_project_kem_wrapper_round_trip(self):
        keypair = generate_keypair()
        shared_key, ciphertext = encaps(bytes(keypair["ek"]))
        self.assertEqual(shared_key, decaps(bytes(keypair["dk"]), ciphertext))


class TestKeystoreAndSerialization(unittest.TestCase):
    def test_keystore_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "keystore.json")
            create_keystore("passphrase", path)
            store_key_in_keystore("passphrase", "key", {"value": "secret"}, path)
            self.assertEqual(retrieve_key_from_keystore("passphrase", "key", path), {"value": "secret"})
            with self.assertRaises(ValueError):
                retrieve_key_from_keystore("wrong", "key", path)

    def test_json_ciphertext_serialization_shape(self):
        from src.core import read_ciphertext_with_iv, write_ciphertext_with_iv

        with tempfile.TemporaryDirectory() as directory:
            path = "cipher.json"
            blocks = [np.arange(8, dtype=np.int64)]
            metadata = {"n": 8, "chaining_mode": "ecb"}
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                write_ciphertext_with_iv(path, "json", blocks, metadata, b"seed", "tag", b"iv", 100.0)
                loaded_metadata, seed, loaded_blocks, tag, iv, timestamp, nonce = read_ciphertext_with_iv(path, "json")
            finally:
                os.chdir(previous_directory)
            self.assertEqual(loaded_metadata, metadata)
            self.assertEqual(seed, b"seed")
            self.assertTrue(np.array_equal(loaded_blocks[0], blocks[0]))
            self.assertEqual((tag, iv, timestamp, nonce), ("tag", b"iv", 100.0, None))


class TestMenu(unittest.TestCase):
    def test_options_parses_values(self):
        answers = ["8", "2", "3", "5", "n", "", "97"]
        with patch("builtins.input", side_effect=answers):
            self.assertEqual(options(), (8, 2, 3, 5, "n", None, 97))


if __name__ == "__main__":
    unittest.main()
