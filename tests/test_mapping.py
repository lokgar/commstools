import numpy as np
import pytest

from commstools import mapping


def test_qam_mapping(backend_device, xp):
    """Verify QAM mapping produces the correct number of symbols."""
    # Use xp to create array on device
    bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1])

    syms = mapping.map_bits(bits, modulation="qam", order=16)

    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2


def test_psk_mapping(backend_device, xp):
    """Verify PSK mapping produces the correct number of symbols."""
    bits = xp.array([0, 1])

    syms = mapping.map_bits(bits, modulation="psk", order=2)

    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2


def test_demap_dimensions_mimo(backend_device, xp):
    """Test that demap_symbols preserves multidimensional structure."""
    modulation = "qam"
    order = 4

    # Creates input shape (2 streams, 4 symbols)
    # Total symbols = 8
    # Total bits = 16
    bits_in = xp.zeros(16, dtype=xp.int32)
    bits_in[::2] = 1

    # Map to symbols (flat) first
    symbols_flat = mapping.map_bits(bits_in, modulation, order)

    # Reshape to (2, 4) to simulate MIMO
    symbols_mimo = symbols_flat.reshape(2, 4)

    # Demap
    bits_out = mapping.demap_symbols(symbols_mimo, modulation, order)

    # Verify strict shape compliance
    # We expect (2, 4 * 2) = (2, 8)
    expected_shape = (2, 8)
    assert bits_out.shape == expected_shape
    # Flatten both to compare content (bits_out is flat if reshape logic failed, but shape check guards it)
    assert xp.array_equal(bits_out.flatten(), bits_in)


def test_soft_demap_sign_correctness(backend_device, xp):
    """LLR sign should match hard decision for noiseless symbols."""
    modulation = "qam"
    order = 16

    # Generate known bits and map to symbols
    bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=xp.int32)
    symbols = mapping.map_bits(bits, modulation, order)

    # Very low noise variance (high SNR) for near-perfect decisions
    noise_var = 1e-6

    llrs = mapping.demap_symbols_soft(
        symbols, modulation, order, noise_var, method="maxlog"
    )

    # Hard decision from LLR: bit = 0 if LLR > 0, else 1
    hard_from_llr = (llrs < 0).astype(xp.int32)

    # Should match original bits
    assert xp.array_equal(hard_from_llr, bits), "LLR signs don't match original bits"


def test_soft_demap_roundtrip(backend_device, xp):
    """Hard decision from LLR should match direct hard demapping."""
    modulation = "psk"
    order = 8

    # Generate random symbols
    bits = xp.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=xp.int32)
    symbols = mapping.map_bits(bits, modulation, order)

    # Soft demap with very low noise
    noise_var = 1e-6
    llrs = mapping.demap_symbols_soft(symbols, modulation, order, noise_var)

    # Hard decision from LLR
    hard_from_llr = (llrs < 0).astype(xp.int32)

    # Direct hard demapping
    hard_direct = mapping.demap_symbols(symbols, modulation, order)

    assert xp.array_equal(hard_from_llr, hard_direct)


def test_soft_demap_exact_vs_maxlog(backend_device, xp):
    """Exact and max-log methods should give similar results at high SNR."""
    modulation = "qam"
    order = 4

    bits = xp.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=xp.int32)
    symbols = mapping.map_bits(bits, modulation, order)

    noise_var = 0.01  # Moderate SNR

    llrs_maxlog = mapping.demap_symbols_soft(
        symbols, modulation, order, noise_var, method="maxlog"
    )
    llrs_exact = mapping.demap_symbols_soft(
        symbols, modulation, order, noise_var, method="exact"
    )

    # Signs should match
    assert xp.array_equal(xp.sign(llrs_maxlog), xp.sign(llrs_exact))

    # Values should be reasonably close (within 20% for QPSK at this SNR)
    # Max-log is an approximation, so some difference is expected
    ratio = xp.abs(llrs_maxlog) / (xp.abs(llrs_exact) + 1e-10)
    assert xp.all(ratio > 0.5) and xp.all(ratio < 2.0)


def test_soft_demap_mimo_shape(backend_device, xp):
    """Soft demapping should preserve MIMO channel structure."""
    modulation = "qam"
    order = 4

    # 2 streams, 4 symbols each
    bits = xp.zeros(16, dtype=xp.int32)
    symbols = mapping.map_bits(bits, modulation, order).reshape(2, 4)

    noise_var = 0.1
    llrs = mapping.demap_symbols_soft(symbols, modulation, order, noise_var)

    # Expected shape: (2, 4*2) = (2, 8)
    expected_shape = (2, 8)
    assert llrs.shape == expected_shape


def test_gray_code_edge_cases():
    """Verify Gray code generation for boundary bit depths."""
    import numpy as np
    import pytest

    # n = 0
    assert np.array_equal(mapping.gray_code(0), np.array([0]))
    assert np.array_equal(mapping.gray_to_binary(0), np.array([0]))

    # n = 2
    # binary: 0, 1, 2, 3
    # gray: 0, 1, 3, 2
    # gray_to_binary should be inverse: mapping[0]=0, mapping[1]=1, mapping[3]=2, mapping[2]=3
    assert np.array_equal(mapping.gray_to_binary(2), np.array([0, 1, 3, 2]))

    # n < 0
    with pytest.raises(ValueError):
        mapping.gray_code(-1)
    with pytest.raises(ValueError):
        mapping.gray_to_binary(-1)


def test_8qam_mapping(backend_device, xp):
    """Verify 8-QAM (rectangular) mapping and round-trip."""
    bits = xp.array([0, 0, 0, 1, 1, 1], dtype=xp.int32)
    syms = mapping.map_bits(bits, modulation="qam", order=8)
    assert len(syms) == 2

    bits_out = mapping.demap_symbols(syms, modulation="qam", order=8)
    assert xp.array_equal(bits, bits_out)


def test_cross_qam_32_mapping(backend_device, xp):
    """Verify 32-QAM (Cross) mapping and round-trip."""
    # 5 bits per symbol. 2 symbols = 10 bits.
    bits = xp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=xp.int32)
    syms = mapping.map_bits(bits, modulation="qam", order=32)
    assert len(syms) == 2

    bits_out = mapping.demap_symbols(syms, modulation="qam", order=32)
    assert xp.array_equal(bits, bits_out)


def test_soft_demap_loop_based(backend_device, xp):
    """Verify loop-based soft demapping (vectorized=False)."""
    modulation = "qam"
    order = 16
    bits = xp.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=xp.int32)
    symbols = mapping.map_bits(bits, modulation, order)
    noise_var = 0.1

    llrs_vec = mapping.demap_symbols_soft(
        symbols, modulation, order, noise_var, vectorized=True
    )
    llrs_loop = mapping.demap_symbols_soft(
        symbols, modulation, order, noise_var, vectorized=False
    )

    assert xp.allclose(llrs_vec, llrs_loop)

    # Test exact method
    llrs_exact = mapping.demap_symbols_soft(
        symbols, modulation, order, noise_var, method="exact", vectorized=False
    )
    # Signs should match
    assert xp.array_equal(xp.sign(llrs_exact), xp.sign(llrs_vec))

    # Test error for unknown method (vectorized)
    import pytest

    with pytest.raises(ValueError, match="Unknown method"):
        mapping.demap_symbols_soft(
            symbols, modulation, order, noise_var, method="unknown", vectorized=True
        )

    # Test error for unknown method (loop-based)
    with pytest.raises(ValueError, match="Unknown method"):
        mapping.demap_symbols_soft(
            symbols, modulation, order, noise_var, method="unknown", vectorized=False
        )


def test_gray_constellation_advanced(backend_device, xp):
    """Verify constellation generation edge cases."""
    # 1. Unipolar forced by string
    const_unipol = mapping.gray_constellation("pam-unipol", 4)
    assert np.min(const_unipol) >= 0

    # 2. Custom scheme with hyphen
    const_custom = mapping.gray_constellation("my-pam", 4)
    assert len(const_custom) == 4

    # 3. Order error
    with pytest.raises(ValueError, match="at least 2"):
        mapping.gray_constellation("psk", 1)

    # 4. QAM non-power-of-2 (already handled by log2 usually but let's check explicitly)
    # Actually gray_constellation checks if it's power of 2
    with pytest.raises(ValueError, match="power of 2"):
        mapping.gray_constellation("qam", 7)

    # 5. Unknown modulation
    with pytest.raises(ValueError, match="Unsupported modulation type"):
        mapping.gray_constellation("unknown", 4)


def test_map_demap_unipolar(backend_device, xp):
    """Verify bit mapping and demapping with unipolar ASK/PAM."""
    bits = xp.array([0, 1, 1, 0])
    # Map to unipolar
    syms = mapping.map_bits(bits, "ask", 4, unipolar=True)
    assert xp.all(syms >= 0)

    # Demap from unipolar
    bits_rx = mapping.demap_symbols(syms, "ask", 4, unipolar=True)
    assert xp.array_equal(bits, bits_rx)

    # Soft demap from unipolar
    llrs = mapping.demap_symbols_soft(syms, "ask", 4, noise_var=0.1, unipolar=True)
    # Signs should allow recovering bits: LLR > 0 -> 0, LLR < 0 -> 1
    bits_soft = (llrs < 0).astype(xp.int32)
    assert xp.array_equal(bits, bits_soft)


def test_mapping_more(backend_device, xp):
    """Verify more mapping edge cases."""
    # Order error in map_bits
    with pytest.raises(ValueError, match="at least 2"):
        mapping.map_bits(xp.array([0, 0]), "psk", 1)

    # Order error in demap_symbols
    with pytest.raises(ValueError, match="at least 2"):
        mapping.demap_symbols(xp.array([0]), "psk", 1)


def test_mapping_order_errors(backend_device, xp):
    """Verify order validation for power-of-2 requirements."""
    # map_bits
    with pytest.raises(ValueError, match="power of 2"):
        mapping.map_bits(xp.array([0, 0]), "psk", 3)

    # demap_symbols_soft
    with pytest.raises(ValueError, match="power of 2"):
        mapping.demap_symbols_soft(xp.array([0.1]), "psk", 3, noise_var=0.1)


def test_demap_symbols_empty_shape(backend_device, xp):
    """Verify demapping behavior for effectively scalar inputs."""
    # 0-dim array (scalar)
    s = xp.array(1.0)
    # constellation will be [-1, 1] for psk-2
    # 1.0 is bit 1? No, 1.0 is mapped to bit?
    # gray_constillation(psk, 2) -> [-1, 1]
    # index 0 is -1, index 1 is 1.
    # so 1.0 -> index 1 -> bit 1.
    bits = mapping.demap_symbols(s, "psk", 2)
    assert bits.size == 1
    assert int(bits) == 1
