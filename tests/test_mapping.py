import pytest
import numpy as np

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


def test_gray_code_edge_cases(backend_device, xp):
    """Verify Gray code generation for boundary bit depths."""

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
    assert xp.min(const_unipol) >= 0

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
    assert int(bits.item()) == 1


def test_gray_code_zero(backend_device, xp):
    """Verify Gray code for 0 bits."""
    assert np.array_equal(mapping.gray_code(0), [0])
    assert np.array_equal(mapping.gray_to_binary(0), [0])


def test_gray_code_negative(backend_device, xp):
    """Verify error for negative bits."""
    with pytest.raises(ValueError, match="n must be non-negative"):
        mapping.gray_code(-1)
    with pytest.raises(ValueError, match="n must be non-negative"):
        mapping.gray_to_binary(-1)


def test_constellation_unsupported(backend_device, xp):
    """Verify error for unknown modulation type."""
    with pytest.raises(ValueError, match="Unsupported modulation type"):
        mapping.gray_constellation("chaos", 4)


def test_constellation_order_error(backend_device, xp):
    """Verify errors for non-matching orders."""
    with pytest.raises(ValueError, match="Order must be at least 2"):
        mapping.gray_constellation("qam", 1)

    with pytest.raises(ValueError, match="Order must be power of 2"):
        mapping.gray_constellation("qam", 10)

    # Test internal helpers напрямую if needed, but они usually called via gray_constellation
    # Triggering the internal ValueError in _gray_psk
    with pytest.raises(ValueError, match="Order must be power of 2"):
        from commstools.mapping import _gray_psk

        _gray_psk(3)

    with pytest.raises(ValueError, match="Order must be power of 2"):
        from commstools.mapping import _gray_ask

        _gray_ask(6)


def test_constellation_fallback_split(backend_device, xp):
    """Test the hyphen split fallback in constellation naming."""
    # "my-qam" -> core scheme is "qam" via 'in' check
    c1 = mapping.gray_constellation("my-qam", 4)
    c2 = mapping.gray_constellation("qam", 4)
    assert xp.allclose(c1, c2)

    # Triggering the split fallback (line 159)
    with pytest.raises(ValueError, match="Unsupported modulation type: unknown"):
        mapping.gray_constellation("custom-unknown", 4)


def test_qam_cross_fallback(backend_device, xp):
    """Trigger the fallback in cross-QAM for small N."""
    # _gray_qam_cross(8) -> n=2, m=1. width=4, height=2.
    # n < 3 triggers n_shift=0 branch
    from commstools.mapping import _gray_qam_cross

    res = _gray_qam_cross(8)
    assert res.shape == (8,)


def test_map_bits_divisibility(backend_device, xp):
    """Verify error when bit count is not divisible by bits per symbol."""
    bits = xp.array([1, 0, 1])  # 3 bits
    with pytest.raises(ValueError, match="must be divisible by bits per symbol"):
        mapping.map_bits(bits, "qam", 16)  # bits_per_symbol = 4


def test_map_bits_dtype_coercion(backend_device, xp):
    """Verify dtype coercion in map_bits for ASK/PAM."""
    bits = xp.array([0, 1, 0, 1])
    # Case: complex dtype requested for ASK
    out = mapping.map_bits(bits, "ask", 4, dtype="complex128")
    assert out.dtype == xp.float64  # c128 -> f64

    out = mapping.map_bits(bits, "ask", 4, dtype="complex64")
    assert out.dtype == xp.float32  # c64 -> f32

    # Case: real dtype requested for ASK
    out = mapping.map_bits(bits, "ask", 4, dtype="float32")
    assert out.dtype == xp.float32


def test_soft_demap_invalid_order(backend_device, xp):
    """Verify error for non-power-of-2 order in soft demapping."""
    with pytest.raises(ValueError, match="Order must be a power of 2"):
        mapping.demap_symbols_soft(xp.ones(1), "qam", 6, 0.1)


def test_soft_demap_invalid_method(backend_device, xp):
    """Verify error for unknown method in soft demapping."""
    with pytest.raises(ValueError, match="Unknown method"):
        mapping.demap_symbols_soft(xp.ones(1), "qam", 4, 0.1, method="magic")

    # Force loop-based to test error there too
    with pytest.raises(ValueError, match="Unknown method"):
        mapping.demap_symbols_soft(
            xp.ones(1), "qam", 4, 0.1, method="magic", vectorized=False
        )
