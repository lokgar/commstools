from commstools import mapping


def test_qam_mapping(backend_device, xp):
    # Use xp to create array on device
    bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1])

    syms = mapping.map_bits(bits, modulation="qam", order=16)

    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2


def test_psk_mapping(backend_device, xp):
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
    k = 4  # log2(16)

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
    k = 3

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
    k = 2

    # 2 streams, 4 symbols each
    bits = xp.zeros(16, dtype=xp.int32)
    symbols = mapping.map_bits(bits, modulation, order).reshape(2, 4)

    noise_var = 0.1
    llrs = mapping.demap_symbols_soft(symbols, modulation, order, noise_var)

    # Expected shape: (2, 4*2) = (2, 8)
    expected_shape = (2, 8)
    assert llrs.shape == expected_shape
