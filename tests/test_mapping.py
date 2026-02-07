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
