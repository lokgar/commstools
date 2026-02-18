import pytest
import numpy as np

from commstools import mapping


def _to_np(arr):
    """Convert any array (NumPy, CuPy, JAX) to a plain NumPy array."""
    if hasattr(arr, "get"):  # CuPy
        return arr.get()
    return np.asarray(arr)


def test_qam_mapping(xp):
    """Verify QAM mapping produces the correct number of symbols."""
    # Use xp to create array on device
    bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1])

    syms = mapping.map_bits(bits, modulation="qam", order=16)

    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2


def test_psk_mapping(xp):
    """Verify PSK mapping produces the correct number of symbols."""
    bits = xp.array([0, 1])

    syms = mapping.map_bits(bits, modulation="psk", order=2)

    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2


def test_demap_dimensions_mimo(xp):
    """Test that demap_symbols_hard preserves multidimensional structure."""
    modulation = "qam"
    order = 4

    # Creates input shape (2 streams, 4 symbols)
    # Total symbols = 8
    # Total bits = 16
    bits_in = xp.zeros(16, dtype="int32")
    bits_in[::2] = 1

    # Map to symbols (flat) first
    symbols_flat = mapping.map_bits(bits_in, modulation, order)

    # Reshape to (2, 4) to simulate MIMO
    symbols_mimo = symbols_flat.reshape(2, 4)

    # Demap
    bits_out = mapping.demap_symbols_hard(symbols_mimo, modulation, order)

    # Verify strict shape compliance
    # We expect (2, 4 * 2) = (2, 8)
    expected_shape = (2, 8)
    assert bits_out.shape == expected_shape
    # Flatten both to compare content (bits_out is flat if reshape logic failed, but shape check guards it)
    assert xp.array_equal(bits_out.flatten(), bits_in)


def test_compute_llr_sign_correctness(xp):
    """LLR sign should match hard decision at high SNR (noiseless symbols)."""
    modulation = "qam"
    order = 16

    bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order)

    llrs = mapping.compute_llr(
        symbols, modulation, order, noise_var=1e-6, method="maxlog"
    )

    hard_from_llr = (np.asarray(llrs) < 0).astype("int32")
    assert np.array_equal(hard_from_llr, _to_np(bits)), (
        "LLR signs don't match original bits"
    )


def test_compute_llr_roundtrip(xp):
    """Hard decision from LLR should match direct hard demapping at high SNR."""
    modulation = "psk"
    order = 8

    bits = xp.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order)

    llrs = mapping.compute_llr(symbols, modulation, order, noise_var=1e-6)
    hard_from_llr = (np.asarray(llrs) < 0).astype("int32")
    hard_direct = _to_np(mapping.demap_symbols_hard(symbols, modulation, order))

    assert np.array_equal(hard_from_llr, hard_direct)


def test_compute_llr_exact_vs_maxlog(xp):
    """Exact and max-log methods should agree on sign and be close in magnitude."""
    modulation = "qam"
    order = 4

    bits = xp.array([0, 0, 0, 1, 1, 0, 1, 1], dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order)

    llrs_maxlog = np.asarray(
        mapping.compute_llr(symbols, modulation, order, noise_var=0.01, method="maxlog")
    )
    llrs_exact = np.asarray(
        mapping.compute_llr(symbols, modulation, order, noise_var=0.01, method="exact")
    )

    assert np.array_equal(np.sign(llrs_maxlog), np.sign(llrs_exact))

    # Max-log is an approximation; values should be within a reasonable ratio
    ratio = np.abs(llrs_maxlog) / (np.abs(llrs_exact) + 1e-10)
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)


def test_compute_llr_mimo_shape(xp):
    """compute_llr should preserve MIMO channel structure."""
    modulation = "qam"
    order = 4

    bits = xp.zeros(16, dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order).reshape(2, 4)

    llrs = mapping.compute_llr(symbols, modulation, order, noise_var=0.1)

    assert llrs.shape == (2, 8)


def test_gray_code_edge_cases():
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


def test_8qam_mapping(xp):
    """Verify 8-QAM (rectangular) mapping and round-trip."""
    bits = xp.array([0, 0, 0, 1, 1, 1], dtype="int32")
    syms = mapping.map_bits(bits, modulation="qam", order=8)
    assert len(syms) == 2

    bits_out = mapping.demap_symbols_hard(syms, modulation="qam", order=8)
    assert xp.array_equal(bits, bits_out)


def test_cross_qam_32_mapping(xp):
    """Verify 32-QAM (Cross) mapping and round-trip."""
    # 5 bits per symbol. 2 symbols = 10 bits.
    bits = xp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype="int32")
    syms = mapping.map_bits(bits, modulation="qam", order=32)
    assert len(syms) == 2

    bits_out = mapping.demap_symbols_hard(syms, modulation="qam", order=32)
    assert xp.array_equal(bits, bits_out)


def test_compute_llr_methods_agree(xp):
    """Verify maxlog and exact methods agree on sign for 16-QAM."""
    modulation = "qam"
    order = 16
    bits = xp.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order)
    noise_var = 0.1

    llrs_maxlog = np.asarray(
        mapping.compute_llr(symbols, modulation, order, noise_var, method="maxlog")
    )
    llrs_exact = np.asarray(
        mapping.compute_llr(symbols, modulation, order, noise_var, method="exact")
    )

    assert np.array_equal(np.sign(llrs_exact), np.sign(llrs_maxlog))

    with pytest.raises(ValueError, match="Unknown method"):
        mapping.compute_llr(symbols, modulation, order, noise_var, method="unknown")


def test_gray_constellation_advanced(xp):
    """Verify constellation generation edge cases."""
    # 1. Unipolar via argument
    const_unipol = mapping.gray_constellation("pam", 4, unipolar=True)
    assert xp.min(const_unipol) >= 0

    # 2. Custom scheme check (now simple string comparison)
    # Note: 'my-pam' no longer automatically resolves to 'ask' unless we use 'pam' or 'ask'
    # Actually gray_constellation uses "ask" in modulation.
    const_custom = mapping.gray_constellation("pam", 4)
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


def test_map_demap_unipolar(xp):
    """Verify bit mapping and demapping with unipolar ASK/PAM."""
    bits = xp.array([0, 1, 1, 0])
    # Map to unipolar
    syms = mapping.map_bits(bits, "ask", 4, unipolar=True)
    assert xp.all(syms >= 0)

    # Demap from unipolar
    bits_rx = mapping.demap_symbols_hard(syms, "ask", 4, unipolar=True)
    assert xp.array_equal(bits, bits_rx)

    # LLR signs should recover the same bits
    llrs = mapping.compute_llr(syms, "ask", 4, noise_var=0.1, unipolar=True)
    bits_soft = (np.asarray(llrs) < 0).astype("int32")
    assert np.array_equal(bits_soft, _to_np(bits))


def test_mapping_more(xp):
    """Verify more mapping edge cases."""
    # Order error in map_bits
    with pytest.raises(ValueError, match="at least 2"):
        mapping.map_bits(xp.array([0, 0]), "psk", 1)

    # Order error in demap_symbols_hard
    with pytest.raises(ValueError, match="at least 2"):
        mapping.demap_symbols_hard(xp.array([0]), "psk", 1)


def test_mapping_order_errors(xp):
    """Verify order validation for power-of-2 requirements."""
    # map_bits
    with pytest.raises(ValueError, match="power of 2"):
        mapping.map_bits(xp.array([0, 0]), "psk", 3)

    # compute_llr
    with pytest.raises(ValueError, match="power of 2"):
        mapping.compute_llr(xp.array([0.1]), "psk", 3, noise_var=0.1)


def test_demap_symbols_empty_shape(xp):
    """Verify demapping behavior for effectively scalar inputs."""
    # 0-dim array (scalar)
    s = xp.array(1.0)
    # constellation will be [-1, 1] for psk-2
    # index 0 is -1, index 1 is 1.
    # so 1.0 -> index 1 -> bit 1.
    bits = mapping.demap_symbols_hard(s, "psk", 2)
    assert bits.size == 1
    assert int(bits.item()) == 1


def test_gray_code_zero():
    """Verify Gray code for 0 bits."""
    assert np.array_equal(mapping.gray_code(0), [0])
    assert np.array_equal(mapping.gray_to_binary(0), [0])


def test_gray_code_negative():
    """Verify error for negative bits."""
    with pytest.raises(ValueError, match="n must be non-negative"):
        mapping.gray_code(-1)
    with pytest.raises(ValueError, match="n must be non-negative"):
        mapping.gray_to_binary(-1)


def test_constellation_unsupported():
    """Verify error for unknown modulation type."""
    with pytest.raises(ValueError, match="Unsupported modulation type"):
        mapping.gray_constellation("chaos", 4)


def test_constellation_order_error():
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


def test_constellation_unsupported_string():
    """Test that unsupported strings raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported modulation type: custom-unknown"):
        mapping.gray_constellation("custom-unknown", 4)


def test_qam_cross_fallback():
    """Trigger the fallback in cross-QAM for small N."""
    # _gray_qam_cross(8) -> n=2, m=1. width=4, height=2.
    # n < 3 triggers n_shift=0 branch
    from commstools.mapping import _gray_qam_cross

    res = _gray_qam_cross(8)
    assert res.shape == (8,)


def test_map_bits_divisibility(xp):
    """Verify error when bit count is not divisible by bits per symbol."""
    bits = xp.array([1, 0, 1])  # 3 bits
    with pytest.raises(ValueError, match="must be divisible by bits per symbol"):
        mapping.map_bits(bits, "qam", 16)  # bits_per_symbol = 4


def test_map_bits_fixed_dtypes(xp):
    """Verify map_bits returns complex64 for PSK/QAM and float32 for ASK/PAM."""
    bits = xp.array([0, 1, 0, 1])

    out_ask = mapping.map_bits(bits, "ask", 4)
    assert out_ask.dtype == "float32"

    out_pam = mapping.map_bits(bits, "pam", 4)
    assert out_pam.dtype == "float32"

    out_qam = mapping.map_bits(bits, "qam", 4)
    assert out_qam.dtype == "complex64"

    out_psk = mapping.map_bits(bits, "psk", 4)
    assert out_psk.dtype == "complex64"


def test_demap_symbols_returns_int8(xp):
    """Verify demap_symbols_hard returns int8 bits matching source_bits dtype."""
    bits = xp.array([0, 1, 0, 1], dtype="int8")
    symbols = mapping.map_bits(bits, "qam", 4)
    demapped = mapping.demap_symbols_hard(symbols, "qam", 4)
    assert demapped.dtype == "int8"


def test_compute_llr_invalid_order(xp):
    """Verify error for non-power-of-2 order."""
    with pytest.raises(ValueError, match="Order must be a power of 2"):
        mapping.compute_llr(xp.ones(1), "qam", 6, 0.1)


def test_compute_llr_invalid_method(xp):
    """Verify error for unknown method."""
    with pytest.raises(ValueError, match="Unknown method"):
        mapping.compute_llr(xp.ones(1), "qam", 4, 0.1, method="magic")


# === compute_llr always-JAX output and differentiability tests ===


def test_compute_llr_numpy_input_returns_jax():
    """NumPy input should return a JAX array."""
    jax = pytest.importorskip("jax")

    bits = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, "qam", 16)

    llrs = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1)

    assert isinstance(llrs, jax.Array), (
        "compute_llr must return jax.Array for NumPy input"
    )


def test_compute_llr_cupy_input_returns_jax():
    """CuPy input should return a JAX array."""
    jax = pytest.importorskip("jax")
    cp = pytest.importorskip("cupy")

    bits = cp.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, "qam", 16)

    llrs = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1)

    assert isinstance(llrs, jax.Array), (
        "compute_llr must return jax.Array for CuPy input"
    )


def test_compute_llr_jax_input_returns_jax():
    """JAX array input should return a JAX array (maxlog)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    bits_np = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols_jax = jnp.asarray(mapping.map_bits(bits_np, "qam", 16))

    llrs = mapping.compute_llr(symbols_jax, "qam", 16, noise_var=0.1, method="maxlog")

    assert isinstance(llrs, jax.Array)
    hard = (np.asarray(llrs) < 0).astype("int32")
    assert np.array_equal(hard, bits_np)


def test_compute_llr_jax_exact_returns_jax():
    """JAX array input should return a JAX array (exact)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    bits_np = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype="int32")
    symbols_jax = jnp.asarray(mapping.map_bits(bits_np, "qam", 16))

    llrs = mapping.compute_llr(symbols_jax, "qam", 16, noise_var=0.01, method="exact")

    assert isinstance(llrs, jax.Array)
    hard = (np.asarray(llrs) < 0).astype("int32")
    assert np.array_equal(hard, bits_np)


def test_compute_llr_gradient():
    """jax.grad through LLRs w.r.t. input symbols should produce finite gradients."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    symbols_jax = jnp.array([0.7 + 0.7j, -0.7 - 0.7j], dtype=jnp.complex64)

    def loss_fn(syms):
        llrs = mapping.compute_llr(syms, "qam", 4, 0.1, method="maxlog")
        return jnp.sum(llrs**2)

    grad = jax.grad(loss_fn)(symbols_jax)

    assert grad.shape == symbols_jax.shape
    assert jnp.all(jnp.isfinite(grad))
    assert not jnp.all(grad == 0)


def test_compute_llr_numpy_vs_jax_input_agree():
    """NumPy and JAX inputs should produce numerically identical LLRs."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    bits = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols_np = mapping.map_bits(bits, "psk", 8)

    for method in ("maxlog", "exact"):
        llrs_from_np = mapping.compute_llr(symbols_np, "psk", 8, 0.05, method=method)
        llrs_from_jax = mapping.compute_llr(
            jnp.asarray(symbols_np), "psk", 8, 0.05, method=method
        )
        np.testing.assert_allclose(
            np.asarray(llrs_from_np), np.asarray(llrs_from_jax), atol=1e-5
        )
