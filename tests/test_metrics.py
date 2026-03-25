"""Tests for performance metrics module."""

import numpy as np
import pytest

from commstools import metrics
from commstools.impairments import apply_awgn
from commstools.helpers import random_symbols


def test_evm_perfect_signal(backend_device, xp):
    """EVM should be 0% for identical tx/rx symbols."""
    symbols = xp.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

    evm_pct, evm_db = metrics.evm(symbols, symbols)

    assert evm_pct < 1e-10
    assert evm_db < -200  # Very negative dB for near-zero EVM


def test_evm_with_known_error(backend_device, xp):
    """EVM with known error magnitude."""
    # Reference: unit power symbols
    tx = xp.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j])
    # Add 10% error (0.1 magnitude error on unit symbols)
    rx = tx + 0.1

    evm_pct, _ = metrics.evm(rx, tx)

    # Error RMS = 0.1, Reference RMS = 1.0 -> EVM = 10%
    assert abs(evm_pct - 10.0) < 1.0  # Within 1% tolerance


def test_snr_matches_applied(backend_device, xp):
    """SNR estimate should approximately match applied AWGN level."""
    # Generate random QPSK symbols
    symbols = random_symbols(10000, "qam", 4, seed=42)

    # Apply known Es/N0
    target_snr_db = 20.0
    noisy = apply_awgn(symbols, esn0_db=target_snr_db, sps=1)

    estimated_snr = metrics.snr(noisy, symbols)

    # Should be within 1 dB of target
    assert abs(estimated_snr - target_snr_db) < 1.0, (
        f"SNR estimate {estimated_snr:.1f} dB far from target {target_snr_db:.1f} dB"
    )


def test_snr_high_snr(backend_device, xp):
    """Very high SNR should return very high estimate."""
    symbols = xp.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

    snr_db = metrics.snr(symbols, symbols)

    assert snr_db > 100  # Essentially infinite for identical signals


def test_ber_no_errors(backend_device, xp):
    """BER = 0 for identical bit sequences."""
    bits = xp.array([0, 1, 0, 1, 1, 0, 0, 1])

    ber_val = metrics.ber(bits, bits)

    assert ber_val == 0.0


def test_ber_known_errors(backend_device, xp):
    """BER calculation with known error count."""
    bits_tx = xp.array([0, 1, 0, 1, 1, 0, 0, 1])
    bits_rx = xp.array([1, 1, 0, 1, 0, 0, 0, 1])  # 2 errors (positions 0 and 4)

    ber_val = metrics.ber(bits_rx, bits_tx)

    expected_ber = 2 / 8  # 0.25
    assert abs(ber_val - expected_ber) < 1e-10


def test_ber_all_errors(backend_device, xp):
    """BER = 1 when all bits are flipped."""
    bits_tx = xp.array([0, 0, 0, 0])
    bits_rx = xp.array([1, 1, 1, 1])

    ber_val = metrics.ber(bits_rx, bits_tx)

    assert ber_val == 1.0


def test_signal_evm_method(backend_device, xp):
    """Test Signal.evm() method using source_symbols as reference."""
    from commstools.core import Signal

    # Create signal with known source_symbols
    sig = Signal.qam(
        order=4, num_symbols=100, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Must resolve symbols before EVM
    sig.resolve_symbols()
    evm_pct, evm_db = sig.evm()
    # Relaxed slightly for robustness logic which might have tiny epsilon effects
    assert evm_pct < 1e-4  # Near-zero EVM for perfect signal


def test_signal_ber_method(backend_device, xp):
    """Test Signal.ber() method using source_bits as reference."""
    from commstools.core import Signal

    # Create signal with known source_bits
    sig = Signal.qam(
        order=4, num_symbols=100, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Must resolve symbols then demap symbols before BER
    sig.resolve_symbols()
    sig.demap_symbols_hard()
    ber_val = sig.ber()
    assert ber_val == 0.0  # Perfect signal, no errors


def test_signal_demap_hard(backend_device, xp, xpt):
    """Test Signal.demap_symbols_hard() hard decision."""
    from commstools.core import Signal

    sig = Signal.qam(
        order=4, num_symbols=50, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Must resolve symbols before demapping
    sig.resolve_symbols()
    # Demap to bits
    sig.demap_symbols_hard()

    # Should match source_bits
    xpt.assert_array_equal(sig.resolved_bits.flatten(), sig.source_bits.flatten())


def test_metrics_more(backend_device, xp):
    """Verify more metrics edge cases."""
    # 1. EVM without normalization (now always normalized)
    # The previous test relied on normalize=False to check unnormalized behavior.
    # Since we removed that argument and enforce normalization, we test that
    # normalized inputs return 0% error even if scaled differently?
    # No, if ref and rx are different but scaled differently, they are both normalized
    # to unit power.
    # Original test: ref=[1,1], rx=[1.1, 1.1].
    # Normalized: ref -> [1/sqrt(2), 1/sqrt(2)] -> RMS=1.
    # rx -> [1.1/sqrt(2*1.1^2), ...] -> [1/sqrt(2), ...].
    # So normalized rx == normalized ref. EVM should be 0.
    ref = xp.array([1.0, 1.0])
    rx = xp.array([1.1, 1.1])
    evm_pct, _ = metrics.evm(rx, ref)
    # 0% because they are identical shapes, just scaled.
    assert evm_pct < 1e-5

    # 2. SNR divide by zero (returns inf in current implementation)
    zero = xp.zeros(10)
    snr_val = metrics.snr(zero, zero)
    assert snr_val == float("inf")

    # 3. BER mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.ber(xp.array([0, 1]), xp.array([0]))


def test_ber_empty(backend_device, xp):
    """Verify BER with empty arrays."""
    assert metrics.ber(xp.array([]), xp.array([])) == 0.0


def test_evm_near_zero_ref(backend_device, xp):
    """Verify EVM behavior when reference signal is near zero."""
    tx = xp.zeros(10)
    rx = xp.ones(10)
    pct, db = metrics.evm(rx, tx)
    assert pct == float("inf")
    assert db == float("inf")



def test_evm_shape_mismatch(backend_device, xp):
    """Verify error on shape mismatch in evm."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.evm(xp.zeros(10), xp.zeros(11))


def test_snr_shape_mismatch(backend_device, xp):
    """Verify error on shape mismatch in snr."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.snr(xp.zeros(10), xp.zeros(11))



def test_evm_array_handling(backend_device, xp):
    """Verify evm multichannel array handling."""
    # 2 channels, one with error, one perfect
    rx = xp.array([[1.0, 1.0], [1.0, 0.8]])
    tx = xp.array([[1.0, 1.0], [1.1, 1.1]])  # Ch 1 is [1.0, 1.0] after norm

    ep, edb = metrics.evm(rx, tx)
    assert ep.shape == (2,)
    assert ep[0] == 0
    assert ep[1] > 0

    # Test low power mask for array
    tx_zero = xp.zeros((2, 2))
    ep, edb = metrics.evm(rx, tx_zero)
    assert xp.all(ep == float("inf"))
    assert xp.all(edb == float("inf"))


def test_snr_scalar_low_power(backend_device, xp):
    """Verify snr returns -inf when reference power is zero."""
    rx = xp.ones(10)
    tx = xp.zeros(10)
    # Ref power is 0, noise is 1.0. SNR should be -inf dB.
    assert metrics.snr(rx, tx) == float("-inf")


def test_snr_array_low_power(backend_device, xp):
    """Verify snr array handling returns -inf per channel when reference power is zero."""
    rx = xp.ones((2, 10))
    tx = xp.zeros((2, 10))
    # Both channels have 0 signal ref
    res = metrics.snr(rx, tx)
    assert xp.all(res == float("-inf"))

    # Mixed case
    tx_mixed = xp.array([xp.ones(10), xp.zeros(10)])
    res = metrics.snr(rx, tx_mixed)
    assert res[0] == float("inf")  # Perfect match for Ch 0
    assert res[1] == float("-inf")  # 0 Signal for Ch 1


def test_ber_length_mismatch(backend_device, xp):
    """Verify error on bit length mismatch."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.ber(xp.array([1, 0]), xp.array([1, 0, 1]))


def test_ber_multichannel(backend_device, xp):
    """Verify multi-channel BER returns per-channel error rates."""
    # 2 channels, each with 10 bits
    tx = xp.array([[0, 1, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]])
    # Channel 0: 1 error at position 0, Channel 1: 2 errors at positions 0,1
    rx = xp.array([[1, 1, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]])

    ber_values = metrics.ber(rx, tx)

    # Should return array of shape (2,)
    assert ber_values.shape == (2,)
    assert float(ber_values[0]) == pytest.approx(1 / 10)
    assert float(ber_values[1]) == pytest.approx(2 / 10)


# =============================================================================
# GMI TESTS
# =============================================================================


def test_gmi_high_snr_approaches_log2m(backend_device, xp):
    """At infinite SNR (perfect LLRs), GMI → log2(M)."""
    from commstools.mapping import map_bits, compute_llr

    k = 4  # bits per symbol for QAM-16
    M = 16
    N = 200
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, N * k).astype("int32")
    symbols = map_bits(xp.asarray(bits), "qam", M)

    # Very low noise → large LLR magnitudes → GMI ≈ log2(M)
    # Use (N, k) shape so gmi() knows k and returns b/cu in [0, log2(M)]
    llrs = compute_llr(symbols, "qam", M, noise_var=1e-6, output="numpy").reshape(N, k)
    gmi_val = metrics.gmi(llrs, bits.reshape(N, k))

    assert gmi_val > np.log2(M) - 0.05, f"High-SNR GMI {gmi_val:.4f} too low"


def test_gmi_low_snr_approaches_zero(backend_device, xp):
    """At very low SNR, LLRs collapse to zero → GMI → 0."""
    from commstools.mapping import map_bits, compute_llr

    k = 4
    M = 16
    N = 500
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, N * k).astype("int32")
    symbols = map_bits(xp.asarray(bits), "qam", M)

    # Huge noise → all LLRs ≈ 0 → log2(1+1)=1 for every bit → GMI ≈ 0
    llrs = compute_llr(symbols, "qam", M, noise_var=1e6, output="numpy").reshape(N, k)
    gmi_val = metrics.gmi(llrs, bits.reshape(N, k))

    assert gmi_val < 0.2, f"Low-SNR GMI {gmi_val:.4f} should be near 0"


def test_gmi_flat_input_returns_per_bit(backend_device, xp):
    """Flat 1D input: gmi() treats k=1 and returns per-bit GMI in [0, 1]."""
    from commstools.mapping import map_bits, compute_llr

    bits = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype="int32")
    symbols = map_bits(xp.asarray(bits), "qam", 4)
    # Flat (N*k,) LLRs — k not known, so gmi() treats as k=1
    llrs = compute_llr(symbols, "qam", 4, noise_var=0.1, output="numpy")
    gmi_val = metrics.gmi(llrs, bits)

    assert 0.0 <= gmi_val <= 1.0


def test_gmi_returns_scalar_float(backend_device, xp):
    """gmi() must return a Python float."""
    from commstools.mapping import map_bits, compute_llr

    k = 2
    N = 4
    bits = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype="int32")
    symbols = map_bits(xp.asarray(bits), "qam", 4)
    llrs = compute_llr(symbols, "qam", 4, noise_var=0.1, output="numpy").reshape(N, k)
    gmi_val = metrics.gmi(llrs, bits.reshape(N, k))

    assert isinstance(gmi_val, float)


def test_gmi_shape_mismatch_raises(backend_device, xp):
    """gmi() should raise ValueError when llrs and tx_bits have different sizes."""
    llrs = np.array([1.0, -1.0, 2.0])
    bits = np.array([0, 1])
    with pytest.raises(ValueError, match="same number of elements"):
        metrics.gmi(llrs, bits)


def test_gmi_2d_bounded_by_log2m(backend_device, xp):
    """For (N, k) input, GMI ∈ [0, k]."""
    from commstools.mapping import map_bits, compute_llr

    k = 2  # QPSK
    N = 100
    rng = np.random.default_rng(55)
    bits = rng.integers(0, 2, N * k).astype("int32")
    symbols = map_bits(xp.asarray(bits), "qam", 4)
    llrs = compute_llr(symbols, "qam", 4, noise_var=0.1, output="numpy").reshape(N, k)
    bits_2d = bits.reshape(N, k)
    gmi_val = metrics.gmi(llrs, bits_2d)
    assert 0.0 <= gmi_val <= np.log2(4)


# =============================================================================
# MI TESTS
# =============================================================================


def test_mi_high_snr_approaches_log2m(backend_device, xp):
    """At high SNR, MI → log2(M)."""
    from commstools.mapping import gray_constellation

    M = 16
    const = gray_constellation("qam", M)
    rng = np.random.default_rng(42)
    # Sample uniformly from constellation (no noise)
    symbols = const[rng.integers(0, M, 500)]

    mi_val = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=1e-8)

    assert mi_val > np.log2(M) - 0.1, f"High-SNR MI {mi_val:.4f} too low"


def test_mi_never_exceeds_log2m(backend_device, xp):
    """MI ≤ log2(M) always (capacity bound)."""
    from commstools.mapping import gray_constellation

    M = 4
    const = gray_constellation("qam", M)
    rng = np.random.default_rng(7)
    symbols = const[rng.integers(0, M, 200)]

    for noise_var in [1e-4, 0.1, 1.0, 10.0]:
        mi_val = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=noise_var)
        assert mi_val <= np.log2(M) + 1e-6, (
            f"MI={mi_val:.4f} exceeded log2(M)={np.log2(M):.4f} at noise_var={noise_var}"
        )


def test_mi_returns_scalar_float(backend_device, xp):
    """mi() must return a Python float."""
    from commstools.mapping import gray_constellation

    M = 4
    const = gray_constellation("qam", M)
    symbols = const[:10]
    mi_val = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=0.1)
    assert isinstance(mi_val, float)


def test_mi_decreases_with_noise(backend_device, xp):
    """MI should decrease as noise increases."""
    from commstools.mapping import gray_constellation

    M = 16
    const = gray_constellation("qam", M)
    rng = np.random.default_rng(99)
    symbols = const[rng.integers(0, M, 500)]

    mi_low_noise = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=0.01)
    mi_high_noise = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=1.0)

    assert mi_low_noise > mi_high_noise
