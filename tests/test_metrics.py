"""Tests for performance metrics module."""

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


def test_signal_demap_hard(backend_device, xp):
    """Test Signal.demap_symbols_hard() hard decision."""
    from commstools.core import Signal

    sig = Signal.qam(
        order=4, num_symbols=50, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Must resolve symbols before demapping
    sig.resolve_symbols()
    # Demap to bits
    bits = sig.demap_symbols_hard()

    # Should match source_bits
    assert xp.array_equal(bits.flatten(), sig.source_bits.flatten())


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


def test_metrics_signal_objects(backend_device, xp):
    """Verify metrics work with Signal objects and resolve metadata."""
    from commstools.core import Signal

    tx = Signal.qam(order=4, num_symbols=100, sps=1, symbol_rate=1e6)
    rx = tx.copy()

    # Test EVM
    metrics.evm(rx, tx)

    # Test SNR
    metrics.snr(rx, tx)


def test_evm_shape_mismatch(backend_device, xp):
    """Verify error on shape mismatch in evm."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.evm(xp.zeros(10), xp.zeros(11))


def test_snr_shape_mismatch(backend_device, xp):
    """Verify error on shape mismatch in snr."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.snr(xp.zeros(10), xp.zeros(11))


def test_evm_signal_extraction(backend_device, xp):
    """Verify evm correctly extracts data from Signal objects."""
    from commstools.core import Signal

    s1 = Signal(samples=xp.ones(10), sampling_rate=1, symbol_rate=1)
    s2 = Signal(samples=xp.ones(10), sampling_rate=1, symbol_rate=1)

    # Symbols match, EVM should be 0
    ep, edb = metrics.evm(s1, s2)
    assert ep == 0
    assert edb == float("-inf")


def test_evm_array_handling(backend_device, xp):
    """Verify evm multichannel array handling (lines 135-143)."""
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
    """Verify snr returns -inf for scalar zero reference (line 238)."""
    rx = xp.ones(10)
    tx = xp.zeros(10)
    # Ref power is 0, noise is 1.0. SNR should be -inf dB.
    assert metrics.snr(rx, tx) == float("-inf")


def test_snr_array_low_power(backend_device, xp):
    """Verify snr array handling for zero reference (lines 243-246)."""
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
    """Verify multi-channel BER logging path (lines 306-311)."""
    # 2 channels, each with 10 bits
    tx = xp.array([[0, 1, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]])
    # Channel 0: 1 error at position 0, Channel 1: 2 errors at positions 0,1
    rx = xp.array([[1, 1, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]])

    ber_values = metrics.ber(rx, tx)

    # Should return array of shape (2,)
    assert ber_values.shape == (2,)
    assert float(ber_values[0]) == pytest.approx(1 / 10)
    assert float(ber_values[1]) == pytest.approx(2 / 10)
