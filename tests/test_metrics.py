"""Tests for performance metrics module."""

from commstools import metrics
from commstools.impairments import add_awgn
from commstools.utils import random_symbols


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
    noisy = add_awgn(symbols, esn0_db=target_snr_db, sps=1)

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


def test_q_factor_from_ber(backend_device, xp):
    """Q-factor from BER follows erfc relationship."""
    # BER = 1e-9 -> Q ≈ 6
    ber_val = 1e-9
    q = metrics.q_factor(ber_value=ber_val)

    # Q should be approximately 6 for BER=1e-9
    assert 5.5 < q < 6.5


def test_q_factor_from_evm(backend_device, xp):
    """Q-factor from EVM approximation."""
    # EVM = 10% -> Q ≈ 10
    evm_pct = 10.0
    q = metrics.q_factor(evm_percent=evm_pct)

    assert abs(q - 10.0) < 0.1


def test_q_factor_zero_ber(backend_device, xp):
    """Zero BER should give infinite Q-factor."""
    q = metrics.q_factor(ber_value=0.0)

    assert q == float("inf")


def test_q_factor_db(backend_device, xp):
    """Q-factor dB conversion."""
    # EVM = 10% -> Q = 10 -> Q_dB = 20 * log10(10) = 20
    q_db = metrics.q_factor_db(evm_percent=10.0)

    expected_db = 20.0
    assert abs(q_db - expected_db) < 0.1


def test_signal_evm_method(backend_device, xp):
    """Test Signal.evm() method using source_symbols as reference."""
    from commstools.core import Signal

    # Create signal with known source_symbols
    sig = Signal.qam(
        order=4, num_symbols=100, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Without noise, EVM should be 0
    evm_pct, evm_db = sig.evm()
    assert evm_pct < 1e-5  # Near-zero EVM for perfect signal


def test_signal_ber_method(backend_device, xp):
    """Test Signal.ber() method using source_bits as reference."""
    from commstools.core import Signal

    # Create signal with known source_bits
    sig = Signal.qam(
        order=4, num_symbols=100, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Without noise, BER should be 0
    ber_val = sig.ber()
    assert ber_val == 0.0  # Perfect signal, no errors


def test_signal_demap_hard(backend_device, xp):
    """Test Signal.demap() hard decision."""
    from commstools.core import Signal

    sig = Signal.qam(
        order=4, num_symbols=50, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Demap to bits
    bits = sig.demap(hard=True)

    # Should match source_bits
    assert xp.array_equal(bits.flatten(), sig.source_bits.flatten())


def test_q_factor_modulation_aware(backend_device, xp):
    """Verify modulation-aware Q-factor calculation using d_min."""
    # EVM = 10%
    evm_pct = 10.0

    # 1. PSK-16
    q_psk = metrics.q_factor(evm_percent=evm_pct, modulation="psk", order=16)
    # d_min_psk = 2 * sin(pi/16) approx 2 * 0.195 = 0.39
    # Q = d_min / (2 * 0.1) approx 0.39 / 0.2 = 1.95
    assert 1.9 < q_psk < 2.0

    # 2. QAM-16 (Square)
    q_qam = metrics.q_factor(evm_percent=evm_pct, modulation="qam", order=16)
    # d_min_qam = sqrt(6/15) approx 0.63
    # Q = 0.63 / 0.2 = 3.15
    assert 3.1 < q_qam < 3.2

    # 3. Invalid params
    import pytest

    with pytest.raises(ValueError):
        metrics.q_factor()
