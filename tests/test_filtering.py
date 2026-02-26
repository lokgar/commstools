"""Tests for digital filtering and pulse shaping tap generation."""

import numpy as np
import pytest

from commstools import filtering

# ============================================================================
# TAP GENERATOR TESTS — these functions always return NumPy arrays regardless
# of backend, so no backend parametrisation is needed.
# ============================================================================


def test_rrc_taps():
    """Verify Root Raised Cosine filter tap length and type."""
    taps = filtering.rrc_taps(sps=4, span=10, rolloff=0.35)
    assert isinstance(taps, np.ndarray)
    assert len(taps) == 10 * 4 + 1


def test_rrc_taps_rolloff_zero(backend_device, xp, xpt):
    """Verify zero-rolloff RRC taps match a normalised sinc function."""
    taps = filtering.rrc_taps(sps=4, rolloff=0, span=8)
    assert len(taps) > 0
    # Compare to normalised sinc on the same device
    t = xp.linspace(-4, 4, len(taps))
    expected = xp.sinc(t)
    expected = expected / xp.sqrt(xp.sum(expected**2))
    xpt.assert_allclose(xp.asarray(taps), expected, atol=1e-3)


def test_rc_taps():
    """Verify Raised Cosine tap length and unit-energy normalisation."""
    taps = filtering.rc_taps(sps=4, rolloff=0.5, span=8)
    assert isinstance(taps, np.ndarray)
    assert len(taps) == 8 * 4 + 1
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_rc_taps_rolloff_zero():
    """Verify zero-rolloff RC tap unit energy (brick-wall filter)."""
    taps = filtering.rc_taps(sps=4, rolloff=0.0, span=8)
    assert isinstance(taps, np.ndarray)
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_gaussian_taps():
    """Verify Gaussian filter tap length and unit-energy normalisation."""
    taps = filtering.gaussian_taps(sps=4, bt=0.3, span=4)
    assert isinstance(taps, np.ndarray)
    assert len(taps) == 4 * 4 + 1
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_smoothrect_taps():
    """Verify smoothrect tap length and unit-energy normalisation."""
    taps = filtering.smoothrect_taps(sps=8, span=4, bt=1.0)
    assert isinstance(taps, np.ndarray)
    assert len(taps) == 4 * 8 + 1
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def _freq_response(taps, nfft=1024):
    """Return (normalised_freqs, magnitude_response) for a tap array."""
    H = np.abs(np.fft.fft(taps, nfft))
    freqs = np.fft.fftfreq(nfft)  # cycles per sample, range [-0.5, 0.5)
    return freqs, H


def test_lowpass_taps():
    """Verify lowpass FIR: correct tap count, passband gain, and Nyquist attenuation."""
    taps = filtering.lowpass_taps(num_taps=63, cutoff=0.2, sampling_rate=1.0)
    assert len(taps) == 63

    freqs, H = _freq_response(taps)
    # DC bin should pass (gain ≈ 1)
    assert H[0] > 0.99, f"DC gain too low: {H[0]:.4f}"
    # Nyquist should be heavily attenuated (Hamming window ≥ 40 dB stopband)
    nyquist_idx = len(H) // 2
    assert H[nyquist_idx] < 0.05, f"Nyquist not attenuated: {H[nyquist_idx]:.4f}"


def test_highpass_taps():
    """Verify highpass FIR: correct tap count, DC attenuation, and passband gain."""
    taps = filtering.highpass_taps(num_taps=63, cutoff=0.2, sampling_rate=1.0)
    assert len(taps) == 63

    freqs, H = _freq_response(taps)
    # DC bin should be blocked
    assert H[0] < 0.05, f"DC not attenuated: {H[0]:.4f}"
    # Nyquist bin should pass
    nyquist_idx = len(H) // 2
    assert H[nyquist_idx] > 0.95, f"Nyquist gain too low: {H[nyquist_idx]:.4f}"


def test_bandpass_taps():
    """Verify bandpass FIR: correct tap count, centre gain, and out-of-band attenuation."""
    low, high = 0.15, 0.35
    taps = filtering.bandpass_taps(
        num_taps=63, low_cutoff=low, high_cutoff=high, sampling_rate=1.0
    )
    assert len(taps) == 63

    freqs, H = _freq_response(taps)
    # Centre of passband (normalised freq 0.25)
    centre_idx = int(0.25 * len(H))
    assert H[centre_idx] > 0.90, f"Passband centre gain too low: {H[centre_idx]:.4f}"
    # DC should be blocked
    assert H[0] < 0.05, f"DC not attenuated: {H[0]:.4f}"
    # Nyquist should be blocked
    nyquist_idx = len(H) // 2
    assert H[nyquist_idx] < 0.05, f"Nyquist not attenuated: {H[nyquist_idx]:.4f}"


def test_bandstop_taps():
    """Verify bandstop FIR: correct tap count, notch rejection, and passband preservation."""
    low, high = 0.15, 0.35
    taps = filtering.bandstop_taps(
        num_taps=63, low_cutoff=low, high_cutoff=high, sampling_rate=1.0
    )
    assert len(taps) == 63

    freqs, H = _freq_response(taps)
    # Centre of stopband should be rejected
    centre_idx = int(0.25 * len(H))
    assert H[centre_idx] < 0.1, f"Notch not deep enough: {H[centre_idx]:.4f}"
    # DC should pass
    assert H[0] > 0.95, f"DC gain too low: {H[0]:.4f}"
    # Nyquist should pass
    nyquist_idx = len(H) // 2
    assert H[nyquist_idx] > 0.95, f"Nyquist gain too low: {H[nyquist_idx]:.4f}"


# ============================================================================
# DISPATCHING FILTER TESTS — these functions operate on the active backend,
# so they require backend parametrisation.
# ============================================================================


def test_fir_filter(backend_device, xp, xpt):
    """Verify FIR filtering output device, shape, and moving-average correctness."""
    data = xp.ones(100)
    taps = xp.ones(5) / 5.0  # Moving average

    filtered = filtering.fir_filter(data, taps)

    assert isinstance(filtered, xp.ndarray)
    assert len(filtered) == len(data)
    # Interior samples of a DC signal through a moving average must stay at 1
    xpt.assert_allclose(filtered[5:-5], xp.ones(90))


def test_matched_filter_normalization(backend_device, xp):
    """Verify matched_filter respects taps_normalization and normalize_output options."""
    # matched_filter dispatches on its input, so use xp arrays throughout
    samples = xp.ones(100)
    pulse = xp.ones(10)

    # Unity-gain normalisation
    out_gain = filtering.matched_filter(samples, pulse, taps_normalization="unity_gain")
    assert out_gain.shape == (100,)
    assert isinstance(out_gain, xp.ndarray)

    # Output amplitude normalisation — peak should be 1
    out_max = filtering.matched_filter(samples, pulse, normalize_output=True)
    assert float(xp.max(xp.abs(out_max))) == pytest.approx(1.0)

    # Invalid normalisation mode must raise
    with pytest.raises(
        ValueError,
        match="Unknown taps_normalization",
    ):
        filtering.matched_filter(samples, pulse, taps_normalization="magic")


def test_shape_pulse_variants(backend_device, xp):
    """Verify shape_pulse produces correct lengths for RC and sinc shapes."""
    symbols = xp.array([1, -1, 1, -1])

    res_rc = filtering.shape_pulse(symbols, sps=4, pulse_shape="rc")
    assert len(res_rc) == 16

    res_sinc = filtering.shape_pulse(symbols, sps=4, pulse_shape="sinc")
    assert len(res_sinc) == 16

    with pytest.raises(ValueError, match="Not implemented pulse shape"):
        filtering.shape_pulse(symbols, sps=4, pulse_shape="magic")


def test_smoothrect_pulse(backend_device, xp):
    """Verify smoothrect pulse shaping output length."""
    symbols = xp.array([1, 1])
    res = filtering.shape_pulse(symbols, sps=8, pulse_shape="smoothrect", filter_span=4)
    assert len(res) == 16


def test_shape_pulse_none_with_rz(backend_device, xp):
    """Verify shape_pulse with pulse_shape='none' and rz=True expands using rect."""
    symbols = xp.array([1, -1, 1], dtype=xp.complex64)
    result = filtering.shape_pulse(symbols, sps=4, pulse_shape="none", rz=True)
    assert result is not None
    assert len(result) > 0
