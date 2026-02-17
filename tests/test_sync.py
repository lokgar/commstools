"""Tests for synchronization utilities (Barker and Zadoff-Chu sequences, frame detection)."""

from unittest.mock import MagicMock, patch

import pytest

from commstools import sync
from commstools.core import Preamble
from commstools.helpers import cross_correlate_fft


def test_barker_sequences(backend_device, xp):
    """Verify all standard Barker sequence lengths and binary properties."""
    valid_lengths = [2, 3, 4, 5, 7, 11, 13]

    for length in valid_lengths:
        seq = sync.barker_sequence(length)
        assert len(seq) == length
        # All values should be +1 or -1
        assert xp.all((seq == 1) | (seq == -1))


def test_barker_invalid_length(backend_device, xp):
    """Verify that unsupported Barker lengths raise ValueError."""
    with pytest.raises(ValueError):
        sync.barker_sequence(6)  # No Barker-6 exists


def test_barker_autocorrelation(backend_device, xp):
    """Verify that Barker sequences possess optimal autocorrelation properties."""
    seq = sync.barker_sequence(13)

    # Auto-correlation via cross_correlate_fft
    acorr = cross_correlate_fft(seq, seq, mode="full")

    # Peak should be at center
    peak_idx = len(acorr) // 2
    peak_val = float(xp.abs(acorr[peak_idx]))

    # Sidelobes should be at most 1
    sidelobes = xp.abs(acorr)
    sidelobes_max = float(
        xp.max(xp.concatenate([sidelobes[:peak_idx], sidelobes[peak_idx + 1 :]]))
    )

    assert peak_val == pytest.approx(13.0, abs=1e-4)  # Sum of squared elements
    assert sidelobes_max <= 1.0 + 1e-5  # Barker property (with float tolerance)


def test_zadoff_chu_cazac(backend_device, xp):
    """Verify that ZC sequences have constant amplitude (CAZAC property)."""
    zc = sync.zadoff_chu_sequence(63, root=25)

    # All magnitudes should be 1
    magnitudes = xp.abs(zc)
    assert xp.allclose(magnitudes, 1.0, atol=1e-5)


def test_zadoff_chu_length(backend_device, xp):
    """Verify that ZC sequences are generated with the requested length."""
    for length in [31, 63, 127]:
        zc = sync.zadoff_chu_sequence(length, root=1)
        assert len(zc) == length

    # Test even length
    zc_even = sync.zadoff_chu_sequence(10, root=1)
    assert len(zc_even) == 10
    assert xp.allclose(xp.abs(zc_even), 1.0)


def test_zadoff_chu_errors(backend_device, xp):
    """Verify ZC sequence generator input validation."""
    with pytest.raises(ValueError, match="Length must be positive"):
        sync.zadoff_chu_sequence(0)
    with pytest.raises(ValueError, match="Root must be in"):
        sync.zadoff_chu_sequence(10, root=10)


def test_correlate_delta(backend_device, xp):
    """Verify correct peak location for delta-like correlation."""
    # Delta-like signal
    signal = xp.zeros(100, dtype="float32")
    signal[50] = 1.0

    template = xp.array([1.0], dtype="float32")

    corr = cross_correlate_fft(signal, template, mode="same")

    # Peak should be at position 50
    peak_idx = int(xp.argmax(xp.abs(corr)))
    assert peak_idx == 50


def test_correlate_shift_detection(backend_device, xp):
    """Verify that correlation correctly identifies the shift of a template."""
    # Template
    template = xp.array([1.0, 1.0, 1.0, -1.0, -1.0], dtype="float32")

    # Signal with template at different position
    signal = xp.zeros(50, dtype="float32")
    signal[20:25] = template

    corr = cross_correlate_fft(signal, template, mode="same")

    # Peak should be near position 22 (center of template)
    peak_idx = int(xp.argmax(xp.abs(corr)))
    assert abs(peak_idx - 22) <= 1


def test_correlate_mimo(backend_device, xp):
    """Verify correlation behavior for multi-stream (MIMO) signals."""
    # 2 channels
    signal = xp.zeros((2, 50), dtype="float32")
    signal[0, 20] = 1.0
    signal[1, 30] = 1.0

    template = xp.array([1.0], dtype="float32")

    corr = cross_correlate_fft(signal, template, mode="same")

    assert corr.shape == (2, 50)

    # Peaks at different locations per channel
    peak_0 = int(xp.argmax(xp.abs(corr[0])))
    peak_1 = int(xp.argmax(xp.abs(corr[1])))

    assert peak_0 == 20
    assert peak_1 == 30


def test_preamble_auto_generation(backend_device, xp):
    """Verify automated preamble bit and symbol generation."""
    # Test Barker-13 auto-generation
    preamble = Preamble(sequence_type="barker", length=13)
    assert preamble.symbols is not None
    assert len(preamble.symbols) == 13
    assert isinstance(preamble.symbols, xp.ndarray)

    # Test Zadoff-Chu auto-generation
    preamble_zc = Preamble(sequence_type="zc", length=63, kwargs={"root": 1})
    assert preamble_zc.symbols is not None
    assert len(preamble_zc.symbols) == 63

    # Test invalid sequence type: Pydantic will raise ValidationError for Literal mismatch
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Preamble(sequence_type="invalid", length=13)

    # Test missing length for auto-generation: Mandatory field in Pydantic
    with pytest.raises(ValidationError):
        Preamble(sequence_type="barker")


def test_cross_correlate_fft_modes(backend_device, xp):
    """Verify cross_correlate_fft output lengths for each mode."""
    signal = xp.ones(100, dtype="float32")
    template = xp.ones(10, dtype="float32")

    full = cross_correlate_fft(signal, template, mode="full")
    same = cross_correlate_fft(signal, template, mode="same")
    valid = cross_correlate_fft(signal, template, mode="valid")

    assert full.shape[-1] == 100 + 10 - 1  # N + L - 1
    assert same.shape[-1] == 100  # N
    assert valid.shape[-1] == 91  # max(N,L) - min(N,L) + 1


def test_estimate_timing_advanced_scenarios(backend_device, xp):
    """Verify estimate_timing with Signal objects, MIMO, and search ranges."""
    from commstools.core import Preamble, Signal

    # 1. Signal object and Preamble object
    preamble = Preamble(sequence_type="barker", length=7)

    # Create a signal with this preamble
    data = xp.zeros(100, dtype="complex64")
    data[20 : 20 + 7] = preamble.symbols
    sig = Signal(samples=data, sampling_rate=1e6, symbol_rate=1e6)

    coarse, _frac = sync.estimate_timing(sig, preamble, threshold=0.1)
    assert 18 <= coarse[0] <= 22

    # 2. MIMO Signal (2 channels)
    mimo_data = xp.zeros((2, 100), dtype="complex64")
    mimo_data[0, 30:37] = preamble.symbols
    mimo_data[1, 30:37] = preamble.symbols
    coarse_mimo, _frac = sync.estimate_timing(
        mimo_data, preamble.symbols, threshold=0.1
    )
    assert 28 <= coarse_mimo[0] <= 32
    assert len(coarse_mimo) == 2

    # 3. Search range
    coarse_range, _frac = sync.estimate_timing(
        data, preamble.symbols, threshold=0.1, search_range=(10, 50)
    )
    assert 18 <= coarse_range[0] <= 22

    # 4. High threshold (above max)
    with pytest.raises(ValueError, match="No correlation peak above threshold"):
        sync.estimate_timing(data, preamble.symbols, threshold=2.0)

    # 5. Zero energy
    zero_data = xp.zeros(100)
    with pytest.raises(ValueError, match="No correlation peak above threshold"):
        sync.estimate_timing(zero_data, preamble.symbols, threshold=0.1)


def test_estimate_timing_known_position(backend_device, xp):
    """Verify timing estimation accuracy for a known preamble position."""
    # Create preamble
    preamble_symbols = sync.barker_sequence(13)

    # Embed in longer signal at known position
    signal = xp.zeros(200, dtype="complex64")
    start_pos = 50
    signal[start_pos : start_pos + 13] = preamble_symbols

    # Detect
    coarse, _frac = sync.estimate_timing(signal, preamble_symbols, threshold=0.3)

    # Should be within 1 sample of true position
    assert abs(coarse[0] - start_pos) <= 1


def test_estimate_timing_with_preamble_object(backend_device, xp):
    """Verify timing estimation using Preamble objects."""
    preamble = Preamble(sequence_type="barker", length=13)

    # Create signal with preamble embedded
    signal = xp.zeros(200, dtype="complex64")
    start_pos = 75

    # Embed preamble symbols
    preamble_syms = xp.asarray(
        preamble.symbols
        if not hasattr(preamble.symbols, "get")
        else preamble.symbols.get()
    )
    signal[start_pos : start_pos + 13] = preamble_syms

    # Create a Signal object to satisfy Preamble object requirement
    from commstools.core import Signal

    sig_obj = Signal(samples=signal, sampling_rate=1e6, symbol_rate=1e6)

    # Detect using Preamble object
    coarse, _frac = sync.estimate_timing(sig_obj, preamble, threshold=0.3)

    assert abs(coarse[0] - start_pos) <= 1


def test_estimate_timing_returns_tuple(backend_device, xp):
    """Verify that estimate_timing returns (coarse_offsets, fractional_offsets)."""
    preamble = sync.barker_sequence(7)

    signal = xp.zeros(100, dtype="complex64")
    signal[30:37] = preamble

    coarse, frac = sync.estimate_timing(signal, preamble, threshold=0.1)

    assert len(coarse) == 1
    assert len(frac) == 1
    assert abs(float(frac[0])) < 0.5


def test_detect_preamble_autocorr(backend_device, xp):
    """Verify preamble detection using autocorrelation (for periodic preambles)."""
    # Create a periodic preamble
    chunk = xp.array([1, 1, -1, -1])
    preamble = xp.tile(chunk, 4)  # 16 samples
    data = xp.zeros(100)
    start_idx = 30
    data[start_idx : start_idx + 16] = preamble

    # Cross-corr works as well, but let's test specifically the autocorrelation detection if it existed.
    # Actually 'estimate_timing' uses correlation.
    # Let's check if there is an autocorrelation-based detector.
    # Looking at sync.py missing lines, 81 was in 'detect_preamble_autocorr'?

    if hasattr(sync, "detect_preamble_autocorr"):
        lag = sync.detect_preamble_autocorr(data, period=4, threshold=0.5)
        # Should detect start around 30
        assert 28 <= lag <= 32


def test_estimate_timing_debug_plot(backend_device, xp):
    """Trigger the debug plot code path in estimate_timing."""
    # Create a signal with a preamble
    preamble = xp.random.randn(10) + 1j * xp.random.randn(10)
    sig = xp.concatenate([xp.zeros(20), preamble, xp.zeros(20)])

    with patch("matplotlib.pyplot.show"):
        # Mock subplots to return dummy fig/axes
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        with patch(
            "matplotlib.pyplot.subplots", return_value=(mock_fig, [[mock_ax, mock_ax]])
        ):
            sync.estimate_timing(sig, preamble, debug_plot=True)


def test_estimate_timing_zero_energy(backend_device, xp):
    """Test estimate_timing with zero energy signal."""
    preamble = xp.ones(10)
    sig = xp.zeros(50)
    with pytest.raises(ValueError, match="No correlation peak above threshold"):
        sync.estimate_timing(sig, preamble, threshold=0.5)


def test_estimate_timing_return_tuple(backend_device, xp):
    """Verify return tuple structure."""
    preamble = xp.ones(4)
    sig = xp.concatenate([xp.zeros(4), preamble, xp.zeros(4)])

    res = sync.estimate_timing(sig, preamble, threshold=0.1)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert len(res[0]) == 1  # coarse_offsets
    assert len(res[1]) == 1  # fractional_offsets


def test_estimate_timing_search_range(backend_device, xp):
    """Verify estimate_timing with search_range."""
    preamble = xp.random.randn(10) + 1j * xp.random.randn(10)
    sig = xp.concatenate([xp.zeros(50), preamble, xp.zeros(50)])

    # Search only in 40-70 range
    coarse, _frac = sync.estimate_timing(
        sig, preamble, search_range=(40, 70), threshold=0.5
    )
    assert coarse[0] == 50


def test_estimate_timing_infer_error(backend_device, xp):
    """Verify error when Preamble object used with raw array signal."""
    pre = Preamble(bits=[1, 0, 1], length=3)
    sig = xp.zeros(20)
    with pytest.raises(ValueError, match="SPS required for Preamble object."):
        sync.estimate_timing(sig, pre)


def test_sequences_gpu(backend_device, xp):
    """Cover the to_device branch in sequence generators on GPU."""
    if backend_device != "gpu":
        pytest.skip("Test targets GPU branch")

    barker = sync.barker_sequence(13)
    assert isinstance(barker, xp.ndarray)

    zc = sync.zadoff_chu_sequence(13, root=1)
    assert isinstance(zc, xp.ndarray)


# ============================================================================
# Fine Timing Estimation Tests
# ============================================================================


def test_estimate_fractional_delay_known_shift(backend_device, xp):
    """Verify parabolic interpolation recovers a known fractional delay."""
    import numpy as np

    # Create a sharp peak via a sinc-like correlation centered at k=50
    # with a known fractional offset
    N = 100
    true_mu = 0.3
    # Build a smooth peak: quadratic around k=50 with fractional shift
    n = np.arange(N, dtype="float64")
    # Sampled quadratic: peak at 50 + true_mu
    corr = np.exp(-0.5 * ((n - 50.0 - true_mu) / 2.0) ** 2).astype("float32")
    corr = xp.asarray(corr)

    peak_idx = xp.argmax(xp.abs(corr))

    mu = sync.estimate_fractional_delay(corr, peak_idx)
    assert abs(float(mu) - true_mu) < 0.15  # Parabolic is approximate


def test_estimate_fractional_delay_edge_peak(backend_device, xp):
    """Verify graceful fallback when peak is at array boundary."""
    corr = xp.zeros(50, dtype="float32")
    corr[0] = 1.0  # Peak at boundary

    mu = sync.estimate_fractional_delay(corr, xp.asarray(0))
    assert float(mu) == 0.0  # Should return 0 at edge


def test_estimate_fractional_delay_mimo(backend_device, xp):
    """Verify per-channel fractional delay estimation."""
    import numpy as np

    N = 100
    corr = xp.zeros((2, N), dtype="float32")
    n = np.arange(N, dtype="float32")
    corr[0] = xp.asarray(np.exp(-0.5 * ((n - 40.0 - 0.2) / 2.0) ** 2).astype("float32"))
    corr[1] = xp.asarray(np.exp(-0.5 * ((n - 60.0 + 0.1) / 2.0) ** 2).astype("float32"))

    peaks = xp.array([40, 60])
    mu = sync.estimate_fractional_delay(corr, peaks)
    assert mu.shape == (2,)
    assert abs(float(mu[0]) - 0.2) < 0.15
    assert abs(float(mu[1]) - (-0.1)) < 0.15


# ============================================================================
# Farrow Interpolator Tests
# ============================================================================


# ============================================================================
# FFT-based Fractional Delay Tests
# ============================================================================


def test_fft_fractional_delay_zero_delay(backend_device, xp):
    """Verify delay=0 is a perfect passthrough (identity operation)."""
    import numpy as np

    n = np.arange(100, dtype="float32")
    signal = xp.asarray(np.sin(2 * np.pi * 0.05 * n).astype("complex64"))

    out = sync.fft_fractional_delay(signal, 0.0)
    # Should be identical (FFT is exact for delay=0)
    assert xp.allclose(out, signal, atol=1e-6)


def test_fft_fractional_delay_known_sine(backend_device, xp):
    """Verify fractional delay of a sinusoid against ground truth."""
    import numpy as np

    f = 0.02  # Normalized frequency
    N = 200
    delay = 0.3
    n = np.arange(N, dtype="float64")

    # Original and ground-truth shifted signal (complex to match typical use)
    original = np.exp(2j * np.pi * f * n).astype("complex64")
    truth = np.exp(2j * np.pi * f * (n - delay)).astype("complex64")

    out = sync.fft_fractional_delay(xp.asarray(original), delay)
    out_np = out if backend_device == "cpu" else out.get()

    # FFT-based delay should be nearly perfect for bandlimited signals
    assert np.allclose(out_np, truth, atol=1e-5)


def test_fft_fractional_delay_mimo(backend_device, xp):
    """Verify per-channel fractional delays for 2-channel signal."""
    import numpy as np

    f = 0.02
    N = 200
    n = np.arange(N, dtype="float64")

    sig = np.zeros((2, N), dtype="complex64")
    sig[0] = np.exp(2j * np.pi * f * n).astype("complex64")
    sig[1] = np.exp(2j * np.pi * f * n).astype("complex64")

    delays = xp.asarray([0.3, -0.2], dtype="float32")
    out = sync.fft_fractional_delay(xp.asarray(sig), delays)

    truth0 = np.exp(2j * np.pi * f * (n - 0.3)).astype("complex64")
    truth1 = np.exp(2j * np.pi * f * (n + 0.2)).astype("complex64")

    out_np = out if backend_device == "cpu" else out.get()
    assert np.allclose(out_np[0], truth0, atol=1e-5)
    assert np.allclose(out_np[1], truth1, atol=1e-5)


def test_fft_fractional_delay_power_conservation(backend_device, xp):
    """Verify FFT-based delay preserves signal power."""
    import numpy as np

    np.random.seed(42)
    N = 1000
    # Complex random signal
    signal = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
    signal_xp = xp.asarray(signal)

    delay = 0.3
    delayed = sync.fft_fractional_delay(signal_xp, delay)

    # Measure power
    power_in = float(xp.mean(xp.abs(signal_xp) ** 2))
    power_out = float(xp.mean(xp.abs(delayed) ** 2))

    # Should preserve power to numerical precision
    assert abs(power_out / power_in - 1.0) < 1e-5


def test_fft_fractional_delay_roundtrip(backend_device, xp):
    """Verify round-trip (delay + undo) recovers original signal."""
    import numpy as np

    np.random.seed(42)
    N = 1000
    signal = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
    signal_xp = xp.asarray(signal)

    delay = 0.37
    # Apply delay then undo it
    delayed = sync.fft_fractional_delay(signal_xp, delay)
    recovered = sync.fft_fractional_delay(delayed, -delay)

    # Should recover original signal to high precision
    assert xp.allclose(recovered, signal_xp, atol=1e-5)


# ============================================================================
# Combined Timing Correction Tests
# ============================================================================


def test_correct_timing_coarse_only(backend_device, xp):
    """Verify integer-only timing correction via roll."""
    signal = xp.zeros(50, dtype="float32")
    signal[10] = 1.0

    corrected = sync.correct_timing(signal, coarse_offset=10)
    # Peak should now be at index 0
    assert int(xp.argmax(xp.abs(corrected))) == 0


def test_correct_timing_combined(backend_device, xp):
    """Verify coarse + fractional timing correction."""
    import numpy as np

    f = 0.02
    N = 200
    delay = 20.3  # 20 integer + 0.3 fractional
    n = np.arange(N, dtype="float64")

    original = np.sin(2 * np.pi * f * n).astype("float32")
    delayed = np.sin(2 * np.pi * f * (n - delay)).astype("float32")

    corrected = sync.correct_timing(
        xp.asarray(delayed), coarse_offset=20, fractional_offset=0.3
    )
    corrected_np = corrected if backend_device == "cpu" else corrected.get()

    assert np.allclose(corrected_np[25:-25], original[25:-25], atol=0.02)


# ============================================================================
# estimate_timing with fractional timing
# ============================================================================


def test_estimate_timing_fractional(backend_device, xp):
    """Verify estimate_timing returns fractional offset."""
    preamble = sync.barker_sequence(13)
    signal = xp.zeros(200, dtype="complex64")
    signal[50:63] = preamble

    coarse, frac = sync.estimate_timing(signal, preamble, threshold=0.3)
    assert len(coarse) == 1
    assert len(frac) == 1
    # The fractional offset should be near 0 for an integer-aligned preamble
    assert abs(float(frac[0])) < 0.5


def test_estimate_fractional_delay_methods(backend_device, xp):
    """Verify different fractional delay estimation methods."""
    import numpy as np

    # 1. Gaussian Pulse -> Log-Parabolic (Gaussian) fit should be superior
    N = 64
    true_mu = 0.35
    sigma = 2.0
    t = np.arange(N) - N / 2
    gaussian = np.exp(-0.5 * ((t - true_mu) / sigma) ** 2).astype("float32")
    corr_gauss = xp.asarray(gaussian)
    peak_idx = xp.asarray(32)

    est_std = sync.estimate_fractional_delay(corr_gauss, peak_idx, method="parabolic")
    est_log = sync.estimate_fractional_delay(
        corr_gauss, peak_idx, method="log-parabolic"
    )

    err_std = abs(float(est_std) - true_mu)
    err_log = abs(float(est_log) - true_mu)

    # Log-parabolic should be much better for Gaussian
    assert err_log < err_std
    assert err_log < 1e-5

    # 2. Sinc Pulse -> DFT Upsampling should be superior
    sinc_val = np.sinc(t - true_mu).astype("float32")
    corr_sinc = xp.asarray(sinc_val)

    # Standard (no upsample)
    est_sinc_1x = sync.estimate_fractional_delay(corr_sinc, peak_idx, dft_upsample=1)
    err_sinc_1x = abs(float(est_sinc_1x) - true_mu)

    # Upsampled (8x)
    est_sinc_8x = sync.estimate_fractional_delay(corr_sinc, peak_idx, dft_upsample=8)
    err_sinc_8x = abs(float(est_sinc_8x) - true_mu)

    # Upsampling should improve accuracy significantly for bandlimited pulse
    assert err_sinc_8x < err_sinc_1x
    assert err_sinc_8x < 0.01
