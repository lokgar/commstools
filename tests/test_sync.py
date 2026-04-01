"""Tests for synchronization utilities (Barker and Zadoff-Chu sequences, frame detection)."""

from unittest.mock import MagicMock, patch

import pytest

from commstools import sync
from commstools.core import Preamble
from commstools.helpers import cross_correlate_fft
from commstools.impairments import apply_iq_imbalance


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


def test_zadoff_chu_cazac(backend_device, xp, xpt):
    """Verify that ZC sequences have constant amplitude (CAZAC property)."""
    zc = sync.zadoff_chu_sequence(63, root=25)

    # All magnitudes should be 1
    magnitudes = xp.abs(zc)
    xpt.assert_allclose(magnitudes, 1.0, atol=1e-5)


def test_zadoff_chu_length(backend_device, xp, xpt):
    """Verify that ZC sequences are generated with the requested length."""
    for length in [31, 63, 127]:
        zc = sync.zadoff_chu_sequence(length, root=1)
        assert len(zc) == length

    # Test even length
    zc_even = sync.zadoff_chu_sequence(10, root=1)
    assert len(zc_even) == 10
    xpt.assert_allclose(xp.abs(zc_even), 1.0)


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
    preamble_zc = Preamble(sequence_type="zc", length=63, root=1)
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
    """Verify estimate_timing with raw arrays, MIMO, and search ranges."""
    from commstools.core import Preamble

    # 1. Raw array and Preamble object
    preamble = Preamble(sequence_type="barker", length=7)

    # Create samples with the preamble embedded
    data = xp.zeros(100, dtype="complex64")
    data[20 : 20 + 7] = preamble.symbols

    coarse, _frac = sync.estimate_timing(data, preamble, threshold=2.0, sps=1)
    assert 18 <= coarse[0] <= 22

    # 2. MIMO Signal (2 channels)
    mimo_data = xp.zeros((2, 100), dtype="complex64")
    mimo_data[0, 30:37] = preamble.symbols
    mimo_data[1, 30:37] = preamble.symbols
    coarse_mimo, _frac = sync.estimate_timing(
        mimo_data, preamble.symbols, threshold=2.0
    )
    assert 28 <= coarse_mimo[0] <= 32
    assert len(coarse_mimo) == 2

    # 3. Search range
    coarse_range, _frac = sync.estimate_timing(
        data, preamble.symbols, threshold=2.0, search_range=(10, 50)
    )
    assert 18 <= coarse_range[0] <= 22

    # 4. High threshold (above max)
    with pytest.raises(ValueError, match="No correlation peak above threshold"):
        sync.estimate_timing(data, preamble.symbols, threshold=100.0)

    # 5. Zero energy
    zero_data = xp.zeros(100)
    with pytest.raises(ValueError, match="No correlation peak above threshold"):
        sync.estimate_timing(zero_data, preamble.symbols, threshold=2.0)


def test_estimate_timing_known_position(backend_device, xp):
    """Verify timing estimation accuracy for a known preamble position."""
    # Create preamble
    preamble_symbols = sync.barker_sequence(13)

    # Embed in longer signal at known position
    signal = xp.zeros(200, dtype="complex64")
    start_pos = 50
    signal[start_pos : start_pos + 13] = preamble_symbols

    # Detect
    coarse, _frac = sync.estimate_timing(signal, preamble_symbols, threshold=2.0)

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
    coarse, _frac = sync.estimate_timing(sig_obj, preamble, threshold=2.0)

    assert abs(coarse[0] - start_pos) <= 1


def test_estimate_timing_returns_tuple(backend_device, xp):
    """Verify that estimate_timing returns (coarse_offsets, fractional_offsets)."""
    preamble = sync.barker_sequence(7)

    signal = xp.zeros(100, dtype="complex64")
    signal[30:37] = preamble

    coarse, frac = sync.estimate_timing(signal, preamble, threshold=2.0)

    assert len(coarse) == 1
    assert len(frac) == 1
    assert abs(float(frac[0])) < 0.5


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
        sync.estimate_timing(sig, preamble, threshold=2.0)


def test_estimate_timing_return_tuple(backend_device, xp):
    """Verify return tuple structure."""
    preamble = xp.ones(4)
    sig = xp.concatenate([xp.zeros(4), preamble, xp.zeros(4)])

    res = sync.estimate_timing(sig, preamble, threshold=2.0)
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
        sig, preamble, search_range=(40, 70), threshold=2.0
    )
    assert coarse[0] == 50


def test_estimate_timing_infer_error(backend_device, xp):
    """Verify error when Preamble object used without sps."""
    pre = Preamble(sequence_type="barker", length=3)
    sig = xp.zeros(20)
    with pytest.raises(ValueError, match="SPS must be provided"):
        sync.estimate_timing(sig, pre)


def test_sequences_device(backend_device, xp):
    """Verify sequence generators return arrays on the active device."""
    barker = sync.barker_sequence(13)
    assert isinstance(barker, xp.ndarray)

    zc = sync.zadoff_chu_sequence(13, root=1)
    assert isinstance(zc, xp.ndarray)


# -----------------------------------------------------------------------------
# Fine Timing Estimation Tests
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Farrow Interpolator Tests
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# FFT-based Fractional Delay Tests
# -----------------------------------------------------------------------------


def test_fft_fractional_delay_zero_delay(backend_device, xp, xpt):
    """Verify delay=0 is a perfect passthrough (identity operation)."""
    import numpy as np

    n = np.arange(100, dtype="float32")
    signal = xp.asarray(np.sin(2 * np.pi * 0.05 * n).astype("complex64"))

    out = sync.fft_fractional_delay(signal, 0.0)
    # Should be identical (FFT is exact for delay=0)
    xpt.assert_allclose(out, signal, atol=1e-6)


def test_fft_fractional_delay_known_sine(backend_device, xp, xpt):
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

    # FFT-based delay should be nearly perfect for bandlimited signals
    xpt.assert_allclose(out, truth, atol=1e-5)


def test_fft_fractional_delay_mimo(backend_device, xp, xpt):
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

    xpt.assert_allclose(out[0], truth0, atol=1e-5)
    xpt.assert_allclose(out[1], truth1, atol=1e-5)


def test_fft_fractional_delay_power_conservation(backend_device, xp):
    """Verify FFT-based delay preserves signal power."""
    import numpy as np

    np.random.seed(42)
    N = 1000
    # Complex random signal
    signal = (np.random.randn(N) + 1j * np.random.randn(N)).astype("complex64")
    signal_xp = xp.asarray(signal)

    delay = 0.3
    delayed = sync.fft_fractional_delay(signal_xp, delay)

    # Measure power
    power_in = float(xp.mean(xp.abs(signal_xp) ** 2))
    power_out = float(xp.mean(xp.abs(delayed) ** 2))

    # Should preserve power to numerical precision
    assert abs(power_out / power_in - 1.0) < 1e-5


def test_fft_fractional_delay_roundtrip(backend_device, xp, xpt):
    """Verify round-trip (delay + undo) recovers original signal."""
    import numpy as np

    np.random.seed(42)
    N = 1000
    signal = (np.random.randn(N) + 1j * np.random.randn(N)).astype("complex64")
    signal_xp = xp.asarray(signal)

    delay = 0.37
    # Apply delay then undo it
    delayed = sync.fft_fractional_delay(signal_xp, delay)
    recovered = sync.fft_fractional_delay(delayed, -delay)

    # Should recover original signal to high precision
    xpt.assert_allclose(recovered, signal_xp, atol=1e-5)


# -----------------------------------------------------------------------------
# Combined Timing Correction Tests
# -----------------------------------------------------------------------------


def test_correct_timing_coarse_only(backend_device, xp):
    """Verify integer-only timing correction via roll."""
    signal = xp.zeros(50, dtype="float32")
    signal[10] = 1.0

    corrected = sync.correct_timing(signal, coarse_offset=10)
    # Peak should now be at index 0
    assert int(xp.argmax(xp.abs(corrected))) == 0


def test_correct_timing_combined(backend_device, xp, xpt):
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

    xpt.assert_allclose(corrected[25:-25], original[25:-25], atol=0.02)


# -----------------------------------------------------------------------------
# estimate_timing with fractional timing
# -----------------------------------------------------------------------------


def test_estimate_timing_fractional(backend_device, xp):
    """Verify estimate_timing returns fractional offset."""
    preamble = sync.barker_sequence(13)
    signal = xp.zeros(200, dtype="complex64")
    signal[50:63] = preamble

    coarse, frac = sync.estimate_timing(signal, preamble, threshold=2.0)
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


# -----------------------------------------------------------------------------
# Coverage Tests for Uncovered Lines
# -----------------------------------------------------------------------------


def test_fft_fractional_delay_scalar_ndarray(backend_device, xp, xpt):
    """Verify fft_fractional_delay with 0-d array delay input."""
    import numpy as np

    f = 0.02
    N = 100
    n = np.arange(N, dtype="float64")
    signal = xp.asarray(np.exp(2j * np.pi * f * n).astype("complex64"))

    # Pass delay as 0-d ndarray (not int/float)
    delay = xp.asarray(0.3)
    out = sync.fft_fractional_delay(signal, delay)
    assert out.shape == (N,)

    truth = np.exp(2j * np.pi * f * (n - 0.3)).astype("complex64")
    xpt.assert_allclose(out, truth, atol=1e-5)


def test_estimate_timing_no_preamble_error(backend_device, xp):
    """Verify estimate_timing raises when no reference is given."""
    sig = xp.zeros(100, dtype="complex64")
    with pytest.raises(
        ValueError,
        match="A 'reference' sequence must be provided",
    ):
        sync.estimate_timing(sig)


def test_estimate_timing_with_preamble_object(backend_device, xp):
    """Verify estimate_timing with explicit Preamble object."""
    from commstools.core import Preamble

    # Create samples with a Barker-7 preamble embedded at sample 40
    barker = sync.barker_sequence(7)
    samples = xp.zeros(200, dtype="complex64")
    samples[40:47] = barker

    preamble = Preamble(sequence_type="barker", length=7)
    coarse, frac = sync.estimate_timing(
        samples, preamble, sps=1, pulse_shape="none", threshold=2.0
    )
    assert abs(int(coarse[0]) - 40) <= 1


def test_estimate_timing_skew_detection(backend_device, xp):
    """Verify skew warning is emitted when MIMO channels have different preamble positions."""

    # 2-channel signal with preambles at slightly different positions
    barker = sync.barker_sequence(7)
    sig = xp.zeros((2, 200), dtype="complex64")
    sig[0, 40:47] = barker
    sig[1, 42:49] = barker  # 2-sample offset -> skew

    # Capture log output to verify skew warning is emitted
    with patch("commstools.sync.logger") as mock_logger:
        coarse, frac = sync.estimate_timing(sig, barker, threshold=2.0)
        # Check that warning was called with skew message
        mock_logger.warning.assert_called()
        call_args = mock_logger.warning.call_args[0][0]
        assert "Skew detected" in call_args

    # Both channels should be detected
    assert len(coarse) == 2
    # Positions differ
    assert abs(int(coarse[0]) - 40) <= 1
    assert abs(int(coarse[1]) - 42) <= 1


def test_correct_timing_per_channel(backend_device, xp):
    """Verify per-channel coarse timing correction using an array of offsets."""

    # 2-channel signal with peaks at different positions
    sig = xp.zeros((2, 50), dtype="complex64")
    sig[0, 10] = 1.0
    sig[1, 20] = 1.0

    # Per-channel shifts
    offsets = xp.array([10, 20])
    corrected = sync.correct_timing(sig, coarse_offset=offsets)

    # Both peaks should be at index 0 after correction
    assert corrected.shape == (2, 50)
    assert int(xp.argmax(xp.abs(corrected[0]))) == 0
    assert int(xp.argmax(xp.abs(corrected[1]))) == 0


def test_correct_timing_fractional_array(backend_device, xp, xpt):
    """Verify fractional offset as array applies per-channel FFT delay and returns 2D output."""
    import numpy as np

    f = 0.02
    N = 200
    n = np.arange(N, dtype="float64")

    # 2-channel signal, each with a different fractional delay
    sig = xp.zeros((2, N), dtype="complex64")
    sig[0] = xp.asarray(np.exp(2j * np.pi * f * (n - 0.3)).astype("complex64"))
    sig[1] = xp.asarray(np.exp(2j * np.pi * f * (n + 0.2)).astype("complex64"))

    # Correct with array of fractional offsets
    fractional = xp.array([0.3, -0.2])
    corrected = sync.correct_timing(sig, coarse_offset=0, fractional_offset=fractional)

    # Should return 2D (not squeezed)
    assert corrected.ndim == 2
    assert corrected.shape == (2, N)

    # Both channels should now be close to the undelayed signal
    truth = np.exp(2j * np.pi * f * n).astype("complex64")
    xpt.assert_allclose(corrected[0], truth, atol=1e-4)
    xpt.assert_allclose(corrected[1], truth, atol=1e-4)


def test_estimate_fractional_delay_dft_edge_fallback(backend_device, xp):
    """Verify DFT upsample with edge peak falls back to standard parabolic estimation."""
    import numpy as np

    N = 100
    true_mu = 0.25
    n = np.arange(N, dtype="float32")

    # Peak at edge (index 2, within half_W=16 of boundary)
    corr = np.exp(-0.5 * ((n - 2.0 - true_mu) / 2.0) ** 2).astype("float32")
    corr = xp.asarray(corr)
    peak_idx = xp.asarray(2)

    # With DFT upsample > 1, but peak near edge → should use standard fallback
    mu = sync.estimate_fractional_delay(corr, peak_idx, dft_upsample=8)
    # Should still produce a reasonable result via standard path
    assert abs(float(mu)) < 0.5


def test_estimate_timing_preamble_kwargs_without_sps(backend_device, xp):
    """Verify estimate_timing raises when preamble is provided but sps is missing."""
    from commstools.core import Preamble

    sig = xp.zeros(100, dtype="complex64")
    pre = Preamble(sequence_type="barker", length=7)
    with pytest.raises(ValueError, match="SPS must be provided"):
        sync.estimate_timing(sig, reference=pre)


# -----------------------------------------------------------------------------
# MIMO UNIQUE-ROOT TIMING SYNC TESTS
# -----------------------------------------------------------------------------


def _make_mimo_signal(xp, channel_matrix, preamble_pos=200, skew=0):
    """Helper: build 2×2 MIMO received signal with unique-root ZC preamble.

    Parameters
    ----------
    xp : module
        Array backend (numpy or cupy).
    channel_matrix : array_like, shape (2, 2)
        Jones-like channel matrix.  ``rx = channel_matrix @ tx``.
    preamble_pos : int
        Sample index where the preamble starts on RX channel 0.
    skew : int
        Additional sample offset applied to RX channel 1 only, simulating
        hardware channel skew.

    Returns
    -------
    rx : array, shape (2, N)
        Received samples with embedded preamble.
    preamble : Preamble
        The ZC preamble object (root=1, length=L).
    L : int
        Preamble sequence length in samples (= 13 at 1 sps).
    """
    import numpy as _np
    from commstools.core import Preamble
    from commstools.helpers import zc_mimo_root

    # ZC-13 with unique roots per TX stream: root 1 for TX0, root 2 for TX1
    L = 13
    zc0 = xp.asarray(
        sync.zadoff_chu_sequence(L, root=zc_mimo_root(0, 1, L)), dtype="complex64"
    )
    zc1 = xp.asarray(
        sync.zadoff_chu_sequence(L, root=zc_mimo_root(1, 1, L)), dtype="complex64"
    )

    # Build TX: both streams transmit simultaneously at preamble_pos
    N = preamble_pos + L + 300
    tx = xp.zeros((2, N), dtype="complex64")
    tx[0, preamble_pos : preamble_pos + L] = zc0
    tx[1, preamble_pos : preamble_pos + L] = zc1

    # Low-level noise
    _rng = _np.random.RandomState(42)
    noise = xp.asarray(
        (_rng.randn(2, N) + 1j * _rng.randn(2, N)).astype("complex64") * 0.05
    )
    tx = tx + noise

    # Apply channel matrix and optional per-channel skew
    H = xp.asarray(channel_matrix, dtype="complex64")
    rx = H @ tx  # (2, N)

    if skew != 0:
        # Roll channel 1 to simulate a timing skew of `skew` samples
        rx = xp.stack([rx[0], xp.roll(rx[1], skew)], axis=0)

    preamble = Preamble(sequence_type="zc", length=L, root=1, num_streams=2)
    return rx, preamble, L


def test_estimate_timing_mimo_identity(backend_device, xp):
    """MIMO unique-root ZC: identity channel, both channels align to preamble_pos."""
    preamble_pos = 200
    rx, preamble, L = _make_mimo_signal(xp, [[1.0, 0.0], [0.0, 1.0]], preamble_pos)

    coarse, frac = sync.estimate_timing(
        rx, preamble, sps=1, pulse_shape="none", threshold=2.0
    )

    for ch in range(2):
        assert abs(int(coarse[ch]) - preamble_pos) <= 1, (
            f"Channel {ch}: expected coarse≈{preamble_pos}, got {int(coarse[ch])}"
        )


def test_estimate_timing_mimo_mixed_channel(backend_device, xp):
    """MIMO unique-root ZC: mixed channel (both streams present on each RX)."""
    import numpy as np

    preamble_pos = 150
    angle = np.radians(40)
    H = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rx, preamble, L = _make_mimo_signal(xp, H, preamble_pos)

    coarse, frac = sync.estimate_timing(
        rx, preamble, sps=1, pulse_shape="none", threshold=2.0
    )

    for ch in range(2):
        assert abs(int(coarse[ch]) - preamble_pos) <= 1, (
            f"Channel {ch}: expected coarse≈{preamble_pos}, got {int(coarse[ch])}"
        )


def test_estimate_timing_mimo_channel_skew(backend_device, xp):
    """MIMO: hardware skew of 5 samples on channel 1 is reflected in per-channel coarse offsets."""
    preamble_pos = 200
    skew = 5
    rx, preamble, L = _make_mimo_signal(
        xp, [[1.0, 0.0], [0.0, 1.0]], preamble_pos, skew=skew
    )

    coarse, frac = sync.estimate_timing(
        rx, preamble, sps=1, pulse_shape="none", threshold=2.0
    )

    # Channel 0 should find preamble_pos; channel 1 is shifted by skew
    assert abs(int(coarse[0]) - preamble_pos) <= 1, (
        f"Ch0: expected {preamble_pos}, got {int(coarse[0])}"
    )
    expected_ch1 = preamble_pos + skew
    assert abs(int(coarse[1]) - expected_ch1) <= 1, (
        f"Ch1: expected {expected_ch1} (pos+skew), got {int(coarse[1])}"
    )
    # Offsets must differ by ~skew (not forced equal)
    assert int(coarse[0]) != int(coarse[1]), (
        "Skew channels must have different coarse offsets"
    )


def test_estimate_timing_mimo_permuted_channel(backend_device, xp):
    """MIMO: pure polarization swap (H = [[0,1],[1,0]]) — RX-0 receives TX-1 and vice versa.

    This is the worst-case for the Hungarian assignment: the initial assignment
    based on cross-correlation peak scores must correctly un-swap the streams.
    The fallback path (X-Y power imbalance recovery) is also exercised here
    when one channel's assigned template has low correlation.
    """
    preamble_pos = 200
    # Permutation channel: RX-0 sees TX-1, RX-1 sees TX-0
    H = [[0.0, 1.0], [1.0, 0.0]]
    rx, preamble, L = _make_mimo_signal(xp, H, preamble_pos)

    coarse, frac = sync.estimate_timing(
        rx, preamble, sps=1, pulse_shape="none", threshold=2.0
    )

    for ch in range(2):
        assert abs(int(coarse[ch]) - preamble_pos) <= 1, (
            f"Channel {ch}: expected coarse≈{preamble_pos}, got {int(coarse[ch])}"
        )


# -----------------------------------------------------------------------------
# DTYPE PRESERVATION TESTS
# -----------------------------------------------------------------------------


def test_fft_fractional_delay_preserves_complex64_dtype(backend_device, xp):
    """fft_fractional_delay: complex64 signal → complex64 output."""
    import numpy as np

    n = np.arange(200)
    sig = xp.asarray(np.exp(2j * np.pi * 0.05 * n).astype(np.complex64))
    out = sync.fft_fractional_delay(sig, 0.3)
    assert out.dtype == xp.complex64, f"Expected complex64, got {out.dtype}"


def test_fft_fractional_delay_preserves_float32_dtype(backend_device, xp):
    """fft_fractional_delay: float32 signal → float32 output."""
    import numpy as np

    n = np.arange(200, dtype=np.float32)
    sig = xp.asarray(np.sin(2 * np.pi * 0.05 * n))
    out = sync.fft_fractional_delay(sig, 0.3)
    assert out.dtype == xp.float32, f"Expected float32, got {out.dtype}"


# =============================================================================
# CARRIER PHASE RECOVERY — VITERBI-VITERBI
# =============================================================================


class TestViterbiViterbi:
    """Tests for recover_carrier_phase_viterbi_viterbi."""

    def _qpsk_symbols(self, xp, N=512, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 4, N)
        angles = (2 * np.pi / 4) * bits + np.pi / 4
        return xp.asarray(np.exp(1j * angles).astype(np.complex64))

    def _qam16_symbols(self, xp, N=512, seed=1):
        import numpy as np
        from commstools.mapping import gray_constellation

        rng = np.random.default_rng(seed)
        const = gray_constellation("qam", 16)
        idx = rng.integers(0, 16, N)
        return xp.asarray(const[idx].astype(np.complex64))

    def test_siso_qpsk_output_shape(self, backend_device, xp):
        """SISO QPSK: output is (N,) float64."""
        syms = self._qpsk_symbols(xp)
        phi_est = sync.recover_carrier_phase_viterbi_viterbi(
            syms, "psk", 4, block_size=32
        )
        assert phi_est.shape == syms.shape
        assert phi_est.dtype == xp.float64

    def test_siso_qpsk_recovers_static_phase(self, backend_device, xp):
        """VV tracks applied phase: difference between rotated and unrotated estimate equals phi_true mod π/2."""
        import numpy as np

        phi_true = 0.3  # radians
        syms = self._qpsk_symbols(xp, N=512)
        # Baseline (no rotation)
        phi_base = float(
            xp.mean(
                sync.recover_carrier_phase_viterbi_viterbi(
                    syms, "psk", 4, block_size=32
                )
            )
        )
        # Rotated by phi_true
        rotated = syms * xp.asarray(np.complex64(np.exp(1j * phi_true)))
        phi_rot = float(
            xp.mean(
                sync.recover_carrier_phase_viterbi_viterbi(
                    rotated, "psk", 4, block_size=32
                )
            )
        )
        # The shift should equal phi_true modulo π/2
        delta = phi_rot - phi_base
        residual = (delta - phi_true + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(residual) < 0.15, (
            f"Phase tracking error too large: {residual:.3f} rad"
        )

    def test_siso_qam16_output_shape(self, backend_device, xp):
        """SISO QAM16: output shape matches input."""
        syms = self._qam16_symbols(xp)
        phi_est = sync.recover_carrier_phase_viterbi_viterbi(
            syms, "qam", 16, block_size=32
        )
        assert phi_est.shape == syms.shape

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO input (C, N): output shape is (C, N)."""
        import numpy as np

        C, N = 2, 256
        syms = xp.asarray(
            np.random.default_rng(5).standard_normal((C, N)).astype(np.float32)
            + 1j * np.random.default_rng(6).standard_normal((C, N)).astype(np.float32)
        )
        phi_est = sync.recover_carrier_phase_viterbi_viterbi(
            syms, "qam", 16, block_size=32
        )
        assert phi_est.shape == (C, N)

    def test_block_size_too_large_raises(self, backend_device, xp):
        """block_size > N should raise ValueError."""
        syms = self._qpsk_symbols(xp, N=16)
        with pytest.raises(ValueError, match="block_size"):
            sync.recover_carrier_phase_viterbi_viterbi(syms, "psk", 4, block_size=64)


# =============================================================================
# CARRIER PHASE RECOVERY — BLIND PHASE SEARCH
# =============================================================================


class TestBPS:
    """Tests for recover_carrier_phase_bps."""

    def _qam16_symbols(self, xp, N=512, seed=2):
        import numpy as np
        from commstools.mapping import gray_constellation

        rng = np.random.default_rng(seed)
        const = gray_constellation("qam", 16)
        idx = rng.integers(0, 16, N)
        return xp.asarray(const[idx].astype(np.complex64))

    def _qpsk_symbols(self, xp, N=512, seed=3):
        import numpy as np
        from commstools.mapping import gray_constellation

        rng = np.random.default_rng(seed)
        const = gray_constellation("qpsk", 4)
        idx = rng.integers(0, 4, N)
        return xp.asarray(const[idx].astype(np.complex64))

    def test_siso_qam16_output_shape(self, backend_device, xp):
        """SISO QAM16 (square QAM fast path): output is (N,) float64."""
        syms = self._qam16_symbols(xp)
        phi_est = sync.recover_carrier_phase_bps(
            syms, "qam", 16, num_test_phases=32, block_size=32
        )
        assert phi_est.shape == syms.shape
        assert phi_est.dtype == xp.float64

    def test_siso_qam16_recovers_static_phase(self, backend_device, xp):
        """BPS should estimate a static QAM16 phase offset to within π/8 tolerance."""
        import numpy as np

        phi_true = 0.25
        syms = self._qam16_symbols(xp, N=512)
        rotated = syms * xp.asarray(np.complex64(np.exp(1j * phi_true)))
        phi_est = sync.recover_carrier_phase_bps(
            rotated, "qam", 16, num_test_phases=64, block_size=32
        )
        phi_mean = float(xp.mean(phi_est))
        # 4-fold ambiguity: allow ±π/8 residual
        residual = (phi_mean - phi_true + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(residual) < 0.15, (
            f"Residual phase error too large: {residual:.3f} rad"
        )

    def test_siso_qpsk_general_path(self, backend_device, xp):
        """SISO QPSK (non-square: triggers general distance path): output shape correct."""
        syms = self._qpsk_symbols(xp, N=256)
        phi_est = sync.recover_carrier_phase_bps(
            syms, "psk", 4, num_test_phases=16, block_size=32
        )
        assert phi_est.shape == syms.shape

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO input (C, N): output shape is (C, N)."""
        import numpy as np

        C, N = 2, 256
        rng = np.random.default_rng(7)
        syms = xp.asarray(
            (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
                np.complex64
            )
        )
        phi_est = sync.recover_carrier_phase_bps(
            syms, "qam", 16, num_test_phases=16, block_size=32
        )
        assert phi_est.shape == (C, N)

    def test_block_size_too_large_raises(self, backend_device, xp):
        """block_size > N should raise ValueError."""
        syms = self._qam16_symbols(xp, N=16)
        with pytest.raises(ValueError, match="block_size"):
            sync.recover_carrier_phase_bps(syms, "qam", 16, block_size=64)


# =============================================================================
# CARRIER PHASE RECOVERY — DECISION-DIRECTED PLL
# =============================================================================


class TestDDPLL:
    """Tests for recover_carrier_phase_pll."""

    def _qpsk_symbols(self, xp, N=512, seed=10):
        import numpy as np
        from commstools.mapping import gray_constellation

        rng = np.random.default_rng(seed)
        const = gray_constellation("qpsk", 4)
        return xp.asarray(const[rng.integers(0, 4, N)].astype(np.complex64))

    def test_siso_output_shape(self, backend_device, xp):
        """SISO: output is (N,) float64."""
        syms = self._qpsk_symbols(xp)
        phi = sync.recover_carrier_phase_pll(syms, "psk", 4)
        assert phi.shape == syms.shape
        assert phi.dtype == xp.float64

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO (C, N): output shape is (C, N)."""
        import numpy as np

        C, N = 2, 256
        rng = np.random.default_rng(11)
        from commstools.mapping import gray_constellation

        const = gray_constellation("qpsk", 4)
        syms = xp.asarray(
            const[rng.integers(0, 4, C * N)].reshape(C, N).astype(np.complex64)
        )
        phi = sync.recover_carrier_phase_pll(syms, "psk", 4)
        assert phi.shape == (C, N)

    def test_second_order_loop(self, backend_device, xp):
        """beta > 0 engages 2nd-order loop without raising."""
        syms = self._qpsk_symbols(xp, N=256)
        phi = sync.recover_carrier_phase_pll(syms, "psk", 4, mu=0.02, beta=1e-4)
        assert phi.shape == syms.shape

    def test_phase_init_applied(self, backend_device, xp):
        """phase_init shifts the starting phase estimate."""
        syms = self._qpsk_symbols(xp, N=256)
        phi_init = 0.5
        phi = sync.recover_carrier_phase_pll(syms, "psk", 4, phase_init=phi_init)
        # First estimate should be close to phase_init before any loop correction
        assert abs(float(phi[0]) - phi_init) < 0.5

    def test_butterworth_output_shape_siso(self, backend_device, xp):
        """loop_filter='butterworth': SISO output shape is (N,)."""
        syms = self._qpsk_symbols(xp, N=512)
        phi = sync.recover_carrier_phase_pll(
            syms,
            "psk",
            4,
            loop_filter="butterworth",
            loop_bandwidth_normalized=1e-3,
        )
        assert phi.shape == syms.shape

    def test_butterworth_output_dtype(self, backend_device, xp):
        """loop_filter='butterworth': output dtype is float64."""
        syms = self._qpsk_symbols(xp, N=256)
        phi = sync.recover_carrier_phase_pll(
            syms,
            "psk",
            4,
            loop_filter="butterworth",
            loop_bandwidth_normalized=1e-3,
        )
        assert phi.dtype == xp.float64

    def test_butterworth_output_finite(self, backend_device, xp):
        """Butterworth loop output should be finite for clean QPSK symbols."""
        import numpy as np
        from commstools.mapping import gray_constellation

        N = 512
        phase_offset = 0.15  # radians
        const = gray_constellation("qpsk", 4)
        rng = np.random.default_rng(42)
        syms_clean = xp.asarray(const[rng.integers(0, 4, N)].astype(np.complex64))
        syms_rotated = syms_clean * np.exp(1j * phase_offset)

        phi = sync.recover_carrier_phase_pll(
            syms_rotated,
            "psk",
            4,
            loop_filter="butterworth",
            loop_bandwidth_normalized=1e-2,
        )
        # Phi should be finite everywhere (no NaN or Inf)
        assert bool(xp.all(xp.isfinite(phi))), (
            "Butterworth PLL output contains non-finite values"
        )

    def test_butterworth_mimo_output_shape(self, backend_device, xp):
        """loop_filter='butterworth': MIMO (C, N) output shape is (C, N)."""
        import numpy as np
        from commstools.mapping import gray_constellation

        C, N = 2, 256
        const = gray_constellation("qpsk", 4)
        rng = np.random.default_rng(3)
        syms = xp.asarray(
            const[rng.integers(0, 4, C * N)].reshape(C, N).astype(np.complex64)
        )
        phi = sync.recover_carrier_phase_pll(
            syms,
            "psk",
            4,
            loop_filter="butterworth",
            loop_bandwidth_normalized=1e-3,
        )
        assert phi.shape == (C, N)

    def test_butterworth_invalid_bandwidth_raises(self, backend_device, xp):
        """loop_bandwidth_normalized outside (0, 0.5) should raise ValueError."""
        syms = self._qpsk_symbols(xp, N=64)
        with pytest.raises(ValueError, match="loop_bandwidth_normalized"):
            sync.recover_carrier_phase_pll(
                syms,
                "psk",
                4,
                loop_filter="butterworth",
                loop_bandwidth_normalized=0.6,
            )

    def test_invalid_loop_filter_raises(self, backend_device, xp):
        """Unknown loop_filter value should raise ValueError."""
        syms = self._qpsk_symbols(xp, N=64)
        with pytest.raises(ValueError, match="loop_filter"):
            sync.recover_carrier_phase_pll(syms, "psk", 4, loop_filter="kalman")


# =============================================================================
# CORRECT TIMING — SCALAR MODES AND PER-CHANNEL VECTORIZED GATHER
# =============================================================================


class TestCorrectTiming:
    """Tests for correct_timing scalar zero/slice modes and per-channel vectorized paths."""

    def test_scalar_zero_mode_positive_shift(self, backend_device, xp):
        """Scalar coarse offset with mode='zero': signal shifts left, tail zero-padded."""
        import numpy as np

        N, shift = 100, 10
        sig = xp.asarray(np.arange(N, dtype=np.complex64))
        out = sync.correct_timing(sig, shift, mode="zero")
        assert out.shape == sig.shape
        # After left-shift by 10: out[0] == sig[10], tail is zeros
        assert float(out[0].real) == pytest.approx(float(sig[shift].real))
        assert float(out[-1].real) == pytest.approx(0.0)

    def test_scalar_zero_mode_negative_shift(self, backend_device, xp):
        """Scalar negative coarse offset with mode='zero': signal shifts right, head zero-padded."""
        import numpy as np

        N, shift = 100, -5
        sig = xp.asarray(np.ones(N, dtype=np.complex64))
        out = sync.correct_timing(sig, shift, mode="zero")
        assert out.shape == sig.shape
        # Head should be zero-padded
        assert float(out[0].real) == pytest.approx(0.0)

    def test_scalar_slice_mode(self, backend_device, xp):
        """Scalar coarse offset with mode='slice': output is shorter by offset."""
        import numpy as np

        N, shift = 100, 15
        sig = xp.asarray(np.ones(N, dtype=np.complex64))
        out = sync.correct_timing(sig, shift, mode="slice")
        assert out.shape[-1] == N - shift

    def test_per_channel_circular_mode(self, backend_device, xp):
        """Per-channel array offset with mode='circular': each channel rolled independently."""
        import numpy as np

        C, N = 2, 64
        rng = np.random.default_rng(20)
        sig = xp.asarray(
            (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
                np.complex64
            )
        )
        offsets = xp.asarray(np.array([3, 7], dtype=np.int64))
        out = sync.correct_timing(sig, offsets, mode="circular")
        assert out.shape == (C, N)

    def test_per_channel_zero_mode(self, backend_device, xp):
        """Per-channel array offset with mode='zero': output same shape, tail zeroed."""
        import numpy as np

        C, N = 2, 64
        sig = xp.asarray(np.ones((C, N), dtype=np.complex64))
        offsets = xp.asarray(np.array([4, 8], dtype=np.int64))
        out = sync.correct_timing(sig, offsets, mode="zero")
        assert out.shape == (C, N)

    def test_per_channel_slice_mode(self, backend_device, xp):
        """Per-channel array offset with mode='slice': output length is N - max(offset)."""
        import numpy as np

        C, N = 2, 64
        offsets = np.array([3, 10])
        sig = xp.asarray(np.ones((C, N), dtype=np.complex64))
        out = sync.correct_timing(sig, xp.asarray(offsets), mode="slice")
        assert out.shape == (C, N - max(offsets))


# =============================================================================
# CORRECT TIMING — INVALID MODE ERROR PATHS
# =============================================================================


class TestCorrectTimingErrors:
    """ValueError for unknown mode — scalar and per-channel paths."""

    def test_scalar_unknown_mode_raises(self, backend_device, xp):
        """Scalar offset with unsupported mode raises ValueError."""
        import numpy as np

        sig = xp.asarray(np.ones(64, dtype=np.complex64))
        with pytest.raises(ValueError, match="Unknown mode"):
            sync.correct_timing(sig, 4, mode="wrap")

    def test_per_channel_unknown_mode_raises(self, backend_device, xp):
        """Per-channel offset with unsupported mode raises ValueError."""
        import numpy as np

        sig = xp.asarray(np.ones((2, 64), dtype=np.complex64))
        offsets = xp.asarray(np.array([2, 4], dtype=np.int64))
        with pytest.raises(ValueError, match="Unknown mode"):
            sync.correct_timing(sig, offsets, mode="wrap")


# =============================================================================
# CORRECT FREQUENCY OFFSET — REAL AND MIMO BRANCHES
# =============================================================================


class TestCorrectFrequencyOffsetBranches:
    """Branches for real-valued and MIMO input in correct_frequency_offset."""

    def test_real_float32_input(self, backend_device, xp):
        """Real float32 input is cast to complex64 and frequency-corrected."""

        N = 256
        t = xp.arange(N, dtype=xp.float32) / 1e6
        # Simple real cosine as stand-in for a real-baseband signal
        sig = xp.cos(t)
        sig = sig.astype(xp.float32)
        out = sync.correct_frequency_offset(sig, offset=5000.0, sampling_rate=1e6)
        assert out.dtype == xp.complex64
        assert out.shape == sig.shape

    def test_mimo_input_broadcasts_mixer(self, backend_device, xp):
        """MIMO (C, N) input: mixer is broadcast over channels without error."""
        import numpy as np

        C, N = 2, 256
        rng = np.random.default_rng(99)
        sig = xp.asarray(
            (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
                np.complex64
            )
        )
        out = sync.correct_frequency_offset(sig, offset=3000.0, sampling_rate=1e6)
        assert out.shape == (C, N)
        assert out.dtype == xp.complex64


class TestIQImbalanceCompensation:
    """Tests for compensate_iq_imbalance_lowdin and compensate_iq_imbalance_gram_schmidt."""

    # κ = |E[r²]| / E[|r|²]: zero for a circular signal, positive for improper
    def _kappa(self, xp, x):
        return float(xp.abs(xp.mean(x**2))) / float(xp.mean(xp.abs(x) ** 2))

    def _make_imbalanced(self, xp, N=8192, seed=42):
        rng = xp.random.RandomState(seed)
        s = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        r = apply_iq_imbalance(s, amplitude_imbalance_db=2.0, phase_imbalance_deg=5.0)
        return s, r

    # --- Löwdin ---

    def test_lowdin_restores_circularity(self, backend_device, xp):
        """Löwdin compensation should drive κ to near zero."""
        _, r = self._make_imbalanced(xp)
        kappa_before = self._kappa(xp, r)
        out = sync.compensate_iq_imbalance_lowdin(r)
        kappa_after = self._kappa(xp, out)
        assert kappa_after < 0.03
        assert kappa_after < kappa_before / 5

    def test_lowdin_iq_balance(self, backend_device, xp):
        """After Löwdin, I and Q should have equal power and be orthogonal."""
        _, r = self._make_imbalanced(xp)
        out = sync.compensate_iq_imbalance_lowdin(r)
        I, Q = out.real, out.imag
        power_ratio = float(xp.mean(I**2)) / float(xp.mean(Q**2))
        cross_corr = float(xp.abs(xp.mean(I * Q))) / float(xp.mean(xp.abs(out) ** 2))
        assert abs(power_ratio - 1.0) < 0.05
        assert cross_corr < 0.02

    def test_lowdin_preserves_power(self, backend_device, xp):
        """Löwdin output power should equal input power."""
        _, r = self._make_imbalanced(xp)
        P_in = float(xp.mean(xp.abs(r) ** 2))
        out = sync.compensate_iq_imbalance_lowdin(r)
        P_out = float(xp.mean(xp.abs(out) ** 2))
        assert abs(P_out - P_in) / P_in < 0.01

    def test_lowdin_identity_on_balanced_signal(self, backend_device, xp, xpt):
        """Löwdin applied to a balanced signal should return it unchanged."""
        N = 8192
        rng = xp.random.RandomState(0)
        s = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        out = sync.compensate_iq_imbalance_lowdin(s)
        xpt.assert_allclose(xp.abs(out), xp.abs(s), atol=0.05)

    def test_lowdin_siso_shape(self, backend_device, xp):
        """Löwdin: SISO (N,) input should return (N,)."""
        _, r = self._make_imbalanced(xp)
        out = sync.compensate_iq_imbalance_lowdin(r)
        assert out.shape == r.shape

    def test_lowdin_mimo_shape(self, backend_device, xp):
        """Löwdin: MIMO (C, N) input should return (C, N)."""
        _, r = self._make_imbalanced(xp)
        r_mimo = xp.stack([r, r])  # (2, N)
        out = sync.compensate_iq_imbalance_lowdin(r_mimo)
        assert out.shape == r_mimo.shape

    def test_lowdin_dtype_preserved(self, backend_device, xp):
        """Löwdin output dtype should match input."""
        _, r = self._make_imbalanced(xp)
        out = sync.compensate_iq_imbalance_lowdin(r)
        assert out.dtype == xp.complex64

    # --- Gram-Schmidt ---

    def test_gram_schmidt_restores_circularity(self, backend_device, xp):
        """Gram-Schmidt compensation should drive κ to near zero."""
        _, r = self._make_imbalanced(xp)
        kappa_before = self._kappa(xp, r)
        out = sync.compensate_iq_imbalance_gram_schmidt(r)
        kappa_after = self._kappa(xp, out)
        assert kappa_after < 0.03
        assert kappa_after < kappa_before / 5

    def test_gram_schmidt_iq_orthogonality(self, backend_device, xp):
        """After Gram-Schmidt, I and Q should be orthogonal."""
        _, r = self._make_imbalanced(xp)
        out = sync.compensate_iq_imbalance_gram_schmidt(r)
        I, Q = out.real, out.imag
        cross_corr = float(xp.abs(xp.mean(I * Q))) / float(xp.mean(xp.abs(out) ** 2))
        assert cross_corr < 0.02

    def test_gram_schmidt_preserves_power(self, backend_device, xp):
        """Gram-Schmidt output power should equal input power."""
        _, r = self._make_imbalanced(xp)
        P_in = float(xp.mean(xp.abs(r) ** 2))
        out = sync.compensate_iq_imbalance_gram_schmidt(r)
        P_out = float(xp.mean(xp.abs(out) ** 2))
        assert abs(P_out - P_in) / P_in < 0.01

    def test_gram_schmidt_siso_shape(self, backend_device, xp):
        """Gram-Schmidt: SISO (N,) input should return (N,)."""
        _, r = self._make_imbalanced(xp)
        out = sync.compensate_iq_imbalance_gram_schmidt(r)
        assert out.shape == r.shape

    def test_gram_schmidt_mimo_shape(self, backend_device, xp):
        """Gram-Schmidt: MIMO (C, N) input should return (C, N)."""
        _, r = self._make_imbalanced(xp)
        r_mimo = xp.stack([r, r])  # (2, N)
        out = sync.compensate_iq_imbalance_gram_schmidt(r_mimo)
        assert out.shape == r_mimo.shape

    def test_gram_schmidt_dtype_preserved(self, backend_device, xp):
        """Gram-Schmidt output dtype should match input."""
        _, r = self._make_imbalanced(xp)
        out = sync.compensate_iq_imbalance_gram_schmidt(r)
        assert out.dtype == xp.complex64
