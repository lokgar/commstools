"""
Comprehensive tests for DSP building blocks and filtering.
"""

import pytest
import numpy as np
from commstools import set_backend
from commstools.dsp import filtering, multirate

# Test both backends
backends = ["numpy"]
try:
    import jax

    backends.append("jax")
except ImportError:
    pass


class TestBuildingBlocks:
    """Test core DSP building block operations."""

    @pytest.mark.parametrize("backend_name", backends)
    def test_expand_basic(self, backend_name):
        """Test basic expand operation."""
        set_backend(backend_name)
        symbols = np.array([1, 2, 3])
        expanded = multirate.expand(symbols, factor=3)

        expected = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0])
        np.testing.assert_array_equal(expanded, expected)

    @pytest.mark.parametrize("backend_name", backends)
    def test_expand_complex(self, backend_name):
        """Test expand with complex values."""
        set_backend(backend_name)
        symbols = np.array([1 + 1j, 2 + 2j])
        expanded = multirate.expand(symbols, factor=2)

        expected = np.array([1 + 1j, 0, 2 + 2j, 0])
        np.testing.assert_array_almost_equal(expanded, expected)

    @pytest.mark.parametrize("backend_name", backends)
    def test_expand_single_symbol(self, backend_name):
        """Test expand with single symbol."""
        set_backend(backend_name)
        symbols = np.array([5.0])
        expanded = multirate.expand(symbols, factor=4)

        expected = np.array([5.0, 0, 0, 0])
        np.testing.assert_array_equal(expanded, expected)

    @pytest.mark.parametrize("backend_name", backends)
    def test_upsample_preserves_length(self, backend_name):
        """Test that upsample produces correct output length."""
        set_backend(backend_name)
        samples = np.array([1.0, 2.0, 3.0, 4.0])
        factor = 3

        upsampled = multirate.upsample(samples, factor=factor)

        assert len(upsampled) == len(samples) * factor

    @pytest.mark.parametrize("backend_name", backends)
    def test_upsample_dc_preservation(self, backend_name):
        """Test that upsample preserves DC component."""
        set_backend(backend_name)
        # Constant signal
        samples = np.ones(20) * 5.0
        factor = 2

        upsampled = multirate.upsample(samples, factor=factor)

        # DC should be preserved
        assert np.abs(np.mean(upsampled) - 5.0) < 0.5

    @pytest.mark.parametrize("backend_name", backends)
    def test_decimate_length(self, backend_name):
        """Test that decimate produces correct output length."""
        set_backend(backend_name)
        samples = np.random.randn(100)
        factor = 5

        decimated = multirate.decimate(samples, factor=factor)

        expected_length = len(samples) // factor
        assert len(decimated) == expected_length

    @pytest.mark.parametrize("backend_name", backends)
    def test_decimate_preserves_low_freq(self, backend_name):
        """Test that decimate preserves signal trend."""
        set_backend(backend_name)
        # Create a slowly varying signal
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz signal

        # Decimate by 10
        decimated = multirate.decimate(signal, factor=10)

        # Should still see the sinusoidal pattern
        # Check zero crossings are preserved (roughly)
        assert len(decimated) == 100
        assert np.min(decimated) < 0
        assert np.max(decimated) > 0

    @pytest.mark.parametrize("backend_name", backends)
    def test_resample_basic(self, backend_name):
        """Test basic resample operation."""
        set_backend(backend_name)
        samples = np.array([1.0, 2.0, 3.0, 4.0])

        # Upsample by 2
        resampled = multirate.resample(samples, up=2, down=1)
        assert len(resampled) == len(samples) * 2

        # Downsample by 2
        resampled = multirate.resample(samples, up=1, down=2)
        assert len(resampled) == len(samples) // 2

    @pytest.mark.parametrize("backend_name", backends)
    def test_resample_rational_rate(self, backend_name):
        """Test resample with rational rate conversion."""
        set_backend(backend_name)
        samples = np.random.randn(100)

        # Resample by 3/2
        resampled = multirate.resample(samples, up=3, down=2)

        expected_length = int(len(samples) * 3 / 2)
        assert abs(len(resampled) - expected_length) <= 1  # Allow for rounding


class TestFilterTaps:
    """Test filter tap generation functions."""

    def test_boxcar_taps_length(self):
        """Test boxcar taps have correct length."""
        taps = filtering.boxcar_taps(sps=8)
        assert len(taps) == 8
        np.testing.assert_array_equal(taps, np.ones(8) / 8)

    def test_gaussian_taps_normalization(self):
        """Test Gaussian taps are properly normalized."""
        sps = 10
        taps = filtering.gaussian_taps(sps=sps, bt=0.3, span=4)

        # Should sum to approximately 1.0 (Unity Gain)
        assert abs(np.sum(taps) - 1.0) < 0.1

    def test_gaussian_taps_bt_effect(self):
        """Test that BT parameter affects Gaussian pulse width."""
        sps = 8

        taps_narrow = filtering.gaussian_taps(sps=sps, bt=0.2, span=4)
        taps_wide = filtering.gaussian_taps(sps=sps, bt=0.5, span=4)

        # Higher BT means wider pulse (lower peak)
        assert np.max(taps_narrow) < np.max(taps_wide)

    def test_rrc_taps_symmetry(self):
        """Test RRC taps are symmetric."""
        taps = filtering.rrc_taps(sps=8, rolloff=0.35, span=4)

        np.testing.assert_array_almost_equal(taps, taps[::-1])

    def test_rrc_taps_unit_energy(self):
        """Test RRC taps have unit energy."""
        taps = filtering.rrc_taps(sps=8, rolloff=0.35, span=6)

        # RRC taps are now Unity Gain normalized, not Unit Energy
        # But let's check sum is 1.0
        gain = np.sum(taps)
        np.testing.assert_almost_equal(gain, 1.0, decimal=5)

    def test_rrc_taps_rolloff_effect(self):
        """Test that rolloff parameter affects bandwidth."""
        sps = 8

        taps_low = filtering.rrc_taps(sps=sps, rolloff=0.1, span=6)
        taps_high = filtering.rrc_taps(sps=sps, rolloff=0.8, span=6)

        # Higher rolloff should have shorter tails (more compact in time)
        # Check energy concentration
        center = len(taps_low) // 2
        window = sps

        energy_center_low = np.sum(
            np.abs(taps_low[center - window : center + window]) ** 2
        )
        energy_center_high = np.sum(
            np.abs(taps_high[center - window : center + window]) ** 2
        )

        # Higher rolloff concentrates energy more in time
        assert energy_center_high > energy_center_low

    def test_sinc_taps_dc_gain(self):
        """Test sinc taps have correct DC gain."""
        # New signature: num_taps, cutoff_norm
        taps = filtering.sinc_taps(num_taps=21, cutoff_norm=0.1)

        # Should sum to approximately 1.0
        assert abs(np.sum(taps) - 1.0) < 0.1


class TestMatchedFiltering:
    """Test matched filtering operations."""

    def test_matched_filter_length(self):
        """Test matched filter output has correct length."""
        samples = np.random.randn(100)
        pulse_taps = filtering.rrc_taps(sps=8, rolloff=0.35, span=4)

        filtered = filtering.matched_filter(samples, pulse_taps, mode="same")

        assert len(filtered) == len(samples)

    def test_matched_filter_is_conjugate_reversed(self):
        """Test matched filter produces peak at correct location."""
        pulse_taps = filtering.rrc_taps(sps=8, rolloff=0.35, span=4)

        # Apply matched filter to impulse
        impulse = np.zeros(100)
        impulse[50] = 1.0
        filtered = filtering.matched_filter(impulse, pulse_taps, mode="same")

        # Peak should be at impulse location
        peak_idx = np.argmax(np.abs(filtered))
        assert peak_idx == 50

    def test_matched_filter_with_pulse_generation(self):
        """Test matched filter with generated pulse taps."""
        samples = np.random.randn(100)
        sps = 8

        # Generate pulse taps and apply matched filter
        pulse_taps = filtering.rrc_taps(sps, rolloff=0.35)
        filtered = filtering.matched_filter(samples, pulse_taps)

        # Should produce output of correct length
        assert len(filtered) == len(samples)
        # Should modify the signal
        assert not np.array_equal(filtered, samples)

    def test_matched_filter_snr_improvement(self):
        """Test that matched filter improves SNR."""
        set_backend("numpy")
        # Create a clean pulse
        sps = 8
        pulse_taps = filtering.rrc_taps(sps, rolloff=0.35, span=6)

        # Create signal: pulse + noise
        signal_len = 200
        signal = np.zeros(signal_len)
        pulse_start = 50
        signal[pulse_start : pulse_start + len(pulse_taps)] = pulse_taps

        # Add noise
        np.random.seed(42)
        noise = np.random.randn(signal_len) * 0.1
        noisy_signal = signal + noise

        # Apply matched filter
        filtered = filtering.matched_filter(noisy_signal, pulse_taps, mode="same")

        # Peak at center should be stronger after filtering
        peak_location = pulse_start + len(pulse_taps) // 2

        # Signal strength at peak
        signal_before = np.abs(noisy_signal[peak_location])
        signal_after = np.abs(filtered[peak_location])

        # Matched filter should concentrate energy
        assert signal_after > signal_before

    def test_matched_filter_taps_normalization(self):
        """Test matched filter with taps normalization."""
        samples = np.random.randn(100)
        sps = 8
        pulse_taps = filtering.rrc_taps(sps, rolloff=0.35)

        # Test energy normalization
        # We can't easily check the internal taps, but we can check the output scaling
        # compared to manual normalization.

        filtered_energy = filtering.matched_filter(
            samples, pulse_taps, taps_normalization="energy"
        )

        # Manual normalization
        taps_energy = pulse_taps / np.sqrt(np.sum(np.abs(pulse_taps) ** 2))
        filtered_manual = filtering.fir_filter(samples, np.conj(taps_energy[::-1]))

        np.testing.assert_allclose(filtered_energy, filtered_manual, rtol=1e-5)

    def test_matched_filter_normalize_output(self):
        """Test matched filter with output normalization."""
        samples = np.random.randn(100)
        sps = 8
        pulse_taps = filtering.rrc_taps(sps, rolloff=0.35)

        filtered = filtering.matched_filter(samples, pulse_taps, normalize_output=True)

        # Max amplitude should be 1.0 (or very close if signal is all zeros, but here random)
        assert abs(np.max(np.abs(filtered)) - 1.0) < 1e-6


class TestShapePulse:
    """Test the high-level shape_pulse function."""

    def test_shape_pulse_length_boxcar(self):
        """Test shape_pulse with boxcar produces correct length."""
        symbols = np.array([1, 0, 1, 0])
        sps = 8

        shaped = filtering.shape_pulse(symbols, sps, filter_span=4, pulse_shape="boxcar")

        assert len(shaped) == len(symbols) * sps

    def test_shape_pulse_length_rrc(self):
        """Test shape_pulse with RRC produces correct length."""
        symbols = np.array([1, 0, 1, 0])
        sps = 8

        shaped = filtering.shape_pulse(
            symbols, sps, filter_span=4, pulse_shape="rrc", rrc_rolloff=0.35
        )

        # RRC uses 'same' mode, so length matches expanded signal
        assert len(shaped) == len(symbols) * sps

    def test_shape_pulse_none(self):
        """Test shape_pulse with no pulse shaping."""
        symbols = np.array([1, 2, 3])
        sps = 4

        shaped = filtering.shape_pulse(symbols, sps, filter_span=4, pulse_shape="none")

        # Should be equivalent to expand * 1.0 (since we normalize to max 1.0)
        expanded = multirate.expand(symbols, sps)
        # normalize_max_amplitude will make the max value 1.0
        # Since expanded has 1, 2, 3, max is 3.
        # So output should be expanded / 3.
        # Wait, shape_pulse normalizes the OUTPUT.
        # If pulse_shape="none", h=[1]. Output is expanded.
        # Max of expanded is 3. So output will be expanded / 3.

        expected = expanded / np.max(np.abs(expanded))
        np.testing.assert_allclose(shaped, expected, rtol=1e-6)

    def test_shape_pulse_complex_symbols(self):
        """Test shape_pulse with complex symbols."""
        symbols = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        sps = 8

        shaped = filtering.shape_pulse(
            symbols, sps, filter_span=4, pulse_shape="rrc", rrc_rolloff=0.35
        )

        assert shaped.dtype == np.complex128 or shaped.dtype == np.complex64
        assert len(shaped) == len(symbols) * sps


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_expand_replaces_upsample(self):
        """Test that expand is the new name for upsample."""
        symbols = np.array([1, 2, 3])
        factor = 4

        # expand function should work
        expanded = multirate.expand(symbols, factor)

        # Should produce correct output
        assert len(expanded) == len(symbols) * factor
        assert expanded[0] == symbols[0]
        assert expanded[factor] == symbols[1]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_input(self):
        """Test handling of empty input arrays."""
        empty = np.array([])

        # Expand should handle empty input
        result = multirate.expand(empty, factor=2)
        assert len(result) == 0

    def test_single_sample_filters(self):
        """Test filters with single sample input."""
        single = np.array([5.0])
        sps = 4

        shaped = filtering.shape_pulse(single, sps, filter_span=4, pulse_shape="boxcar")
        assert len(shaped) == sps

    def test_very_large_sps(self):
        """Test with very large samples per symbol."""
        symbols = np.array([1.0, 0.0])
        sps = 100

        expanded = multirate.expand(symbols, sps)
        assert len(expanded) == len(symbols) * sps

    def test_fractional_sps_handling(self):
        """Test that fractional sps values work with resample."""
        symbols = np.array([1.0, 2.0])
        sps = 7.5  # Fractional

        # boxcar shape_pulse handles floats via resample
        shaped = filtering.shape_pulse(symbols, sps, filter_span=4, pulse_shape="boxcar")
        # Length should be approximately len(symbols) * sps
        expected_len = int(len(symbols) * sps)
        assert abs(len(shaped) - expected_len) <= 1
