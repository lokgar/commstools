"""
Comprehensive tests for DSP building blocks and filters.
"""

import pytest
import numpy as np
from commstools import set_backend
from commstools.dsp import filters

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
        expanded = filters.expand(symbols, factor=3)

        expected = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0])
        np.testing.assert_array_equal(expanded, expected)

    @pytest.mark.parametrize("backend_name", backends)
    def test_expand_complex(self, backend_name):
        """Test expand with complex values."""
        set_backend(backend_name)
        symbols = np.array([1 + 1j, 2 + 2j])
        expanded = filters.expand(symbols, factor=2)

        expected = np.array([1 + 1j, 0, 2 + 2j, 0])
        np.testing.assert_array_almost_equal(expanded, expected)

    @pytest.mark.parametrize("backend_name", backends)
    def test_expand_single_symbol(self, backend_name):
        """Test expand with single symbol."""
        set_backend(backend_name)
        symbols = np.array([5.0])
        expanded = filters.expand(symbols, factor=4)

        expected = np.array([5.0, 0, 0, 0])
        np.testing.assert_array_equal(expanded, expected)

    @pytest.mark.parametrize("backend_name", backends)
    def test_upsample_preserves_length(self, backend_name):
        """Test that upsample produces correct output length."""
        set_backend(backend_name)
        samples = np.array([1.0, 2.0, 3.0, 4.0])
        factor = 3

        upsampled = filters.upsample(samples, factor=factor, filter_type="sinc")

        assert len(upsampled) == len(samples) * factor

    @pytest.mark.parametrize("backend_name", backends)
    def test_upsample_dc_preservation(self, backend_name):
        """Test that upsample preserves DC component."""
        set_backend(backend_name)
        # Constant signal
        samples = np.ones(20) * 5.0
        factor = 2

        upsampled = filters.upsample(samples, factor=factor, filter_type="sinc")

        # DC should be preserved
        assert np.abs(np.mean(upsampled) - 5.0) < 0.5

    @pytest.mark.parametrize("backend_name", backends)
    def test_decimate_length(self, backend_name):
        """Test that decimate produces correct output length."""
        set_backend(backend_name)
        samples = np.random.randn(100)
        factor = 5

        decimated = filters.decimate(samples, factor=factor)

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
        decimated = filters.decimate(signal, factor=10)

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
        resampled = filters.resample(samples, up=2, down=1)
        assert len(resampled) == len(samples) * 2

        # Downsample by 2
        resampled = filters.resample(samples, up=1, down=2)
        assert len(resampled) == len(samples) // 2

    @pytest.mark.parametrize("backend_name", backends)
    def test_resample_rational_rate(self, backend_name):
        """Test resample with rational rate conversion."""
        set_backend(backend_name)
        samples = np.random.randn(100)

        # Resample by 3/2
        resampled = filters.resample(samples, up=3, down=2)

        expected_length = int(len(samples) * 3 / 2)
        assert abs(len(resampled) - expected_length) <= 1  # Allow for rounding


class TestFilterTaps:
    """Test filter tap generation functions."""

    def test_boxcar_taps_length(self):
        """Test boxcar taps have correct length."""
        taps = filters.boxcar_taps(sps=8)
        assert len(taps) == 8
        np.testing.assert_array_equal(taps, np.ones(8))

    def test_gaussian_taps_normalization(self):
        """Test Gaussian taps are properly normalized."""
        sps = 10
        taps = filters.gaussian_taps(sps=sps, bt=0.3, span=4)

        # Should sum to approximately sps
        assert abs(np.sum(taps) - sps) < 0.1

    def test_gaussian_taps_bt_effect(self):
        """Test that BT parameter affects Gaussian pulse width."""
        sps = 8

        taps_narrow = filters.gaussian_taps(sps=sps, bt=0.2, span=4)
        taps_wide = filters.gaussian_taps(sps=sps, bt=0.5, span=4)

        # Higher BT means wider pulse (lower peak)
        assert np.max(taps_narrow) < np.max(taps_wide)

    def test_rrc_taps_symmetry(self):
        """Test RRC taps are symmetric."""
        taps = filters.rrc_taps(sps=8, rolloff=0.35, span=4)

        np.testing.assert_array_almost_equal(taps, taps[::-1])

    def test_rrc_taps_unit_energy(self):
        """Test RRC taps have unit energy."""
        taps = filters.rrc_taps(sps=8, rolloff=0.35, span=6)

        energy = np.sum(np.abs(taps) ** 2)
        np.testing.assert_almost_equal(energy, 1.0, decimal=5)

    def test_rrc_taps_rolloff_effect(self):
        """Test that rolloff parameter affects bandwidth."""
        sps = 8

        taps_low = filters.rrc_taps(sps=sps, rolloff=0.1, span=6)
        taps_high = filters.rrc_taps(sps=sps, rolloff=0.8, span=6)

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
        sps = 10
        taps = filters.sinc_taps(sps=sps, bandwidth=0.5, span=6)

        # Should sum to approximately sps
        assert abs(np.sum(taps) - sps) < 0.5

    def test_get_taps_factory(self):
        """Test get_taps factory function."""
        sps = 8

        # Test all filter types
        taps_none = filters.get_taps("none", sps)
        assert len(taps_none) == 1

        taps_boxcar = filters.get_taps("boxcar", sps)
        assert len(taps_boxcar) == sps

        taps_gaussian = filters.get_taps("gaussian", sps, bt=0.3, span=4)
        assert len(taps_gaussian) > sps

        taps_rrc = filters.get_taps("rrc", sps, rolloff=0.35, span=4)
        assert len(taps_rrc) > sps

        taps_sinc = filters.get_taps("sinc", sps, bandwidth=0.5, span=4)
        assert len(taps_sinc) > sps

    def test_get_taps_invalid_type(self):
        """Test get_taps raises error for invalid filter type."""
        with pytest.raises(ValueError):
            filters.get_taps("invalid_filter", sps=8)


class TestMatchedFiltering:
    """Test matched filtering operations."""

    def test_matched_filter_length(self):
        """Test matched filter output has correct length."""
        samples = np.random.randn(100)
        pulse_taps = filters.rrc_taps(sps=8, rolloff=0.35, span=4)

        filtered = filters.matched_filter(samples, pulse_taps, mode="same")

        assert len(filtered) == len(samples)

    def test_matched_filter_is_conjugate_reversed(self):
        """Test matched filter produces peak at correct location."""
        pulse_taps = filters.rrc_taps(sps=8, rolloff=0.35, span=4)

        # Apply matched filter to impulse
        impulse = np.zeros(100)
        impulse[50] = 1.0
        filtered = filters.matched_filter(impulse, pulse_taps, mode="same")

        # Peak should be at impulse location
        peak_idx = np.argmax(np.abs(filtered))
        assert peak_idx == 50

    def test_matched_filter_with_pulse_generation(self):
        """Test matched filter with generated pulse taps."""
        samples = np.random.randn(100)
        sps = 8

        # Generate pulse taps and apply matched filter
        pulse_taps = filters.rrc_taps(sps, rolloff=0.35)
        filtered = filters.matched_filter(samples, pulse_taps)

        # Should produce output of correct length
        assert len(filtered) == len(samples)
        # Should modify the signal
        assert not np.array_equal(filtered, samples)

    def test_matched_filter_snr_improvement(self):
        """Test that matched filter improves SNR."""
        # Create a clean pulse
        sps = 8
        pulse_taps = filters.rrc_taps(sps, rolloff=0.35, span=6)

        # Create signal: pulse + noise
        signal_len = 200
        signal = np.zeros(signal_len)
        pulse_start = 50
        signal[pulse_start : pulse_start + len(pulse_taps)] = pulse_taps

        # Add noise
        noise = np.random.randn(signal_len) * 0.5
        noisy_signal = signal + noise

        # Apply matched filter
        filtered = filters.matched_filter(noisy_signal, pulse_taps, mode="same")

        # Peak at center should be stronger after filtering
        peak_location = pulse_start + len(pulse_taps) // 2

        # Signal strength at peak
        signal_before = np.abs(noisy_signal[peak_location])
        signal_after = np.abs(filtered[peak_location])

        # Matched filter should concentrate energy
        assert signal_after > signal_before


class TestShapePulse:
    """Test the high-level shape_pulse function."""

    def test_shape_pulse_length_boxcar(self):
        """Test shape_pulse with boxcar produces correct length."""
        symbols = np.array([1, 0, 1, 0])
        sps = 8

        shaped = filters.shape_pulse(symbols, sps, pulse_shape="boxcar")

        assert len(shaped) == len(symbols) * sps

    def test_shape_pulse_length_rrc(self):
        """Test shape_pulse with RRC produces correct length."""
        symbols = np.array([1, 0, 1, 0])
        sps = 8

        shaped = filters.shape_pulse(symbols, sps, pulse_shape="rrc", rolloff=0.35)

        # RRC uses 'same' mode, so length matches expanded signal
        assert len(shaped) == len(symbols) * sps

    def test_shape_pulse_none(self):
        """Test shape_pulse with no pulse shaping."""
        symbols = np.array([1, 2, 3])
        sps = 4

        shaped = filters.shape_pulse(symbols, sps, pulse_shape="none")

        # Should be equivalent to expand
        expanded = filters.expand(symbols, sps)
        np.testing.assert_array_equal(shaped, expanded)

    def test_shape_pulse_complex_symbols(self):
        """Test shape_pulse with complex symbols."""
        symbols = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        sps = 8

        shaped = filters.shape_pulse(symbols, sps, pulse_shape="rrc", rolloff=0.35)

        assert shaped.dtype == np.complex128 or shaped.dtype == np.complex64
        assert len(shaped) == len(symbols) * sps


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_expand_replaces_upsample(self):
        """Test that expand is the new name for upsample."""
        symbols = np.array([1, 2, 3])
        factor = 4

        # expand function should work
        expanded = filters.expand(symbols, factor)

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
        result = filters.expand(empty, factor=2)
        assert len(result) == 0

    def test_single_sample_filters(self):
        """Test filters with single sample input."""
        single = np.array([5.0])
        sps = 4

        shaped = filters.shape_pulse(single, sps, pulse_shape="boxcar")
        assert len(shaped) == sps

    def test_very_large_sps(self):
        """Test with very large samples per symbol."""
        symbols = np.array([1.0, 0.0])
        sps = 100

        expanded = filters.expand(symbols, sps)
        assert len(expanded) == len(symbols) * sps

    def test_fractional_sps_handling(self):
        """Test that fractional sps values work with resample."""
        symbols = np.array([1.0, 2.0])
        sps = 7.5  # Fractional

        # boxcar shape_pulse handles floats via resample
        shaped = filters.shape_pulse(symbols, sps, pulse_shape="boxcar")
        # Length should be approximately len(symbols) * sps
        expected_len = int(len(symbols) * sps)
        assert abs(len(shaped) - expected_len) <= 1
