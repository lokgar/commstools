import numpy as np
from commstools.dsp import sequences, mapping, filters
from commstools import waveforms
from commstools import Signal


class TestSequences:
    def test_prbs_generation(self):
        # PRBS7 length 127
        seq = sequences.prbs(length=127, order=7)
        assert len(seq) == 127
        assert np.all(np.isin(seq, [0, 1]))

        # Check repeatability
        seq2 = sequences.prbs(length=127, order=7)
        np.testing.assert_array_equal(seq, seq2)


class TestMapping:
    def test_ook_map(self):
        bits = np.array([0, 1, 0, 1])
        symbols = mapping.ook_map(bits)
        np.testing.assert_array_equal(symbols, bits)
        assert symbols.dtype == float


class TestFilters:
    def test_boxcar_taps(self):
        taps = filters.boxcar_taps(sps=4)
        assert len(taps) == 4
        np.testing.assert_array_equal(taps, np.ones(4))

    def test_expand(self):
        symbols = np.array([1, 2])
        expanded = filters.expand(symbols, factor=2)
        expected = np.array([1, 0, 2, 0])
        np.testing.assert_array_equal(expanded, expected)

    def test_shape_pulse_boxcar(self):
        symbols = np.array([1, 0])
        # Pass pulse_shape="boxcar" directly
        shaped = filters.shape_pulse(symbols, sps=4, pulse_shape="boxcar")
        # 1 -> 1 1 1 1, 0 -> 0 0 0 0
        # shape_pulse with boxcar now truncates to valid symbol duration
        # len = 2*4 = 8
        assert len(shaped) == 8
        expected_start = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_allclose(shaped, expected_start, atol=1e-10)

    def test_sinc_taps(self):
        taps = filters.sinc_taps(sps=4, bandwidth=0.5, span=4)
        assert len(taps) > 0
        # Check that taps are normalized
        assert abs(np.sum(taps) - 4.0) < 0.1  # Should sum to sps

    def test_fir_filter(self):
        """Test generic FIR filter function."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        taps = filters.boxcar_taps(sps=3)
        filtered = filters.fir_filter(samples, taps, mode="same")
        # Should produce output of same length
        assert len(filtered) == len(samples)
        # Should smooth the signal
        assert not np.array_equal(filtered, samples)

    def test_upsample(self):
        samples = np.array([1.0, 2.0, 3.0])
        upsampled = filters.upsample(samples, factor=2, filter_type="sinc")
        # Output should be 2x length
        assert len(upsampled) == 6

    def test_matched_filter(self):
        # Create a simple pulse
        pulse_taps = np.array([1, 2, 3, 2, 1])
        samples = np.array([0, 0, 1, 0, 0, 0, 0])
        # Apply matched filter
        filtered = filters.matched_filter(samples, pulse_taps, mode="same")
        assert len(filtered) == len(samples)


class TestWaveforms:
    def test_ook_waveform(self):
        bits = np.array([1, 0])
        sig = waveforms.ook(bits, sps=4, pulse_shape="boxcar", sampling_rate=100)

        assert isinstance(sig, Signal)
        assert sig.modulation_format == "OOK"
        assert sig.sampling_rate == 100

        # Check samples
        # 1 -> 1 1 1 1, 0 -> 0 0 0 0
        expected_start = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_allclose(sig.samples[:8], expected_start, atol=1e-10)

    def test_ook_impulse(self):
        bits = np.array([1, 0])
        sig = waveforms.ook(bits, sps=4, pulse_shape="none", sampling_rate=100)

        # 1 -> 1 0 0 0, 0 -> 0 0 0 0
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(sig.samples, expected)

    def test_ook_float_sps(self):
        bits = np.array([1, 0])
        # Use float SPS = 4.0
        sig = waveforms.ook(bits, sps=4.0, pulse_shape="boxcar", sampling_rate=100)

        assert sig.symbol_rate == 25.0

        # Check samples
        expected_start = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_allclose(sig.samples[:8], expected_start, atol=1e-10)
