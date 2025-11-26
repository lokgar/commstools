import numpy as np
from commstools.dsp import sequences, mapping, filters, waveforms
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
    def test_rect_taps(self):
        taps = filters.rect_taps(samples_per_symbol=4)
        assert len(taps) == 4
        np.testing.assert_array_equal(taps, np.ones(4))

    def test_upsample(self):
        symbols = np.array([1, 2])
        upsampled = filters.upsample(symbols, samples_per_symbol=2)
        expected = np.array([1, 0, 2, 0])
        np.testing.assert_array_equal(upsampled, expected)

    def test_shape_pulse_rect(self):
        symbols = np.array([1, 0])
        taps = filters.rect_taps(4)
        shaped = filters.shape_pulse(symbols, taps, samples_per_symbol=4)
        # 1 -> 1 1 1 1, 0 -> 0 0 0 0
        # Convolution mode 'full' will add tail
        # len = 2*4 + 4 - 1 = 11
        assert len(shaped) == 11
        expected_start = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(shaped[:8], expected_start)


class TestWaveforms:
    def test_ook_waveform(self):
        bits = np.array([1, 0])
        sig = waveforms.ook(
            bits, samples_per_symbol=4, pulse_type="rect", sampling_rate=100
        )

        assert isinstance(sig, Signal)
        assert sig.modulation_format == "OOK"
        assert sig.sampling_rate == 100

        # Check samples
        # 1 -> 1 1 1 1, 0 -> 0 0 0 0
        expected_start = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(sig.samples[:8], expected_start)

    def test_ook_impulse(self):
        bits = np.array([1, 0])
        sig = waveforms.ook(
            bits, samples_per_symbol=4, pulse_type="impulse", sampling_rate=100
        )

        # 1 -> 1 0 0 0, 0 -> 0 0 0 0
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(sig.samples, expected)
