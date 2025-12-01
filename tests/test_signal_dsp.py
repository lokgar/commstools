import numpy as np
from commstools import Signal
from commstools.dsp import filters


class TestSignalDSP:
    def test_fir_filter(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sig = Signal(samples=samples, sampling_rate=100, symbol_rate=10)

        taps = filters.boxcar_taps(sps=3)
        filtered_sig = sig.fir_filter(taps)

        assert isinstance(filtered_sig, Signal)
        assert len(filtered_sig.samples) == len(samples)
        assert filtered_sig.sampling_rate == 100

    def test_upsample(self):
        samples = np.array([1.0, 2.0])
        sig = Signal(samples=samples, sampling_rate=100, symbol_rate=10)

        upsampled_sig = sig.upsample(factor=2)

        assert isinstance(upsampled_sig, Signal)
        assert len(upsampled_sig.samples) == 4
        assert upsampled_sig.sampling_rate == 200

    def test_decimate(self):
        samples = np.random.randn(100)
        sig = Signal(samples=samples, sampling_rate=100, symbol_rate=10)

        decimated_sig = sig.decimate(factor=2)

        assert isinstance(decimated_sig, Signal)
        assert len(decimated_sig.samples) == 50
        assert decimated_sig.sampling_rate == 50

    def test_resample(self):
        samples = np.random.randn(100)
        sig = Signal(samples=samples, sampling_rate=100, symbol_rate=10)

        resampled_sig = sig.resample(up=3, down=2)

        assert isinstance(resampled_sig, Signal)
        expected_len = int(100 * 3 / 2)
        assert abs(len(resampled_sig.samples) - expected_len) <= 1
        assert resampled_sig.sampling_rate == 150

    def test_matched_filter(self):
        samples = np.random.randn(100)
        sig = Signal(samples=samples, sampling_rate=100, symbol_rate=10)

        pulse_taps = filters.rrc_taps(sps=4)
        filtered_sig = sig.matched_filter(pulse_taps)

        assert isinstance(filtered_sig, Signal)
        assert len(filtered_sig.samples) == 100
