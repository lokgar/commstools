import numpy as np
import matplotlib.pyplot as plt
from commstools import plotting
from commstools.dsp import filtering


class TestPlotting:
    def test_filter_response(self):
        taps = filtering.rrc_taps(sps=8)

        # Test with creating new figure
        fig, (ax1, ax2, ax3) = plotting.filter_response(taps, sps=8)

        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None

        plt.close(fig)

    def test_filter_response_complex(self):
        taps = filtering.rrc_taps(sps=8) + 1j * filtering.rrc_taps(sps=8)

        fig, (ax1, ax2, ax3) = plotting.filter_response(taps, sps=8)

        assert fig is not None
        plt.close(fig)

    def test_time_domain(self):
        samples = np.array([1.0, 2.0, 3.0])
        fig, ax = plotting.time_domain(samples, sampling_rate=100)
        assert fig is not None
        plt.close(fig)

    def test_eye_diagram(self):
        samples = np.array([1.0, -1.0, 1.0, -1.0] * 4)
        fig, ax = plotting.eye_diagram(samples, sps=2)
        assert fig is not None
        plt.close(fig)

    def test_psd(self):
        samples = np.random.randn(100)
        fig, ax = plotting.psd(samples, sampling_rate=100)
        assert fig is not None
        plt.close(fig)
