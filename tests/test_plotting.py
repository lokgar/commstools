import numpy as np
import matplotlib.pyplot as plt
from commstools import plotting
from commstools.dsp import filters


class TestPlotting:
    def test_filter_response(self):
        taps = filters.rrc_taps(sps=8)

        # Test with creating new figure
        fig, (ax1, ax2, ax3) = plotting.filter_response(taps, fs=100)

        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None

        plt.close(fig)

    def test_filter_response_complex(self):
        taps = filters.rrc_taps(sps=8) + 1j * filters.rrc_taps(sps=8)

        fig, (ax1, ax2, ax3) = plotting.filter_response(taps, fs=100)

        assert fig is not None
        plt.close(fig)
