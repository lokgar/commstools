"""Tests for signal visualization and plotting tools."""

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commstools.plotting import (
    _create_subplot_grid,
    _plot_eye_traces,
    apply_default_theme,
    constellation,
    eye_diagram,
    filter_response,
    ideal_constellation,
    psd,
    time_domain,
)


@pytest.mark.parametrize("type", ["line", "hist"])
def test_eye_diagram_real(backend_device, xp, type):
    """Verify eye diagram generation for real-valued signals."""
    # Random binary data
    samples = xp.random.randn(1000)
    sps = 4

    # Run plot
    try:
        fig, ax = eye_diagram(samples, sps=sps, type=type, show=False)
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    except ImportError:
        if backend_device == "gpu":
            pytest.skip("Skipping GPU plot test if import failure")
        raise


def test_eye_diagram_complex(backend_device, xp):
    """Verify eye diagram generation for complex-valued signals (I/Q)."""
    samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    sps = 4

    # Default (no ax provided) -> should create 2 subplots
    fig, ax = eye_diagram(samples, sps=sps, show=False)
    assert fig is not None
    assert isinstance(ax, (list, tuple, np.ndarray))
    assert len(ax) == 2
    plt.close(fig)

    # With provided ax
    fig, axes = plt.subplots(2, 1)
    fig, ax_ret = eye_diagram(samples, sps=sps, ax=axes, show=False)
    assert ax_ret is axes
    plt.close(fig)


def test_time_domain(backend_device, xp):
    """Verify time-domain waveform plotting."""
    samples = xp.arange(100)
    fig, ax = time_domain(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    plt.close(fig)


def test_psd(backend_device, xp):
    """Verify Power Spectral Density (PSD) plotting."""
    samples = xp.random.randn(256)
    fig, ax = psd(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    plt.close(fig)


def test_filter_response(backend_device, xp):
    """Verify filter response plotting (Impulse, Mag, Phase)."""
    taps = xp.array([1, 0.5, 0.25])
    fig, axes = filter_response(taps, sps=1.0, show=False)
    assert fig is not None
    assert len(axes) == 3
    plt.close(fig)


def test_constellation_1d(backend_device, xp):
    """Verify basic constellation density plot generation."""
    samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    fig, ax = constellation(samples, bins=50, show=False)
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_constellation_overlay_ideal(backend_device, xp):
    """Verify constellation plot with theoretical overlay enabled."""
    samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    fig, ax = constellation(
        samples, bins=50, overlay_ideal=True, modulation="qam", order=16, show=False
    )
    assert fig is not None
    plt.close(fig)


def test_constellation_mimo(backend_device, xp):
    """Verify MIMO constellation plotting uses an optimized grid layout."""
    # 4 channels -> should be 2x2 grid
    samples = xp.random.randn(4, 1000) + 1j * xp.random.randn(4, 1000)
    fig, axes = constellation(samples, bins=50, show=False)
    assert fig is not None
    assert axes.shape == (2, 2)  # 2x2 grid
    plt.close(fig)


def test_psd_mimo_grid(backend_device, xp):
    """Verify MIMO PSD plotting uses an optimized grid layout."""
    # 4 channels -> should be 2x2 grid
    samples = xp.random.randn(4, 256)
    fig, axes = psd(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    assert axes.shape == (2, 2)  # 2x2 grid, not (1, 4)
    plt.close(fig)


def test_time_domain_mimo_grid(backend_device, xp):
    """Verify MIMO time-domain plotting uses an optimized grid layout."""
    # 4 channels -> should be 2x2 grid
    samples = xp.random.randn(4, 100)
    fig, axes = time_domain(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    assert axes.shape == (2, 2)  # 2x2 grid, not (1, 4)
    plt.close(fig)


@pytest.mark.parametrize("plot_func", [psd, time_domain])
def test_multichannel_plots(backend_device, xp, plot_func):
    """Verify multichannel plotting for PSD and Time-Domain."""
    samples = xp.random.randn(2, 256)
    fig, axes = plot_func(samples, sampling_rate=1.0, show=False)
    assert fig is not None
    # For 2 channels, it should return 2 axes
    assert axes.size == 2
    plt.close(fig)


def test_eye_diagram_multichannel(backend_device, xp):
    """Verify eye diagram for multichannel complex signals."""
    samples = xp.random.randn(2, 100) + 1j * xp.random.randn(2, 100)
    # 2 channels, complex -> 2x2 grid = 4 subplots
    fig, axes = eye_diagram(samples, sps=4, show=False)
    assert fig is not None
    assert axes.shape == (2, 2)
    plt.close(fig)


def test_eye_diagram_axis_error(backend_device, xp):
    """Verify error when too few axes are provided for multichannel eye diagram."""
    samples = xp.zeros((2, 100))
    fig, ax = plt.subplots(1)
    with pytest.raises(ValueError, match="Not enough axes"):
        eye_diagram(samples, sps=4, ax=[ax], show=False)
    plt.close(fig)

    with pytest.raises(ValueError, match="must provide a list of axes"):
        eye_diagram(samples, sps=4, ax=ax, show=False)
    plt.close(fig)


def test_apply_theme():
    """Verify applying the visual theme."""
    apply_default_theme()


def test_psd_wavelength(backend_device, xp):
    """Verify PSD plotting with wavelength axis."""
    samples = xp.random.randn(256)
    # Wavelength requires positive frequencies, using real signal gets +/- f
    fig, ax = psd(
        samples, sampling_rate=1e15, x_axis="wavelength", domain="OPT", show=False
    )
    assert fig is not None
    assert "Wavelength" in ax.get_xlabel()
    plt.close(fig)


def test_psd_auto_scale(backend_device, xp):
    """Verify PSD auto-scaling for different frequency ranges."""
    samples = xp.random.randn(256)

    # THz
    fig, ax = psd(samples, sampling_rate=2e12, show=False)
    assert "THz" in ax.get_xlabel()
    plt.close(fig)

    # MHz
    fig, ax = psd(samples, sampling_rate=2e6, show=False)
    assert "MHz" in ax.get_xlabel()
    plt.close(fig)


def test_multichannel_overlay(backend_device, xp):
    """Verify overlaying multichannel signals on a single axis."""
    samples = xp.random.randn(2, 256)
    fig, ax = plt.subplots()

    # PSD overlay
    psd(samples, sampling_rate=1.0, ax=ax, show=False)

    # Time domain overlay
    time_domain(samples, sampling_rate=1.0, ax=ax, show=False)
    plt.close(fig)


def test_filter_response_axis_error(backend_device, xp):
    """Verify behavior when wrong number of axes are provided for filter response."""
    taps = xp.array([1, 0, 0, 1])
    fig, ax = plt.subplots(1)
    # Should warn and create new figure
    filter_response(taps, ax=ax, show=False)

    # Should accept list of 3 axes
    fig, axes = plt.subplots(3, 1)
    filter_response(taps, ax=axes, show=False)
    plt.close("all")


def test_ideal_constellation_basic(backend_device, xp):
    """Verify ideal constellation plotting."""
    fig, ax = ideal_constellation("qam", 16, show=False)
    assert fig is not None
    plt.close(fig)

    # Error case
    ret = ideal_constellation("invalid", 4, show=False)
    assert ret is None


def test_constellation_histogram_overlay_error(backend_device, xp):
    """Verify warning when overlaying ideal on histogram constellation with bad mod."""
    samples = xp.random.randn(100) + 1j * xp.random.randn(100)
    # bins > 0 triggers histogram mode
    constellation(
        samples, bins=10, overlay_ideal=True, modulation="invalid", order=4, show=False
    )
    plt.close("all")


@patch("matplotlib.font_manager.findfont")
def test_apply_theme_fallback(mock_find):
    """Trigger the font fallback in apply_default_theme."""
    mock_find.side_effect = ValueError("Font not found")
    apply_default_theme()


def test_subplot_grid(backend_device, xp):
    """Verify subplot grid calculation."""
    assert _create_subplot_grid(1) == (1, 1)
    assert _create_subplot_grid(2) == (1, 2)
    assert _create_subplot_grid(3) == (2, 2)
    assert _create_subplot_grid(5) == (3, 2)


def test_psd_axis_overlay_warning(caplog, backend_device, xp):
    """Verify warning when single axis is provided for multichannel PSD."""
    sig = xp.ones((2, 256))
    fig, ax = plt.subplots()
    with patch("matplotlib.pyplot.show"):
        psd(sig, ax=ax)
    assert "Multiple channels detected but single axis provided" in caplog.text


def test_psd_wavelength_warning(caplog, backend_device, xp):
    """Verify wavelength warning."""
    sig = xp.ones(256)
    with patch("matplotlib.pyplot.show"):
        psd(sig, x_axis="wavelength", domain="RF")
    assert "Wavelength plotting is typically used for optical signals" in caplog.text


def test_psd_auto_scale_small(backend_device, xp):
    """Test auto-scaling for small frequencies (Hz)."""
    sig = xp.ones(256)
    fig, ax = psd(sig, sampling_rate=1.0)
    assert "Hz" in ax.get_xlabel()


def test_time_domain_limits(caplog, backend_device, xp):
    """Verify symbol limit warnings in time_domain."""
    sig = xp.ones(100)
    # request 200 symbols when only 100 samples available (sps=1)
    with patch("matplotlib.pyplot.show"):
        time_domain(sig, num_symbols=200, sps=1.0)
    assert "Limit exceeds number of symbols" in caplog.text


def test_time_domain_auto_scale(backend_device, xp):
    """Verify time axis auto-scaling."""
    sig = xp.ones(100)

    # ns
    _, ax = time_domain(sig, sampling_rate=1e10)
    assert "ns" in ax.get_xlabel()

    # ps
    _, ax = time_domain(sig, sampling_rate=1e13)
    assert "ps" in ax.get_xlabel()

    # us
    _, ax = time_domain(sig, sampling_rate=1e7)
    assert "µs" in ax.get_xlabel()

    # ms
    _, ax = time_domain(sig, sampling_rate=1e4)
    assert "ms" in ax.get_xlabel()


def test_eye_diagram_sps_error(backend_device, xp):
    """Verify error for non-integer sps."""
    with pytest.raises(ValueError, match="sps must be an integer"):
        eye_diagram(xp.ones(10), sps=2.5)


def test_eye_diagram_trace_len_error(backend_device, xp):
    """Verify error when signal too short for eye trace."""
    with pytest.raises(
        ValueError, match="Signal is shorter than the required trace length"
    ):
        _plot_eye_traces(
            xp.ones(5), sps=10, num_symbols=2, ax=MagicMock(), type="line", title=None
        )


def test_eye_diagram_invalid_type(backend_device, xp):
    """Verify error for unknown eye type."""
    with pytest.raises(ValueError, match="Unknown type"):
        _plot_eye_traces(
            xp.ones(100), sps=4, num_symbols=2, ax=MagicMock(), type="magic", title=None
        )


def test_filter_response_no_sps(backend_device, xp):
    """Test filter_response without sps."""
    filter_response(xp.ones(10))


def test_constellation_histogram_overlay_warning(caplog, backend_device, xp):
    """Verify warning when overlaying ideal on histogram."""
    import logging

    caplog.set_level(logging.WARNING)
    # Constellation with bins > 0 triggers histogram
    # If modulation is None, it should warn
    constellation(xp.ones(10) + 1j, bins=10, overlay_ideal=True, modulation=None)
    assert "Modulation and order must be provided" in caplog.text
