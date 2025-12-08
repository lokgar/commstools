import pytest
import numpy as np
import matplotlib.pyplot as plt
from commstools.plotting import eye_diagram, time_domain, psd, filter_response

# Plotting tests usually just check that no exception is raised and figures are created.
# We don't check visual correctness here.


@pytest.mark.parametrize("type", ["line", "hist"])
def test_eye_diagram_real(backend_device, xp, type):
    # Random binary data
    samples = xp.random.randn(1000)
    sps = 4

    # Move to backend ?? plotting handles backend internally via ensure_on_backend usually,
    # or expects inputs.
    # plotting.py: samples = ensure_on_backend(samples)

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
    samples = xp.arange(100)
    fig, ax = time_domain(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    plt.close(fig)


def test_psd(backend_device, xp):
    samples = xp.random.randn(128)
    fig, ax = psd(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    plt.close(fig)


def test_filter_response(backend_device, xp):
    taps = xp.array([1, 0.5, 0.25])
    fig, axes = filter_response(taps, sps=1.0, show=False)
    assert fig is not None
    assert len(axes) == 3
    plt.close(fig)
