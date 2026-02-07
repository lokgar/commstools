from commstools.core import Signal


def test_dual_pol_initialization(backend_device, xp):
    # Create (N_samples, N_channels) data
    # Use xp (numpy or cupy)
    samples = xp.random.randn(2, 100) + 1j * xp.random.randn(2, 100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    # Time-last convention check
    assert sig.samples.shape == (2, 100)
    assert sig.num_streams == 2


def test_dual_pol_upsample(backend_device, xp):
    samples = xp.random.randn(2, 100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    sig.upsample(2)
    assert sig.samples.shape == (2, 200)
    assert sig.sampling_rate == 2.0


def test_dual_pol_decimate(backend_device, xp):
    samples = xp.random.randn(2, 200)
    sig = Signal(samples=samples, sampling_rate=2.0, symbol_rate=1.0)

    # Decimate by 2
    sig.decimate(2)
    assert sig.samples.shape == (2, 100)
    assert sig.sampling_rate == 1.0


def test_dual_pol_resample(backend_device, xp):
    samples = xp.random.randn(2, 100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    # Resample by 1.5 (up 3, down 2)
    sig.resample(up=3, down=2)
    assert sig.samples.shape == (2, 150)
    assert sig.sampling_rate == 1.5


def test_dual_pol_frequency_shift(backend_device, xp):
    # Create a DC signal on both channels
    samples = xp.ones((2, 100), dtype=complex)
    sig = Signal(samples=samples, sampling_rate=100.0, symbol_rate=100.0)

    # Shift by 25 Hz (1/4 of Fs) -> should become exp(j*pi/2 * n) = j, -1, -j, 1...
    sig.shift_frequency(25.0)

    # Check phase rotation on first few samples of first channel

    # Check phase rotation on first few samples of first channel
    # n=0 -> 1
    # n=1 -> 1 * exp(j * 2pi * 25/100 * 1) = exp(j * pi/2) = j
    expected_sample_1 = xp.exp(1j * xp.pi / 2)

    # Use xp.allclose for comparison
    assert xp.allclose(sig.samples[0, 0], 1.0, atol=1e-6)
    assert xp.allclose(sig.samples[0, 1], expected_sample_1, atol=1e-6)
    # Check second channel n=1 too (since input was ones)
    assert xp.allclose(sig.samples[1, 1], expected_sample_1, atol=1e-6)


def test_dual_pol_fir_filter(backend_device, xp):
    # Impulse on both channels at index 0
    samples = xp.zeros((2, 10))
    samples[:, 0] = 1.0
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    # Simple moving average filter: [0.5, 0.5]
    taps = xp.array([0.5, 0.5])

    sig.fir_filter(taps)

    # Check non-zero content
    assert xp.any(sig.samples != 0)
    # Check shape preserved
    assert sig.samples.shape == (2, 10)


def test_siso_fallback(backend_device, xp):
    # Ensure 1D signal still works creates 1 polarization
    samples = xp.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)
    assert sig.num_streams == 1
    assert sig.samples.ndim == 1
