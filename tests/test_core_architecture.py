import pytest
import numpy as np
from commstools import Signal, ProcessingBlock, set_backend, get_backend, using_backend

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

class GainBlock(ProcessingBlock):
    def __init__(self, gain: float):
        self.gain = gain
        
    def process(self, signal: Signal) -> Signal:
        backend = get_backend()
        # Ensure signal is on current backend or handle mixed types
        # For this test, we'll assume the signal is already on the correct backend
        # or that the backend operations handle it.
        
        # Ideally, we should use signal.to(backend.name) if we want to enforce it,
        # but for a simple gain block, element-wise multiplication should work 
        # if the types are compatible.
        
        new_samples = signal.samples * self.gain
        return Signal(
            samples=new_samples,
            sample_rate=signal.sample_rate,
            center_freq=signal.center_freq,
            modulation_format=signal.modulation_format
        )

def test_signal_creation_numpy():
    set_backend("numpy")
    samples = np.array([1.0, 2.0, 3.0])
    sig = Signal(samples=samples, sample_rate=100.0)
    
    assert isinstance(sig.samples, np.ndarray)
    assert sig.sample_rate == 100.0
    assert sig.duration == 0.03
    
    time_axis = sig.time_axis()
    assert isinstance(time_axis, np.ndarray)
    assert np.allclose(time_axis, [0.0, 0.01, 0.02])

@pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not available")
def test_signal_creation_jax():
    set_backend("jax")
    samples = jnp.array([1.0, 2.0, 3.0])
    sig = Signal(samples=samples, sample_rate=100.0)
    
    assert isinstance(sig.samples, type(jnp.array([]))) or hasattr(sig.samples, 'device_buffer')
    assert sig.sample_rate == 100.0
    
    time_axis = sig.time_axis()
    # Check if it's a JAX array
    assert hasattr(time_axis, 'device_buffer') or isinstance(time_axis, type(jnp.array([])))

def test_backend_switching():
    set_backend("numpy")
    samples = np.array([1.0+1j, 2.0+2j])
    sig = Signal(samples=samples, sample_rate=10.0)
    
    assert isinstance(sig.samples, np.ndarray)
    
    if _JAX_AVAILABLE:
        # Convert to JAX
        sig_jax = sig.to("jax")
        assert hasattr(sig_jax.samples, 'device_buffer') or isinstance(sig_jax.samples, type(jnp.array([])))
        
        # Convert back to Numpy
        sig_numpy = sig_jax.to("numpy")
        assert isinstance(sig_numpy.samples, np.ndarray)
        assert np.allclose(sig_numpy.samples, samples)

def test_processing_block():
    set_backend("numpy")
    samples = np.array([1.0, 2.0])
    sig = Signal(samples=samples, sample_rate=10.0)
    
    gain_block = GainBlock(gain=2.0)
    processed_sig = gain_block(sig)
    
    assert np.allclose(processed_sig.samples, [2.0, 4.0])
    
    if _JAX_AVAILABLE:
        with using_backend("jax"):
            sig_jax = sig.to("jax")
            processed_sig_jax = gain_block(sig_jax)
            assert hasattr(processed_sig_jax.samples, 'device_buffer') or isinstance(processed_sig_jax.samples, type(jnp.array([])))
            # We need to convert back to numpy to check values easily or use jnp.allclose
            assert jnp.allclose(processed_sig_jax.samples, jnp.array([2.0, 4.0]))

def test_spectrum():
    set_backend("numpy")
    # Simple sine wave
    fs = 100.0
    t = np.arange(100) / fs
    f0 = 10.0
    samples = np.exp(1j * 2 * np.pi * f0 * t)
    sig = Signal(samples=samples, sample_rate=fs)
    
    freqs, psd = sig.spectrum()
    
    assert isinstance(freqs, np.ndarray)
    assert isinstance(psd, np.ndarray)
    
    # Peak should be at 10 Hz
    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]
    assert np.isclose(peak_freq, f0, atol=1.0)

