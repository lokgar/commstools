import pytest
import numpy as np
from commstools import Signal, set_backend

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def gain_func(signal: Signal, gain: float) -> Signal:
    return signal.update(signal.samples * gain)


def test_signal_creation_numpy():
    set_backend("numpy")
    samples = np.array([1.0, 2.0, 3.0])
    sig = Signal(samples=samples, sampling_rate=100.0, symbol_rate=1)

    assert isinstance(sig.samples, np.ndarray)
    assert sig.sampling_rate == 100.0
    assert sig.duration == 0.03

    time_axis = sig.time_axis()
    assert isinstance(time_axis, np.ndarray)
    assert np.allclose(time_axis, [0.0, 0.01, 0.02])


@pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not available")
def test_signal_creation_jax():
    set_backend("jax")
    samples = jnp.array([1.0, 2.0, 3.0])
    sig = Signal(samples=samples, sampling_rate=100.0, symbol_rate=1)

    assert isinstance(sig.samples, type(jnp.array([]))) or hasattr(
        sig.samples, "device_buffer"
    )
    assert sig.sampling_rate == 100.0

    time_axis = sig.time_axis()
    # Check if it's a JAX array
    assert hasattr(time_axis, "device_buffer") or isinstance(
        time_axis, type(jnp.array([]))
    )


def test_backend_switching():
    set_backend("numpy")
    samples = np.array([1.0 + 1j, 2.0 + 2j])
    sig = Signal(samples=samples, sampling_rate=10.0, symbol_rate=1)

    assert isinstance(sig.samples, np.ndarray)

    if _JAX_AVAILABLE:
        # Convert to JAX
        sig_jax = sig.to("jax")
        assert hasattr(sig_jax.samples, "device_buffer") or isinstance(
            sig_jax.samples, type(jnp.array([]))
        )

        # Convert back to Numpy
        sig_numpy = sig_jax.to("numpy")
        assert isinstance(sig_numpy.samples, np.ndarray)
        assert np.allclose(sig_numpy.samples, samples)


def test_functional_processing():
    set_backend("numpy")
    samples = np.array([1.0, 2.0])
    sig = Signal(samples=samples, sampling_rate=10.0, symbol_rate=1)

    processed_sig = gain_func(sig, gain=2.0)

    assert np.allclose(processed_sig.samples, [2.0, 4.0])
    assert processed_sig.sampling_rate == 10.0

    if _JAX_AVAILABLE:
        # Switch global backend to JAX
        set_backend("jax")

        # Ensure signal is on JAX (auto-alignment or explicit)
        sig_jax = sig.to("jax")

        processed_sig_jax = gain_func(sig_jax, gain=2.0)
        assert hasattr(processed_sig_jax.samples, "device_buffer") or isinstance(
            processed_sig_jax.samples, type(jnp.array([]))
        )
        # We need to convert back to numpy to check values easily or use jnp.allclose
        assert jnp.allclose(processed_sig_jax.samples, jnp.array([2.0, 4.0]))

        # Reset backend
        set_backend("numpy")
