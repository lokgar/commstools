from typing import Union, TYPE_CHECKING
import numpy as np

from .backend import ArrayType

if TYPE_CHECKING:
    from .signal import Signal


def add_gaussian_noise(
    signal: Union[ArrayType, "Signal"], snr_db: float
) -> Union[ArrayType, "Signal"]:
    """
    Adds Gaussian noise to a signal to achieve a specified SNR.

    Args:
        signal: The input signal (array or Signal object).
        snr_db: The desired Signal-to-Noise Ratio in decibels.

    Returns:
        The signal with added Gaussian noise. If input is Signal, returns a new Signal object.
    """
    # Check if signal is a Signal object (avoiding circular import issues by checking attribute or type name if strictly needed,
    # but TYPE_CHECKING import + runtime check for explicit class is problematic if not imported.
    # We can check for 'samples' attribute or import inside function).
    from .signal import Signal

    from .backend import ensure_on_backend, get_xp

    if isinstance(signal, Signal):
        samples = signal.samples
    else:
        samples = signal

    samples = ensure_on_backend(samples)
    xp = get_xp()

    signal_power = xp.mean(xp.abs(samples) ** 2)
    snr_linear = 10 ** (snr_db / 10)

    # Handle very low SNR or infinite noise case
    # We avoid division by zero or negative SNRs if possible, though physics suggests positive linear SNR.
    # But if snr_linear is effectively 0, noise power is infinite.
    if snr_linear <= 1e-20:
        # Fallback to avoid div by zero, make noise massive relative to signal
        # or just use a safe epsilon.
        noise_power = signal_power / 1e-20
    else:
        noise_power = signal_power / snr_linear

    noise_std = xp.sqrt(noise_power)

    # Handle complex signals
    is_complex = xp.iscomplexobj(samples)

    if is_complex:
        # For complex noise, power is split between real and imag
        noise_std_component = xp.sqrt(noise_power / 2)

        # Generate noise on backend
        noise = xp.random.normal(
            0, noise_std_component, samples.shape
        ) + 1j * xp.random.normal(0, noise_std_component, samples.shape)
    else:
        # Real noise
        noise = xp.random.normal(0, noise_std, samples.shape)

    # noise is already on backend
    noisy_samples = samples + noise

    if isinstance(signal, Signal):
        sig = signal.copy()
        sig.samples = noisy_samples
        return sig
    else:
        return noisy_samples
