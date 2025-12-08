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
    if snr_linear <= 0:
        # If SNR is very low or negative, noise power will be very high
        noise_power = signal_power / 1e-10  # Effectively infinite noise
    else:
        noise_power = signal_power / snr_linear

    noise_std = xp.sqrt(noise_power)

    # Handle complex signals
    is_complex = xp.iscomplexobj(samples)

    if is_complex:
        # For complex noise, power is split between real and imag
        noise_std_component = xp.sqrt(noise_power / 2)

        # PROVISIONAL: Generate noise on host using NumPy and move to backend
        from .backend import to_host

        noise_std_host = float(to_host(noise_std_component))

        shape = samples.shape
        noise = np.random.normal(0, noise_std_host, shape) + 1j * np.random.normal(
            0, noise_std_host, shape
        )
    else:
        # Real noise
        from .backend import to_host

        noise_std_host = float(to_host(noise_std))
        shape = samples.shape
        noise = np.random.normal(0, noise_std_host, shape)

    # Move noise to backend
    noise = xp.asarray(noise, dtype=samples.dtype)

    noisy_samples = samples + noise

    if isinstance(signal, Signal):
        # Return new Signal preserving metadata
        return Signal(
            samples=noisy_samples,
            sampling_rate=signal.sampling_rate,
            symbol_rate=signal.symbol_rate,
            modulation_format=signal.modulation_format,
        )
    else:
        return noisy_samples
