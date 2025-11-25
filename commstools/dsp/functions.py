"""DSP utility functions for signal processing and sequence generation.

These functions demonstrate how to use the global SystemConfig for
functions that don't necessarily take a Signal as input.
"""

import numpy as np
from typing import Optional
from ..core.signal import Signal
from ..core.backend import get_backend
from ..core.config import get_config


def generate_pilot_sequence(
    length: Optional[int] = None,
    modulation: Optional[str] = None,
) -> np.ndarray:
    """Generate a pilot sequence for channel estimation and synchronization.

    This function demonstrates using global config for parameters when
    no Signal object is available as input.

    Args:
        length: Length of the pilot sequence. If None, uses config.sequence_length
        modulation: Modulation format. If None, uses config.modulation_format

    Returns:
        Complex pilot sequence as numpy array

    Example:
        >>> config = SystemConfig(
        ...     sampling_rate=1e6,
        ...     sequence_length=128,
        ...     modulation_format='QPSK'
        ... )
        >>> set_config(config)
        >>> pilots = generate_pilot_sequence()  # Uses config values
        >>> # or override:
        >>> pilots = generate_pilot_sequence(length=256, modulation='16QAM')
    """
    config = get_config()

    # Use provided parameters or fall back to config
    if length is None:
        if config is not None:
            length = config.sequence_length
        else:
            length = 128  # default

    if modulation is None:
        if config is not None:
            modulation = config.modulation_format
        else:
            modulation = "QPSK"  # default

    # Generate sequence based on modulation
    if modulation.upper() == "QPSK":
        # Simple QPSK pilot: random ±1±j
        real_part = np.random.choice([-1, 1], size=length)
        imag_part = np.random.choice([-1, 1], size=length)
        pilots = (real_part + 1j * imag_part) / np.sqrt(2)
    elif modulation.upper() in ["16QAM", "64QAM"]:
        # For QAM, use QPSK pilots (more robust)
        real_part = np.random.choice([-1, 1], size=length)
        imag_part = np.random.choice([-1, 1], size=length)
        pilots = (real_part + 1j * imag_part) / np.sqrt(2)
    else:
        # Default to BPSK
        pilots = np.random.choice([-1, 1], size=length).astype(complex)

    return pilots


def generate_training_signal(
    samples_per_symbol: Optional[int] = None,
    length: Optional[int] = None,
) -> Signal:
    """Generate a training signal for equalizer adaptation.

    This creates a Signal object using global config parameters.

    Args:
        samples_per_symbol: Samples per symbol. If None, uses config value
        length: Number of symbols. If None, uses config.sequence_length

    Returns:
        Training signal as a Signal object

    Example:
        >>> config = SystemConfig(
        ...     sampling_rate=1e6,
        ...     samples_per_symbol=4,
        ...     sequence_length=128
        ... )
        >>> set_config(config)
        >>> training_sig = generate_training_signal()
    """
    config = get_config()
    backend = get_backend()

    # Get parameters from config or use defaults
    if samples_per_symbol is None:
        if config is not None and config.samples_per_symbol is not None:
            samples_per_symbol = config.samples_per_symbol
        else:
            samples_per_symbol = 4  # default

    if length is None:
        if config is not None:
            length = config.sequence_length
        else:
            length = 128  # default

    # Generate symbol sequence
    modulation = config.modulation_format if config else "QPSK"
    symbols = generate_pilot_sequence(length=length, modulation=modulation)

    # Upsample to create training signal
    upsampled = np.zeros(length * samples_per_symbol, dtype=complex)
    upsampled[::samples_per_symbol] = symbols

    # Convert to backend array
    samples = backend.array(upsampled)

    # Create Signal - will use config if available via use_config
    if config is not None:
        return Signal(samples=samples, use_config=True)
    else:
        # Fallback if no config
        return Signal(samples=samples, sampling_rate=1e6)


def matched_filter(
    signal: Signal,
    roll_off: Optional[float] = None,
    filter_length: Optional[int] = None,
) -> Signal:
    """Apply a root raised cosine matched filter.

    Demonstrates a DSP function that takes a Signal and can use config
    for optional parameters.

    Args:
        signal: Input signal to filter
        roll_off: Roll-off factor (0 to 1). If None, uses config.filter_roll_off
        filter_length: Filter length in symbols. If None, uses config.equalizer_taps

    Returns:
        Filtered signal

    Example:
        >>> sig = Signal(samples=data, ...)
        >>> filtered = matched_filter(sig)  # Uses config.filter_roll_off
    """
    config = get_config()

    # Ensure signal is on the global backend
    signal = signal.ensure_backend()

    # Get parameters: explicit > config > defaults
    if roll_off is None:
        if config is not None:
            roll_off = config.filter_roll_off
        else:
            roll_off = 0.35

    if filter_length is None:
        if config is not None:
            filter_length = config.equalizer_taps
        else:
            filter_length = 15

    # For simplicity, use a basic low-pass filter as placeholder
    # In production, this would be a proper RRC filter
    # This is just to demonstrate the config usage pattern

    # Simple moving average as placeholder
    # kernel = backend.ones(filter_length) / filter_length

    # Convolve (simplified - would use scipy.signal.convolve in production)
    # For now, just return signal with metadata preserved
    filtered_samples = signal.samples  # Placeholder

    return signal.update(samples=filtered_samples)


def add_awgn(
    signal: Signal,
    snr_db: Optional[float] = None,
    noise_power: Optional[float] = None,
) -> Signal:
    """Add Additive White Gaussian Noise to a signal.

    Args:
        signal: Input signal
        snr_db: Target SNR in dB. If None, uses config.snr_db
        noise_power: Noise power. If provided, overrides snr_db

    Returns:
        Noisy signal

    Example:
        >>> config = SystemConfig(sampling_rate=1e6, snr_db=20)
        >>> set_config(config)
        >>> sig = Signal(samples=clean_data, ...)
        >>> noisy = add_awgn(sig)  # Uses config.snr_db = 20
    """
    config = get_config()

    # Ensure signal is on the global backend
    signal = signal.ensure_backend()
    backend = signal.backend

    # Calculate noise power
    if noise_power is None:
        if snr_db is None:
            if config is not None and config.snr_db is not None:
                snr_db = config.snr_db
            else:
                snr_db = 20  # default

        # Calculate signal power
        signal_power = backend.mean(backend.abs(signal.samples) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

    # Generate complex Gaussian noise
    noise_real = np.random.randn(*signal.samples.shape) * np.sqrt(noise_power / 2)
    noise_imag = np.random.randn(*signal.samples.shape) * np.sqrt(noise_power / 2)
    noise = backend.array(noise_real + 1j * noise_imag)

    return signal.update(samples=signal.samples + noise)
