from typing import Optional, Literal
from ..core.signal import Signal
from ..core.config import get_config
from ..core.backend import get_backend, ArrayType
from . import mapping, filters


def ook(
    data_bits: ArrayType,
    samples_per_symbol: Optional[int] = None,
    pulse_type: Literal["rect", "impulse"] = "rect",
    center_freq: Optional[float] = None,
    sampling_rate: Optional[float] = None,
) -> Signal:
    """
    Generates an On-Off Keying (OOK) signal.

    Args:
        data_bits: Input binary data.
        samples_per_symbol: Number of samples per symbol.
        pulse_type: Type of pulse shaping ('rect' or 'impulse').
        center_freq: Center frequency in Hz.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Signal object containing the OOK waveform.
    """
    config = get_config()

    # Resolve parameters
    if samples_per_symbol is None:
        if config and config.samples_per_symbol:
            samples_per_symbol = config.samples_per_symbol
        else:
            samples_per_symbol = 1  # Default to 1 if not specified

    if sampling_rate is None:
        if config:
            sampling_rate = config.sampling_rate
        else:
            raise ValueError("sampling_rate must be provided or set in global config")

    if center_freq is None:
        if config:
            center_freq = config.center_freq
        else:
            center_freq = 0.0

    # 1. Map bits to symbols
    symbols = mapping.ook_map(data_bits)

    # 2. Apply pulse shaping
    if pulse_type == "rect":
        samples = filters.apply_pulse_shape(symbols, samples_per_symbol, kind="rect")
    elif pulse_type == "impulse":
        # Impulse is just upsampling with zeros, which is effectively custom shaping with a unit impulse filter
        # Or we can expose 'upsample' directly?
        # The user requirement was: "impulse type should be just upsampler function with zero"
        # In filters.py we have 'upsample'.
        samples = filters.upsample(symbols, samples_per_symbol)
    else:
        raise ValueError(f"Unknown pulse_type: {pulse_type}")

    # 3. Create Signal
    return Signal(
        samples=samples,
        sampling_rate=sampling_rate,
        center_freq=center_freq,
        modulation_format="OOK",
    )
