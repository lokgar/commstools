from typing import Optional, Any
from .core.signal import Signal

from .core.backend import ArrayType
from .dsp import mapping, filters


def ook(
    data_bits: ArrayType,
    sampling_rate: Optional[float] = None,
    sps: Optional[float] = None,
    pulse_shape: str = "none",
    **kwargs: Any,
) -> Signal:
    """
    Generates an On-Off Keying (OOK) signal.

    Args:
        data_bits: Input binary data.
        sampling_rate: Sampling rate in Hz.
        sps: Samples per symbol.
        pulse_shape: Type of pulse shaping ('none', 'boxcar', 'gaussian', 'rrc', 'sinc').
        **kwargs: Additional arguments for specific filters (e.g., rolloff, bt, span).

    Returns:
        Signal object containing the OOK waveform.
    """
    # 0. Resolve parameters
    if sampling_rate is None:
        raise ValueError("sampling_rate must be provided")

    if sps is None:
        sps = 1  # Default to 1 if not specified

    # 1. Map bits to symbols
    symbols = mapping.ook_map(data_bits)

    span = kwargs.pop("span", 10)

    # 2. Apply pulse shaping
    samples = filters.shape_pulse(
        symbols=symbols, sps=sps, span=span, pulse_shape=pulse_shape, **kwargs
    )

    # 3. Create Signal
    return Signal(
        samples=samples,
        sampling_rate=sampling_rate,
        symbol_rate=sampling_rate / sps,
        modulation_format="OOK",
    )
