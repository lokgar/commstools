from typing import Optional, Any
from ..core.signal import Signal
from ..core.config import get_config
from ..core.backend import ArrayType
from . import mapping, filters


def ook(
    data_bits: ArrayType,
    sampling_rate: Optional[float] = None,
    samples_per_symbol: Optional[int] = None,
    pulse_shape: str = "delta",
    **kwargs: Any,
) -> Signal:
    """
    Generates an On-Off Keying (OOK) signal.

    Args:
        data_bits: Input binary data.
        sampling_rate: Sampling rate in Hz.
        samples_per_symbol: Number of samples per symbol.
        pulse_shape: Type of pulse shaping ('delta', 'rect', 'gaussian', 'rrc').
        **kwargs: Additional arguments for specific filters (e.g., alpha, bt, span).

    Returns:
        Signal object containing the OOK waveform.
    """
    config = get_config()

    # 0. Resolve parameters
    if sampling_rate is None:
        if config:
            sampling_rate = config.sampling_rate
        else:
            raise ValueError("sampling_rate must be provided or set in global config")

    if samples_per_symbol is None:
        if config and config.samples_per_symbol:
            samples_per_symbol = config.samples_per_symbol
        else:
            samples_per_symbol = 1  # Default to 1 if not specified

    # 1. Map bits to symbols
    symbols = mapping.ook_map(data_bits)

    # 2. Apply pulse shaping
    # Unified approach: Get taps -> Shape pulse
    taps = filters.get_taps(pulse_shape, samples_per_symbol, **kwargs)
    samples = filters.shape_pulse(symbols, taps, samples_per_symbol)

    # 3. Create Signal
    return Signal(
        samples=samples,
        sampling_rate=sampling_rate,
        modulation_format="OOK",
    )
