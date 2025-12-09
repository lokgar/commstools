from typing import Optional

import numpy as np

from . import filtering, mapping, sequences, utils
from .backend import get_sp, get_xp
from .signal import Signal


def generate_waveform(
    modulation: str,
    order: int,
    num_symbols: int,
    sps: float,
    symbol_rate: float,
    pulse_shape: str = "rrc",
    rrc_rolloff: float = 0.35,
    smoothrect_bt: float = 1.0,
    gaussian_bt: float = 0.3,
    filter_span: int = 10,
    seed: Optional[int] = None,
) -> Signal:
    """
    Generate a complete waveform with specified parameters.

    Args:
        modulation: Modulation type ('psk', 'qam', 'ask').
        order: Modulation order.
        num_symbols: Number of symbols to generate.
        sps: Samples per symbol.
        symbol_rate: Symbol rate in Hz.
        pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc').
        rrc_rolloff: Roll-off factor for RRC/RC filters.
        gaussian_bt: Bandwidth-Time product for Gaussian filter.
        filter_span: Filter span in symbols.
        seed: Random seed.

    Returns:
        Signal object containing the generated waveform.
    """
    # calculate number of bits
    k = int(np.log2(order))
    num_bits = num_symbols * k

    # Generate random bits
    bits = sequences.random_bits(num_bits, seed=seed)

    # Map to symbols
    symbols = mapping.map_bits(bits, modulation=modulation, order=order)

    # Apply pulse shaping
    samples = filtering.shape_pulse(
        symbols,
        sps=sps,
        pulse_shape=pulse_shape,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        smoothrect_bt=smoothrect_bt,
        gaussian_bt=gaussian_bt,
    )

    # Create Signal object
    # Sampling rate is symbol_rate * sps
    return Signal(
        samples=samples,
        sampling_rate=symbol_rate * sps,
        symbol_rate=symbol_rate,
        modulation_format=f"{order}-{modulation.upper()}",
    )


def pam_waveform(
    order: int,
    bipolar: bool,
    num_symbols: int,
    sps: int,
    symbol_rate: float,
    pulse_shape: str = "rect",
    rrc_rolloff: float = 0.35,
    smoothrect_bt: float = 1.0,
    gaussian_bt: float = 0.3,
    filter_span: int = 10,
    seed: Optional[int] = None,
) -> Signal:
    """
    Generate a PAM waveform (NRZ).

    Args:
        order: Modulation order (2, 4, 8, etc.).
        bipolar: Whether to use bipolar (True) or unipolar (False) PAM.
        num_symbols: Number of symbols to generate.
        sps: Samples per symbol (integer).
        symbol_rate: Symbol rate in Hz.
        pulse_shape: Pulse shaping type ('rect', 'rrc', 'rc', 'gaussian', 'smoothrect', 'none').
        rrc_rolloff: Roll-off factor for RRC/RC filters.
        smoothrect_bt: BT product for smoothrect filter.
        gaussian_bt: BT product for Gaussian filter.
        filter_span: Filter span in symbols.
        seed: Random seed.

    Returns:
        Signal object.
    """
    sig = generate_waveform(
        modulation="ask-bipol" if bipolar else "ask-unipol",
        order=order,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        pulse_shape=pulse_shape,
        rrc_rolloff=rrc_rolloff,
        smoothrect_bt=smoothrect_bt,
        gaussian_bt=gaussian_bt,
        filter_span=filter_span,
        seed=seed,
    )
    sig.modulation_format = f"PAM-{order}{'-bipol' if bipolar else '-unipol'}"
    return sig


def rzpam_waveform(
    order: int,
    bipolar: bool,
    num_symbols: int,
    sps: int,
    symbol_rate: float,
    pulse_shape: str = "rect",
    smoothrect_bt: float = 1.0,
    filter_span: int = 10,
    seed: Optional[int] = None,
) -> Signal:
    """
    Generate a RZ (Return-to-Zero) PAM waveform.

    Args:
        order: Modulation order (2, 4, 8, etc.).
        bipolar: Whether to use bipolar (True) or unipolar (False) PAM.
        num_symbols: Number of symbols to generate.
        sps: Samples per symbol (integer).
        symbol_rate: Symbol rate in Hz.
        pulse_shape: 'rect' (simplest) or 'smoothrect'. Others are prohibited.
        smoothrect_bt: BT product for smoothrect filter.
        filter_span: Filter span in symbols.
        seed: Random seed.

    Returns:
        Signal object.
    """
    # Ensure even sps
    if sps % 2 != 0:
        raise ValueError("For correct RZ duty cycle, `sps` must be even")

    # Prohibit complex pulses for RZ as requested/planned
    allowed_rz_pulses = ["rect", "smoothrect"]
    if pulse_shape not in allowed_rz_pulses:
        raise ValueError(
            f"Pulse shape '{pulse_shape}' is not allowed for RZ PAM. "
            f"Allowed: {allowed_rz_pulses}"
        )

    xp = get_xp()

    # Generate symbols
    k = int(xp.log2(order))
    if 2**k != order:
        raise ValueError(f"PAM order must be power of 2, got {order}")
    num_bits = num_symbols * k
    bits = sequences.random_bits(num_bits, seed=seed)
    symbols = mapping.map_bits(
        bits, modulation="ask-bipol" if bipolar else "ask-unipol", order=order
    )

    # Apply RZ Pulse Shaping
    if pulse_shape == "rect":
        sp = get_sp()
        h = xp.ones(int(sps / 2))
        samples = utils.normalize(
            sp.signal.resample_poly(symbols, int(sps), 1, window=h), "max_amplitude"
        )

    elif pulse_shape == "smoothrect":
        # RZ Smooth Rect: 50% width (0.5 symbol duration)
        h = filtering.smoothrect_taps(
            sps=sps, span=filter_span, bt=smoothrect_bt, pulse_width=0.5
        )

        # Apply pulse using polyphase resampling
        sp = get_sp()
        samples = utils.normalize(
            sp.signal.resample_poly(symbols, int(sps), 1, window=h), "max_amplitude"
        )

    # Return Signal with format RZ-PAM-M
    return Signal(
        samples=samples,
        sampling_rate=symbol_rate * sps,
        symbol_rate=symbol_rate,
        modulation_format=f"RZ-PAM-{order}{'-bipol' if bipolar else '-unipol'}",
    )


def psk_waveform(
    order: int,
    num_symbols: int,
    sps: float,
    symbol_rate: float,
    pulse_shape: str = "rrc",
    rrc_rolloff: float = 0.35,
    gaussian_bt: float = 0.3,
    filter_span: int = 10,
    seed: Optional[int] = None,
) -> Signal:
    """Wrapper for PSK waveform generation."""
    return generate_waveform(
        modulation="psk",
        order=order,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        pulse_shape=pulse_shape,
        rrc_rolloff=rrc_rolloff,
        gaussian_bt=gaussian_bt,
        filter_span=filter_span,
        seed=seed,
    )


def qam_waveform(
    order: int,
    num_symbols: int,
    sps: float,
    symbol_rate: float,
    pulse_shape: str = "rrc",
    rrc_rolloff: float = 0.35,
    gaussian_bt: float = 0.3,
    filter_span: int = 10,
    seed: Optional[int] = None,
) -> Signal:
    """Wrapper for QAM waveform generation."""
    return generate_waveform(
        modulation="qam",
        order=order,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        pulse_shape=pulse_shape,
        rrc_rolloff=rrc_rolloff,
        gaussian_bt=gaussian_bt,
        filter_span=filter_span,
        seed=seed,
    )
