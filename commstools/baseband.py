"""
Baseband waveform generation.

This module provides functions to generate baseband digital communication signals,
including support for:
- Pulse Amplitude Modulation (PAM), both unipolar and bipolar.
- Phase Shift Keying (PSK).
- Quadrature Amplitude Modulation (QAM).
- Return-to-Zero (RZ) signaling.
- Various pulse shaping filters (Rectangular, RRC, RC, Gaussian, SmoothRect).
"""

from typing import Optional

import numpy as np
import scipy.signal

from . import filtering, mapping, sequences, utils
from .core import Signal
from .logger import logger


def generate_baseband(
    modulation: str,
    order: int,
    num_symbols: int,
    sps: float,
    symbol_rate: float,
    pulse_shape: str = "rrc",
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
    smoothrect_bt: float = 1.0,
    gaussian_bt: float = 0.3,
    seed: Optional[int] = None,
) -> Signal:
    """
    Generate a baseband waveform with specified parameters.

    Args:
        modulation: Modulation scheme ('psk', 'qam', 'ask').
        order: Modulation order.
        num_symbols: Number of symbols to generate.
        sps: Samples per symbol.
        symbol_rate: Symbol rate in Hz.
        pulse_shape: Pulse shape ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc').
        filter_span: Filter span in symbols.
        rrc_rolloff: Roll-off factor for RRC shaping filter.
        rc_rolloff: Roll-off factor for RC shaping filter.
        smoothrect_bt: Bandwidth-Time product for SmoothRect shaping filter.
        gaussian_bt: Bandwidth-Time product for Gaussian shaping filter.
        seed: Random seed.

    Returns:
        A `Signal` instance containing the generated waveform, including source bits and metadata.
    """
    logger.info(
        f"Generating {modulation.upper()} baseband: order={order}, "
        f"symbols={num_symbols}, sps={sps:.2f}, symbol_rate={utils.format_si(symbol_rate, 'Baud')}."
    )
    # calculate number of bits
    k = int(np.log2(order))
    num_bits = num_symbols * k

    # Generate random bits (NumPy)
    bits = sequences.random_bits(num_bits, seed=seed)

    # Map to symbols (NumPy)
    symbols = mapping.map_bits(bits, modulation=modulation, order=order)

    # Apply pulse shaping (handles dispatch, returns NumPy here as input is NumPy)
    samples = filtering.shape_pulse(
        symbols,
        sps=sps,
        pulse_shape=pulse_shape,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        rc_rolloff=rc_rolloff,
        smoothrect_bt=smoothrect_bt,
        gaussian_bt=gaussian_bt,
    )

    # Create Signal object
    # Default backend logic in Signal will keep it on CPU (NumPy) unless to("gpu") called
    return Signal(
        samples=samples,
        sampling_rate=symbol_rate * sps,
        symbol_rate=symbol_rate,
        modulation_scheme=f"{modulation.upper()}",
        modulation_order=order,
        source_bits=bits,
        pulse_shape=pulse_shape,
        pulse_params={
            "filter_span": filter_span,
            "rrc_rolloff": rrc_rolloff,
            "rc_rolloff": rc_rolloff,
            "smoothrect_bt": smoothrect_bt,
            "gaussian_bt": gaussian_bt,
        },
    )


def pam(
    order: int,
    bipolar: bool,
    num_symbols: int,
    sps: int,
    symbol_rate: float,
    pulse_shape: str = "rect",
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
    smoothrect_bt: float = 1.0,
    gaussian_bt: float = 0.3,
    seed: Optional[int] = None,
) -> Signal:
    """
    Generate a PAM baseband waveform (NRZ).

    Args:
        order: Modulation order (2, 4, 8, etc.).
        bipolar: Whether to use bipolar (True) or unipolar (False) PAM.
        num_symbols: Number of symbols to generate.
        sps: Samples per symbol (integer).
        symbol_rate: Symbol rate in Hz.
        pulse_shape: Pulse shape ('rect', 'rrc', 'rc', 'gaussian', 'smoothrect', 'none').
        filter_span: Filter span in symbols.
        rrc_rolloff: Roll-off factor for RRC shaping filter.
        rc_rolloff: Roll-off factor for RC shaping filter.
        smoothrect_bt: BT product for smoothrect shaping filter.
        gaussian_bt: BT product for Gaussian shaping filter.
        seed: Random seed.

    Returns:
        A `Signal` instance with PAM samples and metadata.
    """
    logger.info(
        f"Generating PAM ({'Bipolar' if bipolar else 'Unipolar'}): "
        f"order={order}, symbols={num_symbols}, pulse={pulse_shape}."
    )
    sig = generate_baseband(
        modulation="ask",
        order=order,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        pulse_shape=pulse_shape,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        rc_rolloff=rc_rolloff,
        smoothrect_bt=smoothrect_bt,
        gaussian_bt=gaussian_bt,
        seed=seed,
    )

    if not bipolar:
        xp = sig.xp
        # Shift so minimum is 0
        sig.samples = sig.samples - xp.min(sig.samples)
        # Normalize so max amplitude is 1 (standard for unipolar)
        sig.samples = utils.normalize(sig.samples, "max_amplitude")

    sig.modulation_scheme = f"PAM{'-BIPOL' if bipolar else '-UNIPOL'}"
    return sig


def rzpam(
    order: int,
    bipolar: bool,
    num_symbols: int,
    sps: int,
    symbol_rate: float,
    pulse_shape: str = "rect",
    filter_span: int = 10,
    smoothrect_bt: float = 1.0,
    seed: Optional[int] = None,
) -> Signal:
    """
    Generate a RZ (Return-to-Zero) PAM baseband waveform.

    Args:
        order: Modulation order (2, 4, 8, etc.).
        bipolar: Whether to use bipolar (True) or unipolar (False) PAM.
        num_symbols: Number of symbols to generate.
        sps: Samples per symbol (integer).
        symbol_rate: Symbol rate in Hz.
        pulse_shape: 'rect' (simplest) or 'smoothrect'. Others are prohibited.
        smoothrect_bt: BT product for smoothrect shaping filter.
        filter_span: Filter span in symbols.
        seed: Random seed.

    Returns:
        A `Signal` instance with RZ-PAM samples and metadata.
    """
    logger.info(
        f"Generating RZ-PAM ({'Bipolar' if bipolar else 'Unipolar'}): "
        f"order={order}, symbols={num_symbols}, pulse={pulse_shape}."
    )
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

    xp = np

    # Generate symbols
    k = int(xp.log2(order))
    if 2**k != order:
        raise ValueError(f"PAM order must be power of 2, got {order}")
    num_bits = num_symbols * k
    bits = sequences.random_bits(num_bits, seed=seed)
    symbols = mapping.map_bits(bits, modulation="ask", order=order)

    if not bipolar:
        # For Unipolar RZ, shift symbols so the lowest level is 0
        symbols = symbols - xp.min(symbols)

    # Apply RZ Pulse Shaping
    if pulse_shape == "rect":
        h = xp.ones(int(sps / 2))
        samples = utils.normalize(
            scipy.signal.resample_poly(symbols, int(sps), 1, window=h), "max_amplitude"
        )

    elif pulse_shape == "smoothrect":
        # RZ Smooth Rect: 50% width (0.5 symbol duration)
        h = filtering.smoothrect_taps(
            sps=sps, span=filter_span, bt=smoothrect_bt, pulse_width=0.5
        )

        # Apply pulse using polyphase resampling
        samples = utils.normalize(
            scipy.signal.resample_poly(symbols, int(sps), 1, window=h), "max_amplitude"
        )

    # Return Signal with format RZ-PAM-M
    return Signal(
        samples=samples,
        sampling_rate=symbol_rate * sps,
        symbol_rate=symbol_rate,
        modulation_scheme=f"RZ-PAM{'-BIPOL' if bipolar else '-UNIPOL'}",
        modulation_order=order,
        source_bits=bits,
        pulse_shape=pulse_shape,
        pulse_params={
            "filter_span": filter_span,
            "smoothrect_bt": smoothrect_bt,
            "pulse_width": 0.5,
        },
    )


def psk(
    order: int,
    num_symbols: int,
    sps: float,
    symbol_rate: float,
    pulse_shape: str = "rrc",
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
    smoothrect_bt: float = 1.0,
    gaussian_bt: float = 0.3,
    seed: Optional[int] = None,
) -> Signal:
    """Wrapper for PSK baseband waveform generation."""
    return generate_baseband(
        modulation="psk",
        order=order,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        pulse_shape=pulse_shape,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        rc_rolloff=rc_rolloff,
        smoothrect_bt=smoothrect_bt,
        gaussian_bt=gaussian_bt,
        seed=seed,
    )


def qam(
    order: int,
    num_symbols: int,
    sps: float,
    symbol_rate: float,
    pulse_shape: str = "rrc",
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
    smoothrect_bt: float = 1.0,
    gaussian_bt: float = 0.3,
    seed: Optional[int] = None,
) -> Signal:
    """Wrapper for QAM baseband waveform generation."""
    return generate_baseband(
        modulation="qam",
        order=order,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        pulse_shape=pulse_shape,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        rc_rolloff=rc_rolloff,
        smoothrect_bt=smoothrect_bt,
        gaussian_bt=gaussian_bt,
        seed=seed,
    )
