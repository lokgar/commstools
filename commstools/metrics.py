"""
Performance metrics for digital communications.

This module provides signal quality and error rate metrics:
- **EVM** (Error Vector Magnitude): Measures distortion between received and reference symbols.
- **SNR estimation**: Data-aided SNR estimation from known reference.
- **BER** (Bit Error Rate): Ratio of erroneous bits.
- **Q-factor**: Quality metric derived from BER or EVM.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch
from .logger import logger

if TYPE_CHECKING:
    from .core import Signal


def evm(
    rx_symbols: Union[ArrayType, "Signal"],
    tx_symbols: Union[ArrayType, "Signal"],
    normalize: bool = True,
) -> Tuple[float, float]:
    """
    Compute Error Vector Magnitude (EVM) between received and reference symbols.

    EVM measures the deviation of received symbols from ideal constellation points,
    expressed as a percentage of the reference signal's RMS amplitude.

    Args:
        rx_symbols: Received symbols (array or Signal). Shape: (..., N).
        tx_symbols: Reference/transmitted symbols (array or Signal). Shape: (..., N).
        normalize: If True, normalize EVM by RMS of reference (standard definition).
                   If False, return raw RMS error.

    Returns:
        Tuple of (evm_percent, evm_db):
            - evm_percent: EVM as percentage (0-100+).
            - evm_db: EVM in dB (20*log10(evm_percent/100)).

    Note:
        For MIMO signals with shape (Channels, N), computes EVM across all samples.
    """
    from .core import Signal

    # Extract samples if Signal objects
    if isinstance(rx_symbols, Signal):
        rx = (
            rx_symbols.source_symbols
            if rx_symbols.source_symbols is not None
            else rx_symbols.samples
        )
    else:
        rx = rx_symbols

    if isinstance(tx_symbols, Signal):
        tx = (
            tx_symbols.source_symbols
            if tx_symbols.source_symbols is not None
            else tx_symbols.samples
        )
    else:
        tx = tx_symbols

    rx, xp, _ = dispatch(rx)
    tx = xp.asarray(tx)

    # Error vector: difference between received and reference
    error = rx - tx

    # RMS of error vector
    error_power = xp.mean(xp.abs(error) ** 2)
    error_rms = xp.sqrt(error_power)

    if normalize:
        # Normalize by RMS of reference signal
        ref_power = xp.mean(xp.abs(tx) ** 2)
        ref_rms = xp.sqrt(ref_power)

        # Avoid division by zero
        if float(ref_rms) < 1e-20:
            logger.warning("Reference signal power near zero, EVM undefined.")
            return float("inf"), float("inf")

        evm_ratio = float(error_rms / ref_rms)
    else:
        evm_ratio = float(error_rms)

    evm_percent = evm_ratio * 100.0
    evm_db = 20.0 * np.log10(evm_ratio) if evm_ratio > 0 else float("-inf")

    logger.debug(f"EVM: {evm_percent:.2f}% ({evm_db:.2f} dB)")
    return evm_percent, evm_db


def snr_estimate(
    rx_symbols: Union[ArrayType, "Signal"],
    tx_symbols: Union[ArrayType, "Signal"],
) -> float:
    """
    Estimate SNR from received symbols using known reference (data-aided).

    Computes SNR as the ratio of signal power to error power:
        SNR = P_signal / P_noise = P_tx / P_error

    Args:
        rx_symbols: Received symbols (array or Signal). Shape: (..., N).
        tx_symbols: Reference/transmitted symbols (array or Signal). Shape: (..., N).

    Returns:
        Estimated SNR in dB.

    Note:
        This is a data-aided estimate requiring known transmitted symbols.
        For blind estimation, use dedicated algorithms (M2M4, etc.).
    """
    from .core import Signal

    if isinstance(rx_symbols, Signal):
        rx = (
            rx_symbols.source_symbols
            if rx_symbols.source_symbols is not None
            else rx_symbols.samples
        )
    else:
        rx = rx_symbols

    if isinstance(tx_symbols, Signal):
        tx = (
            tx_symbols.source_symbols
            if tx_symbols.source_symbols is not None
            else tx_symbols.samples
        )
    else:
        tx = tx_symbols

    rx, xp, _ = dispatch(rx)
    tx = xp.asarray(tx)

    # Signal power (reference)
    signal_power = xp.mean(xp.abs(tx) ** 2)

    # Noise power (error)
    error = rx - tx
    noise_power = xp.mean(xp.abs(error) ** 2)

    # Avoid division by zero
    if float(noise_power) < 1e-20:
        logger.debug("Noise power near zero, SNR very high.")
        return float("inf")

    snr_linear = float(signal_power / noise_power)
    snr_db = 10.0 * np.log10(snr_linear)

    logger.debug(f"SNR estimate: {snr_db:.2f} dB")
    return snr_db


def ber(
    bits_rx: ArrayType,
    bits_tx: ArrayType,
) -> float:
    """
    Compute Bit Error Rate (BER) between received and transmitted bits.

    Args:
        bits_rx: Received bit sequence (0s and 1s). Shape: (..., N).
        bits_tx: Transmitted bit sequence (0s and 1s). Shape: (..., N).

    Returns:
        BER as a ratio in [0, 1]. BER = num_errors / total_bits.

    Note:
        Both arrays must have the same shape.
    """
    bits_rx, xp, _ = dispatch(bits_rx)
    bits_tx = xp.asarray(bits_tx)

    # Flatten for comparison
    rx_flat = bits_rx.flatten()
    tx_flat = bits_tx.flatten()

    if rx_flat.size != tx_flat.size:
        raise ValueError(
            f"Bit sequence lengths must match: rx={rx_flat.size}, tx={tx_flat.size}"
        )

    total_bits = rx_flat.size
    if total_bits == 0:
        return 0.0

    # Count errors (XOR gives 1 where bits differ)
    errors = xp.sum(rx_flat != tx_flat)
    ber_value = float(errors) / float(total_bits)

    logger.debug(f"BER: {ber_value:.2e} ({int(errors)}/{total_bits} errors)")
    return ber_value


def q_factor(
    ber_value: Optional[float] = None,
    evm_percent: Optional[float] = None,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
) -> float:
    """
    Compute Q-factor from BER or EVM.

    Q-factor is a quality metric commonly used in optical communications.
    It represents the "distance" (in standard deviations) between signal levels.

    Args:
        ber_value: Bit Error Rate (0 to 0.5). Uses Q = sqrt(2) * erfc^-1(2*BER).
        evm_percent: EVM in percent. Uses modulation-aware calculation.
        modulation: Modulation type ('psk', 'qam', 'pam'). Required for EVM.
        order: Modulation order. Required for EVM.

    Returns:
        Q-factor (linear, dimensionless).

    Raises:
        ValueError: If neither ber_value nor evm_percent provided,
            or if EVM given without modulation info.

    Note:
        - Q > 6 typically indicates good performance (BER < 1e-9).
        - For EVM: Q = d_min / (2 * σ) where σ relates to EVM.
        - Different modulations have different d_min relative to symbol power.
    """
    from scipy.special import erfcinv

    if ber_value is not None:
        if ber_value <= 0:
            return float("inf")
        if ber_value >= 0.5:
            return 0.0

        # Q = sqrt(2) * erfc^-1(2 * BER)
        q = np.sqrt(2) * erfcinv(2 * ber_value)
        logger.debug(f"Q-factor from BER: {q:.2f}")
        return float(q)

    elif evm_percent is not None:
        if evm_percent <= 0:
            return float("inf")

        evm_ratio = evm_percent / 100.0

        # Modulation-aware Q-factor: Q = d_min / (2 * EVM * sqrt(Es))
        # For normalized constellations (Es = 1), Q = d_min / (2 * EVM)
        if modulation is None or order is None:
            # Fallback to simple approximation (assumes QPSK-like)
            q = 1.0 / evm_ratio
            logger.warning(
                "Q-factor from EVM without modulation info - using QPSK approximation."
            )
        else:
            mod = modulation.lower()
            k = int(np.log2(order)) if order > 1 else 1

            # Compute d_min relative to unit average power
            if mod == "psk":
                # PSK: d_min = 2 * sin(π/M)
                d_min = 2 * np.sin(np.pi / order)
            elif mod in ("qam", "pam", "ask"):
                # Square QAM: d_min = sqrt(6 / (M - 1)) for unit power
                # PAM/ASK: d_min = sqrt(12 / (M^2 - 1)) for unit power
                if mod == "qam" and k % 2 == 0:
                    # Square QAM
                    d_min = np.sqrt(6 / (order - 1))
                else:
                    # PAM or non-square QAM
                    d_min = np.sqrt(12 / (order**2 - 1))
            else:
                # Unknown modulation - fallback
                d_min = 2.0 / np.sqrt(order)

            # Q = d_min / (2 * σ), where σ ≈ EVM (for unit power)
            q = d_min / (2 * evm_ratio)
            logger.debug(f"Q-factor from EVM ({mod}-{order}): {q:.2f}")

        return float(q)

    else:
        raise ValueError("Must provide either ber_value or evm_percent.")


def q_factor_db(
    ber_value: Optional[float] = None,
    evm_percent: Optional[float] = None,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
) -> float:
    """
    Compute Q-factor in dB.

    Args:
        ber_value: Bit Error Rate (0 to 0.5).
        evm_percent: EVM in percent.
        modulation: Modulation type ('psk', 'qam', 'pam').
        order: Modulation order.

    Returns:
        Q-factor in dB: 20 * log10(Q).
    """
    q = q_factor(
        ber_value=ber_value, evm_percent=evm_percent, modulation=modulation, order=order
    )
    if q <= 0:
        return float("-inf")
    return 20.0 * np.log10(q)
