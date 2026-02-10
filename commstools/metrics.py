"""
Performance metrics for digital communications.

This module provides high-performance routines for evaluating signal quality,
error rates, and reliability across different computational backends.

Functions
---------
evm :
    Calculates Error Vector Magnitude between received and reference symbols.
snr :
    Performs data-aided SNR estimation.
ber :
    Computes Bit Error Rate between bit sequences.
q_factor :
    Calculates communication quality Q-factor from BER or EVM.
q_factor_db :
    Expresses Q-factor in decibels.
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
    Computes Error Vector Magnitude (EVM) between received and reference symbols.

    EVM is a standard measure of signal quality that quantifies the deviation
    of received symbols from their ideal constellation points. It is
    calculated as the ratio of the RMS error to the RMS amplitude of the
    reference signal.

    Parameters
    ----------
    rx_symbols : array_like or Signal
        Received symbols to evaluate. Shape: (..., N_symbols).
    tx_symbols : array_like or Signal
        Ideal reference (transmitted) symbols. Shape: (..., N_symbols).
    normalize : bool, default True
        If True, normalizes the error by the RMS amplitude of the reference
        signal (standard definition).

    Returns
    -------
    evm_percent : float
        EVM expressed as a percentage (e.g., 5.0 for 5%).
    evm_db : float
        EVM expressed in decibels (dB), calculated as $20 \log_{10}(EVM_{ratio})$.

    Notes
    -----
    For MIMO or multichannel signals, the EVM is computed across all spatial
    and temporal samples to provide a single aggregate metric.
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


def snr(
    rx_symbols: Union[ArrayType, "Signal"],
    tx_symbols: Union[ArrayType, "Signal"],
) -> float:
    """
    Estimates SNR from received symbols using a known reference (Data-Aided).

    SNR is computed as the ratio of average signal power to noise (error)
    power:
    $SNR_{linear} = \frac{P_{signal}}{P_{noise}} = \frac{E[|tx|^2]}{E[|rx - tx|^2]}$

    Parameters
    ----------
    rx_symbols : array_like or Signal
        Received noisy symbols. Shape: (..., N_symbols).
    tx_symbols : array_like or Signal
        Known transmitted symbols. Shape: (..., N_symbols).

    Returns
    -------
    float
        Estimated SNR in dB.

    Notes
    -----
    This method requires knowledge of the transmitted data. For non-data-aided
    (blind) estimation, more complex algorithms (e.g., moments-based M2M4)
    are required.
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
    Computes the Bit Error Rate (BER) between two bit sequences.

    Parameters
    ----------
    bits_rx : array_like
        Received bit sequence (binary 0/1). Shape: (..., N).
    bits_tx : array_like
        Original transmitted bit sequence (binary 0/1). Shape: (..., N).

    Returns
    -------
    float
        BER as a ratio in the range [0, 1]. Calculated as `errors / total_bits`.

    Notes
    -----
    The calculation is backend-agnostic and supports comparison of massive
    bit sequences on both CPU and GPU.
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
    Computes the Q-factor from BER or EVM.

    The Q-factor is a dimensionless quality metric that represents the
    normalized distance between constellation levels in terms of standard
    deviation. It is widely used in optical and high-speed communications.

    Parameters
    ----------
    ber_value : float, optional
        Bit Error Rate (0 to 0.5). If provided, calculates
        $Q = \sqrt{2} \cdot \text{erfc}^{-1}(2 \cdot BER)$.
    evm_percent : float, optional
        EVM in percent. Requires `modulation` and `order` for accurate mapping.
    modulation : {"psk", "qam", "ask", "pam"}, optional
        Modulation type used for EVM-based calculation.
    order : int, optional
        Modulation order (e.g., 4, 16, 64).

    Returns
    -------
    float
        Linear Q-factor.

    Raises
    ------
    ValueError
        If neither `ber_value` nor `evm_percent` is provided.

    Notes
    -----
    - For EVM, $Q = \frac{d_{min}}{2\sigma}$, where $d_{min}$ is the
      minimum distance between points in a unit-power constellation.
    - $Q > 6$ typically indicates excellent performance ($BER < 10^{-9}$).
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
    Computes the Q-factor in decibels ($20 \log_{10}(Q)$).

    Parameters
    ----------
    ber_value : float, optional
        Bit Error Rate.
    evm_percent : float, optional
        EVM in percent.
    modulation : str, optional
        Modulation identifier.
    order : int, optional
        Modulation order.

    Returns
    -------
    float
        Q-factor in dB.
    """
    q = q_factor(
        ber_value=ber_value, evm_percent=evm_percent, modulation=modulation, order=order
    )
    if q <= 0:
        return float("-inf")
    return 20.0 * np.log10(q)
