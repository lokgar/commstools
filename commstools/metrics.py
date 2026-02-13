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
) -> Tuple[Union[float, ArrayType], Union[float, ArrayType]]:
    """
    Computes Error Vector Magnitude (EVM) between received and reference symbols.

    EVM is a standard measure of signal quality that quantifies the deviation
    of received symbols from their ideal constellation points.

    Calculation is performed per-channel (independent gain normalization) to
    properly handle MIMO/multichannel signals.

    Parameters
    ----------
    rx_symbols : array_like or Signal
        Received symbols to evaluate. Shape: (..., N_symbols).
    tx_symbols : array_like or Signal
        Ideal reference (transmitted) symbols. Shape: (..., N_symbols).

    Returns
    -------
    evm_percent : float or ndarray
        EVM expressed as a percentage. Returns an array if input is multichannel.
    evm_db : float or ndarray
        EVM expressed in decibels (dB).

    Notes
    -----
    For MIMO, metrics are calculated independently for each stream (row).
    """
    from .core import Signal
    from . import helpers

    # Extract samples
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

    # Ensure shape consistency
    if rx.shape != tx.shape:
        raise ValueError(f"Shape mismatch: rx {rx.shape} != tx {tx.shape}")

    # Determine axis for time/symbols (assumed last)
    axis = -1

    # helper to check normalization
    def _is_normalized(arr, ax):
        pwr = xp.mean(xp.abs(arr) ** 2, axis=ax)
        return xp.allclose(pwr, 1.0, atol=1e-3)

    # Robustness: Check if normalized, if not -> normalize per channel
    if not _is_normalized(rx, axis):
        rx = helpers.normalize(rx, axis=axis, mode="average_power")

    if not _is_normalized(tx, axis):
        tx = helpers.normalize(tx, axis=axis, mode="average_power")

    # Check for zero power ref
    # If tx was zeros, normalize might have left it as zeros (safe/helpers.py behavior)
    ref_pwr = xp.mean(xp.abs(tx) ** 2, axis=axis)
    low_pwr_mask = ref_pwr < 1e-20

    if xp.any(low_pwr_mask):
        logger.warning("Reference signal power near zero in one or more channels.")

    # Error vector
    error = rx - tx

    # RMS of error vector (per channel)
    error_power = xp.mean(xp.abs(error) ** 2, axis=axis)
    error_rms = xp.sqrt(error_power)

    # Ref RMS is 1.0 by definition (per channel)
    evm_ratio = error_rms

    # Apply infinity where reference was zero
    # We must operate on matched types.
    # If scalar:
    if evm_ratio.ndim == 0:
        if low_pwr_mask:
            return float("inf"), float("inf")
        evm_percent = float(evm_ratio * 100.0)
        with np.errstate(divide="ignore"):
            evm_db = (
                float(20.0 * xp.log10(evm_ratio)) if evm_ratio > 0 else float("-inf")
            )
        return evm_percent, evm_db

    # If array:
    evm_percent = evm_ratio * 100.0
    with np.errstate(divide="ignore"):
        evm_db = 20.0 * xp.log10(evm_ratio)

    # Overwrite indices
    evm_percent[low_pwr_mask] = float("inf")
    evm_db[low_pwr_mask] = float("inf")

    return evm_percent, evm_db


def snr(
    rx_symbols: Union[ArrayType, "Signal"],
    tx_symbols: Union[ArrayType, "Signal"],
) -> Union[float, ArrayType]:
    """
    Estimates SNR from received symbols using a known reference (Data-Aided).

    SNR is computed as the ratio of average signal power to noise (error)
    power:
    $SNR_{linear} \approx \frac{1.0}{E[|rx - tx|^2]}$

    Calculation is performed per-channel (independent gain normalization).

    Parameters
    ----------
    rx_symbols : array_like or Signal
        Received noisy symbols. Shape: (..., N_symbols).
    tx_symbols : array_like or Signal
        Known transmitted symbols. Shape: (..., N_symbols).

    Returns
    -------
    float or ndarray
        Estimated SNR in dB. Returns array if input is multichannel.
    """
    from .core import Signal
    from . import helpers

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

    if rx.shape != tx.shape:
        raise ValueError(f"Shape mismatch: rx {rx.shape} != tx {tx.shape}")

    axis = -1

    def _is_normalized(arr, ax):
        pwr = xp.mean(xp.abs(arr) ** 2, axis=ax)
        return xp.allclose(pwr, 1.0, atol=1e-3)

    # Normalize per channel if needed
    if not _is_normalized(rx, axis):
        rx = helpers.normalize(rx, axis=axis, mode="average_power")

    if not _is_normalized(tx, axis):
        tx = helpers.normalize(tx, axis=axis, mode="average_power")

    # Check for zero power ref for SNR too
    ref_pwr = xp.mean(xp.abs(tx) ** 2, axis=axis)
    low_pwr_mask = ref_pwr < 1e-20

    # Signal power (reference) is 1.0 per channel effectively, BUT
    # if original ref was zero, SNR is 0 (-inf dB).
    signal_power = 1.0

    # Noise power (error)
    error = rx - tx
    noise_power = xp.mean(xp.abs(error) ** 2, axis=axis)

    # Calculate SNR
    # Handle zeros in noise_power
    with np.errstate(divide="ignore"):
        snr_linear = signal_power / noise_power
        snr_db = 10.0 * xp.log10(snr_linear)

    # Handle zero reference case
    # If noise is zero, SNR is Inf (perfect signal), even if ref was zero.
    # If noise is present but ref was zero, SNR is -Inf (0 linear).
    zero_noise_mask = noise_power < 1e-20

    if snr_db.ndim == 0:
        if zero_noise_mask:
            return float("inf")
        if low_pwr_mask:
            return float("-inf")
        return float(snr_db)

    # Array case: Overwrite with -inf where (low_pwr_mask AND NOT zero_noise_mask)
    # Note: where zero_noise_mask is True, snr_db is already 'inf' from division
    mask_to_neg_inf = low_pwr_mask & (~zero_noise_mask)
    snr_db[mask_to_neg_inf] = float("-inf")

    return snr_db


# TODO: check precision for ber
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
