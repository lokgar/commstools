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

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch
from .logger import logger

if TYPE_CHECKING:
    from .core import Signal


def _extract_symbols(s: Union[ArrayType, "Signal"]) -> ArrayType:
    """Return the array from a Signal (preferring source_symbols) or pass through."""
    from .core import Signal

    if isinstance(s, Signal):
        return s.source_symbols if s.source_symbols is not None else s.samples
    return s


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
    from . import helpers

    rx, xp, _ = dispatch(_extract_symbols(rx_symbols))
    tx = xp.asarray(_extract_symbols(tx_symbols))

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
        logger.info(f"EVM: {evm_percent:.2f}% ({evm_db:.2f} dB)")
        return evm_percent, evm_db

    # If array:
    evm_percent = evm_ratio * 100.0
    with np.errstate(divide="ignore"):
        evm_db = 20.0 * xp.log10(evm_ratio)

    # Overwrite indices
    evm_percent[low_pwr_mask] = float("inf")
    evm_db[low_pwr_mask] = float("inf")

    for ch in range(evm_percent.shape[0]):
        logger.info(
            f"EVM Ch{ch}: {float(evm_percent[ch]):.2f}% ({float(evm_db[ch]):.2f} dB)"
        )

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
    from . import helpers

    rx, xp, _ = dispatch(_extract_symbols(rx_symbols))
    tx = xp.asarray(_extract_symbols(tx_symbols))

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
        snr_val = float(snr_db)
        logger.info(f"SNR: {snr_val:.2f} dB")
        return snr_val

    # Array case: Overwrite with -inf where (low_pwr_mask AND NOT zero_noise_mask)
    # Note: where zero_noise_mask is True, snr_db is already 'inf' from division
    mask_to_neg_inf = low_pwr_mask & (~zero_noise_mask)
    snr_db[mask_to_neg_inf] = float("-inf")

    for ch in range(snr_db.shape[0]):
        logger.info(f"SNR Ch{ch}: {float(snr_db[ch]):.2f} dB")

    return snr_db


def ber(
    bits_rx: ArrayType,
    bits_tx: ArrayType,
) -> Union[float, ArrayType]:
    """
    Computes the Bit Error Rate (BER) between two bit sequences.

    Calculation is performed per-channel (along ``axis=-1``) to properly
    handle MIMO/multichannel signals.

    Parameters
    ----------
    bits_rx : array_like
        Received bit sequence (binary 0/1). Shape: (..., N).
    bits_tx : array_like
        Original transmitted bit sequence (binary 0/1). Shape: (..., N).

    Returns
    -------
    float or ndarray
        BER as a ratio in [0, 1]. Scalar for 1D input, array for multichannel.

    Notes
    -----
    The calculation is backend-agnostic and supports comparison of massive
    bit sequences on both CPU and GPU.
    """
    bits_rx, xp, _ = dispatch(bits_rx)
    bits_tx = xp.asarray(bits_tx)

    if bits_rx.shape != bits_tx.shape:
        raise ValueError(f"Shape mismatch: rx {bits_rx.shape} != tx {bits_tx.shape}")

    if bits_rx.size == 0:
        return 0.0

    # Per-stream error count along time axis
    errors = xp.sum(bits_rx != bits_tx, axis=-1)
    total = bits_rx.shape[-1]
    ber_values = errors / total

    if ber_values.ndim == 0:
        ber_value = float(ber_values)
        logger.info(f"BER: {ber_value:.2e} ({int(errors)}/{total} errors)")
        return ber_value

    for ch in range(ber_values.shape[0]):
        logger.info(
            f"BER Ch{ch}: {float(ber_values[ch]):.2e} "
            f"({int(errors[ch])}/{total} errors)"
        )
    return ber_values
