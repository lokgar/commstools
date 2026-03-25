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
gmi :
    Computes Generalised Mutual Information (GMI) from per-bit LLRs.
mi :
    Estimates Mutual Information (MI) under a Gaussian channel assumption.
"""

from typing import Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch
from .logger import logger


def evm(
    rx_symbols: ArrayType,
    tx_symbols: ArrayType,
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

    rx, xp, _ = dispatch(rx_symbols)
    tx = xp.asarray(tx_symbols)

    # Ensure shape consistency
    if rx.shape != tx.shape:
        raise ValueError(f"Shape mismatch: rx {rx.shape} != tx {tx.shape}")

    # Determine axis for time/symbols
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
    rx_symbols: ArrayType,
    tx_symbols: ArrayType,
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

    rx, xp, _ = dispatch(rx_symbols)
    tx = xp.asarray(tx_symbols)

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


def gmi(
    llrs: ArrayType,
    tx_bits: ArrayType,
) -> float:
    r"""
    Computes the Generalised Mutual Information (GMI) from per-bit LLRs.

    GMI is the achievable information rate under bit-interleaved coded
    modulation (BICM) and is the standard capacity metric for coded optical /
    wireless systems.  It uses pre-computed LLRs so the modulation format
    does not need to be known by this function.

    .. math::

        \text{GMI} = \sum_{b=0}^{k-1}
            \left\{
                1 - \frac{1}{N} \sum_{n=1}^{N}
                \log_2\!\left(1 + e^{-\Lambda_b[n]\,(1 - 2\,c_b[n])}\right)
            \right\}

    where :math:`\Lambda_b[n]` is the LLR for bit :math:`b` of symbol
    :math:`n`, and :math:`c_b[n] \in \{0, 1\}` is the transmitted bit.

    The per-term softplus is computed in a numerically stable form:

    .. math::

        \log_2(1 + e^{-x}) = \frac{1}{\ln 2}
            \bigl[\ln(1 + e^{-|x|}) + \max(0, -x)\bigr]

    which avoids overflow for large :math:`|x|`.

    Parameters
    ----------
    llrs : array_like
        Per-bit LLRs. Accepted shapes:

        * ``(N * k,)`` — flat, all bits concatenated.
        * ``(N, k)`` — symbols along axis 0, bits along axis 1.

        where *N* is the number of symbols and *k = log₂(M)* bits per symbol.
        Positive LLR → bit 0 more likely; negative → bit 1 more likely.
        LLRs from :func:`~commstools.mapping.compute_llr` match this convention.
    tx_bits : array_like
        Transmitted bits (0/1 integers), same shape as ``llrs``.

    Returns
    -------
    float
        GMI in bits per channel use (b/cu). In the range ``[0, log₂(M)]``;
        equals ``log₂(M)`` at infinite SNR and approaches 0 at very low SNR.

    References
    ----------
    A. Alvarado, E. Agrell, D. Lavery, R. Maher, and P. Bayvel,
    "Replacing the soft-decision FEC limit paradigm in the design of optical
    communication systems," *J. Lightw. Technol.*, vol. 33, no. 20, 2015.

    Examples
    --------
    >>> llrs = compute_llr(rx_symbols, "qam", 16, noise_var=0.1, output='numpy')
    >>> gmi_value = gmi(llrs, tx_bits)
    """
    llrs_arr = np.asarray(llrs, dtype=np.float64)
    bits_arr = np.asarray(tx_bits, dtype=np.float64)

    # Detect k (bits per symbol) from 2D shape before flattening.
    # For (N, k) input: sum over k bit positions → GMI ∈ [0, k].
    # For 1D (N·k,) input: k=1, returns per-bit GMI ∈ [0, 1].
    k = llrs_arr.shape[1] if llrs_arr.ndim == 2 else 1

    llrs_np = llrs_arr.ravel()
    bits_np = bits_arr.ravel()

    if llrs_np.shape != bits_np.shape:
        raise ValueError(
            f"llrs and tx_bits must have the same number of elements. "
            f"Got {llrs_np.shape} vs {bits_np.shape}."
        )

    # Signed LLR: positive when transmitted bit is 0, negative when bit is 1.
    # The argument to log2(1 + exp(·)) is: -LLR * (1 - 2*bit)
    # = -LLR when bit=0  (should be positive → small loss)
    # = +LLR when bit=1  (should be negative → small loss at high SNR)
    x = -llrs_np * (1.0 - 2.0 * bits_np)

    # Numerically stable softplus: log2(1 + exp(x)) = log1p(exp(-|x|))/ln2 + max(0,x)/ln2
    ln2 = np.log(2.0)
    softplus = (np.log1p(np.exp(-np.abs(x))) + np.maximum(0.0, x)) / ln2

    # GMI = Σ_{b=0}^{k-1} (1 - mean_n[softplus_b]) = k * (1 - mean_all[softplus])
    gmi_value = float(k * (1.0 - np.mean(softplus)))
    logger.info(f"GMI: {gmi_value:.4f} b/cu")
    return gmi_value


def mi(
    symbols_rx: ArrayType,
    modulation: str,
    order: int,
    noise_var: float,
) -> float:
    r"""
    Estimates the Mutual Information (MI) under a Gaussian channel assumption.

    MI is the Shannon capacity of the discrete-input continuous-output (DCMC)
    AWGN channel.  It upper-bounds GMI and is achievable with optimal
    (non-binary) coding.

    Uses the Monte-Carlo estimator with Gaussian likelihoods:

    .. math::

        p(s_m \mid r) \propto \exp\!\left(-\frac{|r - s_m|^2}{\sigma^2}\right)

    .. math::

        \text{MI} = \log_2 M +
            \frac{1}{N} \sum_{n=1}^{N}
            \sum_{m=1}^{M} p(s_m \mid r_n)
            \log_2 p(s_m \mid r_n)

    Computed via the log-sum-exp trick for numerical stability.

    Parameters
    ----------
    symbols_rx : array_like
        Received noisy symbols. Shape: ``(N,)``.
    modulation : str
        Modulation type: ``'psk'``, ``'qam'``, or ``'ask'``.
    order : int
        Modulation order *M*.
    noise_var : float
        Complex noise variance :math:`\sigma^2` of the AWGN model.
        For unit-power symbols at :math:`E_s/N_0` (dB):
        :math:`\sigma^2 = 10^{-E_s/N_0 / 10}`.

    Returns
    -------
    float
        MI in bits per channel use (b/cu). In the range ``[0, log₂(M)]``.

    References
    ----------
    G. Böcherer, F. Steiner, and P. Schulte, "Bandwidth efficient and
    rate-matched low-density parity-check coded modulation," *IEEE Trans.
    Commun.*, vol. 63, no. 12, 2015.

    Examples
    --------
    >>> mi_value = mi(rx_symbols, "qam", 64, noise_var=0.05)
    """
    from .mapping import gray_constellation

    # Move to CPU first (handles CuPy arrays via .get())
    symbols_cpu, _, _ = dispatch(symbols_rx)
    rx = np.asarray(
        symbols_cpu.get() if hasattr(symbols_cpu, "get") else symbols_cpu,
        dtype=np.complex128,
    ).ravel()  # (N,)
    constellation = gray_constellation(modulation, order).astype(np.complex128)  # (M,)
    M = len(constellation)

    # Log-likelihoods: log p(r | s_m) = -|r - s_m|² / σ²  (drop constant)
    # Shape: (N, M)
    diff = rx[:, None] - constellation[None, :]       # (N, M)
    log_liks = -(diff.real**2 + diff.imag**2) / noise_var

    # log p(s_m | r) = log_liks - log(sum_m exp(log_liks))  [log-sum-exp]
    log_sum = np.log(np.sum(np.exp(log_liks - log_liks.max(axis=1, keepdims=True)), axis=1))
    log_sum += log_liks.max(axis=1)                   # undo the shift

    log_posterior = log_liks - log_sum[:, None]        # (N, M), log base e
    posterior = np.exp(log_posterior)                  # (N, M), sum-to-1 per row

    # MI = log2(M) + (1/N) Σ_n Σ_m p(s_m|r_n) · log2 p(s_m|r_n)
    # Entropy of posterior (negative cross-entropy with uniform prior)
    mi_nats = np.sum(posterior * log_posterior, axis=1).mean()  # Σ_m p·log_e(p)
    mi_value = float(np.log2(M) + mi_nats / np.log(2.0))
    mi_value = float(np.clip(mi_value, 0.0, np.log2(M)))

    logger.info(f"MI: {mi_value:.4f} b/cu  (max {np.log2(M):.2f} b/cu)")
    return mi_value
