"""
Performance metrics for digital communications.

This module provides high-performance routines for evaluating signal quality,
error rates, and reliability across different computational backends.

Functions
---------
evm :
    Error Vector Magnitude — data-aided or blind (decision-directed) mode.
snr :
    Data-aided SNR estimation.
ber :
    Bit Error Rate between bit sequences.
ser :
    Symbol Error Rate using ML hard decisions against a known constellation.
gmi :
    Generalised Mutual Information (GMI) from per-bit LLRs.
mi :
    Mutual Information (MI) under a Gaussian channel assumption.
"""

from typing import Optional, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch
from .logger import logger


def evm(
    rx_symbols: ArrayType,
    tx_symbols: Optional[ArrayType] = None,
    *,
    mode: str = "data_aided",
    modulation: Optional[str] = None,
    order: Optional[int] = None,
) -> Tuple[Union[float, ArrayType], Union[float, ArrayType]]:
    """
    Computes Error Vector Magnitude (EVM).

    Two modes are supported:

    * ``"data_aided"`` *(default)* — uses known transmitted symbols as the
      reference (classical simulation metric, most accurate).
    * ``"blind"`` — decision-directed: the nearest ideal constellation point
      to each received sample is used as the reference.  No knowledge of the
      transmitted sequence is required.  Equivalent to measuring the *blob
      size* around each constellation cluster, as a spectrum analyser would.
      At very low SNR the estimate is optimistically biased because erroneous
      decisions pull the centroid toward the wrong cluster.  The caller is
      responsible for passing a gain-corrected signal; no implicit gain
      normalisation is applied (``gray_constellation`` always returns
      unit-power constellations).

    Calculation is performed per-channel (independent gain normalisation) to
    properly handle MIMO/multichannel signals.

    Parameters
    ----------
    rx_symbols : array_like
        Received symbols. Shape: ``(..., N_symbols)``.
    tx_symbols : array_like, optional
        Ideal reference (transmitted) symbols. Shape: ``(..., N_symbols)``.
        Required when ``mode="data_aided"``, ignored when ``mode="blind"``.
    mode : {"data_aided", "blind"}, default "data_aided"
        Estimation mode (keyword-only).
    modulation : str, optional
        Modulation type (``"psk"``, ``"qam"``, ``"ask"``).
        Required when ``mode="blind"``.
    order : int, optional
        Modulation order *M*. Required when ``mode="blind"``.

    Returns
    -------
    evm_percent : float or ndarray
        EVM as a percentage. Scalar for SISO, array ``(N_ch,)`` for MIMO.
    evm_db : float or ndarray
        EVM in decibels (dB).

    Raises
    ------
    ValueError
        If ``mode="data_aided"`` and ``tx_symbols`` is ``None``, or
        ``mode="blind"`` and ``modulation``/``order`` are missing, or
        an unknown mode string is given.

    Notes
    -----
    For MIMO, metrics are calculated independently for each stream (row).
    """
    from . import helpers

    _VALID_MODES = ("data_aided", "blind")
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {_VALID_MODES}.")

    rx, xp, _ = dispatch(rx_symbols)
    axis = -1

    def _is_normalized(arr, ax):
        pwr = xp.mean(xp.abs(arr) ** 2, axis=ax)
        return xp.allclose(pwr, 1.0, atol=1e-3)

    # --- Build reference ---
    if mode == "data_aided":
        if tx_symbols is None:
            raise ValueError(
                "mode='data_aided' requires tx_symbols. "
                "Pass tx_symbols or use mode='blind'."
            )
        tx = xp.asarray(tx_symbols)
        if rx.shape != tx.shape:
            raise ValueError(f"Shape mismatch: rx {rx.shape} != tx {tx.shape}")

        if not _is_normalized(rx, axis):
            rx = helpers.normalize(rx, axis=axis, mode="average_power")
        if not _is_normalized(tx, axis):
            tx = helpers.normalize(tx, axis=axis, mode="average_power")

    else:  # blind
        if modulation is None or order is None:
            raise ValueError(
                "mode='blind' requires modulation and order. "
                "Example: evm(rx, mode='blind', modulation='qam', order=16)."
            )
        from .mapping import gray_constellation

        # gray_constellation always returns unit-average-power constellations.
        # No gain correction of rx is applied here — the caller is responsible
        # for passing a gain-corrected signal at the expected constellation power.
        constellation = xp.asarray(gray_constellation(modulation, order))  # (M,) unit power

        # ML hard decision: nearest constellation point per symbol.
        # rx shape (..., N) → (..., N, 1) vs (M,) → (..., N, M)
        dist = xp.abs(rx[..., None] - constellation) ** 2
        tx = constellation[xp.argmin(dist, axis=-1)]  # same shape as rx

    # --- EVM computation (shared) ---
    ref_pwr = xp.mean(xp.abs(tx) ** 2, axis=axis)
    low_pwr_mask = ref_pwr < 1e-20
    if xp.any(low_pwr_mask):
        logger.warning("Reference signal power near zero in one or more channels.")

    error = rx - tx
    error_power = xp.mean(xp.abs(error) ** 2, axis=axis)
    evm_ratio = xp.sqrt(error_power)  # ref power is 1.0 after normalisation

    if evm_ratio.ndim == 0:
        if low_pwr_mask:
            return float("inf"), float("inf")
        evm_percent = float(evm_ratio * 100.0)
        with np.errstate(divide="ignore"):
            evm_db = (
                float(20.0 * xp.log10(evm_ratio)) if evm_ratio > 0 else float("-inf")
            )
        logger.info(f"EVM [{mode}]: {evm_percent:.2f}% ({evm_db:.2f} dB)")
        return evm_percent, evm_db

    evm_percent = evm_ratio * 100.0
    with np.errstate(divide="ignore"):
        evm_db = 20.0 * xp.log10(evm_ratio)

    evm_percent[low_pwr_mask] = float("inf")
    evm_db[low_pwr_mask] = float("inf")

    for ch in range(evm_percent.shape[0]):
        logger.info(
            f"EVM [{mode}] Ch{ch}: {float(evm_percent[ch]):.2f}%"
            f" ({float(evm_db[ch]):.2f} dB)"
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


def ser(
    rx_symbols: ArrayType,
    tx_symbols: ArrayType,
    modulation: str,
    order: int,
) -> Union[float, ArrayType]:
    """
    Computes the Symbol Error Rate (SER) using ML hard decisions.

    Each received symbol is decided to its nearest constellation point
    (minimum Euclidean distance).  The decision is then compared to the
    corresponding transmitted symbol.  SER measures symbol-level errors
    before bit demapping and does not assume Gray coding, making it
    distinct from BER.

    Computation is fully vectorised and backend-agnostic (NumPy / CuPy).

    Parameters
    ----------
    rx_symbols : array_like
        Received (noisy) symbols. Shape: ``(..., N_symbols)``.
    tx_symbols : array_like
        Transmitted (ideal) symbols. Shape: ``(..., N_symbols)``.
    modulation : str
        Modulation type: ``"psk"``, ``"qam"``, or ``"ask"``.
    order : int
        Modulation order *M*.

    Returns
    -------
    float or ndarray
        SER as a ratio in ``[0, 1]``. Scalar for SISO input,
        array of shape ``(N_ch,)`` for MIMO input.

    Notes
    -----
    Both ``rx_symbols`` and ``tx_symbols`` are decided against the same
    constellation so the function is robust to any floating-point rounding
    in the reference values.
    """
    from .mapping import gray_constellation

    rx, xp, _ = dispatch(rx_symbols)
    tx = xp.asarray(tx_symbols)

    if rx.shape != tx.shape:
        raise ValueError(f"Shape mismatch: rx {rx.shape} != tx {tx.shape}")

    constellation = xp.asarray(gray_constellation(modulation, order))  # (M,)

    # Broadcast: rx/tx (..., N) → (..., N, 1) vs constellation (M,) → (..., N, M)
    dist_rx = xp.abs(rx[..., None] - constellation) ** 2   # (..., N, M)
    dist_tx = xp.abs(tx[..., None] - constellation) ** 2   # (..., N, M)

    dec_rx = xp.argmin(dist_rx, axis=-1)  # (..., N) — index into constellation
    dec_tx = xp.argmin(dist_tx, axis=-1)  # (..., N)

    errors = xp.sum(dec_rx != dec_tx, axis=-1)
    total = rx.shape[-1]
    ser_values = errors / total

    if ser_values.ndim == 0:
        ser_val = float(ser_values)
        logger.info(f"SER: {ser_val:.2e} ({int(errors)}/{total} errors)")
        return ser_val

    for ch in range(ser_values.shape[0]):
        logger.info(
            f"SER Ch{ch}: {float(ser_values[ch]):.2e} "
            f"({int(errors[ch])}/{total} errors)"
        )
    return ser_values


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
    llrs_arr, xp, _ = dispatch(llrs)
    llrs_arr = llrs_arr.astype(xp.float64)
    bits_arr = xp.asarray(tx_bits, dtype=xp.float64)

    # Detect k (bits per symbol) from 2D shape before flattening.
    # For (N, k) input: sum over k bit positions → GMI ∈ [0, k].
    # For 1D (N·k,) input: k=1, returns per-bit GMI ∈ [0, 1].
    k = llrs_arr.shape[1] if llrs_arr.ndim == 2 else 1

    llrs_flat = llrs_arr.ravel()
    bits_flat = bits_arr.ravel()

    if llrs_flat.shape != bits_flat.shape:
        raise ValueError(
            f"llrs and tx_bits must have the same number of elements. "
            f"Got {llrs_flat.shape} vs {bits_flat.shape}."
        )

    # Signed LLR: positive when transmitted bit is 0, negative when bit is 1.
    # The argument to log2(1 + exp(·)) is: -LLR * (1 - 2*bit)
    # = -LLR when bit=0  (should be positive → small loss)
    # = +LLR when bit=1  (should be negative → small loss at high SNR)
    x = -llrs_flat * (1.0 - 2.0 * bits_flat)

    # Numerically stable softplus: log2(1 + exp(x)) = log1p(exp(-|x|))/ln2 + max(0,x)/ln2
    ln2 = xp.log(xp.asarray(2.0, dtype=xp.float64))
    softplus = (xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(xp.asarray(0.0, dtype=xp.float64), x)) / ln2

    # GMI = Σ_{b=0}^{k-1} (1 - mean_n[softplus_b]) = k * (1 - mean_all[softplus])
    gmi_value = float(k * (1.0 - xp.mean(softplus)))
    logger.info(f"GMI: {gmi_value:.4f} b/cu")
    return gmi_value


def mi(
    symbols_rx: ArrayType,
    modulation: str,
    order: int,
    noise_var: float,
    pmf: Optional[np.ndarray] = None,
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
        Complex noise variance :math:`\sigma^2` of the AWGN model
        **referenced to the normalised constellation** (unit average power
        under the uniform distribution, i.e. the same scale as
        :func:`~commstools.mapping.gray_constellation`).
        For unit-power symbols at :math:`E_s/N_0` (dB):
        :math:`\sigma^2 = 10^{-E_s/N_0 / 10}`.
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(M,)`` for PS-QAM. When provided, the
        non-uniform prior ``P(sₘ)`` is incorporated into the likelihood
        computation and the result is clipped to ``[0, H(X)]`` instead of
        ``[0, log₂M]``. Pass ``signal.ps_pmf`` for PS-QAM signals.
        ``None`` (default) assumes uniform prior.

    Returns
    -------
    float
        MI in bits per channel use (b/cu). In the range ``[0, H(X)]``
        where ``H(X) = log₂M`` for uniform and ``H(X) < log₂M`` for PS.

    .. warning:: **PS-QAM scale convention**

        ``symbols_rx`` must be on the **same scale** as the normalised
        :func:`~commstools.mapping.gray_constellation` (unit average power
        under the uniform distribution).

        :meth:`~commstools.core.Signal.ps_qam` transmits at unit symbol
        power (``shape_pulse`` normalises), so samples from
        :attr:`~commstools.core.Signal.samples` and from
        :func:`~commstools.impairments.apply_awgn` are already on the
        correct scale and can be passed directly.

        However, after :meth:`~commstools.core.Signal.resolve_symbols`
        the receiver re-normalises to unit average power, which shifts the
        PS symbols from the :math:`\{s_m\}` grid to
        :math:`\{c \cdot s_m\}` (where :math:`c = 1/\sqrt{E_{PS}}`).
        Passing ``resolved_symbols`` directly will give slightly incorrect
        MI values.  Use :meth:`~commstools.core.Signal.mi` instead —
        it applies the :math:`E_{PS}` scale correction automatically.

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

    rx, xp, _ = dispatch(symbols_rx)
    rx = rx.ravel().astype(xp.complex128)             # (N,)
    constellation = xp.asarray(
        gray_constellation(modulation, order), dtype=xp.complex128
    )                                                  # (M,)
    M = len(constellation)
    ln2 = float(xp.log(xp.asarray(2.0, dtype=xp.float64)))
    log2_M = float(xp.log2(xp.asarray(float(M), dtype=xp.float64)))

    # Prior: uniform or PS-shaped
    if pmf is not None:
        pmf_arr = np.asarray(pmf, dtype=np.float64).clip(1e-300, None)
        log_prior = xp.asarray(np.log(pmf_arr), dtype=xp.float64)  # (M,) nats
        nz = pmf_arr > 0
        h_x_bits = float(-np.sum(pmf_arr[nz] * np.log2(pmf_arr[nz])))
    else:
        log_prior = xp.full(M, -xp.log(xp.asarray(float(M), dtype=xp.float64)),
                            dtype=xp.float64)          # log(1/M)
        h_x_bits = log2_M

    # Log-joint: log P(sₘ) + log p(r | sₘ)   (drop noise-normalisation constant)
    # Shape: (N, M)
    diff = rx[:, None] - constellation[None, :]        # (N, M)
    log_liks = log_prior[None, :] - (diff.real**2 + diff.imag**2) / noise_var

    # log p(s_m | r) via log-sum-exp for numerical stability
    lse_shift = log_liks.max(axis=1, keepdims=True)    # (N, 1)
    log_sum = xp.log(xp.sum(xp.exp(log_liks - lse_shift), axis=1)) + lse_shift[:, 0]

    log_posterior = log_liks - log_sum[:, None]        # (N, M), log base e
    posterior = xp.exp(log_posterior)                  # (N, M), sum-to-1 per row

    # MI = H(X) + (1/N) Σ_n Σ_m p(s_m|r_n) · log2 p(s_m|r_n)
    mi_nats = float(xp.sum(posterior * log_posterior, axis=1).mean())
    mi_value = float(np.clip(h_x_bits + mi_nats / ln2, 0.0, h_x_bits))

    logger.info(f"MI: {mi_value:.4f} b/cu  (max {h_x_bits:.2f} b/cu)")
    return mi_value
