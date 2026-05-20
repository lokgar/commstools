"""
Frequency offset estimation and correction utilities.

This module provides routines for carrier frequency offset (FOE) estimation
and correction, including blind spectral methods, multi-lag autocorrelation,
pilot-aided estimation, blockwise time-varying FOE, and exact complex mixing
correction.

Functions
---------
estimate_frequency_offset_mth_power :
    Blind FOE via M-th power spectral method with Jacobsen sub-bin interpolation.
estimate_frequency_offset_mengali_morelli :
    Blind or data-aided FOE via multi-lag autocorrelation (Mengali-Morelli); lock range
    extends to full Nyquist [-fs/2, fs/2] regardless of block length.
estimate_frequency_offset_pilots :
    Scattered-pilot FOE via (optionally SNR-weighted) least-squares phase slope fitting.
estimate_frequency_offset_blockwise :
    Time-varying FOE via sliding-window approach with cubic-spline interpolation.
correct_frequency_offset :
    Applies frequency offset correction via exact complex mixing (no bin quantisation).
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch, to_device
from .logger import logger


def _modulation_power_m(modulation: str, order: int) -> int:
    """
    Return the exponent M for M-th power spectral methods.

    Parameters
    ----------
    modulation : str
        Modulation type string (case-insensitive).
    order : int
        Modulation order.

    Returns
    -------
    int
        M = ``order`` for PSK; M = 4 for QAM and other schemes.

    Notes
    -----
    M=4 is exact only for **square** QAM constellations (4, 16, 64, 256, …)
    which have perfect 4-fold rotational symmetry.  For cross-QAM (32, 128,
    512-QAM) the 4th power leaves residual modulation spurs; a warning is
    emitted.  For PAM/ASK the M-th power law does not apply; prefer
    pilot-aided or data-aided estimators.
    """
    mod = modulation.lower()
    if "psk" in mod:
        if order > 4:
            logger.warning(
                f"{order}-PSK: M={order}th-power raises noise variance by M² — "
                "VV/FOE reliability degrades severely for order > 4. "
                "Prefer BPS or pilot-aided CPR for 8-PSK and higher."
            )
        return order  # M-th power exactly removes M-PSK modulation

    if "qam" in mod:
        side = int(order**0.5)
        if side * side == order:
            return 4  # Square QAM: 4-fold rotational symmetry, 4th power is exact
        # Cross-QAM (32, 128, 512-QAM): 4-fold symmetry is only approximate
        logger.warning(
            f"{order}-QAM is not square — 4th-power FOE/CPR will have residual "
            "modulation spurs. Prefer pilot-aided or data-aided estimation."
        )
        return 4

    # PAM, ASK, or unrecognised scheme
    logger.warning(
        f"Modulation '{modulation}' (order {order}): M=4 is a heuristic. "
        "4th-power methods are unreliable for non-QAM/PSK formats. "
        "Prefer pilot-aided or data-aided estimation."
    )
    return 4


# Lazy-compiled Numba kernel for the M&M iterative bootstrap.
_NUMBA_MM: dict = {}


def _get_numba_mm_bootstrap():
    """JIT-compile and cache the Numba M&M iterative bootstrap kernel.

    Returns
    -------
    callable
        Numba-compiled ``_mm_bootstrap_loop``.
    """
    if "mm" not in _NUMBA_MM:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _mm_bootstrap_loop(theta, amp, M_val, fs):
            """Iterative Mengali-Morelli bootstrap compiled to machine code.

            Predicts each lag's phase from the running weighted frequency
            estimate accumulated from all previous lags, then folds it
            into the weighted sum.  Sequential data dependency prevents
            vectorisation; Numba removes Python-interpreter overhead.

            Parameters
            ----------
            theta : (L,) float64
                Wrapped phase of R[m] at each lag (output of np.angle).
            amp : (L,) float64
                Magnitude |R[m]| at each lag (output of np.abs).
            M_val : float64
                Modulation power (1 for data-aided/generic, order for PSK,
                4 for QAM).
            fs : float64
                Sampling rate in Hz.

            Returns
            -------
            float64
                Estimated frequency offset in Hz.
            """
            two_pi = 2.0 * np.pi
            L = len(theta)

            # Lag 1 initialisation (m=1, m²=1)
            Theta_0 = theta[0]
            w0 = amp[0] * amp[0]  # m² · |R|² at m=1
            f_hat = Theta_0 * fs / (two_pi * M_val)
            w_sum = w0 if w0 > 1e-30 else 1e-30
            wf_sum = w_sum * f_hat

            for m_idx in range(1, L):
                m_val = float(m_idx + 1)
                predicted = two_pi * f_hat * M_val * m_val / fs
                diff = predicted - theta[m_idx]
                # round() is the C math round — unboxed, no Python overhead
                correction = round(diff / two_pi)
                Theta_m = theta[m_idx] + two_pi * correction
                f_m = Theta_m * fs / (two_pi * m_val * M_val)
                w_m = m_val * m_val * amp[m_idx] * amp[m_idx]
                w_sum += w_m
                wf_sum += w_m * f_m
                f_hat = wf_sum / w_sum

            return f_hat

        _NUMBA_MM["mm"] = _mm_bootstrap_loop

    return _NUMBA_MM["mm"]


def estimate_frequency_offset_mth_power(
    samples: ArrayType,
    sampling_rate: float,
    modulation: str,
    order: int,
    search_range: Optional[Tuple[float, float]] = None,
    nfft: Optional[int] = None,
    interpolation: str = "jacobsen",
    combine_channels: bool = False,
    debug_plot: bool = False,
) -> Union[float, np.ndarray]:
    """
    Estimates frequency offset using the M-th power law (nonlinear spectral method).

    Raises the signal to the M-th power to eliminate PSK/QAM modulation,
    producing a tone at M·Δf.  A spectral peak search with sub-bin
    interpolation gives frequency resolution well below the FFT bin width.

    Parameters
    ----------
    samples : array_like
        Complex IQ samples. Shape: (N,) or (C, N). For MIMO, channel
        magnitude spectra are summed for robust peak detection, then
        per-channel sub-bin interpolation is applied at the shared peak bin.
    sampling_rate : float
        Sampling rate in Hz.
    modulation : str
        Modulation scheme (case-insensitive): 'psk', 'qam', 'bpsk', etc.
    order : int
        Modulation order (2, 4, 16, 64, ...).
    search_range : tuple of float, optional
        ``(f_min, f_max)`` in Hz to limit the frequency offset search.
        The spectral search is mapped to ``[M·f_min, M·f_max]``.
        Default: full spectrum.
    nfft : int, optional
        FFT size. Default: next power of 2 ≥ len(samples).
    interpolation : {'jacobsen', 'parabolic'}, default 'jacobsen'
        Sub-bin interpolation method.

        * ``'jacobsen'``: uses complex FFT values around the peak bin —
          corrects rectangular-window sinc bias analytically.  More
          accurate than parabolic for short observation windows.
        * ``'parabolic'``: classic parabolic fit on FFT magnitudes.
    combine_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, return a single
        magnitude-weighted mean estimate as ``float``; if ``False``
        (default), return per-channel estimates as ``np.ndarray`` of
        shape ``(C,)``.  SISO inputs always return ``float``.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the M-th power
        spectrum per channel with the detected peak and sub-bin result.

    Returns
    -------
    float or np.ndarray
        Estimated frequency offset in Hz. For SISO or when
        ``combine_channels=True``: scalar ``float``. For MIMO with
        ``combine_channels=False``: ``np.ndarray`` of shape ``(C,)``
        with one estimate per channel.

    Notes
    -----
    M is determined by the modulation type:

    - PSK / BPSK: M = ``order`` (e.g. 4 for QPSK, 8 for 8-PSK).
    - QAM: M = 4 (the 4th power removes quadrature phase; residual
      amplitude modulation is suppressed by subtracting the per-channel
      mean of ``signal^M`` before the FFT).

    **Lock range:** ``[-fs/(2M), fs/(2M)]``. For QPSK at 1 GHz → ±125 MHz.
    Use ``search_range`` to reduce false-peak probability.

    **Jacobsen interpolation** (Jacobsen & Kootsookos, IEEE Signal Process.
    Mag., 2007):

    .. math::

        \\delta = \\operatorname{Re}\\!\\left[
            \\frac{X[k-1] - X[k+1]}{2X[k] - X[k-1] - X[k+1]}
        \\right]

    where *X* are complex FFT values at the peak bin *k* and its
    neighbours.  Unlike the parabolic fit (which operates on magnitudes
    and has a sinc-function bias for small NFFT), the Jacobsen estimator
    is unbiased for a rectangular window.

    References
    ----------
    M. Luise and R. Reggiannini, "Carrier frequency recovery in all-digital
    modems for burst-mode transmissions," IEEE Trans. Commun., 1995.

    E. Jacobsen and P. Kootsookos, "Fast, accurate frequency estimators,"
    IEEE Signal Process. Mag., vol. 24, no. 3, pp. 123-125, May 2007.
    """
    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]  # (1, N)
    C, N = samples.shape

    if N < 8:
        raise ValueError(
            f"Signal too short for spectral FOE (N={N}). Minimum 8 samples required."
        )

    M = _modulation_power_m(modulation, order)
    mod_lower = modulation.lower()

    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(N)))  # guaranteed Python int

    # Promote for numerical accuracy during power computation: (C, N)
    s_c = samples.astype(
        xp.complex128 if samples.dtype == xp.complex64 else samples.dtype
    )

    # M-th power removes modulation → tone at M·Δf; shape stays (C, N)
    x_M = s_c**M

    # For QAM: subtract per-channel mean to suppress DC spike from amplitude residuals
    if M == 4 and "qam" in mod_lower:
        x_M = x_M - xp.mean(x_M, axis=-1, keepdims=True)

    # Batched FFT across all channels: single kernel call on GPU → (C, nfft)
    X_M = xp.fft.fft(x_M, n=nfft, axis=-1)
    # CPU freq array; device version allocated only when search_range masking is needed.
    freqs_np = np.fft.fftfreq(nfft, d=1.0 / sampling_rate)  # (nfft,) — always on CPU

    mag = xp.abs(X_M)  # (C, nfft)

    # Restrict search to [M·f_min, M·f_max] when search_range is given;
    # do this BEFORE zeroing DC so the mask operates on the raw spectrum.
    if search_range is not None:
        tone_lo = M * min(search_range)
        tone_hi = M * max(search_range)
        freqs = xp.asarray(freqs_np)  # device version only when mask is needed
        mask = (freqs >= tone_lo) & (freqs <= tone_hi)  # (nfft,)
        if not bool(xp.any(mask)):
            raise ValueError(
                f"search_range {search_range} Hz produces an empty search "
                f"window in the M={M} scaled spectrum."
            )
        mag = xp.where(mask[None, :], mag, xp.zeros_like(mag))

    # Zero DC after masking (Δf = 0 is degenerate; also removes any residual
    # QAM DC not caught by mean subtraction above).
    mag[:, 0] = 0.0

    # For MIMO, sum spectra across channels for coherent accumulation.
    # For SISO this is a no-op (C=1).
    mag_combined = xp.sum(mag, axis=0)  # (nfft,)

    k_peak = int(xp.argmax(mag_combined))
    k_safe = max(1, min(k_peak, nfft - 2))

    # Per-channel sub-bin interpolation at the shared global peak bin.
    # Using the combined-magnitude peak for robustness (coherent noise averaging);
    # per-channel complex/magnitude values give the channel-specific fractional correction.
    f_per_ch = []
    for c in range(C):
        if interpolation == "jacobsen":
            xa_c = complex(X_M[c, k_safe - 1])
            xb_c = complex(X_M[c, k_safe])
            xc_c = complex(X_M[c, k_safe + 1])
            d_c = 2 * xb_c - xa_c - xc_c
            mu_c = float(((xa_c - xc_c) / d_c).real) if abs(d_c) > 1e-30 else 0.0
        else:
            a_c = float(mag[c, k_safe - 1])
            b_c = float(mag[c, k_safe])
            cc_ = float(mag[c, k_safe + 1])
            d_c = a_c - 2 * b_c + cc_
            mu_c = 0.5 * (a_c - cc_) / d_c if abs(d_c) > 1e-15 else 0.0
        mu_c = max(-0.5, min(0.5, mu_c))
        f_per_ch.append((freqs_np[k_safe] + mu_c * (sampling_rate / nfft)) / M)

    logger.info(
        f"FOE (M-th power, M={M}): {[f'{f:.2f}' for f in f_per_ch]} Hz "
        f"[nfft={nfft}, interp={interpolation}, search_range={search_range}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.frequency_offset_spectrum(
            mag_spectrum=to_device(mag, "cpu"),
            freqs=freqs_np,
            M=M,
            k_peaks=np.array([k_peak] * C),
            f_estimates=f_per_ch,
            search_range=search_range,
            show=True,
        )

    if was_1d:
        return float(f_per_ch[0])
    if combine_channels:
        weights = [float(mag[c, k_peak]) for c in range(C)]
        combined = float(np.average(f_per_ch, weights=weights))
        logger.info(
            f"FOE (M-th power, M={M}): combined={combined:.2f} Hz "
            f"(magnitude-weighted mean of {C} channels)"
        )
        return combined
    return np.array(f_per_ch)


def estimate_frequency_offset_mengali_morelli(
    samples: ArrayType,
    sampling_rate: float,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    ref_signal: Optional[ArrayType] = None,
    max_lag: Optional[int] = None,
    combine_channels: bool = False,
    debug_plot: bool = False,
) -> Union[float, np.ndarray]:
    """
    Estimates frequency offset via the Mengali-Morelli multi-lag autocorrelation.

    Uses multiple autocorrelation lags m = 1 … L combined with MVUE weights
    to extend the lock range to the full Nyquist interval ``[-fs/2, fs/2]``
    while remaining Cramér-Rao-efficient.  This is the recommended estimator
    when the frequency offset may be large (exceeding the Kay / differential
    lock range of ``fs/(2M)``) and a pilot or data-aided reference is
    available for pre-processing.

    Three input modes:

    1. **Data-aided** (``ref_signal`` provided): derotates samples by the
       known reference before estimating.
    2. **Blind M-PSK/QAM** (``modulation`` + ``order``): applies M-th power
       pre-processing.  Lock range remains ``[-fs/2, fs/2]`` for all lags
       after bootstrap unwrapping from lag 1.
    3. **Generic blind** (no arguments): assumes a constant-envelope signal.

    Parameters
    ----------
    samples : array_like
        Complex IQ samples. Shape: (N,) or (C, N).
    sampling_rate : float
        Sampling rate in Hz.
    modulation : str, optional
        Modulation type (case-insensitive). Required for blind M-th power mode.
        Ignored when ``ref_signal`` is provided.
    order : int, optional
        Modulation order. Required with ``modulation`` for blind mode.
    ref_signal : array_like, optional
        Ideal transmitted waveform used to derotate ``samples`` before
        autocorrelation.  The function computes ``y[n] = samples[n] *
        conj(ref[n])``, which cancels the data modulation and leaves a
        complex tone at Δf.

        **What to pass:** the noiseless baseband waveform as it would
        appear at the receiver input *without* any frequency offset or
        carrier phase — i.e. the output of your pulse shaper / DAC model,
        at the same sampling rate and the same number of samples-per-symbol
        as ``samples``.  Concretely:

        * If ``samples`` is pulse-shaped at ``sps`` samples/symbol, pass
          the corresponding pulse-shaped reference (e.g. ``Signal.samples``
          from a freshly generated :class:`~commstools.core.Signal` with no
          impairments).
        * If ``samples`` has already been matched-filtered and decimated to
          1 SPS, pass the 1-SPS symbol sequence (``Signal.source_symbols``).
        * Do **not** pass raw bits or a preamble sequence of different
          length — ``ref_signal`` must be sample-aligned and have the same
          length ``N`` as ``samples``.

        Amplitude normalisation does not affect the estimate (only the
        phase of each lag is used), so there is no need to normalise
        ``ref_signal`` to unit power.

        Shape: ``(N,)``, ``(1, N)`` (broadcast to all MIMO channels), or
        ``(C, N)`` for independent per-channel references.

        .. warning::
            **Sample-exact timing alignment is required.**  The product
            ``samples[n] * conj(ref[n])`` must refer to the *same* symbol
            period at every index ``n``.  A single-sample misalignment
            causes the autocorrelation to average over mismatched symbol
            pairs, collapsing the phase ramp to ≈ 0 Hz for i.i.d. data.
            Call :func:`estimate_timing` and trim / interpolate ``samples``
            to integer-sample alignment before passing to this function.
    max_lag : int, optional
        Maximum autocorrelation lag L.  Default: ``N // 4``, clamped to
        ``[1, N // 2]``.  Increasing L improves noise averaging at the cost
        of using shorter sub-sequences for each lag.
    combine_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, return a single
        autocorrelation-energy-weighted mean estimate as ``float``; if
        ``False`` (default), return per-channel estimates as
        ``np.ndarray`` of shape ``(C,)``.
        SISO inputs always return ``float``.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing per-channel
        autocorrelation magnitude ``|R[m]|`` and wrapped phase ``∠R[m]``
        vs lag, with the expected phase ramp overlaid.

    Returns
    -------
    float or np.ndarray
        Estimated frequency offset in Hz. For SISO or when
        ``combine_channels=True``: scalar ``float``. For MIMO with
        ``combine_channels=False``: ``np.ndarray`` of shape ``(C,)``
        with one estimate per channel.

    Notes
    -----
    **Algorithm** (Mengali & Morelli, 1997):

    1. Pre-process signal ``y`` (data-aided, blind M-th power, or generic blind).

    2. Compute normalised autocorrelation at lags m = 1 … L:

       .. math::

           R[m] = \\frac{1}{N-m} \\sum_{n=0}^{N-1-m} y^*[n]\\, y[n+m]

    3. Bootstrap phase from lag 1:
       :math:`\\theta_1 = \\angle R[1]` gives a coarse estimate that is
       unambiguous within ``[-fs/(2M), fs/(2M)]`` (or ``[-fs/2, fs/2]``
       for data-aided / generic mode).

    4. Unwrap higher lags using the lag-1 prediction:

       .. math::

           \\Theta[m] = \\angle R[m]
               + 2\\pi \\cdot \\operatorname{round}\\!
                 \\left(\\frac{m\\,\\theta_1 - \\angle R[m]}{2\\pi}\\right)

       This extends the effective lock range of every lag to
       ``[-fs/(2M), fs/(2M)]`` regardless of lag number.

    5. Per-lag frequency estimate:
       :math:`f[m] = \\Theta[m] \\cdot f_s / (2\\pi m)`

    6. Combine with SNR-magnitude weights
       :math:`w[m] \\propto m^2 |R[m]|^2` — upweights high lags for variance
       reduction while discarding lags where amplitude-modulation residuals
       (e.g. QAM) or noise degrade the autocorrelation:

       .. math::

           \\hat{f} = \\frac{\\sum_{m=1}^{L} m^2 |R[m]|^2\\, f[m]}
                            {\\sum_{m=1}^{L} m^2 |R[m]|^2}

    For MIMO, the bootstrap is run independently per channel; the result
    is a per-channel array by default.

    **Lock range:** ``[-fs/(2M), fs/(2M)]`` for blind M-th power mode;
    ``[-fs/2, fs/2]`` (full Nyquist) for data-aided or generic blind mode.

    References
    ----------
    U. Mengali and M. Morelli, "Data-aided frequency estimation for burst
    digital transmission," *IEEE Trans. Commun.*, vol. 45, no. 1,
    pp. 23-25, Jan. 1997.

    S. M. Kay, "A fast and accurate single-frequency estimator," *IEEE
    Trans. Acoust. Speech Signal Process.*, vol. 37, no. 12, Dec. 1989.
    """
    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]  # (1, N)
    C, N = samples.shape

    # Pre-process: data-aided, blind M-th power, or generic blind
    if ref_signal is not None:
        ref, _, _ = dispatch(ref_signal)
        if ref.ndim == 1:
            ref = ref[None, :]
        ref = xp.asarray(ref)
        y = samples * xp.conj(ref)  # derotate → complex tone at Δf
        M = 1
    elif modulation is not None and order is not None:
        M = _modulation_power_m(modulation, order)
        y = samples**M  # removes PSK/QAM modulation
    else:
        y = samples
        M = 1

    # Choose max_lag L
    L = max_lag if max_lag is not None else N // 4
    L = max(1, min(L, N // 2))

    # Autocorrelation at all lags 1..L via the Wiener-Khinchin theorem.
    # IFFT(|FFT(y)|²)[l] = Σ_n conj(y[n]) · y[n+l]  (linear, not circular,
    # when zero-padded to nfft_r ≥ N+L).
    # This replaces a Python loop of L GPU kernel launches with 2 FFT calls,
    # keeping all heavy computation on the device (GPU or CPU backend).
    nfft_r = 1 << int(np.ceil(np.log2(N + L)))  # smallest power-of-2 ≥ N+L
    Y_r = xp.fft.fft(y, n=nfft_r, axis=-1)  # (C, nfft_r)
    R_all = xp.fft.ifft(Y_r * xp.conj(Y_r), axis=-1)  # (C, nfft_r) per-ch autocorr

    # Unbiased per-channel autocorrelation at lags 1..L: R_per_ch[c, m-1] = R[c, m] / (N-m)
    lags_xp = xp.arange(1, L + 1, dtype=xp.float64)  # (L,) on device
    R_per_ch = R_all[:, 1 : L + 1] / (N - lags_xp[None, :])  # (C, L)

    # Transfer R_per_ch (C, L) to CPU once; run the iterative bootstrap per channel.
    # The bootstrap is a sequential scan (each step depends on the previous),
    # so Numba on CPU beats GPU here. Single transfer avoids repeated device round-trips.
    R_per_ch_np = to_device(R_per_ch, "cpu")  # (C, L) complex128

    _mm_kernel = _get_numba_mm_bootstrap()
    f_per_ch = []
    for c_idx in range(C):
        R_c_np = R_per_ch_np[c_idx]  # (L,)
        theta_c = np.angle(R_c_np)
        amp_c = np.abs(R_c_np)
        f_per_ch.append(
            float(_mm_kernel(theta_c, amp_c, float(M), float(sampling_rate)))
        )

    mode_str = "data-aided" if ref_signal is not None else f"blind M={M}"
    logger.info(
        f"FOE (Mengali-Morelli, {mode_str}): {[f'{f:.2f}' for f in f_per_ch]} Hz "
        f"[L={L} lags, N={N}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.mm_autocorrelation(
            R_np=R_per_ch_np,
            f_est=f_per_ch,
            sampling_rate=sampling_rate,
            M=M,
            show=True,
        )

    if was_1d:
        return float(f_per_ch[0])
    if combine_channels:
        # Weight by total autocorrelation energy per channel
        weights = [float(np.sum(np.abs(R_per_ch_np[c]) ** 2)) for c in range(C)]
        combined = float(np.average(f_per_ch, weights=weights))
        logger.info(
            f"FOE (Mengali-Morelli, {mode_str}): combined={combined:.2f} Hz "
            f"(autocorrelation-weighted mean of {C} channels)"
        )
        return combined
    return np.array(f_per_ch)


def estimate_frequency_offset_pilots(
    samples: ArrayType,
    sampling_rate: float,
    pilot_indices: ArrayType,
    pilot_values: ArrayType,
    snr_weighted: bool = True,
    combine_channels: bool = False,
    debug_plot: bool = False,
) -> Union[float, np.ndarray]:
    """
    Estimates frequency offset from pilot symbols via phase slope fitting.

    Extracts the received phase at each pilot position, demodulates against
    the known pilot values to obtain the residual phase, unwraps the pilot
    phase sequence, then fits a (optionally SNR-weighted) least-squares line
    to the unwrapped phase as a function of pilot sample time.  The slope
    gives the frequency offset: ``Δf = slope / (2π)``.

    Parameters
    ----------
    samples : array_like
        Received complex samples. Shape: (N,) or (C, N).
    sampling_rate : float
        Sampling rate in Hz.
    pilot_indices : array_like of int
        Sample indices of pilot positions in increasing order. Shape: (P,).
        Must be unique and sorted.  Supports any pilot arrangement:

        * **Comb (scattered):** uniform grid, e.g. every 16th sample.
          Lock range determined by comb spacing.
        * **Block (contiguous cluster):** e.g. ``[0, 1, ..., L-1]``.
        * **Multi-block:** e.g. a front preamble and a mid-burst pilot
          cluster.  Lock range is set by the **largest gap** between any
          two consecutive pilot indices (see Notes).
    pilot_values : array_like
        Known transmitted pilot symbols at the corresponding indices.
        Shape: (P,) for shared pilots (broadcast to all MIMO channels),
        or (C, P) for per-channel pilots.
    snr_weighted : bool, default True
        If ``True``, weights each pilot by its received power ``|r|²``
        (SNR proxy) in the least-squares phase-slope fit.  This is the
        **WLSQ** estimator and is significantly more robust when pilot SNR
        varies across the block (e.g. due to PMD nulls or spectral ripple).
        Set to ``False`` for the standard unweighted OLS slope.
    combine_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, return a single
        pilot-power-weighted mean estimate as ``float``; if ``False``
        (default), return per-channel estimates as ``np.ndarray`` of
        shape ``(C,)``.
        SISO inputs always return ``float``.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the unwrapped pilot
        phase sequence and the fitted frequency-slope line.

    Returns
    -------
    float or np.ndarray
        Estimated frequency offset in Hz. For SISO or when
        ``combine_channels=True``: scalar ``float``. For MIMO with
        ``combine_channels=False``: ``np.ndarray`` of shape ``(C,)``
        with one estimate per channel.

    Notes
    -----
    The demodulated pilot phase follows:

    .. math::

        \\hat{\\phi}[k] = 2\\pi \\Delta f \\cdot t_k + \\phi_0 + \\text{noise}

    where :math:`t_k = \\text{pilot\\_indices}[k] / f_s`.

    **Unweighted (OLS):** the minimum-variance unbiased estimator for
    equal-noise pilots (Tretter, 1985):

    .. math::

        \\hat{\\Delta f} = \\frac{1}{2\\pi}
            \\frac{\\sum_k (t_k - \\bar{t})(\\hat{\\phi}[k] - \\bar{\\phi})}
                  {\\sum_k (t_k - \\bar{t})^2}

    **SNR-weighted (WLSQ):** pilots are weighted by received power
    :math:`v_k = |r_k|^2` (normalised to unit mean), giving:

    .. math::

        \\hat{\\Delta f} = \\frac{1}{2\\pi}
            \\frac{\\sum_k v_k(t_k - \\bar{t}_v)(\\hat{\\phi}[k] - \\bar{\\phi}_v)}
                  {\\sum_k v_k(t_k - \\bar{t}_v)^2}

    where :math:`\\bar{t}_v` and :math:`\\bar{\\phi}_v` are the
    weighted means.  This reduces variance by 30-50 % when pilot SNR
    varies significantly across the burst.

    **Lock range:** ``xp.unwrap`` bridges each gap between consecutive
    pilot indices.  The gap that limits the lock range is the largest one:

    .. math::

        |\\Delta f| < \\frac{f_s}{2 \\cdot \\text{max\\_gap}}

    where ``max_gap`` is the maximum spacing (in samples) between any two
    consecutive entries of ``pilot_indices``.

    * Comb with period *d*: ``max_gap = d``, lock range ``= fs/(2d)``.
    * Two-block pilots with front block ``[0..L-1]`` and back block
      ``[N-L..N-1]``: ``max_gap = N-2L``.  For large *N*, this can be very
      small — use :func:`estimate_frequency_offset_mth_power` as coarse stage.

    References
    ----------
    S. A. Tretter, "Estimating the frequency of a noisy sinusoid by linear
    regression," *IEEE Trans. Inf. Theory*, vol. 31, no. 6, pp. 832-835,
    Nov. 1985.
    """
    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]
    C, N = samples.shape

    pilot_indices_np = to_device(pilot_indices, "cpu").astype(np.intp)
    pilot_values_xp = xp.asarray(pilot_values)
    P = len(pilot_indices_np)

    if pilot_values_xp.ndim == 1:
        pilot_values_xp = xp.broadcast_to(pilot_values_xp[None, :], (C, P))

    # Extract and demodulate: phase = angle(r · conj(s)) = 2π·Δf·t + φ₀ + noise
    r_pilots = samples[:, pilot_indices_np]  # (C, P)
    phi_pilots = xp.angle(r_pilots * xp.conj(pilot_values_xp))  # (C, P)

    # Unwrap in float64; cp.unwrap preserves input dtype so cast before calling
    phi_pilots_u = xp.unwrap(phi_pilots.astype(xp.float64), axis=-1)  # (C, P)

    t_xp = xp.asarray(pilot_indices_np.astype(np.float64)) / sampling_rate  # (P,)

    if snr_weighted:
        # WLSQ: weight each pilot by received power |r|² (averaged across channels).
        # Normalise to unit mean so weights don't affect the units of the slope.
        pwr = xp.mean(xp.abs(r_pilots) ** 2, axis=0)  # (P,) — mean over channels
        v = pwr / (xp.mean(pwr) + 1e-30)  # (P,) normalised weights

        # Weighted means
        v_sum = xp.sum(v)
        t_mean_v = xp.sum(v * t_xp) / v_sum  # scalar
        t_c = t_xp - t_mean_v  # (P,) centred
        phi_mean_v = (
            xp.sum(v[None, :] * phi_pilots_u, axis=-1, keepdims=True) / v_sum
        )  # (C,1)
        phi_c = phi_pilots_u - phi_mean_v  # (C, P)

        # Weighted normal equations: slope = Σ v·(t-t̄)·(φ-φ̄) / Σ v·(t-t̄)²
        t_var_w = float(xp.sum(v * t_c**2))  # scalar
        slopes = xp.sum(v[None, :] * phi_c * t_c[None, :], axis=-1) / t_var_w  # (C,)
    else:
        # Unweighted OLS: centered normal equations (Tretter 1985 MVUE).
        t_c = t_xp - xp.mean(t_xp)  # (P,) centred
        t_var = float(xp.dot(t_c, t_c))  # scalar Σ(t-t̄)²
        phi_c = phi_pilots_u - xp.mean(phi_pilots_u, axis=-1, keepdims=True)  # (C, P)
        slopes = xp.sum(phi_c * t_c[None, :], axis=-1) / t_var  # (C,)

    max_gap = int(np.max(np.diff(pilot_indices_np))) if P > 1 else 0
    lock_range = sampling_rate / (2 * max_gap) if max_gap > 0 else float("inf")
    wt_str = "WLSQ" if snr_weighted else "OLS"

    f_per_ch = [float(slopes[c]) / (2.0 * np.pi) for c in range(C)]

    logger.info(
        f"FOE (pilots, {wt_str}): {[f'{f:.2f}' for f in f_per_ch]} Hz "
        f"[P={P}, max_gap={max_gap} samples, lock_range=±{lock_range:.1f} Hz]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.pilot_phase_estimate(
            pilot_indices=pilot_indices_np,
            phi_pilots_u=to_device(phi_pilots_u, "cpu"),
            f_est=f_per_ch,
            sampling_rate=sampling_rate,
            show=True,
        )

    if was_1d:
        return float(f_per_ch[0])
    if combine_channels:
        # Weight by mean received pilot power per channel
        pwr_per_ch = to_device(xp.mean(xp.abs(r_pilots) ** 2, axis=-1), "cpu")  # (C,)
        weights = [float(pwr_per_ch[c]) for c in range(C)]
        combined = float(np.average(f_per_ch, weights=weights))
        logger.info(
            f"FOE (pilots, {wt_str}): combined={combined:.2f} Hz "
            f"(pilot-power-weighted mean of {C} channels)"
        )
        return combined
    return np.array(f_per_ch)


def estimate_frequency_offset_blockwise(
    samples: ArrayType,
    sampling_rate: float,
    block_size: int = 4096,
    overlap: float = 0.5,
    method: str = "mth_power",
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    debug_plot: bool = False,
) -> np.ndarray:
    r"""
    Estimates a time-varying frequency offset via a sliding-window approach.

    Divides the signal into overlapping blocks, runs a scalar FOE on each
    block, cubic-spline interpolates the block estimates to a dense per-sample
    grid, then integrates to obtain a phase trajectory suitable for
    :func:`correct_carrier_phase`.

    Parameters
    ----------
    samples : array_like
        Complex IQ samples. Shape: ``(N,)`` or ``(C, N)``. For MIMO only the
        first channel (row 0) is used for estimation; the returned phase array
        is 1-D regardless.
    sampling_rate : float
        Sampling rate in Hz.
    block_size : int, default 4096
        Number of samples per analysis block.
    overlap : float, default 0.5
        Fractional overlap between consecutive blocks (``[0, 1)``).  Block
        centers are spaced ``step = round(block_size * (1 - overlap))``
        samples apart.  Values ≥ 0.5 are recommended to avoid under-sampling
        fast frequency drifts.
    method : {"mth_power", "mengali_morelli"}, default "mth_power"
        Per-block estimator.

        * ``"mth_power"`` — M-th power spectral method with Jacobsen
          sub-bin interpolation (:func:`estimate_frequency_offset_mth_power`).
          Requires ``modulation`` and ``order``.
        * ``"mengali_morelli"`` — multi-lag autocorrelation MVUE
          (:func:`estimate_frequency_offset_mengali_morelli`).
          Requires ``modulation`` and ``order`` for blind operation.
    modulation : str, optional
        Modulation scheme (e.g. ``'qam'``, ``'psk'``).  Required for blind
        estimation with either method.
    order : int, optional
        Modulation order (e.g. 4, 16, 64).  Required alongside ``modulation``.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-block frequency
        estimates, the interpolated frequency trajectory, and the integrated
        phase trajectory.

    Returns
    -------
    np.ndarray, shape ``(N,)`` float64
        Per-sample phase trajectory :math:`\theta(n)` in radians.  Apply via::

            corrected = correct_carrier_phase(samples, theta)

        Positive :math:`\theta(n)` corresponds to a positive instantaneous
        frequency offset (carrier ahead of nominal).

    Notes
    -----
    **Pipeline:**

    1. Slice into overlapping blocks centered at
       :math:`t_k = \lfloor k \cdot \text{step} + \text{block\_size}/2 \rceil`
       for :math:`k = 0, 1, \ldots, B-1`.
    2. Run ``method`` on each block to obtain :math:`\Delta f[k]` in Hz.
    3. Interpolate :math:`\Delta f[k]` to a dense per-sample grid:

       * Interior (between first and last block centre): PCHIP spline.
       * Exterior (before first / after last block centre): constant clamp to
         the nearest edge estimate.  Extrapolation diverges rapidly for
         noisy block estimates and is avoided.
       * Fallback to linear interpolation when fewer than 4 blocks are
         available (cubic requires ≥ 4 nodes for numerical stability).

    4. Integrate:
       :math:`\theta(n) = \frac{2\pi}{f_s} \sum_{m=0}^{n} \Delta f_\text{dense}(m)`.

    **Minimum block count:** At least 2 blocks are required for interpolation.
    If the signal is shorter than ``2 * block_size * (1 - overlap)`` samples,
    the function falls back to a single-block global estimate.
    """
    if method not in ("mth_power", "mengali_morelli"):
        raise ValueError(
            f"method must be 'mth_power' or 'mengali_morelli', got {method!r}."
        )
    if method in ("mth_power", "mengali_morelli") and (
        modulation is None or order is None
    ):
        raise ValueError(
            f"method={method!r} requires modulation and order for blind estimation."
        )
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")

    samples_arr, _, _ = dispatch(samples)
    # Use first channel for MIMO inputs
    if samples_arr.ndim == 2:
        sig1d = to_device(samples_arr[0], "cpu")
    else:
        sig1d = to_device(samples_arr, "cpu")
    sig1d = np.asarray(sig1d)
    N = len(sig1d)

    step = max(1, round(block_size * (1.0 - overlap)))
    # Block start indices
    starts = list(range(0, N - block_size + 1, step))
    if not starts:
        # Signal shorter than one block: estimate over full signal
        starts = [0]
        block_size = N

    t_centers = np.array([s + block_size / 2.0 for s in starts], dtype=np.float64)
    df_estimates = np.empty(len(starts), dtype=np.float64)

    for k, s in enumerate(starts):
        block = sig1d[s : s + block_size]
        if method == "mth_power":
            est = estimate_frequency_offset_mth_power(
                block, sampling_rate, modulation, order
            )
        else:
            est = estimate_frequency_offset_mengali_morelli(
                block, sampling_rate, modulation=modulation, order=order
            )
        df_estimates[k] = float(est)

    # Interpolate to per-sample grid
    n_grid = np.arange(N, dtype=np.float64)
    n_blocks = len(t_centers)
    if n_blocks == 1:
        df_dense = np.full(N, df_estimates[0], dtype=np.float64)
    else:
        from scipy.interpolate import make_interp_spline  # noqa: PLC0415

        # Use cubic spline (k=3) for the interior when ≥ 4 nodes; fall back to
        # linear (k=1) for small block counts. Clamp n_grid to [t_centers[0],
        # t_centers[-1]] to avoid diverging polynomial extrapolation at the edges.
        k = 3 if n_blocks >= 4 else 1
        spline = make_interp_spline(t_centers, df_estimates, k=k)
        df_dense = spline(np.clip(n_grid, t_centers[0], t_centers[-1]))

    # Integrate: θ(n) = (2π / fs) * cumsum(Δf_dense)
    phase_trajectory = (2.0 * np.pi / sampling_rate) * np.cumsum(df_dense)

    logger.debug(
        f"FOE blockwise: {n_blocks} blocks, method={method}, "
        f"freq range=[{df_estimates.min():.2f}, {df_estimates.max():.2f}] Hz, "
        f"total phase drift={float(phase_trajectory[-1]):.3f} rad"
    )

    if debug_plot:
        from . import plotting as _plotting  # noqa: PLC0415

        _plotting.foe_blockwise_result(
            t_centers=t_centers,
            df_estimates=df_estimates,
            n_grid=n_grid,
            df_dense=df_dense,
            phase_trajectory=phase_trajectory,
            sampling_rate=sampling_rate,
            show=True,
        )

    return phase_trajectory


def correct_frequency_offset(
    samples: ArrayType,
    sampling_rate: float,
    offset: Union[float, np.ndarray],
) -> ArrayType:
    """
    Applies frequency offset correction by exact complex mixing.

    Unlike :func:`spectral.shift_frequency`, this function applies the
    correction **without bin quantization**, preserving the full precision
    of a sub-bin estimate (e.g. from parabolic interpolation in
    :func:`estimate_frequency_offset_mth_power`).

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: (..., N).
    sampling_rate : float
        Sampling rate in Hz.
    offset : float or np.ndarray
        Estimated frequency offset in Hz. Either a scalar (same correction
        applied to all channels) or a 1-D array of shape ``(C,)`` as
        returned by the per-channel ``estimate_frequency_offset_*``
        functions when ``combine_channels=False``.  A per-channel array
        requires ``samples`` to have shape ``(C, N)``.

    Returns
    -------
    array_like
        Frequency-corrected samples, same shape and dtype as input.
    """
    samples, xp, _ = dispatch(samples)
    offset_arr = xp.asarray(offset)
    per_channel = offset_arr.ndim >= 1 and offset_arr.size > 1

    if per_channel:
        logger.debug(
            f"Applying per-channel frequency offset correction: "
            f"{[f'{f:.4f}' for f in offset_arr.flat]} Hz "
            f"(sampling_rate={sampling_rate:.0f} Hz)"
        )
    else:
        logger.debug(
            f"Applying frequency offset correction: {float(offset_arr):.4f} Hz "
            f"(sampling_rate={sampling_rate:.0f} Hz)"
        )

    n = samples.shape[-1]
    t = xp.arange(n) / sampling_rate

    if xp.iscomplexobj(samples):
        target_dtype = samples.dtype
    else:
        target_dtype = xp.complex64 if samples.dtype == xp.float32 else xp.complex128

    if per_channel:
        # Build a (C, N) mixer — one distinct tone per channel
        C = samples.shape[0]
        offsets_xp = xp.asarray(offset_arr.reshape(-1)[:C], dtype=xp.float64)  # (C,)
        phase = -2.0 * xp.pi * offsets_xp[:, None] * t[None, :]  # (C, N)
        mixer = xp.exp(1j * phase).astype(target_dtype)
    else:
        # Scalar path — single mixer broadcast over all channels
        phase = -2.0 * xp.pi * float(offset_arr) * t
        mixer = xp.exp(1j * phase).astype(target_dtype)
        if samples.ndim > 1:
            mixer = mixer.reshape((1,) * (samples.ndim - 1) + (-1,))

    return samples * mixer
