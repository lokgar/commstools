"""
Synchronization and timing estimation utilities.

This module provides routines for time and frequency synchronization,
including the generation of optimal synchronization sequences (Barker,
Zadoff-Chu), robust timing estimation via cross-correlation,
and fine timing estimation and correction.

Functions
---------
barker_sequence :
    Generates Barker codes with optimal auto-correlation.
zadoff_chu_sequence :
    Generates Constant Amplitude Zero Auto-Correlation (CAZAC) sequences.
estimate_fractional_delay :
    Estimates sub-sample timing offset via parabolic interpolation.
fft_fractional_delay :
    Applies fractional sample delay using FFT-based frequency-domain method
    (ideal for bandlimited signals, perfect power preservation).
estimate_timing :
    Estimates coarse (integer) and fractional timing offsets via
    cross-correlation with a known reference sequence.
correct_timing :
    Combined coarse (integer) and fine (fractional) timing correction.
estimate_frequency_offset_mth_power :
    Blind FOE via M-th power spectral method with Jacobsen sub-bin interpolation.
estimate_frequency_offset_mengali_morelli :
    Blind or data-aided FOE via multi-lag autocorrelation (Mengali-Morelli); lock range
    extends to full Nyquist [-fs/2, fs/2] regardless of block length.
estimate_frequency_offset_pilots :
    Scattered-pilot FOE via (optionally SNR-weighted) least-squares phase slope fitting.
correct_frequency_offset :
    Applies frequency offset correction via complex mixing.
recover_carrier_phase_pll :
    Streaming CPR via Decision-Directed PLL (1st/2nd-order loop); Numba-compiled
    inner loop for CPU performance; GPU-transparent via CPU offload.
recover_carrier_phase_viterbi_viterbi :
    Block-based CPR via M-th power law (Viterbi-Viterbi) for PSK/QAM symbols.
recover_carrier_phase_bps :
    Blind Phase Search CPR for QAM constellations (Pfau et al.).
recover_carrier_phase_tikhonov :
    MAP CPR with Tikhonov/Wiener prior; RTS Kalman smoother on VV block phases.
recover_carrier_phase_pilots :
    Pilot-aided CPR with phase unwrapping and interpolation across the symbol grid.
    Single-carrier only; for OFDM CPE tracking see 5G NR PTRS / DVB-T2.
correct_carrier_phase :
    Applies per-symbol phase correction by complex rotation.
compensate_iq_imbalance_lowdin :
    Blind IQ imbalance compensation via Löwdin symmetric orthogonalisation.
compensate_iq_imbalance_gram_schmidt :
    Blind IQ imbalance compensation via Gram-Schmidt sequential orthogonalisation.
"""

from typing import Optional, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch, is_cupy_available, to_device
from .core import Preamble
from .logger import logger

# Window length for DFT-upsampling in estimate_fractional_delay()
_DFT_WINDOW = 33

# Standard Barker codes
_BARKER_SEQUENCES = {
    2: [1, -1],
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def barker_sequence(length: int) -> ArrayType:
    """
    Generates a Barker sequence of the specified length.

    Barker sequences are binary sequences (+1, -1) with optimal cyclic
    auto-correlation properties, where the sidelobes are at most 1. They are
    widely used for frame synchronization and pulse compression.

    Parameters
    ----------
    length : {2, 3, 4, 5, 7, 11, 13}
        Total length of the Barker sequence.

    Returns
    -------
    array_like
        BPSK symbols (+1.0, -1.0). Shape: (length,).
        Backend depends on system availability (CuPy if available, else NumPy).

    Raises
    ------
    ValueError
        If the requested length is not a valid Barker length.

    Examples
    --------
    >>> barker_sequence(7)
    array([ 1.,  1.,  1., -1., -1.,  1., -1.], dtype=float32)
    """
    if length not in _BARKER_SEQUENCES:
        valid = sorted(_BARKER_SEQUENCES.keys())
        raise ValueError(f"No Barker sequence of length {length}. Valid: {valid}")

    seq = np.array(_BARKER_SEQUENCES[length], dtype="float32")

    if is_cupy_available():
        seq = to_device(seq, "gpu")

    logger.debug(f"Generated Barker-{length} sequence.")
    return seq


def zadoff_chu_sequence(length: int, root: int = 1) -> ArrayType:
    r"""
    Generates a Zadoff-Chu (ZC) synchronization sequence.

    ZC sequences are Complex-valued, Constant Amplitude Zero
    Auto-Correlation (CAZAC) sequences. They possess the unique property
    that their periodic auto-correlation is zero at all non-zero lags,
    and their DFT is also a ZC sequence. This makes them ideal for
    timing and frequency synchronization in systems like LTE and 5G NR.

    Parameters
    ----------
    length : int
        The sequence length ($N_{ZC}$). For optimal cross-correlation
        properties, this should be a prime number.
    root : int, default 1
        The root index ($u$). Must be relatively prime to `length`.

    Returns
    -------
    array_like
        Complex Zadoff-Chu symbols of unit magnitude.
        Shape: (length,). Data type: `complex64`.

    Notes
    -----
    - For odd lengths: $x[n] = \exp(-j \frac{\pi u n (n+1)}{N_{ZC}})$
    - For even lengths: $x[n] = \exp(-j \frac{\pi u n^2}{N_{ZC}})$
    - ZC sequences have exceptionally low Peak-to-Average Power Ratio (PAPR).
    """
    if length < 1:
        raise ValueError("Length must be positive.")
    if root < 1 or root >= length:
        raise ValueError(f"Root must be in [1, {length - 1}].")

    # Determine backend
    if is_cupy_available():
        import cupy as cp

        xp = cp
    else:
        xp = np

    # ZC formula: x[n] = exp(-j * pi * u * n * (n+1) / N)
    n = xp.arange(length)
    if length % 2 == 0:
        # Even length: x[n] = exp(-j * pi * u * n^2 / N)
        seq = xp.exp(-1j * xp.pi * root * n * n / length)
    else:
        # Odd length: x[n] = exp(-j * pi * u * n * (n+1) / N)
        seq = xp.exp(-1j * xp.pi * root * n * (n + 1) / length)

    seq = seq.astype(xp.complex64)

    logger.debug(f"Generated ZC sequence: length={length}, root={root}.")
    return seq


def estimate_fractional_delay(
    correlation: ArrayType,
    peak_indices: ArrayType,
    dft_upsample: int = 1,
    method: str = "log-parabolic",
) -> ArrayType:
    """
    Estimates sub-sample timing offset via parabolic interpolation.

    Given a correlation array and the integer peak positions, fits a
    parabola (or Gaussian) through the three points around each peak to
    estimate the fractional offset with sub-sample precision.

    When the input is **complex**, the peak is phase-rotated to the real
    axis before fitting. This preserves the true peak shape (unlike
    fitting to ``|R|``) and improves accuracy by 2-5x for typical
    matched-filter outputs.

    If ``dft_upsample > 1``, performs a "Zoom FFT" (DFT zero-padding)
    around the peak to interpolate the correlation function onto a finer
    grid before fitting. This significantly reduces bias error.

    Parameters
    ----------
    correlation : array_like
        Correlation values — complex or real magnitude.
        Shape: ``(N,)`` or ``(C, N)``.
    peak_indices : array_like
        Integer peak positions. Shape: scalar or ``(C,)``.
    dft_upsample : int, default 1
        Upsampling factor for DFT-based interpolation.
        Values > 1 perform zero-padded FFT interpolation on a window
        around the peak (typically 33 samples).
    method : {'parabolic', 'log-parabolic'}, default 'log-parabolic'
        Fitting method:
        - 'parabolic': Standard parabolic fit. Good for general peaks.
        - 'log-parabolic': Fits a parabola to log(y), equivalent to a
          Gaussian fit. Often more accurate for bandlimited pulses.

    Returns
    -------
    array_like
        Fractional offset per channel, in [-0.5, 0.5). Shape: ``(C,)`` or scalar.
    """
    correlation, xp, _ = dispatch(correlation)
    peak_indices = xp.asarray(peak_indices)
    scalar_input = peak_indices.ndim == 0

    if correlation.ndim == 1:
        correlation = correlation[None, :]  # (1, N)
    if peak_indices.ndim == 0:
        peak_indices = peak_indices[None]  # (1,)

    N = correlation.shape[-1]
    C = correlation.shape[0]
    ch_idx = xp.arange(C)

    k = peak_indices.astype(int)
    mu = xp.zeros(C, dtype=correlation.real.dtype)

    def _calculate_mu(r_prev, r_curr, r_next, xp, method):
        if xp.iscomplexobj(r_curr):
            phase = xp.exp(-1j * xp.angle(r_curr))
            alpha = (r_prev * phase).real
            beta = (r_curr * phase).real
            gamma = (r_next * phase).real
        else:
            alpha = r_prev
            beta = r_curr
            gamma = r_next

        if method == "log-parabolic":
            eps = 1e-12
            alpha = xp.maximum(alpha, eps)
            beta = xp.maximum(beta, eps)
            gamma = xp.maximum(gamma, eps)
            alpha, beta, gamma = xp.log(alpha), xp.log(beta), xp.log(gamma)

        denom = 2.0 * (alpha - 2.0 * beta + gamma)
        safe_denom = xp.where(xp.abs(denom) > 1e-12, denom, xp.ones_like(denom))
        mu_val = (alpha - gamma) / safe_denom

        # Mask invalid fits
        valid = xp.abs(denom) > 1e-12
        mu_val = xp.where(valid, mu_val, xp.zeros_like(mu_val))
        return xp.clip(mu_val, -0.5, 0.5)

    half_W = _DFT_WINDOW // 2
    interior_mask = (k >= half_W) & (k < N - half_W)

    # -------------------------------------------------------------------------
    # Path A: DFT Upsampling
    # -------------------------------------------------------------------------
    if dft_upsample > 1:
        if xp.any(interior_mask):
            valid_indices = xp.where(interior_mask)[0]
            k_valid = k[interior_mask]

            offsets = xp.arange(-half_W, half_W + 1)
            gather_idx = k_valid[:, None] + offsets[None, :]
            windows = correlation[valid_indices[:, None], gather_idx]

            specs = xp.fft.fft(windows, axis=-1)
            pos_limit = (_DFT_WINDOW + 1) // 2
            neg_len = _DFT_WINDOW - pos_limit
            target_len = _DFT_WINDOW * dft_upsample
            padded_specs = xp.zeros((len(valid_indices), target_len), dtype=specs.dtype)
            padded_specs[:, :pos_limit] = specs[:, :pos_limit]
            padded_specs[:, -neg_len:] = specs[:, pos_limit:]

            upsampled = xp.fft.ifft(padded_specs, axis=-1) * dft_upsample
            up_mag = xp.abs(upsampled)
            k_up = xp.argmax(up_mag, axis=-1)
            k_up_safe = xp.clip(k_up, 1, target_len - 2)

            row_idx = xp.arange(len(valid_indices))
            r_prev = upsampled[row_idx, k_up_safe - 1]
            r_curr = upsampled[row_idx, k_up_safe]
            r_next = upsampled[row_idx, k_up_safe + 1]

            mu_up = _calculate_mu(r_prev, r_curr, r_next, xp, method)

            # Position relative to window start: k_up + mu_up
            # Center of window is at index (half_W * dft_upsample)?
            # No, standard ZoomFFT mapping.
            # Time axis of upsampled is 0 to W*M-1.
            # Original sample k corresponds to center of window.
            # Window covers [k - half_W, k + half_W].
            # Index 0 of upsampled corresponds to k - half_W.
            # So offset from k is: -half_W + (k_up + mu_up) / M.
            offset_samples = -half_W + (k_up_safe + mu_up) / dft_upsample
            mu[interior_mask] = offset_samples

    # -------------------------------------------------------------------------
    # Path B: Standard (Primary when M=1, fallback for edge channels when M>1)
    # -------------------------------------------------------------------------
    # The trigger is determined purely by which channels still need a result:
    #   - M=1: all channels (Path A was skipped entirely)
    #   - M>1: only ~interior_mask channels (edges Path A cannot process)
    # Do NOT use `mu == 0` as a proxy — zero is a valid fractional offset and
    # would wrongly overwrite DFT results for on-centre correlation peaks.
    if dft_upsample == 1:
        calc_mask = xp.ones(C, dtype=bool)
    else:
        calc_mask = ~interior_mask  # only edge channels need the fallback

    if xp.any(calc_mask):
        interior_all = (k >= 1) & (k < N - 1)
        k_all_safe = xp.clip(k, 1, N - 2)

        r_prev = correlation[ch_idx, k_all_safe - 1]
        r_curr = correlation[ch_idx, k_all_safe]
        r_next = correlation[ch_idx, k_all_safe + 1]

        mu_std = _calculate_mu(r_prev, r_curr, r_next, xp, method)
        mu_std = xp.where(interior_all, mu_std, xp.zeros_like(mu_std))

        if dft_upsample == 1:
            mu = mu_std
        else:
            # Merge: DFT result for interior channels, standard for edge channels
            mu = xp.where(interior_mask, mu, mu_std)

    if scalar_input:
        return mu[0]
    return mu


def fft_fractional_delay(
    samples: ArrayType,
    delay: Union[float, ArrayType],
) -> ArrayType:
    """
    Applies fractional sample delay using FFT-based frequency-domain method.

    This is the mathematically ideal method for bandlimited signals. It
    applies the delay as a phase shift in the frequency domain, which is
    equivalent to ideal sinc interpolation in the time domain. Unlike
    polynomial interpolators (e.g., Farrow), this method perfectly preserves
    signal power and has no bandwidth limitations.

    Parameters
    ----------
    samples : array_like
        Input signal. Shape: (N,) or (C, N).
    delay : float or array_like
        Fractional delay in samples. Positive = delay (shift right).
        Scalar applies the same delay to all channels.
        Array of shape (C,) applies per-channel delays.

    Returns
    -------
    array_like
        Delayed signal with the same shape as input.

    Notes
    -----
    The delay is applied in the frequency domain as:

    .. math::
        Y(f) = X(f) \\cdot e^{-j 2\\pi f \\cdot \\text{delay} / N}

    This is equivalent to ideal sinc interpolation and is the optimal
    fractional delay method for bandlimited signals.

    Advantages over polynomial interpolators (Farrow):
    - Perfect power preservation (0 dB loss)
    - No bandwidth limitations
    - Numerically stable
    - Ideal for communications signals

    References
    ----------
    T. I. Laakso et al., "Splitting the unit delay," IEEE Signal
    Processing Magazine, 1996.
    """
    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]  # (1, N)

    C, N = samples.shape

    # Convert delay to array
    if isinstance(delay, (int, float)):
        delay_arr = xp.full(C, delay, dtype=samples.real.dtype)
    else:
        delay_arr = xp.asarray(delay, dtype=samples.real.dtype)
        if delay_arr.ndim == 0:
            delay_arr = delay_arr[None]

    # FFT
    spec = xp.fft.fft(samples, axis=-1)

    # Frequency axis: normalized frequencies in cycles/sample
    freqs = xp.fft.fftfreq(N, d=1.0)

    # Phase shift: exp(-j * 2 * pi * f * delay)
    # Positive delay -> phase ramp that shifts signal to the right.
    # Computed at float64 accuracy, then cast to match spec's dtype to prevent
    # -2j * xp.pi (complex128) from promoting complex64 spectra.
    phase_shift = xp.exp(-2j * xp.pi * freqs[None, :] * delay_arr[:, None])
    phase_shift = phase_shift.astype(spec.dtype)

    # Apply phase shift
    spec_delayed = spec * phase_shift

    # IFFT
    result = xp.fft.ifft(spec_delayed, axis=-1)

    # Dtype restoration: mirrors the impairments.py pattern.
    if not xp.iscomplexobj(samples):
        # Real input: fractional delay is a real-valued operation
        result = result.real
    elif result.dtype != samples.dtype:
        # Complex input: ifft may return complex128 from complex64 input
        result = result.astype(samples.dtype)

    if was_1d:
        return result[0]
    return result


def estimate_timing(
    samples: ArrayType,
    reference: Optional[Union[ArrayType, "Preamble"]] = None,
    threshold: float = 3.0,
    sps: Optional[int] = None,
    pulse_shape: Optional[str] = None,
    filter_params: Optional[dict] = None,
    search_range: Optional[Tuple[int, int]] = None,
    dft_upsample: int = 1,
    fractional_method: str = "log-parabolic",
    debug_plot: bool = False,
) -> Tuple[ArrayType, ArrayType]:
    """
    Estimates coarse and fractional timing offsets via cross-correlation.

    Performs a sliding cross-correlation between the received signal and
    a known reference sequence to determine the coarse (integer sample)
    timing offset per channel.  Additionally estimates the fractional
    (sub-sample) timing offset using parabolic interpolation on the
    correlation peak.

    The reference can be:

    * A :class:`~commstools.core.Preamble` object — the waveform is
      reconstructed internally using ``sps``, ``pulse_shape``, and
      ``filter_params``.
    * A raw array — used directly as the correlation template.  This
      can be the original transmitted signal, a known training sequence,
      or any sub-sequence of the signal.  The array must be at the same
      sampling rate as ``samples``.

    For MIMO ZC preambles each TX stream uses a unique root (assigned by
    :func:`~commstools.helpers.zc_mimo_root`).  All templates are correlated
    against all RX channels; the Hungarian assignment algorithm maps each TX
    root to the RX channel it matches best, and only that template's
    correlation is used to find each channel's peak independently, so
    **hardware skew between RX channels is preserved in the returned
    per-channel offsets**.

    Parameters
    ----------
    samples : array_like
        Received signal samples. Shape: ``(N,)`` or ``(C, N)``.
    reference : array_like or Preamble
        Known reference sequence for correlation.

        * **Preamble object:** the waveform is reconstructed via
          ``Preamble.to_signal()`` using the provided ``sps``,
          ``pulse_shape``, and ``filter_params``.  ``sps`` is required
          in this mode.
        * **Raw array:** used directly as the correlation template at the
          current sampling rate.  Shape: ``(L,)`` for a single template
          or ``(C_tx, L)`` for per-channel templates (MIMO).
    threshold : float, default 3.0
        Detection threshold for the correlation metric, defined as the Peak-to-Average
        Power Ratio (PAPR) of the cross-correlation magnitude. A value >= 3.0 ensures
        reliable peak prominence above the noise floor.
    sps : int, optional
        Samples per symbol.  Required when ``reference`` is a
        :class:`~commstools.core.Preamble` object; ignored for raw arrays.
    pulse_shape : str, optional
        Pulse shaping filter type (e.g. ``'rrc'``).  Used only when
        ``reference`` is a Preamble object.  Defaults to ``'rrc'``.
    filter_params : dict, optional
        Additional filter parameters (``beta``, ``span``, etc.) passed
        to the pulse shaper.  Used only when ``reference`` is a Preamble
        object.  Defaults to ``{}``.
    search_range : tuple of int, optional
        A ``(start, end)`` sample range to restrict detection.
        Defaults to the full signal length.
    dft_upsample : int, default 1
        Factor for DFT-based upsampling of correlation peak.
        Use > 1 for high-precision fractional delay estimation.
    fractional_method : {'parabolic', 'log-parabolic'}, default 'log-parabolic'
        Fitting method for fractional delay estimation.
    debug_plot : bool, default False
        If ``True``, plots the correlation magnitude for debugging.

    Returns
    -------
    coarse_offsets : ArrayType
        Integer sample offsets where the reference sequence begins,
        per RX channel.  Shape: ``(N_channels,)``.  Each channel's
        offset is estimated independently so hardware skew between
        channels is preserved.
    fractional_offsets : ArrayType
        Sub-sample timing offsets in [-0.5, 0.5) per channel.
        Shape: ``(N_channels,)``.

    Raises
    ------
    ValueError
        If no correlation peak is found that satisfies the threshold
        criteria, or if ``reference`` is not provided.

    Notes
    -----
    - The returned coarse offset corresponds to the very first sample of
      the detected reference sequence.
    - **Synchronization Strategy**: For signals with oversampling (SPS > 1),
      correlating with a shaped/oversampled reference (e.g., generated via
      ``Preamble.to_signal(...)``) typically yields superior timing precision
      and SNR compared to correlating with 1 SPS symbols.  Ensure the
      ``reference`` argument matches the sampling rate of the input
      ``samples``.
    """
    from .helpers import cross_correlate_fft

    # 1. Resolve Inputs & Metadata
    if filter_params is None:
        filter_params = {}

    if not hasattr(samples, "ndim"):
        raise TypeError(
            f"estimate_timing() expects an array of samples, got {type(samples).__name__}."
        )

    sig_array, xp, _ = dispatch(samples)

    # 2. Build reference waveform
    ref_waveform = None

    if isinstance(reference, Preamble):
        # Preamble object: reconstruct the shaped waveform at the target SPS.
        if sps is None:
            raise ValueError("SPS must be provided when using a Preamble object.")

        ref_waveform = xp.asarray(
            reference.to_signal(
                sps=sps,
                symbol_rate=1.0,
                pulse_shape=pulse_shape or "rrc",
                **filter_params,
            ).samples
        )
        # ensure 2-D (C_tx, L*sps) for the correlation engine
        if ref_waveform.ndim == 1:
            ref_waveform = ref_waveform[None, :]

    elif reference is not None:
        # Raw array: use directly as the correlation template.
        ref_waveform = xp.asarray(reference)
    else:
        raise ValueError(
            "A 'reference' sequence must be provided (Preamble object or raw array)."
        )

    # 3. Correlation Strategy
    # Signal: (C, N) or (N,)
    # Reference: (C, L) or (L,) or (1, L)

    # Ensure dimensions match for broadcast/multichannel correlation
    if sig_array.ndim == 1:
        sig_array = sig_array[None, :]  # Treat as 1 channel

    if ref_waveform.ndim == 1:
        ref_waveform = ref_waveform[None, :]  # Treat as 1 template

    # Ensure reference is on the same device as the signal.
    ref_waveform = to_device(ref_waveform, "cpu" if xp is np else "gpu")

    num_sig_ch = sig_array.shape[0]

    # Apply search range
    offset = 0
    if search_range is not None:
        start, end = search_range
        sig_processing = sig_array[:, start:end]
        offset = start
    else:
        sig_processing = sig_array

    # === Vectorized Correlation (FFT) via shared helper ===
    L = ref_waveform.shape[-1]
    C_tx = ref_waveform.shape[0]

    if C_tx > 1:
        from scipy.optimize import linear_sum_assignment

        # MIMO unique-root: correlate every TX template against every RX channel.
        # corr_all[rx, tx, lag] — shape (C_rx, C_tx, N_lag)
        corr_all = xp.stack(
            [
                cross_correlate_fft(
                    sig_processing, ref_waveform[t : t + 1], mode="positive_lags"
                )
                for t in range(C_tx)
            ],
            axis=1,
        )

        # === Best-assignment peak finding ===
        corr_all_mag = xp.abs(corr_all)  # (C_rx, C_tx, N_lag)
        S_np = to_device(xp.max(corr_all_mag, axis=-1), "cpu")  # (C_rx, C_tx)
        _, assignment = linear_sum_assignment(-S_np)  # assignment[rx] = best tx

        # Per-channel correlation magnitude using only the assigned template.
        corr_incoherent = xp.stack(
            [corr_all_mag[rx, assignment[rx]] for rx in range(num_sig_ch)], axis=0
        )  # (C_rx, N_lag)
        peak_indices = xp.argmax(corr_incoherent, axis=-1)  # (C_rx,)
        # Assigned-template complex correlation for fractional delay interpolation.
        corr = xp.stack(
            [corr_all[rx, assignment[rx]] for rx in range(num_sig_ch)], axis=0
        )  # (C_rx, N_lag) complex
    else:
        corr = cross_correlate_fft(sig_processing, ref_waveform, mode="positive_lags")

    # Magnitude
    corr_mag = xp.abs(corr)

    # === Per-Channel Analysis ===
    # For SISO / broadcast single template: find peak from magnitude
    if C_tx == 1:
        peak_indices = xp.argmax(corr_mag, axis=-1)  # Shape (C,)

    # === Per-Channel Normalization ===
    # Normalized correlation metric: use local signal energy around each
    # detected peak (L-sample window) instead of global mean.  This gives
    # metric ≈ 1.0 for a perfectly matched reference regardless of the
    # noise/data content elsewhere in the signal.
    e_ref = xp.sum(xp.abs(ref_waveform) ** 2, axis=-1)  # (C_tx,) or scalar
    if e_ref.ndim == 0:
        e_ref = xp.full(num_sig_ch, e_ref)
    else:
        e_ref = xp.full(num_sig_ch, xp.mean(e_ref))

    # Local signal energy: sum |sig|² in the L-sample window at each peak
    N_sig = sig_processing.shape[-1]
    e_s = xp.empty(num_sig_ch, dtype=sig_processing.real.dtype)
    for ch in range(num_sig_ch):
        pk = int(peak_indices[ch])
        end_idx = min(pk + L, N_sig)
        e_s[ch] = xp.sum(xp.abs(sig_processing[ch, pk:end_idx]) ** 2)

    norm_factors = xp.sqrt(e_ref * e_s)
    norm_factors = xp.maximum(norm_factors, 1e-12)

    # Calculate per-channel coherence (diagnostic mathematically absolute bound, [0, 1])
    if C_tx > 1:
        peak_vals = xp.max(
            corr_incoherent, axis=-1
        )  # (C,) incoherent — matches peak detection
        mean_vals = xp.mean(corr_incoherent, axis=-1)
    else:
        peak_vals = xp.max(corr_mag, axis=-1)  # (C,)
        mean_vals = xp.mean(corr_mag, axis=-1)

    coherence = peak_vals / norm_factors
    coherence = xp.clip(coherence, 0.0, 1.0)

    # Primary timing metric: PAPR (Peak-to-Average magnitude)
    # This evaluates visual prominence against the noise floor, ensuring
    # extreme robustness to CFO phase-rotation and pulse-shape mismatch.
    mean_vals = xp.maximum(mean_vals, 1e-12)
    metrics = peak_vals / mean_vals

    # Log coherence diagnostics
    for ch in range(num_sig_ch):
        c_val = float(coherence[ch])
        p_val = float(metrics[ch])
        if c_val < 0.5 and p_val >= threshold:
            logger.warning(
                f"Channel {ch}: Peak phase coherence is very low ({c_val:.2f}), but "
                f"peak is visually prominent (PAPR={p_val:.1f} >= {threshold}). "
                f"This suggests strong Carrier Frequency Offset (CFO) or uncompensated "
                f"dispersion destroying phase alignment over the sequence length."
            )
        else:
            logger.debug(
                f"Channel {ch}: Peak prominence = {p_val:.1f}, coherence = {c_val:.2f}"
            )

    # === MIMO fallback: X-Y power imbalance with polarization mixing ===
    if C_tx > 1 and float(xp.max(metrics)) >= threshold:
        fallback_applied = False
        for rx in range(num_sig_ch):
            if float(metrics[rx]) >= threshold:
                continue
            best_alt_tx, best_alt_metric = None, float(metrics[rx])
            for tx in range(C_tx):
                if tx == int(assignment[rx]):
                    continue
                alt_peak = float(xp.max(corr_all_mag[rx, tx]))
                alt_mean = float(xp.mean(corr_all_mag[rx, tx]))
                alt_metric = alt_peak / max(alt_mean, 1e-12)
                if alt_metric > best_alt_metric:
                    best_alt_metric, best_alt_tx = alt_metric, tx
            if best_alt_tx is not None:
                logger.warning(
                    f"Channel {rx}: assigned ZC root {int(assignment[rx])} metric "
                    f"{float(metrics[rx]):.3f} < threshold {threshold}. "
                    f"Possible X-Y power imbalance. "
                    f"Falling back to ZC root {best_alt_tx} "
                    f"(metric {best_alt_metric:.3f})."
                )
                assignment[rx] = best_alt_tx
                corr_incoherent[rx] = corr_all_mag[rx, best_alt_tx]
                corr[rx] = corr_all[rx, best_alt_tx]
                peak_indices[rx] = xp.argmax(corr_incoherent[rx])
                fallback_applied = True
        if fallback_applied:
            # Recompute local energy and metrics for updated peaks
            for ch in range(num_sig_ch):
                pk = int(peak_indices[ch])
                end_idx = min(pk + L, N_sig)
                e_s[ch] = xp.sum(xp.abs(sig_processing[ch, pk:end_idx]) ** 2)
            norm_factors = xp.sqrt(e_ref * e_s)
            norm_factors = xp.maximum(norm_factors, 1e-12)
            peak_vals = xp.max(corr_incoherent, axis=-1)
            metrics = xp.clip(peak_vals / norm_factors, 0.0, 1.0)
            corr_mag = xp.abs(corr)

    # === Threshold Check ===
    max_metric = float(xp.max(metrics))
    if max_metric < threshold:
        raise ValueError(
            f"No correlation peak above threshold {threshold} (max: {max_metric:.3f})"
        )
    for _ch in range(num_sig_ch):
        _m = float(metrics[_ch])
        if _m < threshold:
            logger.warning(
                f"Channel {_ch}: correlation metric {_m:.3f} is below threshold "
                f"{threshold}. Coarse offset for this channel may be unreliable."
            )

    # === Skew Check (Robust) ===
    if num_sig_ch > 1:
        valid_mask = metrics > threshold

        if xp.sum(valid_mask) > 1:
            valid_peaks = peak_indices[valid_mask]
            spread = int(xp.max(valid_peaks) - xp.min(valid_peaks))

            if spread > 0:
                logger.warning(
                    f"Skew detected among valid channels! "
                    f"Valid Peaks: {valid_peaks.tolist()}. Spread: {spread} samples."
                )
            else:
                logger.info(f"Channels aligned (spread {spread}).")

    # Each channel's peak is found independently so hardware skew is preserved.
    coarse_offsets = xp.maximum(0, peak_indices + offset)

    if debug_plot:
        from . import plotting as _plotting

        _plotting.timing_correlation(
            corr_mag=to_device(corr_mag, "cpu"),
            peak_indices=to_device(peak_indices, "cpu"),
            norm_factors=to_device(mean_vals, "cpu"),
            threshold=threshold,
            offset=offset,
            show=True,
        )

    # === Fine Timing (Parabolic Interpolation) ===
    fractional_offsets = estimate_fractional_delay(
        corr, peak_indices, dft_upsample=dft_upsample, method=fractional_method
    )

    logger.info(
        f"Timing estimated. Coarse: {coarse_offsets.tolist()}, "
        f"Fractional: {fractional_offsets.tolist()}, "
        f"Metrics: {metrics.tolist()}"
    )

    return coarse_offsets, fractional_offsets


def correct_timing(
    samples: ArrayType,
    coarse_offset: Union[int, ArrayType],
    fractional_offset: Union[float, ArrayType] = 0.0,
    mode: str = "circular",
) -> ArrayType:
    """
    Combined coarse and fine timing correction.

    Applies an integer sample shift followed by fractional sample
    interpolation using FFT-based frequency-domain delay.

    Parameters
    ----------
    samples : array_like
        Input signal. Shape: (N,) or (C, N).
    coarse_offset : int or array_like
        Integer sample offset(s) to correct. Positive values shift
        the signal left (i.e., remove leading samples).
        Scalar or shape (C,) for per-channel offsets.
    fractional_offset : float or array_like, default 0.0
        Fractional sample delay(s) in [-0.5, 0.5) to correct via
        FFT-based interpolation. Scalar or shape (C,).
    mode : {'circular', 'zero', 'slice'}, default 'circular'
        How to handle boundary samples after the coarse shift:

        - ``'circular'``: Wrap-around (``xp.roll``). Output has the
          same length as input.  Correct for periodic signals or
          unit tests; **not** suitable for burst frames.
        - ``'zero'``: Shift left, fill trailing samples with zeros.
          Same output shape as input.  Correct for burst reception
          where tail wrapping would corrupt the payload.
        - ``'slice'``: Discard leading samples; no tail artifact.
          For a scalar offset the output length is ``N - offset``.
          For per-channel offsets the output length is
          ``N - max(offset)`` so all channels are aligned to the
          same common overlap region — the correct approach for
          offline MIMO timing-skew correction.

    Returns
    -------
    array_like
        Timing-corrected signal.  Same shape as input for
        ``'circular'`` and ``'zero'``; shorter for ``'slice'``.

    Notes
    -----
    **Reference sequence is not removed.**  After correction,
    ``signal[..., 0]`` corresponds to the first sample of the detected
    reference (e.g. preamble, training sequence).  All three modes align
    the reference start to index 0 — they do not strip it from the output.
    Use :meth:`~commstools.core.SingleCarrierFrame.get_structure_map`
    to locate preamble and payload regions, or manually slice using the
    structure map if you need to process the body in isolation.

    The fractional correction uses FFT-based frequency-domain delay, which
    is mathematically ideal for bandlimited signals and perfectly preserves
    signal power (unlike polynomial interpolators).
    """
    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]

    num_ch = samples.shape[0]
    N = samples.shape[-1]

    # === Coarse correction (integer shift) ===
    coarse_offset = xp.asarray(coarse_offset)

    if coarse_offset.ndim == 0:
        # --- Scalar: same shift for all channels ---
        shift = int(coarse_offset)
        if mode == "circular":
            samples = xp.roll(samples, -shift, axis=-1)
        elif mode == "zero":
            result = xp.zeros_like(samples)
            if shift > 0:
                result[..., : N - shift] = samples[..., shift:]
            elif shift < 0:
                result[..., -shift:] = samples[..., : N + shift]
            samples = result
        elif mode == "slice":
            samples = samples[..., shift:]
        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Choose 'circular', 'zero', or 'slice'."
            )

    else:
        # --- Per-channel: vectorized gather (avoids one GPU→CPU sync per channel) ---
        coarse_int = coarse_offset.astype(xp.int64)  # (C,) on device
        col_base = xp.arange(N, dtype=xp.int64)[None, :]  # (1, N)
        row_idx = xp.arange(num_ch)[:, None]  # (C, 1)

        if mode == "circular":
            col_idx = (col_base + coarse_int[:, None]) % N  # (C, N)
            samples = samples[row_idx, col_idx]

        elif mode == "zero":
            col_raw = col_base + coarse_int[:, None]  # (C, N)
            gathered = samples[row_idx, xp.clip(col_raw, 0, N - 1)]
            samples = xp.where(col_raw < N, gathered, xp.zeros_like(gathered))

        elif mode == "slice":
            # Align all channels to common overlap: N - max(offset) samples
            max_shift = int(xp.max(coarse_int))  # one GPU sync
            common_len = N - max_shift
            col_idx_s = (
                xp.arange(common_len, dtype=xp.int64)[None, :] + coarse_int[:, None]
            )  # (C, common_len)
            samples = samples[row_idx, col_idx_s]

        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Choose 'circular', 'zero', or 'slice'."
            )

    # === Fine correction (fractional via FFT) ===
    # Correction removes the delay: negate the fractional offset
    if isinstance(fractional_offset, (int, float)):
        apply_frac = abs(fractional_offset) > 1e-9
    else:
        fractional_offset = xp.asarray(fractional_offset)
        apply_frac = bool(xp.any(xp.abs(fractional_offset) > 1e-9))

    if apply_frac:
        samples = fft_fractional_delay(samples, -fractional_offset)

    if mode == "slice":
        logger.warning(
            f"correct_timing(mode='slice'): output is shorter than input "
            f"(trimmed by up to {int(xp.max(xp.asarray(coarse_offset)))} samples). "
            "Signal length metadata (e.g. duration) will no longer match the original."
        )
    logger.info(
        f"Timing corrected: coarse={coarse_offset.tolist() if hasattr(coarse_offset, 'tolist') else coarse_offset}, "
        f"fractional={'applied' if apply_frac else 'skipped'}, mode={mode!r}."
    )

    if was_1d:
        return samples[0]
    return samples


# -----------------------------------------------------------------------------
# FOE/CPR helpers
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Frequency Offset Estimation (FOE)
# -----------------------------------------------------------------------------


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
    IEEE Signal Process. Mag., vol. 24, no. 3, pp. 123–125, May 2007.
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
    weighted means.  This reduces variance by 30–50 % when pilot SNR
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
    regression," *IEEE Trans. Inf. Theory*, vol. 31, no. 6, pp. 832–835,
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
    sps: int = 2,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
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
    sps : int, default 2
        Samples per symbol (passed to ``estimate_frequency_offset_mth_power``
        for the M-th power modulation mapping).
    modulation : str, optional
        Modulation scheme (e.g. ``'qam'``, ``'psk'``).  Required for blind
        estimation with either method.
    order : int, optional
        Modulation order (e.g. 4, 16, 64).  Required alongside ``modulation``.

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
    3. Cubic-spline-interpolate :math:`\Delta f[k]` at block centers to a
       dense per-sample grid :math:`\Delta f_\text{dense}(n)`, with
       ``fill_value='extrapolate'`` to cover the first and last partial blocks.
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
    if len(t_centers) == 1:
        df_dense = np.full(N, df_estimates[0], dtype=np.float64)
    else:
        from scipy.interpolate import interp1d  # noqa: PLC0415

        interp_fn = interp1d(
            t_centers, df_estimates, kind="cubic", fill_value="extrapolate"
        )
        df_dense = interp_fn(n_grid)

    # Integrate: θ(n) = (2π / fs) * cumsum(Δf_dense)
    phase_trajectory = (2.0 * np.pi / sampling_rate) * np.cumsum(df_dense)

    logger.debug(
        f"FOE blockwise: {len(starts)} blocks, method={method}, "
        f"freq range=[{df_estimates.min():.2f}, {df_estimates.max():.2f}] Hz, "
        f"total phase drift={float(phase_trajectory[-1]):.3f} rad"
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


# -----------------------------------------------------------------------------
# Carrier Phase Recovery (CPR)
# -----------------------------------------------------------------------------

# Lazy-compiled Numba kernels for sequential CPR algorithms.
_NUMBA_PLL: dict = {}


def _get_numba_dd_pll():
    """JIT-compile and cache the Numba DD-PLL sample-wise loop kernel.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_loop``.
    """
    if "dd_pll" not in _NUMBA_PLL:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_loop(sym_r, sym_i, const_r, const_i, mu, beta, phi0, freq0):
            """Inner DD-PLL loop compiled to machine code by Numba.

            Parameters
            ----------
            sym_r, sym_i : (N,) float64
                Real and imaginary parts of received symbols.
            const_r, const_i : (M,) float64
                Real and imaginary parts of reference constellation.
            mu : float64
                Proportional (phase) gain — corrects the instantaneous phase error.
            beta : float64
                Integral (frequency) gain — tracks residual frequency drift.
                Set to 0.0 for a 1st-order loop.
            phi0 : float64
                Initial phase state in radians.
            freq0 : float64
                Initial frequency correction state in radians/symbol.

            Returns
            -------
            phase_est : (N,) float64
                Per-symbol phase trajectory φ[n].
            """
            N = len(sym_r)
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            freq = freq0

            for n in range(N):
                # Rotate received symbol by current phase estimate:
                # y[n] = s[n] · exp(−jφ[n])
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                yr = sym_r[n] * cos_phi + sym_i[n] * sin_phi
                yi = -sym_r[n] * sin_phi + sym_i[n] * cos_phi

                # Hard decision: argmin_{c ∈ C} |y − c|²
                min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                d_r = const_r[0]
                d_i = const_i[0]
                for k in range(1, M):
                    d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                    if d2 < min_d2:
                        min_d2 = d2
                        d_r = const_r[k]
                        d_i = const_i[k]

                # Cross-product phase error:  e = Im(y · d*) = yi·d_r − yr·d_i
                e = yi * d_r - yr * d_i

                # Record the phase used to derotate symbol n — before the update.
                phase_est[n] = phi

                # 2nd-order loop filter (reduces to 1st order when beta=0):
                #   φ[n+1] = φ[n] + μ·e[n] + ν[n]
                #   ν[n]   = ν[n−1] + β·e[n]
                phi = phi + mu * e + freq
                freq = freq + beta * e

            return phase_est

        _NUMBA_PLL["dd_pll"] = _dd_pll_loop

    return _NUMBA_PLL["dd_pll"]


def _get_numba_dd_pll_butterworth():
    """JIT-compile and cache the Numba DD-PLL loop with Butterworth loop filter.

    The Butterworth loop filter replaces the simple PI (proportional-integral)
    structure with a 2nd-order IIR biquad.  This gives a flatter passband and
    sharper roll-off, improving phase noise rejection near the loop bandwidth.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_bw_loop``.
    """
    if "dd_pll_bw" not in _NUMBA_PLL:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_bw_loop(sym_r, sym_i, const_r, const_i, phi0, b0, b1, b2, a1, a2):
            """DD-PLL inner loop with a 2nd-order Butterworth loop filter.

            Parameters
            ----------
            sym_r, sym_i : (N,) float64
                Real and imaginary parts of received symbols.
            const_r, const_i : (M,) float64
                Real and imaginary parts of reference constellation.
            phi0 : float64
                Initial phase state in radians.
            b0, b1, b2 : float64
                Numerator coefficients of the 2nd-order Butterworth IIR filter.
            a1, a2 : float64
                Denominator coefficients (a[1], a[2]; a[0] is normalised to 1).

            Returns
            -------
            phase_est : (N,) float64
                Per-symbol phase trajectory φ[n].
            """
            N = len(sym_r)
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0

            # Biquad Direct Form II Transposed state variables
            w1 = 0.0
            w2 = 0.0

            for n in range(N):
                # Rotate received symbol by current phase estimate
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                yr = sym_r[n] * cos_phi + sym_i[n] * sin_phi
                yi = -sym_r[n] * sin_phi + sym_i[n] * cos_phi

                # Hard decision: argmin_{c ∈ C} |y − c|²
                min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                d_r = const_r[0]
                d_i = const_i[0]
                for k in range(1, M):
                    d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                    if d2 < min_d2:
                        min_d2 = d2
                        d_r = const_r[k]
                        d_i = const_i[k]

                # Cross-product phase error: e = Im(y · d*) = yi·d_r − yr·d_i
                e = yi * d_r - yr * d_i

                # Record phase before update
                phase_est[n] = phi

                # Biquad (Direct Form II Transposed):
                #   v[n] = b0·e[n] + w1[n-1]
                #   w1[n] = b1·e[n] - a1·v[n] + w2[n-1]
                #   w2[n] = b2·e[n] - a2·v[n]
                v_out = b0 * e + w1
                w1 = b1 * e - a1 * v_out + w2
                w2 = b2 * e - a2 * v_out

                phi = phi + v_out

            return phase_est

        _NUMBA_PLL["dd_pll_bw"] = _dd_pll_bw_loop

    return _NUMBA_PLL["dd_pll_bw"]


def _get_numba_dd_pll_joint():
    """JIT-compile and cache the joint-channel DD-PLL PI kernel.

    Averages the cross-product phase error across C channels at each symbol
    before updating the single shared phase/frequency state.  This is the
    MVUE joint estimator for shared-LO systems.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_joint_loop``.
    """
    if "dd_pll_joint" not in _NUMBA_PLL:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_joint_loop(sym_r, sym_i, const_r, const_i, mu, beta, phi0, freq0):
            """Joint-channel DD-PLL with PI loop filter.

            Parameters
            ----------
            sym_r, sym_i : (C, N) float64
                Real and imaginary parts of received symbols, all channels.
            const_r, const_i : (M,) float64
                Reference constellation.
            mu, beta, phi0, freq0 : float64
                Loop parameters — same semantics as ``_dd_pll_loop``.

            Returns
            -------
            phase_est : (N,) float64
                Single shared phase trajectory (broadcast to all channels by caller).
            """
            C = sym_r.shape[0]
            N = sym_r.shape[1]
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            freq = freq0

            for n in range(N):
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                e_sum = 0.0
                for c in range(C):
                    yr = sym_r[c, n] * cos_phi + sym_i[c, n] * sin_phi
                    yi = -sym_r[c, n] * sin_phi + sym_i[c, n] * cos_phi
                    min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                    d_r = const_r[0]
                    d_i = const_i[0]
                    for k in range(1, M):
                        d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                        if d2 < min_d2:
                            min_d2 = d2
                            d_r = const_r[k]
                            d_i = const_i[k]
                    e_sum += yi * d_r - yr * d_i
                # Average error across channels — MVUE for shared LO
                e = e_sum / float(C)
                phase_est[n] = phi
                phi = phi + mu * e + freq
                freq = freq + beta * e

            return phase_est

        _NUMBA_PLL["dd_pll_joint"] = _dd_pll_joint_loop

    return _NUMBA_PLL["dd_pll_joint"]


def _get_numba_dd_pll_joint_butterworth():
    """JIT-compile and cache the joint-channel DD-PLL Butterworth kernel.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_joint_bw_loop``.
    """
    if "dd_pll_joint_bw" not in _NUMBA_PLL:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_joint_bw_loop(
            sym_r, sym_i, const_r, const_i, phi0, b0, b1, b2, a1, a2
        ):
            """Joint-channel DD-PLL with 2nd-order Butterworth loop filter.

            Parameters
            ----------
            sym_r, sym_i : (C, N) float64
            const_r, const_i : (M,) float64
            phi0 : float64
            b0, b1, b2, a1, a2 : float64
                Butterworth biquad coefficients.

            Returns
            -------
            phase_est : (N,) float64
            """
            C = sym_r.shape[0]
            N = sym_r.shape[1]
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            w1 = 0.0
            w2 = 0.0

            for n in range(N):
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                e_sum = 0.0
                for c in range(C):
                    yr = sym_r[c, n] * cos_phi + sym_i[c, n] * sin_phi
                    yi = -sym_r[c, n] * sin_phi + sym_i[c, n] * cos_phi
                    min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                    d_r = const_r[0]
                    d_i = const_i[0]
                    for k in range(1, M):
                        d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                        if d2 < min_d2:
                            min_d2 = d2
                            d_r = const_r[k]
                            d_i = const_i[k]
                    e_sum += yi * d_r - yr * d_i
                e = e_sum / float(C)
                phase_est[n] = phi
                v_out = b0 * e + w1
                w1 = b1 * e - a1 * v_out + w2
                w2 = b2 * e - a2 * v_out
                phi = phi + v_out

            return phase_est

        _NUMBA_PLL["dd_pll_joint_bw"] = _dd_pll_joint_bw_loop

    return _NUMBA_PLL["dd_pll_joint_bw"]


_NUMBA_RTS: dict = {}


def _get_numba_rts_smoother():
    """JIT-compile and cache the Numba RTS-smoother kernel.

    Returns
    -------
    callable
        Numba-compiled ``_rts_loop``.
    """
    if "rts" not in _NUMBA_RTS:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _rts_loop(phi_obs, sigma_p2, sigma_v2):
            """Rauch-Tung-Striebel smoother — Numba inner kernel.

            Parameters
            ----------
            phi_obs : (B,) float64
            sigma_p2 : float64
            sigma_v2 : float64

            Returns
            -------
            (B,) float64
            """
            B = len(phi_obs)
            x_filt = np.empty(B, dtype=np.float64)
            P_filt = np.empty(B, dtype=np.float64)
            x_pred = np.empty(B, dtype=np.float64)
            P_pred = np.empty(B, dtype=np.float64)

            x_filt[0] = phi_obs[0]
            P_filt[0] = sigma_v2

            for k in range(1, B):
                x_pred[k] = x_filt[k - 1]
                P_pred[k] = P_filt[k - 1] + sigma_p2
                K = P_pred[k] / (P_pred[k] + sigma_v2)
                x_filt[k] = x_pred[k] + K * (phi_obs[k] - x_pred[k])
                P_filt[k] = (1.0 - K) * P_pred[k]

            x_smooth = x_filt.copy()
            for k in range(B - 2, -1, -1):
                G = P_filt[k] / P_pred[k + 1]
                x_smooth[k] = x_filt[k] + G * (x_smooth[k + 1] - x_pred[k + 1])

            return x_smooth

        _NUMBA_RTS["rts"] = _rts_loop

    return _NUMBA_RTS["rts"]


def recover_carrier_phase_pll(
    symbols: ArrayType,
    modulation: str,
    order: int,
    mu: float = 1e-2,
    beta: float = 0.0,
    phase_init: float = 0.0,
    loop_filter: str = "pi",
    loop_bandwidth_normalized: float = 1e-3,
    joint_channels: bool = False,
    cycle_slip_correction: bool = True,
    cycle_slip_history: int = 10000,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    r"""
    Carrier phase recovery via a Decision-Directed Phase-Locked Loop (DD-PLL).

    Tracks the carrier phase symbol-by-symbol using hard decisions as phase
    references.  A 1st-order loop (``beta=0``) corrects static or slowly
    varying phase noise; a 2nd-order loop (``beta > 0``) additionally tracks
    a residual frequency offset left over after coarse FOE.

    This is the standard streaming CPR for hardware implementations: it is
    modulation-format agnostic (works for any QAM/PSK order) and converges
    much faster than block-based methods (VV, BPS) after equalizer pull-in.

    .. warning::
        The DD-PLL requires reliable decisions at the input.  For a cold
        start the first ``~1/mu`` symbols may show slow convergence.
        A common strategy is to pre-converge with BPS or a short preamble
        and feed the resulting phase as ``phase_init``.

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filtering and FOE.
        Shape: ``(N,)`` or ``(C, N)``.
    modulation : str
        Modulation scheme (case-insensitive): ``'qam'``, ``'psk'``, etc.
        Used to fetch the reference constellation via
        :func:`~commstools.mapping.gray_constellation`.
    order : int
        Modulation order (4, 16, 64, …).
    mu : float, default 1e-2
        Proportional gain — controls convergence speed and steady-state
        jitter.  Larger ``mu`` converges faster but amplifies noise.
        Typical range: ``1e-3`` (high-SNR, high-order QAM) to ``5e-2``
        (QPSK, low latency).
    beta : float, default 0.0
        Integral gain — enables 2nd-order frequency tracking.
        Set ``beta > 0`` when a residual frequency offset remains after
        FOE (e.g. ``beta ≈ mu² / 4``).  Zero gives a 1st-order loop.
    phase_init : float, default 0.0
        Initial phase state in radians.  Use the last sample of a
        preceding BPS or pilot-aided estimate to warm-start the loop.
    loop_filter : {"pi", "butterworth"}, default "pi"
        Loop filter type.

        * ``"pi"`` (default) — classic proportional-integral filter
          controlled by ``mu`` and ``beta``.
        * ``"butterworth"`` — 2nd-order Butterworth IIR loop filter
          designed via bilinear (Tustin) transform.  Controlled by
          ``loop_bandwidth_normalized`` instead of ``mu``/``beta``.
          Provides a flatter passband and sharper roll-off than PI,
          improving phase noise rejection at the cost of a small
          transient overshoot.

        When ``loop_filter='butterworth'``, ``mu`` and ``beta`` are
        ignored.
    loop_bandwidth_normalized : float, default 1e-3
        Normalised one-sided loop bandwidth as a fraction of the symbol
        rate (i.e. in the range ``(0, 0.5)``).  Only used when
        ``loop_filter='butterworth'``.  Typical values: ``1e-4``
        (narrow, low phase noise) to ``1e-2`` (wide, fast tracking).
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, average the cross-product
        phase error across all channels at each symbol before updating the
        shared loop state.  Both polarisations drive a single phase/frequency
        trajectory, giving ~√C variance reduction for shared-LO systems.
        The output ``phi_full[ch]`` rows are all identical.
        Has no effect for SISO (C = 1).
    cycle_slip_correction : bool, default True
        If ``True``, apply :func:`correct_cycle_slips` to the per-symbol
        phase trajectory after the loop, to detect and fix sudden ``π/2``
        jumps caused by incorrect hard decisions near the branch boundary.
    cycle_slip_history : int, default 10000
        ``history_length`` passed to :func:`correct_cycle_slips`.
        Default is higher than for block-phase methods because the trajectory
        is per-symbol (not per-block).
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).

    Returns
    -------
    array_like
        Per-symbol phase estimate φ[n] in radians.
        Shape matches ``symbols``.  Same backend as input.

    Notes
    -----
    **Algorithm** (per sample n):

    .. math::

        y[n]       &= s[n] \cdot e^{-j\hat{\phi}[n]} \\
        \hat{d}[n] &= \operatorname{argmin}_{c \in \mathcal{C}}
                       \lvert y[n] - c \rvert^2 \\
        e[n]       &= \operatorname{Im}\!\bigl(y[n]\,\hat{d}^*[n]\bigr) \\
        \hat{\phi}[n+1] &= \hat{\phi}[n] + \mu e[n] + \nu[n] \\
        \nu[n]     &= \nu[n-1] + \beta e[n]

    where :math:`\nu` is the integral (frequency) state of the loop.

    **Backend notes:** The inner loop is inherently sequential (each sample
    depends on the previous phase state) and is compiled with Numba
    (``@njit``) for CPU performance.  When the input lives on a GPU
    (CuPy), samples are transparently moved to CPU for processing and
    the result is moved back — acceptable because the CPR loop is not
    the throughput bottleneck.

    **M-fold phase ambiguity:** Like VV and BPS, the DD-PLL may converge
    to any of the M constellation-symmetry-equivalent phases.  Resolve
    via a pilot symbol or known reference after CPR.

    References
    ----------
    I. Fatadin, D. Ives, and S. J. Savory, "Blind equalization and
    carrier phase recovery in a 16-QAM optical coherent system," *J.
    Lightw. Technol.*, vol. 27, no. 15, pp. 3042-3049, Aug. 2009.

    Md. S. Faruk and S. J. Savory, "Digital signal processing for coherent
    transceivers employing multilevel formats," *J. Lightw. Technol.*,
    vol. 35, no. 5, pp. 1125-1141, Mar. 2017, Sec. VIII.A, refs [65, 108].

    J. G. Proakis, *Digital Communications*, 4th ed., McGraw-Hill, 2001,
    ch. 6 (carrier phase synchronisation).
    """
    from .helpers import normalize
    from .mapping import gray_constellation

    if loop_filter not in ("pi", "butterworth"):
        raise ValueError(
            f"loop_filter must be 'pi' or 'butterworth', got {loop_filter!r}."
        )
    if loop_filter == "butterworth" and not (0.0 < loop_bandwidth_normalized < 0.5):
        raise ValueError(
            f"loop_bandwidth_normalized must be in (0, 0.5), got {loop_bandwidth_normalized}."
        )
    if loop_filter == "butterworth":
        if mu != 1e-2 or beta != 0.0:
            logger.warning(
                "loop_filter='butterworth': mu and beta are ignored. "
                "Use loop_bandwidth_normalized to control loop bandwidth."
            )

    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    # Normalise to unit average power so the effective loop gain is mu regardless
    # of input amplitude.  The error signal is e[n] = Im(y[n]*d_hat*), which
    # scales with signal amplitude; without this, the effective gain is mu*A
    # (where A is the RMS amplitude), making loop bandwidth input-dependent.
    symbols = normalize(symbols, mode="average_power", axis=-1)

    # Constellation on CPU (decisions are scalar operations in the loop)
    const_np = gray_constellation(modulation, order).astype(np.complex128)
    const_r = const_np.real.copy()
    const_i = const_np.imag.copy()

    # Move to CPU for sequential processing
    if xp is not np:
        symbols_cpu = to_device(symbols, "cpu")
    else:
        symbols_cpu = symbols

    phi_full = np.zeros((C, N), dtype=np.float64)
    use_joint = joint_channels and C > 1

    # Pre-build (C, N) float64 views used by joint kernels
    if use_joint:
        symbols_np = symbols_cpu.astype(np.complex128)
        sym_r_all = np.ascontiguousarray(symbols_np.real)  # (C, N) float64
        sym_i_all = np.ascontiguousarray(symbols_np.imag)

    if loop_filter == "butterworth":
        import scipy.signal as _ss  # noqa: PLC0415

        # Design 2nd-order Butterworth lowpass at loop_bandwidth_normalized
        # (normalised by Nyquist = 0.5 symbol rate, so Wn = 2 * lbw).
        b_arr, a_arr = _ss.butter(
            2, 2.0 * loop_bandwidth_normalized, btype="low", analog=False
        )
        b0, b1, b2 = float(b_arr[0]), float(b_arr[1]), float(b_arr[2])
        a1, a2 = float(a_arr[1]), float(a_arr[2])

        if use_joint:
            jbw_kernel = _get_numba_dd_pll_joint_butterworth()
            phi_joint = jbw_kernel(
                sym_r_all,
                sym_i_all,
                const_r,
                const_i,
                float(phase_init),
                b0,
                b1,
                b2,
                a1,
                a2,
            )
            for ch in range(C):
                phi_full[ch] = phi_joint
        else:
            bw_kernel = _get_numba_dd_pll_butterworth()
            for ch in range(C):
                sym = symbols_cpu[ch].astype(np.complex128)
                phi_full[ch] = bw_kernel(
                    sym.real.copy(),
                    sym.imag.copy(),
                    const_r,
                    const_i,
                    float(phase_init),
                    b0,
                    b1,
                    b2,
                    a1,
                    a2,
                )
        loop_desc = f"Butterworth, BW={loop_bandwidth_normalized:.2g}"
    else:
        if use_joint:
            j_kernel = _get_numba_dd_pll_joint()
            phi_joint = j_kernel(
                sym_r_all,
                sym_i_all,
                const_r,
                const_i,
                float(mu),
                float(beta),
                float(phase_init),
                0.0,
            )
            for ch in range(C):
                phi_full[ch] = phi_joint
        else:
            pi_kernel = _get_numba_dd_pll()
            for ch in range(C):
                sym = symbols_cpu[ch].astype(np.complex128)
                phi_full[ch] = pi_kernel(
                    sym.real.copy(),
                    sym.imag.copy(),
                    const_r,
                    const_i,
                    float(mu),
                    float(beta),
                    float(phase_init),
                    0.0,
                )
        loop_order = "2nd" if beta > 0.0 else "1st"
        loop_desc = f"PI {loop_order}-order, mu={mu}, beta={beta}"

    if cycle_slip_correction:
        if use_joint:
            # All rows are identical — correct once and broadcast
            phi_full[0] = correct_cycle_slips(
                phi_full[0],
                symmetry=4,
                history_length=cycle_slip_history,
                threshold=cycle_slip_threshold,
            )
            for ch in range(1, C):
                phi_full[ch] = phi_full[0]
        else:
            for ch in range(C):
                phi_full[ch] = correct_cycle_slips(
                    phi_full[ch],
                    symmetry=4,
                    history_length=cycle_slip_history,
                    threshold=cycle_slip_threshold,
                )

    # Move result back to original device
    if xp is not np:
        phi_full = xp.asarray(phi_full)

    phi_mean_deg = float(np.mean(phi_full)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full)) * 180.0 / np.pi
    logger.info(
        f"CPR (DD-PLL, {loop_desc}): phase mean={phi_mean_deg:.2f}°, "
        f"std={phi_std_deg:.2f}° [C={C}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full if xp is np else to_device(phi_full, "cpu"),
            show=True,
            title=f"CPR — DD-PLL ({loop_desc})",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def recover_carrier_phase_viterbi_viterbi(
    symbols: ArrayType,
    modulation: str,
    order: int,
    block_size: int = 32,
    joint_channels: bool = False,
    cycle_slip_correction: bool = True,
    cycle_slip_history: int = 1000,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Carrier phase recovery via the Viterbi-Viterbi (M-th power) algorithm.

    Block-based blind phase estimation for PSK and QAM symbols. Raises each
    block of symbols to the M-th power to remove modulation, extracts the
    block phase, resolves the M-fold ambiguity by unwrapping, then
    interpolates to per-symbol resolution.

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filter. Shape: (N,) or (C, N).
    modulation : str
        Modulation scheme (case-insensitive): 'psk', 'qam', etc.
    order : int
        Modulation order.
    block_size : int, default 32
        Number of symbols per estimation block. Larger blocks reduce
        variance but reduce tracking bandwidth for fast phase noise.
        Typical range: 16-128 for QAM; as low as 1 for PSK (data cancels
        exactly in the M-th power for M-PSK constellations).
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the M-th-power block
        phasors ``S_b`` across all channels before phase extraction.
        The resulting single trajectory is broadcast to all C output rows.
        Reduces variance by ~√C for shared-LO systems.  SISO-safe.
    cycle_slip_correction : bool, default True
        If ``True``, apply cycle-slip detection and correction
        (:func:`correct_cycle_slips`) after M-fold unwrap, before
        interpolation.
    cycle_slip_history : int, default 1000
        ``history_length`` passed to :func:`correct_cycle_slips`.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-symbol phase
        trajectory alongside the block-phase estimates.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians. Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    Algorithm for each block ``b``:

    .. math::
        S_b = \\sum_{n \\in \\text{block } b} s[n]^M, \\quad
        \\hat{\\phi}_b = \\angle(S_b) / M

    **M-fold ambiguity resolution:** block phases are scaled by M,
    unwrapped in the 2π domain, then re-divided by M.  A global ``2π/M``
    phase ambiguity always remains — resolve it via a known pilot or
    preamble reference.

    .. warning::
        **Phase-unwrapping slip risk:** the unwrapper assumes consecutive block
        phases differ by less than ``π/M``.  Two independent effects can
        violate this:

        1. **Phase noise (high linewidth):** for a combined linewidth
           :math:`\\Delta\\nu`, the safety condition is:

           .. math::

               \\Delta\\nu \\cdot T_{\\text{block}} < 0.05 \\cdot f_s

           where :math:`T_{\\text{block}} = \\text{block\\_size} / f_s`.
           For example, 100 kHz linewidth at 32 Gbaud is safe up to
           ``block_size ≈ 16 000``, but 1 MHz linewidth requires
           ``block_size ≤ 1 600``.

        2. **Insufficient averaging for QAM (dominant at small block_size):**
           For PSK constellations every point maps to the *same* M-th power
           value, so the data modulation cancels exactly even with a single
           symbol per block.  For QAM constellations with ``order > 4`` the
           M-th power of individual symbols is **not** constant — it varies
           by constellation point.  Averaging over a block suppresses this
           residual, but a small ``block_size`` leaves significant variance
           that exceeds the unwrap threshold.  The minimum block size for
           reliable unwrapping scales roughly as
           :math:`4\\lceil\\sqrt{\\text{order}}\\rceil`:

           * 16-QAM → ``block_size ≥ 16``
           * 64-QAM → ``block_size ≥ 32``
           * 256-QAM → ``block_size ≥ 64``

           Using a ``block_size`` below this threshold will cause persistent
           ``2π/M`` phase slips regardless of SNR.

        When operating near or above the phase-noise limit, prefer
        :func:`recover_carrier_phase_bps`, which uses a brute-force phase
        search and does not require phase unwrapping.

    References
    ----------
    A. J. Viterbi and A. M. Viterbi, "Nonlinear estimation of PSK-modulated
    carrier phase with application to burst digital transmission," IEEE
    Trans. Inf. Theory, 1983.
    """
    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]  # (1, N)
    C, N = symbols.shape

    M = _modulation_power_m(modulation, order)

    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    if N_blocks == 0:
        raise ValueError(
            f"Signal length {N} is shorter than block_size={block_size}. "
            "Reduce block_size or use a longer symbol sequence."
        )

    # For QAM with order > 4 the M-th power of individual symbols does NOT cancel
    # the data modulation (unlike PSK, where every M-PSK point gives (c/|c|)^M = 1).
    # Sufficient block averaging is required so that the block-phase variance stays
    # below the π/M unwrap threshold.  The practical minimum scales as 4·ceil(√order).
    if "qam" in modulation.lower() and order > 4:
        _min_bs = max(8, 4 * int(np.ceil(order**0.5)))
        if block_size < _min_bs:
            logger.warning(
                f"CPR (VV): block_size={block_size} is too small for {order}-QAM. "
                f"Individual QAM symbols' M-th powers do not cancel the data modulation; "
                f"insufficient averaging causes block-phase variance that exceeds the "
                f"π/M unwrap threshold, producing persistent 2π/M phase slips. "
                f"Recommended minimum for {order}-QAM: block_size ≥ {_min_bs}."
            )

    # Reshape for block processing: (C, N_blocks, block_size).
    # Promote to complex128 for the M-th power — identical to estimate_frequency_offset_mth_power.
    # On GPU, complex64^4 loses precision near the ±π/M unwrap boundary, causing
    # spurious branch flips for high-order QAM with small block sizes.
    blocks = symbols[:, :N_trunc].reshape(C, N_blocks, block_size)
    blocks_c = blocks.astype(
        xp.complex128 if blocks.dtype == xp.complex64 else blocks.dtype
    )

    # For QAM, project to unit circle before the M-th power (normalized VV).
    # This removes outer-ring amplitude dominance and makes the π/M QAM bias
    # correction exact (by the 4-fold rotational symmetry of the constellation).
    # PSK is already constant-modulus; normalization is a no-op.
    if "qam" in modulation.lower():
        mag = xp.abs(blocks_c)
        blocks_c = blocks_c / xp.maximum(mag, 1e-15 * xp.max(mag))

    S_b = xp.sum(blocks_c**M, axis=-1)  # (C, N_blocks)

    # Block centre positions for interpolation (uniform spacing = block_size)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)

    phi_full = xp.zeros((C, N), dtype=xp.float64)
    phi_blocks_out = xp.zeros((C, N_blocks), dtype=xp.float64)

    if joint_channels and C > 1:
        # Sum M-th-power phasors across channels → single block-phase trajectory
        S_b_joint = xp.sum(S_b, axis=0)  # (N_blocks,)
        phi_raw_joint = xp.angle(S_b_joint) / M
        phi_u_joint = xp.unwrap((phi_raw_joint * M).astype(xp.float64)) / M
        if "qam" in modulation.lower():
            phi_u_joint = phi_u_joint - (np.pi / M)
        if cycle_slip_correction:
            phi_u_joint_np = correct_cycle_slips(
                to_device(phi_u_joint, "cpu"),
                4,
                cycle_slip_history,
                cycle_slip_threshold,
            )
            phi_u_joint = xp.asarray(phi_u_joint_np)
        phi_interp = xp.interp(all_positions, block_centers, phi_u_joint)
        for ch in range(C):
            phi_full[ch] = phi_interp
            phi_blocks_out[ch] = phi_u_joint
    else:
        # Raw block phase in [-π/M, π/M)
        phi_raw = xp.angle(S_b) / M  # (C, N_blocks)

        # M-fold unwrap: scale into 2π domain, unwrap, re-scale back.
        # Cast to float64 before unwrap — cp.unwrap preserves input dtype so float32
        # would lose precision during the discontinuity test (diff vs 2π threshold).
        phi_u = (
            xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M
        )  # (C, N_blocks)

        # QAM bias correction.
        if "qam" in modulation.lower():
            phi_u = phi_u - (np.pi / M)

        # MIMO M-fold alignment: align every channel to channel 0's branch.
        # Skipped in joint mode (all channels share the same trajectory).
        if C > 1:
            for ch in range(1, C):
                diff = float(xp.mean(phi_u[ch] - phi_u[0]))
                k = round(diff * M / (2 * np.pi))
                phi_u[ch] = phi_u[ch] - k * (2 * np.pi / M)

        # xp.interp is 1D-only; loop over C channels.
        for ch in range(C):
            phi_u_ch = phi_u[ch]
            if cycle_slip_correction:
                phi_u_ch_np = correct_cycle_slips(
                    to_device(phi_u_ch, "cpu"),
                    4,
                    cycle_slip_history,
                    cycle_slip_threshold,
                )
                phi_u_ch = xp.asarray(phi_u_ch_np)
            phi_full[ch] = xp.interp(all_positions, block_centers, phi_u_ch)
            phi_blocks_out[ch] = phi_u_ch

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    mode_str = "joint" if (joint_channels and C > 1) else "independent"
    logger.info(
        f"CPR (Viterbi-Viterbi, M={M}, {mode_str}): phase mean={phi_mean_deg:.2f}°, "
        f"std={phi_std_deg:.2f}° [{N_blocks} blocks × {block_size} symbols, C={C}, "
        f"cycle_slip_correction={cycle_slip_correction}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full_np,
            block_centers=to_device(block_centers, "cpu"),
            phi_blocks=to_device(phi_blocks_out, "cpu"),
            show=True,
            title="CPR — Viterbi-Viterbi",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def recover_carrier_phase_bps(
    symbols: ArrayType,
    modulation: str,
    order: int,
    num_test_phases: int = 64,
    block_size: int = 32,
    joint_channels: bool = False,
    cycle_slip_correction: bool = True,
    cycle_slip_history: int = 1000,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Carrier phase recovery via Blind Phase Search (BPS).

    Tests ``num_test_phases`` candidate rotation angles over ``[0, π/2)``
    (exploiting 4-fold QAM symmetry), selects the candidate that minimises
    the block-averaged sum of minimum squared distances to the reference
    constellation, and interpolates to per-symbol resolution.

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filter. Shape: (N,) or (C, N).
    modulation : str
        Modulation scheme (case-insensitive). Used to fetch the reference
        constellation via
        :func:`~commstools.mapping.gray_constellation`.
    order : int
        Modulation order.
    num_test_phases : int, default 64
        Number of candidate phase offsets B. Resolution is ``π/(2B)``
        rad per step. More candidates improve accuracy at higher compute cost.
    block_size : int, default 32
        Number of symbols per block for error-metric averaging.
        Very small values (< 4) make the 4-fold phase unwrap unreliable
        because noise on a single-symbol metric causes the best-candidate
        index to jump between non-adjacent phase bins between consecutive
        blocks, triggering false unwrap corrections.  Recommended
        minimum: ``block_size ≥ 4``.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the distance metrics
        across all channels before selecting the best phase candidate.
        The resulting single phase trajectory is broadcast to all C rows
        of the output (all channels identical before ambiguity resolution).
        Reduces phase estimation variance by ~√C for shared-LO systems.
        Has no effect for SISO (C = 1).
    cycle_slip_correction : bool, default True
        If ``True``, apply cycle-slip detection and correction
        (:func:`correct_cycle_slips`) to the block-phase trajectory
        after 4-fold unwrap, before interpolation.
    cycle_slip_history : int, default 1000
        ``history_length`` passed to :func:`correct_cycle_slips`.
        Number of past corrected blocks used for linear extrapolation.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-symbol phase
        trajectory alongside the block-phase estimates.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians. Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    Algorithm:

    1. Candidates: :math:`\\phi_k = k\\,\\pi/(2B)` for :math:`k=0,\\ldots,B-1`.
    2. Rotate: ``x_rot[n, k] = symbols[n] · exp(-j·φ_k)``. Shape: (N, B).
    3. Min dist: ``d²[n,k] = min_c |x_rot[n,k]-c|²``. Shape: (N, B).
    4. Block sum: ``metric[b,k] = Σ d²[n,k]`` over block ``b``.
    5. Best phase: ``φ_b = candidates[argmin(metric[b,:])]``.
    6. 4-fold unwrap, optional cycle-slip correction, per-symbol interpolation.

    .. note::
        The candidate search covers :math:`[0, \\pi/2)` and the unwrap
        exploits the **4-fold** symmetry of square QAM constellations.
        For PSK modulations whose symmetry order differs from 4, the
        candidate range and unwrap fold should be adjusted accordingly.

    **Memory:** The ``(N, B, M_const)`` distance tensor scales as
    ``N·B·M_const·8`` bytes.  For N=10 000, B=64, M=256 → ~1.3 GB.
    Reduce ``num_test_phases`` or process shorter segments for high-order
    constellations.

    **4-fold ambiguity:** A global ``π/2`` phase offset remains after
    unwrapping.  Resolve via a pilot or preamble phase reference.

    References
    ----------
    T. Pfau, S. Hoffmann, and R. Noe, "Hardware-efficient coherent digital
    receiver concept with feedforward carrier recovery for M-QAM
    constellations," J. Lightw. Technol., 2009.
    """
    from .mapping import gray_constellation

    from .helpers import normalize

    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    # Normalise each channel to unit average power so the metric is computed at
    # the same scale as the reference constellation (gray_constellation returns
    # unit-average-power points).  BPS is a phase estimator; it must be
    # amplitude-agnostic.
    symbols = normalize(symbols, mode="average_power", axis=-1)

    # Reference constellation on the same device
    const_np = gray_constellation(modulation, order)
    const_xp = xp.asarray(const_np)  # (M_const,)

    # Candidate test phases over [0, π/2)
    B = num_test_phases
    candidates = xp.arange(B, dtype=symbols.real.dtype) * (np.pi / 2.0 / B)  # (B,)

    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    if N_blocks == 0:
        raise ValueError(
            f"Signal length {N} is shorter than block_size={block_size}. "
            "Reduce block_size or use a longer symbol sequence."
        )

    # Very small block_size makes the 4-fold phase unwrap unreliable: with only
    # one or two symbols per block the noise on the distance-metric argmin causes
    # large candidate-index jumps between consecutive blocks, triggering false
    # 4-fold unwrap corrections.  Warn early so users diagnose this easily.
    if block_size < 4:
        logger.warning(
            f"CPR (BPS): block_size={block_size} is very small. "
            f"Averaging the distance metric over only {block_size} symbol(s) per block "
            f"makes the 4-fold phase unwrap unreliable. Recommended minimum: block_size ≥ 4."
        )

    # block_centers[b] = b * block_size + block_size/2  (consistent with VV)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2

    all_positions = xp.arange(N, dtype=xp.float64)

    # Pre-compute interpolation indices and weights (identical for every channel).
    # block b is "to the left" of position n when its centre b*bs + bs/2 <= n
    #   => b <= (n - bs/2) / bs  => idx_left = floor((n - bs/2) / bs)
    idx_left = xp.clip(
        xp.floor((all_positions - block_size / 2) / block_size).astype(xp.int64),
        0,
        N_blocks - 2,
    )  # (N,)
    idx_right = idx_left + 1  # (N,)
    t_interp = xp.clip(
        (all_positions - block_centers[idx_left]) / block_size, 0.0, 1.0
    )  # (N,)

    # Pre-compute phasors for all B candidates once (avoid redundant exp per channel)
    dtype_c = xp.complex64 if symbols.dtype == xp.complex64 else xp.complex128
    phasors = xp.exp(-1j * candidates.astype(xp.float64)).astype(dtype_c)  # (B,)

    # For square QAM (order a perfect square): the nearest constellation point
    # can be found in O(1) per symbol via per-component rounding, eliminating
    # the (CHUNK, B, M_const) distance tensor entirely.
    side = int(order**0.5)
    is_sq_qam = ("qam" in modulation.lower()) and (side * side == order)
    if is_sq_qam:
        # Sorted unique real levels of the constellation (shape: (side,))
        levels = xp.sort(xp.unique(const_xp.real))
        d_grid = float(levels[1] - levels[0])  # uniform grid spacing
        lev_min = float(levels[0])

    float_dtype = xp.float32 if symbols.dtype == xp.complex64 else xp.float64

    # Chunk size for N axis: bounds peak memory of the distance tensor.
    # Always a multiple of block_size so each chunk covers a whole number of
    # blocks exactly.  Rounded up to the nearest multiple ≥ 1024.
    CHUNK_N = max(block_size, ((1024 + block_size - 1) // block_size) * block_size)

    phi_full = xp.zeros((C, N), dtype=xp.float64)
    phi_blocks = xp.zeros((C, N_blocks), dtype=xp.float64)

    # Accumulate per-channel distance metrics (N_blocks, B) for all channels.
    metrics_all = xp.zeros((C, N_blocks, B), dtype=float_dtype)

    for ch in range(C):
        sym = symbols[ch, :N_trunc]  # (N_trunc,)

        for n0 in range(0, N_trunc, CHUNK_N):
            n1 = min(n0 + CHUNK_N, N_trunc)
            x_rot = sym[n0:n1, None] * phasors[None, :]  # (CHUNK, B)

            if is_sq_qam:
                # O(1) nearest-point: round each component to the nearest grid level
                r_idx = xp.clip(
                    xp.round((x_rot.real - lev_min) / d_grid).astype(xp.int64),
                    0,
                    side - 1,
                )
                i_idx = xp.clip(
                    xp.round((x_rot.imag - lev_min) / d_grid).astype(xp.int64),
                    0,
                    side - 1,
                )
                r_near = levels[r_idx]  # (CHUNK, B)
                i_near = levels[i_idx]  # (CHUNK, B)
                chunk_min_d = (
                    (x_rot.real - r_near) ** 2 + (x_rot.imag - i_near) ** 2
                ).astype(float_dtype)
            else:
                # General: (CHUNK, B, M_const) — bounded by CHUNK_N
                d_sq = xp.abs(x_rot[:, :, None] - const_xp[None, None, :]) ** 2
                chunk_min_d = xp.min(d_sq, axis=-1).astype(float_dtype)

            b0 = n0 // block_size
            n_b = (n1 - n0) // block_size
            metrics_all[ch, b0 : b0 + n_b] = chunk_min_d.reshape(
                n_b, block_size, B
            ).sum(axis=1)

    # Phase estimation: joint (sum metrics across channels) or independent per channel.
    if joint_channels and C > 1:
        metric_joint = xp.sum(metrics_all, axis=0)  # (N_blocks, B)
        best_k_joint = xp.argmin(metric_joint, axis=-1)  # (N_blocks,)
        phi_b_joint = candidates[best_k_joint]  # (N_blocks,)
        phi_u_joint = xp.unwrap(phi_b_joint.astype(xp.float64) * 4, axis=-1) / 4
        if cycle_slip_correction:
            phi_u_joint_np = correct_cycle_slips(
                to_device(phi_u_joint, "cpu"),
                4,
                cycle_slip_history,
                cycle_slip_threshold,
            )
            phi_u_joint = xp.asarray(phi_u_joint_np)
        for ch in range(C):
            phi_full[ch] = (
                phi_u_joint[idx_left] * (1.0 - t_interp)
                + phi_u_joint[idx_right] * t_interp
            )
            phi_blocks[ch] = phi_u_joint
    else:
        for ch in range(C):
            metric = metrics_all[ch]  # (N_blocks, B)
            best_k = xp.argmin(metric, axis=-1)  # (N_blocks,)
            phi_b = candidates[best_k]  # (N_blocks,)
            phi_u = xp.unwrap(phi_b.astype(xp.float64) * 4, axis=-1) / 4
            if cycle_slip_correction:
                phi_u_np = correct_cycle_slips(
                    to_device(phi_u, "cpu"), 4, cycle_slip_history, cycle_slip_threshold
                )
                phi_u = xp.asarray(phi_u_np)
            phi_full[ch] = (
                phi_u[idx_left] * (1.0 - t_interp) + phi_u[idx_right] * t_interp
            )
            phi_blocks[ch] = phi_u

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    mode_str = "joint" if (joint_channels and C > 1) else "independent"
    logger.info(
        f"CPR (BPS, B={B}, {mode_str}): phase mean={phi_mean_deg:.2f}°, std={phi_std_deg:.2f}° "
        f"[{N_blocks} blocks × {block_size} symbols, C={C}, "
        f"cycle_slip_correction={cycle_slip_correction}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full_np,
            block_centers=to_device(block_centers, "cpu"),
            phi_blocks=to_device(phi_blocks, "cpu"),
            show=True,
            title="CPR — Blind Phase Search",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def _rts_smoother_1d(
    phi_obs: np.ndarray,
    sigma_p2: float,
    sigma_v2: float,
) -> np.ndarray:
    """Rauch-Tung-Striebel (RTS) Kalman smoother for a 1-D random-walk state.

    Uses the Numba-compiled kernel (:func:`_get_numba_rts_smoother`) when
    available; falls back to a pure-Python loop otherwise.  Always runs on
    CPU — call with a NumPy array; the caller is responsible for
    ``to_device`` conversion.

    State model  : x[k+1] = x[k] + w[k],   w ~ N(0, sigma_p2)
    Observation  : y[k]   = x[k] + v[k],   v ~ N(0, sigma_v2)

    Parameters
    ----------
    phi_obs : (B,) float64
        Noisy block-phase observations in radians (e.g. from VV).
    sigma_p2 : float
        Process noise variance per block (Wiener phase noise increment).
    sigma_v2 : float
        Observation noise variance (VV estimator variance per block).

    Returns
    -------
    (B,) float64
        MAP-smoothed phase trajectory.
    """
    return _get_numba_rts_smoother()(phi_obs, float(sigma_p2), float(sigma_v2))


def _sskf_smoother_1d(
    phi_obs: ArrayType,
    sigma_p2: float,
    sigma_v2: float,
    sp,
    xp,
) -> ArrayType:
    """Steady-state Kalman smoother via zero-phase IIR filter (filtfilt).

    Approximates the RTS smoother by replacing the sequential Kalman
    recurrence with a 1st-order IIR filter whose gain is solved analytically
    from the discrete algebraic Riccati equation.  The bidirectional
    ``filtfilt`` call makes it equivalent to the RTS smoother in steady state.

    Backend-aware: uses ``sp.signal.filtfilt`` where ``sp`` is
    ``scipy`` (CPU) or ``cupyx.scipy`` (GPU) as returned by
    :func:`~commstools.backend.dispatch`.

    The approximation is excellent when ``B >> 1/K_∞``
    (typically ``B > 20``).  For ``B < 7`` (``filtfilt`` minimum), falls
    back to the exact :func:`_rts_smoother_1d` on CPU.

    Parameters
    ----------
    phi_obs : (B,) float64, on the target device
        Noisy block-phase observations in radians.
    sigma_p2, sigma_v2 : float
        Process and observation noise variances per block.
    sp : module
        ``scipy`` or ``cupyx.scipy``, from :func:`~commstools.backend.dispatch`.
    xp : module
        ``numpy`` or ``cupy``, from :func:`~commstools.backend.dispatch`.

    Returns
    -------
    (B,) float64, same device as ``phi_obs``.
    """
    # filtfilt requires at least padlen * 2 + 1 samples; padlen = 3 * max(len(b), len(a)) = 6
    if len(phi_obs) < 7:
        phi_np = to_device(phi_obs, "cpu")
        return xp.asarray(_rts_smoother_1d(phi_np, sigma_p2, sigma_v2))

    # Steady-state prediction error covariance from discrete Riccati equation:
    #   p² - σ_p²·p - σ_p²·σ_v² = 0  →  p = (σ_p² + √(σ_p⁴ + 4σ_p²σ_v²)) / 2
    p_ss = (sigma_p2 + float(np.sqrt(sigma_p2**2 + 4.0 * sigma_p2 * sigma_v2))) / 2.0
    K_ss = p_ss / (p_ss + sigma_v2)

    # Forward IIR:  y[k] = (1-K)·y[k-1] + K·x[k]
    #   H(z) = K / (1 - (1-K)·z⁻¹)
    # filtfilt applies forward + backward  →  zero-phase, ≡ RTS smoother at
    # steady state.
    b = [K_ss]
    a = [1.0, -(1.0 - K_ss)]
    return sp.signal.filtfilt(b, a, phi_obs)


def recover_carrier_phase_tikhonov(
    symbols: ArrayType,
    modulation: str,
    order: int,
    linewidth_symbol_periods: float,
    block_size: int = 32,
    snr_db: Optional[float] = None,
    method: str = "exact",
    joint_channels: bool = False,
    cycle_slip_correction: bool = True,
    cycle_slip_history: int = 1000,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    r"""
    Carrier phase recovery via MAP estimation with a Tikhonov/Wiener phase
    noise prior (Colavolpe et al., 2005).

    Extends the Viterbi-Viterbi block estimator with a Kalman smoother
    matched to the laser phase noise statistics.  Two smoother backends are
    available via ``method``:

    * ``'exact'`` — full Rauch-Tung-Striebel (RTS) smoother; Numba-compiled.
      Exact for all sequence lengths; runs on CPU.
    * ``'sskf'`` — steady-state Kalman filter approximation via zero-phase
      IIR (``filtfilt``); backend-aware (stays on GPU when input is on GPU).
      Approximation holds for ``N_blocks >> 1/K_∞`` (~20+ blocks typical).

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filter and FOE.
        Shape: ``(N,)`` or ``(C, N)``.
    modulation : str
        Modulation scheme (case-insensitive): ``'psk'``, ``'qam'``, etc.
    order : int
        Modulation order.
    linewidth_symbol_periods : float
        Combined linewidth-symbol-time product :math:`\Delta\nu \cdot T_s`.
        Typical values: ``1e-5`` (narrow laser, 32 GBd), ``5e-4`` (wide
        laser / high baud rate).  Sets the Kalman process noise variance:
        :math:`\sigma_p^2 = 2\pi \cdot \Delta\nu T_s \cdot N_b`.
    block_size : int, default 32
        Symbols per VV estimation block.  Same trade-off as for
        :func:`recover_carrier_phase_viterbi_viterbi`.
    snr_db : float or None, default None
        Per-symbol SNR in dB.  Used to compute the VV observation noise
        variance :math:`\sigma_v^2 \approx 1/(M^2 \cdot \mathrm{SNR} \cdot N_b)`.
        If ``None``, defaults to 20 dB with a warning — provide the actual
        operating SNR for the optimal smoother bandwidth.
    method : {'exact', 'sskf'}, default 'exact'
        Smoother implementation:

        * ``'exact'``: full RTS smoother (:func:`_rts_smoother_1d`); Numba
          kernel when available.  Sequential CPU recurrence; exact for any
          ``N_blocks``.
        * ``'sskf'``: steady-state approximation via ``filtfilt``
          (:func:`_sskf_smoother_1d`); runs on the input device (GPU-native
          when data is on GPU).  Excellent for ``N_blocks ≥ 20``; for
          ``N_blocks < 7`` silently falls back to ``'exact'``.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the M-th-power block
        phasors across all channels before the VV phase extraction and
        Kalman smoother.  The single smoothed trajectory is broadcast to
        all C output rows.  Reduces variance by ~√C for shared-LO systems.
    cycle_slip_correction : bool, default True
        If ``True``, apply cycle-slip detection and correction
        (:func:`correct_cycle_slips`) after the Kalman smoother, before
        interpolation.
    cycle_slip_history : int, default 1000
        ``history_length`` passed to :func:`correct_cycle_slips`.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-symbol phase
        trajectory with the Kalman-smoothed block phases.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians.  Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    **Algorithm:**

    1. Compute VV block phases using normalized M-th power (unit-circle
       projection before raising to the M-th power removes QAM amplitude
       bias; see :func:`recover_carrier_phase_viterbi_viterbi`).
    2. Apply the Kalman smoother with:

       .. math::

           \sigma_p^2 &= 2\pi \cdot \Delta\nu T_s \cdot N_b \\
           \sigma_v^2 &\approx \frac{1}{M^2 \cdot \mathrm{SNR} \cdot N_b}

       where :math:`N_b` = ``block_size`` and :math:`M` is the modulation
       exponent from :func:`_modulation_power_m`.
    3. Interpolate smoothed block phases to per-symbol resolution (linear,
       consistent with VV).

    **M-fold ambiguity:** same as VV — a residual ``2π/M`` phase offset
    always remains.  Resolve via a pilot or preamble reference.

    References
    ----------
    G. Colavolpe, A. Barbieri, and G. Caire, "Algorithms for iterative
    decoding in the presence of strong phase noise," *IEEE J. Sel. Areas
    Commun.*, vol. 23, no. 9, pp. 1748-1757, Sep. 2005.

    A. J. Viterbi and A. M. Viterbi, "Nonlinear estimation of PSK-modulated
    carrier phase with application to burst digital transmission," *IEEE
    Trans. Inf. Theory*, 1983.
    """
    if method not in ("exact", "sskf"):
        raise ValueError(f"Unknown method {method!r}. Choose 'exact' or 'sskf'.")

    symbols, xp, sp = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    M = _modulation_power_m(modulation, order)

    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    if N_blocks == 0:
        raise ValueError(
            f"Signal length {N} is shorter than block_size={block_size}. "
            "Reduce block_size or use a longer symbol sequence."
        )

    # Same data-residual constraint as VV: for QAM with order > 4 the M-th power
    # does not cancel per symbol.  Block phase variance can exceed π/M before the
    # Kalman smoother is applied, causing unwrap slips that the smoother cannot fix.
    if "qam" in modulation.lower() and order > 4:
        _min_bs = max(8, 4 * int(np.ceil(order**0.5)))
        if block_size < _min_bs:
            logger.warning(
                f"CPR (Tikhonov): block_size={block_size} is too small for {order}-QAM. "
                f"Block phases are estimated via Viterbi-Viterbi; the data-residual "
                f"constraint is identical — see recover_carrier_phase_viterbi_viterbi. "
                f"Recommended minimum for {order}-QAM: block_size ≥ {_min_bs}."
            )

    # Smoother noise parameters
    if snr_db is None:
        logger.warning(
            "CPR (Tikhonov): snr_db not provided — defaulting to 20 dB. "
            "Pass the operating SNR for the optimal smoother bandwidth."
        )
        snr_lin = 100.0  # 20 dB default
    else:
        snr_lin = 10.0 ** (snr_db / 10.0)

    sigma_p2 = float(2.0 * np.pi * linewidth_symbol_periods * block_size)
    sigma_v2 = float(1.0 / (M**2 * snr_lin * block_size))

    # VV block phase estimation with unit-circle normalisation for QAM
    blocks = symbols[:, :N_trunc].reshape(C, N_blocks, block_size)
    blocks_c = blocks.astype(
        xp.complex128 if blocks.dtype == xp.complex64 else blocks.dtype
    )
    if "qam" in modulation.lower():
        mag = xp.abs(blocks_c)
        blocks_c = blocks_c / xp.maximum(mag, 1e-15 * xp.max(mag))

    S_b = xp.sum(blocks_c**M, axis=-1)  # (C, N_blocks)

    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)
    phi_full = xp.zeros((C, N), dtype=xp.float64)

    if joint_channels and C > 1:
        # Sum M-th-power phasors → single VV estimate → single Kalman pass
        S_b_joint = xp.sum(S_b, axis=0)  # (N_blocks,)
        phi_raw_joint = xp.angle(S_b_joint) / M
        phi_u_joint = xp.unwrap((phi_raw_joint * M).astype(xp.float64)) / M
        if "qam" in modulation.lower():
            phi_u_joint = phi_u_joint - (np.pi / M)

        # Kalman smoother on the joint trajectory
        if method == "exact":
            phi_u_joint_np = to_device(phi_u_joint, "cpu")
            phi_smooth_joint_np = _rts_smoother_1d(phi_u_joint_np, sigma_p2, sigma_v2)
            phi_smooth_joint = xp.asarray(phi_smooth_joint_np)
        else:
            phi_smooth_joint = _sskf_smoother_1d(
                phi_u_joint, sigma_p2, sigma_v2, sp, xp
            )
            phi_smooth_joint_np = to_device(phi_smooth_joint, "cpu")

        if cycle_slip_correction:
            phi_smooth_joint_np = correct_cycle_slips(
                to_device(phi_smooth_joint, "cpu"),
                4,
                cycle_slip_history,
                cycle_slip_threshold,
            )
            phi_smooth_joint = xp.asarray(phi_smooth_joint_np)

        phi_interp = xp.interp(all_positions, block_centers, phi_smooth_joint)
        for ch in range(C):
            phi_full[ch] = phi_interp
        phi_smooth_np = np.tile(to_device(phi_smooth_joint, "cpu"), (C, 1))
    else:
        phi_raw = xp.angle(S_b) / M
        phi_u = (
            xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M
        )  # (C, N_blocks)

        if "qam" in modulation.lower():
            phi_u = phi_u - (np.pi / M)

        if C > 1:
            for ch in range(1, C):
                diff = float(xp.mean(phi_u[ch] - phi_u[0]))
                k = round(diff * M / (2 * np.pi))
                phi_u[ch] = phi_u[ch] - k * (2 * np.pi / M)

        # Kalman smoother — dispatch on method
        if method == "exact":
            phi_u_np = to_device(phi_u, "cpu")  # (C, N_blocks) float64
            phi_smooth_np = np.empty_like(phi_u_np)
            for ch in range(C):
                phi_smooth_np[ch] = _rts_smoother_1d(phi_u_np[ch], sigma_p2, sigma_v2)
            phi_smooth = xp.asarray(phi_smooth_np)
        else:  # method == "sskf"
            phi_smooth = xp.empty_like(phi_u)
            for ch in range(C):
                phi_smooth[ch] = _sskf_smoother_1d(
                    phi_u[ch], sigma_p2, sigma_v2, sp, xp
                )
            phi_smooth_np = to_device(phi_smooth, "cpu")

        for ch in range(C):
            phi_s_ch = phi_smooth[ch]
            if cycle_slip_correction:
                phi_s_ch_np = correct_cycle_slips(
                    to_device(phi_s_ch, "cpu"),
                    4,
                    cycle_slip_history,
                    cycle_slip_threshold,
                )
                phi_s_ch = xp.asarray(phi_s_ch_np)
                phi_smooth_np[ch] = to_device(phi_s_ch, "cpu")
            phi_full[ch] = xp.interp(all_positions, block_centers, phi_s_ch)

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    mode_str = "joint" if (joint_channels and C > 1) else "independent"
    logger.info(
        f"CPR (Tikhonov-{method.upper()}, M={M}, {mode_str}): "
        f"phase mean={phi_mean_deg:.2f}°, std={phi_std_deg:.2f}° "
        f"[{N_blocks} blocks × {block_size}, σ_p²={sigma_p2:.2e}, σ_v²={sigma_v2:.2e}, "
        f"C={C}, cycle_slip_correction={cycle_slip_correction}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full_np,
            block_centers=to_device(block_centers, "cpu"),
            phi_blocks=phi_smooth_np,
            show=True,
            title=f"CPR — Tikhonov-{method.upper()}",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def recover_carrier_phase_pilots(
    symbols: ArrayType,
    pilot_indices: ArrayType,
    pilot_values: ArrayType,
    interpolation: str = "linear",
    joint_channels: bool = False,
    cycle_slip_correction: bool = True,
    cycle_slip_history: int = 1000,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Carrier phase recovery using known pilot symbols.

    Computes the phase error at each pilot position, unwraps the pilot
    phase sequence, and interpolates across the full symbol grid.

    Parameters
    ----------
    symbols : array_like
        Received 1-SPS complex symbols. Shape: (N,) or (C, N).
    pilot_indices : array_like of int
        Indices of pilot symbols within the frame, in increasing order.
        Shape: (P,).
    pilot_values : array_like
        Known transmitted pilot constellation points.
        Shape: (P,) for shared pilots (broadcast to all MIMO channels),
        or (C, P) for per-channel pilots.
    interpolation : {'linear', 'cubic'}, default 'linear'
        Interpolation method between pilot positions.  Both modes loop over
        MIMO channels (``xp.interp`` and ``CubicSpline`` are 1D-only);
        C is typically 1-4 so the overhead is negligible.  ``'cubic'`` uses
        :class:`scipy.interpolate.CubicSpline` (CPU) or
        :class:`cupyx.scipy.interpolate.CubicSpline` (GPU) with natural
        boundary conditions (zero second derivative at endpoints) and
        constant-hold extrapolation outside the pilot span.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, perform coherent complex
        averaging of ``r_pilot * conj(s_pilot)`` across all channels before
        calling ``angle()``.  This avoids wrap-around artefacts that arise
        when averaging phases directly, and reduces variance by ~√C for
        shared-LO systems.  The resulting single phase trajectory is broadcast
        to all C output rows.  Has no effect for SISO (C = 1).
    cycle_slip_correction : bool, default True
        If ``True``, apply :func:`correct_cycle_slips` to the unwrapped pilot
        phase sequence before interpolation, with ``symmetry=1`` (correction
        quantum ``2π``) to detect and fix wrap-around errors introduced by
        ``xp.unwrap`` at large inter-pilot gaps.
    cycle_slip_history : int, default 1000
        ``history_length`` passed to :func:`correct_cycle_slips`.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the unwrapped pilot
        phase sequence and the interpolated phase trajectory.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians. Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    Phase at pilot position :math:`k`:

    .. math::
        \\hat{\\phi}[k] = \\angle\\!\\left(
            r[\\mathrm{pilot\\_indices}[k]] \\cdot
            s^*[\\mathrm{pilot\\_values}[k]]
        \\right)

    For ``'linear'``, boundary extrapolation holds the first/last pilot value
    (constant hold — identical to ``numpy.interp`` behaviour).  For
    ``'cubic'``, the spline uses natural boundary conditions (zero second
    derivative at endpoints), and symbols before the first pilot or after
    the last pilot are filled with the respective boundary pilot value,
    preventing edge oscillation.

    .. note::
        **Single-carrier use only.**  This function tracks carrier phase across
        a linear symbol stream using scattered pilot positions.  For OFDM
        systems, phase noise is tracked as *common phase error* (CPE) across
        pilot *subcarriers* within each OFDM symbol (e.g. 5G NR PTRS,
        DVB-T2 continual pilots) — a structurally different problem not
        covered here.

    References
    ----------
    S. J. Savory, "Digital filters for coherent optical receivers,"
    Optics Express, 2008.
    """
    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    pilot_indices_np = to_device(pilot_indices, "cpu").astype(np.intp)
    pilot_indices_xp = xp.asarray(pilot_indices, dtype=xp.float64)
    pilot_values_xp = xp.asarray(pilot_values)
    P = len(pilot_indices_np)

    # Broadcast shared pilots (P,) → (C, P) for all channels
    if pilot_values_xp.ndim == 1:
        pilot_values_xp = xp.broadcast_to(pilot_values_xp[None, :], (C, P))

    # Phase at each pilot position: angle(r_pilot · conj(s_pilot))
    r_pilots = symbols[:, pilot_indices_np]  # (C, P)

    if joint_channels and C > 1:
        # Coherent complex averaging before angle() — avoids wrap-around artefacts
        # that arise from averaging phases directly (e.g. antipodal channels).
        z_joint = xp.mean(r_pilots * xp.conj(pilot_values_xp), axis=0)  # (P,)
        phi_joint_u = xp.unwrap(xp.angle(z_joint).astype(xp.float64))  # (P,)
        if cycle_slip_correction:
            phi_joint_np = to_device(phi_joint_u, "cpu")
            phi_joint_np = correct_cycle_slips(
                phi_joint_np,
                symmetry=1,
                history_length=cycle_slip_history,
                threshold=cycle_slip_threshold,
            )
            phi_joint_u = xp.asarray(phi_joint_np)
        # Broadcast to (C, P) — read-only, downstream code only reads phi_pilots_u[ch]
        phi_pilots_u = xp.broadcast_to(phi_joint_u[None, :], (C, P))
    else:
        phi_pilots = xp.angle(r_pilots * xp.conj(pilot_values_xp))  # (C, P)
        # Unwrap along the pilot axis in float64 (cp.unwrap preserves input dtype;
        # casting before avoids precision loss in the discontinuity test for float32 input)
        phi_pilots_u = xp.unwrap(phi_pilots.astype(xp.float64), axis=-1)  # (C, P)
        if cycle_slip_correction:
            phi_pilots_u_np = to_device(phi_pilots_u, "cpu")
            for ch in range(C):
                phi_pilots_u_np[ch] = correct_cycle_slips(
                    phi_pilots_u_np[ch],
                    symmetry=1,
                    history_length=cycle_slip_history,
                    threshold=cycle_slip_threshold,
                )
            phi_pilots_u = xp.asarray(phi_pilots_u_np)

    all_positions = xp.arange(N, dtype=xp.float64)

    if interpolation == "linear":
        # xp.interp handles non-uniform pilot spacing natively, is boundary-safe
        # (extrapolates with first/last pilot value), and avoids the divide-by-zero
        # guards that the searchsorted form required.  Loop over C channels because
        # xp.interp is 1D-only; overhead is negligible for typical C = 1–4.
        phi_full = xp.zeros((C, N), dtype=xp.float64)
        for ch in range(C):
            phi_full[ch] = xp.interp(all_positions, pilot_indices_xp, phi_pilots_u[ch])

    elif interpolation == "cubic":
        # CubicSpline is inherently per-channel (1D y input); loop is unavoidable.
        # Both scipy (CPU) and cupyx.scipy (GPU) share the same API.
        phi_full = xp.zeros((C, N), dtype=xp.float64)
        if xp is not np:
            from cupyx.scipy.interpolate import CubicSpline
        else:
            from scipy.interpolate import CubicSpline

        for ch in range(C):
            phi_ch = phi_pilots_u[ch]  # already float64
            cs = CubicSpline(pilot_indices_xp, phi_ch, bc_type="natural")
            # Evaluate the spline only within the pilot span; constant-hold outside.
            first_idx = int(pilot_indices_np[0])
            last_idx = int(pilot_indices_np[-1])
            phi_full[ch, first_idx : last_idx + 1] = cs(
                all_positions[first_idx : last_idx + 1]
            )
            if first_idx > 0:
                phi_full[ch, :first_idx] = phi_ch[0]
            if last_idx < N - 1:
                phi_full[ch, last_idx + 1 :] = phi_ch[-1]

    else:
        raise ValueError(
            f"Unknown interpolation method: {interpolation!r}. "
            "Choose 'linear' or 'cubic'."
        )

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    logger.info(
        f"CPR (pilot-aided, {interpolation}): phase mean={phi_mean_deg:.2f}°, "
        f"std={phi_std_deg:.2f}° [P={P} pilots, C={C}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        phi_pilots_u_np = to_device(phi_pilots_u, "cpu")
        _plotting.pilot_phase_estimate(
            pilot_indices=pilot_indices_np,
            phi_pilots_u=phi_pilots_u_np,
            phi_full=phi_full_np,
            show=True,
            title="CPR — Pilot-Aided Phase",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def correct_carrier_phase(
    symbols: ArrayType,
    phase_vector: ArrayType,
) -> ArrayType:
    """
    Applies carrier phase correction to a symbol sequence.

    Rotates each symbol by the negative of the estimated phase to cancel
    the carrier phase offset:

    .. math::
        y[n] = s[n] \\cdot e^{-j\\,\\hat{\\phi}[n]}

    Parameters
    ----------
    symbols : array_like
        Complex symbols. Shape: (N,) or (C, N).
    phase_vector : array_like
        Per-symbol phase estimates in radians.  Shape: (N,) for SISO, or
        broadcastable to ``symbols.shape`` for MIMO.

    Returns
    -------
    array_like
        Phase-corrected symbols, same shape and dtype as ``symbols``.
    """
    symbols, xp, _ = dispatch(symbols)
    logger.debug(f"Applying carrier phase correction: shape={symbols.shape}")
    phase_vector_xp = xp.asarray(phase_vector)
    phasor = xp.exp(-1j * phase_vector_xp)
    if phasor.dtype != symbols.dtype:
        phasor = phasor.astype(symbols.dtype)
    return symbols * phasor


# -----------------------------------------------------------------------------
# Cycle-slip correction
# -----------------------------------------------------------------------------

_NUMBA_CYCLE_SLIP: dict = {}


def _get_numba_cycle_slip():
    """JIT-compile and cache the Numba cycle-slip correction kernel.

    Returns
    -------
    callable
        Numba-compiled ``_cycle_slip_loop``.
    """
    if "cs" not in _NUMBA_CYCLE_SLIP:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _cycle_slip_loop(phi_u, symmetry, history_length, threshold):
            """Cycle-slip detection and correction via linear extrapolation.

            Scans the block-phase trajectory ``phi_u`` sequentially.  For each
            block, linearly extrapolates from up to ``history_length`` past
            *corrected* blocks.  When the deviation exceeds ``threshold``, a
            ``π/2`` step correction is applied.

            The linear regression is maintained in O(1) per step using
            incremental sufficient statistics (Welford-style running sums):
            ``Sx``, ``Sy``, ``Sxx``, ``Sxy``, ``n``.

            Parameters
            ----------
            phi_u : (B,) float64
                Block-phase trajectory after M-fold unwrap (modified in place).
            symmetry : int
                Rotational symmetry order; correction quantum = ``2π/symmetry``.
                Pass 4 for QAM (all BPS, VV, Tikhonov use 4-fold symmetry).
            history_length : int
                Maximum number of past corrected blocks used for extrapolation.
                Use ``min(b, history_length)`` at each step.
            threshold : float64
                Deviation from extrapolated value that triggers a correction
                (radians).  Default in the caller: ``π/4``.

            Returns
            -------
            (B,) float64
                Corrected block-phase trajectory (same array, modified in place).
            """
            two_pi = 2.0 * np.pi
            quantum = two_pi / float(symmetry)
            B = len(phi_u)

            # Incremental linear regression state
            # We store the window as a circular buffer of (x, y) pairs.
            buf_x = np.empty(history_length, dtype=np.float64)
            buf_y = np.empty(history_length, dtype=np.float64)
            buf_head = 0  # index of next write position (circular)
            n_buf = 0  # current number of valid entries

            Sx = 0.0
            Sy = 0.0
            Sxx = 0.0
            Sxy = 0.0

            for b in range(B):
                x_b = float(b)
                y_b = phi_u[b]

                if n_buf == 0:
                    # First block: trust it unconditionally
                    buf_x[buf_head % history_length] = x_b
                    buf_y[buf_head % history_length] = y_b
                    buf_head += 1
                    n_buf += 1
                    Sx += x_b
                    Sy += y_b
                    Sxx += x_b * x_b
                    Sxy += x_b * y_b
                    continue

                if n_buf < min(10, history_length):
                    # Zero-slope (constant) extrapolation for first few blocks
                    # Prevents noisy early blocks from cementing false slips
                    phi_pred = buf_y[(buf_head - 1) % history_length]
                else:
                    # Linear extrapolation: slope = (n·Sxy − Sx·Sy) / (n·Sxx − Sx²)
                    n_f = float(n_buf)
                    denom = n_f * Sxx - Sx * Sx
                    if abs(denom) > 1e-30:
                        slope = (n_f * Sxy - Sx * Sy) / denom
                        intercept = (Sy - slope * Sx) / n_f
                    else:
                        slope = 0.0
                        intercept = Sy / n_f
                    phi_pred = slope * x_b + intercept

                diff = y_b - phi_pred
                # Round to nearest correction quantum
                k = round(diff / quantum)
                if abs(diff) > threshold and k != 0:
                    phi_u[b] -= float(k) * quantum
                    y_b = phi_u[b]

                # Add corrected value to circular buffer, evicting oldest if full
                if n_buf == history_length:
                    # Evict the oldest entry
                    old_idx = buf_head % history_length
                    ox = buf_x[old_idx]
                    oy = buf_y[old_idx]
                    Sx -= ox
                    Sy -= oy
                    Sxx -= ox * ox
                    Sxy -= ox * oy
                    buf_x[old_idx] = x_b
                    buf_y[old_idx] = y_b
                    Sx += x_b
                    Sy += y_b
                    Sxx += x_b * x_b
                    Sxy += x_b * y_b
                    buf_head += 1
                else:
                    idx = buf_head % history_length
                    buf_x[idx] = x_b
                    buf_y[idx] = y_b
                    Sx += x_b
                    Sy += y_b
                    Sxx += x_b * x_b
                    Sxy += x_b * y_b
                    buf_head += 1
                    n_buf += 1

            return phi_u

        _NUMBA_CYCLE_SLIP["cs"] = _cycle_slip_loop

    return _NUMBA_CYCLE_SLIP["cs"]


def correct_cycle_slips(
    phi_u: np.ndarray,
    symmetry: int = 4,
    history_length: int = 1000,
    threshold: float = np.pi / 4,
) -> np.ndarray:
    """
    Detects and corrects cycle slips in a block-phase trajectory.

    After ``xp.unwrap`` resolves the M-fold ambiguity, residual cycle slips
    may remain where the unwrapper chose the wrong quadrant.  This function
    scans the trajectory sequentially: for each block it extrapolates the
    expected phase from up to ``history_length`` past corrected blocks using
    a linear fit.  When the deviation exceeds ``threshold``, the block is
    corrected by the nearest integer multiple of ``2π/symmetry``.

    Algorithm: linear extrapolation from the previous
    ``history_length`` corrected phases; correction quantum = ``π/2`` for
    4-fold QAM symmetry; threshold = ``π/4``.

    Parameters
    ----------
    phi_u : (B,) float64
        Block-phase trajectory on CPU after M-fold unwrap (e.g. output of
        ``xp.unwrap(phi_raw * M) / M``).  **Modified in place.**
    symmetry : int, default 4
        Rotational symmetry order of the constellation.  Correction quantum
        is ``2π/symmetry``.  Use 4 for all square QAM constellations and BPS
        (which always searches over ``[0, π/2)``).  For M-PSK use ``symmetry = M``.
    history_length : int, default 1000
        Number of past corrected blocks used for linear extrapolation.
        Reduce for short bursts.
    threshold : float, default π/4
        Deviation from the extrapolated phase that triggers a correction.
        ``π/4`` is the midpoint between adjacent correction quanta for 4-fold
        symmetry.

    Returns
    -------
    (B,) float64
        Corrected block-phase trajectory (same NumPy array).

    Notes
    -----
    Runs on CPU only (sequential scan; Numba-compiled).
    The caller should transfer ``phi_u`` to CPU before calling and move
    the result back to the device if needed.
    """
    phi_u = np.asarray(phi_u, dtype=np.float64)
    kernel = _get_numba_cycle_slip()
    return kernel(phi_u, int(symmetry), int(history_length), float(threshold))


# -----------------------------------------------------------------------------
# Phase ambiguity resolution
# -----------------------------------------------------------------------------


def resolve_phase_ambiguity(
    symbols: ArrayType,
    ref_symbols: ArrayType,
    modulation: str,
    order: int,
    symmetry_order: Optional[int] = None,
) -> ArrayType:
    """
    Resolves rotational phase ambiguity after blind carrier phase recovery.

    Blind CPR methods (VV, BPS, Tikhonov) cannot distinguish between
    ``symmetry_order`` rotational copies of the constellation.  This function
    tests all candidate rotations, scores each by Symbol Error Rate (SER)
    against the known transmitted symbols, and returns the symbols rotated by
    the best candidate.

    For MIMO inputs each channel is resolved independently — after MIMO
    equalisation the output streams may land on different ambiguity branches.

    Parameters
    ----------
    symbols : array_like
        Received complex symbols after CPR and ``correct_carrier_phase``.
        Shape: ``(N,)`` or ``(C, N)``.
    ref_symbols : array_like
        Known transmitted symbols (unit-average-power normalised).
        Shape: ``(N,)`` or ``(C, N)``.
    modulation : str
        Modulation scheme (case-insensitive): ``'qam'``, ``'psk'``, etc.
    order : int
        Modulation order.
    symmetry_order : int, optional
        Number of rotationally equivalent constellation copies to test.
        Defaults to 4 for QAM (4-fold ``π/2`` symmetry) and ``order`` for
        PSK.  Override for non-standard constellations.

    Returns
    -------
    array_like
        Phase-ambiguity-resolved symbols, same shape and dtype as ``symbols``.
    """
    from .metrics import ser as _ser

    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    ref = xp.asarray(ref_symbols)
    if ref.ndim == 1:
        ref = ref[None, :]
    if ref.shape[0] == 1 and C > 1:
        ref = xp.broadcast_to(ref, (C, N))

    if symmetry_order is None:
        symmetry_order = 4 if "qam" in modulation.lower() else order

    step = 2.0 * np.pi / symmetry_order
    candidates = [
        xp.exp(1j * k * step).astype(symbols.dtype) for k in range(symmetry_order)
    ]

    out = xp.empty_like(symbols)
    for ch in range(C):
        best_k = 0
        best_ser = float("inf")
        for k, rot in enumerate(candidates):
            rotated = symbols[ch] * rot
            s = float(xp.mean(xp.asarray(_ser(rotated, ref[ch], modulation, order))))
            if s < best_ser:
                best_ser = s
                best_k = k
        out[ch] = symbols[ch] * candidates[best_k]
        logger.info(
            f"Phase ambiguity resolution: ch={ch}, best_k={best_k}, "
            f"rotation={best_k * step * 180.0 / np.pi:.1f}°, SER={best_ser:.4f}"
        )

    if was_1d:
        return out[0]
    return out


def compensate_iq_imbalance_lowdin(samples: ArrayType) -> ArrayType:
    """
    Blind IQ imbalance compensation via Löwdin symmetric orthogonalisation.

    Treats the I and Q components as a 2-D real vector and applies the
    symmetric whitening transform :math:`W = M^{-1/2}` (where *M* is the
    :math:`2 \\times 2` second-moment matrix) so that the corrected I and Q
    channels have equal power and zero cross-correlation.  Unlike
    Gram-Schmidt, the transform is symmetric: both branches are adjusted
    equally, minimising the total distortion introduced.

    The output power equals the input power.

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).

    Returns
    -------
    array_like
        IQ-corrected signal, same shape and dtype as input.

    Notes
    -----
    *Algorithm* (per channel):

    1. Form the :math:`2 \\times N` real data matrix :math:`X = [I;\\ Q]`.
    2. Compute the :math:`2 \\times 2` second-moment matrix
       :math:`M = X X^\\top / N`.
    3. Factorise :math:`M = V \\Lambda V^\\top` (symmetric eigendecomposition).
    4. Apply the whitening matrix :math:`W = V \\operatorname{diag}(\\lambda^{-1/2}) V^\\top`.
    5. Rescale the output to restore the original signal power.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.5, phase_imbalance_deg=4.0)
    >>> s_hat = compensate_iq_imbalance_lowdin(r)
    """
    logger.info("Applying Löwdin IQ imbalance compensation.")

    samples, xp, _ = dispatch(samples)

    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[xp.newaxis, :]  # (1, N)

    C, N = samples.shape
    result = xp.empty_like(samples)

    for ch in range(C):
        r = samples[ch]  # (N,)
        P_in = xp.mean(xp.abs(r) ** 2)

        # 2×N real data matrix: rows = [I, Q]
        X = xp.stack([r.real, r.imag])  # (2, N)

        # 2×2 second-moment matrix
        M = (X @ X.T) / N  # (2, 2)

        # Symmetric whitening: W = M^{-1/2} = V @ diag(1/sqrt(lam)) @ V.T
        lam, V = xp.linalg.eigh(M)  # lam: (2,), V: (2, 2)
        W = (V * (1.0 / xp.sqrt(lam))) @ V.T  # (2, 2)

        # Apply whitening — X_corr has identity second-moment matrix
        X_corr = W @ X  # (2, N)

        # Restore input power: E[|s_hat|^2] = P_in
        s_corr = (X_corr[0] + 1j * X_corr[1]) * xp.sqrt(P_in / 2.0)

        if s_corr.dtype != samples.dtype:
            s_corr = s_corr.astype(samples.dtype)

        result[ch] = s_corr

    if was_1d:
        return result[0]
    return result


def compensate_iq_imbalance_gram_schmidt(samples: ArrayType) -> ArrayType:
    """
    Blind IQ imbalance compensation via Gram-Schmidt sequential orthogonalisation.

    Uses the I branch as the reference axis.  The Q branch is orthogonalised
    against I and both are normalised to unit RMS before being recombined.
    This is the classical GSOP approach used in analogue front-end calibration.

    The output power equals the input power.

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).

    Returns
    -------
    array_like
        IQ-corrected signal, same shape and dtype as input.

    Notes
    -----
    *Algorithm* (per channel):

    1. Normalise I to unit RMS: :math:`\\hat{I} = I / \\sigma_I`.
    2. Remove I-projection from Q: :math:`Q_\\perp = Q - \\langle \\hat{I}, Q \\rangle \\hat{I}`.
    3. Normalise :math:`Q_\\perp` to unit RMS: :math:`\\hat{Q} = Q_\\perp / \\sigma_{Q_\\perp}`.
    4. Recombine and rescale to preserve input power.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.5, phase_imbalance_deg=4.0)
    >>> s_hat = compensate_iq_imbalance_gram_schmidt(r)
    """
    logger.info("Applying Gram-Schmidt IQ imbalance compensation.")

    samples, xp, _ = dispatch(samples)

    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[xp.newaxis, :]  # (1, N)

    C, N = samples.shape
    result = xp.empty_like(samples)

    for ch in range(C):
        r = samples[ch]  # (N,)
        P_in = xp.mean(xp.abs(r) ** 2)

        I = r.real
        Q = r.imag

        # Step 1: Normalise I (reference branch)
        sigma_I = xp.sqrt(xp.mean(I**2))
        I_norm = I / sigma_I

        # Step 2: Orthogonalise Q against I
        rho = xp.mean(I_norm * Q)  # scalar projection coefficient
        Q_orth = Q - rho * I_norm

        # Step 3: Normalise orthogonalised Q
        sigma_Q = xp.sqrt(xp.mean(Q_orth**2))
        Q_norm = Q_orth / sigma_Q

        # Step 4: Recombine and restore input power
        s_corr = (I_norm + 1j * Q_norm) * xp.sqrt(P_in / 2.0)

        if s_corr.dtype != samples.dtype:
            s_corr = s_corr.astype(samples.dtype)

        result[ch] = s_corr

    if was_1d:
        return result[0]
    return result
