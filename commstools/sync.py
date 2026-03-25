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
    preamble correlation and parabolic interpolation.
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
recover_carrier_phase_decision_directed :
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
    preamble: Optional[Union[ArrayType, "Preamble"]] = None,
    threshold: float = 0.5,
    sps: Optional[int] = None,
    pulse_shape: Optional[str] = None,
    filter_params: Optional[dict] = None,
    search_range: Optional[Tuple[int, int]] = None,
    dft_upsample: int = 1,
    fractional_method: str = "log-parabolic",
    debug_plot: bool = False,
) -> Tuple[ArrayType, ArrayType]:
    """
    Estimates coarse and fractional timing offsets via preamble correlation.

    Performs a sliding cross-correlation between the received signal and
    the expected preamble to determine the coarse (integer sample) timing
    offset per channel.  Additionally estimates the fractional (sub-sample)
    timing offset using parabolic interpolation on the correlation peak.

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
        Received signal samples.

    preamble : array_like or Preamble, optional
        Known preamble.
    threshold : float, default 0.5
        Detection threshold normalized between 0 and 1.

    sps : int, optional
        Samples per symbol. Must be provided explicitly.
    pulse_shape : str, optional
        Pulse shaping filter. Defaults to ``'rrc'`` if not provided.
    filter_params : dict, optional
        Additional filter parameters (beta, span, etc.). Defaults to ``{}``
        if None.
    search_range : tuple of int, optional
        A ``(start, end)`` sample range to restrict detection.
        Defaults to the full signal length.
    dft_upsample : int, default 1
        Factor for DFT-based upsampling of correlation peak.
        Use > 1 for high-precision fractional delay estimation.
    fractional_method : {'parabolic', 'log-parabolic'}, default 'log-parabolic'
        Fitting method for fractional delay estimation.
    debug_plot : bool, default False
        If True, plots the correlation magnitude for debugging.

    Returns
    -------
    coarse_offsets : ArrayType
        Integer sample offsets where the frame begins, per RX channel.
        Shape: ``(N_channels,)``.  Each channel's offset is estimated
        independently so hardware skew between channels is preserved.
    fractional_offsets : ArrayType
        Sub-sample timing offsets in [-0.5, 0.5) per channel.
        Shape: ``(N_channels,)``.

    Raises
    ------
    ValueError
        If no correlation peak is found that satisfies the threshold criteria.

    Notes
    -----
    - The returned coarse offset corresponds to the very first sample of
      the detected preamble.
    - **Synchronization Strategy**: For signals with oversampling (SPS > 1),
      correlating with a shaped/oversampled preamble (e.g., generated via
      ``Preamble.to_signal(...)``) typically yields superior timing precision
      and SNR compared to correlating with 1 SPS symbols. Ensure the
      ``preamble`` argument matches the sampling rate of the input ``samples``.
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

    # 2. Reconstruct/Get Preamble Waveform
    preamble_waveform = None
    resolved_preamble = preamble

    if isinstance(resolved_preamble, Preamble):
        if sps is None:
            raise ValueError("SPS must be provided when using a Preamble object.")

        preamble_waveform = xp.asarray(
            resolved_preamble.to_signal(
                sps=sps,
                symbol_rate=1.0,
                pulse_shape=pulse_shape or "rrc",
                **filter_params,
            ).samples
        )
        # ensure 2-D (C_tx, L*sps) for the correlation engine
        if preamble_waveform.ndim == 1:
            preamble_waveform = preamble_waveform[None, :]

    elif resolved_preamble is not None:
        preamble_waveform = xp.asarray(resolved_preamble)
    else:
        raise ValueError(
            "Either 'signal.frame' with a preamble, or 'preamble' argument must be provided."
        )

    # 3. Correlation Strategy
    # Signal: (C, N) or (N,)
    # Preamble: (C, L) or (L,) or (1, L)

    # Ensure dimensions match for broadcast/multichannel correlation
    if sig_array.ndim == 1:
        sig_array = sig_array[None, :]  # Treat as 1 channel

    if preamble_waveform.ndim == 1:
        preamble_waveform = preamble_waveform[None, :]  # Treat as 1 template

    # Ensure preamble is on the same device as the signal.  The preamble can be
    # constructed on GPU (e.g. via Preamble.to_signal() with GPU default) while
    # the received signal was loaded from a .npy file and lives on CPU.
    preamble_waveform = to_device(preamble_waveform, "cpu" if xp is np else "gpu")

    # If signal has C channels, we expect preamble to be compatible.
    # Case 1: Signal (C, N), Preamble (C, L) -> Correlate row-by-row, sum magnitudes.
    # Case 2: Signal (C, N), Preamble (1, L) -> Broadcast preamble to all?
    #         If mode="same", Preamble is (C, L) already.

    # We will compute correlation for each channel pair (k, k)
    # This assumes checking for "preamble on this channel".

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
    L = preamble_waveform.shape[-1]
    C_tx = preamble_waveform.shape[0]

    if C_tx > 1:
        from scipy.optimize import linear_sum_assignment

        # MIMO unique-root: correlate every TX template against every RX channel.
        # corr_all[rx, tx, lag] — shape (C_rx, C_tx, N_lag)
        corr_all = xp.stack(
            [
                cross_correlate_fft(
                    sig_processing, preamble_waveform[t : t + 1], mode="positive_lags"
                )
                for t in range(C_tx)
            ],
            axis=1,
        )

        # === Best-assignment peak finding ===
        # Score matrix S[rx, tx] = max_lag |corr_all[rx, tx, :]|.
        # Solve the linear assignment problem to find the unique TX template per
        # RX channel that maximises the total peak score.  For a unitary optical
        # channel this correctly identifies which ZC root each RX channel
        # received most strongly, giving a higher peak-to-floor ratio than the
        # incoherent sum (which adds cross-correlation floor terms from all
        # non-matching templates).
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
        corr = cross_correlate_fft(
            sig_processing, preamble_waveform, mode="positive_lags"
        )

    # Magnitude
    corr_mag = xp.abs(corr)

    # === Per-Channel Analysis ===
    # For SISO / broadcast single template: find peak from magnitude
    if C_tx == 1:
        peak_indices = xp.argmax(corr_mag, axis=-1)  # Shape (C,)

    # === Per-Channel Normalization ===
    # Calculate Energy per channel
    e_p = xp.sum(xp.abs(preamble_waveform) ** 2, axis=-1)  # (C_tx,) or scalar
    if e_p.ndim == 0:  # SISO: broadcast scalar energy to all channels
        e_p = xp.full(num_sig_ch, e_p)
    else:
        # Multiple TX templates: broadcast mean energy to all RX channels
        e_p = xp.full(num_sig_ch, xp.mean(e_p))

    e_s = xp.mean(xp.abs(sig_processing) ** 2, axis=-1) * L  # (C,)

    norm_factors = xp.sqrt(e_p * e_s)
    norm_factors = xp.maximum(norm_factors, 1e-12)

    # Calculate per-channel metrics.
    # For MIMO use the assigned-template correlation magnitude (consistent with
    # peak_indices); for SISO corr_incoherent is not computed so fall back to
    # the coherent magnitude.
    if C_tx > 1:
        peak_vals = xp.max(
            corr_incoherent, axis=-1
        )  # (C,) incoherent — matches peak detection
    else:
        peak_vals = xp.max(corr_mag, axis=-1)  # (C,)
    metrics = peak_vals / norm_factors
    metrics = xp.clip(metrics, 0.0, 1.0)

    # === MIMO fallback: X-Y power imbalance with polarization mixing ===
    # Only enters when at least one channel already has a good peak — that anchor
    # confirms the frame is present. For each failing channel, try the templates
    # that weren't assigned to it; if a better correlation is found, borrow it.
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
                alt_metric = min(alt_peak / float(norm_factors[rx]), 1.0)
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
        # Check skew among valid channels
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

    # Frame Start Calculation (Per-Channel)
    # Each channel's peak is found independently so hardware skew is preserved.
    coarse_offsets = xp.maximum(0, peak_indices + offset)

    if debug_plot:
        from . import plotting as _plotting

        _plotting.timing_correlation(
            corr_mag=to_device(corr_mag, "cpu"),
            peak_indices=to_device(peak_indices, "cpu"),
            norm_factors=to_device(norm_factors, "cpu"),
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
    **Preamble is not removed.**  After correction, ``signal[..., 0]``
    corresponds to the first sample of the preamble (the frame start).
    All three modes align the frame start to index 0 — they do not strip
    the preamble from the output.  Use
    :meth:`~commstools.core.SingleCarrierFrame.get_structure_map`
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
    shared_lo_check: bool = True,
    shared_lo_tol_hz: Optional[float] = None,
    debug_plot: bool = False,
) -> float:
    """
    Estimates frequency offset using the M-th power law (nonlinear spectral method).

    Raises the signal to the M-th power to eliminate PSK/QAM modulation,
    producing a tone at M·Δf.  A spectral peak search with sub-bin
    interpolation gives frequency resolution well below the FFT bin width.

    Parameters
    ----------
    samples : array_like
        Complex IQ samples. Shape: (N,) or (C, N). For MIMO, channel
        spectra are summed before peak detection (coherent accumulation).
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
    shared_lo_check : bool, default True
        For MIMO inputs (C > 1): also find the peak independently per
        channel and warn if the inter-channel spread exceeds
        ``shared_lo_tol_hz``.  Assumes a shared LO (e.g. dual-polarisation
        coherent optical).  Set to ``False`` for independent-LO systems.
    shared_lo_tol_hz : float, optional
        Spread threshold for the shared-LO sanity check.
        Default: ``0.005 * sampling_rate`` (0.5 % of sampling rate).

    Returns
    -------
    float
        Estimated frequency offset in Hz. Positive means the received
        signal is shifted up relative to the local oscillator.

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

    if interpolation == "jacobsen":
        # Jacobsen estimator: uses complex FFT values — unbiased for rectangular window.
        # δ = Re[(X[k-1] − X[k+1]) / (2·X[k] − X[k-1] − X[k+1])]
        # Sum over channels for the combined (coherent) spectrum.
        X_combined = xp.sum(X_M, axis=0)  # (nfft,)
        xa = complex(X_combined[k_safe - 1])
        xb = complex(X_combined[k_safe])
        xc = complex(X_combined[k_safe + 1])
        denom_j = 2 * xb - xa - xc
        mu = float(((xa - xc) / denom_j).real) if abs(denom_j) > 1e-30 else 0.0
    else:
        # Parabolic fallback on magnitudes (legacy, biased for small nfft)
        a = float(mag_combined[k_safe - 1])
        b_ = float(mag_combined[k_safe])
        c_ = float(mag_combined[k_safe + 1])
        denom_p = a - 2 * b_ + c_
        mu = 0.5 * (a - c_) / denom_p if abs(denom_p) > 1e-15 else 0.0

    mu = max(-0.5, min(0.5, mu))
    f_est = (freqs_np[k_safe] + mu * (sampling_rate / nfft)) / M

    # Per-channel shared-LO sanity check for MIMO
    if C > 1 and shared_lo_check:
        tol = (
            shared_lo_tol_hz if shared_lo_tol_hz is not None else 0.005 * sampling_rate
        )
        f_per_ch = []
        for c_idx in range(C):
            k_c = int(xp.argmax(mag[c_idx]))
            k_c_safe = max(1, min(k_c, nfft - 2))
            if interpolation == "jacobsen":
                xa_c = complex(X_M[c_idx, k_c_safe - 1])
                xb_c = complex(X_M[c_idx, k_c_safe])
                xc_c = complex(X_M[c_idx, k_c_safe + 1])
                d_c = 2 * xb_c - xa_c - xc_c
                mu_c = (
                    float(((xa_c - xc_c) / d_c).real)
                    if abs(d_c) > 1e-30
                    else 0.0
                )
            else:
                a_c = float(mag[c_idx, k_c_safe - 1])
                b_c = float(mag[c_idx, k_c_safe])
                c_c = float(mag[c_idx, k_c_safe + 1])
                d_c = a_c - 2 * b_c + c_c
                mu_c = 0.5 * (a_c - c_c) / d_c if abs(d_c) > 1e-15 else 0.0
            mu_c = max(-0.5, min(0.5, mu_c))
            f_per_ch.append((freqs_np[k_c_safe] + mu_c * (sampling_rate / nfft)) / M)
        spread = max(f_per_ch) - min(f_per_ch)
        logger.info(
            f"FOE (M-th power, M={M}): {f_est:.2f} Hz "
            f"[per-ch: {[f'{f:.2f}' for f in f_per_ch]} Hz, spread: {spread:.2f} Hz, "
            f"nfft={nfft}, interp={interpolation}]"
        )
        if spread > tol:
            logger.warning(
                f"FOE (M-th power): per-channel spread {spread:.2f} Hz "
                f"exceeds shared-LO tolerance {tol:.2f} Hz. "
                f"Per-channel estimates: {[f'{f:.2f}' for f in f_per_ch]} Hz. "
                "Possible timing misalignment or independent LOs per channel."
            )
    else:
        logger.info(
            f"FOE (M-th power, M={M}): {f_est:.2f} Hz "
            f"[nfft={nfft}, interp={interpolation}, search_range={search_range}]"
        )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.frequency_offset_spectrum(
            mag_spectrum=to_device(mag_combined[None, :], "cpu"),
            freqs=freqs_np,
            M=M,
            k_peaks=np.array([k_peak]),
            f_estimates=[f_est],
            search_range=search_range,
            show=True,
        )

    return float(f_est)


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
    shared_lo_check: bool = True,
    shared_lo_tol_hz: Optional[float] = None,
    debug_plot: bool = False,
) -> float:
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
    shared_lo_check : bool, default True
        For MIMO (C > 1): warn if per-channel estimates diverge beyond
        ``shared_lo_tol_hz`` (shared-LO assumption check).
    shared_lo_tol_hz : float, optional
        Spread threshold for shared-LO sanity check.
        Default: ``0.005 * sampling_rate``.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the autocorrelation
        magnitude ``|R[m]|`` and wrapped phase ``∠R[m]`` vs lag, with the
        expected phase ramp overlaid.

    Returns
    -------
    float
        Estimated frequency offset in Hz.

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

    For MIMO, autocorrelations are summed coherently across channels before
    phase extraction, assuming a shared LO.

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

    # Coherent sum across channels (shared-LO assumption), then divide by C
    R_combined = xp.sum(R_per_ch, axis=0) / C  # (L,)

    # Transfer only R_combined (L complex128 = ~16 KB) to CPU.  The iterative
    # bootstrap is a sequential scan — each step reads f_hat produced by the
    # previous one, preventing GPU parallelisation.  We use a Numba-compiled
    # kernel (same lazy-cache pattern as DD-PLL / RTS smoother) for ~250×
    # speedup over the plain Python loop; falls back to pure Python if Numba
    # is not installed.
    R_np = to_device(R_combined, "cpu")  # (L,) complex128
    theta_np = np.angle(R_np)  # (L,) float64
    amp_np = np.abs(R_np)  # (L,) float64

    # Iterative bootstrap — see _mm_bootstrap_loop docstring for the algorithm.
    # Weights: w[m] ∝ m² · |R[m]|²  (M&M MVUE × SNR proxy)
    _mm_kernel = _get_numba_mm_bootstrap()
    f_est = float(_mm_kernel(theta_np, amp_np, float(M), float(sampling_rate)))

    mode_str = "data-aided" if ref_signal is not None else f"blind M={M}"

    # Per-channel shared-LO sanity check for MIMO.
    # R_per_ch[c, :] was already computed above — transfer (C, L) once, then
    # run the same Numba kernel per channel (no extra GPU work).
    spread = None
    f_per_ch = None
    if C > 1 and shared_lo_check:
        tol = (
            shared_lo_tol_hz if shared_lo_tol_hz is not None else 0.005 * sampling_rate
        )
        f_per_ch = []
        R_per_ch_np = to_device(R_per_ch, "cpu")  # (C, L) — single transfer
        for c_idx in range(C):
            R_c_np = R_per_ch_np[c_idx]  # (L,) — already on CPU
            theta_c = np.angle(R_c_np)
            amp_c = np.abs(R_c_np)
            f_per_ch.append(
                float(_mm_kernel(theta_c, amp_c, float(M), float(sampling_rate)))
            )
        spread = max(f_per_ch) - min(f_per_ch)

    if f_per_ch is not None:
        logger.info(
            f"FOE (Mengali-Morelli, {mode_str}): {f_est:.2f} Hz "
            f"[L={L} lags, N={N}, per-ch: {[f'{f:.2f}' for f in f_per_ch]} Hz, "
            f"spread: {spread:.2f} Hz]"
        )
    else:
        logger.info(
            f"FOE (Mengali-Morelli, {mode_str}): {f_est:.2f} Hz [L={L} lags, N={N}]"
        )

    if spread is not None and spread > tol:
        logger.warning(
            f"FOE (Mengali-Morelli): per-channel spread {spread:.2f} Hz "
            f"exceeds shared-LO tolerance {tol:.2f} Hz. "
            f"Per-channel estimates: {[f'{f:.2f}' for f in f_per_ch]} Hz. "
            "Possible timing misalignment or independent LOs per channel."
        )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.mm_autocorrelation(
            R_np=R_np,
            f_est=f_est,
            sampling_rate=sampling_rate,
            M=M,
            show=True,
        )

    return f_est


def estimate_frequency_offset_pilots(
    samples: ArrayType,
    sampling_rate: float,
    pilot_indices: ArrayType,
    pilot_values: ArrayType,
    snr_weighted: bool = True,
    shared_lo_check: bool = True,
    shared_lo_tol_hz: Optional[float] = None,
    debug_plot: bool = False,
) -> float:
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
    shared_lo_check : bool, default True
        For MIMO inputs (C > 1): compare per-channel slope estimates and
        warn if the inter-channel spread exceeds ``shared_lo_tol_hz``.
        Assumes a shared LO (e.g. dual-polarisation coherent optical).
        Set to ``False`` for independent-LO systems.
    shared_lo_tol_hz : float, optional
        Spread threshold for the shared-LO sanity check.
        Default: ``0.005 * sampling_rate`` (0.5 % of sampling rate).

    Returns
    -------
    float
        Estimated frequency offset in Hz.  For MIMO (shared LO), the
        per-channel slope estimates are averaged to a single scalar.

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

    # Combined estimate: mean of per-channel slopes (valid for shared LO)
    f_est = float(xp.mean(slopes)) / (2.0 * np.pi)

    max_gap = int(np.max(np.diff(pilot_indices_np))) if P > 1 else 0
    lock_range = sampling_rate / (2 * max_gap) if max_gap > 0 else float("inf")
    wt_str = "WLSQ" if snr_weighted else "OLS"

    if C > 1:
        f_per_ch = [float(slopes[c]) / (2.0 * np.pi) for c in range(C)]
        spread = max(f_per_ch) - min(f_per_ch)
        logger.info(
            f"FOE (pilots, {wt_str}): {f_est:.2f} Hz "
            f"[per-ch: {[f'{f:.2f}' for f in f_per_ch]} Hz, spread: {spread:.2f} Hz, "
            f"P={P}, max_gap={max_gap}, lock_range=±{lock_range:.1f} Hz]"
        )
        if shared_lo_check:
            tol = (
                shared_lo_tol_hz
                if shared_lo_tol_hz is not None
                else 0.005 * sampling_rate
            )
            if spread > tol:
                logger.warning(
                    f"FOE (pilots): per-channel spread {spread:.2f} Hz "
                    f"exceeds shared-LO tolerance {tol:.2f} Hz. "
                    f"Per-channel estimates: {[f'{f:.2f}' for f in f_per_ch]} Hz. "
                    "Possible timing misalignment or independent LOs per channel."
                )
    else:
        logger.info(
            f"FOE (pilots, {wt_str}): {f_est:.2f} Hz "
            f"[P={P}, max_gap={max_gap} samples, lock_range=±{lock_range:.1f} Hz]"
        )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.pilot_phase_estimate(
            pilot_indices=pilot_indices_np,
            phi_pilots_u=to_device(phi_pilots_u, "cpu"),
            f_est=f_est,
            sampling_rate=sampling_rate,
            show=True,
        )

    return f_est


def correct_frequency_offset(
    samples: ArrayType,
    sampling_rate: float,
    offset: float,
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
    offset : float
        Estimated frequency offset in Hz (as returned by the
        ``estimate_frequency_offset_*`` functions).

    Returns
    -------
    array_like
        Frequency-corrected samples, same shape and dtype as input.
    """
    samples, xp, _ = dispatch(samples)
    logger.debug(
        f"Applying frequency offset correction: {offset:.4f} Hz (sampling_rate={sampling_rate:.0f} Hz)"
    )
    n = samples.shape[-1]
    t = xp.arange(n) / sampling_rate

    # Negate offset: correction reverses the carrier frequency error
    phase = -2.0 * np.pi * float(offset) * t
    mixer = xp.exp(1j * phase)  # complex128

    if xp.iscomplexobj(samples):
        target_dtype = samples.dtype
    else:
        target_dtype = xp.complex64 if samples.dtype == xp.float32 else xp.complex128
    mixer = mixer.astype(target_dtype)

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


def recover_carrier_phase_decision_directed(
    symbols: ArrayType,
    modulation: str,
    order: int,
    mu: float = 1e-2,
    beta: float = 0.0,
    phase_init: float = 0.0,
    loop_filter: str = "pi",
    loop_bandwidth_normalized: float = 1e-3,
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

    if loop_filter == "butterworth":
        import scipy.signal as _ss  # noqa: PLC0415

        # Design 2nd-order Butterworth lowpass at loop_bandwidth_normalized
        # (normalised by Nyquist = 0.5 symbol rate, so Wn = 2 * lbw).
        b_arr, a_arr = _ss.butter(
            2, 2.0 * loop_bandwidth_normalized, btype="low", analog=False
        )
        b0, b1, b2 = float(b_arr[0]), float(b_arr[1]), float(b_arr[2])
        a1, a2 = float(a_arr[1]), float(a_arr[2])

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
        Typical range: 16-128.

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
        phases differ by less than `pi/M`.  For high phase noise this
        assumption can be violated, causing a persistent `2*pi/M` phase
        step in the output.

        A rough safety condition is:

        .. math::

            \\Delta\\nu \\cdot T_{\\text{block}} < 0.05 \\cdot f_s

        where :math:`\\Delta\\nu` is the combined linewidth (Hz),
        :math:`T_{\\text{block}} = \\text{block\\_size} / f_s` is the block
        duration, and :math:`f_s` is the symbol rate.  For example, 100 kHz
        linewidth at 32 Gbaud is safe up to ``block_size ≈ 16 000``, but
        1 MHz linewidth requires ``block_size ≤ 1 600``.

        When operating near or above this limit, prefer
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

    # Raw block phase in [-π/M, π/M)
    phi_raw = xp.angle(S_b) / M  # (C, N_blocks)

    # M-fold unwrap: scale into 2π domain, unwrap, re-scale back.
    # Cast to float64 before unwrap — cp.unwrap preserves input dtype so float32
    # would lose precision during the discontinuity test (diff vs 2π threshold).
    phi_u = xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M  # (C, N_blocks)

    # QAM bias correction.
    # With unit-circle normalisation, E[(d/|d|)^M] is real negative by the
    # 4-fold rotational symmetry of any square QAM constellation: all points
    # map to angles of the form π/4 + k·π/2, whose 4th powers are all −1.
    # This gives a deterministic π/M offset; subtracting it is now exact.
    # PSK has no such bias (all (d/|d|)^M = 1 are real positive).
    if "qam" in modulation.lower():
        phi_u = phi_u - (np.pi / M)

    # MIMO M-fold alignment.
    # VV resolves the M-fold ambiguity independently per channel, so different
    # streams can land on different 2π/M branches.  For a coherent system with a
    # shared local oscillator the carrier phase is common — align every channel
    # to channel 0's branch by rounding the mean inter-channel difference to the
    # nearest integer multiple of 2π/M.
    if C > 1:
        for ch in range(1, C):
            diff = float(xp.mean(phi_u[ch] - phi_u[0]))
            k = round(diff * M / (2 * np.pi))
            phi_u[ch] = phi_u[ch] - k * (2 * np.pi / M)

    # Block centre positions for interpolation (uniform spacing = block_size)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)

    # xp.interp is 1D-only; loop over C channels.  C is typically 1–4 so the
    # overhead is negligible.  A searchsorted-based batched form gives ULP-level
    # differences at block boundaries that can flip the M-fold branch for
    # high-order QAM with small block sizes, so xp.interp is preferred here.
    phi_full = xp.zeros((C, N), dtype=xp.float64)
    for ch in range(C):
        phi_full[ch] = xp.interp(all_positions, block_centers, phi_u[ch])

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    logger.info(
        f"CPR (Viterbi-Viterbi, M={M}): phase mean={phi_mean_deg:.2f}°, "
        f"std={phi_std_deg:.2f}° [{N_blocks} blocks × {block_size} symbols, C={C}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full_np,
            block_centers=to_device(block_centers, "cpu"),
            phi_blocks=to_device(phi_u, "cpu"),
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
    6. 4-fold unwrap and per-symbol interpolation.

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

    for ch in range(C):
        sym = symbols[ch, :N_trunc]  # (N_trunc,)

        # Accumulate block-average error metric: (N_blocks, B).
        # CHUNK_N is a multiple of block_size by construction, so each chunk
        # covers a whole number of blocks with no remainder.
        metric = xp.zeros((N_blocks, B), dtype=float_dtype)

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
            metric[b0 : b0 + n_b] = chunk_min_d.reshape(n_b, block_size, B).sum(axis=1)

        # Best candidate index per block and 4-fold unwrap
        best_k = xp.argmin(metric, axis=-1)  # (N_blocks,)
        phi_b = candidates[best_k]  # (N_blocks,)
        phi_u = xp.unwrap(phi_b.astype(xp.float64) * 4, axis=-1) / 4

        # Interpolate to per-symbol resolution using pre-computed weights
        phi_full[ch] = phi_u[idx_left] * (1.0 - t_interp) + phi_u[idx_right] * t_interp
        phi_blocks[ch] = phi_u

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    logger.info(
        f"CPR (BPS, B={B}): phase mean={phi_mean_deg:.2f}°, std={phi_std_deg:.2f}° "
        f"[{N_blocks} blocks × {block_size} symbols, C={C}]"
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
    debug_plot: bool = False,
) -> ArrayType:
    r"""
    Carrier phase recovery via MAP estimation with a Tikhonov/Wiener phase
    noise prior (Colavolpe et al., 2005).

    Extends the Viterbi-Viterbi block estimator with a Kalman smoother
    matched to the laser phase noise statistics.  Two smoother backends are
    available via ``method``:

    * ``'exact'`` — full Rauch-Tung-Striebel (RTS) smoother; Numba-compiled
      when available, pure-Python fallback otherwise.  Exact for all
      sequence lengths; runs on CPU.
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
    phi_raw = xp.angle(S_b) / M
    phi_u = xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M  # (C, N_blocks)

    if "qam" in modulation.lower():
        phi_u = phi_u - (np.pi / M)

    if C > 1:
        for ch in range(1, C):
            diff = float(xp.mean(phi_u[ch] - phi_u[0]))
            k = round(diff * M / (2 * np.pi))
            phi_u[ch] = phi_u[ch] - k * (2 * np.pi / M)

    # Kalman smoother — dispatch on method
    if method == "exact":
        # Sequential RTS: offload to CPU; Numba kernel used when available.
        phi_u_np = to_device(phi_u, "cpu")  # (C, N_blocks) float64
        phi_smooth_np = np.empty_like(phi_u_np)
        for ch in range(C):
            phi_smooth_np[ch] = _rts_smoother_1d(phi_u_np[ch], sigma_p2, sigma_v2)
        phi_smooth = xp.asarray(phi_smooth_np)
    else:  # method == "sskf"
        # Steady-state approximation via filtfilt — stays on the input device.
        phi_smooth = xp.empty_like(phi_u)
        for ch in range(C):
            phi_smooth[ch] = _sskf_smoother_1d(phi_u[ch], sigma_p2, sigma_v2, sp, xp)
        phi_smooth_np = to_device(phi_smooth, "cpu")

    # Per-symbol interpolation (linear, consistent with VV)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)

    phi_full = xp.zeros((C, N), dtype=xp.float64)
    for ch in range(C):
        phi_full[ch] = xp.interp(all_positions, block_centers, phi_smooth[ch])

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    logger.info(
        f"CPR (Tikhonov-{method.upper()}, M={M}): phase mean={phi_mean_deg:.2f}°, "
        f"std={phi_std_deg:.2f}° [{N_blocks} blocks × {block_size}, "
        f"σ_p²={sigma_p2:.2e}, σ_v²={sigma_v2:.2e}, C={C}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full_np,
            block_centers=to_device(block_centers, "cpu"),
            phi_blocks=phi_smooth_np,
            show=True,
            title="CPR — Tikhonov-RTS",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def recover_carrier_phase_pilots(
    symbols: ArrayType,
    pilot_indices: ArrayType,
    pilot_values: ArrayType,
    interpolation: str = "linear",
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
        C is typically 1–4 so the overhead is negligible.  ``'cubic'`` uses
        :class:`scipy.interpolate.CubicSpline` (CPU) or
        :class:`cupyx.scipy.interpolate.CubicSpline` (GPU) with natural
        boundary conditions (zero second derivative at endpoints) and
        constant-hold extrapolation outside the pilot span.

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
    phi_pilots = xp.angle(r_pilots * xp.conj(pilot_values_xp))  # (C, P)

    # Unwrap along the pilot axis in float64 (cp.unwrap preserves input dtype;
    # casting before avoids precision loss in the discontinuity test for float32 input)
    phi_pilots_u = xp.unwrap(phi_pilots.astype(xp.float64), axis=-1)  # (C, P)

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
