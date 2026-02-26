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
"""

from typing import Optional, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch, is_cupy_available, to_device
from .core import Preamble, Signal, SignalInfo
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
    # Path B: Standard (Fallback or Primary)
    # -------------------------------------------------------------------------
    if dft_upsample == 1 or xp.any(mu == 0):
        # We compute this for channels NOT handled by DFT path (or all if M=1)
        # For simplicity, finding mask of channels needing calculation
        if dft_upsample > 1:
            # interior_mask was defined above
            calc_mask = ~interior_mask
        else:
            calc_mask = xp.ones(C, dtype=bool)

        if xp.any(calc_mask):
            # We calculate for ALL safe indices, then blend.

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
    signal: ArrayType,
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
    signal : array_like
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
    signal, xp, _ = dispatch(signal)
    was_1d = signal.ndim == 1
    if was_1d:
        signal = signal[None, :]  # (1, N)

    C, N = signal.shape

    # Convert delay to array
    if isinstance(delay, (int, float)):
        delay_arr = xp.full(C, delay, dtype=signal.real.dtype)
    else:
        delay_arr = xp.asarray(delay, dtype=signal.real.dtype)
        if delay_arr.ndim == 0:
            delay_arr = delay_arr[None]

    # FFT
    spec = xp.fft.fft(signal, axis=-1)

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
    if not xp.iscomplexobj(signal):
        # Real input: fractional delay is a real-valued operation
        result = result.real
    elif result.dtype != signal.dtype:
        # Complex input: ifft may return complex128 from complex64 input
        result = result.astype(signal.dtype)

    if was_1d:
        return result[0]
    return result


def estimate_timing(
    signal: Union[ArrayType, "Signal"],
    preamble: Optional[Union[ArrayType, "Preamble"]] = None,
    threshold: float = 0.5,
    info: Optional["SignalInfo"] = None,
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
    offset. Additionally estimates the fractional (sub-sample) timing
    offset using parabolic interpolation on the correlation peak.
    Supports SISO and MIMO configurations, handling different preamble
    modes (e.g., 'same', 'time_orthogonal') automatically if
    ``SignalInfo`` is provided.

    Parameters
    ----------
    signal : array_like or Signal
        Received signal samples.
    preamble : array_like or Preamble, optional
        The known preamble symbols (reference sequence). If None, it will be
        inferred from ``signal.signal_info`` or ``info`` argument.
    threshold : float, default 0.5
        Detection threshold normalized between 0 and 1.
    info : SignalInfo, optional
        Metadata describing the frame structure. If provided, it overrides
        any info attached to ``signal``.
    sps : int, optional
        Samples per symbol. Inferred from ``signal`` if not provided.
    pulse_shape : str, optional
        Pulse shaping filter. Inferred from ``signal`` if not provided.
    filter_params : dict, optional
        Additional filter parameters (beta, span, etc.). Inferred if None.
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
        Integer sample offsets where the frame begins, per channel.
        Shape: ``(N_channels,)``.
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
      ``preamble`` argument matches the sampling rate of the input ``signal``.
    """
    from .helpers import cross_correlate_fft, expand_preamble_mimo

    # 1. Resolve Inputs & Metadata
    sig_array = None

    if isinstance(signal, Signal):
        sig_array = signal.samples
        if info is None:
            info = signal.signal_info
        if sps is None:
            sps = int(round(signal.sps))
        if pulse_shape is None:
            pulse_shape = signal.pulse_shape or "rrc"
        if filter_params is None:
            filter_params = {
                "filter_span": signal.filter_span,
                "rrc_rolloff": signal.rrc_rolloff,
                "rc_rolloff": signal.rc_rolloff,
                "gaussian_bt": signal.gaussian_bt,
                "smoothrect_bt": signal.smoothrect_bt,
            }
    else:
        sig_array = signal

    if filter_params is None:
        filter_params = {}

    sig_array, xp, _ = dispatch(sig_array)

    # 2. Reconstruct/Get Preamble Waveform
    # We prioritize reconstructing from 'info' to get correct MIMO structure
    preamble_waveform = None

    # Check if we can reconstruct from Info
    if info is not None and info.preamble_type is not None:
        # Reconstruct Preamble Object
        p_obj = Preamble(
            sequence_type=info.preamble_type,
            length=info.preamble_seq_len,
            **info.preamble_kwargs or {},
        )

        # Generate Base Waveform
        # Must strictly use SPS/Filter from signal to match
        if sps is None:
            raise ValueError("SPS must be provided or inferred from Signal.")

        p_sig = p_obj.to_signal(
            sps=sps,
            symbol_rate=1.0,  # dummy, affects only sampling_rate metadata
            pulse_shape=pulse_shape or "rrc",
            **filter_params,
        )
        base_waveform = p_sig.samples

        # Apply MIMO Structure (Same logic as SingleCarrierFrame.to_signal)
        # This ensures we correlate against exactly what was transmitted
        num_streams = info.num_streams
        mode = info.preamble_mode or "same"

        preamble_waveform = expand_preamble_mimo(base_waveform, num_streams, mode)

    # Fallback to manual preamble argument
    elif preamble is not None:
        if isinstance(preamble, Preamble):
            if sps is None:
                raise ValueError("SPS required for Preamble object.")
            p_sig = preamble.to_signal(
                sps=sps, symbol_rate=1.0, pulse_shape=pulse_shape, **filter_params
            )
            preamble_waveform = p_sig.samples
        else:
            preamble_waveform = xp.asarray(preamble)
    else:
        raise ValueError("Either 'info' or 'preamble' must be provided.")

    # 3. Correlation Strategy
    # Signal: (C, N) or (N,)
    # Preamble: (C, L) or (L,) or (1, L)

    # Ensure dimensions match for broadcast/multichannel correlation
    if sig_array.ndim == 1:
        sig_array = sig_array[None, :]  # Treat as 1 channel

    if preamble_waveform.ndim == 1:
        preamble_waveform = preamble_waveform[None, :]  # Treat as 1 template

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

    corr = cross_correlate_fft(sig_processing, preamble_waveform, mode="full")

    # Extract positive lags only (valid start positions).
    # In full mode (scipy convention), lag 0 is at index L-1.
    corr = corr[..., L - 1 :]  # length = sig_processing.shape[-1]

    # Magnitude
    corr_mag = xp.abs(corr)

    # === Per-Channel Analysis ===
    # Find peak index per channel
    peak_indices = xp.argmax(corr_mag, axis=-1)  # Shape (C,)

    # === Per-Channel Normalization ===
    # Calculate Energy per channel
    e_p = xp.sum(xp.abs(preamble_waveform) ** 2, axis=-1)  # (C,) or scalar
    if e_p.ndim == 0:  # SISO: broadcast scalar energy to all channels
        e_p = xp.full(num_sig_ch, e_p)

    e_s = xp.mean(xp.abs(sig_processing) ** 2, axis=-1) * L  # (C,)

    norm_factors = xp.sqrt(e_p * e_s)
    norm_factors = xp.maximum(norm_factors, 1e-12)

    # Calculate per-channel metrics
    peak_vals = xp.max(corr_mag, axis=-1)  # (C,)
    metrics = peak_vals / norm_factors
    metrics = xp.clip(metrics, 0.0, 1.0)

    # === Threshold Check ===
    max_metric = float(xp.max(metrics))
    if max_metric < threshold:
        raise ValueError(
            f"No correlation peak above threshold {threshold} (max: {max_metric:.3f})"
        )

    # === Skew Check (Robust) ===
    if num_sig_ch > 1:
        # Check skew among valid channels
        valid_mask = metrics > (threshold * 0.8)

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
    coarse_offsets = peak_indices + offset
    coarse_offsets = xp.maximum(0, coarse_offsets)

    if debug_plot:
        import matplotlib.pyplot as plt

        # Layout: N rows for channels (No Combined)
        n_rows = num_sig_ch
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5 * n_rows), squeeze=False)

        # Plot Per-Channel
        for i in range(num_sig_ch):
            ax1 = axes[i][0]
            ax2 = axes[i][1]

            c_ch = to_device(corr_mag[i], "cpu")
            pk_idx = int(peak_indices[i])
            pk_val = float(c_ch[pk_idx])
            metric_val = float(metrics[i])

            norm_val_i = float(norm_factors[i])
            abs_thresh = threshold * norm_val_i

            # Ax1: Overall
            ax1.plot(c_ch, label=f"Ch {i} (Metric: {metric_val:.2f})")
            ax1.axhline(pk_val, color="r", linestyle="--", alpha=0.3, label="Max")
            if abs_thresh > 0:
                ax1.axhline(
                    y=abs_thresh,
                    color="g",
                    linestyle=":",
                    label=f"Thresh ({abs_thresh:.1f})",
                )
            ax1.set_title(f"Channel {i} - Overall")
            ax1.legend(loc="upper right", fontsize="small")
            ax1.grid(True, alpha=0.3)

            # Ax2: Zoom
            zoom_w = 40
            s_z = max(0, pk_idx - zoom_w)
            e_z = min(len(c_ch), pk_idx + zoom_w)

            ax2.plot(np.arange(s_z, e_z), c_ch[s_z:e_z], label="Peak Area")
            ax2.axvline(pk_idx, color="r", linestyle="--", label=f"Pk {pk_idx}")
            if abs_thresh > 0:
                ax2.axhline(y=abs_thresh, color="g", linestyle=":", label="Thresh")
            ax2.set_title(f"Channel {i} - Detail")
            ax2.legend(loc="upper right", fontsize="small")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

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
    signal: ArrayType,
    coarse_offset: Union[int, ArrayType],
    fractional_offset: Union[float, ArrayType] = 0.0,
) -> ArrayType:
    """
    Combined coarse and fine timing correction.

    Applies an integer sample shift followed by fractional sample
    interpolation using FFT-based frequency-domain delay.

    Parameters
    ----------
    signal : array_like
        Input signal. Shape: (N,) or (C, N).
    coarse_offset : int or array_like
        Integer sample offset(s) to correct. Positive values shift
        the signal left (i.e., remove leading samples).
        Scalar or shape (C,) for per-channel offsets.
    fractional_offset : float or array_like, default 0.0
        Fractional sample delay(s) in [-0.5, 0.5) to correct via
        FFT-based interpolation. Scalar or shape (C,).

    Returns
    -------
    array_like
        Timing-corrected signal with the same shape as input.

    Notes
    -----
    The fractional correction uses FFT-based frequency-domain delay, which
    is mathematically ideal for bandlimited signals and perfectly preserves
    signal power (unlike polynomial interpolators).
    """
    signal, xp, _ = dispatch(signal)
    was_1d = signal.ndim == 1
    if was_1d:
        signal = signal[None, :]

    num_ch = signal.shape[0]
    N = signal.shape[-1]

    # === Coarse correction (integer shift) ===
    coarse_offset = xp.asarray(coarse_offset)
    if coarse_offset.ndim == 0:
        # Scalar: same shift for all channels
        signal = xp.roll(signal, -int(coarse_offset), axis=-1)
    else:
        # Per-channel shift via vectorized index gathering
        shifts = xp.asarray([-int(coarse_offset[ch]) for ch in range(num_ch)])
        col_idx = (xp.arange(N)[None, :] - shifts[:, None]) % N  # (C, N)
        row_idx = xp.arange(num_ch)[:, None]  # (C, 1)
        signal = signal[row_idx, col_idx]

    # === Fine correction (fractional via FFT) ===
    # Correction removes the delay: negate the fractional offset
    if isinstance(fractional_offset, (int, float)):
        apply_frac = abs(fractional_offset) > 1e-9
    else:
        fractional_offset = xp.asarray(fractional_offset)
        apply_frac = bool(xp.any(xp.abs(fractional_offset) > 1e-9))

    if apply_frac:
        signal = fft_fractional_delay(signal, -fractional_offset)

    if was_1d:
        return signal[0]
    return signal
