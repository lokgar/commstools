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
    Blind FOE via M-th power spectral method with parabolic sub-bin interpolation.
estimate_frequency_offset_differential :
    Blind or data-aided FOE via differential auto-correlation (Kay's estimator).
estimate_frequency_offset_data_aided :
    Preamble-based FOE using known training sequence (Kay/Fitz ML estimator).
correct_frequency_offset :
    Applies frequency offset correction via complex mixing.
recover_carrier_phase_viterbi_viterbi :
    Block-based CPR via M-th power law (Viterbi-Viterbi) for PSK/QAM symbols.
recover_carrier_phase_bps :
    Blind Phase Search CPR for QAM constellations (Pfau et al.).
recover_carrier_phase_pilots :
    Pilot-aided CPR with phase unwrapping and interpolation across the symbol grid.
correct_carrier_phase :
    Applies per-symbol phase correction by complex rotation.
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


# ============================================================================
# FOE/CPR helpers
# ============================================================================



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
    """
    mod = modulation.lower()
    if "psk" in mod or "bpsk" in mod:
        return order
    return 4  # QAM, PAM — 4th power is the standard choice


# ============================================================================
# Frequency Offset Estimation (FOE)
# ============================================================================


def estimate_frequency_offset_mth_power(
    signal: ArrayType,
    fs: float,
    modulation: str,
    order: int,
    search_range: Optional[Tuple[float, float]] = None,
    nfft: Optional[int] = None,
) -> float:
    """
    Estimates frequency offset using the M-th power law (nonlinear spectral method).

    Raises the signal to the M-th power to eliminate PSK/QAM modulation,
    producing a tone at M·Δf. A spectral peak search with parabolic
    interpolation gives sub-bin frequency resolution.

    Parameters
    ----------
    signal : array_like
        Complex IQ samples. Shape: (N,) or (C, N). For MIMO, each channel
        is estimated independently and the results are averaged.
    fs : float
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
        FFT size. Default: next power of 2 ≥ len(signal).

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
      amplitude modulation is suppressed by subtracting the mean of
      ``signal^M`` before the FFT).

    **Lock range:** ``[-fs/(2M), fs/(2M)]``. For QPSK at 1 GHz → ±125 MHz.
    Use ``search_range`` to reduce false-peak probability.

    Sub-bin accuracy is achieved by parabolic interpolation on the three
    FFT magnitude bins surrounding the spectral peak.

    References
    ----------
    M. Luise and R. Reggiannini, "Carrier frequency recovery in all-digital
    modems for burst-mode transmissions," IEEE Trans. Commun., 1995.
    """
    signal, xp, _ = dispatch(signal)
    was_1d = signal.ndim == 1
    if was_1d:
        signal = signal[None, :]  # (1, N)
    C, N = signal.shape

    M = _modulation_power_m(modulation, order)
    mod_lower = modulation.lower()

    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(N)))

    # Promote for numerical accuracy during power computation: (C, N)
    s_c = signal.astype(xp.complex128 if signal.dtype == xp.complex64 else signal.dtype)

    # M-th power removes modulation → tone at M·Δf; shape stays (C, N)
    x_M = s_c**M

    # For QAM: subtract per-channel mean to suppress DC spike from amplitude residuals
    if M == 4 and "qam" in mod_lower:
        x_M = x_M - xp.mean(x_M, axis=-1, keepdims=True)

    # Batched FFT across all channels: single kernel call on GPU → (C, nfft)
    X_M = xp.fft.fft(x_M, n=nfft, axis=-1)
    # Keep two copies: device array for masking, CPU array for scalar peak indexing
    freqs = xp.fft.fftfreq(nfft, d=1.0 / fs)   # (nfft,) — on device
    freqs_np = np.fft.fftfreq(nfft, d=1.0 / fs) # (nfft,) — always on CPU

    mag = xp.abs(X_M)       # (C, nfft)
    mag[:, 0] = 0.0          # zero DC for all channels

    # Restrict search to [M·f_min, M·f_max] when search_range is given
    if search_range is not None:
        tone_lo = M * min(search_range)
        tone_hi = M * max(search_range)
        mask = (freqs >= tone_lo) & (freqs <= tone_hi)  # (nfft,)
        if not bool(xp.any(mask)):
            raise ValueError(
                f"search_range {search_range} Hz produces an empty search "
                f"window in the M={M} scaled spectrum."
            )
        mag = xp.where(mask[None, :], mag, xp.zeros_like(mag))

    # Peak bin per channel: (C,)
    k_peaks = xp.argmax(mag, axis=-1)

    # Parabolic interpolation: C scalar operations — loop overhead is negligible
    estimates = []
    for ch in range(C):
        k_peak = int(k_peaks[ch])
        k_safe = max(1, min(k_peak, nfft - 2))
        a = float(mag[ch, k_safe - 1])
        b_ = float(mag[ch, k_safe])
        c = float(mag[ch, k_safe + 1])
        denom = a - 2 * b_ + c
        mu = 0.5 * (a - c) / denom if abs(denom) > 1e-15 else 0.0
        mu = max(-0.5, min(0.5, mu))
        f_tone = freqs_np[k_peak] + mu * (fs / nfft)
        estimates.append(f_tone / M)

    return float(np.mean(estimates))


def estimate_frequency_offset_differential(
    signal: ArrayType,
    fs: float,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    ref_signal: Optional[ArrayType] = None,
    weighted: bool = True,
) -> float:
    """
    Estimates frequency offset via differential phase (auto-correlation).

    Three modes, selected automatically based on the supplied arguments:

    1. **Data-aided** (``ref_signal`` provided): derotates ``signal`` by
       ``conj(ref_signal)`` before applying the estimator.  Unaffected by
       modulation, highest-accuracy mode.
    2. **Blind M-PSK/QAM** (``modulation`` and ``order`` provided, no
       ``ref_signal``): applies M-th power pre-processing to remove
       modulation before estimating.
    3. **Generic blind** (neither provided): estimates frequency directly
       from raw differential phases.  Best for constant-envelope signals.

    Parameters
    ----------
    signal : array_like
        Received complex samples. Shape: (N,) or (C, N).
    fs : float
        Sampling rate in Hz.
    modulation : str, optional
        Modulation type (case-insensitive). Required for blind M-th power
        mode.  Ignored when ``ref_signal`` is provided.
    order : int, optional
        Modulation order.  Required with ``modulation`` for blind mode.
    ref_signal : array_like, optional
        Known reference signal at the same sample rate. Shape: (N,) or
        (C, N).  Used to derotate ``signal`` before estimation.
    weighted : bool, default True
        If ``True``, applies Kay's sinc-weighted estimator for improved
        low-SNR performance.  If ``False``, uses an unweighted sum.

    Returns
    -------
    float
        Estimated frequency offset in Hz.  For MIMO, the per-channel
        estimates are averaged.

    Notes
    -----
    The core estimator computes:

    .. math::
        \\hat{f} = \\frac{f_s}{2\\pi} \\angle\\!
            \\left[\\sum_{n=0}^{N-2} w[n]\\, y[n+1]\\, y^*[n]\\right]

    where ``y`` is the pre-processed signal and ``w[n]`` are Kay's
    sinc-squared weights (``w[n] = 1`` when ``weighted=False``).

    **Lock range (blind M-th power mode):** ``[-fs/(2M), fs/(2M)]``.
    Use :func:`estimate_frequency_offset_mth_power` as a coarse stage
    first if the offset may exceed this range.

    References
    ----------
    S. M. Kay, "A fast and accurate single-frequency estimator," IEEE
    Trans. Acoust. Speech Signal Process., 1989.
    """
    signal, xp, _ = dispatch(signal)
    was_1d = signal.ndim == 1
    if was_1d:
        signal = signal[None, :]  # (1, N)
    C, N = signal.shape

    # Determine pre-processing mode and M
    if ref_signal is not None:
        ref, _, _ = dispatch(ref_signal)
        if ref.ndim == 1:
            ref = ref[None, :]
        ref = xp.asarray(ref)
        y = signal * xp.conj(ref)  # derotate → complex tone at Δf
        M = 1
    elif modulation is not None and order is not None:
        M = _modulation_power_m(modulation, order)
        y = signal**M  # M-th power removes PSK/QAM modulation
    else:
        y = signal
        M = 1

    # Differential product: y[n+1] · conj(y[n])  → (C, N-1)
    y_diff = y[..., 1:] * xp.conj(y[..., :-1])

    if weighted:
        # Kay's sinc-squared weights: downweight samples at large lags
        L = N - 1
        n_np = np.arange(L, dtype=np.float64)
        n_c = (L - 1) / 2.0
        half = max(n_c, 1.0)
        w_np = 1.0 - ((n_np - n_c) / half) ** 2
        w_np = np.maximum(w_np, 0.0)
        w_np /= w_np.sum()
        w = xp.asarray(w_np)
        # Weighted complex phasor sum → angle
        weighted_sum = xp.sum(w * y_diff, axis=-1)  # (C,)
        f_per_ch = xp.angle(weighted_sum) * (fs / (2 * np.pi))
    else:
        f_per_ch = xp.angle(xp.sum(y_diff, axis=-1)) * (fs / (2 * np.pi))

    f_est = float(xp.mean(f_per_ch))
    return f_est / M


def estimate_frequency_offset_data_aided(
    signal: ArrayType,
    preamble_samples: ArrayType,
    fs: float,
    offset: int = 0,
) -> float:
    """
    Estimates frequency offset using a known preamble (data-aided).

    Extracts a preamble-length window from the received signal, demodulates
    it against the known preamble to isolate the complex tone at Δf, then
    applies Kay's ML estimator on the result.

    Parameters
    ----------
    signal : array_like
        Received complex samples. Shape: (N,) or (C, N).
    preamble_samples : array_like
        Known preamble at the same sample rate as ``signal``.
        Shape: (L,) or (C, L).
    fs : float
        Sampling rate in Hz.
    offset : int, default 0
        Integer sample index where the preamble begins in the received
        signal.  Typically the ``coarse_offset`` returned by
        :func:`estimate_timing`.

    Returns
    -------
    float
        Estimated frequency offset in Hz.

    Notes
    -----
    After demodulation, ``y[n] = r_p[n] · p*[n]`` is a noisy complex
    exponential at Δf.  Kay's estimator (``weighted=True``) is applied for
    maximum-likelihood performance at high SNR.

    **Lock range:** ``~[-fs/(2L), fs/(2L)]`` where ``L`` is the preamble
    length.  For large offsets, use
    :func:`estimate_frequency_offset_mth_power` as a coarse stage first,
    correct it, then call this function for fine tuning.

    References
    ----------
    M. Fitz, "Further results in the unified analysis of digital
    communication systems," IEEE Trans. Commun., 1994.
    """
    signal, xp, _ = dispatch(signal)
    was_1d = signal.ndim == 1
    if was_1d:
        signal = signal[None, :]  # (1, N)

    preamble_samples, _, _ = dispatch(preamble_samples)
    if preamble_samples.ndim == 1:
        preamble_samples = preamble_samples[None, :]
    preamble_samples = xp.asarray(preamble_samples)

    L = preamble_samples.shape[-1]
    r_p = signal[..., offset : offset + L]  # (C, L)

    # Demodulate against known preamble → complex tone at Δf
    y = r_p * xp.conj(preamble_samples)

    # Apply Kay's estimator (data-aided: no M-th power, M=1)
    return estimate_frequency_offset_differential(y, fs, weighted=True)


def correct_frequency_offset(
    samples: ArrayType,
    offset: float,
    fs: float,
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
    offset : float
        Estimated frequency offset in Hz (as returned by the
        ``estimate_frequency_offset_*`` functions).
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    array_like
        Frequency-corrected samples, same shape and dtype as input.
    """
    samples, xp, _ = dispatch(samples)
    n = samples.shape[-1]
    t = xp.arange(n) / fs

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


# ============================================================================
# Carrier Phase Recovery (CPR)
# ============================================================================


def recover_carrier_phase_viterbi_viterbi(
    symbols: ArrayType,
    modulation: str,
    order: int,
    block_size: int = 32,
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

    # Reshape for block processing: (C, N_blocks, block_size)
    blocks = symbols[:, :N_trunc].reshape(C, N_blocks, block_size)
    S_b = xp.sum(blocks**M, axis=-1)  # (C, N_blocks)

    # Raw block phase in [-π/M, π/M)
    phi_raw = xp.angle(S_b) / M  # (C, N_blocks)

    # M-fold unwrap: scale into 2π domain, unwrap, re-scale back.
    # Cast to float64 before unwrap — cp.unwrap preserves input dtype so float32
    # would lose precision during the discontinuity test (diff vs 2π threshold).
    phi_u = xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M  # (C, N_blocks)

    # Block centre positions for interpolation (uniform spacing = block_size)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)

    # Batched linear interpolation across all channels simultaneously.
    # xp.interp is 1D-only, so we use searchsorted + vectorised indexing.
    # block_centers is uniformly spaced, so the segment length is always block_size.
    idx_right = xp.clip(
        xp.searchsorted(block_centers, all_positions), 1, N_blocks - 1
    )  # (N,)
    idx_left = idx_right - 1
    t_interp = xp.clip(
        (all_positions - block_centers[idx_left]) / block_size, 0.0, 1.0
    )  # (N,) — clamp handles extrapolation at both edges

    phi_u_f64 = phi_u.astype(xp.float64)  # (C, N_blocks)
    phi_full = (
        phi_u_f64[:, idx_left] * (1.0 - t_interp)
        + phi_u_f64[:, idx_right] * t_interp
    )  # (C, N)

    if was_1d:
        return phi_full[0]
    return phi_full


def recover_carrier_phase_bps(
    symbols: ArrayType,
    modulation: str,
    order: int,
    num_test_phases: int = 64,
    block_size: int = 32,
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

    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

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

    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)

    # Pre-compute interpolation indices and weights (identical for every channel)
    idx_right = xp.clip(
        xp.searchsorted(block_centers, all_positions), 1, N_blocks - 1
    )
    idx_left = idx_right - 1
    t_interp = xp.clip(
        (all_positions - block_centers[idx_left]) / block_size, 0.0, 1.0
    )

    # Pre-compute phasors for all B candidates once (avoid redundant exp per channel)
    phasors = xp.exp(
        (-1j * candidates.astype(xp.float64)).astype(
            xp.complex64 if symbols.dtype == xp.complex64 else xp.complex128
        )
    )  # (B,)

    # For square QAM (order a perfect square): the nearest constellation point
    # can be found in O(1) per symbol via per-component rounding, eliminating
    # the (CHUNK, B, M_const) distance tensor entirely.
    side = int(order**0.5)
    is_sq_qam = ("qam" in modulation.lower()) and (side * side == order)
    if is_sq_qam:
        # Sorted unique real levels of the constellation (shape: (side,))
        levels = xp.sort(xp.unique(const_xp.real))
        d_grid = float(levels[1] - levels[0])   # uniform grid spacing
        lev_min = float(levels[0])

    float_dtype = xp.float32 if symbols.dtype == xp.complex64 else xp.float64

    # Chunk size for N axis: bounds peak memory of the distance tensor.
    # CHUNK × B × M_const × 4 bytes ≤ ~32 MB regardless of signal length.
    CHUNK_N = 1024

    phi_full = xp.zeros((C, N), dtype=xp.float64)

    for ch in range(C):
        sym = symbols[ch, :N_trunc]  # (N_trunc,)

        # Allocate min-distance accumulator once per channel: (N_trunc, B)
        min_dist = xp.empty((N_trunc, B), dtype=float_dtype)

        for n0 in range(0, N_trunc, CHUNK_N):
            n1 = min(n0 + CHUNK_N, N_trunc)
            x_rot = sym[n0:n1, None] * phasors[None, :]  # (CHUNK, B)

            if is_sq_qam:
                # O(1) nearest-point: round each component to the nearest grid level
                r_idx = xp.clip(
                    xp.round((x_rot.real - lev_min) / d_grid).astype(xp.int64),
                    0, side - 1,
                )
                i_idx = xp.clip(
                    xp.round((x_rot.imag - lev_min) / d_grid).astype(xp.int64),
                    0, side - 1,
                )
                r_near = levels[r_idx]   # (CHUNK, B)
                i_near = levels[i_idx]   # (CHUNK, B)
                min_dist[n0:n1] = (
                    (x_rot.real - r_near) ** 2 + (x_rot.imag - i_near) ** 2
                ).astype(float_dtype)
            else:
                # General: (CHUNK, B, M_const) — bounded by CHUNK_N
                d_sq = xp.abs(x_rot[:, :, None] - const_xp[None, None, :]) ** 2
                min_dist[n0:n1] = xp.min(d_sq, axis=-1).astype(float_dtype)

        # Block-average error metric: (N_blocks, B)
        metric = xp.sum(min_dist.reshape(N_blocks, block_size, B), axis=1)

        # Best candidate index per block and 4-fold unwrap
        best_k = xp.argmin(metric, axis=-1)               # (N_blocks,)
        phi_b = candidates[best_k]                         # (N_blocks,)
        phi_u = xp.unwrap(phi_b.astype(xp.float64) * 4, axis=-1) / 4

        # Interpolate to per-symbol resolution using pre-computed weights
        phi_full[ch] = phi_u[idx_left] * (1.0 - t_interp) + phi_u[idx_right] * t_interp

    if was_1d:
        return phi_full[0]
    return phi_full


def recover_carrier_phase_pilots(
    symbols: ArrayType,
    pilot_indices: ArrayType,
    pilot_values: ArrayType,
    interpolation: str = "linear",
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
        Interpolation method between pilot positions.  ``'linear'`` is
        fully vectorised across all channels.  ``'cubic'`` uses
        :class:`scipy.interpolate.CubicSpline` (CPU) or
        :class:`cupyx.scipy.interpolate.CubicSpline` (GPU).

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

    Boundary extrapolation uses the first/last pilot phase (hold-and-extend),
    avoiding erratic values at frame edges.

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

    pilot_indices_np = np.asarray(pilot_indices, dtype=np.intp)
    pilot_indices_xp = xp.asarray(pilot_indices, dtype=xp.float64)
    pilot_values_xp = xp.asarray(pilot_values)
    P = len(pilot_indices_np)

    # Broadcast shared pilots (P,) → (C, P) for all channels
    if pilot_values_xp.ndim == 1:
        pilot_values_xp = xp.broadcast_to(pilot_values_xp[None, :], (C, P))

    # Phase at each pilot position: angle(r_pilot · conj(s_pilot))
    r_pilots = symbols[:, pilot_indices_np]  # (C, P)
    phi_pilots = xp.angle(r_pilots * xp.conj(pilot_values_xp))  # (C, P)

    # Unwrap along the pilot axis and promote to float64
    phi_pilots_u = xp.unwrap(phi_pilots, axis=-1).astype(xp.float64)  # (C, P)

    all_positions = xp.arange(N, dtype=xp.float64)

    if interpolation == "linear":
        # Fully vectorised across all channels using searchsorted.
        # Pilot indices may be non-uniformly spaced, so segment lengths vary.
        idx_right = xp.clip(
            xp.searchsorted(pilot_indices_xp, all_positions), 1, P - 1
        )  # (N,)
        idx_left = idx_right - 1
        seg_len = pilot_indices_xp[idx_right] - pilot_indices_xp[idx_left]  # (N,)
        # Guard against degenerate zero-length segments (identical consecutive pilots)
        safe_len = xp.where(seg_len > 0, seg_len, xp.ones_like(seg_len))
        t_interp = xp.clip(
            (all_positions - pilot_indices_xp[idx_left]) / safe_len, 0.0, 1.0
        )  # (N,)
        phi_full = (
            phi_pilots_u[:, idx_left] * (1.0 - t_interp[None, :])
            + phi_pilots_u[:, idx_right] * t_interp[None, :]
        )  # (C, N)

    elif interpolation == "cubic":
        # CubicSpline is inherently per-channel (1D y input); loop is unavoidable.
        # Both scipy (CPU) and cupyx.scipy (GPU) share the same API.
        phi_full = xp.empty((C, N), dtype=xp.float64)
        if xp is not np:
            from cupyx.scipy.interpolate import CubicSpline
        else:
            from scipy.interpolate import CubicSpline

        for ch in range(C):
            phi_ch = phi_pilots_u[ch]  # already float64
            cs = CubicSpline(pilot_indices_xp, phi_ch, extrapolate=True)
            phi_full[ch] = cs(all_positions)

    else:
        raise ValueError(
            f"Unknown interpolation method: {interpolation!r}. "
            "Choose 'linear' or 'cubic'."
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
    phase_vector_xp = xp.asarray(phase_vector)
    phasor = xp.exp(-1j * phase_vector_xp)
    if phasor.dtype != symbols.dtype:
        phasor = phasor.astype(symbols.dtype)
    return symbols * phasor
