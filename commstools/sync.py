"""
Synchronization and frame detection utilities.

This module provides routines for time and frequency synchronization,
including the generation of optimal synchronization sequences (Barker,
Zadoff-Chu) and robust frame detection algorithms using cross-correlation.

Functions
---------
barker_sequence :
    Generates Barker codes with optimal auto-correlation.
zadoff_chu_sequence :
    Generates Constant Amplitude Zero Auto-Correlation (CAZAC) sequences.
correlate :
    Multi-backend cross-correlation for sequence detection.
detect_frame :
    Identifies frame start position via preamble correlation.
"""

from typing import Optional, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch, is_cupy_available, to_device
from .core import Preamble, Signal, SignalInfo
from .logger import logger

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

    seq = np.array(_BARKER_SEQUENCES[length], dtype=np.float32)

    if is_cupy_available():
        seq = to_device(seq, "gpu")

    logger.debug(f"Generated Barker-{length} sequence.")
    return seq


def zadoff_chu_sequence(length: int, root: int = 1) -> ArrayType:
    """
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


def correlate(
    signal: ArrayType,
    template: ArrayType,
    mode: str = "full",
    normalize: bool = False,
) -> ArrayType:
    """
    Cross-correlates a signal with a template sequence.

    Uses FFT-based correlation for optimal performance on both CPU and GPU.
    This is essential for detecting synchronization sequences (preambles)
    in received data.

    Parameters
    ----------
    signal : array_like
        Input signal samples. Shape: (N_samples,) or (N_channels, N_samples).
    template : array_like
        The reference sequence to correlate against. Shape: (N_template,).
    mode : {"full", "same", "valid"}, default "full"
        The output size mode of the correlation.
    normalize : bool, default False
        If True, the output is normalized by the template energy.

    Returns
    -------
    array_like
        Correlation metric. Shape depends on `mode` and `signal` dimensions.
    """
    signal, xp, sp = dispatch(signal)
    template = xp.asarray(template)

    # Handle MIMO: correlate each channel independently
    if signal.ndim == 2:
        # Apply along time axis (axis=-1)
        results = []
        for ch in range(signal.shape[0]):
            corr = _correlate_1d(signal[ch], template, mode, normalize, xp, sp)
            results.append(corr)
        return xp.stack(results, axis=0)

    return _correlate_1d(signal, template, mode, normalize, xp, sp)


def _correlate_1d(
    signal: ArrayType,
    template: ArrayType,
    mode: str,
    normalize: bool,
    xp,
    sp,
) -> ArrayType:
    """
    Internal helper for 1D cross-correlation.

    Parameters
    ----------
    signal : array_like
        The input sequence.
    template : array_like
        The reference sequence.
    mode : {"full", "same", "valid"}
        Correlation output mode.
    normalize : bool
        Whether to apply energy-based normalization.
    xp : module
        Active array backend (NumPy/CuPy).
    sp : module
        Active signal processing backend (SciPy/CuPyX).

    Returns
    -------
    array_like
        The correlation metric.
    """
    # Use scipy's correlate which handles mode correctly
    # For matched filtering, we want correlation (not convolution)
    # scipy.correlate computes: sum_k(signal[n+k] * conj(template[k]))

    corr = sp.signal.correlate(signal, template, mode=mode)

    if normalize:
        # Normalize by template energy
        template_energy = xp.sum(xp.abs(template) ** 2)
        if template_energy > 0:
            corr = corr / xp.sqrt(template_energy)

            # Also normalize by signal energy in each window (for true normalized correlation)
            # This is expensive, so we use a simpler approximation
            signal_energy = xp.sum(xp.abs(signal) ** 2)
            if signal_energy > 0:
                corr = corr / xp.sqrt(signal_energy / len(signal) * len(template))

    return corr


def detect_frame(
    signal: Union[ArrayType, "Signal"],
    preamble: Optional[Union[ArrayType, "Preamble"]] = None,
    threshold: float = 0.5,
    info: Optional["SignalInfo"] = None,
    sps: Optional[int] = None,
    pulse_shape: Optional[str] = None,
    filter_params: Optional[dict] = None,
    return_metric: bool = False,
    search_range: Optional[Tuple[int, int]] = None,
    debug_plot: bool = False,
) -> Union[ArrayType, Tuple[ArrayType, ArrayType]]:
    """
    Detects the start of a frame via preamble correlation.

    This function performs a sliding cross-correlation between the received
    signal and the expected preamble. It supports SISO and MIMO configurations,
    handling different preamble modes (e.g., 'same', 'time_orthogonal') automatically
    if `SignalInfo` is provided.

    Parameters
    ----------
    signal : array_like or Signal
        Received signal samples.
    preamble : array_like or Preamble, optional
        The known preamble symbols (reference sequence). If None, it will be
        inferred from `signal.signal_info` or `info` argument.
    threshold : float, default 0.5
        Detection threshold normalized between 0 and 1.
    info : SignalInfo, optional
        Metadata describing the frame structure. If provided, it overrides
        any info attached to `signal`.
    sps : int, optional
        Samples per symbol. Inferred from `signal` if not provided.
    pulse_shape : str, optional
        Pulse shaping filter. Inferred from `signal` if not provided.
    filter_params : dict, optional
        Additional filter parameters (beta, span, etc.). Inferred if None.
    return_metric : bool, default False
        If True, returns both the detected index and the peak metric.
    search_range : tuple of int, optional
        A `(start, end)` sample range to restrict detection.
        Defaults to the full signal length.
    debug_plot : bool, default False
        If True, plots the correlation magnitude for debugging.

    Returns
    -------
    frame_starts : ArrayType
        The sample indices where the frame begins for each channel. Shape: (N_channels,).
    peak_metrics : ArrayType, optional
        The normalized correlation coefficients for each channel. Shape: (N_channels,).
        Only returned if `return_metric=True`.

    Raises
    ------
    ValueError
        If no correlation peak is found that satisfies the threshold criteria.

    Notes
    -----
    - The returned index is compensated for the preamble length and corresponds
      to the very first sample of the detected preamble.
    - **Synchronization Strategy**: For signals with oversampling ($SPS > 1$),
      correlating with a shaped/oversampled preamble (e.g., generated via
      `Preamble.to_signal(...)`) typically yields superior timing precision
      and SNR compared to correlating with 1 SPS symbols. Ensure the `preamble`
      argument matches the sampling rate of the input `signal`.
    """
    from .helpers import expand_preamble_mimo

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

    # --- Vectorized Correlation (FFT) ---
    # We want independent correlation for each channel.
    # Signal: (C, N). Preamble: (1, L) or (C, L).
    # Broadcasting handles (1, L) -> (C, L).

    N = sig_processing.shape[-1]
    L = preamble_waveform.shape[-1]
    n_fft = 1 << (N + L - 1).bit_length()

    # Calculate FFTs
    SIG = xp.fft.fft(sig_processing, n_fft, axis=-1)
    PRE = xp.fft.fft(preamble_waveform, n_fft, axis=-1)

    # Multiply in frequency domain (Correlation = Convolution with time-reversed conjugate)
    # FFT correlation gives circular convolution.
    # Result[k] corresponds to lag k.
    CORR_F = SIG * xp.conj(PRE)
    corr_raw = xp.fft.ifft(CORR_F, axis=-1)

    # Truncate to relevant lags (0 to N-L)
    # Positive lags [0, N] correspond to valid start positions where P fits in S.
    corr = corr_raw[..., :N]

    # Magnitude
    corr_mag = xp.abs(corr)

    # --- Per-Channel Analysis ---
    # Find peak index per channel
    peak_indices = xp.argmax(corr_mag, axis=-1)  # Shape (C,)

    # --- Per-Channel Normalization ---
    # Calculate Energy per channel
    e_p = xp.sum(xp.abs(preamble_waveform) ** 2, axis=-1)  # (C,) or broadcasted
    if e_p.ndim < 1 or e_p.shape[0] != num_sig_ch:  # Handle scalar broadcast
        e_p = xp.broadcast_to(e_p, (num_sig_ch,))

    e_s = xp.mean(xp.abs(sig_processing) ** 2, axis=-1) * L  # (C,)

    norm_factors = xp.sqrt(e_p * e_s)
    norm_factors = xp.maximum(norm_factors, 1e-12)

    # Calculate per-channel metrics
    peak_vals = xp.max(corr_mag, axis=-1)  # (C,)
    metrics = peak_vals / norm_factors
    metrics = xp.clip(metrics, 0.0, 1.0)

    # --- Threshold Check ---
    max_metric = float(xp.max(metrics))
    if max_metric < threshold:
        raise ValueError(
            f"No correlation peak above threshold {threshold} (max: {max_metric:.3f})"
        )

    # --- Skew Check (Robust) ---
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
                logger.debug(f"Channels aligned (spread {spread}).")

    # Frame Start Calculation (Per-Channel)
    frame_starts = peak_indices + offset
    frame_starts = xp.maximum(0, frame_starts)

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

    logger.debug(
        f"Frame detected. Starts: {frame_starts.tolist()}, Metrics: {metrics.tolist()}"
    )

    if return_metric:
        return frame_starts, metrics
    return frame_starts
