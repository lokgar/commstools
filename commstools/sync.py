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
estimate_frequency_offset_pilots :
    Scattered-pilot FOE via least-squares phase slope fitting.
correct_frequency_offset :
    Applies frequency offset correction via complex mixing.
recover_carrier_phase_viterbi_viterbi :
    Block-based CPR via M-th power law (Viterbi-Viterbi) for PSK/QAM symbols.
recover_carrier_phase_bps :
    Blind Phase Search CPR for QAM constellations (Pfau et al.).
recover_carrier_phase_pilots :
    Pilot-aided CPR with phase unwrapping and interpolation across the symbol grid.
recover_carrier_phase_decision_directed :
    Streaming CPR via Decision-Directed PLL (1st/2nd-order loop); Numba-compiled
    inner loop for CPU performance; GPU-transparent via CPU offload.
correct_carrier_phase :
    Applies per-symbol phase correction by complex rotation.
"""

from typing import Optional, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch, is_cupy_available, to_device
from .core import Preamble, Signal
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
    against all RX channels; the incoherent sum over templates is used to
    find each channel's peak independently, so **hardware skew between RX
    channels is preserved in the returned per-channel offsets**.

    Parameters
    ----------
    signal : array_like or Signal
        Received signal samples.

    preamble : array_like or Preamble, optional
        Known preamble.
    threshold : float, default 0.5
        Detection threshold normalized between 0 and 1.

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
      ``preamble`` argument matches the sampling rate of the input ``signal``.
    """
    from .helpers import cross_correlate_fft, zc_mimo_root

    # 1. Resolve Inputs & Metadata
    sig_array = None

    if isinstance(signal, Signal):
        sig_array = signal.samples
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
    # We prioritize reconstructing from signal.frame to get correct MIMO structure.
    preamble_waveform = None

    frame = getattr(signal, "frame", None) if isinstance(signal, Signal) else None

    resolved_preamble = preamble
    num_streams = 1

    if resolved_preamble is None and frame is not None:
        if hasattr(frame, "preamble") and frame.preamble is not None:
            resolved_preamble = frame.preamble
        elif getattr(signal, "signal_type", None) == "Preamble":
            resolved_preamble = frame

    if frame is not None:
        num_streams = getattr(frame, "num_streams", 1)

    if isinstance(resolved_preamble, Preamble):
        if sps is None:
            raise ValueError("SPS must be provided or inferred from Signal.")

        primary = resolved_preamble

        if num_streams > 1 and primary.sequence_type == "zc":
            # MIMO ZC: per-stream unique-root waveforms matching TX.
            stream_waveforms = []
            for k in range(num_streams):
                root_k = zc_mimo_root(k, primary.root, primary.length)
                pk_sig = Preamble(
                    sequence_type="zc",
                    length=primary.length,
                    root=root_k,
                ).to_signal(
                    sps=sps,
                    symbol_rate=1.0,
                    pulse_shape=pulse_shape or "rrc",
                    **filter_params,
                )
                stream_waveforms.append(pk_sig.samples)
            preamble_waveform = xp.stack(
                [xp.asarray(w) for w in stream_waveforms], axis=0
            )  # (C_tx, L*sps)
        else:
            # SISO or non-ZC MIMO: broadcast single template.
            p_sig = primary.to_signal(
                sps=sps,
                symbol_rate=1.0,
                pulse_shape=pulse_shape or "rrc",
                **filter_params,
            )
            preamble_waveform = xp.tile(
                xp.asarray(p_sig.samples)[None, :], (num_streams, 1)
            )

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

        # Per-channel correlation uses only the assigned template.
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
    # For MIMO use the incoherent sum (consistent with peak_indices); for SISO
    # corr_incoherent is not computed so fall back to the coherent magnitude.
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
    signal: ArrayType,
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
    signal : array_like
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
    signal, xp, _ = dispatch(signal)
    was_1d = signal.ndim == 1
    if was_1d:
        signal = signal[None, :]

    num_ch = signal.shape[0]
    N = signal.shape[-1]

    # === Coarse correction (integer shift) ===
    coarse_offset = xp.asarray(coarse_offset)

    if coarse_offset.ndim == 0:
        # --- Scalar: same shift for all channels ---
        shift = int(coarse_offset)
        if mode == "circular":
            signal = xp.roll(signal, -shift, axis=-1)
        elif mode == "zero":
            result = xp.zeros_like(signal)
            if shift > 0:
                result[..., : N - shift] = signal[..., shift:]
            elif shift < 0:
                result[..., -shift:] = signal[..., : N + shift]
            signal = result
        elif mode == "slice":
            signal = signal[..., shift:]
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
            signal = signal[row_idx, col_idx]

        elif mode == "zero":
            col_raw = col_base + coarse_int[:, None]  # (C, N)
            gathered = signal[row_idx, xp.clip(col_raw, 0, N - 1)]
            signal = xp.where(col_raw < N, gathered, xp.zeros_like(gathered))

        elif mode == "slice":
            # Align all channels to common overlap: N - max(offset) samples
            max_shift = int(xp.max(coarse_int))  # one GPU sync
            common_len = N - max_shift
            col_idx_s = (
                xp.arange(common_len, dtype=xp.int64)[None, :] + coarse_int[:, None]
            )  # (C, common_len)
            signal = signal[row_idx, col_idx_s]

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
        signal = fft_fractional_delay(signal, -fractional_offset)

    logger.info(
        f"Timing corrected: coarse={coarse_offset.tolist() if hasattr(coarse_offset, 'tolist') else coarse_offset}, "
        f"fractional={'applied' if apply_frac else 'skipped'}, mode={mode!r}."
    )

    if was_1d:
        return signal[0]
    return signal


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
    signal: ArrayType,
    fs: float,
    modulation: str,
    order: int,
    search_range: Optional[Tuple[float, float]] = None,
    nfft: Optional[int] = None,
    debug_plot: bool = False,
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
    # CPU freq array used for parabola scalar indexing; device version only when
    # search_range masking is needed (avoids a spurious GPU allocation on every call).
    freqs_np = np.fft.fftfreq(nfft, d=1.0 / fs)  # (nfft,) — always on CPU

    mag = xp.abs(X_M)  # (C, nfft)
    mag[:, 0] = 0.0  # zero DC for all channels

    # Restrict search to [M·f_min, M·f_max] when search_range is given
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

    # For MIMO, sum spectra across channels before peak detection so the
    # estimator benefits from coherent accumulation rather than averaging
    # independent (noisy) per-channel estimates.  For SISO this is a no-op
    # (C=1, sum = identity).
    mag_combined = xp.sum(mag, axis=0)  # (nfft,)

    k_peak_combined = int(xp.argmax(mag_combined))
    k_safe = max(1, min(k_peak_combined, nfft - 2))
    a = float(mag_combined[k_safe - 1])
    b_ = float(mag_combined[k_safe])
    c = float(mag_combined[k_safe + 1])
    denom = a - 2 * b_ + c
    mu = 0.5 * (a - c) / denom if abs(denom) > 1e-15 else 0.0
    mu = max(-0.5, min(0.5, mu))
    f_est = (freqs_np[k_peak_combined] + mu * (fs / nfft)) / M

    logger.info(
        f"FOE (M-th power, M={M}): {f_est:.2f} Hz "
        f"[nfft={nfft}, search_range={search_range}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        # Pass the combined spectrum and a single peak index for the plot
        _plotting.frequency_offset_spectrum(
            mag_spectrum=to_device(mag_combined[None, :], "cpu"),
            freqs=freqs_np,
            M=M,
            k_peaks=np.array([k_peak_combined]),
            f_estimates=[f_est],
            search_range=search_range,
            show=True,
        )

    return f_est


def estimate_frequency_offset_differential(
    signal: ArrayType,
    fs: float,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    ref_signal: Optional[ArrayType] = None,
    weighted: bool = True,
    debug_plot: bool = False,
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
       from raw differential phases. Best for constant-envelope signals.

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
        # Coherent MIMO combining: sum weighted complex y_diff across all
        # channels before taking the angle.  Every channel observes the
        # same frequency offset Δf; their complex differential products
        # add coherently while incoherent per-channel noise averages down
        # as 1/√C.  Taking angle(sum_channels) is equivalent to a
        # maximum-likelihood estimate under equal-power, independent noise.
        # This is far better than angle(sum_c) per channel → mean(angle),
        # where a single noisy channel can pull the mean-of-angles off.
        combined_sum = xp.sum(w * y_diff, axis=(-2, -1))  # scalar complex
    else:
        combined_sum = xp.sum(y_diff)  # scalar complex, sum over C and N-1

    f_est = float(xp.angle(combined_sum)) * (fs / (2 * np.pi)) / M
    mode_str = "data-aided" if ref_signal is not None else f"blind M={M}"
    logger.info(f"FOE (differential, {mode_str}): {f_est:.2f} Hz")

    if debug_plot:
        from . import plotting as _plotting

        _plotting.differential_phase_trajectory(
            y_diff=to_device(y_diff, "cpu"),
            f_est=f_est,
            fs=fs,
            M=M,
            show=True,
        )

    return f_est


def estimate_frequency_offset_pilots(
    signal: ArrayType,
    pilot_indices: ArrayType,
    pilot_values: ArrayType,
    fs: float,
    debug_plot: bool = False,
) -> float:
    """
    Estimates frequency offset from pilot symbols via phase slope fitting.

    Extracts the received phase at each pilot position, demodulates against
    the known pilot values to obtain the residual phase, unwraps the pilot
    phase sequence, then fits a least-squares line to the unwrapped phase as
    a function of pilot sample time.  The slope gives the frequency offset:
    ``Δf = slope / (2π)``.

    Parameters
    ----------
    signal : array_like
        Received complex samples. Shape: (N,) or (C, N).
    pilot_indices : array_like of int
        Sample indices of pilot positions in increasing order. Shape: (P,).
        Must be unique and sorted.  Supports any pilot arrangement:

        * **Comb (scattered):** uniform grid, e.g. every 16th sample.
          Lock range determined by comb spacing.
        * **Block (contiguous cluster):** e.g. ``[0, 1, ..., L-1]``.
          For contiguous blocks the differential estimator
          (``estimate_frequency_offset_differential`` with
          ``ref_signal=``) provides ML performance.
        * **Multi-block:** e.g. a front preamble and a mid-burst pilot
          cluster.  Lock range is set by the **largest gap** between any
          two consecutive pilot indices (see Notes).
    pilot_values : array_like
        Known transmitted pilot symbols at the corresponding indices.
        Shape: (P,) for shared pilots (broadcast to all MIMO channels),
        or (C, P) for per-channel pilots.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    float
        Estimated frequency offset in Hz. For MIMO, per-channel estimates
        are averaged to a single scalar.

    Notes
    -----
    The demodulated pilot phase follows:

    .. math::

        \\hat{\\phi}[k] = 2\\pi \\Delta f \\cdot t_k + \\phi_0 + \\text{noise}

    where :math:`t_k = \\text{pilot\\_indices}[k] / f_s`.  The
    minimum-variance unbiased estimator for :math:`\\Delta f` is the
    least-squares slope of the unwrapped phase vs. time, solved via the
    centered normal equations:

    .. math::

        \\hat{\\Delta f} = \\frac{1}{2\\pi}
            \\frac{\\sum_k (t_k - \\bar{t})(\\hat{\\phi}[k] - \\bar{\\phi})}
                  {\\sum_k (t_k - \\bar{t})^2}

    **Lock range:** ``xp.unwrap`` bridges each gap between consecutive
    pilot indices.  The gap that limits the lock range is the largest one:

    .. math::

        |\\Delta f| < \\frac{f_s}{2 \\cdot \\max_k (t_{k+1} - t_k)^{-1}}
            = \\frac{f_s}{2 \\cdot \\text{max\\_gap}}

    where ``max_gap`` is the maximum spacing (in samples) between any two
    consecutive entries of ``pilot_indices``.

    * Comb with period *d*: ``max_gap = d``, lock range ``= fs/(2d)``.
    * Two-block pilots with front block ``[0..L-1]`` and back block
      ``[N-L..N-1]``: ``max_gap = N-2L``, lock range ``= fs/(2(N-2L))``.
      For large *N*, this can be very small.  Use
      :func:`estimate_frequency_offset_mth_power` as a coarse stage first.

    **Accuracy vs. data-aided:** For the same number of pilots P, widely
    spaced pilots achieve a lower CRLB than a contiguous block because the
    long time baseline increases the denominator :math:`\\sum (t_k-\\bar t)^2`.

    References
    ----------
    S. A. Tretter, "Estimating the frequency of a noisy sinusoid by linear
    regression," *IEEE Trans. Inf. Theory*, vol. 31, no. 6, pp. 832–835,
    Nov. 1985.
    """
    signal, xp, _ = dispatch(signal)
    was_1d = signal.ndim == 1
    if was_1d:
        signal = signal[None, :]
    C, N = signal.shape

    pilot_indices_np = to_device(pilot_indices, "cpu").astype(np.intp)
    pilot_values_xp = xp.asarray(pilot_values)
    P = len(pilot_indices_np)

    if pilot_values_xp.ndim == 1:
        pilot_values_xp = xp.broadcast_to(pilot_values_xp[None, :], (C, P))

    # Extract and demodulate: phase = angle(r · conj(s)) = 2π·Δf·t + φ₀ + noise
    r_pilots = signal[:, pilot_indices_np]  # (C, P)
    phi_pilots = xp.angle(r_pilots * xp.conj(pilot_values_xp))  # (C, P)

    # Unwrap in float64; cp.unwrap preserves input dtype so cast before calling
    phi_pilots_u = xp.unwrap(phi_pilots.astype(xp.float64), axis=-1)  # (C, P)

    # Centered normal equations on-backend — vectorised across all C channels.
    # Centering avoids cancellation and matches the MVUE from Tretter (1985).
    t_xp = xp.asarray(pilot_indices_np.astype(np.float64)) / fs  # (P,)
    t_c = t_xp - xp.mean(t_xp)  # (P,) centred
    t_var = float(xp.dot(t_c, t_c))  # scalar Σ(t-t̄)²

    phi_c = phi_pilots_u - xp.mean(phi_pilots_u, axis=-1, keepdims=True)  # (C, P)
    slopes = xp.sum(phi_c * t_c[None, :], axis=-1) / t_var  # (C,)

    f_est = float(xp.mean(slopes)) / (2.0 * np.pi)
    max_gap = int(np.max(np.diff(pilot_indices_np))) if P > 1 else 0
    lock_range = fs / (2 * max_gap) if max_gap > 0 else float("inf")
    logger.info(
        f"FOE (pilots): {f_est:.2f} Hz "
        f"[P={P} pilots, max_gap={max_gap} samples, lock_range=±{lock_range:.1f} Hz]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.pilot_phase_estimate(
            pilot_indices=pilot_indices_np,
            phi_pilots_u=to_device(phi_pilots_u, "cpu"),
            f_est=f_est,
            fs=fs,
            show=True,
        )

    return f_est


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
    logger.debug(
        f"Applying frequency offset correction: {offset:.4f} Hz (fs={fs:.0f} Hz)"
    )
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


# -----------------------------------------------------------------------------
# Carrier Phase Recovery (CPR)
# -----------------------------------------------------------------------------

# Lazy-compiled Numba kernels for sequential CPR algorithms.
_NUMBA_PLL: dict = {}


def _get_numba_dd_pll():
    """JIT-compile and cache the Numba DD-PLL sample-wise loop kernel.

    Returns
    -------
    callable or None
        Numba-compiled ``_dd_pll_loop``, or ``None`` if Numba is not installed.
    """
    if "dd_pll" not in _NUMBA_PLL:
        try:
            import numba  # noqa: PLC0415
        except ImportError:
            _NUMBA_PLL["dd_pll"] = None
            return None

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

                # 2nd-order loop filter (reduces to 1st order when beta=0):
                #   φ[n+1] = φ[n] + μ·e[n] + ν[n]
                #   ν[n]   = ν[n−1] + β·e[n]
                phi = phi + mu * e + freq
                freq = freq + beta * e

                phase_est[n] = phi

            return phase_est

        _NUMBA_PLL["dd_pll"] = _dd_pll_loop

    return _NUMBA_PLL["dd_pll"]


def _dd_pll_numpy(sym, const, mu, beta, phi0, freq0):
    """Pure-NumPy DD-PLL fallback (sequential Python loop over N samples).

    Used when Numba is not installed.  Correct but slow for large N.
    """
    logger.warning(
        "DD-PLL: Numba not installed — running a sequential Python loop. "
        "Install numba for a 10–100× speedup: `uv add numba`."
    )
    N = len(sym)
    phase_est = np.empty(N, dtype=np.float64)
    phi = float(phi0)
    freq = float(freq0)
    const_np = to_device(const, "cpu")

    for n in range(N):
        y = sym[n] * np.exp(-1j * phi)
        d = const_np[np.argmin(np.abs(y - const_np) ** 2)]
        e = float(np.imag(y * np.conj(d)))
        phi = phi + mu * e + freq
        freq = freq + beta * e
        phase_est[n] = phi

    return phase_est


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
    S_b = xp.sum(blocks_c**M, axis=-1)  # (C, N_blocks)

    # Raw block phase in [-π/M, π/M)
    phi_raw = xp.angle(S_b) / M  # (C, N_blocks)

    # M-fold unwrap: scale into 2π domain, unwrap, re-scale back.
    # Cast to float64 before unwrap — cp.unwrap preserves input dtype so float32
    # would lose precision during the discontinuity test (diff vs 2π threshold).
    phi_u = xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M  # (C, N_blocks)

    # QAM bias correction.
    # For square QAM, E[d^M] is always real and negative: every diagonal symbol
    # d = a(1±j) gives d^4 ∝ (1+j)^4 = -4, and these corner-like points dominate
    # the mean.  This introduces a deterministic π/M offset in phi_u that rotates
    # the corrected constellation away from its canonical orientation.  PSK has
    # no such bias (all d^M are real positive).  Subtracting π/M removes it.
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

    # block_centers[b] = b * block_size + half_bs  (integer-aligned float64)
    half_bs = block_size // 2
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + half_bs
    all_positions = xp.arange(N, dtype=xp.float64)

    # Pre-compute interpolation indices and weights (identical for every channel).
    # Integer floor division is bitwise-exact for integer positions, avoiding the
    # ambiguous side='left'/'right' behaviour of searchsorted at block boundaries.
    #   block b is "to the left" of position n when b*bs + half_bs <= n
    #   => b <= (n - half_bs) / bs  => idx_left = floor((n - half_bs) / bs)
    pos_int = all_positions.astype(xp.int64)  # (N,)
    idx_left = xp.clip((pos_int - half_bs) // block_size, 0, N_blocks - 2)  # (N,)
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
    # CHUNK × B × M_const × 4 bytes ≤ ~32 MB regardless of signal length.
    CHUNK_N = 1024

    phi_full = xp.zeros((C, N), dtype=xp.float64)

    for ch in range(C):
        sym = symbols[ch, :N_trunc]  # (N_trunc,)

        # Accumulate block-average error metric: (N_blocks, B).
        # CHUNK_N (1024) is an exact multiple of block_size (32), so each chunk
        # covers a whole number of blocks with no remainder — no edge-case needed.
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
            show=True,
            title="CPR — Blind Phase Search",
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


def recover_carrier_phase_decision_directed(
    symbols: ArrayType,
    modulation: str,
    order: int,
    mu: float = 1e-2,
    beta: float = 0.0,
    phase_init: float = 0.0,
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
    the throughput bottleneck.  Install ``numba`` for a 10-100x speedup
    over the pure-Python fallback.

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

    # Select kernel: Numba (fast) or pure-NumPy fallback (correct but slow)
    numba_kernel = _get_numba_dd_pll()

    phi_full = np.zeros((C, N), dtype=np.float64)

    for ch in range(C):
        sym = symbols_cpu[ch].astype(np.complex128)  # (N,)
        sym_r = sym.real.copy()
        sym_i = sym.imag.copy()

        if numba_kernel is not None:
            phi_full[ch] = numba_kernel(
                sym_r,
                sym_i,
                const_r,
                const_i,
                float(mu),
                float(beta),
                float(phase_init),
                0.0,
            )
        else:
            phi_full[ch] = _dd_pll_numpy(
                sym,
                const_np,
                float(mu),
                float(beta),
                float(phase_init),
                0.0,
            )

    # Move result back to original device
    if xp is not np:
        phi_full = xp.asarray(phi_full)

    phi_mean_deg = float(np.mean(phi_full)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full)) * 180.0 / np.pi
    loop_order = "2nd" if beta > 0.0 else "1st"
    logger.info(
        f"CPR (DD-PLL, {loop_order}-order): phase mean={phi_mean_deg:.2f}°, "
        f"std={phi_std_deg:.2f}° [mu={mu}, beta={beta}, C={C}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full if xp is np else to_device(phi_full, "cpu"),
            show=True,
            title=f"CPR — DD-PLL ({loop_order}-order)",
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
