"""Linear block equalizers: zero-forcing and static tap application."""

from __future__ import annotations

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..filtering import _ols_backward, _ols_forward
from ..logger import logger
from ._common import _build_padded_samples, _normalize_inputs

# -----------------------------------------------------------------------------
# BLOCK equalization
# -----------------------------------------------------------------------------


def zf_equalizer(
    samples: ArrayType,
    channel_estimate: ArrayType,
    noise_variance: float = 0.0,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Zero-Forcing / MMSE frequency-domain block equalizer.

    Given a channel impulse response estimate, equalizes the received signal
    in the frequency domain. Supports both SISO and MIMO channels.

    When ``noise_variance=0``, this is a pure Zero-Forcing equalizer.
    When ``noise_variance > 0``, this becomes an MMSE equalizer that
    regularizes spectral nulls to avoid noise enhancement.

    Parameters
    ----------
    samples : array_like
        Received samples. Shape: ``(N,)`` for SISO or ``(C, N)`` for MIMO.
    channel_estimate : array_like
        Channel impulse response. Shape: ``(L,)`` for SISO or
        ``(C, C, L)`` for MIMO (each ``[i, j]`` is the FIR from input j
        to output i).
    noise_variance : float, default 0.0
        Noise variance (sigma^2) for MMSE regularization.
        ``0.0`` gives pure ZF.

    Returns
    -------
    array_like
        Equalized samples. Same shape and backend as input.
    """
    logger.info(f"ZF/MMSE equalizer: noise_variance={noise_variance:.2e}")
    samples, xp, _ = dispatch(samples)
    channel_estimate = xp.asarray(channel_estimate)

    # Capture BEFORE any reshape so the later frequency-domain branch
    # can still take the efficient scalar path for SISO channels.
    siso_channel = channel_estimate.ndim == 1

    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]
        if siso_channel:
            channel_estimate = channel_estimate[None, None, :]

    num_ch, N = samples.shape
    L = channel_estimate.shape[-1] if channel_estimate.ndim > 0 else 1
    reg = noise_variance if noise_variance > 0 else 1e-12

    # N_fft must satisfy discard = N_fft//4 >= L so the IIR filter's causal
    # and anti-causal tails both decay within the discarded guard regions.
    N_fft = max(1024, 1 << (max(1, 4 * L) - 1).bit_length())

    logger.debug(f"ZF/MMSE internals: N={N}, L={L}, num_ch={num_ch}, N_fft={N_fft}")

    # --- Shared OLS forward pass: pad → stride_tricks → batch FFT ---
    Y, meta = _ols_forward(samples, N_fft)  # Y: (num_ch, num_blocks, N_fft)

    if siso_channel:
        # SISO: scalar frequency-domain ZF/MMSE inversion.
        # channel_estimate was reshaped to (1,1,L) for was_1d; flatten to 1D.
        H = xp.fft.fft(channel_estimate.reshape(-1), n=N_fft)
        W = xp.conj(H) / (xp.abs(H) ** 2 + reg)
        X_hat_f = Y * W
    else:
        # MIMO: per-bin (CxC) matrix inversion — cannot reduce to scalar multiply.
        H_f = xp.fft.fft(channel_estimate, n=N_fft, axis=-1)
        Hk = xp.transpose(H_f, (2, 0, 1))  # (N_fft, C_rx, C_tx)
        Hk_H = xp.conj(xp.transpose(Hk, (0, 2, 1)))  # (N_fft, C_tx, C_rx)
        HHh = Hk @ Hk_H  # (N_fft, C_rx, C_rx)

        eye = xp.eye(num_ch, dtype=samples.dtype)[None, :, :]
        if num_ch == 2:
            # Fast-path: Explicit 2x2 matrix inversion via Cramer's Rule avoiding
            # GPU LAPACK solver overheads (e.g. cuSOLVER batch inversion).
            H_reg = HHh + reg * eye
            a = H_reg[:, 0, 0]
            b = H_reg[:, 0, 1]
            c = H_reg[:, 1, 0]
            d = H_reg[:, 1, 1]
            det = a * d - b * c

            inv_term = xp.empty_like(H_reg)
            inv_term[:, 0, 0] = d / det
            inv_term[:, 0, 1] = -b / det
            inv_term[:, 1, 0] = -c / det
            inv_term[:, 1, 1] = a / det
        else:
            inv_term = xp.linalg.inv(HHh + reg * eye)

        Wk = Hk_H @ inv_term  # (N_fft, C_tx, C_rx)

        # Vectorized batch matrix multiplication across frequency bins and blocks.
        # Uses explicit batched GEMM instead of einsum for better GPU utilization.
        Y_t = xp.transpose(Y, (2, 0, 1))  # (N_fft, num_ch, num_blocks)
        X_hat_k = Wk @ Y_t  # (N_fft, C_tx, C_rx) @ (N_fft, C_rx, num_blocks)
        X_hat_f = xp.transpose(X_hat_k, (1, 2, 0))  # (num_ch, num_blocks, N_fft)

    # --- Shared OLS backward pass: batch IFFT → symmetric discard → reshape ---
    out = _ols_backward(X_hat_f, meta)

    if debug_plot:
        from .. import plotting as _plotting  # lazy import avoids circular dep

        _plotting.plot_zf_equalizer_response(
            channel_estimate=to_device(channel_estimate, "cpu"),
            noise_variance=noise_variance,
            show=True,
        )

    return out[0] if was_1d else out


def apply_taps(
    samples: ArrayType,
    weights: ArrayType,
    sps: int = 2,
    normalize: bool = True,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
) -> ArrayType:
    """Apply frozen equalizer taps to a signal (inference pass, no weight updates).

    Performs the butterfly FIR forward pass using pre-converged weights,
    decimating by ``sps`` to produce one output symbol per ``sps`` input samples.
    Suitable for reusing frozen taps from a prior equalizer run on a new signal
    without re-running adaptation.

    The forward pass implements::

        y[i, n] = sum_j sum_t conj(W[i,j,t]) * x[j, n*sps + t]

    which is the same inner computation as the Numba/JAX adaptive-equalizer
    kernels, fully vectorized over ``n`` via a single batched ``einsum``.

    Parameters
    ----------
    samples : array_like
        Input samples. Shape: ``(N_samples,)`` for SISO or
        ``(C, N_samples)`` for MIMO. Typically at ``sps`` samples/symbol.
    weights : array_like
        Frozen tap weights, typically ``EqualizerResult.weights`` from a
        prior equalizer run. Shape: ``(num_taps,)`` for SISO or
        ``(C, C, num_taps)`` for MIMO butterfly.
    sps : int, default 2
        Samples per symbol. Output length is ``N_samples // sps``.
        Unlike the adaptive equalizers, any ``sps >= 1`` is accepted.
    normalize : bool, default True
        If ``True``, normalize ``samples`` to unit symbol power before
        filtering (same pre-processing as the adaptive equalizers via
        ``_normalize_inputs``). Set to ``False`` if the caller has already
        scaled the input consistently with the original training run.
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor.  When provided and
        ``normalize=True``, skips RMS recomputation and uses this value
        directly.  Typically ``EqualizerResult.input_norm_factor`` from the
        prior equalizer run that produced ``weights``.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block.  See ``lms()``
        for the full description; behaviour is identical.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy when ``samples_prefix`` is ``None``.  See
        ``lms()`` for the full description; behaviour is identical.

    Returns
    -------
    array_like
        Equalized symbols. Shape: ``(N_sym,)`` for SISO or
        ``(C, N_sym)`` for MIMO, where ``N_sym = N_samples // sps``.
        Resides on the same backend as ``samples``.

    Examples
    --------
    Freeze taps from a training run and apply to a new capture::

        result = equalization.lms(train_signal, training_symbols, num_taps=31)
        y = equalization.apply_taps(new_signal, result.weights,
                                    input_norm_factor=result.input_norm_factor)
    """
    samples, xp, _ = dispatch(samples)
    weights = xp.asarray(weights)

    was_1d = samples.ndim == 1

    # Promote SISO → MIMO shapes for uniform butterfly code path
    if was_1d:
        samples = samples[None, :]  # (N,) → (1, N)
    if weights.ndim == 1:
        weights = weights[None, None, :]  # (T,) → (1, 1, T)

    C, N = samples.shape
    num_taps = weights.shape[-1]
    n_sym = N // sps

    _at_eq_norm = None
    if normalize:
        samples, _, _at_eq_norm = _normalize_inputs(
            samples, None, sps, input_norm_factor=input_norm_factor
        )

    # Pad left by center tap so window[0] is center-aligned with sample 0,
    # pad right to fill the last window completely — mirrors LMS pre-processing.
    c_tap = num_taps // 2
    pad_left = c_tap
    pad_right = n_sym * sps - N + num_taps - 1 - pad_left
    if samples_prefix is None and pad_mode == "zeros":
        samples_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))
    else:
        # _build_padded_samples is backend-generic: pad on the input device
        # instead of round-tripping the full signal through the host.
        samples_padded = xp.ascontiguousarray(
            _build_padded_samples(
                samples.astype(xp.complex64),
                pad_left,
                pad_right,
                samples_prefix,
                pad_mode,
                _at_eq_norm,
                sps,
            )
        )

    # Zero-copy sliding-window view: (C, N_sym, num_taps)
    # windows[c, n, t] = samples_padded[c, n*sps + t]
    s0, s1 = samples_padded.strides
    windows = xp.lib.stride_tricks.as_strided(
        samples_padded,
        shape=(C, n_sym, num_taps),
        strides=(s0, sps * s1, s1),
    )

    # Butterfly forward pass — single batched GEMM (dispatches to cuBLAS on GPU)
    # y[i, n] = Σ_j Σ_t conj(W[i,j,t]) * windows[j, n, t]
    y = xp.einsum("ijt,jnt->in", xp.conj(weights), windows)  # (C, N_sym)

    return y[0] if was_1d else y
