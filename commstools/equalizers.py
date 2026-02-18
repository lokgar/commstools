"""
Adaptive and block equalization algorithms.

This module provides production-grade equalizer implementations for
compensating inter-symbol interference (ISI) and channel distortion
in digital communication systems.

Adaptive algorithms (LMS, RLS, CMA) use JAX ``lax.scan`` for compiled
sequential weight updates on both CPU and GPU. Block algorithms (ZF/MMSE)
use NumPy/CuPy dispatch for vectorized frequency-domain processing.

All adaptive equalizers support a **butterfly MIMO** topology: for an input
of shape ``(C, N)``, the equalizer maintains a ``(C, C, num_taps)`` weight
matrix so that each output stream is a filtered combination of *all* input
streams. This enables cross-channel interference cancellation (e.g.
dual-polarization demultiplexing in coherent optical, spatial MIMO demux).

Input SPS Convention
--------------------
Adaptive equalizers require T/2-spaced input (2 samples/symbol), the
industry standard for coherent optical and many wireless systems. The
equalizer outputs one symbol per 2 input samples, decimating to symbol rate.
A ``ValueError`` is raised if ``sps != 2``.

Functions
---------
lms :
    Least Mean Squares / Normalized LMS adaptive equalizer.
rls :
    Recursive Least Squares adaptive equalizer.
cma :
    Constant Modulus Algorithm blind equalizer.
zf_equalizer :
    Zero-Forcing / MMSE frequency-domain block equalizer.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .backend import ArrayType, _get_jax, dispatch, from_jax, to_jax, to_device
from .logger import logger


# ============================================================================
# RESULT CONTAINER
# ============================================================================


@dataclass
class EqualizerResult:
    """Container for equalizer outputs.

    Attributes
    ----------
    y_hat : ArrayType
        Equalized output symbols. Shape: ``(N_sym,)`` or ``(C, N_sym)``.
    weights : ArrayType
        Final tap weight vector. Shape: ``(num_taps,)`` for SISO
        or ``(C, C, num_taps)`` for MIMO butterfly.
    error : ArrayType
        Error signal history. Shape: ``(N_sym,)`` or ``(C, N_sym)``.
    weights_history : ArrayType or None
        Tap weight evolution over time. Only populated when
        ``store_weights=True``.
    """

    y_hat: ArrayType
    weights: ArrayType
    error: ArrayType
    weights_history: Optional[ArrayType] = None


# ============================================================================
# JAX JIT CACHE — ADAPTIVE EQUALIZER KERNELS
# ============================================================================
#
# Alignment convention (center-tap):
#   At scan index k, the input window spans samples[k*sps : k*sps + num_taps].
#   The center-tap at index num_taps//2 references sample k*sps + num_taps//2.
#   Training symbols are pre-shifted by (num_taps//2) in the public functions
#   so that desired[k] matches the symbol at the center of the window.

_JITTED_EQ = {}


def _get_jitted_lms(num_taps, stride, n_train, const_size, num_ch, normalize):
    """Returns JIT-compiled LMS/NLMS butterfly scan."""
    key = ("lms", num_taps, stride, n_train, const_size, num_ch, normalize)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def lms_scan(x_input, training_padded, constellation, w_init, step_size):
            # x_input: (C, N_samples)    training_padded: (C, N_sym)
            # constellation: (M,)        w_init: (C, C, num_taps)

            def step(W, idx):
                sample_idx = idx * stride

                def get_win(ch):
                    return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

                X_wins = jax.vmap(get_win)(x_input)  # (C, num_taps)

                # Butterfly: y_i = sum_j conj(W[i,j]) . X_wins[j]
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)  # (C,)

                # Training or decision-directed
                def slicer(ch_y):
                    return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

                dd = jax.vmap(slicer)(y)  # (C,)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)

                e = d - y  # (C,)

                # Step size: NLMS normalizes by input power
                if normalize:
                    power = jnp.real(jnp.sum(X_wins * jnp.conj(X_wins))) + 1e-10
                    mu_eff = step_size / power
                else:
                    mu_eff = step_size

                W_new = W + mu_eff * jnp.einsum("i,jt->ijt", e, jnp.conj(X_wins))
                return W_new, (y, e, W_new)

            n_sym = training_padded.shape[1]
            W_final, (y_hat, errors, w_hist) = jax.lax.scan(
                step, w_init, jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = lms_scan
    return _JITTED_EQ[key]


def _get_jitted_rls(num_taps, stride, n_train, const_size, num_ch):
    """Returns JIT-compiled RLS butterfly scan."""
    key = ("rls", num_taps, stride, n_train, const_size, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def rls_scan(x_input, training_padded, constellation, w_init, P_init, lam):
            # x_input: (C, N_samples)    training_padded: (C, N_sym)
            # w_init: (C, C, num_taps)   P_init: (C, C*num_taps, C*num_taps)

            def step(carry, idx):
                W, P = carry
                sample_idx = idx * stride

                def get_win(ch):
                    return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

                X_wins = jax.vmap(get_win)(x_input)  # (C, num_taps)

                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)

                def slicer(ch_y):
                    return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

                dd = jax.vmap(slicer)(y)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e = d - y

                x_bar = X_wins.flatten()  # (C * num_taps,)

                def rls_update(args):
                    w_row, P_i, e_i = args
                    w_flat = w_row.flatten()
                    Px = P_i @ x_bar  # P @ u
                    denom = lam + jnp.dot(jnp.conj(x_bar), Px)  # λ + u^H P u
                    k = Px / denom  # gain vector
                    w_flat_new = w_flat + k * jnp.conj(e_i)  # w + k * e*
                    P_new = (P_i - jnp.outer(k, jnp.conj(x_bar) @ P_i)) / lam
                    return w_flat_new.reshape(num_ch, num_taps), P_new

                W_new, P_new = jax.vmap(rls_update)((W, P, e))
                return (W_new, P_new), (y, e, W_new)

            n_sym = training_padded.shape[1]
            (W_final, _), (y_hat, errors, w_hist) = jax.lax.scan(
                step, (w_init, P_init), jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = rls_scan
    return _JITTED_EQ[key]


def _get_jitted_cma(num_taps, stride, num_ch, normalize, n_sym):
    """Returns JIT-compiled CMA butterfly scan."""
    key = ("cma", num_taps, stride, num_ch, normalize, n_sym)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def cma_scan(x_input, w_init, step_size, r2):
            def step(W, idx):
                sample_idx = idx * stride

                def get_win(ch):
                    return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

                X_wins = jax.vmap(get_win)(x_input)  # (C, num_taps)
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)

                # CMA error: e_i = y_i * (|y_i|^2 - R2)
                e = y * (jnp.abs(y) ** 2 - r2)

                if normalize:
                    power = jnp.real(jnp.sum(X_wins * jnp.conj(X_wins))) + 1e-10
                    mu_eff = step_size / power
                else:
                    mu_eff = step_size

                W_new = W - mu_eff * jnp.einsum("i,jt->ijt", e, jnp.conj(X_wins))
                return W_new, (y, e, W_new)

            W_final, (y_hat, errors, w_hist) = jax.lax.scan(
                step, w_init, jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = cma_scan
    return _JITTED_EQ[key]


# ============================================================================
# SHARED HELPERS
# ============================================================================


def _init_butterfly_weights(num_ch, num_taps, jnp):
    """Center-tap identity initialization for butterfly weight matrix.

    Returns ``(C, C, num_taps)`` with diagonal center taps = 1.
    """
    W = jnp.zeros((num_ch, num_ch, num_taps), dtype=jnp.complex64)
    center = num_taps // 2
    for i in range(num_ch):
        W = W.at[i, i, center].set(1.0 + 0j)
    return W


def _prepare_training(
    training_symbols, reference_constellation, num_ch, n_sym, num_taps, sps=2
):
    """Build constellation and center-tap-aligned training array.

    Training symbols are delayed to match the equalizer's center-tap delay.
    Delay in symbols = ``(num_taps // 2) // sps``.

    Returns
    -------
    constellation_np : ndarray (M,) complex64
    train_full : ndarray (C, n_sym) complex64
    n_train_aligned : int
    """
    # Delay in samples is num_taps // 2. Convert to symbols.
    delay = (num_taps // 2) // sps

    if reference_constellation is not None:
        constellation_np = (
            np.asarray(to_device(reference_constellation, "cpu"))
            .flatten()
            .astype(np.complex64)
        )
    elif training_symbols is not None:
        train_flat = np.asarray(to_device(training_symbols, "cpu")).flatten()
        constellation_np = np.unique(np.round(train_flat, decimals=8)).astype(
            np.complex64
        )
    else:
        raise ValueError(
            "Either training_symbols or reference_constellation must be provided."
        )

    if training_symbols is not None:
        train_np = np.asarray(to_device(training_symbols, "cpu")).astype(np.complex64)
        if train_np.ndim == 1:
            train_np = (
                np.tile(train_np[None, :], (num_ch, 1))
                if num_ch > 1
                else train_np[None, :]
            )
        n_raw = train_np.shape[1]
        n_train_aligned = max(0, min(n_raw - delay, n_sym))
        train_full = np.zeros((num_ch, n_sym), dtype=np.complex64)
        if n_train_aligned > 0:
            train_full[:, :n_train_aligned] = train_np[
                :, delay : delay + n_train_aligned
            ]
    else:
        n_train_aligned = 0
        train_full = np.zeros((num_ch, n_sym), dtype=np.complex64)

    return constellation_np, train_full, n_train_aligned


def _unpack_result(
    y_hat_jax, errors_jax, W_final_jax, w_hist_jax, was_1d, store_weights
):
    """Convert JAX scan outputs to backend arrays with proper shape."""
    y_hat = from_jax(y_hat_jax).T  # (N_sym, C) -> (C, N_sym)
    errors = from_jax(errors_jax).T
    W_final = from_jax(W_final_jax)

    if was_1d:
        y_hat = y_hat[0]
        errors = errors[0]
        W_final = W_final[0, 0]

    w_history = None
    if store_weights:
        w_history = from_jax(w_hist_jax)
        if was_1d:
            w_history = w_history[:, 0, 0, :]

    return EqualizerResult(
        y_hat=y_hat, weights=W_final, error=errors, weights_history=w_history
    )


def _validate_sps(sps, num_taps):
    """Validate SPS is exactly 2 (T/2-spaced) and num_taps is sufficient."""
    if sps != 2:
        raise ValueError(
            f"Adaptive equalizers require 2 samples/symbol "
            f"(T/2-spaced input). Got sps={sps}."
        )
    if num_taps < 2 * sps:
        logger.warning(
            f"num_taps={num_taps} is small for sps={sps}. "
            f"Recommend num_taps >= {4 * sps} for fractionally-spaced equalization."
        )


# ============================================================================
# ADAPTIVE EQUALIZERS
# ============================================================================


def lms(
    samples: ArrayType,
    training_symbols: Optional[ArrayType] = None,
    num_taps: int = 21,
    step_size: float = 0.01,
    reference_constellation: Optional[ArrayType] = None,
    sps: int = 2,
    normalize: bool = True,
    store_weights: bool = False,
) -> EqualizerResult:
    """
    Least Mean Squares adaptive equalizer with butterfly MIMO support.

    Supports data-aided (training) and decision-directed (DD) modes.
    When ``training_symbols`` are provided, the equalizer uses them for the
    initial convergence phase, then switches to DD mode using the
    ``reference_constellation`` for hard-decision slicing.

    For MIMO inputs ``(C, N)``, a butterfly ``(C, C, num_taps)`` filter
    structure is used so each output is a weighted sum of all input streams,
    enabling cross-channel interference cancellation.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: ``(N_samples,)`` for SISO or
        ``(C, N_samples)`` for MIMO butterfly equalization.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
    training_symbols : array_like, optional
        Known transmitted symbols (at symbol rate, 1 SPS).
        Shape: ``(N_train,)`` for SISO or ``(C, N_train)`` for MIMO.
        If None, pure DD mode (requires ``reference_constellation``).
    num_taps : int, default 21
        Number of equalizer taps per FIR filter. For fractionally-spaced
        equalization (sps > 1), use at least ``4 * sps`` taps.
    step_size : float, default 0.01
        LMS step size (mu). When ``normalize=True`` (NLMS, default),
        this is the normalized step size in (0, 2) — typical: 0.01 to 0.1.
        When ``normalize=False``, this is the raw step size.
    reference_constellation : array_like, optional
        Constellation points for DD slicing. Shape: ``(M,)``.
        Required if ``training_symbols`` is None.
    sps : int, default 2
        Samples per symbol at the input. Must be 2 (T/2-spaced).
        The equalizer outputs one symbol per ``sps`` input samples.
    normalize : bool, default True
        If True, use Normalized LMS (NLMS): step size is divided by
        the instantaneous input power ``||x||^2``. This makes convergence
        robust to input power variations and is strongly recommended.
    store_weights : bool, default False
        If True, stores weight trajectory in ``weights_history``.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory. Arrays reside on the same backend as input.
    """
    logger.info(
        f"LMS equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"normalize={normalize}"
    )
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for adaptive equalizers.")

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    n_sym = (n_samples - num_taps + 1) // stride
    if n_sym <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {num_taps} taps with sps={sps}."
        )

    constellation_np, train_full, n_train_aligned = _prepare_training(
        training_symbols, reference_constellation, num_ch, n_sym, num_taps, sps=sps
    )

    # Convert to JAX — preserves device (CuPy → JAX GPU via DLPack)
    x_jax = to_jax(samples)
    if was_1d:
        x_jax = x_jax[None, :]

    # Ensure all JAX arrays are on the same device as the input
    try:
        if hasattr(x_jax, "device"):
            platform = x_jax.device.platform
        else:
            platform = list(x_jax.devices())[0].platform
    except Exception:
        platform = "cpu"

    train_jax = to_jax(train_full, device=platform)
    const_jax = to_jax(constellation_np, device=platform)
    w_init = _init_butterfly_weights(num_ch, num_taps, jnp)
    w_init = to_jax(w_init, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)

    scan_fn = _get_jitted_lms(
        num_taps, stride, n_train_aligned, len(constellation_np), num_ch, normalize
    )
    y_jax, e_jax, W_jax, wh_jax = scan_fn(x_jax, train_jax, const_jax, w_init, mu_jax)

    return _unpack_result(y_jax, e_jax, W_jax, wh_jax, was_1d, store_weights)


def rls(
    samples: ArrayType,
    training_symbols: Optional[ArrayType] = None,
    num_taps: int = 21,
    forgetting_factor: float = 0.99,
    delta: float = 0.01,
    reference_constellation: Optional[ArrayType] = None,
    sps: int = 2,
    store_weights: bool = False,
) -> EqualizerResult:
    """
    Recursive Least Squares adaptive equalizer with butterfly MIMO support.

    RLS converges faster than LMS at the cost of higher per-symbol
    complexity (maintains an inverse correlation matrix per output stream).

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
    training_symbols : array_like, optional
        Known symbols for data-aided adaptation (at symbol rate, 1 SPS).
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    forgetting_factor : float, default 0.99
        RLS forgetting factor (lambda). Range: (0, 1].
        Values close to 1 give longer memory.
    delta : float, default 0.01
        Regularization for initial inverse correlation matrix
        ``P = (1/delta) * I``.
    reference_constellation : array_like, optional
        Constellation points for DD slicing.
    sps : int, default 1
        Samples per symbol at the input (stride for output decimation).
    store_weights : bool, default False
        If True, stores weight trajectory.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory.
    """
    logger.info(
        f"RLS equalizer: num_taps={num_taps}, lambda={forgetting_factor}, sps={sps}"
    )
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for adaptive equalizers.")

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    n_sym = (n_samples - num_taps + 1) // stride
    if n_sym <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {num_taps} taps with sps={sps}."
        )

    constellation_np, train_full, n_train_aligned = _prepare_training(
        training_symbols, reference_constellation, num_ch, n_sym, num_taps, sps=sps
    )

    x_jax = to_jax(samples)
    if was_1d:
        x_jax = x_jax[None, :]

    try:
        if hasattr(x_jax, "device"):
            platform = x_jax.device.platform
        else:
            platform = list(x_jax.devices())[0].platform
    except Exception:
        platform = "cpu"

    train_jax = to_jax(train_full, device=platform)
    const_jax = to_jax(constellation_np, device=platform)
    w_init = _init_butterfly_weights(num_ch, num_taps, jnp)
    w_init = to_jax(w_init, device=platform)

    regressor_dim = num_ch * num_taps
    P_init = jnp.stack([jnp.eye(regressor_dim, dtype=jnp.complex64) / delta] * num_ch)
    P_init = to_jax(P_init, device=platform)
    lam_jax = to_jax(jnp.float32(forgetting_factor), device=platform)

    scan_fn = _get_jitted_rls(
        num_taps, stride, n_train_aligned, len(constellation_np), num_ch
    )
    y_jax, e_jax, W_jax, wh_jax = scan_fn(
        x_jax, train_jax, const_jax, w_init, P_init, lam_jax
    )

    return _unpack_result(y_jax, e_jax, W_jax, wh_jax, was_1d, store_weights)


def cma(
    samples: ArrayType,
    num_taps: int = 21,
    step_size: float = 0.01,
    r2: Optional[float] = None,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    sps: int = 2,
    normalize: bool = True,
    store_weights: bool = False,
) -> EqualizerResult:
    """
    Constant Modulus Algorithm blind equalizer with butterfly MIMO support.

    CMA minimizes the Godard dispersion criterion and requires no training
    symbols. It is the standard blind equalizer for constant-modulus signals
    (PSK) and near-constant-modulus signals (low-order QAM).

    CMA recovers the signal up to a phase ambiguity. A phase recovery step
    (e.g. Viterbi-Viterbi, pilot-aided) is typically needed after CMA.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    step_size : float, default 0.01
        CMA step size (mu). When ``normalize=True`` (default), this is
        the normalized step size.
    r2 : float, optional
        Godard radius ``R2 = E[|a|^4] / E[|a|^2]``. If None, computed
        from ``modulation``/``order`` or defaults to 1.0 (PSK).
    modulation : str, optional
        Modulation type for auto-computing R2 (e.g. ``"psk"``, ``"qam"``).
    order : int, optional
        Modulation order for auto-computing R2.
    sps : int, default 1
        Samples per symbol at the input.
    normalize : bool, default True
        If True, normalize step size by instantaneous input power.
    store_weights : bool, default False
        If True, stores weight trajectory.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, CMA error history, and optionally
        weight trajectory.
    """
    logger.info(
        f"CMA equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"normalize={normalize}"
    )
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for adaptive equalizers.")

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    # Compute R2
    if r2 is None:
        if modulation is not None and order is not None:
            from .mapping import gray_constellation

            const = gray_constellation(modulation, order)
            r2 = float(np.mean(np.abs(const) ** 4) / np.mean(np.abs(const) ** 2))
            logger.debug(f"CMA R2 from {modulation.upper()}-{order}: {r2:.4f}")
        else:
            r2 = 1.0

    n_sym = (n_samples - num_taps + 1) // stride
    if n_sym <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {num_taps} taps with sps={sps}."
        )

    x_jax = to_jax(samples)
    if was_1d:
        x_jax = x_jax[None, :]

    try:
        platform = x_jax.device.platform
    except Exception:
        platform = "cpu"

    w_init = _init_butterfly_weights(num_ch, num_taps, jnp)
    w_init = to_jax(w_init, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    r2_jax = to_jax(jnp.float32(r2), device=platform)

    # n_sym is static (part of cache key) — required for jnp.arange inside JIT
    scan_fn = _get_jitted_cma(num_taps, stride, num_ch, normalize, n_sym)
    y_jax, e_jax, W_jax, wh_jax = scan_fn(x_jax, w_init, mu_jax, r2_jax)

    return _unpack_result(y_jax, e_jax, W_jax, wh_jax, was_1d, store_weights)


# ============================================================================
# BLOCK EQUALIZERS
# ============================================================================


def zf_equalizer(
    samples: ArrayType,
    channel_estimate: ArrayType,
    noise_variance: float = 0.0,
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

    was_1d = samples.ndim == 1
    reg = noise_variance if noise_variance > 0 else 1e-12

    if was_1d:
        N = samples.shape[0]
        H = xp.fft.fft(channel_estimate, n=N)
        W = xp.conj(H) / (xp.abs(H) ** 2 + reg)
        return xp.fft.ifft(xp.fft.fft(samples) * W)

    num_ch, N = samples.shape

    if channel_estimate.ndim == 1:
        # SISO channel applied per-channel independently
        H = xp.fft.fft(channel_estimate, n=N)
        W = xp.conj(H) / (xp.abs(H) ** 2 + reg)
        Y = xp.fft.fft(samples, n=N, axis=-1)
        return xp.fft.ifft(Y * W[None, :], axis=-1)

    # Full MIMO: (C, C, L)
    H_f = xp.fft.fft(channel_estimate, n=N, axis=-1)  # (C, C, N)
    Y = xp.fft.fft(samples, n=N, axis=-1)  # (C, N)
    X_hat = xp.zeros_like(Y)

    eye = xp.eye(num_ch, dtype=samples.dtype)
    for k in range(N):
        Hk = H_f[:, :, k]
        HHh = Hk @ xp.conj(Hk).T
        Wk = xp.conj(Hk).T @ xp.linalg.inv(HHh + reg * eye)
        X_hat[:, k] = Wk @ Y[:, k]

    return xp.fft.ifft(X_hat, axis=-1)
