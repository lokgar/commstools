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

import functools
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
    num_train_symbols : int
        Number of data-aided training symbols actually consumed by the equalizer.
        This value is used to discard transients (the training portion) before
        computing steady-state metrics like EVM, SNR, and BER on the blind
        decision-directed portion of the output.
    """

    y_hat: ArrayType
    weights: ArrayType
    error: ArrayType
    weights_history: Optional[ArrayType] = None
    num_train_symbols: int = 0


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


def _get_jitted_lms(num_taps, stride, const_size, num_ch, normalize):
    """Returns JIT-compiled LMS/NLMS butterfly scan."""
    key = ("lms", num_taps, stride, const_size, num_ch, normalize)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def lms_scan(
            x_input, training_padded, constellation, w_init, step_size, n_train
        ):
            # x_input: (C, N_samples)    training_padded: (C, N_sym)
            # constellation: (M,)        w_init: (C, C, num_taps)
            # n_train: int32 scalar — dynamic (no retrace on change)

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


def _get_jitted_rls(num_taps, stride, const_size, num_ch):
    """Returns JIT-compiled RLS butterfly scan."""
    key = ("rls", num_taps, stride, const_size, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def rls_scan(
            x_input, training_padded, constellation, w_init, P_init, lam, n_train
        ):
            # x_input: (C, N_samples)    training_padded: (C, N_sym)
            # w_init: (C, C, num_taps)   P_init: (C, C*num_taps, C*num_taps)
            # n_train: int32 scalar — dynamic (no retrace on change)

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


def _get_jitted_cma(num_taps, stride, num_ch, normalize):
    """Returns JIT-compiled CMA butterfly scan."""
    key = ("cma", num_taps, stride, num_ch, normalize)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        # n_sym (arg 4) is static because lax.scan requires a compile-time
        # iteration count.  JAX's own bounded LRU cache handles retracing
        # when n_sym changes, preventing unbounded growth of _JITTED_EQ.
        @functools.partial(jax.jit, static_argnums=(4,))
        def cma_scan(x_input, w_init, step_size, r2, n_sym):
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
# BLOCK-ADAPTIVE KERNELS (GPU-optimized)
# ============================================================================
#
# Instead of lax.scan over individual symbols, these kernels scan over
# *blocks* of B symbols.  Within each block:
#   1. Parallel forward pass  (vmap) — apply frozen weights to B symbols
#   2. Parallel error compute (vmap) — B errors in one shot
#   3. Single weight update          — average gradient across the block
#
# When block_size=1 this is mathematically identical to the sample-by-sample
# kernel, just with higher per-step overhead.  For block_size >= 8 the
# GPU gets enough parallel work per step to amortise kernel-launch latency.


def _get_jitted_block_lms(num_taps, stride, const_size, num_ch, normalize, block_size):
    """Returns JIT-compiled block-LMS butterfly scan."""
    key = ("block_lms", num_taps, stride, const_size, num_ch, normalize, block_size)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        B = block_size

        @functools.partial(jax.jit, static_argnums=(6,))
        def block_lms_scan(
            x_input,
            training_padded,
            constellation,
            w_init,
            step_size,
            n_train,
            n_blocks,
            n_sym,
        ):
            # x_input: (C, N_samples)  training_padded: (C, n_sym_padded)
            # n_blocks: int32 — number of blocks to scan over

            def extract_window(ch, sample_idx):
                return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

            def forward_one(W, sym_idx):
                """Forward pass + error for a single symbol with frozen W."""
                sample_idx = sym_idx * stride
                X_wins = jax.vmap(extract_window, in_axes=(0, None))(
                    x_input, sample_idx
                )  # (C, num_taps)
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)  # (C,)
                return y, X_wins

            def slicer(ch_y):
                return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

            def block_step(W, block_idx):
                # Symbol indices for this block
                base = block_idx * B
                sym_indices = base + jnp.arange(B)  # (B,)

                # Parallel forward pass over B symbols
                y_block, X_block = jax.vmap(lambda idx: forward_one(W, idx))(
                    sym_indices
                )  # y_block: (B, C), X_block: (B, C, num_taps)

                # Desired symbols: training or DD
                d_train = jax.lax.dynamic_slice(
                    training_padded, (0, base), (num_ch, B)
                ).T  # (B, C)
                dd = jax.vmap(lambda y_i: jax.vmap(slicer)(y_i))(y_block)  # (B, C)
                # Per-symbol mask: use training if sym_idx < n_train
                mask = (sym_indices < n_train)[:, None]  # (B, 1)
                d_block = jnp.where(mask, d_train, dd)  # (B, C)

                e_block = d_block - y_block  # (B, C)

                # Mask out ghost errors from out-of-bounds padding
                valid_mask = (sym_indices < n_sym)[:, None]
                e_block = jnp.where(valid_mask, e_block, 0.0)

                # Average gradient across the block
                grads = jax.vmap(
                    lambda e_i, X_i: jnp.einsum("i,jt->ijt", e_i, jnp.conj(X_i))
                )(e_block, X_block)  # (B, C, C, num_taps)
                avg_grad = jnp.mean(grads, axis=0)  # (C, C, num_taps)

                if normalize:
                    # Average power across block
                    powers = jax.vmap(
                        lambda X_i: jnp.real(jnp.sum(X_i * jnp.conj(X_i)))
                    )(X_block)  # (B,)
                    avg_power = jnp.mean(powers) + 1e-10
                    mu_eff = step_size / avg_power
                else:
                    mu_eff = step_size

                W_new = W + mu_eff * avg_grad
                return W_new, (y_block, e_block, W_new)

            W_final, (y_all, e_all, w_hist) = jax.lax.scan(
                block_step, w_init, jnp.arange(n_blocks)
            )
            # y_all: (n_blocks, B, C) → reshape to (n_blocks*B, C)
            y_flat = y_all.reshape(-1, num_ch)
            e_flat = e_all.reshape(-1, num_ch)
            # w_hist: (n_blocks, C, C, num_taps) — one per block
            return y_flat, e_flat, W_final, w_hist

        _JITTED_EQ[key] = block_lms_scan
    return _JITTED_EQ[key]


def _get_jitted_block_cma(num_taps, stride, num_ch, normalize, block_size):
    """Returns JIT-compiled block-CMA butterfly scan."""
    key = ("block_cma", num_taps, stride, num_ch, normalize, block_size)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        B = block_size

        @functools.partial(jax.jit, static_argnums=(4,))
        def block_cma_scan(x_input, w_init, step_size, r2, n_blocks, n_sym):

            def extract_window(ch, sample_idx):
                return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

            def forward_one(W, sym_idx):
                sample_idx = sym_idx * stride
                X_wins = jax.vmap(extract_window, in_axes=(0, None))(
                    x_input, sample_idx
                )
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)
                return y, X_wins

            def block_step(W, block_idx):
                base = block_idx * B
                sym_indices = base + jnp.arange(B)

                y_block, X_block = jax.vmap(lambda idx: forward_one(W, idx))(
                    sym_indices
                )  # (B, C), (B, C, num_taps)

                # CMA error: e = y * (|y|^2 - R2)
                e_block = y_block * (jnp.abs(y_block) ** 2 - r2)  # (B, C)

                # Mask out ghost errors from out-of-bounds padding
                valid_mask = (sym_indices < n_sym)[:, None]
                e_block = jnp.where(valid_mask, e_block, 0.0)

                grads = jax.vmap(
                    lambda e_i, X_i: jnp.einsum("i,jt->ijt", e_i, jnp.conj(X_i))
                )(e_block, X_block)
                avg_grad = jnp.mean(grads, axis=0)

                if normalize:
                    powers = jax.vmap(
                        lambda X_i: jnp.real(jnp.sum(X_i * jnp.conj(X_i)))
                    )(X_block)
                    avg_power = jnp.mean(powers) + 1e-10
                    mu_eff = step_size / avg_power
                else:
                    mu_eff = step_size

                W_new = W - mu_eff * avg_grad  # CMA uses gradient descent (minus)
                return W_new, (y_block, e_block, W_new)

            W_final, (y_all, e_all, w_hist) = jax.lax.scan(
                block_step, w_init, jnp.arange(n_blocks)
            )
            y_flat = y_all.reshape(-1, num_ch)
            e_flat = e_all.reshape(-1, num_ch)
            return y_flat, e_flat, W_final, w_hist

        _JITTED_EQ[key] = block_cma_scan
    return _JITTED_EQ[key]


def _get_jitted_block_rls(num_taps, stride, const_size, num_ch, block_size):
    """Returns JIT-compiled block-RLS butterfly scan (hybrid).

    Phase 1 (parallel): Apply frozen weights to B symbols → Y_block, E_block
    Phase 2 (sequential): Update (W, P) via inner lax.scan over B symbols
    """
    key = ("block_rls", num_taps, stride, const_size, num_ch, block_size)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        B = block_size

        @functools.partial(jax.jit, static_argnums=(7,))
        def block_rls_scan(
            x_input,
            training_padded,
            constellation,
            w_init,
            P_init,
            lam,
            n_train,
            n_blocks,
            n_sym,
        ):
            def extract_window(ch, sample_idx):
                return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

            def forward_one(W, sym_idx):
                sample_idx = sym_idx * stride
                X_wins = jax.vmap(extract_window, in_axes=(0, None))(
                    x_input, sample_idx
                )
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)
                return y, X_wins

            def slicer(ch_y):
                return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

            def block_step(carry, block_idx):
                W, P = carry
                base = block_idx * B
                sym_indices = base + jnp.arange(B)

                # Phase 1: parallel forward pass with frozen weights
                y_block, X_block = jax.vmap(lambda idx: forward_one(W, idx))(
                    sym_indices
                )  # (B, C), (B, C, num_taps)

                # Desired symbols
                d_train = jax.lax.dynamic_slice(
                    training_padded, (0, base), (num_ch, B)
                ).T
                dd = jax.vmap(lambda y_i: jax.vmap(slicer)(y_i))(y_block)
                mask = (sym_indices < n_train)[:, None]
                d_block = jnp.where(mask, d_train, dd)
                e_block = d_block - y_block  # (B, C)

                # Mask out ghost errors from out-of-bounds padding
                valid_mask = (sym_indices < n_sym)[:, None]
                e_block = jnp.where(valid_mask, e_block, 0.0)

                # Phase 2: sequential RLS weight update over B symbols
                def rls_inner_step(carry_inner, b_idx):
                    W_i, P_i = carry_inner
                    X_i = X_block[b_idx]  # (C, num_taps)
                    e_i = e_block[b_idx]  # (C,)
                    x_bar = X_i.flatten()  # (C * num_taps,)

                    def rls_update(args):
                        w_row, P_ch, e_ch = args
                        w_flat = w_row.flatten()
                        Px = P_ch @ x_bar
                        denom = lam + jnp.dot(jnp.conj(x_bar), Px)
                        k = Px / denom
                        w_flat_new = w_flat + k * jnp.conj(e_ch)
                        P_new = (P_ch - jnp.outer(k, jnp.conj(x_bar) @ P_ch)) / lam
                        return w_flat_new.reshape(num_ch, num_taps), P_new

                    W_new, P_new = jax.vmap(rls_update)((W_i, P_i, e_i))
                    return (W_new, P_new), None

                (W_new, P_new), _ = jax.lax.scan(rls_inner_step, (W, P), jnp.arange(B))

                return (W_new, P_new), (y_block, e_block, W_new)

            (W_final, _), (y_all, e_all, w_hist) = jax.lax.scan(
                block_step, (w_init, P_init), jnp.arange(n_blocks)
            )
            y_flat = y_all.reshape(-1, num_ch)
            e_flat = e_all.reshape(-1, num_ch)
            return y_flat, e_flat, W_final, w_hist

        _JITTED_EQ[key] = block_rls_scan
    return _JITTED_EQ[key]


# ============================================================================
# SHARED HELPERS
# ============================================================================


def _get_center_tap(num_taps, sps):
    """Center tap index aligned with symbol-bearing sample positions.

    For ``sps=2`` (T/2-spaced), ensures the center tap falls on an
    even-indexed sample position within the filter window, which
    corresponds to a symbol instant in the input signal. This prevents
    initialization on inter-symbol samples that may have low energy
    (e.g. near zero crossings after matched filtering or with
    zero-stuffed signals).
    """
    center = num_taps // 2
    if sps == 2 and center % 2 != 0:
        center -= 1
    return center


def _normalize_inputs(samples, training_symbols, sps, xp):
    """Normalize samples so expected symbol-rate power matches training power.

    For fractionally-spaced equalizers operating on un-synchronized data,
    the fractional timing phase is unknown. Measuring power using a stride
    (e.g., ``samples[..., ::sps]``) is dangerous because it could sample
    the zero-crossings, severely underestimating the true signal power and
    causing instability.

    Instead, we measure the wideband signal power (average power across all
    samples). For a unit-energy Nyquist pulse, the total discrete power is
    ``E[|symbol|^2] / sps``. Therefore, we robustly estimate the peak symbol
    RMS by multiplying the total global RMS by ``sqrt(sps)``.
    """
    from commstools.helpers import normalize as c_normalize, rms

    # Robust symbol-power estimate invariant to arbitrary fractional delays
    global_rms = rms(samples, axis=-1, keepdims=True)
    sym_rms = global_rms * xp.sqrt(xp.asarray(sps, dtype=global_rms.dtype))

    # Avoid div by 0 just in case
    sym_rms = xp.where(sym_rms == 0, 1.0, sym_rms)
    samples = samples / sym_rms

    if training_symbols is not None:
        training_symbols = c_normalize(training_symbols, "average_power", axis=-1)

    return samples, training_symbols


def _pad_to_block(n_sym, block_size):
    """Compute number of blocks and padded symbol count.

    Returns ``(n_blocks, n_sym_padded)`` where
    ``n_sym_padded = n_blocks * block_size >= n_sym``.
    """
    n_blocks = (n_sym + block_size - 1) // block_size
    return n_blocks, n_blocks * block_size


def _init_butterfly_weights(num_ch, num_taps, jnp, sps=2):
    """Center-tap identity initialization for butterfly weight matrix.

    Returns ``(C, C, num_taps)`` with diagonal center taps = 1.
    The center tap is aligned with a symbol-bearing sample position
    via :func:`_get_center_tap`.
    """
    W = jnp.zeros((num_ch, num_ch, num_taps), dtype="complex64")
    center = _get_center_tap(num_taps, sps)
    W = W.at[jnp.arange(num_ch), jnp.arange(num_ch), center].set(1.0 + 0j)
    return W


def _prepare_training(
    training_symbols,
    num_ch,
    n_sym,
    num_taps,
    sps=2,
    num_train_symbols=None,
):
    """Build center-tap-aligned training array.

    Training symbols are delayed to match the equalizer's center-tap delay.
    Delay in symbols = ``center_tap // sps``.

    Arrays are kept on their original device (NumPy or CuPy) to avoid
    unnecessary CPU round-trips.

    Returns
    -------
    train_full : ndarray (C, n_sym) complex64
        On the same backend as ``training_symbols`` (or NumPy if None).
    n_train_aligned : int
    """
    center = _get_center_tap(num_taps, sps)
    delay = center // sps

    if training_symbols is not None:
        # Keep training data on its original device
        train_arr, xp, _ = dispatch(training_symbols)
        train_arr = train_arr.astype("complex64")
        if train_arr.ndim == 1:
            train_arr = (
                xp.tile(train_arr[None, :], (num_ch, 1))
                if num_ch > 1
                else train_arr[None, :]
            )
        n_raw = train_arr.shape[1]
        n_train_aligned = max(0, min(n_raw - delay, n_sym))
        if num_train_symbols is not None:
            n_train_aligned = min(n_train_aligned, num_train_symbols)

        train_full = xp.zeros((num_ch, n_sym), dtype="complex64")
        if n_train_aligned > 0:
            train_full[:, :n_train_aligned] = train_arr[
                :, delay : delay + n_train_aligned
            ]
    else:
        n_train_aligned = 0
        train_full = np.zeros((num_ch, n_sym), dtype="complex64")

    return train_full, n_train_aligned


def _unpack_result(
    y_hat_jax,
    errors_jax,
    W_final_jax,
    w_hist_jax,
    was_1d,
    store_weights,
    n_sym=None,
    xp=np,
    num_train_symbols=0,
):
    """Convert JAX scan outputs to backend arrays with proper shape.

    Parameters
    ----------
    n_sym : int, optional
        Truncate output to this many symbols (used by block mode to
        remove zero-padding). If None, no truncation.
    """
    y_hat = xp.asarray(from_jax(y_hat_jax).T)  # (N_sym, C) -> (C, N_sym)
    errors = xp.asarray(from_jax(errors_jax).T)
    W_final = xp.asarray(from_jax(W_final_jax))

    if n_sym is not None:
        y_hat = y_hat[..., :n_sym]
        errors = errors[..., :n_sym]

    if was_1d:
        y_hat = y_hat[0]
        errors = errors[0]
        W_final = W_final[0, 0]

    w_history = None
    if store_weights:
        w_history = xp.asarray(from_jax(w_hist_jax))
        if was_1d:
            w_history = w_history[:, 0, 0, :]

    return EqualizerResult(
        y_hat=y_hat,
        weights=W_final,
        error=errors,
        weights_history=w_history,
        num_train_symbols=num_train_symbols,
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
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    sps: int = 2,
    normalize: bool = True,
    store_weights: bool = False,
    block_size: int = 1,
    num_train_symbols: Optional[int] = None,
    device: Optional[str] = None,
) -> EqualizerResult:
    """
    Least Mean Squares adaptive equalizer with butterfly MIMO support.

    Supports data-aided (training) and decision-directed (DD) modes.
    When ``training_symbols`` are provided, the equalizer uses them for the
    initial convergence phase, then switches to DD mode using the
    ``modulation`` and ``order`` parameters, or a constellation auto-inferred
    from the training symbols for hard-decision slicing.

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
    modulation : str, optional
        Modulation scheme (e.g., 'psk', 'qam', 'pam') for DD slicing.
        Required if ``training_symbols`` is None.
    order : int, optional
        Modulation order (e.g., 4, 16).
    unipolar : bool, default False
        If True, indicates the modulation is unipolar (e.g., unipolar PAM).
    sps : int, default 2
        Samples per symbol at the input. Must be 2 (T/2-spaced).
        The equalizer natively filters intermediate transition points by
        decimating to symbol rate (shifting the mathematical window by
        exactly ``sps`` samples) to compute robust DD slicing errors.
    normalize : bool, default True
        If True, use Normalized LMS (NLMS): step size is divided by
        the instantaneous input power ``||x||^2``. This makes convergence
        robust to input power variations and is strongly recommended.
    store_weights : bool, default False
        If True, stores weight trajectory in ``weights_history``.
    block_size : int, default 1
        Number of symbols processed in parallel per weight update.
        ``block_size=1`` is the classical sample-by-sample algorithm.
        Larger values (8–64) improve GPU throughput by reducing
        ``lax.scan`` iterations at the cost of slower adaptation.
    num_train_symbols : int, optional
        Limits the number of training symbols used. If provided, the
        equalizer will forcefully switch to blind Decision-Directed (DD)
        mode after this many symbols, even if more training symbols
        are available in the array.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory. Arrays reside on the same backend as input.
    """
    logger.info(
        f"LMS equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"normalize={normalize}, block_size={block_size}"
    )
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for adaptive equalizers.")

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    # RMS-normalize samples; scale training to match symbol-rate power
    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)
    samples, training_symbols = _normalize_inputs(samples, training_symbols, sps, xp)

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

    # For block mode, pad n_sym to a multiple of block_size
    if block_size > 1:
        n_blocks, n_sym_padded = _pad_to_block(n_sym, block_size)
    else:
        n_sym_padded = n_sym

    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        reference_constellation = gray_constellation(
            modulation, order, unipolar=unipolar
        )
    elif training_symbols is not None:
        # Infer constellation from unique training values, handling on the current backend
        _, _xp, _ = dispatch(training_symbols)
        train_flat = _xp.reshape(training_symbols, (-1,))
        reference_constellation = _xp.unique(_xp.round(train_flat, decimals=8))
    else:
        raise ValueError("modulation and order must be provided for DD mode.")

    constellation_np = (
        np.asarray(to_device(reference_constellation, "cpu"))
        .flatten()
        .astype("complex64")
    )

    train_full, n_train_aligned = _prepare_training(
        training_symbols,
        num_ch,
        n_sym_padded,
        num_taps,
        sps=sps,
        num_train_symbols=num_train_symbols,
    )

    # Convert to JAX — preserves device or overrides if `device` is given
    x_jax = to_jax(samples, device=device)
    if was_1d:
        x_jax = x_jax[None, :]

    # Ensure all JAX arrays are on the target device
    try:
        platform = (
            device.lower()
            if device is not None
            else (
                x_jax.device.platform
                if hasattr(x_jax, "device")
                else list(x_jax.devices())[0].platform
            )
        )
    except Exception:
        platform = "cpu"

    train_jax = to_jax(train_full, device=platform)
    const_jax = to_jax(constellation_np, device=platform)
    w_init = _init_butterfly_weights(num_ch, num_taps, jnp, sps=sps)
    w_init = to_jax(w_init, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)
    n_sym_jax = to_jax(jnp.int32(n_sym), device=platform)

    if block_size > 1:
        scan_fn = _get_jitted_block_lms(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            normalize,
            block_size,
        )
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            w_init,
            mu_jax,
            n_train_jax,
            n_blocks,  # static arg — must be Python int
            n_sym_jax,
        )
    else:
        scan_fn = _get_jitted_lms(
            num_taps, stride, len(constellation_np), num_ch, normalize
        )
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, train_jax, const_jax, w_init, mu_jax, n_train_jax
        )

    return _unpack_result(
        y_jax,
        e_jax,
        W_jax,
        wh_jax,
        was_1d,
        store_weights,
        n_sym=n_sym if block_size > 1 else None,
        xp=xp,
        num_train_symbols=int(n_train_aligned),
    )


def rls(
    samples: ArrayType,
    training_symbols: Optional[ArrayType] = None,
    num_taps: int = 21,
    forgetting_factor: float = 0.99,
    delta: float = 0.01,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    sps: int = 2,
    store_weights: bool = False,
    block_size: int = 1,
    num_train_symbols: Optional[int] = None,
    device: Optional[str] = None,
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
    modulation : str, optional
        Modulation scheme (e.g., 'psk', 'qam', 'pam') for DD slicing.
        Required if ``training_symbols`` is None.
    order : int, optional
        Modulation order (e.g., 4, 16).
    unipolar : bool, default False
        If True, indicates the modulation is unipolar (e.g., unipolar PAM).
    sps : int, default 2
        Samples per symbol at the input. Must be 2 (T/2-spaced).
        The equalizer natively filters intermediate transition points by
        decimating to symbol rate (shifting the mathematical window by
        exactly ``sps`` samples) to compute robust DD slicing errors.
    store_weights : bool, default False
        If True, stores weight trajectory.
    block_size : int, default 1
        Number of symbols per weight update block. ``block_size=1`` is
        standard sample-by-sample RLS. Larger values use a hybrid
        approach: parallel forward pass + sequential P/W update within
        each block, reducing outer-scan overhead.
    num_train_symbols : int, optional
        Limits the number of training symbols used. If provided, the
        equalizer will forcefully switch to blind Decision-Directed (DD)
        mode after this many symbols, even if more training symbols
        are available in the array.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory.
    """
    logger.info(
        f"RLS equalizer: num_taps={num_taps}, lambda={forgetting_factor}, "
        f"sps={sps}, block_size={block_size}"
    )
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for adaptive equalizers.")

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    # RMS-normalize samples; scale training to match symbol-rate power
    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)
    samples, training_symbols = _normalize_inputs(samples, training_symbols, sps, xp)

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

    # For block mode, pad n_sym to a multiple of block_size
    if block_size > 1:
        n_blocks, n_sym_padded = _pad_to_block(n_sym, block_size)
    else:
        n_sym_padded = n_sym

    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        reference_constellation = gray_constellation(
            modulation, order, unipolar=unipolar
        )
    elif training_symbols is not None:
        # Infer constellation from unique training values, handling on the current backend
        _, _xp, _ = dispatch(training_symbols)
        train_flat = _xp.reshape(training_symbols, (-1,))
        reference_constellation = _xp.unique(_xp.round(train_flat, decimals=8))
    else:
        raise ValueError("modulation and order must be provided for DD mode.")

    constellation_np = (
        np.asarray(to_device(reference_constellation, "cpu"))
        .flatten()
        .astype("complex64")
    )

    train_full, n_train_aligned = _prepare_training(
        training_symbols,
        num_ch,
        n_sym_padded,
        num_taps,
        sps=sps,
        num_train_symbols=num_train_symbols,
    )

    x_jax = to_jax(samples, device=device)
    if was_1d:
        x_jax = x_jax[None, :]

    try:
        platform = (
            device.lower()
            if device is not None
            else (
                x_jax.device.platform
                if hasattr(x_jax, "device")
                else list(x_jax.devices())[0].platform
            )
        )
    except Exception:
        platform = "cpu"

    train_jax = to_jax(train_full, device=platform)
    const_jax = to_jax(constellation_np, device=platform)
    w_init = _init_butterfly_weights(num_ch, num_taps, jnp, sps=sps)
    w_init = to_jax(w_init, device=platform)

    regressor_dim = num_ch * num_taps
    P_init = jnp.stack([jnp.eye(regressor_dim, dtype="complex64") / delta] * num_ch)
    P_init = to_jax(P_init, device=platform)
    lam_jax = to_jax(jnp.float32(forgetting_factor), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)
    n_sym_jax = to_jax(jnp.int32(n_sym), device=platform)

    if block_size > 1:
        scan_fn = _get_jitted_block_rls(
            num_taps, stride, len(constellation_np), num_ch, block_size
        )
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            w_init,
            P_init,
            lam_jax,
            n_train_jax,
            n_blocks,  # static arg — must be Python int
            n_sym_jax,
        )
    else:
        scan_fn = _get_jitted_rls(num_taps, stride, len(constellation_np), num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, train_jax, const_jax, w_init, P_init, lam_jax, n_train_jax
        )

    return _unpack_result(
        y_jax,
        e_jax,
        W_jax,
        wh_jax,
        was_1d,
        store_weights,
        n_sym=n_sym if block_size > 1 else None,
        xp=xp,
        num_train_symbols=int(n_train_aligned),
    )


def cma(
    samples: ArrayType,
    num_taps: int = 21,
    step_size: float = 0.01,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    sps: int = 2,
    normalize: bool = True,
    store_weights: bool = False,
    block_size: int = 1,
    device: Optional[str] = None,
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
    modulation : str, optional
        Modulation type for auto-computing Godard radius R2 (e.g. ``"psk"``, ``"qam"``).
        If None, defaults to R2=1.0.
    order : int, optional
        Modulation order for auto-computing R2.
    unipolar : bool, default False
        Use unipolar constellation for auto-computing R2.
    sps : int, default 2
        Samples per symbol at the input. Must be 2 (T/2-spaced).
        The equalizer natively filters intermediate transition points by
        decimating to symbol rate (shifting the mathematical window by
        exactly ``sps`` samples) to compute robust blind errors.
    normalize : bool, default True
        If True, normalize step size by instantaneous input power.
    store_weights : bool, default False
        If True, stores weight trajectory.
    block_size : int, default 1
        Number of symbols processed in parallel per weight update.
        ``block_size=1`` is the classical sample-by-sample algorithm.
        Larger values (8–64) improve GPU throughput. CMA is more
        sensitive to block size than LMS due to its non-convex cost
        surface — recommend ``block_size <= 32``.
    device : str, optional
        Target device for equalizer execution (e.g. ``"cpu"`` or ``"gpu"``).
        If provided, the inputs are moved to this device for processing
        and then securely returned to their original device format.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, CMA error history, and optionally
        weight trajectory.
    """
    logger.info(
        f"CMA equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"normalize={normalize}, block_size={block_size}"
    )
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for adaptive equalizers.")

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    # RMS-normalize samples to unit symbol-rate power (CMA has no training)
    samples, _ = _normalize_inputs(samples, None, sps, xp)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    # Compute R2 from the (now unit-power) constellation
    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        const = gray_constellation(modulation, order, unipolar=unipolar)
        r2 = float(np.mean(np.abs(const) ** 4) / np.mean(np.abs(const) ** 2))
        logger.debug(f"CMA R2 from {modulation.upper()}-{order}: {r2:.4f}")
    else:
        r2 = 1.0

    n_sym = (n_samples - num_taps + 1) // stride
    if n_sym <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {num_taps} taps with sps={sps}."
        )

    x_jax = to_jax(samples, device=device)
    if was_1d:
        x_jax = x_jax[None, :]

    try:
        platform = (
            device.lower()
            if device is not None
            else (
                x_jax.device.platform
                if hasattr(x_jax, "device")
                else list(x_jax.devices())[0].platform
            )
        )
    except Exception:
        platform = "cpu"

    w_init = _init_butterfly_weights(num_ch, num_taps, jnp, sps=sps)
    w_init = to_jax(w_init, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    r2_jax = to_jax(jnp.float32(r2), device=platform)
    n_sym_jax = to_jax(jnp.int32(n_sym), device=platform)

    if block_size > 1:
        n_blocks, n_sym_padded = _pad_to_block(n_sym, block_size)
        scan_fn = _get_jitted_block_cma(num_taps, stride, num_ch, normalize, block_size)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax,
            w_init,
            mu_jax,
            r2_jax,
            n_blocks,  # static arg
            n_sym_jax,
        )
    else:
        scan_fn = _get_jitted_cma(num_taps, stride, num_ch, normalize)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(x_jax, w_init, mu_jax, r2_jax, n_sym)

    return _unpack_result(
        y_jax,
        e_jax,
        W_jax,
        wh_jax,
        was_1d,
        store_weights,
        n_sym=n_sym if block_size > 1 else None,
        xp=xp,
    )


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

    # Vectorized loop-free MIMO inversion across frequencies
    Hk = xp.transpose(H_f, (2, 0, 1))  # (N, C, C)
    Hk_H = xp.conj(xp.transpose(Hk, (0, 2, 1)))  # (N, C, C)
    HHh = Hk @ Hk_H

    eye = xp.eye(num_ch, dtype=samples.dtype)[None, :, :]
    inv_term = xp.linalg.inv(HHh + reg * eye)
    Wk = Hk_H @ inv_term  # (N, C, C)

    Y_t = xp.transpose(Y)[:, :, None]  # (N, C, 1)
    X_hat_t = Wk @ Y_t  # (N, C, 1)
    X_hat = xp.transpose(X_hat_t[..., 0])  # (C, N)

    return xp.fft.ifft(X_hat, axis=-1)
