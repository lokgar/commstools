"""Pilot-tone-based polarization demultiplexing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..filtering import fir_filter, lowpass_taps
from ..frequency import find_bias_tone
from ..logger import logger

# -----------------------------------------------------------------------------
# TONE-BASED POLARIZATION DEMULTIPLEXING
# -----------------------------------------------------------------------------


def demultiplex_polarization_tones_static(
    samples: ArrayType,
    sampling_rate: float,
    tone_frequencies: Sequence[float],
    *,
    refine_tones: bool = True,
    search_band: float | None = None,
    normalize: bool = True,
    return_matrix: bool = False,
) -> ArrayType | tuple[ArrayType, ArrayType]:
    r"""
    One-shot polarization demux from distinct per-stream CW pilot tones.

    Undoes a **frequency-flat** polarization / spatial mixing by inverting the
    channel's Jones matrix, which is read directly off pilot tones placed at a
    *distinct* frequency on each transmitted stream (see ``add_pilot_tone`` with
    a per-channel ``frequency`` list).

    **Principle.**  With tone ``f_j`` carried *only* by transmitted stream ``j``
    and a frequency-flat mixing ``r = J s`` (e.g.
    ``apply_polarization_mixing``), the complex amplitude of tone ``f_j`` in
    received channel ``i`` is ``T[i, j] = J[i, j] · α_j`` (``α_j`` = the TX tone
    amplitude of stream ``j``).  Hence the measured tone-phasor matrix factors as
    ``T = J · diag(α)``, and ``W = pinv(T)`` unmixes::

        W r = diag(1/α) · J^{-1} J s = diag(1/α) · s,

    recovering each stream up to a trivial per-stream complex scale ``1/α_j``
    (removed by ``normalize`` and/or downstream CPR).  Because each tone uniquely
    labels its stream, output row ``j`` always corresponds to ``tone_frequencies[j]``
    — there is **no polarization-permutation ambiguity** (unlike blind CMA; cf.
    ``resolve_polarization_permutation``).

    **Speed.**  Tones added with ``add_pilot_tone`` are grid-quantized
    (buffer-periodic), so a single DFT bin is the exact, mutually-orthogonal,
    maximum-likelihood estimator of each tone phasor.  The whole operation is two
    small GEMMs (extract ``T`` via a ``(C,N)·(N,K)`` product, apply ``W`` via a
    ``(K,C)·(C,N)`` product) plus one ``KxK`` inverse — no iteration, no
    convergence, far below the cost of a full FFT.

    **Scope.**  Frequency-flat **and time-invariant** SOP only — the whole-record
    DFT bin estimates a *single* Jones matrix.  If the state of polarization
    drifts appreciably over the capture (long records and/or low baud rate, so
    the wall-clock duration exceeds the SOP coherence time), the averaged tone
    phasor is biased and attenuated and the one-shot inverse leaves residual,
    time-growing crosstalk; use ``demultiplex_polarization_tones_dynamic``
    instead.  For PMD / DGD use the returned unmixer (``return_matrix=True``) to
    seed a butterfly equalizer (``cma`` / ``block_lms``).

    Parameters
    ----------
    samples : array_like
        Received MIMO samples. Shape ``(C, N)`` — time on the last axis.
    sampling_rate : float
        Sampling rate f_s in Hz.
    tone_frequencies : sequence of float
        The ``K`` distinct per-stream tone frequencies in Hz (as added at the
        TX, in transmitted-stream order).  Require ``K <= C``.  Output row ``j``
        corresponds to ``tone_frequencies[j]``.
    refine_tones : bool, default True
        If ``True``, sub-bin-refine each tone centre with ``find_bias_tone``
        (on the receive channel where that tone is strongest) before extraction,
        absorbing a residual carrier frequency offset that has dragged the tone
        off its nominal bin.  If ``False``, extract exactly at
        ``tone_frequencies``.
    search_band : float, optional
        Half-width in Hz of the per-tone peak search when ``refine_tones=True``.
        Defaults to ``4 * f_s / N`` (a few FFT bins).  Widen it if the carrier
        offset can exceed that, but keep it inside the tone-to-data guard so the
        data band never wins the argmax.
    normalize : bool, default True
        If ``True``, rescale each demuxed row so its mean power equals the mean
        per-channel input power, removing the arbitrary ``1/α_j`` per-stream
        scale and preserving the library power invariant.
    return_matrix : bool, default False
        If ``True``, also return the ``(K, C)`` unmixing matrix ``W``.

    Returns
    -------
    demuxed : array_like
        Demultiplexed streams. Shape ``(K, N)`` — one row per recovered
        *stream* (``K`` = number of tones), **not** per receive channel: the
        ``C`` input channels are mapped down to the ``K`` transmitted streams.
        In the usual square dual-pol case ``K == C == 2`` the two coincide.
        Same complex dtype and backend as the input.  Row ``j`` is the stream
        that carried ``tone_frequencies[j]``.
    W : array_like, optional
        Returned only if ``return_matrix=True``: the ``(K, C)`` unmixing matrix
        (``complex128``), i.e. the estimated inverse Jones matrix up to per-stream
        scaling.  Suitable as a seed for a butterfly equalizer.

    Raises
    ------
    ValueError
        If ``samples`` is not 2-D, ``tone_frequencies`` is empty, ``K > C``, or
        any tone frequency lies outside ``(-f_s/2, f_s/2)``.

    See Also
    --------
    add_pilot_tone : Add the per-stream tones at the transmitter.
    demultiplex_polarization_tones_dynamic : Time-varying (drifting-SOP) demux.
    """
    samples, xp, _ = dispatch(samples)
    if samples.ndim != 2:
        raise ValueError(
            "demultiplex_polarization_tones_static requires a 2-D (C, N) MIMO "
            f"input; got ndim={samples.ndim}."
        )
    C, N = samples.shape

    f_tones = [float(f) for f in tone_frequencies]
    K = len(f_tones)
    if K == 0:
        raise ValueError("tone_frequencies must contain at least one frequency.")
    if K > C:
        raise ValueError(
            f"got K={K} tones but only C={C} receive channels; need K <= C to "
            "unmix (one tone per transmitted stream)."
        )
    nyq = sampling_rate / 2.0
    for f in f_tones:
        if not (-nyq < f < nyq):
            raise ValueError(
                f"tone_frequencies entry {f} must lie in (-fs/2, fs/2) = "
                f"(±{nyq:.3g}) Hz."
            )

    # Promote to complex128: the tone-phasor matrix inverse is precision-sensitive
    # (CLAUDE.md), and extraction is a single GEMM regardless of width.
    samples_c = xp.asarray(samples, dtype=xp.complex128)  # (C, N)
    n = xp.arange(N, dtype=xp.float64)
    two_pi = 2.0 * np.pi

    def _extract(freqs: Sequence[float]) -> ArrayType:
        # Tone-phasor matrix T[i, j] = (1/N) Σ_n samples[i, n] exp(-j2π f_j n/fs).
        # Conj-exponential basis B (K, N); phase wrapped in float64 before exp.
        fc = xp.asarray(freqs, dtype=xp.float64).reshape(K, 1)  # (K, 1)
        ph = two_pi * fc * n[None, :] / sampling_rate  # (K, N)
        ph = ph - xp.round(ph / two_pi) * two_pi
        basis = xp.exp(-1j * ph)  # (K, N) complex128
        return (samples_c @ basis.T) / N  # (C, K)

    df = sampling_rate / N
    T = _extract(f_tones)  # (C, K)

    if refine_tones:
        if search_band is None:
            search_band = 4.0 * df
        # Pick, per tone, the channel where it is strongest (one host transfer),
        # then sub-bin refine the centre there and re-extract.
        best_ch = to_device(xp.argmax(xp.abs(T), axis=0), "cpu")  # (K,) host ints
        f_used = [
            find_bias_tone(
                samples_c[int(best_ch[j])],
                sampling_rate,
                target_frequency=f_tones[j],
                search_band=search_band,
            )
            for j in range(K)
        ]
        T = _extract(f_used)
    else:
        f_used = f_tones

    # Unmix.  pinv covers the over-determined C > K case and equals the inverse
    # when square; kept in complex128 (inversion is precision-sensitive).
    W = xp.linalg.pinv(T)  # (K, C)
    demuxed = W @ samples_c  # (K, N) complex128

    if normalize:
        p_in = xp.mean(xp.abs(samples) ** 2)  # scalar: mean per-channel input power
        p_out = xp.mean(xp.abs(demuxed) ** 2, axis=-1, keepdims=True)  # (K, 1)
        scale = xp.sqrt(p_in / xp.where(p_out > 0, p_out, 1.0))
        demuxed = demuxed * scale

    demuxed = demuxed.astype(samples.dtype)

    logger.info(
        "demultiplex_polarization_tones: f_tones=%s Hz, refine=%s [C=%d, K=%d, N=%d]",
        [f"{f:.3g}" for f in f_used],
        refine_tones,
        C,
        K,
        N,
    )

    if return_matrix:
        return demuxed, W
    return demuxed


def demultiplex_polarization_tones_dynamic(
    samples: ArrayType,
    sampling_rate: float,
    tone_frequencies: Sequence[float],
    *,
    track_bandwidth: float,
    num_taps: int | None = None,
    grid_step: int | None = None,
    refine_tones: bool = True,
    search_band: float | None = None,
    normalize: bool = True,
    trim_edges: bool = False,
    return_matrix: bool = False,
    apply: bool = True,
) -> ArrayType | tuple[Any, ...]:
    r"""
    Time-varying polarization demux from distinct per-stream CW pilot tones.

    Drifting-SOP counterpart of ``demultiplex_polarization_tones_static``.  Where
    the static routine reads a *single* Jones matrix from a whole-record DFT bin,
    this one **tracks** a slowly rotating frequency-flat mixing ``r(n) = J(n) s(n)``
    by following each pilot tone continuously in time.

    **Principle.**  Tone ``f_j`` is a CW carried *only* by transmitted stream
    ``j`` (amplitude ``α_j``).  Mixing receive channel ``i`` down by ``f_j`` and
    low-pass filtering isolates that tone's slowly-varying phasor::

        z_ij(n) = LPF{ r_i(n) · exp(-j2π f_j n / f_s) } ≈ J_ij(n) · α_j,

    because every other tone ``f_k`` (k ≠ j) lands at ``f_k - f_j`` and the data
    band is pushed away from DC, so the LPF rejects them.  Running this for all
    ``K`` tones (one mix-down per *distinct* tone frequency) yields a continuous
    estimate of the whole Jones matrix ``T(n) = J(n) diag(α)``, shape ``(C, K, N)``.
    Inverting ``T`` on a decimated time grid and interpolating back to full rate
    gives a per-sample unmixer ``W(n) = pinv(T(n))`` with::

        W(n) r(n) = diag(1/α) · s(n),

    recovering each stream up to the same trivial per-stream scale ``1/α_j`` as
    the static routine.  As with the static version each tone uniquely labels its
    stream, so there is **no polarization-permutation ambiguity**: output row
    ``j`` corresponds to ``tone_frequencies[j]``.

    **Tracking-bandwidth trade-off.**  ``track_bandwidth`` (the LPF cut-off) is
    the single design knob and is bounded on both sides:

    * It must be **≥ the SOP rotation rate**, or ``W(n)`` lags the true ``J(n)``
      and residual crosstalk remains (lag bias).
    * It must be **≤ the guard** to the nearest other tone and to the data band,
      or those leak into ``z_ij`` and corrupt the estimate.  The hard ceiling is
      roughly half the smallest tone spacing.

    If the SOP rotates faster than the available tone spacing permits to track,
    the tones are simply spaced too closely for that drift — a real feasibility
    limit; a warning is logged when ``2·track_bandwidth`` (plus the FIR
    transition) encroaches on the nearest tone spacing.

    Parameters
    ----------
    samples : array_like
        Received MIMO samples. Shape ``(C, N)`` — time on the last axis.
    sampling_rate : float
        Sampling rate f_s in Hz.
    tone_frequencies : sequence of float
        The ``K`` distinct per-stream tone frequencies in Hz (as added at the
        TX, in transmitted-stream order).  Require ``K <= C``.  Output row ``j``
        corresponds to ``tone_frequencies[j]``.
    track_bandwidth : float
        One-sided LPF cut-off in Hz — the polarization-tracking bandwidth.  Set
        it a few times above the expected SOP rotation rate but well below the
        smallest tone spacing (see the trade-off above).
    num_taps : int, optional
        Length of the tracking low-pass FIR.  Defaults to ``~3.3·f_s/track_bandwidth``
        (the Hamming transition width that resolves ``track_bandwidth``), forced
        odd and clipped below ``N``.  Increase for sharper neighbour-tone
        rejection at the cost of longer edge transients.
    grid_step : int, optional
        Decimation (in samples) of the grid on which ``T(n)`` is inverted.
        Defaults to ``max(1, floor(f_s / (4·track_bandwidth)))`` — i.e. oversample
        the tracked process ~4x.  ``W`` is linearly interpolated between grid
        points, so a finer grid costs more inverses but tracks marginally better.
    refine_tones : bool, default True
        If ``True``, sub-bin-refine each tone centre with ``find_bias_tone`` (on
        the receive channel where it is strongest) before mixing down, absorbing
        a residual carrier frequency offset.
    search_band : float, optional
        Half-width in Hz of the per-tone peak search when ``refine_tones=True``.
        Defaults to ``4 · f_s / N``.
    normalize : bool, default True
        If ``True``, rescale each demuxed row so its mean power equals the mean
        per-channel input power (removes the ``1/α_j`` scale; preserves the
        library power invariant).  When ``trim_edges=True`` the power is measured
        over the retained interior only.
    trim_edges : bool, default False
        The tracking FIR is applied with centred (``'same'``) convolution, so the
        Jones estimate — and hence ``W(n)`` — is unreliable within ``num_taps//2``
        samples of each record end (the convolution averages in zero-padding
        there).  The **data is never filtered**, so timing is unaffected, but
        those edge samples carry residual crosstalk.  If ``True``, drop them:
        ``demuxed`` is returned as the reliable interior ``(K, N - 2·g)`` with
        ``g = num_taps//2``, together with a ``valid`` slice giving the retained
        sample range in **original** coordinates (so full-length references align
        as ``ref[..., valid]``).
    return_matrix : bool, default False
        If ``True``, also return the decimated unmixer stack ``W_grid`` and the
        sample positions ``grid_positions`` it was evaluated at (suitable for
        seeding a time-varying butterfly equalizer).  ``W_grid`` / ``grid_positions``
        always span the **full** record, even when ``trim_edges=True``.
    apply : bool, default True
        If ``True`` (default), interpolate ``W(n)`` to full rate and apply it,
        returning the demuxed signal as documented below.  If ``False``,
        **matrix-only mode**: skip the ``O(N)`` interpolate-and-apply entirely and
        return just ``(W_grid, grid_positions)`` (``return_matrix`` is implied).
        Use this when only the unmixer stack is needed — e.g. to make a
        PDL/unitarity decision and then apply a *different* factor (a polar unitary
        ``Qᴴ(n)``) without paying for a demux that would be discarded.  ``normalize``
        and ``trim_edges`` act on the applied signal, so they have **no effect**
        when ``apply=False``.

    Returns
    -------
    demuxed : array_like
        Demultiplexed streams. Same complex dtype and backend as the input; row
        ``j`` carried ``tone_frequencies[j]``.  Shape ``(K, N)``, or
        ``(K, N - 2·(num_taps//2))`` when ``trim_edges=True``.  **Omitted** when
        ``apply=False`` (the return is then ``(W_grid, grid_positions)``).
    valid : slice, optional
        Returned only if ``trim_edges=True``: the ``slice(g, N - g)`` of original
        sample indices retained in ``demuxed`` (``g = num_taps//2``).  Always
        precedes ``W_grid`` in the output tuple.
    W_grid : array_like, optional
        Returned only if ``return_matrix=True``: the ``(G, K, C)`` stack of
        per-grid-point unmixing matrices (``complex128``).
    grid_positions : array_like, optional
        Returned only if ``return_matrix=True``: the ``(G,)`` sample indices
        (``float64``) at which ``W_grid`` was evaluated.

    Raises
    ------
    ValueError
        If ``samples`` is not 2-D, ``tone_frequencies`` is empty, ``K > C``,
        ``track_bandwidth`` is not positive, or any tone frequency lies outside
        ``(-f_s/2, f_s/2)``.

    See Also
    --------
    demultiplex_polarization_tones_static : One-shot static-SOP demux (faster).
    add_pilot_tone : Add the per-stream tones at the transmitter.
    """
    samples, xp, _ = dispatch(samples)
    if samples.ndim != 2:
        raise ValueError(
            "demultiplex_polarization_tones_dynamic requires a 2-D (C, N) MIMO "
            f"input; got ndim={samples.ndim}."
        )
    C, N = samples.shape

    f_tones = [float(f) for f in tone_frequencies]
    K = len(f_tones)
    if K == 0:
        raise ValueError("tone_frequencies must contain at least one frequency.")
    if K > C:
        raise ValueError(
            f"got K={K} tones but only C={C} receive channels; need K <= C to "
            "unmix (one tone per transmitted stream)."
        )
    if not (track_bandwidth > 0):
        raise ValueError(f"track_bandwidth must be positive; got {track_bandwidth}.")
    nyq = sampling_rate / 2.0
    for f in f_tones:
        if not (-nyq < f < nyq):
            raise ValueError(
                f"tone_frequencies entry {f} must lie in (-fs/2, fs/2) = "
                f"(±{nyq:.3g}) Hz."
            )

    df = sampling_rate / N
    two_pi = 2.0 * xp.pi
    samples_c = xp.asarray(samples, dtype=xp.complex128)  # (C, N)
    n = xp.arange(N, dtype=xp.float64)

    # --- Tracking low-pass design + feasibility check ------------------------
    if num_taps is None:
        # Hamming transition width ≈ 3.3·fs/num_taps; size it to resolve the
        # requested tracking bandwidth.  Force odd; keep it shorter than N.
        num_taps = int(round(3.3 * sampling_rate / track_bandwidth))
        num_taps += 1 - (num_taps % 2)  # nearest odd >= value
        num_taps = max(num_taps, 3)
    num_taps = min(int(num_taps), (N // 2) * 2 - 1)
    h = lowpass_taps(sampling_rate, num_taps, track_bandwidth)

    # Edge guard: 'same' convolution corrupts num_taps//2 samples at each end.
    # num_taps is clipped < N above, so the retained interior is always non-empty.
    guard = num_taps // 2 if trim_edges else 0

    if K > 1:
        sorted_f = sorted(f_tones)
        d_min = min(b - a for a, b in zip(sorted_f, sorted_f[1:]))
        transition = 3.3 * sampling_rate / num_taps
        if 2.0 * track_bandwidth + transition >= d_min:
            logger.warning(
                "demultiplex_polarization_tones_dynamic: tracking bandwidth "
                "(%.3g Hz, FIR transition %.3g Hz) approaches the nearest tone "
                "spacing %.3g Hz — neighbouring tones may leak into the Jones "
                "estimate. Reduce track_bandwidth or widen the tone spacing.",
                track_bandwidth,
                transition,
                d_min,
            )

    # --- Optional sub-bin tone refinement (reuse the static one-shot bin to
    #     pick, per tone, the receive channel where it is strongest) ----------
    def _static_bin(freqs: Sequence[float]) -> ArrayType:
        fc = xp.asarray(freqs, dtype=xp.float64).reshape(K, 1)
        ph = two_pi * fc * n[None, :] / sampling_rate
        ph = ph - xp.round(ph / two_pi) * two_pi
        return (samples_c @ xp.exp(-1j * ph).T) / N  # (C, K)

    if refine_tones:
        if search_band is None:
            search_band = 4.0 * df
        best_ch = to_device(xp.argmax(xp.abs(_static_bin(f_tones)), axis=0), "cpu")
        f_used = [
            find_bias_tone(
                samples_c[int(best_ch[j])],
                sampling_rate,
                target_frequency=f_tones[j],
                search_band=search_band,
            )
            for j in range(K)
        ]
    else:
        f_used = f_tones

    # --- Track the Jones matrix: mix each tone to DC and low-pass ------------
    # T_t[i, j, n] = LPF{ r_i(n) · exp(-j2π f_j n/fs) } ≈ J_ij(n)·α_j.
    # fir_filter uses centred ('same') linear-phase convolution, so the LPF
    # group delay is compensated and z aligns in time with the input.
    f_arr = xp.asarray([float(fj) for fj in f_used], dtype=xp.float64)  # (K,)
    # The phase ramp 2π·f·n/fs reaches ~1e7 rad over a long capture, so it MUST be
    # formed and wrapped in float64 before the complex exp — float32 here loses the
    # integer turn count and corrupts every tone phasor.  Everything downstream (the
    # mix-down product and the averaging LPF) is well-conditioned, so it drops to
    # complex64: half the bandwidth / temporary size on the (C, K, N) hot arrays,
    # with FFT-FIR round-off (~√N_fft·ε ≈ 5e-5) far below the demux crosstalk floor.
    ph = two_pi * f_arr[:, None] * n[None, :] / sampling_rate  # (K, N) float64
    ph = ph - xp.round(ph / two_pi) * two_pi  # wrap in float64 (essential)
    carrier = xp.exp(-1j * ph).astype(xp.complex64)  # (K, N)
    mixed = samples_c.astype(xp.complex64)[:, None, :] * carrier[None, :, :]  # (C,K,N)
    # One batched linear-phase FIR over (C·K) rows instead of K separate calls.
    # fir_filter is signal-driven precision, so a complex64 input stays complex64.
    T_t = cast(ArrayType, fir_filter(mixed.reshape(C * K, N), h, axis=-1)).reshape(
        C, K, N
    )

    # --- Invert on a decimated grid -----------------------------------------
    if grid_step is None:
        grid_step = max(1, int(sampling_rate / (4.0 * track_bandwidth)))
    grid_step = int(min(max(grid_step, 1), N))
    idx = xp.arange(0, N, grid_step)
    if int(idx[-1]) != N - 1:
        idx = xp.concatenate([idx, xp.asarray([N - 1])])  # pin the last sample
    G = int(idx.shape[0])
    grid_positions = idx.astype(xp.float64)

    # Promote the grid samples back to double for the (sensitive) batched inverse —
    # the Gram/inverse follows the RLS-style double-precision convention, while the
    # O(N) tracker FIR above ran in complex64.  G is small, so this cast is cheap.
    Tg = xp.moveaxis(T_t[:, :, idx], 2, 0).astype(xp.complex128)  # (G, C, K)
    Th = xp.conj(xp.swapaxes(Tg, -1, -2))  # (G, K, C)
    gram = Th @ Tg  # (G, K, K)
    # Tikhonov regularisation keeps the batched inverse well-conditioned when the
    # instantaneous SOP nearly aligns two streams (gram → singular).
    diag_mean = xp.real(xp.trace(gram, axis1=-2, axis2=-1)) / K  # (G,)
    eye = xp.eye(K, dtype=xp.complex128)
    ridge = (1e-9 * diag_mean)[:, None, None] * eye[None, :, :]
    Wg = xp.linalg.inv(gram + ridge) @ Th  # (G, K, C) — batched, CPU+GPU

    if not apply:
        # Matrix-only mode: skip the O(N) interpolate-and-apply.  normalize and
        # trim_edges act on the applied signal, so they are no-ops here.
        logger.info(
            "demultiplex_polarization_tones_dynamic (matrix-only): f_tones=%s Hz, "
            "refine=%s, track_bw=%.3g Hz, taps=%d, grid_step=%d, G=%d "
            "[C=%d, K=%d, N=%d]",
            [f"{f:.3g}" for f in f_used],
            refine_tones,
            track_bandwidth,
            num_taps,
            grid_step,
            G,
            C,
            K,
            N,
        )
        return Wg, grid_positions

    # --- Interpolate W to full rate and apply (block-vectorised over the grid) ---
    # Within a grid cell W(n) is the exact linear blend W[g] + (W[g+1]-W[g])·t, so
    # each uniform cell collapses to two batched C×C GEMMs — W[g]·X and dW·(X·t) —
    # instead of materialising a per-sample (L,K,C) matrix and an einsum matvec.
    # complex64 on this O(N) data path: the demux apply is a well-conditioned mix
    # with no long accumulation, so single precision is exact enough (the grid
    # inverse stayed double); the bulk traffic then moves at half the bandwidth.
    Wc = Wg.astype(xp.complex64)  # (G, K, C)
    xc = samples_c.astype(xp.complex64)  # (C, N)
    demuxed = xp.empty((K, N), dtype=xp.complex64)
    nblk = (N - 1) // grid_step  # full uniform cells cover [0, nblk·grid_step)
    bulk = nblk * grid_step
    if nblk > 0:
        Xb = xc[:, :bulk].reshape(C, nblk, grid_step).transpose(1, 0, 2)  # (nblk,C,gs)
        W0 = Wc[:nblk]  # (nblk, K, C)
        dW = Wc[1 : nblk + 1] - W0
        t = (xp.arange(grid_step, dtype=xp.float32) / grid_step).astype(xp.complex64)
        y = W0 @ Xb + dW @ (Xb * t[None, None, :])  # (nblk, K, grid_step)
        demuxed[:, :bulk] = y.transpose(1, 0, 2).reshape(K, bulk)
    if bulk < N:  # tail (< grid_step, spans the pinned last cell) — per-sample blend
        nn = n[bulk:]
        lo = xp.clip(xp.searchsorted(grid_positions, nn, side="right") - 1, 0, G - 2)
        frac = ((nn - grid_positions[lo]) / (grid_positions[lo + 1] - grid_positions[lo])).astype(
            xp.complex64
        )
        W_full = Wc[lo] + (Wc[lo + 1] - Wc[lo]) * frac[:, None, None]  # (L, K, C)
        demuxed[:, bulk:] = xp.einsum("lkc,cl->kl", W_full, xc[:, bulk:])

    # Drop the FIR edge transient (the data is untouched; only W is unreliable
    # there).  ``valid`` reports the retained range in original coordinates.
    valid = slice(guard, N - guard)
    demuxed = demuxed[:, valid]

    if normalize:
        p_in = xp.mean(xp.abs(samples[:, valid]) ** 2)
        p_out = xp.mean(xp.abs(demuxed) ** 2, axis=-1, keepdims=True)  # (K, 1)
        scale = xp.sqrt(p_in / xp.where(p_out > 0, p_out, 1.0))
        demuxed = demuxed * scale

    demuxed = demuxed.astype(samples.dtype)

    logger.info(
        "demultiplex_polarization_tones_dynamic: f_tones=%s Hz, refine=%s, "
        "track_bw=%.3g Hz, taps=%d, grid_step=%d, G=%d [C=%d, K=%d, N=%d]",
        [f"{f:.3g}" for f in f_used],
        refine_tones,
        track_bandwidth,
        num_taps,
        grid_step,
        G,
        C,
        K,
        N,
    )

    out: tuple[Any, ...] = (demuxed,)
    if trim_edges:
        out = out + (valid,)
    if return_matrix:
        out = out + (Wg, grid_positions)
    return out[0] if len(out) == 1 else out
