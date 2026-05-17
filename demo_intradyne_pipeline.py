#!/usr/bin/env python3
"""
demo_intradyne_pipeline.py — End-to-end intradyne DP-coherent DSP demo.

Walks through the offline DSP pipeline for a tiled-DAC waveform captured by
an intradyne dual-pol coherent receiver, exercising the workflow we converged
on during debugging:

  1.  Load TX reference, strip the digital-FO baked at TX, bring to the ADC
      sample rate.
  2.  Load capture, form 2-pol complex, fix IQ imbalance.  No pre-trim — we
      truncate AFTER timing alignment so we don't waste a frame.
  3.  Coarse FOE on the first DAC-buffer segment using the bias-leakage tone
      with log-parabolic sub-bin interpolation.  Applies one constant shift
      to the whole capture so cross-correlation phase coherence survives the
      timing step.
  4.  Timing alignment using cross-correlation on the first ~2 tiles so the
      search lands on the FIRST tile peak (not the strongest, which would
      typically live deep in the capture and force a circular roll).
      `mode='slice'` — keeps the phase trajectory continuous at sample 0.
  5.  Truncate to an integer number of frames after the slice.
  6.  Per-segment refined FOE: bias-tone sub-bin estimate on each tile,
      cubic-spline-interpolate Δf(t) between segment centres, integrate to a
      continuous per-sample phase trajectory, apply via `correct_carrier_phase`.
      No discontinuities at segment boundaries (the key constraint when
      tracking a fast-drifting carrier).  Equivalent in structure to
      `estimate_frequency_offset_blockwise`, but plugs the bias tone in as
      the per-block estimator instead of M-th-power (which has poor SNR on
      16-QAM and ignores the bias tone entirely).
  7.  Resample reference and capture down to 2 sps; matched-filter the RX.
  8.  Tile `source_bits`/`source_symbols` to match the aligned frame count
      so metrics tile cleanly.
  9.  Iterative warm-started butterfly block-LMS with BPS CPR.  We also pass
      `training_symbols` to the final full-signal call (workaround for the
      CPR-state-reset-on-warm-start issue tracked as Issue A in
      DSP_DEBUG_PLAN.md).
 10.  Standalone BPS-CPR on the equalized output, phase-ambiguity resolution,
      hard demap.
 11.  Metrics — EVM (blind), SER, SNR.  SER/SNR discard `2 × num_train` symbols
      to skip past the post-training CPR-convergence region.

Run with:
    uv run python demo_intradyne_pipeline.py
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from commstools import Signal, load_npz
from commstools.backend import dispatch
from commstools.equalization import block_lms
from commstools.sync import (
    compensate_iq_imbalance_lowdin,
    correct_carrier_phase,
    correct_frequency_offset,
    correct_timing,
    estimate_timing,
    recover_carrier_phase_bps,
)


TX_PATH = "/home/lokgar/repos/trmhi304-p2p/waveforms/signal_16qam"
RX_PATH = "/home/lokgar/repos/trmhi304-p2p/captures/capture_16qam_intradyne_long.npy"


# =============================================================================
# Helpers — bias-tone FOE + continuous-phase per-segment refinement (Option B)
# =============================================================================

def _to_cpu(x):
    """Return ``x`` as a 1-D CPU numpy array regardless of backend."""
    return x.get() if hasattr(x, "get") else np.asarray(x)


def find_bias_tone(seg_1d_cpu, fs, target_hz=None, search_band_hz=None):
    """
    Sub-bin PSD-peak FOE.

    Locate the strongest spectral peak in `seg_1d_cpu` and refine its
    position via log-parabolic fit on the three bins around the argmax.
    Returns the refined peak frequency in Hz.

    Optional restriction: pass both ``target_hz`` and ``search_band_hz`` to
    limit the search to ``[target_hz ± search_band_hz]``.  Useful for
    tracking the bias tone after a coarse correction has placed it near a
    known target frequency, so the wider-band signal doesn't fool argmax.

    Parameters
    ----------
    seg_1d_cpu : 1-D complex numpy array (CPU)
    fs : sampling rate in Hz
    target_hz, search_band_hz : optional band restriction in Hz
    """
    x = np.asarray(seg_1d_cpu)
    N = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    Pxx = spec.real ** 2 + spec.imag ** 2
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))

    if target_hz is not None and search_band_hz is not None:
        band = np.where(np.abs(f - target_hz) < search_band_hz)[0]
        k = int(band[int(np.argmax(Pxx[band]))])
    else:
        k = int(np.argmax(Pxx))

    if 0 < k < N - 1:
        ym = np.log(max(Pxx[k - 1], 1e-30))
        y0 = np.log(max(Pxx[k],     1e-30))
        yp = np.log(max(Pxx[k + 1], 1e-30))
        denom = ym - 2.0 * y0 + yp
        delta = 0.5 * (ym - yp) / denom if abs(denom) > 1e-30 else 0.0
    else:
        delta = 0.0
    return float(f[k] + delta * (fs / N))


def piecewise_carrier_phase(samples_1d_cpu, fs, block_size, overlap, estimator):
    """
    Build a per-sample carrier-phase trajectory from a per-block FOE.

    Mirrors the structure of `estimate_frequency_offset_blockwise` (overlap,
    cubic-spline interpolation between block centres, ``cumsum``-to-phase)
    but accepts an arbitrary per-block estimator callable.  This is the
    "Option B" pattern from the debugging discussion: same continuous-phase
    machinery, custom estimator under the hood.

    Parameters
    ----------
    samples_1d_cpu : (N,) complex numpy array, CPU
    fs : sampling rate (Hz)
    block_size : analysis block length in samples
    overlap : fractional overlap in ``[0, 1)``
    estimator : callable ``(block_1d_cpu, fs) -> df_hz``

    Returns
    -------
    theta : (N,) float64 — per-sample phase trajectory in radians.
            Apply via ``correct_carrier_phase(samples, theta)``.
    """
    N = len(samples_1d_cpu)
    step = max(1, round(block_size * (1.0 - overlap)))
    starts = list(range(0, N - block_size + 1, step)) or [0]

    t_centers = np.array([s + block_size / 2.0 for s in starts], dtype=np.float64)
    dfs = np.array(
        [estimator(samples_1d_cpu[s : s + block_size], fs) for s in starts],
        dtype=np.float64,
    )
    print(
        f"    per-segment Δf:  mean = {dfs.mean():+.1f} Hz,  "
        f"std = {dfs.std():.1f} Hz,  "
        f"range = [{dfs.min():+.1f}, {dfs.max():+.1f}] Hz "
        f"({len(starts)} segments)"
    )

    n_grid = np.arange(N, dtype=np.float64)
    kind = "cubic" if len(starts) >= 4 else "linear"
    df_dense = interp1d(
        t_centers, dfs,
        kind=kind, bounds_error=False,
        fill_value=(dfs[0], dfs[-1]),
    )(n_grid)
    return (2.0 * np.pi / fs) * np.cumsum(df_dense)


# =============================================================================
# Pipeline
# =============================================================================

def main():
    # ─────────────────────────────────────────────────────────────────────
    # 1. Reference TX waveform, brought to the ADC sample rate (8 sps)
    # ─────────────────────────────────────────────────────────────────────
    print("[1] Loading TX reference …")
    tx = load_npz(TX_PATH)
    _, xp, _ = dispatch(tx.samples)

    digoff = tx.digital_frequency_offset           # +300 MHz baked at TX
    tx.shift_frequency(-digoff)                    # strip it from the reference
    tx.resample(up=1, down=2)                      # 4 GSps → 2 GSps  (16 sps → 8 sps)
    tile_len_adc = tx.samples.shape[-1]            # one DAC buffer at the ADC rate
    print(f"    tile_len_adc = {tile_len_adc} samples @ {tx.sampling_rate/1e9:.1f} GSps "
          f"(sps = {tx.sps:.0f}),  digoff = {digoff:+.1f} Hz")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Capture: form 2-pol complex, fix IQ imbalance.  No pre-trim — we
    #    truncate after timing alignment so we don't waste a frame.
    # ─────────────────────────────────────────────────────────────────────
    print("[2] Loading capture, fixing IQ imbalance …")
    rx_raw = xp.load(RX_PATH)
    rx_samples = xp.array([
        rx_raw[0] + 1j * rx_raw[1],
        rx_raw[2] + 1j * rx_raw[3],
    ])
    rx_samples = compensate_iq_imbalance_lowdin(xp.conj(rx_samples))

    rx = Signal(
        samples=rx_samples,
        sampling_rate=tx.sampling_rate,
        symbol_rate=tx.symbol_rate,
        mod_scheme=tx.frame.payload_mod_scheme,
        mod_order=tx.frame.payload_mod_order,
        frame=tx.frame,
        pulse_shape=tx.pulse_shape,
    )
    n_buf_raw = rx.samples.shape[-1] / tile_len_adc
    print(f"    rx.samples.shape = {tuple(rx.samples.shape)}  "
          f"({n_buf_raw:.2f} DAC buffers in the raw capture)")

    # ─────────────────────────────────────────────────────────────────────
    # 3. Coarse FOE on the FIRST tile — drives the timing step.
    #    Use the bias-leakage tone in the PSD with log-parabolic sub-bin
    #    interpolation.  One sub-bin pass is enough precision for timing's
    #    phase coherence; per-segment refinement comes later.
    # ─────────────────────────────────────────────────────────────────────
    print("[3] Coarse FOE on first segment (bias-tone, sub-bin) …")
    first_seg_cpu = _to_cpu(rx.samples[0, :tile_len_adc])
    f_b_coarse = find_bias_tone(first_seg_cpu, rx.sampling_rate)
    coarse_shift = f_b_coarse + digoff             # puts bias at -digoff, signal at 0
    print(f"    bias-tone @ {f_b_coarse:+.3f} MHz, "
          f"applying shift = {coarse_shift/1e6:+.3f} MHz")
    rx.samples = correct_frequency_offset(rx.samples, rx.sampling_rate, coarse_shift)

    # ─────────────────────────────────────────────────────────────────────
    # 4. Timing — first peak, slice mode.  Restrict the cross-correlation
    #    search to the first 2 tiles so estimate_timing's argmax lands on the
    #    FIRST tile peak (otherwise the strongest of many near-identical
    #    tile-peaks wins, typically pointing deep into the capture).
    # ─────────────────────────────────────────────────────────────────────
    print("[4] Timing alignment (first peak, mode='slice') …")
    coarse, fract = estimate_timing(
        rx.samples,
        tx.samples,
        sps=rx.sps,
        search_range=(0, 2 * tile_len_adc),
        threshold=3,
        debug_plot=True,
    )
    coarse_cpu = _to_cpu(coarse)
    fract_cpu = _to_cpu(fract)
    print(f"    coarse = {coarse_cpu.tolist()}, fract = {fract_cpu.tolist()}, "
          f"cross-pol skew = {abs(int(coarse_cpu[0]) - int(coarse_cpu[1]))} samples")

    # `mode='slice'` discards `coarse` leading samples — no circular wrap, no
    # phase discontinuity at the front of the equalizer input.  Fractional
    # delay is applied to the FULL pre-slice buffer (recent fix in
    # `correct_timing`), so the new sample 0 is uncontaminated by FFT
    # wrap-around.
    rx.samples = correct_timing(rx.samples, coarse, fract, mode="slice")

    # ─────────────────────────────────────────────────────────────────────
    # 5. Truncate to an integer number of frames after the slice.
    #    Doing this *after* the slice (not before) preserves up to one extra
    #    frame relative to a pre-trim, since the raw trailing partial tile
    #    can now contribute when it's bigger than `max(coarse)`.
    # ─────────────────────────────────────────────────────────────────────
    n_seg_adc = rx.samples.shape[-1] // tile_len_adc
    rx.samples = rx.samples[:, : n_seg_adc * tile_len_adc]
    print(f"[5] Truncated to {n_seg_adc} full frames @ ADC rate "
          f"({rx.samples.shape[-1]} samples per pol)")

    # ─────────────────────────────────────────────────────────────────────
    # 6. Per-segment refined FOE with a continuous-phase trajectory.
    #    Search around the post-coarse bias-tone location (-digoff) in each
    #    segment, compute the residual drift, cubic-spline-interpolate, and
    #    integrate to a per-sample phase array applied via
    #    correct_carrier_phase.  No phase discontinuities at boundaries.
    # ─────────────────────────────────────────────────────────────────────
    print(f"[6] Per-segment refined FOE ({n_seg_adc} segments × {tile_len_adc} samples) …")

    rx_cpu = _to_cpu(rx.samples[0])

    def per_segment_estimator(blk_cpu, fs):
        # After the coarse correction, the bias tone sits near -digoff.
        # Any deviation = the residual frequency drift to be removed.
        f_b_k = find_bias_tone(
            blk_cpu, fs,
            target_hz=-digoff,
            search_band_hz=20e6,   # ±20 MHz around -digoff is well clear of the signal
        )
        return f_b_k + digoff      # δf_k = current bias offset relative to target

    theta_cpu = piecewise_carrier_phase(
        rx_cpu, rx.sampling_rate,
        block_size=tile_len_adc,
        overlap=0.5,
        estimator=per_segment_estimator,
    )
    theta = xp.asarray(theta_cpu)
    rx.samples = correct_carrier_phase(rx.samples, theta)

    # ─────────────────────────────────────────────────────────────────────
    # 7. Resample reference and capture to 2 sps; the polyphase anti-alias
    #    filter kills the now-stationary bias tone at -digoff = -300 MHz
    #    (outside ±250 MHz Nyquist at 500 MSps).
    # ─────────────────────────────────────────────────────────────────────
    print("[7] Resampling reference and RX to 2 sps …")
    rx.resample(sps_out=2)
    tx.resample(sps_out=2)
    tile_len = tx.samples.shape[-1]                # one frame at 2 sps

    # Re-truncate to integer frames (the resampler can change length by a few
    # samples; cheap to re-align here so source-symbol tiling stays exact).
    n_seg = rx.samples.shape[-1] // tile_len
    rx.samples = rx.samples[:, : n_seg * tile_len]
    print(f"    rx.samples.shape = {tuple(rx.samples.shape)}  "
          f"({n_seg} frames @ 2 sps)")
    rx.plot_psd(show=True, title="After FOE + timing + per-segment refinement (2 sps)")

    # ─────────────────────────────────────────────────────────────────────
    # 8. Matched filter + tile source bits/symbols to the aligned frame count.
    # ─────────────────────────────────────────────────────────────────────
    print("[8] Matched filter + tiling source bits/symbols …")
    rx.source_bits = xp.tile(tx.frame.payload_bits, (1, n_seg))
    rx.source_symbols = xp.tile(tx.frame.payload_symbols, (1, n_seg))
    rx.matched_filter()
    rx.plot_psd(show=True, title="Synced and MFed PSD")

    # ─────────────────────────────────────────────────────────────────────
    # 9. Iterative warm-started butterfly block-LMS with BPS CPR.
    #    Known limitation: CPR state resets to 0 on every block_lms entry
    #    (DSP_DEBUG_PLAN.md Issue A).  Workaround used here: pass
    #    `training_symbols` to the final full-signal call too, so the
    #    equalizer trains over the front of the stream while BPS spins up,
    #    instead of going straight to DD and corrupting the warm-started
    #    taps.
    # ─────────────────────────────────────────────────────────────────────
    print("[9] Iterative block-LMS + BPS CPR …")
    NUM_EQ = 5
    NUM_TRAIN = 2 ** 13
    NUM_TAPS = 75
    BLOCK_SIZE = 128
    STEP_SIZE = 2e-4

    eq_kwargs = dict(
        modulation=rx.mod_scheme,
        order=rx.mod_order,
        num_taps=NUM_TAPS,
        step_size=STEP_SIZE,
        sps=rx.sps,
        block_size=BLOCK_SIZE,
        cpr_type="bps",
        cpr_bps_joint_channels=True,
        cpr_bps_test_phases=1024,
        cpr_bps_block_size=32,
        cpr_cycle_slip_correction=False,
        cpr_cycle_slip_history=16,
        cpr_cycle_slip_threshold=np.pi / 4,
        plot_smoothing=200,
    )

    res = None
    for i in range(NUM_EQ):
        res = block_lms(
            samples=rx.samples[:, : NUM_TRAIN * 2],
            training_symbols=rx.source_symbols[:, :NUM_TRAIN],
            w_init=res.weights if res is not None else None,
            debug_plot=(i == 0 or i == NUM_EQ - 1),
            **eq_kwargs,
        )

    # Final pass over the full sequence — pass training too (Issue A workaround).
    res = block_lms(
        samples=rx.samples,
        training_symbols=rx.source_symbols[:, :NUM_TRAIN],
        w_init=res.weights,
        debug_plot=True,
        **eq_kwargs,
    )

    eq = rx.copy()
    eq.samples = res.y_hat
    eq.sampling_rate = rx.symbol_rate
    eq.plot_constellation(show=True, overlay_ideal=True)

    # ─────────────────────────────────────────────────────────────────────
    # 10. Standalone BPS-CPR on the equalized output, then phase ambiguity
    #     resolution and hard demap.
    # ─────────────────────────────────────────────────────────────────────
    print("[10] Standalone BPS-CPR + phase ambiguity + demap …")
    phases_bps = recover_carrier_phase_bps(
        eq.samples,
        modulation=eq.mod_scheme,
        order=eq.mod_order,
        debug_plot=True,
        block_size=32,
        num_test_phases=1024,
        cycle_slip_correction=False,
        cycle_slip_history=16,
        cycle_slip_threshold=np.pi / 4,
        joint_channels=True,
    )

    eqfoecpr = eq.copy()
    eqfoecpr.samples = correct_carrier_phase(eqfoecpr.samples, phases_bps)
    eqfoecpr.resolve_symbols()
    eqfoecpr.resolve_phase_ambiguity()
    eqfoecpr.demap_symbols_hard()
    eqfoecpr.plot_constellation(show=True, overlay_ideal=True)

    # ─────────────────────────────────────────────────────────────────────
    # 11. Metrics.  EVM "blind" measures blob size against the nearest
    #     constellation point.  SER/SNR discard `2 × NUM_TRAIN` symbols to
    #     skip past the post-training CPR convergence region (Issue A).
    # ─────────────────────────────────────────────────────────────────────
    print("[11] Metrics …")
    eqfoecpr.evm(num_train_symbols=NUM_TRAIN, mode="blind")
    eqfoecpr.ser(num_train_symbols=NUM_TRAIN * 2)
    eqfoecpr.snr(num_train_symbols=NUM_TRAIN * 2)


if __name__ == "__main__":
    main()
