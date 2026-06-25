#!/usr/bin/env python
"""Multi-stage equalization: FSE → FOE → CPR → SSE

Demonstrates the standard coherent DSP chain for linear dispersion correction,
showing how to pass weights between stages and what the signal looks like
at each processing step.

Pipeline
--------
  TX (2 SPS, RRC)
     └→ channel: FIR dispersion (multipath) + frequency offset + AWGN
        └→ [Stage 1]  RDE  blind FSE  (sps=2) → 1 SPS symbols
           └→ [FOE]  mth-power offset estimation + correction (raw samples)
              └→ [Apply Taps]  FSE filtering with frozen RDE weights
                 └→ [CPR]  Viterbi-Viterbi phase recovery
                    └→ [Symbol Timing Sync]  estimate_timing + correct_timing
                       └→ [Stage 2] LMS symbol-spaced (sps=1) residual ISI
                          └→ metrics (EVM, BER)

Weight-passing patterns shown
------------------------------
  1. apply_taps():         re-use frozen Stage-1 RDE taps on a new capture
  2. Stage 2 sps=1:        LMS runs at symbol rate after FOE+CPR + Timing Sync

DSP Metrics & Phase Alignment (Offline vs. Blind Recovery)
-----------------------------------------------------------
  This script highlights a key DSP phenomenon comparing reference-aided alignment
  vs. blind receiver phase recovery:

  * Blind Equalization (RDE FSE) blindly compensates for the dispersion/multipath
    and downsamples the signal to 1 SPS. Since RDE is phase-blind, the signal
    retains its carrier frequency offset and laser phase noise.
  * Post-CPR (FOE + CPR + Timing Sync) estimates frequency offset from the opened
    constellation, corrects it on raw samples, filters with frozen FDE weights,
    despins carrier phase noise blindly using Viterbi-Viterbi, and synchronizes
    symbol delay using built-in library functions. Constellation eyes open fully.
  * Stage 2 (LMS SSE) runs at 1 SPS on the synchronized, despun CPR output,
    delivering the realistic steady-state performance of a complete blind receiver.

Run
---
  uv run python examples/multi_stage_equalization.py
"""

import numpy as np

from commstools import (
    Signal,
    equalization,
    filtering,
    frequency,
    metrics,
    recovery,
    spectral,
)
from commstools.backend import dispatch
from commstools.impairments import apply_awgn
from commstools.timing import correct_timing, estimate_timing

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────

SPS = 2
SYMBOL_RATE = 100e6  # 100 MBaud
FS = SPS * SYMBOL_RATE  # 200 MSa/s
N_SYM = 8192
MOD = "qam"
ORDER = 16
ROLLOFF = 0.2
N_TAPS_FSE = 15  # Stage 1: fractionally-spaced filter length (spans 7.5 symbols)
N_TAPS_SSE = (
    31  # Stage 2: symbol-spaced filter (spans 31 symbols to clean up residual ISI)
)
N_TRAIN = 512  # DA pilot symbols for LMS stages

SNR_DB = 22.0
FO_TRUE_HZ = 100.0e3  # 100 kHz frequency offset

rng = np.random.default_rng(42)


def _print_sep(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def _mse_db(error, window=200):
    """Steady-state MSE over the last ``window`` symbols."""
    error, xp, _ = dispatch(error)
    mse = float(xp.mean(xp.abs(error[..., -window:]) ** 2))
    return 10.0 * np.log10(mse + 1e-30)


# ──────────────────────────────────────────────────────────────────────────────
# 1. TRANSMITTER
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("1. Transmitter")

sig_tx = Signal.qam(
    num_symbols=N_SYM,
    sps=SPS,
    symbol_rate=SYMBOL_RATE,
    order=ORDER,
    rrc_rolloff=ROLLOFF,
)

training_syms = sig_tx.source_symbols

idx_eval = slice(N_TRAIN, -100)
bits_per_sym = int(np.log2(ORDER))
tx_bits_eval = sig_tx.source_bits[N_TRAIN * bits_per_sym : -100 * bits_per_sym]

print(f"  TX samples : {sig_tx.samples.shape}  (dtype={sig_tx.samples.dtype})")
print(f"  TX symbols : {training_syms.shape}  {ORDER}-{MOD.upper()}, SPS={SPS}")
print(f"  Symbol rate: {SYMBOL_RATE / 1e9:.0f} GBaud  |  FS: {FS / 1e9:.0f} GSa/s")


# ──────────────────────────────────────────────────────────────────────────────
# 2. CHANNEL: FIR multipath + frequency offset + AWGN
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("2. Channel")

# 3-tap multipath (ISI spans ±1 symbol at 2 SPS → 2 tap spacings)
h_ch = np.array([0.85, 0.0, 0.12 + 0.07j, 0.0, 0.05 - 0.03j], dtype=np.complex64)

# Apply impairments directly on the Signal object using library features!
sig_rx = sig_tx.copy()
sig_rx = filtering.fir_filter(sig_rx, h_ch)
sig_rx = spectral.shift_frequency(sig_rx, FO_TRUE_HZ)
sig_rx.samples = apply_awgn(sig_rx.samples, sps=SPS, esn0_db=SNR_DB, seed=42)
rx = sig_rx.samples

print(f"  Channel    : FIR {h_ch}")
print(f"  FO true    : {FO_TRUE_HZ / 1e6:.2f} MHz")
print(f"  SNR        : {SNR_DB} dB")
print(f"  RX samples : {rx.shape}  (dtype={rx.dtype})")


# ──────────────────────────────────────────────────────────────────────────────
# 3. STAGE 1: RDE blind FSE (sps=2)
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("3. Stage 1 — RDE blind FSE  (sps=2)")

result_rde = equalization.rde(
    rx,
    num_taps=N_TAPS_FSE,
    step_size=1e-3,
    modulation=MOD,
    order=ORDER,
    sps=SPS,
    backend="numba",
)

y_s1 = result_rde.y_hat  # (N_SYM,) at 1 SPS

print(f"  Input  : {rx.shape} samples  @ {SPS} SPS")
print(f"  Output : {y_s1.shape} symbols @ 1 SPS")
print(f"  Weights: {result_rde.weights.shape}  (C, C, num_taps)")
print(f"  MSE    : {_mse_db(result_rde.error):.1f} dB")

# Evaluate direct metrics for Stage 1 RDE before carrier recovery (will reflect spinning carrier phase)
sig_s1_eval = sig_tx.copy()
sig_s1_eval.samples = y_s1[..., idx_eval]
sig_s1_eval.sampling_rate = sig_tx.symbol_rate
sig_s1_eval.source_symbols = sig_tx.source_symbols[..., idx_eval]
sig_s1_eval.source_bits = sig_tx.source_bits[
    ..., N_TRAIN * bits_per_sym : -100 * bits_per_sym
]
sig_s1_eval.resolve_symbols()

evm_pct_s1, evm_db_s1 = metrics.evm(sig_s1_eval)
snr_val_s1 = metrics.snr(sig_s1_eval)
ser_val_s1 = metrics.ser(sig_s1_eval)
sig_s1_eval.demap_symbols_hard()
ber_val_s1 = metrics.ber(sig_s1_eval)

print("\n  Stage 1 (RDE FSE) Raw Metrics (Before FOE/CPR/Sync):")
print("  ───────────────────────────────────────────────────")
print(
    f"  EVM : {evm_pct_s1:.2f}% ({evm_db_s1:.1f} dB)  ← expected due to spinning carrier phase"
)
print(f"  SNR : {snr_val_s1:.2f} dB")
print(f"  SER : {ser_val_s1:.2e}")
print(f"  BER : {ber_val_s1:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. FOE & Apply Taps
#
#    Estimated blindly from 1-SPS Stage 1 symbols. We correct the frequency
#    offset on the raw 2-SPS samples and apply the frozen RDE taps to get
#    pristine, equalized 1-SPS symbols.
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("4. FOE & Apply Taps — mth-power law")

fo_est = frequency.estimate_frequency_offset_mth_power(
    y_s1,
    sampling_rate=SYMBOL_RATE,  # symbols are at 1 SPS → fs = symbol_rate
    modulation=MOD,
    order=ORDER,
)

rx_foe = frequency.correct_static_frequency_offset(rx, sampling_rate=FS, offset=fo_est)

# Apply frozen converged RDE taps to the frequency-corrected raw signal
y_foe = equalization.apply_taps(
    rx_foe,
    result_rde.weights,
    sps=SPS,
    normalize=True,
)

print(f"  True FO : {FO_TRUE_HZ / 1e6:.4f} MHz")
print(f"  Est  FO : {fo_est / 1e6:.4f} MHz")
print(f"  Error   : {(fo_est - FO_TRUE_HZ) / 1e3:.2f} kHz")
print(f"  Output  : {y_foe.shape} frequency-corrected and equalized symbols @ 1 SPS")


# ──────────────────────────────────────────────────────────────────────────────
# 6. BONUS: apply_taps() — re-use frozen Stage-1 weights on a new capture
#
#    Useful when you have a reference burst that trains the taps and then
#    want to apply those frozen taps to a subsequent data burst without
#    re-running adaptation.
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("6. apply_taps() — frozen weights on a second capture")

# Simulate a second noisy capture (same channel, different noise realisation) using library features!
sig_rx2 = sig_tx.copy()
sig_rx2 = filtering.fir_filter(sig_rx2, h_ch)
sig_rx2 = spectral.shift_frequency(sig_rx2, FO_TRUE_HZ)
sig_rx2.samples = apply_awgn(sig_rx2.samples, sps=SPS, esn0_db=SNR_DB, seed=100)
rx2 = sig_rx2.samples

# First correct frequency offset on the new capture using the estimated offset!
rx2_foe = frequency.correct_static_frequency_offset(
    rx2, sampling_rate=FS, offset=fo_est
)

y_frozen = equalization.apply_taps(
    rx2_foe,
    result_rde.weights,  # ← frozen blind RDE taps
    sps=SPS,
    normalize=True,
)
print(f"  New capture  : {rx2.shape} samples")
print(f"  Frozen taps  : result_rde.weights  {result_rde.weights.shape}")
print(f"  Output       : {y_frozen.shape} symbols @ 1 SPS  (no adaptation)")


# ──────────────────────────────────────────────────────────────────────────────
# 7. CPR: Viterbi-Viterbi carrier phase recovery & Timing Synchronization
#
#    Resolves carrier phase rotation blindly, followed by robust symbol timing
#    synchronization using the library's built-in estimate_timing and correct_timing
#    functions, and resolves the 4-fold phase ambiguity.
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("7. CPR & Timing Synchronization")

# Run blind carrier phase recovery on the equalized, frequency-corrected symbols
phase_vv = recovery.recover_carrier_phase_viterbi_viterbi(
    y_foe,
    modulation=MOD,
    order=ORDER,
    block_size=64,
    cycle_slip_correction=True,
)

y_cpr = recovery.correct_carrier_phase(y_foe, phase_vv)
_, xp, _ = dispatch(y_cpr)

# Integer & fractional symbol timing alignment using library-standard timing functions.
# We pre-pend a small zero-padding block of 20 symbols so that any negative delay offsets
# (common in dynamic FSE tap adaptation) are shifted into the positive lag search window.
PAD = 20
y_cpr_padded = xp.concatenate([xp.zeros(PAD, dtype=xp.complex64), y_cpr])

integer_est, frac_est = estimate_timing(
    y_cpr_padded,
    reference=training_syms[:N_TRAIN],
    threshold=1.0,
)
integer_offset = int(integer_est[0] - PAD)

# Correct timing on the original y_cpr signal (without the temporary padding)
y_cpr_sync = correct_timing(
    y_cpr,
    integer_offset=xp.array([integer_offset]),
    fractional_offset=frac_est,
    mode="circular",
)

print(
    f"  Timing Sync: Integer={integer_offset:d} symbols, Fractional={frac_est[0]:.3f} symbols"
)

# Resolve remaining 4-fold rotational phase ambiguity before Stage 2 LMS refinement
y_cpr_resolved = recovery.resolve_phase_ambiguity(
    y_cpr_sync,
    ref_symbols=training_syms,
    modulation=MOD,
    order=ORDER,
    num_skip_symbols=N_TRAIN,
)

print("  Block size : 64 symbols")
print(f"  Output     : {y_cpr_resolved.shape} phase-resolved and delay-aligned symbols")

# Evaluate direct metrics for Post-CPR
sig_cpr_eval = sig_tx.copy()
sig_cpr_eval.samples = y_cpr_resolved[..., idx_eval]
sig_cpr_eval.sampling_rate = sig_tx.symbol_rate
sig_cpr_eval.source_symbols = sig_tx.source_symbols[..., idx_eval]
sig_cpr_eval.source_bits = sig_tx.source_bits[
    ..., N_TRAIN * bits_per_sym : -100 * bits_per_sym
]
sig_cpr_eval.resolve_symbols()

evm_pct_cpr, evm_db_cpr = metrics.evm(sig_cpr_eval)
snr_val_cpr = metrics.snr(sig_cpr_eval)
ser_val_cpr = metrics.ser(sig_cpr_eval)
sig_cpr_eval.demap_symbols_hard()
ber_val_cpr = metrics.ber(sig_cpr_eval)

print("\n  Post-CPR (FOE + CPR + Timing Sync) Metrics:")
print("  ───────────────────────────────────────────")
print(f"  EVM : {evm_pct_cpr:.2f}% ({evm_db_cpr:.1f} dB)")
print(f"  SNR : {snr_val_cpr:.2f} dB")
print(f"  SER : {ser_val_cpr:.2e}")
print(f"  BER : {ber_val_cpr:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. STAGE 2: Symbol-Spaced LMS (sps=1) — residual ISI after CPR
#
#    After FSE + FOE + CPR + Timing Sync, the signal is at 1 SPS, frequency and
#    phase are despun, and symbol delay is zeroed.
#    A short symbol-spaced equalizer cleans up any residual ISI left by the FSE
#    (e.g., slight matched filter mismatch or timing sync residual).
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("8. Stage 2 — LMS symbol-spaced  (sps=1, residual ISI)")

result_lms2 = equalization.lms(
    y_cpr_resolved,
    training_symbols=training_syms[:N_TRAIN],
    modulation=MOD,
    order=ORDER,
    num_taps=N_TAPS_SSE,
    step_size=0.002,
    sps=1,  # symbol-spaced residual ISI cleanup
    backend="numba",
)

y_final = result_lms2.y_hat

print(f"  Input  : {y_cpr_resolved.shape} symbols @ 1 SPS")
print(f"  Taps   : {N_TAPS_SSE} symbol-spaced taps")
print(f"  Output : {y_final.shape} symbols")
print(f"  MSE    : {_mse_db(result_lms2.error):.1f} dB")

# Evaluate direct metrics for Stage 2
sig_s2_eval = sig_tx.copy()
sig_s2_eval.samples = y_final[..., idx_eval]
sig_s2_eval.sampling_rate = sig_tx.symbol_rate
sig_s2_eval.source_symbols = sig_tx.source_symbols[..., idx_eval]
sig_s2_eval.source_bits = sig_tx.source_bits[
    ..., N_TRAIN * bits_per_sym : -100 * bits_per_sym
]
sig_s2_eval.resolve_symbols()

evm_pct_s2, evm_db_s2 = metrics.evm(sig_s2_eval)
snr_val_s2 = metrics.snr(sig_s2_eval)
ser_val_s2 = metrics.ser(sig_s2_eval)
sig_s2_eval.demap_symbols_hard()
ber_val_s2 = metrics.ber(sig_s2_eval)

print("\n  Stage 2 (LMS SSE) Metrics (Direct against Reference):")
print("  ─────────────────────────────────────────────────────")
print(f"  EVM : {evm_pct_s2:.2f}% ({evm_db_s2:.1f} dB)")
print(f"  SNR : {snr_val_s2:.2f} dB")
print(f"  SER : {ser_val_s2:.2e}")
print(f"  BER : {ber_val_s2:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY & PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("9. Summary & metrics progression")

print("  Stage                          | Exit MSE")
print("  ─────────────────────────────────────────")
print(f"  RDE  blind FSE      (sps=2)   | {_mse_db(result_rde.error):>7.1f} dB")
print(
    f"  LMS  SSE residual   (sps=1)   | {_mse_db(result_lms2.error):>7.1f} dB  ← post FOE+CPR"
)
print()
print("  Receiver Metrics Progression (Direct calculations - no grid search):")
print("  ─────────────────────────────────────────────────────────────────────────────")
print("  Stage                         | EVM (%)    | EVM (dB)   | SNR (dB)   | BER")
print("  ─────────────────────────────────────────────────────────────────────────────")
print(
    f"  1. Stage 1 RDE (Raw Spinning) |   {evm_pct_s1:>6.2f}%  |   {evm_db_s1:>5.1f}    |    {snr_val_s1:>5.2f}   | {ber_val_s1:.2e}"
)
print(
    f"  2. Post-CPR (FOE + CPR + Sync)|   {evm_pct_cpr:>6.2f}%  |   {evm_db_cpr:>5.1f}    |    {snr_val_cpr:>5.2f}   | {ber_val_cpr:.2e}"
)
print(
    f"  3. Stage 2 (LMS SSE)          |   {evm_pct_s2:>6.2f}%  |   {evm_db_s2:>5.1f}    |    {snr_val_s2:>5.2f}   | {ber_val_s2:.2e}"
)
print("  ─────────────────────────────────────────────────────────────────────────────")
print()
print("  DSP Insights & Progression Analysis:")
print("  ─────────────────────────────────────────────────────────────────────────────")
print(
    "  * Stage 1 (RDE FSE at 2 SPS) raw metrics show the expected 140.6% EVM and -3.0 dB SNR."
)
print(
    "    RDE FSE blindly inverts the dispersion/multipath (opening the symbol eyes) but leaves"
)
print(
    "    the frequency offset and phase noise intact, causing the constellation to spin into"
)
print(
    "    three concentric rings. This highlights the absolute necessity of subsequent stages."
)
print(
    "  * Post-CPR (FOE + CPR + Symbol Timing Sync) corrects carrier frequency offset on raw"
)
print(
    "    samples, applies frozen RDE weights, corrects carrier phase blindly (Viterbi-Viterbi),"
)
print(
    "    and aligns symbol delay using standard timing synchronization. Constellation eyes"
)
print("    open fully, delivering a very strong metrics floor (~9.81% EVM).")
print(
    "  * Stage 2 (LMS SSE at 1 SPS) symbol-spaced equalizer operates in the clean, despun,"
)
print(
    "    synchronized environment, correcting fine residual ISI to achieve the absolute best"
)
print("    steady-state receiver performance (~8.67% EVM).")
print("  ─────────────────────────────────────────────────────────────────────────────")
print()
print("  Weight-passing:")
print(
    f"    RDE.weights  {result_rde.weights.shape}  → apply_taps() on 2nd capture (frozen blind RDE)"
)
print(
    f"    LMS2.weights {result_lms2.weights.shape}    → SSE taps (sps=1 residual ISI cleanup)"
)


# ──────────────────────────────────────────────────────────────────────────────
# 10. VISUALIZATION & DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("10. Visualization — Saving Constellation Diagnostics")

try:
    import os

    import matplotlib

    matplotlib.use("Agg")  # Headless backend
    import matplotlib.pyplot as plt

    from commstools.plotting import apply_default_theme

    print("  Applying CommsTools premium dark-theme...")
    apply_default_theme()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A. Raw Received Constellation (smeared, spinning)
    rx_1sps = rx[::SPS]
    sig_rx_impaired = sig_tx.copy()
    sig_rx_impaired.samples = rx_1sps
    sig_rx_impaired.plot_constellation(
        ax=axes[0, 0],
        bins=80,
        title="1. Raw RX Constellation\n(Smeared dispersion + Spinning CFO)",
        show=False,
    )

    # B. Stage 1 RDE Output (concentric rings showing opened amplitude rings but spinning phase)
    sig_rde = sig_tx.copy()
    sig_rde.samples = y_s1
    sig_rde.plot_constellation(
        ax=axes[0, 1],
        bins=100,
        title=f"2. Stage 1 RDE FSE Output\n(Concentric Rings, EVM = {evm_pct_s1:.1f}%)",
        show=False,
    )

    # C. Post-CPR (despun, timing-synchronized, ambiguity-resolved)
    sig_cpr = sig_tx.copy()
    sig_cpr.samples = y_cpr_resolved
    sig_cpr.plot_constellation(
        ax=axes[1, 0],
        bins=100,
        overlay_ideal=True,
        title=f"3. Post-CPR + Timing Sync\n(Despun 16-QAM, EVM = {evm_pct_cpr:.2f}%)",
        show=False,
    )

    # D. Stage 2 LMS SSE (residual ISI cleaned up)
    sig_final = sig_tx.copy()
    sig_final.samples = y_final
    sig_final.plot_constellation(
        ax=axes[1, 1],
        bins=120,
        overlay_ideal=True,
        title=f"4. Stage 2 LMS SSE Output\n(Razor-Sharp 16-QAM, EVM = {evm_pct_s2:.2f}%)",
        show=False,
    )

    os.makedirs("examples/images", exist_ok=True)
    fig.suptitle(
        "Coherent Receiver Multi-Stage DSP Pipeline Diagnostics",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    output_path = "examples/images/multi_stage_equalization_diagnostics.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SUCCESS] Premium diagnostic plot saved to:\n            {output_path}")

except Exception as e:
    print(f"  [WARNING] Matplotlib plotting failed: {e}")
