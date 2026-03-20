#!/usr/bin/env python
"""Multi-stage equalization: FSE → FOE → CPR → SSE

Demonstrates the standard coherent DSP chain for residual ISI correction,
showing how to pass weights between stages and what the signal looks like
at each processing step.

Pipeline
--------
  TX (2 SPS, RRC)
    └→ channel: FIR multipath + frequency offset + AWGN
       └→ [Stage 1]  CMA  blind FSE  (sps=2) → 1 SPS symbols
          └→ [Stage 1b] LMS warm-started from CMA weights (sps=2)
             └→ [FOE]  mth-power offset estimation + correction
                └→ [CPR]  Viterbi-Viterbi phase recovery
                   └→ [Stage 2] LMS symbol-spaced (sps=1) residual ISI
                      └→ metrics (EVM, BER)

Weight-passing patterns shown
------------------------------
  1. CMA → LMS (w_init):  same sps=2, warm-start speeds up LMS DA phase
  2. apply_taps():         re-use frozen Stage-1 taps on a new capture
  3. Stage 2 sps=1:        LMS runs at symbol rate after FOE+CPR

Run
---
  uv run python examples/multi_stage_equalization.py
"""

import numpy as np

from commstools import Signal
from commstools import equalization, sync
from commstools.backend import to_device

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────

SPS          = 2
SYMBOL_RATE  = 10e9        # 10 GBaud
FS           = SPS * SYMBOL_RATE  # 20 GSa/s
N_SYM        = 8192
MOD          = "qam"
ORDER        = 16
ROLLOFF      = 0.2
N_TAPS_FSE   = 31          # Stage 1 & 1b: fractionally-spaced filter length
N_TAPS_SSE   = 7           # Stage 2: short symbol-spaced filter (residual ISI)
N_TRAIN      = 512         # DA pilot symbols for LMS stages

SNR_DB       = 22.0
FO_TRUE_HZ   = 1.2e6       # 1.2 MHz frequency offset

rng = np.random.default_rng(42)


def _print_sep(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def _mse_db(error, window=200):
    """Steady-state MSE over the last ``window`` symbols."""
    e = np.asarray(error)
    mse = float(np.mean(np.abs(e[..., -window:]) ** 2))
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

training_syms = to_device(sig_tx.source_symbols, "cpu").astype(np.complex64)  # (N_SYM,) at 1 SPS

print(f"  TX samples : {sig_tx.samples.shape}  (dtype={sig_tx.samples.dtype})")
print(f"  TX symbols : {training_syms.shape}  {ORDER}-{MOD.upper()}, SPS={SPS}")
print(f"  Symbol rate: {SYMBOL_RATE/1e9:.0f} GBaud  |  FS: {FS/1e9:.0f} GSa/s")


# ──────────────────────────────────────────────────────────────────────────────
# 2. CHANNEL: FIR multipath + frequency offset + AWGN
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("2. Channel")

# 3-tap multipath (ISI spans ±1 symbol at 2 SPS → 2 tap spacings)
h_ch = np.array([0.85, 0.0, 0.12 + 0.07j, 0.0, 0.05 - 0.03j], dtype=np.complex64)
tx_np = to_device(sig_tx.samples, "cpu").astype(np.complex64)
rx = np.convolve(tx_np, h_ch, mode="same")

# Frequency offset
t = np.arange(len(rx), dtype=np.float64) / FS
rx = rx * np.exp(1j * 2.0 * np.pi * FO_TRUE_HZ * t).astype(np.complex64)

# AWGN
sig_pwr = float(np.mean(np.abs(rx) ** 2))
noise_var = sig_pwr / (10 ** (SNR_DB / 10))
noise = np.sqrt(noise_var / 2) * (
    rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape)
).astype(np.complex64)
rx = (rx + noise).astype(np.complex64)

print(f"  Channel    : FIR {h_ch}")
print(f"  FO true    : {FO_TRUE_HZ/1e6:.2f} MHz")
print(f"  SNR        : {SNR_DB} dB")
print(f"  RX samples : {rx.shape}  (dtype={rx.dtype})")


# ──────────────────────────────────────────────────────────────────────────────
# 3. STAGE 1: CMA blind FSE (sps=2)
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("3. Stage 1 — CMA blind FSE  (sps=2)")

result_cma = equalization.cma(
    rx,
    num_taps=N_TAPS_FSE,
    step_size=3e-4,
    modulation=MOD,
    order=ORDER,
    sps=SPS,
    backend="numba",
)

y_s1 = result_cma.y_hat  # (N_SYM,) at 1 SPS

print(f"  Input  : {rx.shape} samples  @ {SPS} SPS")
print(f"  Output : {y_s1.shape} symbols @ 1 SPS")
print(f"  Weights: {result_cma.weights.shape}  (C, C, num_taps)")
print(f"  MSE    : {_mse_db(result_cma.error):.1f} dB")

# Note: EVM vs tx symbols is not computed here — CMA leaves a residual
# M-fold phase ambiguity (resolved in Stage 1b by DA training symbols).


# ──────────────────────────────────────────────────────────────────────────────
# 4. STAGE 1b: LMS refinement warm-started from CMA weights (sps=2)
#
#    w_init = result_cma.weights  ← weight hand-off
#    CMA converges blindly; LMS then tightens with known pilots.
#    The warm-start skips the initial LMS pull-in transient.
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("4. Stage 1b — LMS refinement  (sps=2, w_init=CMA.weights)")

result_lms1 = equalization.lms(
    rx,
    training_symbols=training_syms[:N_TRAIN],   # DA phase
    modulation=MOD,
    order=ORDER,
    num_taps=N_TAPS_FSE,
    step_size=0.004,
    sps=SPS,
    w_init=result_cma.weights,                  # ← weight hand-off from CMA
    backend="numba",
)

y_s1b = result_lms1.y_hat  # (N_SYM,) at 1 SPS

print(f"  Input      : {rx.shape} samples  @ {SPS} SPS  (same as Stage 1)")
print(f"  w_init     : CMA result.weights  {result_cma.weights.shape}")
print(f"  DA pilots  : {N_TRAIN} symbols  →  DD for remainder")
print(f"  Output     : {y_s1b.shape} symbols @ 1 SPS")
print(f"  MSE Stage1 : {_mse_db(result_cma.error):.1f} dB  (CMA, no pilots)")
print(f"  MSE Stage1b: {_mse_db(result_lms1.error):.1f} dB  (LMS, warm-started)")


# ──────────────────────────────────────────────────────────────────────────────
# 5. BONUS: apply_taps() — re-use frozen Stage-1b weights on a new capture
#
#    Useful when you have a reference burst that trains the taps and then
#    want to apply those frozen taps to a subsequent data burst without
#    re-running adaptation.
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("5. apply_taps() — frozen weights on a second capture")

# Simulate a second noisy capture (same channel, different noise realisation)
noise2 = np.sqrt(noise_var / 2) * (
    rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape)
).astype(np.complex64)
rx2 = np.convolve(tx_np, h_ch, mode="same")
t2 = np.arange(len(rx2), dtype=np.float64) / FS
rx2 = rx2 * np.exp(1j * 2.0 * np.pi * FO_TRUE_HZ * t2).astype(np.complex64) + noise2

y_frozen = equalization.apply_taps(
    rx2,
    result_lms1.weights,   # ← frozen taps from the training burst
    sps=SPS,
    normalize=True,
)
print(f"  New capture  : {rx2.shape} samples")
print(f"  Frozen taps  : result_lms1.weights  {result_lms1.weights.shape}")
print(f"  Output       : {y_frozen.shape} symbols @ 1 SPS  (no adaptation)")


# ──────────────────────────────────────────────────────────────────────────────
# 6. FOE: frequency offset estimation + correction
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("6. FOE — mth-power law  (on 1 SPS symbols from Stage 1b)")

fo_est = sync.estimate_frequency_offset_mth_power(
    y_s1b,
    fs=SYMBOL_RATE,        # symbols are at 1 SPS → fs = symbol_rate
    modulation=MOD,
    order=ORDER,
)

y_foe = sync.correct_frequency_offset(y_s1b, fo_est, fs=SYMBOL_RATE)

print(f"  True FO : {FO_TRUE_HZ/1e6:.4f} MHz")
print(f"  Est  FO : {fo_est/1e6:.4f} MHz")
print(f"  Error   : {(fo_est - FO_TRUE_HZ)/1e3:.2f} kHz")
print(f"  Output  : {y_foe.shape} symbols (FO-corrected)")


# ──────────────────────────────────────────────────────────────────────────────
# 7. CPR: Viterbi-Viterbi carrier phase recovery
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("7. CPR — Viterbi-Viterbi")

phase_vv = sync.recover_carrier_phase_viterbi_viterbi(
    y_foe,
    modulation=MOD,
    order=ORDER,
    block_size=32,
)

y_cpr = sync.correct_carrier_phase(y_foe, phase_vv)

print("  Block size : 32 symbols")
print(f"  Output     : {y_cpr.shape} phase-corrected symbols")


# ──────────────────────────────────────────────────────────────────────────────
# 8. STAGE 2: Symbol-Spaced LMS (sps=1) — residual ISI after CPR
#
#    After FSE + FOE + CPR, the signal is at 1 SPS and phase is aligned.
#    A short symbol-spaced equalizer cleans up any residual ISI left by the
#    FSE (e.g. imperfect matched filter, slight chromatic dispersion tail).
#
#    Key: sps=1 is now accepted — no upsampling back to 2 SPS needed.
# ──────────────────────────────────────────────────────────────────────────────

_print_sep("8. Stage 2 — LMS symbol-spaced  (sps=1, residual ISI)")

result_lms2 = equalization.lms(
    y_cpr,
    training_symbols=training_syms[:N_TRAIN],
    modulation=MOD,
    order=ORDER,
    num_taps=N_TAPS_SSE,       # short filter: only residual ISI spans 1–2 symbols
    step_size=0.002,
    sps=1,                     # ← symbol-spaced — enabled by relaxed sps check
    backend="numba",
)

y_final = result_lms2.y_hat

print(f"  Input  : {y_cpr.shape} symbols @ 1 SPS  (after FOE+CPR)")
print(f"  Taps   : {N_TAPS_SSE}  (short; residual ISI only)")
print(f"  Output : {y_final.shape} symbols")
print(f"  MSE    : {_mse_db(result_lms2.error):.1f} dB")


# ──────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
# MSE progression tells the story: each stage reduces residual distortion.
# Final EVM vs. training_syms is not computed here because VV CPR leaves a
# 4-fold (π/2) phase ambiguity for QAM — comparing to tx symbols directly
# gives a meaningless number unless the ambiguity is resolved by pilots or
# decision-rotation search.  Use pilot-aided CPR if you need absolute EVM.

_print_sep("9. MSE progression summary")

print("  Stage                          | Exit MSE")
print("  ─────────────────────────────────────────")
print(f"  CMA  blind FSE      (sps=2)   | {_mse_db(result_cma.error):>7.1f} dB")
print(f"  LMS  warm-start FSE (sps=2)   | {_mse_db(result_lms1.error):>7.1f} dB  ← DA pilot gain")
print(f"  LMS  SSE residual   (sps=1)   | {_mse_db(result_lms2.error):>7.1f} dB  ← post FOE+CPR")
print()
print("  Weight-passing:")
print(f"    CMA.weights  {result_cma.weights.shape}  → w_init for LMS Stage 1b  (warm-start)")
print(f"    LMS1b.weights {result_lms1.weights.shape}  → apply_taps() on 2nd capture (frozen)")
print(f"    LMS2.weights {result_lms2.weights.shape}    → short SSE taps (sps=1 residual ISI)")
