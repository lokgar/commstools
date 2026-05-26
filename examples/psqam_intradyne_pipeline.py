#!/usr/bin/env python
"""Coherent DSP pipeline for Probabilistically-Shaped QAM (PS-QAM).

End-to-end synthetic pipeline that takes PS-QAM through the standard
coherent receive chain — IQ-imbalance compensation, fractionally-spaced
LMS, blind-phase-search CPR, hard demap, information-theoretic metrics —
to verify that the library's scaling is consistent for shaped
constellations.

The Tx generates symbols on the normalised ``{s_m}`` grid with average
power ``E_PS = Σ P(s_m) |s_m|² < 1``; :func:`shape_pulse` then normalises
the transmit waveform to unit symbol power, so the underlying
constellation in ``Signal.samples`` is implicitly ``{s_m / √E_PS}``.
Every library primitive used below accepts ``pmf=`` (or auto-forwards
``Signal.ps_pmf``) so the receive chain sees a consistent constellation
scale end-to-end without manual rescaling.

Pipeline
--------
  TX (psqam, 2 sps, RRC)
    └→ channel: multipath FIR + PMD + IQ imbalance + phase noise + AWGN
       └→ Löwdin IQ compensation
          └→ butterfly LMS-FSE  (sps=2, pmf=ps_pmf)
             └→ blind-phase-search CPR  (pmf=ps_pmf)
                └→ resolve symbols + phase ambiguity (auto-forwards pmf)
                   └→ hard demap (auto-forwards pmf)
                      └→ metrics: EVM, SNR, BER, SER, GMI, MI

Library APIs that accept ``pmf=`` or auto-forward ``Signal.ps_pmf``
-------------------------------------------------------------------
*  :func:`commstools.equalization.lms` (and ``rls`` / ``block_lms``)
*  :func:`commstools.recovery.recover_carrier_phase_bps`
*  :func:`commstools.recovery.resolve_phase_ambiguity`
*  :func:`commstools.mapping.demap_symbols_hard`
*  :func:`commstools.metrics.evm` (blind mode) / :func:`metrics.ser`
*  :meth:`commstools.core.Signal.demap_symbols_hard` /
   :meth:`Signal.resolve_phase_ambiguity` / :meth:`Signal.evm` /
   :meth:`Signal.ser` / :meth:`Signal.ber` / :meth:`Signal.mi` /
   :meth:`Signal.gmi`

Run
---
  uv run python examples/psqam_intradyne_pipeline.py
"""

import numpy as np

from commstools import Signal
from commstools.backend import to_device
from commstools.equalization import lms
from commstools.impairments import (
    apply_awgn,
    apply_iq_imbalance,
    apply_pmd,
    compensate_iq_imbalance_lowdin,
)
from commstools.mapping import gray_constellation
from commstools.recovery import (
    correct_carrier_phase,
    recover_carrier_phase_bps,
)

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────

SPS = 2
SYMBOL_RATE = 32e9                  # 32 GBaud
ORDER = 64
ENTROPY = 5.4                       # bits/symbol  (cap = log2(64) = 6.0)
ROLLOFF = 0.1
N_SYM = 1 << 16                     # 65 536 symbols
FILTER_SPAN = 10
NUM_STREAMS = 2                     # dual-pol

# Channel
ESN0_DB = 22.0
IQ_AMP_DB = 0.4
IQ_PHASE_DEG = 3.0
PMD_DGD_S = 2e-12
PMD_THETA = np.pi / 5
LASER_LINEWIDTH_HZ = 50e3           # combined Tx+Rx phase noise

# Equalizer
NUM_TAPS = 31
NUM_TRAIN = 1 << 13                 # 8 192 training symbols
STEP_SIZE = 1e-3

# CPR
BPS_TEST_PHASES = 128
BPS_BLOCK_SIZE = 32

SEED = 0xC0DE
rng = np.random.default_rng(SEED)


def _sep(title: str) -> None:
    print(f"\n{'─' * 64}\n  {title}\n{'─' * 64}")


def _fmt(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return f"{float(arr):.4g}"
    return "[" + ", ".join(f"{float(v):.4g}" for v in arr.ravel()) + "]"


# ──────────────────────────────────────────────────────────────────────────────
# 1. TRANSMITTER  ─  PS-QAM at unit symbol power
# ──────────────────────────────────────────────────────────────────────────────

_sep("1. Transmitter — PS-QAM")

tx = Signal.psqam(
    num_symbols=N_SYM,
    sps=SPS,
    symbol_rate=SYMBOL_RATE,
    order=ORDER,
    entropy=ENTROPY,
    pulse_shape="rrc",
    rrc_rolloff=ROLLOFF,
    filter_span=FILTER_SPAN,
    num_streams=NUM_STREAMS,
    seed=SEED,
).to("cpu")

pmf = np.asarray(tx.ps_pmf, dtype=np.float64)
const_unit = gray_constellation("qam", ORDER)
E_PS = float(np.dot(pmf, np.abs(const_unit) ** 2))

print(f"  Order        : {ORDER}-QAM")
print(f"  Entropy H(X) : {ENTROPY:.3f} bits/symbol  (max = {np.log2(ORDER):.1f})")
print(f"  E_PS         : {E_PS:.4f}  ({-10 * np.log10(E_PS):.2f} dB below uniform QAM)")
print(f"  Streams      : {NUM_STREAMS}  (dual-pol)")
print(f"  Tx samples   : {tx.samples.shape}  dtype={tx.samples.dtype}")
print(f"  Symbol rate  : {SYMBOL_RATE / 1e9:.1f} GBaud  |  fs = {SPS * SYMBOL_RATE / 1e9:.1f} GSa/s")

source_symbols = np.asarray(to_device(tx.source_symbols, "cpu"), dtype=np.complex64)
source_bits = np.asarray(to_device(tx.source_bits, "cpu"))


# ──────────────────────────────────────────────────────────────────────────────
# 2. CHANNEL  ─  multipath FIR + PMD + IQ imbalance + phase noise + AWGN
# ──────────────────────────────────────────────────────────────────────────────

_sep("2. Channel impairments")

rx_samples = np.ascontiguousarray(to_device(tx.samples, "cpu"), dtype=np.complex64)

# Per-pol 3-tap multipath (residual ISI for the FSE).
h_ch = np.array([0.92, 0.0, 0.10 + 0.06j, 0.0, 0.04 - 0.02j], dtype=np.complex64)
rx_samples = np.stack(
    [np.convolve(rx_samples[c], h_ch, mode="same") for c in range(NUM_STREAMS)]
).astype(np.complex64)

# Polarisation mode dispersion (frequency-dependent SOP rotation + DGD).
rx_samples = np.asarray(
    apply_pmd(rx_samples, tx.sampling_rate, dgd=PMD_DGD_S, theta=PMD_THETA)
).astype(np.complex64)

# Laser phase noise (Wiener random walk, applied to both pols identically — they
# share an LO).  BPS-CPR downstream tracks this.
fs = tx.sampling_rate
phase_walk = np.cumsum(
    rng.standard_normal(rx_samples.shape[-1])
    * np.sqrt(2 * np.pi * LASER_LINEWIDTH_HZ / fs)
)
rx_samples = rx_samples * np.exp(1j * phase_walk).astype(np.complex64)[None, :]

# Receiver IQ imbalance.
rx_samples = np.asarray(
    apply_iq_imbalance(rx_samples, IQ_AMP_DB, IQ_PHASE_DEG)
).astype(np.complex64)

# AWGN calibrated against the actual sample power (PS-QAM safe).
rx_samples = np.asarray(
    apply_awgn(rx_samples, sps=SPS, esn0_db=ESN0_DB, seed=SEED)
).astype(np.complex64)

print(f"  Multipath FIR: |h| = {np.abs(h_ch).tolist()}")
print(f"  PMD          : DGD={PMD_DGD_S * 1e12:.1f} ps, θ={PMD_THETA:.3f} rad")
print(f"  Phase noise  : {LASER_LINEWIDTH_HZ / 1e3:.0f} kHz combined linewidth")
print(f"  IQ imbalance : {IQ_AMP_DB:.2f} dB amp, {IQ_PHASE_DEG:.1f}° phase")
print(f"  Es/N0        : {ESN0_DB:.1f} dB")


# ──────────────────────────────────────────────────────────────────────────────
# 3. RECEIVER FRONT-END  ─  IQ compensation
# ──────────────────────────────────────────────────────────────────────────────

_sep("3. Receiver front-end")

rx = Signal(
    samples=rx_samples,
    sampling_rate=tx.sampling_rate,
    symbol_rate=tx.symbol_rate,
    mod_scheme="PS-QAM",
    mod_order=ORDER,
    ps_pmf=tx.ps_pmf,
    pulse_shape="rrc",
    rrc_rolloff=ROLLOFF,
    filter_span=FILTER_SPAN,
).to("cpu")

rx.samples = compensate_iq_imbalance_lowdin(rx.samples)
rx.source_bits = source_bits.copy()
rx.source_symbols = source_symbols.copy()

print(f"  IQ comp      : Löwdin orthogonalisation")
print(f"  RX samples   : {rx.samples.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. EQUALIZER  ─  Butterfly LMS-FSE  (sps=2, pmf=ps_pmf)
#
# The FSE absorbs Tx pulse-shaping, channel multipath and PMD in one shot.
# ``pmf=tx.ps_pmf`` scales the DD slicer constellation to ``{s_m / √E_PS}``
# so post-training decision-directed adaptation works on PS-QAM.
# Two passes warm-started from each other tighten the taps.
# ──────────────────────────────────────────────────────────────────────────────

_sep("4. Butterfly LMS-FSE  (pmf=ps_pmf)")

eq_kwargs = dict(
    modulation="qam",
    order=ORDER,
    num_taps=NUM_TAPS,
    step_size=STEP_SIZE,
    sps=SPS,
    pmf=tx.ps_pmf,                  # ← PS-QAM scale correction
    backend="numba",
)

res = lms(
    samples=rx.samples,
    training_symbols=rx.source_symbols[:, :NUM_TRAIN],
    **eq_kwargs,
)
mse1 = float(np.mean(np.abs(np.asarray(res.error)[..., -512:]) ** 2))
print(f"  pass 1 (cold) : MSE_tail = {10 * np.log10(mse1 + 1e-30):>6.2f} dB")

res = lms(
    samples=rx.samples,
    training_symbols=rx.source_symbols[:, :NUM_TRAIN],
    w_init=res.weights,
    input_norm_factor=res.input_norm_factor,
    **eq_kwargs,
)
mse2 = float(np.mean(np.abs(np.asarray(res.error)[..., -512:]) ** 2))
print(f"  pass 2 (warm) : MSE_tail = {10 * np.log10(mse2 + 1e-30):>6.2f} dB")
print(f"  y_hat shape   : {res.y_hat.shape}")
print(f"  y_hat power   : {_fmt([float(np.mean(np.abs(np.asarray(res.y_hat)[c])**2)) for c in range(NUM_STREAMS)])}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. CPR  ─  Standalone Blind Phase Search  (pmf=ps_pmf)
#
# Tracks the phase-noise walk left after the equaliser.  The metric uses
# nearest-neighbour distance to ``gray_constellation`` rescaled to
# ``{s_m / √E_PS}`` so the search is unbiased for shaped constellations.
# ──────────────────────────────────────────────────────────────────────────────

_sep("5. BPS carrier phase recovery  (pmf=ps_pmf)")

y_eq = np.asarray(res.y_hat, dtype=np.complex64)
phi_bps = recover_carrier_phase_bps(
    y_eq,
    modulation="qam",
    order=ORDER,
    num_test_phases=BPS_TEST_PHASES,
    block_size=BPS_BLOCK_SIZE,
    joint_channels=True,            # shared LO across pols
    pmf=tx.ps_pmf,                  # ← PS-QAM scale correction
)
y_cpr = correct_carrier_phase(y_eq, phi_bps)

print(f"  Test phases  : {BPS_TEST_PHASES}")
print(f"  Block size   : {BPS_BLOCK_SIZE} symbols")
print(f"  Phase trajectory range : "
      f"{float(np.asarray(phi_bps).min()):+.2f} → {float(np.asarray(phi_bps).max()):+.2f} rad")


# ──────────────────────────────────────────────────────────────────────────────
# 6. RESOLVE + HARD DEMAP  (Signal methods auto-forward ps_pmf)
# ──────────────────────────────────────────────────────────────────────────────

_sep("6. Phase-ambiguity resolution + hard demap")

eq = rx.copy()
eq.samples = y_cpr
eq.sampling_rate = rx.symbol_rate             # 1 SPS
eq.source_symbols = rx.source_symbols
eq.source_bits = rx.source_bits

eq.resolve_symbols()                          # avg-power=1 on {s_m / √E_PS}
eq.resolve_phase_ambiguity()                  # auto-forwards self.ps_pmf
eq.demap_symbols_hard()                       # auto-forwards self.ps_pmf


# ──────────────────────────────────────────────────────────────────────────────
# 7. METRICS  (all PS-QAM aware via auto-forwarded self.ps_pmf)
# ──────────────────────────────────────────────────────────────────────────────

_sep("7. Metrics")

discard = 2 * NUM_TRAIN

evm_pct, evm_db = eq.evm(num_train_symbols=discard, mode="data_aided")
snr_db = eq.snr(num_train_symbols=discard)
ser_val = eq.ser(num_train_symbols=discard)
ber_val = eq.ber(num_train_symbols=discard)

# AWGN-equivalent noise variance from per-channel SNR (referenced to {s_m}).
snr_lin = 10 ** (np.asarray(snr_db, dtype=np.float64) / 10.0)
noise_var = float(1.0 / np.mean(snr_lin))
mi_val = eq.mi(noise_var=noise_var)
gmi_val = eq.gmi(noise_var=noise_var, method="maxlog")

print(f"  EVM (data-aided) : {_fmt(evm_pct)} %   ({_fmt(evm_db)} dB)")
print(f"  SNR              : {_fmt(snr_db)} dB")
print(f"  SER              : {_fmt(ser_val)}")
print(f"  BER              : {_fmt(ber_val)}")
print(f"  MI               : {mi_val:.4f} b/cu   (cap H(X) = {ENTROPY:.3f})")
print(f"  GMI              : {gmi_val:.4f} b/cu")

assert mi_val <= ENTROPY + 1e-3, f"MI {mi_val:.3f} > H(X) {ENTROPY:.3f}"

_sep("Done")
print("  Tweak ESN0_DB / ENTROPY / PMD_DGD_S / LASER_LINEWIDTH_HZ to explore the")
print("  PS-QAM operating envelope.  GMI saturates at H(X) — the right rate")
print("  metric for coded-system comparisons.")
