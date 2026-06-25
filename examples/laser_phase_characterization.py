#!/usr/bin/env python
"""Carrier-phase characterization: drift, phase noise & linewidth.

Synthetic dual-pol coherent pipeline that **injects known ground truth** —
a deterministic sinusoidal frequency wander, Wiener phase noise of a known
combined linewidth, and calibrated AWGN — then recovers it with the
:mod:`commstools.analysis` estimators and checks each against truth.

Methodology (see ``laser_characterization_in_coherent.md``)
-----------------------------------------------------------
  TX (16-QAM, 2 sps, RRC, dual-pol)
    └→ channel: multipath FIR + frequency wander Δf(t) + Wiener phase noise
       (shared LO) + AWGN
       └→ LMS-FSE *with inline PLL CPR* so the taps converge under the
          phase noise / drift  →  res.weights
          └→ apply_taps WITHOUT CPR  →  carrier phase left intact in y_eq
             └→ data-aided trajectory  φ = unwrap(angle(y·conj(d)))
                └→ separate drift / phase-noise  →  drift & linewidth metrics
                   └→ FM-noise PSD + β-line + white-FM floor, Allan deviation

The injected impairments are the *beat* of the (shared) LO and signal lasers,
so every recovered linewidth is the combined Δν and every drift is relative —
exactly what the equalizer/CPR must track.

Run
---
  uv run python examples/laser_phase_characterization.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from commstools import analysis, filtering, plotting, qam
from commstools.backend import to_device
from commstools.equalization import apply_taps, lms
from commstools.impairments import apply_awgn, apply_phase_noise

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
SPS = 2
SYMBOL_RATE = 32e9  # 32 GBaud
ORDER = 16
ROLLOFF = 0.1
N_SYM = 1 << 19  # 524 288 symbols → low-frequency PSD resolution
FILTER_SPAN = 10
NUM_STREAMS = 2  # dual-pol, shared LO

# Channel ground truth
ESN0_DB = 28.0
LASER_linewidth = 1.5e6  # combined Tx+Rx (shared LO trajectory)
DRIFT_AMP_HZ = 6.0e6  # peak of the sinusoidal frequency wander (dominates RW-FM)
DRIFT_PERIODS = 6.0  # wander cycles over the capture (≫1 ⇒ std → A/√2)

# Equalizer
NUM_TAPS = 21
STEP_SIZE = 1e-3
CPR_PLL_BW = 5e-3  # inline PLL bandwidth to converge taps under phase noise

# Analysis
DRIFT_cutoff = 3.0e6  # drift / phase-noise split (> wander rate, < Nyquist)
PSD_NPERSEG = 1 << 13  # Welch segment for the FM-noise PSD
FLOOR_F_MIN = 5.0e6  # white-FM floor band: above drift / β-crossing …
FLOOR_F_MAX = 2.0e8  # … and below the AWGN f² knee

SEED = 0xBEEF
rng = np.random.default_rng(SEED)


def _sep(title: str) -> None:
    print(f"\n{'─' * 66}\n  {title}\n{'─' * 66}")


def _fmt(x) -> str:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return f"{float(arr):.4g}"
    return "[" + ", ".join(f"{float(v):.4g}" for v in arr.ravel()) + "]"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Transmitter
# ──────────────────────────────────────────────────────────────────────────────
_sep("1. Transmitter — 16-QAM dual-pol")

tx = qam(
    num_symbols=N_SYM,
    sps=SPS,
    symbol_rate=SYMBOL_RATE,
    order=ORDER,
    pulse_shape="rrc",
    rrc_rolloff=ROLLOFF,
    filter_span=FILTER_SPAN,
    num_streams=NUM_STREAMS,
    seed=SEED,
).to("cpu")

fs = tx.sampling_rate
print(f"  {ORDER}-QAM | {NUM_STREAMS} pol | {N_SYM} sym | fs = {fs / 1e9:.1f} GSa/s")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Channel — multipath + frequency wander + phase noise + AWGN
# ──────────────────────────────────────────────────────────────────────────────
_sep("2. Channel impairments (known ground truth)")

rx = tx.copy()

# (a) light residual ISI for the FSE
h_ch = np.array([0.95, 0.0, 0.08 + 0.05j, 0.0, 0.03 - 0.02j], dtype=np.complex64)
rx = filtering.fir_filter(rx, h_ch)

# (b) deterministic sinusoidal frequency wander Δf(t), shared across pols.
#     φ_wander(t) = 2π ∫ Δf dt — applied at the sample rate.
n_samp = rx.samples.shape[-1]
t = np.arange(n_samp) / fs
f_wander = DRIFT_PERIODS / (n_samp / fs)  # Hz, ≈ DRIFT_PERIODS cycles / capture
df_t = DRIFT_AMP_HZ * np.sin(2.0 * np.pi * f_wander * t)
phase_wander = 2.0 * np.pi * np.cumsum(df_t) / fs
rx.samples = rx.samples * np.exp(1j * phase_wander).astype(np.complex64)[None, :]

# (c) Wiener phase noise — one shared-LO trajectory (combined linewidth)
rx.samples = apply_phase_noise(
    rx.samples,
    sampling_rate=fs,
    linewidth=LASER_linewidth,
    shared_lo=True,
    seed=SEED,
)

# (d) calibrated AWGN
rx.samples = apply_awgn(rx.samples, sps=SPS, esn0_db=ESN0_DB, seed=SEED)

# Ground-truth references (drift std of a multi-period sinusoid → A/√2)
drift_std_truth = DRIFT_AMP_HZ / np.sqrt(2.0)
noise_var_truth = 10.0 ** (-ESN0_DB / 10.0)  # σ_n² at unit symbol power
print(f"  Multipath    : |h| = {np.abs(h_ch).tolist()}")
print(
    f"  Freq wander  : ±{DRIFT_AMP_HZ / 1e6:.1f} MHz, {DRIFT_PERIODS:.0f} cycles "
    f"→ std ≈ {drift_std_truth / 1e3:.0f} kHz"
)
print(f"  Linewidth    : {LASER_linewidth / 1e6:.2f} MHz combined (shared LO)")
print(f"  Es/N0        : {ESN0_DB:.1f} dB  →  σ_n² = {noise_var_truth:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Equalizer — LMS-FSE with inline PLL CPR (to converge taps), then freeze
# ──────────────────────────────────────────────────────────────────────────────
_sep("3. LMS-FSE + inline PLL CPR  →  freeze taps (CPR off)")

source_symbols = np.asarray(to_device(rx.source_symbols, "cpu"), dtype=np.complex64)

res = lms(
    samples=rx.samples,
    training_symbols=source_symbols,
    modulation="qam",
    order=ORDER,
    num_taps=NUM_TAPS,
    step_size=STEP_SIZE,
    sps=SPS,
    cpr_type="pll",  # inline CPR only so DD-LMS converges under phase noise
    cpr_pll_bandwidth=CPR_PLL_BW,
    cpr_joint_channels=True,  # shared LO across pols
    backend="numba",
)
mse = float(np.mean(np.abs(np.asarray(res.error)[..., -1024:]) ** 2))
print(f"  Tail MSE     : {10 * np.log10(mse + 1e-30):.2f} dB")

# Re-apply the frozen taps WITHOUT CPR — leaves the carrier phase intact.
y_eq = apply_taps(
    rx.samples, res.weights, sps=SPS, input_norm_factor=res.input_norm_factor
)
y_eq = np.asarray(to_device(y_eq, "cpu"), dtype=np.complex64)
print(f"  y_eq         : {y_eq.shape}  (carrier phase preserved)")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Characterize the carrier phase
# ──────────────────────────────────────────────────────────────────────────────
_sep("4. Carrier-phase characterization")

report = analysis.characterize_carrier_phase(
    y_eq,
    source_symbols,
    SYMBOL_RATE,
    drift_cutoff=DRIFT_cutoff,
    noise_var=noise_var_truth,  # additive-noise variance only (see docs)
    nperseg=PSD_NPERSEG,
    f_min=FLOOR_F_MIN,
    f_max=FLOOR_F_MAX,
    channel_pairing="auto",
)

dm = report["drift_metrics"]
lw_inc = report["linewidth_increment"]
lw_beta = report["linewidth_beta"]


def _avg(x):  # average the two pols (shared LO ⇒ identical statistics)
    return float(np.mean(np.asarray(x, dtype=np.float64)))


print("  Frequency drift (residual after the equalizer):")
print(
    f"    std        : {_avg(dm['std']) / 1e3:8.1f} kHz   (injected wander std ≈ {drift_std_truth / 1e3:.0f} kHz"
)
print("                 + in-band random-walk FM ⇒ measured is slightly higher)")
print(f"    peak-peak  : {_avg(dm['pp']) / 1e6:8.2f} MHz")
print("  Linewidth (combined Δν_sig + Δν_LO):")
print(
    f"    increment (lag-slope, AWGN-free) : {_avg(lw_inc['linewidth']) / 1e6:6.3f} MHz   (truth = {LASER_linewidth / 1e6:.2f} MHz)"
)
print(
    f"    FM-PSD white-FM floor            : {_avg(lw_beta['linewidth_floor']) / 1e6:6.3f} MHz   (truth = {LASER_linewidth / 1e6:.2f} MHz)"
)
print(
    f"    β-separation area                : {_avg(lw_beta['linewidth']) / 1e6:6.3f} MHz   "
    f"(needs fine PSD resolution at high baud — see note)"
)
print(f"  Fitted AWGN intercept (2σ_φ²)      : {_fmt(lw_inc['awgn_var'])}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Diagnostic plots
# ──────────────────────────────────────────────────────────────────────────────
_sep("5. Diagnostic plots")

# The library's plotting helpers build the full 2x2 dashboard from the report
# dict (same panels as analysis.characterize_carrier_phase(debug_plot=True),
# but here we keep the figure to save it headlessly).
fig, _ = plotting.carrier_phase_characterization(
    report,
    symbol_rate=SYMBOL_RATE,
    drift_cutoff=DRIFT_cutoff,
    band=(FLOOR_F_MIN, FLOOR_F_MAX),
    floor=LASER_linewidth,  # draw the *injected* white-FM floor as reference
    amp_ref=DRIFT_AMP_HZ,
    show=False,
    title=(
        f"Laser carrier-phase characterization — {LASER_linewidth / 1e6:.2f} MHz Δν, "
        f"±{DRIFT_AMP_HZ / 1e6:.1f} MHz wander, {ESN0_DB:.0f} dB Es/N0"
    ),
)

os.makedirs("examples/images", exist_ok=True)
out = "examples/images/laser_phase_characterization.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Validation
# ──────────────────────────────────────────────────────────────────────────────
_sep("6. Validation against ground truth")

lw_floor = _avg(lw_beta["linewidth_floor"])
lw_increment = _avg(lw_inc["linewidth"])
drift_std = _avg(dm["std"])

ok = True
for name, est, truth, tol in [
    ("drift std", drift_std, drift_std_truth, 0.15),
    ("linewidth (increment)", lw_increment, LASER_linewidth, 0.20),
    ("linewidth (FM-PSD floor)", lw_floor, LASER_linewidth, 0.20),
]:
    rel = abs(est - truth) / truth
    status = "OK " if rel <= tol else "OFF"
    ok &= rel <= tol
    print(
        f"  [{status}] {name:26s}: {est / 1e3:8.1f} kHz vs {truth / 1e3:8.1f} kHz  ({rel * 100:4.1f}%)"
    )

_sep("Done" if ok else "Done (some estimates outside tolerance — tune params)")
print("  The white-FM floor (Δν = π·S_f) is the robust FM-PSD linewidth estimator")
print("  at high baud rates; the β-separation *area* needs much finer low-frequency")
print("  PSD resolution (longer captures) to populate the sub-MHz white-FM band.")
