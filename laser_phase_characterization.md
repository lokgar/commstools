# Laser Phase Characterization: Drift, Phase Noise & Linewidth

Methods used to separate and quantify **carrier frequency instability** (drift) from
**laser phase noise** (linewidth) on coherent dual-pol captures, from the recovered
carrier-phase trajectory. Written as a reference for implementing these as library
analysis functions.

> **Headline result on the test captures** (16-QAM, 250 MBd, `T_sym = 4 ns`):
>
> | Laser | Freq drift (std / p-p) | Phase-noise Δφ/sym | Linewidth (raw / AWGN-corrected) |
> |---|---|---|---|
> | IDP | 83 kHz / ~0.98 MHz | 5.1° | ~320 kHz / ~115 kHz |
> | LXM | 10 kHz / ~0.25 MHz | 6.7° | ~550 kHz / ~210 kHz |
>
> IDP = frequency-unstable + narrow-line; LXM = frequency-stable + broad-line. The two
> impairments are **orthogonal** and must be measured separately.

---

## 1. Signal & phase model

The carrier phase on the recovered symbols decomposes into three additive parts:

```
φ(t) = 2π ∫ Δf(t) dt   +   φ_PN(t)   +   n_φ(t)
       └── drift ──┘       └─ PN ─┘     └ AWGN ┘
```

- **Drift** `2π∫Δf(t)dt` — a slow, *deterministic*, sustained phase ramp from carrier
  frequency offset / laser frequency instability. Low-frequency, large excursions.
- **Phase noise** `φ_PN` — a *stochastic Wiener (random-walk) process* from the laser's
  finite linewidth. Zero-mean **increments** with variance set by the Lorentzian
  linewidth `Δν`. No sustained slope.
- **AWGN angle noise** `n_φ` — per-symbol phase estimation error from additive noise;
  *not* a laser property, must be removed to avoid inflating the linewidth estimate.

These live at different timescales/frequencies, which is what lets us separate them.

---

## 2. Step 1 — Extract a clean carrier-phase trajectory (data-aided)

The estimate must be **bias-free** and **slip-free**. The method:

1. **Lock the equalizer** (joint BPS CPR) to obtain good ISI/polarization-demux taps
   `res.weights`. CPR is only needed *here* so the equalizer converges; it does not
   enter the measurement.
2. **Freeze + re-apply the taps WITHOUT CPR and WITHOUT adaptation** using
   [`apply_taps`](commstools/equalization.py#L7219):

   ```python
   y = apply_taps(rx.samples, res.weights, sps=2,
                  input_norm_factor=res.input_norm_factor)
   ```

   This removes ISI / polarization mixing but **leaves the carrier phase intact** —
   exactly what we want to measure.
3. **Data-aided phase** against the known transmitted symbols `d` (the payload is a
   repeating, known sequence tiled across the capture, so *all* symbols are known —
   not just the training prefix):

   ```python
   phi = np.unwrap(np.angle(y * np.conj(d)).astype(np.float64))   # per channel
   ```

   `y·conj(d)` cancels the data modulation (for QAM, `|d|²` is real-positive so the
   angle is preserved), leaving carrier phase + AWGN.

**Why data-aided, not BPS:** a BPS/feedforward estimate adds its own estimator noise and
can cycle-slip, both of which corrupt a linewidth estimate. Known symbols give the true
phase with only AWGN, and never slip. A constant offset or ±π/2 polarization ambiguity is
irrelevant — we only use the *time variation*.

**Pitfalls**
- **Channel pairing / pol swap:** the equalizer may map pol 0↔1. If `angle(y·conj(d))`
  looks random, swap `d`'s channels (pick the pairing with lower phase-error variance).
- **Unwrap in `float64`** (per the repo's CPR precision rule) — `float32` rounding can
  trigger spurious 2π slips at the wrap boundaries.
- The drift you measure here is **residual after the front-end FOE/drift correction**
  ([`correct_frequency_drift`](commstools/frequency.py)). That residual is what actually
  challenges the downstream DSP. To characterize the *raw* laser frequency wander instead,
  see §6.

---

## 3. Step 2 — Separate drift from phase noise (detrending)

Split `phi` into a slow drift component and a fast residual:

```python
W     = 512                                  # smoothing window (symbols)
drift = np.convolve(phi, np.ones(W)/W, mode="same")   # low-pass → drift
pn    = phi - drift                                   # high-pass → phase noise + AWGN
```

The boxcar moving average is a crude low-pass with first-null cutoff
`f_c ≈ 1/(W·T_sym)` (and ~half-power near `0.44/(W·T_sym)`). Content **below** `f_c` is
called "drift"; **above** is "phase noise". The cutoff choice is a modeling decision:

- `W` too large → fast drift leaks into `pn` and inflates the linewidth.
- `W` too small → genuine low-frequency phase noise gets absorbed into "drift".

A useful **diagnostic by-product**: `std(pn)` after a fixed `W`. A frequency-stable laser
gives a small residual (LXM: 4.8°); an unstable one leaks drift through the smoother
(IDP: 21.9°). That contrast alone flags frequency instability.

For a production implementation prefer a **proper low-pass** (e.g. Butterworth /
zero-phase `filtfilt`) or a **polynomial/spline detrend** over the boxcar, and expose the
cutoff frequency explicitly rather than a window length.

---

## 4. Step 3 — Frequency-instability (drift) metrics

Instantaneous residual frequency offset from the *smoothed* phase:

```python
df = np.diff(drift) / (2*np.pi*T_sym)        # Hz, one value per symbol step
drift_std_kHz = df.std()    / 1e3
drift_pp_MHz  = (df.max() - df.min()) / 1e6
```

Report **std** (typical wander) and **peak-to-peak** (worst-case spin the CPR must
follow). Relate to the BPS tracking limit: a residual `δf` advances the phase by
`2π·δf·T_sym` per symbol; over a BPS window of `K` symbols the intra-window rotation is
`2π·δf·T_sym·K`, which must stay below the QAM quarter-symmetry (`π/4`):

```
δf_max ≈ 1 / (8 · K · T_sym)
```

(e.g. `T_sym=4 ns`: `K=20 → 1.5 MHz`, `K=256 → 122 kHz`). This is *why* a drifting laser
needs a small BPS window and/or inline (joint) CPR, while a stable one tolerates a large
window.

---

## 5. Step 4 — Phase-noise / linewidth metrics

Wiener phase-noise model (Lorentzian lineshape, FWHM linewidth `Δν`): the phase increment
over time `τ` has variance `Var(δφ) = 2π·Δν·τ`. Over one symbol (`τ = T_sym`):

```
Δν = Var(Δφ_PN) / (2π · T_sym)
```

```python
dphi = np.diff(pn)                      # per-symbol phase increment
var_meas = np.var(dphi)
linewidth_raw = var_meas / (2*np.pi*T_sym)
```

### AWGN bias correction (important)

The data-aided phase carries AWGN angle noise with per-symbol variance
`σ_φ² ≈ 1/(2ρ)` (high-SNR approx, `ρ` = linear SNR). Independent symbols make the
increment variance:

```
Var(Δφ_meas) = Var(Δφ_PN) + 2·σ_φ²   →   subtract 2σ_φ² = 1/ρ
```

```python
rho = 10**(snr_dB/10)                   # linear SNR (from metrics.snr)
var_pn = max(var_meas - 1.0/rho, 0.0)   # AWGN-corrected increment variance
linewidth = var_pn / (2*np.pi*T_sym)
```

On the test data this took IDP from ~320→~115 kHz and LXM from ~550→~210 kHz. The
**absolute** value is sensitive to this correction; the **ratio between lasers** measured
identically at similar SNR is robust (~1.8× here).

**Refinement:** for QAM the AWGN angle noise is amplitude-dependent (`σ_φ ∝ 1/|d|`), so a
rigorous implementation should compute `σ_φ²` per symbol from `|d|` and the noise variance,
or restrict the estimate to the outer (high-SNR) constellation ring.

---

## 6. More rigorous alternatives (for a robust library implementation)

The moving-average split above is a quick estimate. Standard lab-grade methods:

- **FM-noise PSD + β-separation line** — the canonical way to extract linewidth.
  Compute the frequency-noise PSD `S_f(f)` from `df` (the differentiated phase), then
  integrate only the part of `S_f(f)` lying **above** the β-separation line
  `S_f(f) = (8 ln2 / π²)·f`; `Δν = √(8 ln2 · A)` where `A` is that integrated area.
  Cleanly separates the white-FM (linewidth) floor from `1/f` drift and technical noise.
- **Allan variance / deviation** of the instantaneous frequency `df` — characterizes
  drift vs white noise vs flicker by the slope of `σ²_Allan(τ)` versus averaging time `τ`.
- **Raw laser frequency wander (pre-equalizer):** instead of the residual after the
  front-end, track the **bias/pilot-tone frequency over sliding windows** on the
  coarse-corrected capture (this is exactly what the pipeline's per-segment estimator
  does via [`find_bias_tone`](commstools/frequency.py)). Gives the laser's own `Δf(t)`
  independent of the equalizer.
- **Spectral linewidth direct:** fit a Lorentzian to the phase-only PSD / the recovered
  carrier spectrum.

---

## 7. Suggested function API (future)

```python
def carrier_phase_trajectory(y_eq, ref_symbols, *, channel_pairing="auto") -> np.ndarray:
    """Data-aided unwrapped carrier phase from frozen-tap output + known symbols."""

def separate_drift_phase_noise(phi, *, cutoff_hz, symbol_rate):
    """Low-pass split → (drift_phase, pn_phase). Returns both components."""

def frequency_drift_metrics(drift_phase, symbol_rate) -> dict:
    """{'df_hz': array, 'std_hz', 'pp_hz'}."""

def linewidth(pn_phase, symbol_rate, *, snr_db=None) -> dict:
    """Wiener linewidth from increment variance, with optional AWGN correction.
    {'linewidth_hz', 'dphi_var', 'awgn_corrected': bool}."""

def fm_noise_psd_linewidth(phi, symbol_rate) -> dict:   # rigorous (§6)
    """β-separation-line linewidth from the FM-noise PSD."""
```

Keep all phase math in `float64`; accept either SISO `(N,)` or MIMO `(C, N)` with time on
the last axis (repo convention); make `symbol_rate`/`T_sym` and the drift/PN cutoff
explicit parameters.

---

## 8. End-to-end reference snippet

```python
# 1. front-end → rx.samples (2 sps), rx.source_symbols (full known, tiled)
# 2. lock equalizer (joint BPS) → res
# 3. freeze taps, no CPR:
y = apply_taps(rx.samples, res.weights, sps=2, input_norm_factor=res.input_norm_factor)
y = to_device(y, "cpu"); d = to_device(rx.source_symbols, "cpu")
n = min(y.shape[-1], d.shape[-1]); y, d = y[:, :n], d[:, :n]
T_sym = 1.0 / float(rx.symbol_rate)

for ch in range(y.shape[0]):
    phi   = np.unwrap(np.angle(y[ch] * np.conj(d[ch])).astype(np.float64))
    drift = np.convolve(phi, np.ones(512)/512, mode="same")
    pn    = phi - drift
    df    = np.diff(drift) / (2*np.pi*T_sym)
    dphi  = np.diff(pn)
    lw_raw = np.var(dphi) / (2*np.pi*T_sym)
    # AWGN-correct with measured SNR ρ:  lw = (var(dphi) - 1/ρ)/(2π T_sym)
```
