# DSP Roadmap

---

## ✅ Completed

- **Bit-First Architecture**: `Signal.source_bits`, `Preamble` class
- **Backend Consistency**: `dispatch()` pattern, `xp.pi`, dtype propagation
- **Precision Control**: `dtype` parameter in mapping/generation functions
- **Es/N0 AWGN**: `add_awgn()` now uses standard Es/N0 with sps handling
- **Performance Metrics** (`metrics.py`): EVM, SNR estimation, BER, Q-factor
- **Synchronization Utilities** (`sync.py`): Correlation, frame detection, Barker/ZC sequences
- **Soft Demapping** (`mapping.py`): LLR calculation with max-log and exact methods

---

## 🔴 Critical - Next Priority

### 1. Symbol Timing Recovery

- Gardner timing error detector
- Mueller-Muller algorithm
- Interpolation-based correction

### 2. Frequency Offset Estimation

- Coarse estimation (data-aided)
- Fine estimation (blind/DD)
- Carrier phase recovery

---

## 🟡 Important - Phase 2

### 4. Channel Coding Module (`coding.py`)

| Component | Purpose |
|-----------|---------|
| **FEC Encoders** | Convolutional, LDPC, Turbo, Polar codes |
| **FEC Decoders** | Viterbi, Belief Propagation, BCJR |
| **Interleaving** | Block, convolutional, random interleavers |
| **CRC** | Error detection for ARQ/HARQ |

**Data structure needed**: `BitSequence` with FEC-aware grouping

### 5. Enhanced Channel Models (`impairments.py`)

| Impairment | Domain |
|------------|--------|
| **Phase noise** (laser linewidth, oscillator jitter) | Both |
| **IQ imbalance** (gain/phase mismatch) | RF |
| **Chromatic Dispersion (CD)** | Optical |
| **Nonlinear effects** (PA compression, fiber Kerr) | Both |

---

## 🟢 Future Additions

### 6. Statistics/Analysis Module (`analysis.py`)

- Autocorrelation for timing/CD estimation
- Cyclostationary analysis
- Probability density estimation

### 7. Extended Data Structures

- **BatchedSignal** for vectorized SNR sweeps
- **CodedFrame** subclass with FEC integration
- OFDM frame variants

---

## References

- *Digital Signal Processing for Coherent Transceivers* (Faruk & Savory, 2017)
- *Digital Coherent Optical Receivers* (Savory, 2010)
