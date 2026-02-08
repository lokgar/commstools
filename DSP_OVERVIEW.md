# DSP Roadmap

should do something with generate_sequence in Frame??

---

## ✅ Completed

- **Bit-First Architecture**: `Signal.source_bits`, `Preamble` class
- **Backend Consistency**: `dispatch()` pattern, `xp.pi`, dtype propagation
- **Precision Control**: `dtype` parameter in mapping/generation functions
- **Es/N0 AWGN**: `add_awgn()` now uses standard Es/N0 with sps handling

---

## 🔴 Critical - Next Priority

### 1. Performance Metrics Module (`metrics.py`)

| Metric | Purpose |
|--------|---------|
| **EVM** | Error Vector Magnitude - primary signal quality metric |
| **SNR Estimation** | Estimate SNR from received symbols |
| **BER** | Bit Error Rate (pre/post-FEC) |
| **Q-factor** | Optical/RF quality metric |

### 2. Synchronization Utilities (`sync.py`)

- Symbol timing recovery (Gardner, Mueller-Muller)
- Frame/preamble detection via correlation
- Coarse/fine frequency offset estimation
- Barker/Zadoff-Chu sequence generation

### 3. Soft Demapping (`mapping.py` extension)

- **LLR calculation** - Essential for coded systems
- Symbol-level decision regions

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
