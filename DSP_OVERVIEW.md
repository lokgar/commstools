# DSP Pre-Implementation Overview

Critical analysis of recommended additions **before** implementing actual DSP algorithms.

---

## 🔴 Critical Additions

### 1. Performance Metrics Module (`metrics.py`)

| Metric | Purpose |
|--------|---------|
| **EVM** | Error Vector Magnitude - primary signal quality metric |
| **SNR Estimation** | Estimate SNR from received symbols |
| **BER** | Bit Error Rate (pre/post-FEC) |
| **Q-factor** | Optical/RF quality metric |
| **OSNR** | Optical SNR for coherent systems |

> Without metrics, you cannot evaluate any DSP algorithm's performance.

### 2. Synchronization Utilities (`sync.py`)

Core building blocks for any practical receiver:
- **Symbol timing recovery** (Gardner, Mueller-Muller)
- **Frame/preamble detection** via correlation
- **Coarse/fine frequency offset estimation**
- **Barker/CAZAC sequence generation**

### 3. Channel Coding Module (`coding.py`) ⚠️ NEW

| Component | Purpose |
|-----------|---------|
| **FEC Encoders** | Convolutional, LDPC, Turbo, Polar codes |
| **FEC Decoders** | Viterbi, Belief Propagation, BCJR |
| **Interleaving** | Block, convolutional, random interleavers |
| **Scrambling** | PRBS-based data whitening |
| **CRC** | Error detection for ARQ/HARQ |

> Channel coding requires working with bits BEFORE symbol mapping, which has implications for the current architecture (see below).

---

## 🟡 Important Additions

### 4. Enhanced Channel Models (`impairments.py`)

| Impairment | Domain |
|------------|--------|
| **Chromatic Dispersion (CD)** | Optical |
| **Polarization Mode Dispersion (PMD)** | Optical |
| **Phase noise** (laser linewidth, oscillator jitter) | Both |
| **IQ imbalance** (gain/phase mismatch) | RF |
| **Nonlinear effects** (PA compression, fiber Kerr) | Both |

### 5. Soft Demapping (`mapping.py` extension)

- **LLR calculation** - Essential for coded systems
- **Symbol-level decision regions**
- **Adaptive constellation** support (geometric shaping)

---

## 🟢 Nice-to-Have Additions

### 6. Statistics/Analysis Module (`analysis.py`)

- Autocorrelation for timing/CD estimation
- Cyclostationary analysis
- Probability density estimation
- Histogram-based statistics

### 7. Data Structures

- **BatchedSignal** class for vectorized SNR sweeps
- **SampleBuffer** for streaming with overlap-save
- Multi-format bit containers

---

## ⚠️ Current Implementation Review

### Bits vs Symbols Design Consideration

**Current approach**: Symbol-first design
- `random_symbols()` generates symbols directly from random indices
- `Signal.generate()` works primarily with symbols
- Bit representation is derived via `demap_symbols()` after the fact

**Implications for channel coding**:

| Aspect | Current Design | Channel Coding Requirement |
|--------|----------------|---------------------------|
| Data flow | `random_indices → symbols → bits` | `bits → encode → interleave → map → symbols` |
| RNG control | Clean (symbol-level seeding) | Must seed at bit level for reproducibility |
| Bits storage | Optional (`demap_source_symbols()`) | Required as primary representation |

**Recommended changes**:
1. Add `random_bits()` → `encode()` → `map_bits()` pipeline
2. Store source bits in `Signal.source_bits` (primary), derive symbols
3. Add `BitSequence` container with FEC-aware grouping methods
4. Separate uncoded/coded symbol generation paths

### Signal Normalization

**Current**: Signals normalized to `max_amplitude=1` consistently via `shape_pulse()`.

**Implication**: Constellation overlay must scale to match (fixed in `constellation()`).

### Backend Consistency

**Current**: Good - consistent `dispatch()` pattern, `xp.pi` usage, dtype propagation.

**Gap**: Some hardcoded NumPy calls in plotting (acceptable for CPU-only matplotlib).

### Frame Structure (`SingleCarrierFrame`)

**Current design**:
- Frame operates at **symbol level**: `payload_len` = number of symbols
- Preamble, pilots, guard intervals all specified in symbols
- Payload symbols generated via `random_symbols()` (index-based RNG)

**Implications for channel coding**:

| Frame Component | Current | FEC Requirement |
|-----------------|---------|-----------------|
| Payload | Random symbols (seeded) | Encoded bits → mapped symbols |
| Payload length | In symbols | Must also track bit count (varies with code rate) |
| Source data | Not stored | Need `source_bits` for BER calculation |

**Structural gaps for FEC integration**:

1. **No bit-level granularity**: Frame length specified in symbols, but FEC operates on bits
   - Code rate R = k/n means k info bits → n coded bits → n/log₂(M) symbols
   - Need: `info_bits` → `coded_bits` → `symbols` chain with length tracking

2. **No codeword boundaries**: Current frame is flat symbol array
   - LDPC/Turbo need codeword-aligned blocks
   - Need: `CodeBlock` abstraction with (info_len, coded_len, codeword_size)

3. **Preamble as raw array**: User must provide pre-mapped sequence
   - OK for known sequences (Barker, Zadoff-Chu)
   - For coded preambles, need integration with coding pipeline

**Recommended Frame evolution**:

```
SingleCarrierFrame (current, uncoded)
       │
       ├── Add: source_bits, coded_bits properties
       │
       └── New: CodedFrame(SingleCarrierFrame)
              │
              ├── fec_encoder: Encoder
              ├── interleaver: Interleaver  
              ├── info_bits_per_codeword: int
              └── num_codewords: int
```

**Migration path**:
1. Keep `SingleCarrierFrame` for uncoded simulations (backward compatible)
2. Add `CodedFrame` subclass with FEC-aware generation
3. Add `source_bits` property that returns demapped payload (for BER)
4. Later: Add OFDM frame variants

---

## Architecture Recommendations

| Current Gap | Recommendation |
|-------------|----------------|
| No batch processing dimension | Add optional `batch` axis to `Signal` |
| Missing precision control for metrics | Add `dtype` parameter to metric functions |
| No streaming support | Implement overlap-save wrapper for long signals |
| Symbol-first architecture | Add bits-first pipeline for coded systems |
| No bit container | Add `BitSequence` class with FEC block awareness |

---

## References

- *Digital Signal Processing for Coherent Transceivers* (Faruk & Savory, 2017)
- *Digital Coherent Optical Receivers* (Savory, 2010)
- *Fundamentals of Coherent Optical Fiber Communications* (Kikuchi, 2016)

