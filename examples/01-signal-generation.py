"""
Example 01: Signal Generation Basics

This example demonstrates the fundamentals of digital signal generation:
- Creating binary sequences (PRBS)
- Mapping bits to symbols
- Generating OOK (On-Off Keying) waveforms
- Understanding samples per symbol (sps)
- Visualizing signals in time and frequency domains

Learning objectives:
- Understand the relationship between symbol rate, sampling rate, and sps
- Learn how digital information is represented as analog waveforms
- Explore the spectral characteristics of different pulse shapes
"""

import numpy as np
import matplotlib.pyplot as plt
from commstools import set_backend, Signal
from commstools.dsp.sequences import prbs
from commstools.waveforms import ook

# Use NumPy backend for this example
set_backend("numpy")

print("=" * 70)
print("EXAMPLE 01: Signal Generation Basics")
print("=" * 70)

# =============================================================================
# Step 1: Generate Binary Data
# =============================================================================
print("\n[Step 1] Generating Binary Data")
print("-" * 70)

# Generate a Pseudo-Random Binary Sequence (PRBS)
# PRBS-7 has a period of 2^7 - 1 = 127 bits
bits = prbs(length=100, order=7, seed=42)

print(f"Generated {len(bits)} bits using PRBS-7")
print(f"First 20 bits: {bits[:20]}")
print(f"Bit statistics: {np.sum(bits)} ones, {len(bits) - np.sum(bits)} zeros")

# =============================================================================
# Step 2: Understanding Sampling Parameters
# =============================================================================
print("\n[Step 2] Understanding Sampling Parameters")
print("-" * 70)

# Define system parameters
symbol_rate = 10e3  # 10 kHz symbol rate (10,000 symbols per second)
sampling_rate = 100e3  # 100 kHz sampling rate (100,000 samples per second)
sps = int(sampling_rate / symbol_rate)  # Samples per symbol

print(f"Symbol rate: {symbol_rate / 1e3:.1f} kHz")
print(f"Sampling rate: {sampling_rate / 1e3:.1f} kHz")
print(f"Samples per symbol (sps): {sps}")
print(f"\nThis means each symbol is represented by {sps} samples")
print(f"Total signal duration: {len(bits) / symbol_rate * 1000:.2f} ms")

# =============================================================================
# Step 3: Generate Signals with Different Pulse Shapes
# =============================================================================
print("\n[Step 3] Generating Signals with Different Pulse Shapes")
print("-" * 70)

# 1. Impulse (no pulse shaping) - just zeros inserted
sig_impulse = ook(bits, sampling_rate=sampling_rate, sps=sps, pulse_shape="none")
print(f"✓ Impulse signal: {len(sig_impulse.samples)} samples")

# 2. Boxcar (rectangular) pulse
sig_boxcar = ook(bits, sampling_rate=sampling_rate, sps=sps, pulse_shape="boxcar")
print(f"✓ Boxcar signal: {len(sig_boxcar.samples)} samples")

# 3. Gaussian pulse (smooth transitions)
sig_gaussian = ook(
    bits, sampling_rate=sampling_rate, sps=sps, pulse_shape="gaussian", bt=0.3, span=4
)
print(f"✓ Gaussian signal: {len(sig_gaussian.samples)} samples")

# 4. Root Raised Cosine (RRC) - optimal for ISI control
sig_rrc = ook(
    bits,
    sampling_rate=sampling_rate,
    sps=sps,
    pulse_shape="rrc",
    rolloff=0.35,
    span=8,
)
print(f"✓ RRC signal: {len(sig_rrc.samples)} samples")

# =============================================================================
# Step 4: Visualize Signals in Time Domain
# =============================================================================
print("\n[Step 4] Visualizing Time Domain Signals")
print("-" * 70)

fig, axs = plt.subplots(4, 1, figsize=(12, 10))

# Plot a short segment to see the pulse shapes clearly
samples_to_plot = 200

for idx, (sig, label) in enumerate(
    [
        (sig_impulse, "Impulse (No Pulse Shaping)"),
        (sig_boxcar, "Boxcar (Rectangular)"),
        (sig_gaussian, "Gaussian (BT=0.3)"),
        (sig_rrc, "RRC (α=0.35)"),
    ]
):
    t = np.arange(samples_to_plot) / sampling_rate * 1e6  # Convert to microseconds
    axs[idx].plot(t, sig.samples[:samples_to_plot].real, linewidth=1.5)
    axs[idx].set_ylabel("Amplitude")
    axs[idx].set_title(label, fontsize=11, fontweight="bold")
    axs[idx].grid(True, alpha=0.3)

    # Mark symbol boundaries
    for i in range(0, samples_to_plot, sps):
        axs[idx].axvline(
            x=i / sampling_rate * 1e6, color="red", linestyle="--", alpha=0.3
        )

axs[-1].set_xlabel("Time (μs)")
plt.tight_layout()
plt.savefig("01_time_domain.png", dpi=150, bbox_inches="tight")
print("✓ Saved time domain plot: 01_time_domain.png")

# =============================================================================
# Step 5: Visualize Power Spectral Density (PSD)
# =============================================================================
print("\n[Step 5] Visualizing Frequency Domain (PSD)")
print("-" * 70)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for idx, (sig, label) in enumerate(
    [
        (sig_impulse, "Impulse"),
        (sig_boxcar, "Boxcar"),
        (sig_gaussian, "Gaussian"),
        (sig_rrc, "RRC"),
    ]
):
    sig.plot_psd(ax=axs[idx], NFFT=2048)
    axs[idx].set_title(label, fontsize=11, fontweight="bold")
    axs[idx].set_ylim(-80, -20)
    axs[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_frequency_domain.png", dpi=150, bbox_inches="tight")
print("✓ Saved frequency domain plot: 01_frequency_domain.png")

# =============================================================================
# Step 6: Key Observations and Learning Points
# =============================================================================
print("\n[Step 6] Key Learning Points")
print("-" * 70)
print("""
1. IMPULSE SIGNAL:
   - Simplest form: just zeros inserted between symbols
   - Wide spectrum (maximum bandwidth)
   - Sharp transitions (not practical for real systems)

2. BOXCAR (RECTANGULAR) SIGNAL:
   - Each symbol period is filled with constant value
   - Spectrum has sinc shape with sidelobes
   - Still has sharp transitions at symbol boundaries

3. GAUSSIAN SIGNAL:
   - Smooth transitions reduce spectral sidelobes
   - BT parameter controls bandwidth-time product
   - Lower BT = narrower bandwidth but more ISI

4. ROOT RAISED COSINE (RRC):
   - Optimal pulse shape for ISI-free communication
   - Rolloff factor (α) controls excess bandwidth
   - Used in most modern communication systems

5. SAMPLES PER SYMBOL (sps):
   - Higher sps → better representation of analog waveform
   - Typical values: 2-16 for simulations
   - Trade-off between accuracy and computation
""")

print("\n" + "=" * 70)
print("Example complete! Check the generated PNG files.")
print("=" * 70)

# Show plots if running interactively
# plt.show()
