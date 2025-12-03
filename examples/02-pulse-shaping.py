"""
Example 02: Pulse Shaping and Filtering

This example demonstrates advanced pulse shaping techniques:
- Understanding the expand (zero-insertion) operation
- Exploring different filter types for pulse shaping
- Understanding filter parameters (span, bandwidth, rolloff)
- Comparing filter frequency responses
- Visualizing eye diagrams to assess signal quality

Learning objectives:
- Learn how pulse shaping works (expand + filter)
- Understand filter design parameters
- Assess signal quality using eye diagrams
- Compare trade-offs between different pulse shapes
"""

import numpy as np
import matplotlib.pyplot as plt
from commstools import set_backend
from commstools.dsp import filtering, sequences, mapping, multirate
from commstools.waveforms import ook

set_backend("numpy")

print("=" * 70)
print("EXAMPLE 02: Pulse Shaping and Filtering")
print("=" * 70)

# =============================================================================
# Step 1: Understanding the Expand Operation
# =============================================================================
print("\n[Step 1] Understanding Zero-Insertion (Expand)")
print("-" * 70)

# Create a simple symbol sequence
symbols = np.array([0, 1, 1, 0, 1])
sps = 4

# Expand: insert zeros between symbols
expanded = multirate.expand(symbols, sps)

print(f"Original symbols: {symbols}")
print(f"After expand (sps={sps}): {expanded}")
print(f"\nNotice: Each symbol is followed by {sps - 1} zeros")
print("This creates spectral images that need to be filtered out!")

# =============================================================================
# Step 2: Filter Tap Design
# =============================================================================
print("\n[Step 2] Designing Filter Taps")
print("-" * 70)

sps = 8  # Use higher sps for better visualization

# Generate different types of filter taps
taps_boxcar = filtering.boxcar_taps(sps)
taps_sinc = filtering.sinc_taps(sps, bandwidth=0.5, span=6)
taps_gaussian = filtering.gaussian_taps(sps, bt=0.3, span=4)
taps_rrc = filtering.rrc_taps(sps, rolloff=0.35, span=6)

print(f"Boxcar taps: {len(taps_boxcar)} coefficients")
print(f"Sinc taps: {len(taps_sinc)} coefficients")
print(f"Gaussian taps: {len(taps_gaussian)} coefficients")
print(f"RRC taps: {len(taps_rrc)} coefficients")

# Visualize filter taps
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for idx, (taps, label) in enumerate(
    [
        (taps_boxcar, "Boxcar"),
        (taps_sinc, "Sinc (BW=0.5)"),
        (taps_gaussian, "Gaussian (BT=0.3)"),
        (taps_rrc, "RRC (α=0.35)"),
    ]
):
    t = np.arange(len(taps)) / sps
    t = t - t[len(t) // 2]  # Center the time axis
    axs[idx].stem(t, taps, basefmt=" ")
    axs[idx].set_xlabel("Time (symbols)")
    axs[idx].set_ylabel("Amplitude")
    axs[idx].set_title(label, fontsize=11, fontweight="bold")
    axs[idx].grid(True, alpha=0.3)
    axs[idx].axhline(y=0, color="k", linewidth=0.5)

plt.tight_layout()
plt.savefig("02_filter_taps.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved filter taps plot: 02_filter_taps.png")

# =============================================================================
# Step 3: Filter Frequency Responses
# =============================================================================
print("\n[Step 3] Analyzing Filter Frequency Responses")
print("-" * 70)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for idx, (taps, label) in enumerate(
    [
        (taps_boxcar, "Boxcar"),
        (taps_sinc, "Sinc"),
        (taps_gaussian, "Gaussian"),
        (taps_rrc, "RRC"),
    ]
):
    # Compute frequency response
    freq_response = np.fft.fft(taps, n=2048)
    freqs = np.fft.fftfreq(2048, d=1 / sps)
    freqs_shifted = np.fft.fftshift(freqs)
    response_shifted = np.fft.fftshift(freq_response)

    # Plot magnitude in dB
    magnitude_db = 20 * np.log10(np.abs(response_shifted) + 1e-12)
    axs[idx].plot(freqs_shifted, magnitude_db, linewidth=1.5)
    axs[idx].set_xlabel("Normalized Frequency")
    axs[idx].set_ylabel("Magnitude (dB)")
    axs[idx].set_title(label, fontsize=11, fontweight="bold")
    axs[idx].set_ylim(-60, 20)
    axs[idx].grid(True, alpha=0.3)
    axs[idx].axvline(x=0, color="r", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("02_frequency_response.png", dpi=150, bbox_inches="tight")
print("✓ Saved frequency response plot: 02_frequency_response.png")

# =============================================================================
# Step 4: Effect of Filter Parameters
# =============================================================================
print("\n[Step 4] Exploring Filter Parameter Effects")
print("-" * 70)

# Generate signals with different parameters
sampling_rate = 100e3
symbol_rate = 10e3
sps = int(sampling_rate / symbol_rate)
bits = sequences.prbs(200, order=7, seed=123)

# Gaussian filters with different BT values
print("\nGaussian filters with varying BT:")
bt_values = [0.2, 0.3, 0.5, 1.0]
gaussian_signals = []
for bt in bt_values:
    sig = ook(bits, sampling_rate=sampling_rate, sps=sps, pulse_shape="gaussian", bt=bt)
    gaussian_signals.append((sig, f"BT={bt}"))
    print(f"  BT={bt}: bandwidth ∝ {bt}/T")

# RRC filters with different rolloff values
print("\nRRC filters with varying rolloff (α):")
rolloff_values = [0.1, 0.35, 0.5, 1.0]
rrc_signals = []
for rolloff in rolloff_values:
    sig = ook(
        bits, sampling_rate=sampling_rate, sps=sps, pulse_shape="rrc", rolloff=rolloff
    )
    rrc_signals.append((sig, f"α={rolloff}"))
    excess_bw = rolloff * 100
    print(f"  α={rolloff}: {excess_bw:.0f}% excess bandwidth")

# =============================================================================
# Step 5: Eye Diagrams for Signal Quality Assessment
# =============================================================================
print("\n[Step 5] Eye Diagrams for Quality Assessment")
print("-" * 70)

# Generate longer sequence for better eye diagram
bits_long = sequences.prbs(500, order=7, seed=456)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

configs = [
    ("none", {}, "No Pulse Shaping"),
    ("boxcar", {}, "Boxcar"),
    ("gaussian", {"bt": 0.3}, "Gaussian (BT=0.3)"),
    ("rrc", {"rolloff": 0.35}, "RRC (α=0.35)"),
]

for idx, (pulse_shape, kwargs, label) in enumerate(configs):
    sig = ook(
        bits_long,
        sampling_rate=sampling_rate,
        sps=sps,
        pulse_shape=pulse_shape,
        **kwargs,
    )
    ax = axs[idx // 2, idx % 2]
    sig.plot_eye(ax=ax, num_symbols=2, plot_type="line")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (symbol periods)")
    ax.set_ylabel("Amplitude")

plt.tight_layout()
plt.savefig("02_eye_diagrams.png", dpi=150, bbox_inches="tight")
print("✓ Saved eye diagrams: 02_eye_diagrams.png")

# =============================================================================
# Step 6: Key Learning Points
# =============================================================================
print("\n[Step 6] Key Learning Points")
print("-" * 70)
print("""
1. PULSE SHAPING PROCESS:
   - Expand: Insert zeros between symbols
   - Filter: Remove spectral images and shape the pulse
   - Result: Smooth analog waveform from digital symbols

2. FILTER PARAMETERS:
   
   GAUSSIAN (bt):
   - Bandwidth-Time product
   - Lower BT → narrower bandwidth, more ISI
   - Higher BT → wider bandwidth, less ISI
   - Typical: 0.2-0.5 for GMSK
   
   RRC (rolloff α):
   - Excess bandwidth factor (0 to 1)
   - α=0: Ideal brick-wall (sinc pulse)
   - α=1: 100% excess bandwidth
   - Typical: 0.2-0.5 for most systems
   
   SPAN:
   - Filter length in symbol periods
   - Longer span → better frequency response
   - Trade-off: computation vs. performance

3. EYE DIAGRAMS:
   - Overlaid symbol traces show signal quality
   - Wide eye opening → low ISI, good SNR margin
   - Closed eye → high ISI, poor signal quality
   - Used for system optimization and debugging

4. TRADE-OFFS:
   - Bandwidth vs. ISI
   - Filter complexity vs. performance
   - Spectral efficiency vs. implementation cost
""")

print("\n" + "=" * 70)
print("Example complete! Check the generated PNG files.")
print("=" * 70)

# plt.show()
