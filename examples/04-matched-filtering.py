"""
Example 04: Matched Filtering and Detection

This example demonstrates matched filtering concepts:
- Understanding matched filter theory
- Implementing matched filters for different pulse shapes
- Exploring SNR improvement
- Comparing transmit and receive pulse shaping
- Eye diagram analysis with matched filtering

Learning objectives:
- Learn why matched filtering is optimal
- Understand the matched filter relationship to pulse shape
- Measure SNR improvement
- Design complete transmit/receive chains
- Assess detection performance
"""

import numpy as np
import matplotlib.pyplot as plt
from commstools import set_backend
from commstools.dsp import filtering, sequences
from commstools.waveforms import ook

set_backend("numpy")

print("=" * 70)
print("EXAMPLE 04: Matched Filtering and Detection")
print("=" * 70)

# =============================================================================
# Step 1: Understanding Matched Filter
# =============================================================================
print("\n[Step 1] Matched Filter Theory")
print("-" * 70)

# Create a simple pulse shape
sps = 8
pulse_taps = filtering.rrc_taps(sps, rolloff=0.35, span=4)

# Matched filter is time-reversed conjugate
matched_taps = np.conj(pulse_taps[::-1])

print(f"Pulse shape: {len(pulse_taps)} taps")
print(f"Matched filter: {len(matched_taps)} taps")
print("\nMatched filter = time-reversed conjugate of pulse shape")
print("For real pulses: matched filter = time-reversed pulse")

# Visualize pulse and matched filter
fig, axs = plt.subplots(3, 1, figsize=(12, 9))

t = np.arange(len(pulse_taps)) / sps - (len(pulse_taps) - 1) / (2 * sps)

axs[0].plot(t, pulse_taps, linewidth=2)
axs[0].set_title("Transmit Pulse Shape (RRC)", fontsize=11, fontweight="bold")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True, alpha=0.3)
axs[0].axhline(y=0, color="k", linewidth=0.5)

axs[1].plot(t, matched_taps, linewidth=2, color="C1")
axs[1].set_title("Matched Filter (Time-Reversed)", fontsize=11, fontweight="bold")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True, alpha=0.3)
axs[1].axhline(y=0, color="k", linewidth=0.5)

# Autocorrelation (convolution of pulse with matched filter)
autocorr = np.convolve(pulse_taps, matched_taps, mode="same")
t_auto = np.arange(len(autocorr)) / sps - (len(autocorr) - 1) / (2 * sps)

axs[2].plot(t_auto, autocorr, linewidth=2, color="C2")
axs[2].set_title(
    "Pulse Autocorrelation (After Matched Filtering)", fontsize=11, fontweight="bold"
)
axs[2].set_xlabel("Time (symbols)")
axs[2].set_ylabel("Amplitude")
axs[2].grid(True, alpha=0.3)
axs[2].axhline(y=0, color="k", linewidth=0.5)
axs[2].axvline(x=0, color="r", linestyle="--", alpha=0.5, label="Sampling instant")
axs[2].legend()

plt.tight_layout()
plt.savefig("04_matched_filter_concept.png", dpi=150, bbox_inches="tight")
print("✓ Saved matched filter concept plot: 04_matched_filter_concept.png")

# =============================================================================
# Step 2: SNR Improvement with Matched Filtering
# =============================================================================
print("\n[Step 2] SNR Improvement Demonstration")
print("-" * 70)

# Generate a clean signal
sampling_rate = 100e3
symbol_rate = 10e3
sps = int(sampling_rate / symbol_rate)
bits = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 0])

# Create signal with RRC pulse shaping
signal = ook(
    bits, sampling_rate=sampling_rate, sps=sps, pulse_shape="rrc", rolloff=0.35
)

# Add noise
SNR_dB = 5  # Low SNR to demonstrate improvement
noise_power = 10 ** (-SNR_dB / 10)
noise = np.random.randn(len(signal.samples)) * np.sqrt(noise_power / 2)
noisy_signal = signal.samples + noise

# Apply matched filter
pulse_taps_rrc = filtering.rrc_taps(sps, rolloff=0.35, span=8)
filtered_signal = filtering.matched_filter(noisy_signal, pulse_taps_rrc, mode="same")

# Calculate SNR improvement
signal_power_before = np.var(noisy_signal)
signal_power_after = np.var(filtered_signal)
noise_power_estimate = noise_power  # Theoretical

# Theoretical SNR improvement is the processing gain
processing_gain = len(pulse_taps_rrc)
theoretical_improvement_dB = 10 * np.log10(processing_gain)

print(f"Input SNR: {SNR_dB:.1f} dB")
print(f"Processing gain: {processing_gain} samples")
print(f"Theoretical SNR improvement: {theoretical_improvement_dB:.1f} dB")
print("\nMatched filtering concentrates signal energy at sampling instants!")

# Visualize the improvement
fig, axs = plt.subplots(3, 1, figsize=(12, 9))

samples_to_plot = 200

axs[0].plot(signal.samples[:samples_to_plot].real, linewidth=1.5, label="Clean signal")
axs[0].set_title("Original Clean Signal", fontsize=11, fontweight="bold")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True, alpha=0.3)
axs[0].legend()

axs[1].plot(noisy_signal[:samples_to_plot].real, linewidth=1, alpha=0.7)
axs[1].set_title(f"Noisy Signal (SNR = {SNR_dB} dB)", fontsize=11, fontweight="bold")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True, alpha=0.3)

axs[2].plot(filtered_signal[:samples_to_plot].real, linewidth=1.5, color="C2")
axs[2].set_title(
    "After Matched Filtering (SNR Improved)", fontsize=11, fontweight="bold"
)
axs[2].set_xlabel("Sample")
axs[2].set_ylabel("Amplitude")
axs[2].grid(True, alpha=0.3)

# Mark sampling instants
for i in range(0, samples_to_plot, sps):
    for ax in axs:
        ax.axvline(x=i, color="r", linestyle="--", alpha=0.2)

plt.tight_layout()
plt.savefig("04_snr_improvement.png", dpi=150, bbox_inches="tight")
print("✓ Saved SNR improvement plot: 04_snr_improvement.png")

# =============================================================================
# Step 3: Matched Filtering for Different Pulse Shapes
# =============================================================================
print("\n[Step 3] Matched Filtering for Different Pulse Shapes")
print("-" * 70)

bits_test = sequences.prbs(50, order=7, seed=789)

pulse_configs = [
    ("boxcar", {}),
    ("gaussian", {"bt": 0.3}),
    ("rrc", {"rolloff": 0.35}),
    ("sinc", {"bandwidth": 0.5}),
]

fig, axs = plt.subplots(len(pulse_configs), 2, figsize=(14, 10))

for idx, (pulse_shape, kwargs) in enumerate(pulse_configs):
    # Generate signal
    sig = ook(
        bits_test,
        sampling_rate=sampling_rate,
        sps=sps,
        pulse_shape=pulse_shape,
        **kwargs,
    )

    # Add noise
    noise = np.random.randn(len(sig.samples)) * np.sqrt(noise_power / 2)
    noisy = sig.samples + noise

    # Apply matched filter
    filtered = filtering.matched_filter_pulse(noisy, pulse_shape, sps, **kwargs)

    # Before matched filtering
    axs[idx, 0].plot(noisy[:150].real, linewidth=1, alpha=0.7)
    axs[idx, 0].set_title(f"{pulse_shape.upper()}: Before Matched Filter", fontsize=10)
    axs[idx, 0].set_ylabel("Amplitude")
    axs[idx, 0].grid(True, alpha=0.3)

    # After matched filtering
    axs[idx, 1].plot(filtered[:150].real, linewidth=1.5, color="C2")
    axs[idx, 1].set_title(f"{pulse_shape.upper()}: After Matched Filter", fontsize=10)
    axs[idx, 1].grid(True, alpha=0.3)

    print(f"✓ {pulse_shape.upper()}: Matched filtering applied")

axs[-1, 0].set_xlabel("Sample")
axs[-1, 1].set_xlabel("Sample")

plt.tight_layout()
plt.savefig("04_matched_filtering_comparison.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved matched filtering comparison: 04_matched_filtering_comparison.png")

# =============================================================================
# Step 4: Eye Diagrams with Matched Filtering
# =============================================================================
print("\n[Step 4] Eye Diagrams: Matched vs. Unmatched")
print("-" * 70)

# Generate longer sequence
bits_long = sequences.prbs(300, order=7, seed=111)
sig_tx = ook(
    bits_long, sampling_rate=sampling_rate, sps=sps, pulse_shape="rrc", rolloff=0.35
)

# Add moderate noise
SNR_eye = 10  # dB
noise_power_eye = 10 ** (-SNR_eye / 10)
noise_eye = (
    np.random.randn(len(sig_tx.samples)) + 1j * np.random.randn(len(sig_tx.samples))
) * np.sqrt(noise_power_eye / 2)
sig_noisy = sig_tx.samples + noise_eye

# Apply matched filter
sig_matched = filtering.matched_filter_pulse(sig_noisy, "rrc", sps, rolloff=0.35)

# Create signal objects for eye diagram plotting
from commstools import Signal

sig_before = Signal(
    samples=sig_noisy,
    sampling_rate=sampling_rate,
    symbol_rate=symbol_rate,
    modulation_format="OOK",
)

sig_after = Signal(
    samples=sig_matched,
    sampling_rate=sampling_rate,
    symbol_rate=symbol_rate,
    modulation_format="OOK",
)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

sig_before.plot_eye(ax=axs[0], num_symbols=2, plot_type="line")
axs[0].set_title("Eye Diagram: Before Matched Filter", fontsize=11, fontweight="bold")

sig_after.plot_eye(ax=axs[1], num_symbols=2, plot_type="line")
axs[1].set_title("Eye Diagram: After Matched Filter", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("04_eye_diagram_matched.png", dpi=150, bbox_inches="tight")
print("✓ Saved eye diagram comparison: 04_eye_diagram_matched.png")
print("\nNotice: Eye opening is larger after matched filtering!")

# =============================================================================
# Step 5: Complete TX/RX Chain
# =============================================================================
print("\n[Step 5] Complete Transmit/Receive Chain")
print("-" * 70)

print("""
TYPICAL COMMUNICATION SYSTEM:

Transmitter:
  bits → map_to_symbols → pulse_shaping → channel
           ↓                    ↓              ↓
        Symbols         TX pulse shape    Add noise

Receiver:
  samples → matched_filter → sample @ T → decision → bits
                 ↓              ↓             ↓
         RX matched filter  Timing sync   Threshold

Key Points:
  - TX pulse + RX matched filter = overall Nyquist pulse
  - For RRC: TX RRC * RX RRC = Raised Cosine (zero ISI)
  - Matched filter maximizes SNR at sampling instant
  - Timing must be synchronized for optimal sampling
""")

# =============================================================================
# Step 6: Key Learning Points
# =============================================================================
print("\n[Step 6] Key Learning Points")
print("-" * 70)
print("""
1. MATCHED FILTER PRINCIPLE:
   - Optimal linear filter for detecting known signal in AWGN
   - Maximizes SNR at the sampling instant
   - Matched filter = time-reversed conjugate of pulse shape
   - For real pulses: matched filter = time-reversed pulse

2. SNR IMPROVEMENT:
   - Processing gain = filter length (in samples)
   - Concentrates signal energy at decision points
   - Critical for reliable detection in noisy channels
   - Trade-off: filter length vs. delay

3. PULSE SHAPE CONSIDERATIONS:
   - RRC: Most common, used in pairs (TX + RX = RC)
   - Gaussian: Constant envelope, used in GMSK
   - Sinc: Theoretical ideal, impractical due to infinite length
   - Boxcar: Simple but poor spectral efficiency

4. PRACTICAL IMPLEMENTATION:
   - Matched filtering at receiver is standard practice
   - Can split filtering between TX and RX
   - Square-root raised cosine (RRC) is common split
   - Overall pulse must control ISI

5. SYSTEM DESIGN:
   - Choose pulse shape based on:
     * Bandwidth constraints
     * Spectral mask requirements
     * Peak-to-average power ratio
     * Implementation complexity
   - Matched filtering is nearly universal in digital comms

6. EYE DIAGRAM INTERPRETATION:
   - Open eye = good SNR, low ISI
   - Closed eye = poor SNR or high ISI
   - Eye width → timing margin
   - Eye height → voltage/SNR margin
   - Use for system optimization
""")

print("\n" + "=" * 70)
print("Example complete! Check the generated PNG files.")
print("=" * 70)

# plt.show()
