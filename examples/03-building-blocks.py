"""
Example 03: DSP Building Blocks

This example demonstrates low-level DSP building blocks:
- Expand (zero-insertion)
- Interpolate (upsample with filtering)
- Decimate (downsample with anti-aliasing)
- Resample (rational rate conversion)
- Practical applications and use cases

Learning objectives:
- Understand multirate signal processing operations
- Learn when and how to use each building block
- Explore rate conversion for flexible system design
- Understand aliasing and imaging concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from commstools import set_backend
from commstools.dsp import filters

set_backend("numpy")

print("=" * 70)
print("EXAMPLE 03: DSP Building Blocks")
print("=" * 70)

# =============================================================================
# Step 1: Expand - Zero Insertion
# =============================================================================
print("\n[Step 1] Expand: Zero-Insertion Operation")
print("-" * 70)

# Create a simple signal
original = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
factor = 4

expanded = filters.expand(original, factor)

print(f"Original signal ({len(original)} samples): {original}")
print(f"Expanded signal ({len(expanded)} samples): {expanded}")
print(f"\nExpansion factor: {factor}x")
print("Note: Zeros are inserted between original samples")

# Visualize expand operation
fig, axs = plt.subplots(3, 1, figsize=(12, 8))

# Original signal
axs[0].stem(np.arange(len(original)), original, basefmt=" ")
axs[0].set_title("Original Signal", fontsize=11, fontweight="bold")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True, alpha=0.3)
axs[0].set_xlim(-1, len(expanded))

# Expanded signal
axs[1].stem(np.arange(len(expanded)), expanded, basefmt=" ")
axs[1].set_title(f"After Expand (factor={factor})", fontsize=11, fontweight="bold")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True, alpha=0.3)
axs[1].set_xlim(-1, len(expanded))

# Frequency domain comparison
fft_orig = np.fft.fft(original, n=1024)
fft_exp = np.fft.fft(expanded, n=1024)
freqs = np.fft.fftfreq(1024)

axs[2].plot(
    np.fft.fftshift(freqs),
    np.fft.fftshift(20 * np.log10(np.abs(fft_orig) + 1e-10)),
    label="Original",
    linewidth=2,
)
axs[2].plot(
    np.fft.fftshift(freqs),
    np.fft.fftshift(20 * np.log10(np.abs(fft_exp) + 1e-10)),
    label="Expanded (with images)",
    linewidth=2,
    alpha=0.7,
)
axs[2].set_title("Frequency Domain: Spectral Images", fontsize=11, fontweight="bold")
axs[2].set_xlabel("Normalized Frequency")
axs[2].set_ylabel("Magnitude (dB)")
axs[2].legend()
axs[2].grid(True, alpha=0.3)
axs[2].set_ylim(-60, 40)

plt.tight_layout()
plt.savefig("03_expand_operation.png", dpi=150, bbox_inches="tight")
print("✓ Saved expand operation plot: 03_expand_operation.png")

# =============================================================================
# Step 2: Interpolate - Expand + Filter
# =============================================================================
print("\n[Step 2] Interpolate: Expand with Anti-Imaging Filter")
print("-" * 70)

# Create a clean signal
t = np.linspace(0, 1, 50)
signal = np.sin(2 * np.pi * 3 * t)  # 3 Hz sine wave

# Interpolate by factor of 4
interpolated = filters.interpolate(signal, factor=4, filter_type="sinc")

print(f"Original signal: {len(signal)} samples")
print(f"Interpolated signal: {len(interpolated)} samples")
print(f"Rate increase: {len(interpolated) / len(signal)}x")
print("\nInterpolation removes spectral images created by expand!")

# Visualize interpolation
fig, axs = plt.subplots(2, 1, figsize=(12, 6))

# Time domain
t_interp = np.linspace(0, 1, len(interpolated))
axs[0].plot(t, signal, "o-", label="Original", markersize=6, linewidth=2)
axs[0].plot(t_interp, interpolated, ".-", label="Interpolated", alpha=0.7)
axs[0].set_title("Interpolation: Clean Upsampling", fontsize=11, fontweight="bold")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Frequency domain
fft_orig = np.fft.fft(signal, n=2048)
fft_interp = np.fft.fft(interpolated, n=2048)
freqs = np.fft.fftfreq(2048)

axs[1].plot(
    np.fft.fftshift(freqs),
    np.fft.fftshift(20 * np.log10(np.abs(fft_orig) + 1e-10)),
    label="Original spectrum",
    linewidth=2,
)
axs[1].plot(
    np.fft.fftshift(freqs),
    np.fft.fftshift(20 * np.log10(np.abs(fft_interp) + 1e-10)),
    label="Interpolated (images removed)",
    linewidth=2,
    alpha=0.7,
)
axs[1].set_title("Frequency Domain: Images Filtered", fontsize=11, fontweight="bold")
axs[1].set_xlabel("Normalized Frequency")
axs[1].set_ylabel("Magnitude (dB)")
axs[1].legend()
axs[1].grid(True, alpha=0.3)
axs[1].set_ylim(-80, 40)

plt.tight_layout()
plt.savefig("03_interpolate_operation.png", dpi=150, bbox_inches="tight")
print("✓ Saved interpolation plot: 03_interpolate_operation.png")

# =============================================================================
# Step 3: Decimate - Anti-Alias + Downsample
# =============================================================================
print("\n[Step 3] Decimate: Anti-Aliasing + Downsampling")
print("-" * 70)

# Create a signal with high and low frequency components
fs = 1000  # 1 kHz sampling rate
t = np.arange(1000) / fs
signal_hf = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

# Decimate by factor of 8
decimated = filters.decimate(signal_hf, factor=8)

print(f"Original signal: {len(signal_hf)} samples at {fs} Hz")
print(f"Decimated signal: {len(decimated)} samples at {fs / 8} Hz")
print("\nDecimation includes anti-aliasing filter to prevent aliasing!")

# Visualize decimation
fig, axs = plt.subplots(3, 1, figsize=(12, 9))

# Original signal
axs[0].plot(t[:200], signal_hf[:200], linewidth=1)
axs[0].set_title("Original Signal (50 Hz + 200 Hz)", fontsize=11, fontweight="bold")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True, alpha=0.3)

# Decimated signal
t_dec = np.arange(len(decimated)) / (fs / 8)
axs[1].plot(t_dec[:25], decimated[:25], "o-", linewidth=2, markersize=5)
axs[1].set_title("Decimated Signal (factor=8)", fontsize=11, fontweight="bold")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True, alpha=0.3)

# Frequency domain
fft_orig = np.fft.fft(signal_hf, n=2048)
fft_dec = np.fft.fft(decimated, n=2048)
freqs_orig = np.fft.fftfreq(2048, d=1 / fs)
freqs_dec = np.fft.fftfreq(2048, d=1 / (fs / 8))

axs[2].plot(
    np.fft.fftshift(freqs_orig),
    np.fft.fftshift(20 * np.log10(np.abs(fft_orig) + 1e-10)),
    label=f"Original ({fs} Hz)",
    linewidth=2,
)
axs[2].plot(
    np.fft.fftshift(freqs_dec),
    np.fft.fftshift(20 * np.log10(np.abs(fft_dec) + 1e-10)),
    label=f"Decimated ({fs / 8} Hz)",
    linewidth=2,
    alpha=0.7,
)
axs[2].axvline(x=fs / 8 / 2, color="r", linestyle="--", alpha=0.5, label="New Nyquist")
axs[2].set_title(
    "Frequency Domain: High Frequencies Filtered", fontsize=11, fontweight="bold"
)
axs[2].set_xlabel("Frequency (Hz)")
axs[2].set_ylabel("Magnitude (dB)")
axs[2].legend()
axs[2].grid(True, alpha=0.3)
axs[2].set_xlim(-300, 300)

plt.tight_layout()
plt.savefig("03_decimate_operation.png", dpi=150, bbox_inches="tight")
print("✓ Saved decimation plot: 03_decimate_operation.png")

# =============================================================================
# Step 4: Resample - Rational Rate Conversion
# =============================================================================
print("\n[Step 4] Resample: Arbitrary Rate Conversion")
print("-" * 70)

# Create a clean sine wave
t = np.linspace(0, 1, 100)
signal_sine = np.sin(2 * np.pi * 5 * t)

# Different resampling ratios
ratios = [(3, 2), (5, 3), (2, 3), (3, 5)]  # (up, down)

fig, axs = plt.subplots(len(ratios), 1, figsize=(12, 10))

for idx, (up, down) in enumerate(ratios):
    resampled = filters.resample(signal_sine, up=up, down=down)
    rate_change = up / down

    t_resample = np.linspace(0, 1, len(resampled))

    axs[idx].plot(t, signal_sine, "o-", alpha=0.5, label="Original", markersize=4)
    axs[idx].plot(
        t_resample, resampled, ".-", label=f"Resampled ({up}/{down})", markersize=5
    )
    axs[idx].set_title(
        f"Rate: {up}/{down} = {rate_change:.2f}x ({len(signal_sine)} → {len(resampled)} samples)",
        fontsize=10,
        fontweight="bold",
    )
    axs[idx].set_ylabel("Amplitude")
    axs[idx].legend(loc="upper right")
    axs[idx].grid(True, alpha=0.3)

    print(
        f"  {up}/{down}: {len(signal_sine)} samples → {len(resampled)} samples ({rate_change:.2f}x)"
    )

axs[-1].set_xlabel("Time")
plt.tight_layout()
plt.savefig("03_resample_operation.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved resample operations plot: 03_resample_operation.png")

# =============================================================================
# Step 5: Practical Application - Multi-Stage Processing
# =============================================================================
print("\n[Step 5] Practical Application: Multi-Stage Processing")
print("-" * 70)

# Scenario: Process signal from 100 kHz to 12.5 kHz (8x decimation)
# Efficient approach: Two-stage decimation (4x then 2x)

fs_original = 100e3
signal_orig = np.sin(2 * np.pi * 5e3 * np.arange(1000) / fs_original)

# Single-stage decimation
decimated_1stage = filters.decimate(signal_orig, factor=8)

# Two-stage decimation (more efficient)
decimated_stage1 = filters.decimate(signal_orig, factor=4)
decimated_2stage = filters.decimate(decimated_stage1, factor=2)

print(f"Original: {len(signal_orig)} samples at {fs_original / 1e3:.1f} kHz")
print(f"\nSingle-stage (÷8):")
print(f"  Result: {len(decimated_1stage)} samples at {fs_original / 8 / 1e3:.1f} kHz")
print(f"\nTwo-stage (÷4 then ÷2):")
print(
    f"  After stage 1: {len(decimated_stage1)} samples at {fs_original / 4 / 1e3:.1f} kHz"
)
print(
    f"  After stage 2: {len(decimated_2stage)} samples at {fs_original / 8 / 1e3:.1f} kHz"
)
print("\nTwo-stage is often more computationally efficient!")

# =============================================================================
# Step 6: Key Learning Points
# =============================================================================
print("\n[Step 6] Key Learning Points")
print("-" * 70)
print("""
1. EXPAND (Zero-Insertion):
   - Inserts zeros between samples
   - Creates spectral images
   - First step in interpolation
   - Fast operation (no filtering)

2. INTERPOLATE:
   - Expand + anti-imaging filter
   - Increases sample rate cleanly
   - Preserves baseband spectrum
   - Used for upsampling signals

3. DECIMATE:
   - Anti-aliasing filter + downsample
   - Reduces sample rate safely
   - Prevents aliasing of high frequencies
   - Used for downsampling signals

4. RESAMPLE:
   - Arbitrary rational rate conversion (up/down)
   - Combines interpolation and decimation
   - Efficient polyphase implementation
   - Used for flexible rate adaptation

5. PRACTICAL CONSIDERATIONS:
   - Multi-stage is often more efficient
   - Filter design trades off:
     * Transition bandwidth
     * Stopband attenuation
     * Filter length (computation)
   - Choosing the right tool:
     * Need spectral images? → expand
     * Need clean upsampling? → interpolate
     * Need downsampling? → decimate
     * Need arbitrary rate? → resample

6. ANTI-ALIASING & ANTI-IMAGING:
   - Aliasing: High frequencies fold into baseband
   - Imaging: Spectral replicas at multiples of fs
   - Both prevented by proper filtering
   - Critical for signal quality
""")

print("\n" + "=" * 70)
print("Example complete! Check the generated PNG files.")
print("=" * 70)

# plt.show()
