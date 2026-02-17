"""
Verification script for FFT-based timing correction fix.

This script demonstrates that the new FFT-based fractional delay preserves
SNR, unlike the old Farrow interpolator which had ~1.5 dB power loss per pass.
"""

import numpy as np
from commstools.core import Signal
from commstools.impairments import add_awgn
from commstools.sync import estimate_timing, correct_timing
from commstools.core import Preamble


def test_timing_correction_snr():
    """
    Test timing correction SNR preservation.

    Simulates the exact scenario from the notebook:
    1. Generate a frame with preamble
    2. Add AWGN noise
    3. Add integer + fractional delay
    4. Estimate and correct timing
    5. Measure SNR before and after correction
    """

    print("=" * 70)
    print("Testing FFT-based Timing Correction (SNR Preservation)")
    print("=" * 70)

    # Generate a clean signal
    np.random.seed(42)
    symbol_rate = 10e9
    sampling_rate = 40e9
    sps = int(sampling_rate / symbol_rate)

    # Generate QAM signal
    sig = Signal.qam(
        num_symbols=10000,
        order=16,
        sps=sps,
        symbol_rate=symbol_rate,
        pulse_shape="rrc",
        seed=42,
    )

    # Generate preamble
    p = Preamble(sequence_type="zc", length=71)
    ps = p.to_signal(sps=sps, symbol_rate=symbol_rate, pulse_shape="rrc")

    # Prepend preamble to signal
    combined = sig.copy()
    combined.samples = np.concatenate([ps.samples, sig.samples])

    # Add noise (30 dB Es/N0)
    clean_signal = combined.copy()
    noisy_signal = combined.copy()
    noisy_signal.samples = add_awgn(noisy_signal.samples, esn0_db=30, sps=sps)

    # Measure SNR before timing offset
    noise_before = noisy_signal.samples - clean_signal.samples
    power_signal = np.mean(np.abs(clean_signal.samples) ** 2)
    power_noise_before = np.mean(np.abs(noise_before) ** 2)
    snr_before_db = 10 * np.log10(power_signal / power_noise_before)

    print(f"\n1. Original signal (before timing offset)")
    print(f"   SNR: {snr_before_db:.2f} dB")

    # Apply integer + fractional delay
    integer_delay = 518
    fractional_delay = 0.37  # This is the key test - fractional delay

    delayed = noisy_signal.copy()
    delayed.samples = np.roll(delayed.samples, integer_delay)

    # Apply fractional delay using FFT method
    from commstools.sync import fft_fractional_delay
    delayed.samples = fft_fractional_delay(delayed.samples, fractional_delay)

    print(f"\n2. Applied timing offset")
    print(f"   Integer delay:     {integer_delay} samples")
    print(f"   Fractional delay:  {fractional_delay:.2f} samples")

    # Estimate timing
    coarse_offset, frac_offset = estimate_timing(
        delayed,
        p,
        sps=sps,
        pulse_shape="rrc",
    )

    print(f"\n3. Estimated timing offset")
    print(f"   Coarse offset:     {int(coarse_offset[0])} samples (expected: {integer_delay})")
    print(f"   Fractional offset: {frac_offset[0]:.3f} samples (expected: {fractional_delay:.3f})")

    # Correct timing
    corrected = delayed.copy()
    corrected.samples = correct_timing(
        corrected.samples,
        coarse_offset[0],
        frac_offset[0],
    )

    # Measure SNR after correction (trim edges to avoid artifacts)
    trim = 1000
    noise_after = corrected.samples[trim:-trim] - clean_signal.samples[trim:-trim]
    power_noise_after = np.mean(np.abs(noise_after) ** 2)
    snr_after_db = 10 * np.log10(power_signal / power_noise_after)

    degradation = snr_before_db - snr_after_db

    print(f"\n4. After timing correction")
    print(f"   SNR: {snr_after_db:.2f} dB")
    print(f"   SNR degradation: {degradation:.3f} dB")

    print("\n" + "=" * 70)
    if degradation < 0.5:
        print("✓ PASS: SNR degradation < 0.5 dB (FFT method preserves SNR!)")
        print(f"  Old Farrow method would have degraded ~1.5 dB")
    else:
        print(f"✗ FAIL: SNR degradation {degradation:.2f} dB is too high")
    print("=" * 70)

    return degradation


if __name__ == "__main__":
    degradation = test_timing_correction_snr()
