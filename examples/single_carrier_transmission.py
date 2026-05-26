#!/usr/bin/env python
"""End-to-end Single Carrier Transmission & Recovery Simulation

Demonstrates the core physical layer digital transceiver chain using CommsTools:
1. Signal Generation: Pulse-shaped 16-QAM
2. Channel impairments: Fractional timing delay, static frequency offset, and AWGN noise
3. Receiver Synchronization:
   - Data-aided timing estimation & alignment (coarse + fractional interpolation)
   - Fine frequency offset estimation (Mengali-Morelli) & correction
   - Matched filtering
   - Carrier phase recovery (Viterbi-Viterbi)
   - Phase ambiguity resolution against Tx symbols
4. Quality Assessment: EVM and BER computation
5. Visualization: Save diagnostic spectrum and constellation plots

Run:
  uv run examples/single_carrier_transmission.py
"""

import os
import matplotlib
# Headless backend for headless execution
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from commstools import Signal
from commstools.backend import to_device
from commstools.impairments import apply_awgn
from commstools.timing import estimate_timing, correct_timing, fft_fractional_delay
from commstools.frequency import estimate_frequency_offset_mengali_morelli, correct_static_frequency_offset
from commstools.recovery import recover_carrier_phase_viterbi_viterbi, correct_carrier_phase, resolve_phase_ambiguity
from commstools.metrics import evm, ber
from commstools.plotting import apply_default_theme

def main():
    print("=" * 70)
    print("  CommsTools SISO Transmission & Synchronization Simulation")
    print("=" * 70)
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 1. PARAMETERS
    # ──────────────────────────────────────────────────────────────────────────────
    SPS = 4
    SYMBOL_RATE = 10e9       # 10 GBaud
    SAMPLING_RATE = SPS * SYMBOL_RATE  # 40 GSa/s
    N_SYM = 5000
    ORDER = 16               # 16-QAM
    ROLLOFF = 0.25           # RRC rolloff
    
    # Channel Impairments
    TIMING_DELAY = 12.35     # samples (coarse 12, fractional 0.35)
    FREQ_OFFSET = 200.0e3     # 200 kHz carrier frequency offset (CFO)
    ESN0_DB = 18.0           # 18 dB Es/N0
    
    seed = 42
    rng = np.random.default_rng(seed)
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 2. TRANSMITTER
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n[1] Transmitter: Generating 16-QAM pulse-shaped signal...")
    sig_tx = Signal.qam(
        order=ORDER,
        num_symbols=N_SYM,
        sps=SPS,
        symbol_rate=SYMBOL_RATE,
        pulse_shape="rrc",
        rrc_rolloff=ROLLOFF,
        seed=seed,
    )
    
    tx_samples = to_device(sig_tx.samples, "cpu").astype(np.complex64)
    tx_symbols = to_device(sig_tx.source_symbols, "cpu").astype(np.complex64)
    
    print(f"    Tx Signal: {len(tx_samples)} samples @ {SAMPLING_RATE/1e9:.1f} GSa/s ({SPS} SPS)")
    print(f"    Tx Symbols: {len(tx_symbols)} QAM-16 constellation symbols")
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 3. CHANNEL IMPAIRMENTS
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n[2] Channel: Injecting impairments...")
    
    # A. Timing Delay (Integer circular shift + fractional FFT-based delay)
    coarse_delay = int(np.round(TIMING_DELAY))
    frac_delay = TIMING_DELAY - coarse_delay
    
    samples_delayed = np.roll(tx_samples, coarse_delay)
    samples_delayed = fft_fractional_delay(samples_delayed, frac_delay)
    
    # B. Frequency Offset
    t = np.arange(len(samples_delayed)) / SAMPLING_RATE
    samples_cfo = samples_delayed * np.exp(1j * 2.0 * np.pi * FREQ_OFFSET * t)
    
    # C. AWGN Noise (respecting SPS energy normalization)
    samples_noisy = apply_awgn(samples_cfo, sps=SPS, esn0_db=ESN0_DB, seed=seed)
    
    print(f"    Injected Delay : {TIMING_DELAY:.2f} samples (coarse={coarse_delay}, fractional={frac_delay:.2f})")
    print(f"    Injected CFO   : {FREQ_OFFSET/1e6:.2f} MHz")
    print(f"    Channel Es/N0  : {ESN0_DB:.1f} dB")
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 4. RECEIVER SYNCHRONIZATION
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n[3] Receiver: Starting recovery pipeline...")
    
    # A. Timing Estimation & Correction
    # We correlate against the original known reference sequence (Tx samples)
    print("    - Performing Timing Synchronization...")
    coarse_est, frac_est = estimate_timing(samples_noisy, reference=tx_samples, sps=SPS)
    
    # Apply inverse timing correction
    samples_aligned = correct_timing(samples_noisy, coarse_offset=coarse_est, fractional_offset=frac_est, mode="circular")
    
    print(f"      Estimated Timing Delay : Coarse={coarse_est[0]}, Fractional={frac_est[0]:.3f} samples")
    print(f"      Timing Error           : {float(coarse_est[0] + frac_est[0] - TIMING_DELAY):.4f} samples")
    
    # B. Frequency Offset Estimation & Correction
    # Mengali-Morelli fine frequency estimation
    print("    - Performing Frequency Recovery...")
    fo_est = estimate_frequency_offset_mengali_morelli(
        samples_aligned, sampling_rate=SAMPLING_RATE, modulation="qam", order=ORDER
    )
    
    # Apply frequency correction
    samples_foc = correct_static_frequency_offset(samples_aligned, sampling_rate=SAMPLING_RATE, offset=fo_est)
    
    print(f"      Estimated CFO : {fo_est/1e6:.4f} MHz")
    print(f"      CFO Error     : {(fo_est - FREQ_OFFSET)/1e3:.2f} kHz")
    
    # C. Matched Filtering
    print("    - Applying Matched Filter...")
    # Wrap in Signal object to leverage metadata-driven matched_filter
    sig_rx = sig_tx.copy()
    sig_rx.samples = samples_foc
    sig_matched = sig_rx.matched_filter()
    
    # D. Decimate to Symbol Rate (1 SPS)
    print("    - Decimating to Symbol Rate (1 SPS)...")
    sig_symbols = sig_matched.copy().decimate_to_symbol_rate()
    
    # E. Carrier Phase Recovery (CPR)
    # Viterbi-Viterbi blind CPR at 1 SPS (symbol rate)
    print("    - Performing Viterbi-Viterbi Carrier Phase Recovery...")
    phase_vv = recover_carrier_phase_viterbi_viterbi(
        sig_symbols.samples, modulation="qam", order=ORDER, block_size=32
    )
    samples_cpr = correct_carrier_phase(sig_symbols.samples, phase_vv)
    sig_symbols.samples = samples_cpr
    
    # F. Resolve Phase Ambiguity
    # Blind CPR leaves a residual π/2 (90 degree) phase ambiguity; resolve it against Tx symbols
    print("    - Resolving Phase Ambiguity...")
    symbols_final = resolve_phase_ambiguity(
        sig_symbols.samples, ref_symbols=tx_symbols, modulation="qam", order=ORDER
    )
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 5. METRICS ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n[4] Performance Analysis:")
    
    # EVM calculation (%), comparing recovered symbols to Tx symbols
    evm_pct, evm_db = evm(symbols_final, tx_symbols)
    
    # Bit Error Rate (BER)
    # Map recovered symbols back to bits and compare with sig_tx.source_bits
    from commstools.mapping import demap_symbols_hard, map_bits
    # Demap recovered symbols
    rx_bits = demap_symbols_hard(symbols_final, modulation="qam", order=ORDER)
    tx_bits = to_device(sig_tx.source_bits, "cpu")
    
    bit_errors = int(np.sum(rx_bits != tx_bits))
    ber_val = bit_errors / len(tx_bits)
    
    print(f"    Exit EVM : {evm_pct:.2f}% ({evm_db:.1f} dB)")
    print(f"    Bit Errors: {bit_errors} / {len(tx_bits)}")
    print(f"    Final BER : {ber_val:.2e}")
    
    # ──────────────────────────────────────────────────────────────────────────────
    # 6. VISUALIZATION AND DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n[5] Visualization: Saving diagnostics plot...")
    apply_default_theme()
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # A. Spectrum (PSD) comparison
    sig_tx.plot_psd(ax=axes[0, 0], label="Tx Clean", show=False)
    # Wrap noisy samples in a Signal to plot its PSD easily
    sig_noisy = sig_tx.copy()
    sig_noisy.samples = samples_noisy
    sig_noisy.plot_psd(ax=axes[0, 0], label="Rx Impaired & Noisy", alpha=0.6, show=False)
    axes[0, 0].set_title("Power Spectral Density")
    axes[0, 0].legend(loc="lower left")
    
    # B. Received (Impaired) Waveform (I/Q)
    time_axis = (np.arange(150) / SAMPLING_RATE) * 1e9  # ns
    axes[0, 1].plot(time_axis, samples_noisy.real[:150], label="I", alpha=0.8)
    axes[0, 1].plot(time_axis, samples_noisy.imag[:150], label="Q", alpha=0.8)
    axes[0, 1].set_xlabel("Time [ns]")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].set_title("Received Impaired Signal (I/Q Waveform)")
    axes[0, 1].legend()
    
    # C. Constellation before synchronization
    # Decimate impaired signal to 1 sps directly (no matched filter or sync)
    sig_impaired_1sps = sig_noisy.copy().decimate_to_symbol_rate(normalize=True)
    sig_impaired_1sps.plot_constellation(ax=axes[1, 0], bins=80, show=False)
    axes[1, 0].set_title("Constellation Before Sync (Spinning + Jitter)")
    
    # D. Constellation after full recovery
    sig_final_1sps = sig_symbols.copy()
    sig_final_1sps.samples = symbols_final
    sig_final_1sps.plot_constellation(ax=axes[1, 1], bins=100, overlay_ideal=True, show=False)
    axes[1, 1].set_title(f"Constellation After Sync (EVM = {evm_pct:.2f}%)")
    
    os.makedirs("examples/images", exist_ok=True)
    fig.suptitle("SISO Transmission & Recovery Diagnostics", fontsize=15, fontweight="bold")
    fig.savefig("examples/images/siso_transmission_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("    Diagnostic plot saved to examples/images/siso_transmission_diagnostics.png")
    print("\n" + "=" * 70)
    print("  Simulation completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
