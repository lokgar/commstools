"""Demonstration of the global SystemConfig pattern.

This example shows three different usage patterns:
1. Traditional way (no config) - works as before
2. Global config way - set once, use everywhere
3. Function-specific config without Signal input
"""

import numpy as np
from commstools import Signal, SystemConfig, set_config, get_config, using_config
from commstools.dsp import (
    generate_pilot_sequence,
    generate_training_signal,
    add_awgn,
    matched_filter,
)


def demo_traditional_no_config():
    """Pattern 1: Traditional approach without config (still works!)"""
    print("\n" + "=" * 70)
    print("PATTERN 1: Traditional Approach (No Config)")
    print("=" * 70)

    # Create signal the old way - explicitly passing all parameters
    samples = np.exp(1j * 2 * np.pi * 50e3 * np.arange(1000) / 1e6)
    sig = Signal(
        samples=samples, sampling_rate=1e6, center_freq=2.4e9, modulation_format="QPSK"
    )

    print(f"Signal created: {sig.samples.shape[0]} samples")
    print(f"Sampling rate: {sig.sampling_rate / 1e6} MHz")
    print(f"Modulation: {sig.modulation_format}")

    # Functions with explicit parameters
    noisy = add_awgn(sig, snr_db=20)
    print(f"Added noise with SNR=20 dB")

    # Generate sequence with explicit parameters
    pilots = generate_pilot_sequence(length=128, modulation="QPSK")
    print(f"Generated pilot sequence: {len(pilots)} symbols")


def demo_global_config():
    """Pattern 2: Global config - set once, use everywhere"""
    print("\n" + "=" * 70)
    print("PATTERN 2: Global Config (Recommended for Complex Simulations)")
    print("=" * 70)

    # Step 1: Define all system parameters in one place
    config = SystemConfig(
        # Physical layer
        sampling_rate=1e6,  # 1 MHz
        center_freq=2.4e9,  # 2.4 GHz
        modulation_format="16QAM",
        # Symbol level
        symbol_rate=250e3,  # 250 kHz symbol rate
        # samples_per_symbol will auto-compute to 4
        # Channel
        snr_db=25,
        frequency_offset=1e3,  # 1 kHz offset
        # DSP
        filter_roll_off=0.25,
        equalizer_taps=21,
        # Sequences
        sequence_length=256,
        pilot_spacing=16,
    )

    print("Created SystemConfig with:")
    print(f"  Sampling rate: {config.sampling_rate / 1e6} MHz")
    print(f"  Modulation: {config.modulation_format}")
    print(f"  Symbol rate: {config.symbol_rate / 1e3} kHz")
    print(f"  Samples/symbol: {config.samples_per_symbol} (auto-computed)")
    print(f"  SNR: {config.snr_db} dB")
    print(f"  Sequence length: {config.sequence_length}")

    # Step 2: Set it globally
    set_config(config)
    print("\nConfig set globally with set_config()")

    # Step 3: Create Signal from config - no need to repeat parameters!
    samples = np.random.randn(1000) + 1j * np.random.randn(1000)
    sig = Signal.from_config(samples=samples)

    print(f"\nSignal created from config:")
    print(f"  Samples: {sig.samples.shape[0]}")
    print(f"  Sampling rate: {sig.sampling_rate / 1e6} MHz (from config)")
    print(f"  Modulation: {sig.modulation_format} (from config)")

    # Step 4: Functions automatically use config values
    print("\nApplying DSP functions (using config values automatically):")

    # Add noise - uses config.snr_db automatically
    noisy = add_awgn(sig)
    print(f"  ✓ Added AWGN (SNR={config.snr_db} dB from config)")

    # Apply matched filter - uses config.filter_roll_off
    filtered = matched_filter(noisy)
    print(f"  ✓ Applied matched filter (roll-off={config.filter_roll_off} from config)")

    # Step 5: Sequence generation without Signal input
    print("\nGenerating sequences (no Signal input needed):")

    pilots = generate_pilot_sequence()  # Uses config values!
    print(f"  ✓ Pilot sequence: {len(pilots)} symbols (from config.sequence_length)")
    print(f"     Modulation: {config.modulation_format} (from config)")

    training = generate_training_signal()  # Uses config values!
    print(f"  ✓ Training signal: {training.samples.shape[0]} samples")
    print(f"     Sampling rate: {training.sampling_rate / 1e6} MHz (from config)")

    # Step 6: Can still override if needed
    print("\nOverriding config values when needed:")
    noisy_low_snr = add_awgn(sig, snr_db=10)  # Override config SNR
    print(
        f"  ✓ Added noise with SNR=10 dB (overriding config value of {config.snr_db} dB)"
    )

    custom_pilots = generate_pilot_sequence(length=512, modulation="QPSK")
    print(f"  ✓ Custom pilots: {len(custom_pilots)} QPSK symbols (overriding config)")


def demo_temporary_config():
    """Pattern 3: Temporary config using context manager"""
    print("\n" + "=" * 70)
    print("PATTERN 3: Temporary Config with Context Manager")
    print("=" * 70)

    # Set a default config
    default_config = SystemConfig(
        sampling_rate=1e6,
        modulation_format="QPSK",
        sequence_length=128,
        snr_db=20,
    )
    set_config(default_config)
    print("Default config: QPSK, SNR=20 dB, length=128")

    # Generate with default config
    pilots1 = generate_pilot_sequence()
    print(
        f"Generated with default: {len(pilots1)} {default_config.modulation_format} symbols"
    )

    # Temporarily use different config
    special_config = SystemConfig(
        sampling_rate=2e6,
        modulation_format="64QAM",
        sequence_length=256,
        snr_db=30,
    )

    print("\nUsing temporary config context:")
    with using_config(special_config):
        # Inside this block, special_config is active
        pilots2 = generate_pilot_sequence()
        print(
            f"  Inside context: {len(pilots2)} {special_config.modulation_format} symbols"
        )

        samples = np.random.randn(500) + 1j * np.random.randn(500)
        sig = Signal.from_config(samples=samples)
        print(f"  Signal created: fs={sig.sampling_rate / 1e6} MHz")

    # Back to default config
    pilots3 = generate_pilot_sequence()
    print(
        f"After context: {len(pilots3)} {default_config.modulation_format} symbols (back to default)"
    )


def demo_save_load_config():
    """Pattern 4: Save and load configurations"""
    print("\n" + "=" * 70)
    print("PATTERN 4: Save/Load Config for Reproducibility")
    print("=" * 70)

    # Create a config
    config = SystemConfig(
        sampling_rate=10e6,
        center_freq=5.8e9,
        modulation_format="64QAM",
        symbol_rate=2.5e6,
        snr_db=28,
        filter_roll_off=0.22,
        sequence_length=512,
    )

    # Save to YAML
    config_file = "/tmp/my_simulation_config.yaml"
    config.to_yaml(config_file)
    print(f"Saved config to: {config_file}")

    # Later, load it back
    loaded_config = SystemConfig.from_yaml(config_file)
    print(f"Loaded config from file:")
    print(f"  Sampling rate: {loaded_config.sampling_rate / 1e6} MHz")
    print(f"  Modulation: {loaded_config.modulation_format}")
    print(f"  SNR: {loaded_config.snr_db} dB")

    # Use the loaded config
    set_config(loaded_config)
    print("\nUsing loaded config for simulation...")
    pilots = generate_pilot_sequence()
    print(f"  Generated {len(pilots)} pilot symbols")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 70)
    print("COMMSTOOLS: SystemConfig Demonstration")
    print("=" * 70)
    print("\nThis demo shows how to use the global SystemConfig in different ways.")
    print("All patterns are valid - choose what works best for your use case!")

    # Run all demos
    demo_traditional_no_config()
    demo_global_config()
    demo_temporary_config()
    demo_save_load_config()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Config is OPTIONAL - old code still works")
    print("2. Config is GLOBAL - no need to pass it around")
    print("3. Functions without Signal input can still access config")
    print("4. Can override config values when needed")
    print("5. Save/load configs for reproducibility")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
