import numpy as np
import matplotlib.pyplot as plt
from commstools import Signal, set_backend, using_backend, get_backend, jit


# 1. Define a processing function
# @jit(static_argnums=1)
def add_noise(signal: Signal, noise_power: float = 0.1) -> Signal:
    """Adds Gaussian noise to the signal."""
    backend = get_backend()
    # Generate complex noise
    noise = (
        backend.array(np.random.randn(*signal.samples.shape))
        + 1j * backend.array(np.random.randn(*signal.samples.shape))
    ) * np.sqrt(noise_power / 2)

    return signal.update(signal.samples + noise)


def main():
    print("=== Commstools Demo (Functional API) ===")

    # Parameters
    fs = 1e6  # 1 MHz sampling rate
    f0 = 50e3  # 50 kHz tone
    duration = 0.001  # 1 ms
    t = np.arange(int(fs * duration)) / fs

    # 2. Create a Signal using NumPy (default)
    print("\n[Numpy Backend]")
    set_backend("numpy")

    # Create a simple complex exponential signal
    samples = np.exp(1j * 2 * np.pi * f0 * t)
    sig = Signal(samples=samples, sampling_rate=fs, center_freq=2.4e9)

    print(f"Signal created: {sig.samples.shape[0]} samples")
    print(f"Backend type: {type(sig.samples)}")

    # Apply processing
    noisy_sig = add_noise(sig, noise_power=0.5)
    print("Applied add_noise function.")

    # 3. Switch to JAX Backend
    print("\n[JAX Backend]")
    try:
        with using_backend("jax"):
            # Move signal to JAX
            sig_jax = sig.to("jax")
            print(f"Signal moved to JAX. Backend type: {type(sig_jax.samples)}")

            # Apply processing (happens on JAX backend, JIT compiled!)
            noisy_sig_jax = add_noise(sig_jax, noise_power=0.5)
            print("Applied add_noise function on JAX backend.")

            # Compute Spectrum using JAX
            freqs, psd = noisy_sig_jax.spectrum()
            print("Computed spectrum using JAX FFT.")

            # Bring back to Numpy for plotting
            freqs_np = np.array(freqs)
            psd_np = np.array(psd)

    except ImportError:
        print("JAX not installed or configured. Skipping JAX demo.")
        # Fallback for plotting if JAX failed
        freqs, psd = noisy_sig.spectrum()
        freqs_np = freqs
        psd_np = psd

    # 4. Visualization
    print("\nVisualizing results...")
    plt.figure(figsize=(12, 5))

    # Time Domain
    plt.subplot(1, 2, 1)
    plt.plot(noisy_sig.time_axis() * 1e3, noisy_sig.samples.real, label="Real")
    plt.plot(
        noisy_sig.time_axis() * 1e3, noisy_sig.samples.imag, label="Imag", alpha=0.7
    )
    plt.title("Time Domain (Numpy)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Frequency Domain
    plt.subplot(1, 2, 2)
    plt.plot(freqs_np / 1e3, 10 * np.log10(psd_np))
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Power (dB)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()  # Commented out for headless environments
    # output_file = "demo_plot.png"
    # plt.savefig(output_file)
    # print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    main()
