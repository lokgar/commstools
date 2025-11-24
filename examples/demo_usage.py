import numpy as np
import matplotlib.pyplot as plt
from commstools import Signal, ProcessingBlock, set_backend, using_backend, get_backend


# 1. Define a custom Processing Block
class AddNoise(ProcessingBlock):
    """Adds Gaussian noise to the signal."""

    def __init__(self, noise_power: float = 0.1):
        self.noise_power = noise_power

    def process(self, signal: Signal) -> Signal:
        backend = get_backend()
        # Generate complex noise
        noise = (
            backend.array(np.random.randn(*signal.samples.shape))
            + 1j * backend.array(np.random.randn(*signal.samples.shape))
        ) * np.sqrt(self.noise_power / 2)

        # Ensure noise is on the correct backend (JAX/Numpy)
        # Note: In a real implementation, we'd use backend-specific random generation
        # for better performance/reproducibility, but this works for a demo.

        new_samples = signal.samples + noise
        return Signal(
            samples=new_samples,
            sample_rate=signal.sample_rate,
            center_freq=signal.center_freq,
            modulation_format=signal.modulation_format,
        )


def main():
    print("=== Commstools Demo ===")

    # Parameters
    fs = 1e6  # 1 MHz sample rate
    f0 = 50e3  # 50 kHz tone
    duration = 0.001  # 1 ms
    t = np.arange(int(fs * duration)) / fs

    # 2. Create a Signal using NumPy (default)
    print("\n[Numpy Backend]")
    set_backend("numpy")

    # Create a simple complex exponential signal
    samples = np.exp(1j * 2 * np.pi * f0 * t)
    sig = Signal(samples=samples, sample_rate=fs, center_freq=2.4e9)

    print(f"Signal created: {sig.samples.shape[0]} samples")
    print(f"Backend type: {type(sig.samples)}")

    # Apply processing
    noise_block = AddNoise(noise_power=0.5)
    noisy_sig = noise_block(sig)
    print("Applied AddNoise block.")

    # 3. Switch to JAX Backend
    print("\n[JAX Backend]")
    try:
        with using_backend("jax"):
            # Move signal to JAX
            sig_jax = sig.to("jax")
            print(f"Signal moved to JAX. Backend type: {type(sig_jax.samples)}")

            # Apply processing (happens on JAX backend)
            # Note: Our simple AddNoise block uses numpy.random, which returns numpy arrays.
            # The backend.array() call in AddNoise handles the conversion to JAX array.
            noisy_sig_jax = noise_block(sig_jax)
            print("Applied AddNoise block on JAX backend.")

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
    plt.show()


if __name__ == "__main__":
    main()
