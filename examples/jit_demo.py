import time
import numpy as np
from commstools import Signal, set_backend, jit


# Define a computationally intensive function
@jit
def complex_operation(signal: Signal) -> Signal:
    # Ensure signal is on the global backend
    signal = signal.ensure_backend()
    backend = signal.backend
    # Perform some heavy math: lots of element-wise operations
    x = signal.samples
    for _ in range(50):
        x = x * 1.001 + backend.exp(1j * backend.abs(x))

    return signal.update(x)


def main():
    print("=== JIT Compilation Demo (Functional API) ===")

    # Create a large signal
    N = 100_000
    print(f"Signal size: {N} samples")
    samples = np.random.randn(N) + 1j * np.random.randn(N)
    sig_np = Signal(samples=samples, sampling_rate=1e6)

    # --- Numpy Backend (No JIT) ---
    set_backend("numpy")

    print("\n[Numpy Backend]")
    start = time.time()
    _ = complex_operation(sig_np)
    print(f"Execution time: {time.time() - start:.4f} s")

    # --- JAX Backend (With JIT) ---
    print("\n[JAX Backend]")
    try:
        # Switch to JAX backend
        set_backend("jax")

        # Move signal to JAX (auto-alignment via ensure_backend is also possible inside function,
        # but explicit conversion here avoids measuring transfer time in execution)
        sig_jax = sig_np.to("jax")

        # Warmup (compilation)
        print("Compiling...")
        start = time.time()
        # Note: We call the function directly, @jit handles the dispatch
        _ = complex_operation(sig_jax).samples.block_until_ready()
        print(f"Compilation + Run time: {time.time() - start:.4f} s")

        # Fast Run
        print("Running JIT-compiled...")
        start = time.time()
        _ = complex_operation(sig_jax).samples.block_until_ready()
        print(f"Execution time: {time.time() - start:.4f} s")

        # Reset backend
        set_backend("numpy")

    except ImportError:
        print("JAX not installed.")


if __name__ == "__main__":
    main()
