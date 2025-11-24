import time
import numpy as np
from commstools import Signal, set_backend, using_backend, get_backend


# Define a computationally intensive function
def complex_operation(signal: Signal) -> Signal:
    backend = get_backend()
    # Perform some heavy math: lots of element-wise operations
    x = signal.samples
    for _ in range(100):
        x = x * 1.001 + backend.exp(1j * backend.abs(x))

    return Signal(
        samples=x,
        sample_rate=signal.sample_rate,
        center_freq=signal.center_freq,
        modulation_format=signal.modulation_format,
    )


def main():
    print("=== JIT Compilation Demo ===")

    # Create a large signal
    N = 100_000
    print(f"Signal size: {N} samples")
    samples = np.random.randn(N) + 1j * np.random.randn(N)
    sig_np = Signal(samples=samples, sample_rate=1e6)

    # --- Numpy Backend (No JIT) ---
    set_backend("numpy")
    backend = get_backend()

    # "JIT" the function (identity for Numpy)
    jitted_op_np = backend.jit(complex_operation)

    print("\n[Numpy Backend]")
    start = time.time()
    _ = jitted_op_np(sig_np)
    print(f"Execution time: {time.time() - start:.4f} s")

    # --- JAX Backend (With JIT) ---
    print("\n[JAX Backend]")
    try:
        with using_backend("jax"):
            backend = get_backend()
            sig_jax = sig_np.to("jax")

            # JIT the function (actual JAX JIT)
            # Note: Because Signal is registered as a Pytree, we can pass it directly!
            jitted_op_jax = backend.jit(complex_operation)

            # Warmup (compilation)
            print("Compiling...")
            start = time.time()
            _ = jitted_op_jax(sig_jax).samples.block_until_ready()
            print(f"Compilation + Run time: {time.time() - start:.4f} s")

            # Fast Run
            print("Running JIT-compiled...")
            start = time.time()
            _ = jitted_op_jax(sig_jax).samples.block_until_ready()
            print(f"Execution time: {time.time() - start:.4f} s")

    except ImportError:
        print("JAX not installed.")


if __name__ == "__main__":
    main()
