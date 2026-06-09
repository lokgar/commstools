# CommsTools: GPU Acceleration & Custom CUDA Kernel Analysis Report

This report evaluates the computational bottlenecks in the [CommsTools](file:///home/lokgar/commstools) library, analyzes the current acceleration strategies (CPU Numba, CuPy, JAX), and provides a technical feasibility study on implementing custom CUDA kernels (via CuPy `RawKernel` or Numba-CUDA) and algorithmic reformulations to optimize high-performance digital communications receiver DSP.

*Additionally check GPU->CPU passes in RDE/CMA/LMS equalizers running on CPU - any way to optimize or not a big speed bottleneck?*

---

## 1. Executive Summary

CommsTools currently uses a hybrid backend strategy via [backend.py](file:///home/lokgar/commstools/commstools/backend.py):
*   **CPU Acceleration**: Leverages Numba (`@njit(cache=True, fastmath=True, nogil=True)`) for sequential symbol-by-symbol operations (equalizer updates, PLL phase tracking, Rauch-Tung-Striebel smoothers).
*   **GPU Acceleration**: Employs vectorized CuPy operations and JAX Ahead-of-Time (AOT) JIT compilation with `jax.lax.scan` for GPU-placed signals.

While this structure is highly performant, our code analysis reveals **significant GPU bottlenecks and device-host synchronization barriers** in both standalone estimators and equalizer implementations:
1.  **Memory-Bandwidth Spikes in Blind Phase Search (BPS)**: The current BPS implementation in [recovery.py](file:///home/lokgar/commstools/commstools/recovery.py) generates massive intermediate distance tensors of shape `(CHUNK, B, M_const)` for non-square QAM, causing severe memory footprint and bandwidth limits.
2.  **Device-Host Synchronization in Block LMS (`block_lms`)**: The block frequency-domain equalizer in [equalization.py](file:///home/lokgar/commstools/commstools/equalization.py) is designed for the GPU, but it contains synchronous CPU-GPU transfers (specifically inside the BPS unwrapping and cycle-slip correction) that block GPU execution pipelining.
3.  **Lack of Native GPU Paths for Sequential Equalizers**: Standalone equalizers (LMS, RLS, CMA, RDE) offload to the CPU when using the `numba` backend on GPU arrays, leading to costly CPU-GPU memory copy round-trips.

**Core Recommendations**:
*   Implement a **BPS Custom CUDA Kernel** via **CuPy `RawKernel`** to eliminate high-density memory allocations on both standalone BPS and `block_lms`.
*   Optimize the BPS unwrapping in `block_lms` to run **completely asynchronously on the GPU** by moving state variables (`bps_prev4`, `bps_offset4`) to the active device, removing the per-block `to_device` synchronization.
*   Reformulate sequential algorithms using block-based architectures (e.g., Block CMA, Delayed LMS) to maximize GPU occupancy.

---

## 2. Deep-Dive Performance Bottleneck Analysis

### 2.1. Blind Phase Search (BPS)
The BPS algorithm in `recover_carrier_phase_bps` ([recovery.py:L854](file:///home/lokgar/commstools/commstools/recovery.py#L854)) searches over $B$ candidate phase rotations for each of the $N$ symbols, computing the minimum Euclidean distance to all $M_{\mathrm{const}}$ constellation points:

$$d^2[n, k] = \min_{c \in \mathcal{C}} \left| s[n]e^{-j\phi_k} - c \right|^2$$

#### Bottleneck:
For general (non-square QAM) constellations, the code computes the distance tensor explicitly:
```python
d_sq = xp.abs(x_rot[:, :, None] - const_xp[None, None, :]) ** 2
chunk_min_d = xp.min(d_sq, axis=-1).astype(float_dtype)
```
For a chunk size of $1024$ symbols, $B=64$ test phases, and a $256$-QAM constellation ($M_{\mathrm{const}}=256$), the `d_sq` tensor requires $1024 \times 64 \times 256 = 16,777,216$ complex values (**134.2 MB of allocation per chunk step**). For a signal of $10^6$ symbols, this results in **~131 GB of global memory writes and reads** due to intermediate allocations, saturating GPU memory bandwidth.

---

### 2.2. Block LMS Equalizer (`block_lms`)
The `block_lms` function ([equalization.py:L5556](file:///home/lokgar/commstools/commstools/equalization.py#L5556)) processes signals in blocks of size `block_size` (e.g., 256 or 512 symbols). Convolution is efficiently handled in the frequency domain using FFTs, but the inline phase recovery and cycle-slip tracking steps introduce substantial overhead on GPU.

#### Bottleneck 1: Synchronous BPS Unwrapping Copy
At the end of each block iteration, the phase unwrapping code does:
```python
# equalization.py: L6141-6143
_delta = to_device(_cumul_dev[:, -1], "cpu")  # (C,) — 16 bytes for C=2
bps_prev4 += _delta
bps_offset4 += _delta
```
Because the accumulators `bps_prev4` and `bps_offset4` are initialized as CPU NumPy arrays, a synchronous device-to-host transfer (`to_device`) is executed **every block**. For $100\text{k}$ symbols with `block_size=256`, this forces **390 synchronous round-trips**, stalling the GPU execution queue and serializing kernel launches.

#### Bottleneck 2: Cycle-Slip Correction CPU Round-Trips
When `cpr_cycle_slip_correction=True`, the code copies the full block phase trajectory to the host and runs the regression filter on the CPU:
```python
# equalization.py: L6152-6153
phi_blk_np = to_device(_phi_f64, "cpu").astype(np.float64)  # (C, B)
phi_corr_np = phi_blk_np.copy()
```
This forces a synchronous `C x B` float64 copy (D→H) and a subsequent write-back (H→D) per block, completely eliminating the speed benefits of GPU-based block equalization.

#### Bottleneck 3: Inline BPS Distance Matrix Allocations
If `cpr_type='bps'`, the candidate phase search is evaluated inside the block loop:
```python
# equalization.py: L6080-6083
d2_all = (xp.abs(rotated[..., None] - constellation[None, None, None, :]) ** 2).real
min_d2 = xp.min(d2_all, axis=-1).astype(xp.float32)
```
For non-square QAM, this creates a 4D tensor `d2_all` of shape `(P, C, B, M_const)` in GPU memory on every block, creating heap pressure and memory transfer bottlenecks.

---

### 2.3. Decision-Directed Phase-Locked Loop (DD-PLL) Standalone CPR
The DD-PLL in `recover_carrier_phase_pll` ([recovery.py:L324](file:///home/lokgar/commstools/commstools/recovery.py#L324)) tracks phase symbol-by-symbol. Due to the feedback loop, the calculation at step $n$ is highly dependent on the phase estimated at step $n-1$:

$$\phi[n+1] = \phi[n] + \mu \cdot \mathrm{Im}\left(y[n]\hat{d}^*[n]\right) + \nu[n]$$

#### Bottleneck:
Because the loop is inherently sequential, the GPU implementation copies samples to the CPU to execute the Numba loop (`symbols_cpu = to_device(symbols, "cpu")`) and copies the result back. This round-trip synchronization introduces latency that disrupts streaming GPU pipelines.

---

### 2.4. Standalone Sequential Equalizers (LMS, RLS, CMA, RDE)
Standalone equalizers in [equalization.py](file:///home/lokgar/commstools/commstools/equalization.py) are either JAX-based (`backend='jax'`) or CPU Numba-based (`backend='numba'`).

#### Bottleneck:
If the user passes CuPy arrays on the GPU and requests `backend='numba'`, the equalizer is forced to copy the entire sample array to the CPU, run the sequential weight-update loop, and write the output back. While JAX scans compile loops to the GPU, they serialise execution and do not fully saturate GPU cores for a single signal stream due to low arithmetic intensity per step.

---

## 3. Single-Stream Parallelization: Custom GPU Kernels & Workarounds

While sequential algorithms are traditionally bound to serial processors like CPUs, we can achieve substantial speedups for **single-stream execution** on GPUs by parallelizing the internal operations of each step, keeping execution state resident in fast GPU memories, and utilizing pipelined mathematical approximations.

### 3.1. Register-Resident Weight Filtering (LMS, CMA, RDE)
For single-stream LMS or CMA, we cannot parallelize across the symbol index $n$. However, we can eliminate global memory latency by keeping the filter weights in GPU **registers** or **shared memory** during the sequential execution.

*   **Problem**: In standard array operations, reading and writing the weight vector $\mathbf{w}_n$ at each step requires round-trips to GPU global memory (200-400 cycles of latency per access).
*   **Custom Kernel Workaround**: Implement a CuPy `RawKernel` where a single thread (or a warp using shuffle instructions) executes the sequential symbol loop.
    1.  **Load to Registers**: Load the initial weights $\mathbf{w}_0$ (size $C \times C \times N_{\mathrm{taps}}$) directly into the GPU thread's **register file** (1-cycle latency). For a $2 \times 2$ MIMO equalizer with 21 taps, this requires $2 \times 2 \times 21 = 84$ complex values, which easily fits in the thread's register allocation (up to 255 32-bit registers per thread on modern architectures).
    2.  **Sequential Symbol Loop**: Loop sequentially over the $N$ symbols. In each step $n$:
        *   Load the current input vector $\mathbf{x}_n$ from global memory (coalesced reads) or shared memory.
        *   Compute the dot product $y_n = \mathbf{w}_n^H \mathbf{x}_n$ entirely in registers.
        *   Compute the error $e_n$ and execute the gradient update directly on the register-allocated weights.
    3.  **Write Back**: Write the output symbol $y_n$ to global memory at each step, but write the final weights $\mathbf{w}_N$ back to global memory **only once** at the end of the entire loop.
*   **Result**: This eliminates $O(N)$ global memory accesses for weight updates, allowing the sequential loop to run at near-register speed on the GPU.

---

### 3.2. Shared-Memory Matrix Updates (RLS)
The Recursive Least Squares (RLS) algorithm is computationally heavy because it maintains and updates the inverse correlation matrix $P_n$ of size $M \times M$ (where $M = C \times N_{\mathrm{taps}}$) at every symbol:

$$P_n = \frac{1}{\lambda} \left( P_{n-1} - \mathbf{k}_n \mathbf{x}_n^H P_{n-1} \right)$$

For $M = 42$ (e.g. $2 \times 2$ MIMO with 21 taps), $P$ has $1764$ complex values. This quadratic complexity ($O(M^2)$ operations per symbol) is a major bottleneck on CPU, but fits GPU architectures when parallelized internally.

*   **Custom Kernel Workaround**: Launch a thread block of size 32 or 64 to process the single stream.
    1.  **Shared Memory Matrix**: Allocate the matrix $P$ in **GPU Shared Memory** (requiring $\approx 14$ KB, well within the 48-96 KB limit of a streaming multiprocessor).
    2.  **Internal Step Parallelization**: In each symbol step $n$:
        *   The threads load the input vector $\mathbf{x}_n$ and compute the matrix-vector product $\mathbf{u} = P \mathbf{x}_n$ in parallel.
        *   Compute the denominator $\lambda + \mathbf{x}_n^H \mathbf{u}$ using a parallel warp reduction.
        *   Compute the Kalman gain vector $\mathbf{k}_n$ in parallel.
        *   Update the $M \times M$ elements of the matrix $P$ in parallel (each thread updating a subset of rows/columns).
    3.  All operations read and write to the shared-memory matrix $P$ with single-cycle latency, completely avoiding global memory accesses.
*   **Result**: Since the $O(M^2)$ work at each step is parallelized across the threads in the block, RLS achieves significant single-stream speedups on GPU, turning a slow sequential algorithm into a highly parallel localized compute block.

---

### 3.3. Pipelined/Delayed LMS & CMA (FDAF Parallelization)
We can break the sequential dependency of single-stream LMS and CMA by reformulating them into the **Delayed LMS (DLMS)** or **Delayed CMA** architecture.
Instead of updating the weights using the immediate error from step $n-1$, we use an error delayed by $D$ symbols:

$$\mathbf{w}_{n+1} = \mathbf{w}_n + \mu \cdot e_{n-D}^* \cdot \mathbf{x}_{n-D}$$

*   **Parallel Execution Model**: The pipeline delay $D$ allows us to process the stream in chunks of size $D$.
    *   For symbols $n \dots n+D-1$, we compute the output signals $y_n \dots y_{n+D-1}$ in parallel as a single **matrix-vector product** using the frozen weight vector $\mathbf{w}_n$:
        
        $$\mathbf{y}_{[n:n+D]} = \mathbf{X}_{[n:n+D]} \mathbf{w}_n$$
        
    *   Once the $D$ outputs and errors are computed in parallel, we update the weights in parallel using the delayed error vectors.
*   **Implementation Workaround**: A custom CUDA kernel or structured CuPy implementation that processes symbols in chunks of $D$. For $D = 8$ or $16$, convergence degradation is negligible, while the step output computation shifts from $D$ sequential dot-products to a single parallelized matrix multiplication.

---

## 4. Single-Stream Optimization in JAX / XLA

JAX compiles Python code to XLA (Accelerated Linear Algebra). To get optimal single-stream speedups in JAX without writing custom C++ kernels, we must structure the JAX graph to guide the XLA compiler into using register/shared memory allocation and parallel execution.

### 4.1. Chunked-Scan Compilation (Exploiting XLA Matrix Engines)
If we implement a single-stream loop using a simple `jax.lax.scan` over the entire sequence, XLA compiles the loop sequentially. To force XLA to utilize parallel GPU execution units on a single stream, we can apply the **Delayed LMS/CMA** reformulation directly in JAX:

```python
# Instead of scanning symbol-by-symbol:
# _, outputs = jax.lax.scan(step_fn, w_init, samples)

# Reshape input into chunks of size D
chunks = samples.reshape(-1, D, C * num_taps)

def chunk_step_fn(w, x_chunk):
    # Compute D outputs in parallel using a vectorized matrix-vector product
    y_chunk = jnp.dot(x_chunk, w)
    # Compute errors and accumulate gradients in parallel
    errors = compute_errors(y_chunk)
    grad = jnp.dot(x_chunk.T, errors)
    # Update weights once per chunk
    w_new = w + mu * grad
    return w_new, y_chunk

_, output_chunks = jax.lax.scan(chunk_step_fn, w_init, chunks)
```
*   **XLA Compilation Behavior**: Because the inner operation `jnp.dot` operates on a $D \times M$ matrix, the XLA compiler generates a parallel matrix multiplication GPU kernel. This utilizes the GPU's Tensor Cores/parallel pipelines at every step of the scan, bypassing the single-thread execution bottleneck.

---

### 4.2. JAX Pallas & Triton Integration for Custom Memory Layouts
Standard JAX loops compile to global memory writes because JAX does not expose shared memory allocation or register layout controls. To optimize RLS or BPS in JAX on a single stream, we can use **JAX Pallas** (JAX's interface for writing custom GPU kernels using Python/Triton):

*   **Pallas Workaround**: Structure the RLS update inside a Pallas kernel using `pallas.program`.
    *   Expose the inverse correlation matrix $P$ as a shared memory scratchpad block.
    *   Use Pallas grid mappings to assign a thread block to the matrix updates, ensuring the intermediate outer-products are computed in registers and accumulated in shared memory without writing back to global memory until the final iteration.

---

## 5. Technical Recommendations & Roadmap

We recommend a phased approach to implementing custom CUDA kernels and algorithmic reformulations in the library:

### Phase 1: High-Priority Optimization (BPS & Block LMS Copies)
*   **BPS Kernel**: Write a CuPy `RawKernel` for the BPS candidate distance loop to replace the large tensor allocation in [recovery.py](file:///home/lokgar/commstools/commstools/recovery.py). (Estimated Speedup: 20x for 256-QAM signals).
*   **Asynchronous block_lms Unwrapping**: Modify `block_lms` to keep the unwrapping state variables (`bps_prev4`, `bps_offset4`) on the GPU, removing the `to_device` synchronization barrier per block.

### Phase 2: Medium-Priority Optimization (Register-Resident & Shared Memory Kernels)
*   **Register-Resident LMS/CMA Kernel**: Implement a CuPy `RawKernel` that keeps equalizer weights in thread registers during the sequential symbol loop.
*   **Shared-Memory RLS Kernel**: Write a block-parallel RLS kernel where the inverse correlation matrix $P$ is stored and updated entirely in shared memory.

### Phase 3: Algorithmic Reformulation & JAX Custom Scans
*   **Block-CMA Equalizer**: Implement a frequency-domain Block-CMA optimizer using the overlap-and-save FDAF structure.
*   **Pipelined JAX Scans**: Implement the Delayed LMS/CMA chunked matrix-update step in JAX to allow the XLA compiler to parallelize single-stream execution.
