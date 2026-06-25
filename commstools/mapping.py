"""
Symbol mapping, demapping, and constellation management.

This module provides high-performance routines for the transition between
digital bits and physical IQ symbols. It supports standardized mapping
strategies (Gray coding) and advanced demapping algorithms (Soft-decision LLR).

Note: The codes and constellations are generated using numpy.

Functions
---------
gray_code :
    Generates Gray code sequences for bit-to-symbol assignment.
gray_to_binary :
    Inverse Gray code mapping for bit retrieval.
gray_constellation :
    Primary interface for generating Gray-coded constellation arrays.
constellation_power :
    Average symbol power ``E[|s|^2]`` (pmf-weighted for PS-QAM); the single
    source of truth for ``E_PS``.
map_bits :
    Maps bit sequences to complex/float symbols.
demap_symbols_hard :
    Performs hard-decision demapping from symbols to bits.
compute_llr :
    Computes Log-Likelihood Ratios (LLRs) for soft-decision decoding.

Probabilistic constellation shaping (PS-QAM)
--------------------------------------------
maxwell_boltzmann :
    Maxwell-Boltzmann PMF over a QAM constellation (literature-scale ``nu``).
ps_entropy :
    Per-symbol Shannon entropy of a Maxwell-Boltzmann distribution.
optimal_nu :
    Solves for the shaping parameter ``nu`` that hits a target entropy.
sample_ps_symbols :
    Draws shaped QAM symbols on the normalized grid from a given PMF.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from .backend import ArrayType, _get_jax, dispatch, is_jax_array, to_device, to_jax
from .core.signal import Signal
from .logger import logger

# Lazy cache for JIT-compiled soft demapping kernels
_JITTED_SOFT_DEMAP: dict[str, Any] = {}


def _get_jitted_soft_demap():
    """
    Returns JIT-compiled maxlog and exact LLR computation functions.

    Functions are defined and compiled lazily on first call to avoid
    importing JAX at module load time.
    """
    if not _JITTED_SOFT_DEMAP:
        jax, jnp, _ = _get_jax()
        if jax is None:
            raise ImportError(
                "JAX is required for soft demapping. Install with: pip install jax"
            )

        @jax.jit
        def maxlog(symbols, constellation, bits_table_t, sigma_sq, log_pmf):
            """Max-log LLR with PS prior.

            symbols (N,), constellation (M,), bits_table_t (k, M), log_pmf (M,).
            Uniform case: pass log_pmf = jnp.zeros(M) — constant offset cancels.
            PS case: log_pmf = log P(sₘ).

            Effective metric: eff_m = d_m/σ² - log P(sₘ)
            LLR_k = min_{b=1} eff - min_{b=0} eff
            """
            distances_sq = (
                jnp.abs(symbols[:, None] - constellation[None, :]) ** 2
            )  # (N, M)
            eff = distances_sq / sigma_sq - log_pmf[None, :]  # (N, M)

            def bit_llr(bit_row):  # (M,)
                d0 = jnp.where(bit_row == 0, eff, jnp.inf)  # (N, M)
                d1 = jnp.where(bit_row == 1, eff, jnp.inf)  # (N, M)
                return jnp.min(d1, axis=1) - jnp.min(d0, axis=1)  # (N,)

            return jax.vmap(bit_llr)(bits_table_t).T  # (k, N) -> (N, k)

        @jax.jit
        def exact(symbols, constellation, bits_table_t, sigma_sq, log_pmf):
            """Exact LLR with PS prior via log-sum-exp.

            symbols (N,), constellation (M,), bits_table_t (k, M), log_pmf (M,).
            Uniform case: pass log_pmf = jnp.zeros(M).
            PS case: log_pmf = log P(sₘ).

            log_terms_m = log P(sₘ) - d_m/σ²
            LLR_k = LSE_{b=0}(log_terms) - LSE_{b=1}(log_terms)
            """
            distances_sq = (
                jnp.abs(symbols[:, None] - constellation[None, :]) ** 2
            )  # (N, M)
            log_terms = log_pmf[None, :] - distances_sq / sigma_sq  # (N, M)

            def bit_llr(bit_row):  # (M,)
                e0 = jnp.where(bit_row == 0, log_terms, -jnp.inf)  # (N, M)
                e1 = jnp.where(bit_row == 1, log_terms, -jnp.inf)  # (N, M)
                return jax.scipy.special.logsumexp(
                    e0, axis=1
                ) - jax.scipy.special.logsumexp(e1, axis=1)  # (N,)

            return jax.vmap(bit_llr)(bits_table_t).T  # (k, N) -> (N, k)

        _JITTED_SOFT_DEMAP["maxlog"] = maxlog
        _JITTED_SOFT_DEMAP["exact"] = exact

    return _JITTED_SOFT_DEMAP["maxlog"], _JITTED_SOFT_DEMAP["exact"]


@lru_cache(maxsize=128)
def gray_code(n: int) -> np.ndarray:
    """
    Generates a Gray code sequence for `n` bits.

    Gray coding ensures that adjacent symbols in the constellation differ
    by only one bit, minimizing bit error rate (BER) for a given symbol
    error rate (SER).

    Parameters
    ----------
    n : int
        Number of bits.

    Returns
    -------
    np.ndarray
        Array of integers representing the Gray code sequence.
        Shape: (2^n,).

    Examples
    --------
    >>> gray_code(2)
    array([0, 1, 3, 2])
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([0], dtype=int)

    # Vectorized implementation: s ^ (s >> 1)
    i = np.arange(1 << n, dtype=int)

    return i ^ (i >> 1)


@lru_cache(maxsize=128)
def gray_to_binary(n: int) -> np.ndarray:
    """
    Computes the inverse Gray code mapping: symbol index to bit pattern.

    In soft demapping, it is necessary to identify which bits correspond to
    each constellation point. Since `gray_code(n)[i]` provides the Gray
    code for natural binary `i`, this inverse mapping finds the natural
    binary representation corresponding to a specific Gray-coded index.

    Parameters
    ----------
    n : int
        Number of bits per symbol.

    Returns
    -------
    np.ndarray
        Array where `result[s]` provides the natural binary integer
        representing the bit pattern for symbol `s`. Shape: (2^n,).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([0], dtype=int)

    # gray_code gives: natural_binary -> gray_symbol
    # We want: gray_symbol -> natural_binary (inverse)
    gray = gray_code(n)
    inverse = np.zeros(1 << n, dtype=int)
    inverse[gray] = np.arange(1 << n)

    return inverse


@lru_cache(maxsize=128)
def gray_constellation(
    modulation: str,
    order: int,
    normalize: bool = True,
    unipolar: bool = False,
) -> np.ndarray:
    """
    Generate constellation points with Gray mapping.

    The returned array is indexed by the symbol's natural binary value.
    `constellation[i]` is the complex/float value for the symbol representing
    the bit pattern for integer `i`.

    Parameters
    ----------
    modulation : {"psk", "qam", "ask", "pam"}
        Modulation type. If string contains 'unipolar', unipolar=True is triggered.
    order : int
        Modulation order (number of symbols). Must be a power of 2.
    normalize : bool, default True
        If True, scales the constellation to unit average power (E_s = 1).
    unipolar : bool, default False
        If True, shifts the constellation to be strictly non-negative (for 'ask'/'pam').
        If the modulation string contains 'unipol', this is automatically forced to True.

    Returns
    -------
    np.ndarray
        Array of constellation points (NumPy). Shape: (order,).
        Complex for 'psk' and 'qam', Float for 'ask'.

    Notes
    -----
    By default, constellations are normalized to **unit average power (E_s = 1)**.
    This ensures that when different modulation schemes are mixed (e.g., PSK
    pilots with QAM payload), they start from a consistent power baseline.
    """
    logger.debug(
        f"Generating Gray-coded constellation: modulation={modulation}, order={order}, normalize={normalize}"
    )
    modulation = modulation.lower()

    # Extract core modulation scheme
    if "psk" in modulation:
        modulation = "psk"
    elif "qam" in modulation:
        modulation = "qam"
    elif "ask" in modulation or "pam" in modulation:
        modulation = "ask"

    if order < 2:
        raise ValueError("Order must be at least 2 for modulation")

    if modulation == "psk":
        result = _gray_psk(order)
    elif modulation == "ask":
        result = _gray_ask(order, unipolar=unipolar)
    elif modulation == "qam":
        k = int(np.log2(order))
        if 2**k != order:
            raise ValueError(f"Order must be power of 2 for {modulation}")

        # Check for 8-QAM
        if order == 8:
            result = _gray_qam_8_rect()
        elif k % 2 == 0:
            result = _gray_qam_square(order)
        else:
            result = _gray_qam_cross(order)
    else:
        raise ValueError(f"Unsupported modulation type: {modulation}")

    if normalize:
        if modulation == "psk":
            # PSK is already on unit circle (E_s = 1)
            pass
        else:
            from . import helpers

            result = helpers.normalize(result, mode="average_power")

    return result


def _gray_psk(order: int) -> np.ndarray:
    """
    Generates an M-PSK constellation with Gray mapping.

    Parameters
    ----------
    order : int
        Modulation order (must be a power of 2).

    Returns
    -------
    np.ndarray
        Complex array of PSK constellation points.
    """
    # Bits per symbol
    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError("Order must be power of 2 for psk")

    # Gray codes for k bits
    gray = gray_code(k)

    # BPSK Special Case: strictly real values
    if order == 2:
        points = np.array([-1.0, 1.0], dtype=float)
        constellation = np.zeros(order, dtype=float)
        constellation[gray] = points
        return constellation

    # Geometric points (phases)
    # Phases are 0, 2pi/M, ..., 2pi(M-1)/M
    phases = np.arange(order) * 2 * np.pi / order
    points = np.exp(1j * phases)

    constellation = np.zeros(order, dtype=complex)
    constellation[gray] = points

    return constellation


def _gray_ask(order: int, unipolar: bool = False) -> np.ndarray:
    """
    Generates an M-ASK constellation with Gray mapping.

    By default, returns bipolar values centered at zero. If unipolar=True,
    shifts values to start from zero.

    Parameters
    ----------
    order : int
        Modulation order (must be a power of 2).

    Returns
    -------
    np.ndarray
        Real-valued array of ASK constellation points.
    """
    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError("Order must be power of 2 for ask")

    gray = gray_code(k)
    # Points are symmetric: -M+1, -M+3, ..., M-3, M-1
    points = np.linspace(-order + 1, order - 1, order)

    if unipolar:
        points = points - np.min(points)

    constellation = np.zeros(order, dtype=float)
    constellation[gray] = points

    return constellation


def _gray_qam_square(order: int) -> np.ndarray:
    """
    Generates a Square M-QAM constellation with Gray mapping.

    Constructed as the Cartesian product of two Gray-coded ASK
    constellations. Valid for orders where log2(M) is even.

    Parameters
    ----------
    order : int
        Modulation order (must be an even power of 2, e.g., 16, 64).

    Returns
    -------
    np.ndarray
        Complex array of square QAM constellation points.
    """
    # M = 2^(2k). I bits = k, Q bits = k.
    k_total = int(np.log2(order))
    n = k_total // 2  # bits per axis

    # Generate ASK for each axis
    m_axis = 2**n
    pam = _gray_ask(m_axis)

    # Cartesian product
    # Symbol s (2n bits). High n bits -> I, Low n bits -> Q.
    s = np.arange(order)
    mask_q = (1 << n) - 1

    idx_i = s >> n
    idx_q = s & mask_q

    i_vals = pam[idx_i]
    q_vals = pam[idx_q]

    constellation = i_vals + 1j * q_vals

    return constellation


def _gray_qam_8_rect() -> np.ndarray:
    """
    Generates an 8-QAM constellation with Rectangular Gray mapping.

    Returns
    -------
    np.ndarray
        Complex array of 8-QAM constellation points.
    """

    points = np.zeros(8, dtype=complex)

    # 000 (I=-3, Q=-1)
    points[0] = -3.0 - 1.0j

    # 001 (I=-3, Q=+1)
    points[1] = -3.0 + 1.0j

    # 010 (I=-1, Q=-1)
    points[2] = -1.0 - 1.0j

    # 011 (I=-1, Q=+1)
    points[3] = -1.0 + 1.0j

    # 100 (I=+3, Q=-1)
    points[4] = 3.0 - 1.0j

    # 101 (I=+3, Q=+1)
    points[5] = 3.0 + 1.0j

    # 110 (I=+1, Q=-1)
    points[6] = 1.0 - 1.0j

    # 111 (I=+1, Q=+1)
    points[7] = 1.0 + 1.0j

    return points


def _gray_qam_cross(order: int) -> np.ndarray:
    """
    Generates a Cross-QAM constellation using a Quasi-Gray mapping.

    Cross constellations are used for QAM orders where log2(M) is odd (e.g., 32, 128)
    to maintain a more circular/compact shape than a purely rectangular grid.

    Parameters
    ----------
    order : int
        Modulation order (must be an odd power of 2).

    Returns
    -------
    np.ndarray
        Complex array of cross QAM constellation points.
    """
    # M = 2^(2k+1).
    k_total = int(np.log2(order))

    # I bits n = ceil(k_total/2) = (k+1)//2
    # Q bits m = floor(k_total/2) = k // 2
    n = (k_total + 1) // 2
    m = k_total // 2

    width = 2**n
    height = 2**m

    # 1. Generate Gray sequences for underlying Rectangular Grid
    gray_i_seq = gray_code(n)
    gray_q_seq = gray_code(m)

    # 2. Build Inverse Lookup Tables (Symbol -> Geometric Index)
    inv_gray_i = np.zeros(width, dtype=int)
    inv_gray_i[gray_i_seq] = np.arange(width)

    inv_gray_q = np.zeros(height, dtype=int)
    inv_gray_q[gray_q_seq] = np.arange(height)

    # 3. Decompose input Symbols into I and Q components
    s = np.arange(order)
    mask_q = (1 << m) - 1
    idx_i_sym = s >> m
    idx_q_sym = s & mask_q  # This is the Gray-coded Symbol Value for Q

    # 4. Map Symbols to Geometric Indices (0..Width-1, 0..Height-1)
    geo_i = inv_gray_i[idx_i_sym]
    geo_q = inv_gray_q[idx_q_sym]

    # 5. Determine Folding Parameters
    n_shift = 0
    if n >= 3:
        n_shift = 2 ** (n - 3)

    if n_shift == 0:
        # Fallback
        val_i = (-width + 1 + 2 * geo_i).astype(float)
        val_q = (-height + 1 + 2 * geo_q).astype(float)
        return val_i + 1j * val_q

    # 6. Identify Wings
    mask_left = geo_i < n_shift
    mask_right = geo_i >= (width - n_shift)

    # 7. Calculate Rotation/Translation
    center_offset = (width - height) // 2

    # Left Wing -> Top Cap
    rotated_i_left = geo_q + center_offset
    rotated_q_left = height + (n_shift - 1 - geo_i)

    # Right Wing -> Bottom Cap
    rotated_i_right = geo_q + center_offset
    rotated_q_right = -1 - (geo_i - (width - n_shift))

    # Apply Updates
    geo_i_final = geo_i.copy()
    geo_q_final = geo_q.copy()

    geo_i_final = np.where(mask_left, rotated_i_left, geo_i_final)
    geo_i_final = np.where(mask_right, rotated_i_right, geo_i_final)

    geo_q_final = np.where(mask_left, rotated_q_left, geo_q_final)
    geo_q_final = np.where(mask_right, rotated_q_right, geo_q_final)

    # Convert Final Geometric Indices to Values
    final_i_vals = (-width + 1 + 2 * geo_i_final).astype(float)
    final_q_vals = (-height + 1 + 2 * geo_q_final).astype(float)

    constellation = final_i_vals + 1j * final_q_vals

    return constellation


def map_bits(
    bits: ArrayType,
    modulation: str,
    order: int,
    unipolar: bool = False,
) -> ArrayType:
    """
    Maps a bit sequence to complex or real symbols.

    This function follows a bit-first architecture: it takes a flat sequence
    of bits and packs them into symbols according to the modulation scheme
    and Gray mapping.

    Output dtype is ``complex64`` for PSK/QAM and ``float32`` for ASK/PAM.

    Parameters
    ----------
    bits : array_like
        Input binary sequence (0s and 1s).
    modulation : {"psk", "qam", "ask", "pam"}
        Modulation scheme.
    order : int
        Modulation order (number of symbols).
    unipolar : bool, default False
        Trigger unipolar mapping for ASK/PAM.

    Returns
    -------
    array_like
        Array of symbols. Shape: (N_bits / log2(order),).
        Backend (NumPy/CuPy) matches the input `bits`.
    """
    logger.debug(f"Mapping bits to {modulation.upper()} {order}-level symbols.")
    bits, xp, _ = dispatch(bits)

    if order < 2:
        raise ValueError("Order must be at least 2 for modulation")

    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError(f"Order must be a power of 2, got {order}")

    if len(bits) % k != 0:
        raise ValueError(
            f"Number of bits ({len(bits)}) must be divisible by bits per symbol ({k})"
        )

    # Reshape bits into symbols (N/k, k)
    num_symbols = len(bits) // k

    # Pack bits into integer indices
    bits_reshaped = bits.reshape((num_symbols, k))
    powers = 2 ** xp.arange(k - 1, -1, -1, dtype=int)
    indices = xp.sum(bits_reshaped * powers, axis=1)

    # Get constellation (returns NumPy)
    constellation = gray_constellation(modulation, order, unipolar=unipolar)

    # Ensure constellation is on the same backend and dtype
    constellation = xp.asarray(constellation)

    # ASK/PAM constellations are real-valued; PSK/QAM are complex.
    mod_lower = modulation.lower()

    if "ask" in mod_lower or "pam" in mod_lower:
        constellation = constellation.astype(xp.float32)
    else:
        constellation = constellation.astype(xp.complex64)

    # Map indices to points
    return constellation[indices]


def demap_symbols_hard(
    symbols: ArrayType | Signal,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    pmf: np.ndarray | None = None,
) -> ArrayType | Signal:
    """
    Maps complex symbols back to a sequence of bits (hard decisions).

    This function performs minimum Euclidean distance decoding to find the
    most likely bit pattern for each received symbol.

    Parameters
    ----------
    symbols : array_like
        Input array of received symbols. Shape: (..., N_symbols).
    modulation : {"psk", "qam", "ask"}
        Modulation type.
    order : int
        Modulation order.
    unipolar : bool, default False
        Trigger unipolar demapping for ASK/PAM.
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(order,)`` for PS-QAM.  When provided, the
        input symbols are scaled by ``sqrt(E_PS)`` (where
        ``E_PS = Σ P(s_m) |s_m|²`` on the normalised grid) before the
        nearest-neighbour search, mapping unit-avg-power resolved symbols
        back to the ``{s_m}`` grid used by ``gray_constellation``.
        Use this when ``symbols`` comes from
        ``resolved_symbols`` of a PS-QAM signal.
        Has no effect for uniform modulations.

    Returns
    -------
    array_like
        Sequence of bits (0s and 1s). Shape: (..., N_symbols * log2(order)).
        The backend (NumPy/CuPy) matches the input `symbols`.
    """
    if isinstance(symbols, Signal):
        sig = symbols
        if sig.signal_type is not None:
            logger.warning(
                "demap_symbols_hard() called on a frame-generated signal — skipping. "
                "Extract the payload segment via frame.get_structure_map() and build "
                "a plain Signal before demapping."
            )
            return sig.copy()
        if sig.mod_scheme is None or sig.mod_order is None:
            raise ValueError("Modulation scheme and order required for demapping.")
        if sig.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Call resolve_symbols(sig) first."
            )
        new = sig.copy()
        new.resolved_bits = demap_symbols_hard(
            sig.resolved_symbols,
            sig.mod_scheme,
            sig.mod_order,
            unipolar=sig.mod_unipolar or False,
            pmf=sig.ps_pmf if pmf is None else pmf,
        )
        return new

    if modulation is None or order is None:
        raise ValueError("demap_symbols_hard() requires modulation and order.")

    logger.debug(f"Demapping {modulation.upper()} {order}-level symbols to bits.")
    symbols, xp, _ = dispatch(symbols)

    # Capture original shape to restore structure later
    # If input is (N,), output (N*k,)
    # If input is (C, N), output (C, N*k)
    original_shape = symbols.shape

    # Ensure 1D for vectorized distance broadcasting
    symbols_flat = symbols.flatten()

    # Get constellation
    constellation = gray_constellation(modulation, order, unipolar=unipolar)

    # PS-QAM: receive-path symbols at unit average power live on the
    # ``{s_m/sqrt(E_PS)}`` grid.  Rescale by ``sqrt(E_PS)`` to bring them back
    # to the ``{s_m}`` grid that ``gray_constellation`` returns, so the
    # nearest-neighbour search is exact.  No-op for uniform modulations.
    if pmf is not None:
        e_ps = constellation_power(constellation, pmf)
        if e_ps < 1.0 - 1e-6:
            symbols_flat = symbols_flat * xp.asarray(
                np.sqrt(e_ps), dtype=symbols_flat.real.dtype
            )

    constellation = xp.asarray(constellation)

    # 1. Find nearest constellation point (Hard Decision)
    k = int(np.log2(order))
    is_sq_qam = (modulation == "qam") and (order != 8) and (k % 2 == 0)

    if is_sq_qam:
        # O(1) component-wise rounding — no (N, M) distance matrix.
        # Levels are evenly spaced; round each axis independently.
        n_ax = k // 2  # bits per axis
        side = 2**n_ax  # points per axis
        levels = xp.sort(xp.unique(constellation.real))  # (side,)
        lev_min = float(levels[0])
        d_grid = float(levels[1] - levels[0])
        # Gray LUT: sorted-level index (geometric) → natural-binary symbol index
        gray_lut = xp.asarray(gray_code(n_ax))  # (side,)
        g_i = xp.clip(
            xp.round((symbols_flat.real - lev_min) / d_grid).astype(xp.int64),
            0,
            side - 1,
        )
        g_q = xp.clip(
            xp.round((symbols_flat.imag - lev_min) / d_grid).astype(xp.int64),
            0,
            side - 1,
        )
        indices = (gray_lut[g_i] << n_ax) | gray_lut[g_q]  # (N_flat,)
    else:
        # General path: chunk N to bound peak memory at (CHUNK_N, M_const).
        CHUNK_N = 4096
        indices = xp.empty(len(symbols_flat), dtype=xp.int64)
        for n0 in range(0, len(symbols_flat), CHUNK_N):
            n1 = min(n0 + CHUNK_N, len(symbols_flat))
            d = xp.abs(symbols_flat[n0:n1, xp.newaxis] - constellation[xp.newaxis, :])
            indices[n0:n1] = xp.argmin(d, axis=1)

    # 2. Convert indices to bits
    # Extract bits from indices: (N, k)
    # We use bit shifting: (index >> shift) & 1
    shifts = xp.arange(k - 1, -1, -1, dtype="int32")
    bits = ((indices[:, xp.newaxis] >> shifts) & 1).astype(xp.int8)

    # 3. Reshape to restore original structure
    # bits is currently (Total_Symbols, k)
    # We flatten it to (Total_Symbols * k)
    flat_bits = bits.flatten()

    # Calculate new shape: replace last dimension D with D*k
    # If original shape was (D1, D2, ..., Dn), new shape is (D1, D2, ..., Dn * k)
    if len(original_shape) > 0:
        new_shape = list(original_shape)
        new_shape[-1] = new_shape[-1] * k
        return flat_bits.reshape(new_shape)
    else:
        # Scalar input case (if supported), returns 1D array of bits
        return flat_bits


def compute_llr(
    symbols: ArrayType,
    modulation: str,
    order: int,
    noise_var: float,
    method: str = "maxlog",
    unipolar: bool = False,
    output: str = "jax",
    pmf: np.ndarray | None = None,
) -> ArrayType:
    """
    Compute Log-Likelihood Ratios (LLRs) for soft-decision decoding.

    Positive LLR → bit 0 more likely; negative → bit 1; magnitude = confidence.
    JAX JIT-compiled with ``jax.vmap`` over bit positions; fully differentiable.

    Parameters
    ----------
    symbols : array_like
        Received noisy symbols. Shape: (..., N_symbols). NumPy, CuPy, or JAX.
    modulation : {"psk", "qam", "ask"}
        Modulation type.
    order : int
        Modulation order.
    noise_var : float
        Complex noise variance sigma^2 referenced to the normalised
        constellation (unit avg power).  sigma^2 = 10^(-EsN0_dB / 10).
    method : {"maxlog", "exact"}, default "maxlog"
        LLR algorithm. ``"maxlog"`` is faster; ``"exact"`` uses log-sum-exp.
    unipolar : bool, default False
        Use unipolar constellation for ASK/PAM.
    output : {"jax", "input", "numpy"}, default "jax"
        Output array type.  ``"jax"`` preserves differentiability;
        ``"input"`` matches the input backend; ``"numpy"`` forces NumPy.
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(order,)`` for PS-QAM.  Pass
        ``maxwell_boltzmann(order, nu)`` to incorporate the non-uniform prior.
        ``None`` assumes uniform prior.

    Returns
    -------
    array_like
        LLR values. Shape: (..., N_symbols * log2(order)).
        Array type determined by ``output``.

    Notes
    -----
    Max-Log: LLR_k ≈ (1/sigma^2) * (min_{S_1^k} |r-s|^2 - min_{S_0^k} |r-s|^2).
    Exact: LLR_k = log sum_{S_0^k} exp(-|r-s|^2/sigma^2) - log sum_{S_1^k} ...

    For PS-QAM, ``symbols`` must be on the same scale as
    ``gray_constellation`` (unit avg power).  After
    ``resolve_symbols`` the receiver renormalises;
    use ``gmi`` instead for correct scale.
    """
    logger.debug(
        f"Computing LLRs for {modulation.upper()} {order}-level (method={method}, output={output})."
    )

    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError(f"Order must be a power of 2, got {order}")
    if method not in ("maxlog", "exact"):
        raise ValueError(f"Unknown method: {method}. Use 'maxlog' or 'exact'.")
    if output not in ("jax", "input", "numpy"):
        raise ValueError(f"Unknown output: {output!r}. Use 'jax', 'input', or 'numpy'.")

    # Convert to JAX if not already
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError(
            "JAX is required for LLR computation. Install with: pip install jax"
        )

    # Build constellation and bits table on CPU once — shared by both paths.
    is_complex = symbols.dtype.kind == "c"
    const = gray_constellation(modulation, order, unipolar=unipolar).astype(
        "complex64" if is_complex else "float32"
    )
    bits_table_np = (
        (
            (
                np.arange(order, dtype="int32")[:, None]
                >> np.arange(k - 1, -1, -1, dtype="int32")
            )
            & 1
        )
        .astype("int32")
        .T
    )
    sigma_np = np.float32(max(noise_var, 1e-20))

    # Build log_pmf: zeros = uniform (constant offset cancels in LLR difference).
    if pmf is not None:
        log_pmf_np = np.log(np.clip(np.asarray(pmf, dtype=np.float32), 1e-40, None))
    else:
        log_pmf_np = np.zeros(order, dtype=np.float32)

    # JAX path
    jax_module, jnp, _ = _get_jax()
    if is_jax_array(symbols):
        assert jnp is not None
        if hasattr(symbols, "shape"):
            original_shape = symbols.shape
        else:
            # Fallback for JAX tracers or odd objects
            original_shape = jnp.shape(symbols)

        jax_symbols_flat = symbols.flatten()
        constellation_jax = jnp.asarray(const)
        bits_table_t_jax = jnp.asarray(bits_table_np)
        sigma_sq = jnp.asarray(sigma_np)
        log_pmf_jax = jnp.asarray(log_pmf_np)

    # NumPy/CuPy path
    else:
        symbols, xp, _ = dispatch(symbols)
        original_shape = symbols.shape

        jax_symbols_flat = to_jax(
            symbols, dtype="complex64" if is_complex else "float32"
        ).flatten()
        device = jax_symbols_flat.device

        assert jax_module is not None
        assert jnp is not None
        # device_put accepts NumPy arrays directly — no intermediate jnp.asarray needed
        constellation_jax = jax_module.device_put(const, device)
        bits_table_t_jax = jax_module.device_put(bits_table_np, device)
        sigma_sq = jax_module.device_put(jnp.asarray(sigma_np), device)
        log_pmf_jax = jax_module.device_put(log_pmf_np, device)

    # Compute LLRs via JIT-compiled kernels
    maxlog_fn, exact_fn = _get_jitted_soft_demap()
    if method == "maxlog":
        llrs = maxlog_fn(
            jax_symbols_flat, constellation_jax, bits_table_t_jax, sigma_sq, log_pmf_jax
        )
    else:
        llrs = exact_fn(
            jax_symbols_flat, constellation_jax, bits_table_t_jax, sigma_sq, log_pmf_jax
        )

    # Reshape to match input structure
    flat_llrs = llrs.flatten()
    if len(original_shape) > 1:
        new_shape = list(original_shape)
        new_shape[-1] = new_shape[-1] * k
        flat_llrs = flat_llrs.reshape(new_shape)

    # Convert output to the requested backend
    if output == "jax":
        return flat_llrs
    elif output == "numpy":
        return np.asarray(flat_llrs)
    else:  # output == "input"
        if is_jax_array(symbols):
            return flat_llrs  # already JAX
        # xp is NumPy or CuPy — convert via NumPy intermediate
        return xp.asarray(np.asarray(flat_llrs))


# =============================================================================
# Probabilistic Shaping (PS-QAM)
# =============================================================================


def constellation_power(
    constellation: ArrayType, pmf: ArrayType | None = None
) -> float:
    r"""Average symbol power ``E[|s|^2]`` of a constellation.

    For a uniform constellation (``pmf=None``) this is the unweighted mean
    power ``mean(|s|^2)``.  For probabilistic shaping it is the pmf-weighted
    power ``Σ_m P(s_m) |s_m|^2`` — the quantity written ``E_PS`` when the
    constellation is on the normalised grid (where it is ``< 1``).

    The value is *grid-agnostic*: it reports the average power of the
    constellation exactly as passed — ``≈1`` for a normalised uniform grid,
    ``E_PS < 1`` for a normalised shaped grid, or the raw integer-grid energy
    (e.g. ``10`` for an unnormalised 16-QAM).  Callers apply their own
    rescaling (``√E_PS`` on the received symbols, or ``1/√E_PS`` on the
    constellation) using the returned value.

    This is the single source of truth for PS-QAM power across the library;
    prefer it over the inline ``Σ pmf·|s|^2`` idiom.

    Parameters
    ----------
    constellation : array_like
        Constellation points, shape ``(M,)``.  NumPy, CuPy, or list.
    pmf : array_like, optional
        Per-point probabilities, shape ``(M,)``, aligned with
        ``constellation`` and summing to 1 (e.g. from
        :func:`maxwell_boltzmann`).  ``None`` (uniform) returns the
        unweighted mean power.

    Returns
    -------
    float
        Host scalar ``E[|s|^2]``.

    Raises
    ------
    ValueError
        If ``pmf`` is supplied and its length does not match the
        constellation.
    """
    const = np.asarray(to_device(constellation, "cpu"))
    energies = np.abs(const).astype(np.float64) ** 2
    if pmf is None:
        return float(np.mean(energies))
    pmf_arr = np.asarray(to_device(pmf, "cpu"), dtype=np.float64).ravel()
    if pmf_arr.shape[0] != energies.shape[0]:
        raise ValueError(
            f"pmf length {pmf_arr.shape[0]} does not match constellation "
            f"length {energies.shape[0]}."
        )
    return float(np.dot(pmf_arr, energies))


@lru_cache(maxsize=256)
def maxwell_boltzmann(order: int, nu: float) -> np.ndarray:
    """
    Computes the Maxwell-Boltzmann PMF over a QAM constellation.

    P(s_m) = exp(-nu * |s_m|^2) / Z(nu), where |s_m|^2 is on the unnormalized
    integer grid (literature-compatible nu scale).  Indexed consistently with
    ``gray_constellation``.

    Parameters
    ----------
    order : int
        QAM modulation order (must be a power of 2, e.g. 16, 64, 256).
    nu : float
        Shaping parameter nu >= 0.  nu = 0 returns the uniform distribution.
        Larger nu concentrates probability on lower-energy points.

    Returns
    -------
    np.ndarray
        PMF array of shape ``(order,)``, dtype float64, summing to 1.
        Cached by ``(order, nu)``.
    """
    # Energies computed on the unnormalized integer grid for literature-compatible ν.
    # The PMF index ordering matches gray_constellation (normalized), since both
    # use the same Gray-code index assignment.
    unnorm_constellation = gray_constellation("qam", order, normalize=False)
    if nu == 0.0:
        return np.full(order, 1.0 / order, dtype=np.float64)
    energies = np.abs(unnorm_constellation) ** 2  # (M,) unnormalized energies
    log_p = -nu * energies
    log_p -= log_p.max()  # shift for numerical stability before exp
    p = np.exp(log_p)
    return p / p.sum()


def ps_entropy(order: int, nu: float) -> float:
    r"""
    Computes the per-symbol Shannon entropy under a Maxwell-Boltzmann distribution.

    H(X) = -sum_m P(s_m) * log2(P(s_m)) [bits/symbol]

    Parameters
    ----------
    order : int
        QAM modulation order.
    nu : float
        MB shaping parameter nu >= 0. nu = 0 returns log2(order).

    Returns
    -------
    float
        Entropy in bits per symbol. In the range (0, log2(order)].
    """
    pmf = maxwell_boltzmann(order, nu)
    nonzero = pmf > 0
    return float(-np.sum(pmf[nonzero] * np.log2(pmf[nonzero])))


def optimal_nu(order: int, entropy_bits: float) -> tuple:
    r"""
    Finds the MB shaping parameter ``ν`` that achieves a target per-symbol entropy.

    Uses ``scipy.optimize.brentq`` to bisect on
    ``ps_entropy(order, ν) - entropy_bits = 0``.

    Parameters
    ----------
    order : int
        QAM modulation order.
    entropy_bits : float
        Target per-symbol entropy in bits. Must be in ``(0, log₂(order)]``.

    Returns
    -------
    nu : float
        Shaping parameter achieving the target entropy.
    achieved_entropy : float
        Actual entropy at the returned ``nu`` (may differ by ``< 1e-6`` bits).

    Raises
    ------
    ValueError
        If ``entropy_bits`` is outside ``(0, log₂(order)]``.
    """
    from scipy.optimize import brentq

    max_h = np.log2(order)
    if not (0 < entropy_bits <= max_h):
        raise ValueError(
            f"entropy_bits must be in (0, {max_h:.3f}] for {order}-QAM, "
            f"got {entropy_bits:.4f}"
        )
    if np.isclose(entropy_bits, max_h, atol=1e-8):
        return 0.0, float(max_h)

    def _obj(nu):
        return ps_entropy(order, nu) - entropy_bits

    # Grow upper bound until entropy drops below target
    nu_hi = 0.01
    while ps_entropy(order, nu_hi) > entropy_bits:
        nu_hi *= 10.0

    nu_opt = float(brentq(_obj, 0.0, nu_hi, xtol=1e-9, rtol=1e-9))
    return nu_opt, ps_entropy(order, nu_opt)


def sample_ps_symbols(
    num_symbols: int,
    order: int,
    pmf: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Draws QAM symbols from a Maxwell-Boltzmann distribution.

    Parameters
    ----------
    num_symbols : int
        Number of symbols to generate.
    order : int
        QAM modulation order.
    pmf : np.ndarray
        Symbol PMF of shape ``(order,)``. Typically from
        ``maxwell_boltzmann``.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Complex-valued symbols drawn from the MB distribution.
        Shape: ``(num_symbols,)``, dtype ``complex64``.
        All values lie exactly on the normalized QAM constellation grid.
    """
    rng = np.random.default_rng(seed)
    constellation = gray_constellation("qam", order).astype(np.complex64)
    indices = rng.choice(order, size=num_symbols, p=np.asarray(pmf, dtype=np.float64))
    return constellation[indices]
