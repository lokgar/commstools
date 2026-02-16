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
map_bits :
    Maps bit sequences to complex/float symbols.
demap_symbols_hard :
    Performs hard-decision demapping from symbols to bits.
demap_symbols_soft :
    Computes Log-Likelihood Ratios (LLRs) for soft-decision decoding.
"""

import numpy as np

from functools import lru_cache

from .backend import ArrayType, dispatch
from .logger import logger


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
    # Force unipolar if the modulation string says so
    if "unipol" in modulation:
        unipolar = True

    # Extract core modulation scheme
    if "psk" in modulation:
        modulation = "psk"
    elif "qam" in modulation:
        modulation = "qam"
    elif "ask" in modulation or "pam" in modulation:
        modulation = "ask"
    else:
        # Fallback to last part for custom schemes if any
        if "-" in modulation:
            modulation = modulation.split("-")[-1]

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
    symbols: ArrayType,
    modulation: str,
    order: int,
    unipolar: bool = False,
) -> ArrayType:
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

    Returns
    -------
    array_like
        Sequence of bits (0s and 1s). Shape: (..., N_symbols * log2(order)).
        The backend (NumPy/CuPy) matches the input `symbols`.
    """
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
    constellation = xp.asarray(constellation)

    # 1. Find nearest constellation point (Hard Decision)
    # We expand dimensions to calculate all-to-all distances
    # symbols: (N,), constellation: (M,)
    # distances shape: (N, M)
    distances = xp.abs(symbols_flat[:, xp.newaxis] - constellation[xp.newaxis, :])
    indices = xp.argmin(distances, axis=1)

    # 2. Convert indices to bits
    k = int(np.log2(order))
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


def demap_symbols_soft(
    symbols: ArrayType,
    modulation: str,
    order: int,
    noise_var: float,
    method: str = "maxlog",
    vectorized: bool = True,
    unipolar: bool = False,
) -> ArrayType:
    """
    Compute Log-Likelihood Ratios (LLRs) for soft-decision demapping.

    LLRs represent the reliability of each bit. Positive values favor bit 0,
    while negative values favor bit 1 (assuming $0 \rightarrow +1$ and
    $1 \rightarrow -1$ mapping convention). The magnitude indicates confidence.

    Parameters
    ----------
    symbols : array_like
        Received noisy symbols. Shape: (..., N_symbols).
    modulation : {"psk", "qam", "ask"}
        Modulation type.
    order : int
        Modulation order.
    noise_var : float
        Noise variance per complex dimension ($\sigma^2$).
        For AWGN with unit-power symbols: $\sigma^2 = 0.5 \cdot 10^{-E_s/N_0 / 10}$.
    method : {"maxlog", "exact"}, default "maxlog"
        Computation algorithm. "maxlog" is much faster; "exact" uses log-sum-exp.
    vectorized : bool, default True
        If True, use high-performance vectorized operations.
        Automatically falls back to loops for extremely large arrays to
        prevent Out-Of-Memory (OOM) errors.
    unipolar : bool, default False
        Trigger unipolar demapping for ASK/PAM.

    Returns
    -------
    array_like
        LLR values. Shape: (..., N_symbols * log2(order)).

    Notes
    -----
    The Max-Log approximation simplifies the exact LLR:
    $LLR \approx \frac{1}{\sigma^2} (\min_{s \in S_1} |r-s|^2 - \min_{s \in S_0} |r-s|^2)$
    """
    logger.debug(
        f"Soft demapping {modulation.upper()} {order}-level (method={method})."
    )
    symbols, xp, _ = dispatch(symbols)

    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError(f"Order must be a power of 2, got {order}")

    # Capture original shape for output reshaping
    original_shape = symbols.shape
    symbols_flat = symbols.flatten()
    num_symbols = symbols_flat.size

    # Get constellation points indexed by symbol value
    constellation = gray_constellation(modulation, order, unipolar=unipolar)
    constellation = xp.asarray(constellation)

    # Build bit mapping table
    # Constellation points in 'constellation' array are already indexed by their bit value
    # (Natural Binary Indexing).
    # So the bits for constellation[s] are simply the bits of integer s.
    s_indices = xp.arange(order, dtype="int32")
    shifts = xp.arange(k - 1, -1, -1, dtype="int32")
    bits_table = ((s_indices[:, xp.newaxis] >> shifts) & 1).astype(
        "int32"
    )  # Shape: (M, k)

    # Avoid division by zero in noise variance
    sigma_sq = max(noise_var, 1e-20)

    # Memory estimate for fully vectorized: (N, M, k) tensor
    # Each element is float32 (4 bytes), threshold at ~500MB
    tensor_elements = num_symbols * order * k
    memory_threshold = 128 * 1024 * 1024  # 128M elements ~ 512MB for float32

    use_vectorized = vectorized and (tensor_elements < memory_threshold)

    if use_vectorized:
        # === FULLY VECTORIZED IMPLEMENTATION ===
        # Compute squared distances: |r - s|² for all symbols
        # symbols_flat: (N,), constellation: (M,)
        # distances_sq: (N, M)
        distances_sq = (
            xp.abs(symbols_flat[:, xp.newaxis] - constellation[xp.newaxis, :]) ** 2
        )

        # Transpose bits_table for broadcasting: (M, k) -> (k, M)
        bits_table_t = bits_table.T  # (k, M)

        # Create masks for all bits simultaneously
        # mask_0[b, m] = True if bit b of symbol m is 0
        mask_0 = bits_table_t == 0  # (k, M)
        mask_1 = bits_table_t == 1  # (k, M)

        if method == "maxlog":
            # Expand distances for all bits: (N, M) -> (N, 1, M) -> broadcast with (k, M)
            # Result: (N, k, M) via broadcasting
            dist_expanded = distances_sq[:, xp.newaxis, :]  # (N, 1, M)

            # Apply masks: where mask is False, set to inf
            # Broadcasting: (N, 1, M) * (k, M) -> (N, k, M)
            dist_0 = xp.where(mask_0, dist_expanded, xp.inf)  # (N, k, M)
            dist_1 = xp.where(mask_1, dist_expanded, xp.inf)  # (N, k, M)

            # Minimum over constellation points
            min_dist_0 = xp.min(dist_0, axis=2)  # (N, k)
            min_dist_1 = xp.min(dist_1, axis=2)  # (N, k)

            # LLR = (1/σ²) * (min_d1 - min_d0)
            llrs = (min_dist_1 - min_dist_0) / sigma_sq  # (N, k)

        elif method == "exact":
            # Exact LLR using log-sum-exp
            neg_exp = -distances_sq / sigma_sq  # (N, M)
            neg_exp_expanded = neg_exp[:, xp.newaxis, :]  # (N, 1, M)

            # Apply masks
            neg_exp_0 = xp.where(mask_0, neg_exp_expanded, -xp.inf)  # (N, k, M)
            neg_exp_1 = xp.where(mask_1, neg_exp_expanded, -xp.inf)  # (N, k, M)

            # Log-sum-exp with numerical stability
            max_0 = xp.max(neg_exp_0, axis=2, keepdims=True)
            max_1 = xp.max(neg_exp_1, axis=2, keepdims=True)

            max_0 = xp.where(xp.isinf(max_0), 0.0, max_0)
            max_1 = xp.where(xp.isinf(max_1), 0.0, max_1)

            log_sum_0 = max_0.squeeze(axis=2) + xp.log(
                xp.sum(xp.exp(neg_exp_0 - max_0), axis=2)
            )
            log_sum_1 = max_1.squeeze(axis=2) + xp.log(
                xp.sum(xp.exp(neg_exp_1 - max_1), axis=2)
            )

            llrs = log_sum_0 - log_sum_1  # (N, k)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'maxlog' or 'exact'.")

    else:
        # === LOOP-BASED IMPLEMENTATION (for large arrays) ===
        logger.debug(
            f"Using loop-based soft demapping (tensor size {tensor_elements} exceeds threshold)."
        )

        # Compute squared distances: (N, M)
        distances_sq = (
            xp.abs(symbols_flat[:, xp.newaxis] - constellation[xp.newaxis, :]) ** 2
        )

        llrs = xp.zeros((num_symbols, k), dtype=symbols_flat.real.dtype)

        for bit_idx in range(k):
            mask_0 = bits_table[:, bit_idx] == 0
            mask_1 = bits_table[:, bit_idx] == 1

            if method == "maxlog":
                dist_0 = xp.where(mask_0, distances_sq, xp.inf)
                dist_1 = xp.where(mask_1, distances_sq, xp.inf)
                min_dist_0 = xp.min(dist_0, axis=1)
                min_dist_1 = xp.min(dist_1, axis=1)
                llrs[:, bit_idx] = (min_dist_1 - min_dist_0) / sigma_sq

            elif method == "exact":
                neg_exp = -distances_sq / sigma_sq
                neg_exp_0 = xp.where(mask_0, neg_exp, -xp.inf)
                neg_exp_1 = xp.where(mask_1, neg_exp, -xp.inf)

                max_0 = xp.max(neg_exp_0, axis=1, keepdims=True)
                max_1 = xp.max(neg_exp_1, axis=1, keepdims=True)
                max_0 = xp.where(xp.isinf(max_0), 0.0, max_0)
                max_1 = xp.where(xp.isinf(max_1), 0.0, max_1)

                log_sum_0 = max_0.squeeze() + xp.log(
                    xp.sum(xp.exp(neg_exp_0 - max_0), axis=1)
                )
                log_sum_1 = max_1.squeeze() + xp.log(
                    xp.sum(xp.exp(neg_exp_1 - max_1), axis=1)
                )
                llrs[:, bit_idx] = log_sum_0 - log_sum_1
            else:
                raise ValueError(f"Unknown method: {method}. Use 'maxlog' or 'exact'.")

    # Reshape LLRs to match input symbol structure
    flat_llrs = llrs.flatten()

    if len(original_shape) > 1:
        new_shape = list(original_shape)
        new_shape[-1] = new_shape[-1] * k
        return flat_llrs.reshape(new_shape)
    else:
        return flat_llrs
