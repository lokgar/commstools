"""
Symbol mapping and constellation generation.

This module handles the mapping of bits to complex symbols and vice versa.
It supports:
- Gray coding and decoding.
- Constellation generation for PSK, QAM (Square, Cross, Rectangular, Star), and ASK.
- Bit-to-symbol mapping.
"""

import numpy as np

from .backend import ArrayType, ensure_on_backend, get_xp


def _gray_code_np(n: int) -> np.ndarray:
    """
    Internal numpy implementation of Gray code.

    Args:
        n: Number of bits.

    Returns:
        Array of integers representing the Gray code sequence.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([0], dtype=int)

    # Vectorized implementation for speed
    i = np.arange(1 << n, dtype=int)
    return i ^ (i >> 1)


def gray_code(n: int) -> ArrayType:
    """
    Generate Gray code sequence of length 2^n.

    Args:
        n: Number of bits.

    Returns:
        Array of integers representing the Gray code sequence.
    """
    return ensure_on_backend(_gray_code_np(n))


def gray_constellation(modulation: str, order: int) -> ArrayType:
    """
    Generate constellation points with Gray mapping.

    The returned array is indexed by the symbol value (0 to order-1).
    constellation[s] is the complex/float value for symbol s.

    Args:
        modulation: Modulation type ('psk', 'qam', 'ask').
        order: Modulation order.

    Returns:
        Array of constellation points on the active backend.
    """
    modulation = modulation.lower()

    if order < 2:
        raise ValueError("Order must be at least 2 for modulation")

    if modulation == "psk":
        result = _gray_psk(order)
    elif modulation == "ask":
        result = _gray_ask(order)
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

    return ensure_on_backend(result)


def _gray_psk(order: int) -> np.ndarray:
    """
    Generate M-PSK constellation with Gray mapping (Numpy).

    Args:
        order: Modulation order (must be a power of 2).

    Returns:
        Complex array of constellation points.
    """
    # Bits per symbol
    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError("Order must be power of 2 for psk")

    # Gray codes for k bits
    gray = _gray_code_np(k)

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


def _gray_ask(order: int) -> np.ndarray:
    """
    Generate M-ASK (Amplitude Shift Keying) constellation with Gray mapping (Numpy).
    ATTENTION: Always returns bipolar values centered at 0.

    Args:
        order: Modulation order (must be a power of 2).

    Returns:
        Real array of constellation points (centered at 0).
    """
    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError("Order must be power of 2 for ask")

    gray = _gray_code_np(k)
    # Points are symmetric: -M+1, -M+3, ..., M-3, M-1
    points = np.linspace(-order + 1, order - 1, order)

    constellation = np.zeros(order, dtype=float)
    constellation[gray] = points
    return constellation


def _gray_qam_square(order: int) -> np.ndarray:
    """
    Generate Square M-QAM constellation (M = 2^2k) with Gray mapping (Numpy).

    Constructed as the Cartesian product of two Gray-coded ASK constellations.

    Args:
        order: Modulation order (must be an even power of 2).

    Returns:
        Complex array of constellation points.
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

    return i_vals + 1j * q_vals


def _gray_qam_8_rect() -> np.ndarray:
    """
    Generate 8-QAM constellation with Rectangular Gray mapping (Numpy).

    Standard rectangular 8-QAM with Gray mapping:
        0 (000) -> (1, 1)
        1 (001) -> (1, 3)
        2 (010) -> (3, 3)
        3 (011) -> (3, 1)
        4 (100) -> (1, -1)
        5 (101) -> (1, -3)
        6 (110) -> (3, -3)
        7 (111) -> (3, -1)

    Args:
        order: Modulation order (must be 8). This arg is kept for signature consistency.

    Returns:
        Complex array of constellation points.
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


def _gray_qam_8_star() -> np.ndarray:
    """
    Generate 8-QAM constellation with 'Star' Gray mapping (Numpy).

    Specific circular constellation maximizing Minimum Euclidean Distance.
    Outer points are at distance (1+sqrt(3)) on axes.
    Inner points are at (±1, ±1).

    Args:
        order: Modulation order (must be 8). This arg is kept for signature consistency.

    Returns:
        Complex array of constellation points.
    """

    # Optimal scaling for star 8-QAM
    # Inner square at ±1. R1 = sqrt(2).
    # Outer points at R2 = 1 + sqrt(3) ~ 2.732.
    a = 1.0 + np.sqrt(3.0)

    # 3-bit Gray mapping indices
    # 0 (000) -> Outer Right
    # 1 (001) -> Inner Top-Right
    # 2 (010) -> Inner Top-Left
    # 3 (011) -> Outer Top
    # 4 (100) -> Inner Bottom-Right
    # 5 (101) -> Outer Bottom
    # 6 (110) -> Outer Left
    # 7 (111) -> Inner Bottom-Left

    points = np.zeros(8, dtype=complex)
    points[0] = a
    points[1] = 1.0 + 1.0j
    points[2] = -1.0 + 1.0j
    points[3] = a * 1j
    points[4] = 1.0 - 1.0j
    points[5] = -a * 1j
    points[6] = -a
    points[7] = -1.0 - 1.0j

    return points


def _gray_qam_cross(order: int) -> np.ndarray:
    """
    Generate Cross-QAM constellation (M = 2^(2k+1)) using a Quasi-Gray mapping (Numpy).

    For Cross-QAM (e.g., 32-QAM, 128-QAM), a perfect Gray code (Hamming distance 1
    for all nearest neighbors) is impossible on the standard cross shape.
    This implementation uses a "Folded" Rectangular mapping approach:
    1. Start with a Rectangular Grid of size 2^(k+1) x 2^k.
    2. Identify "Wing" columns on the left and right.
    3. Rotate and stack these Wings onto the Top and Bottom of the central body.

    This technique yields the standard Cross shape (e.g., 6x6 for 32-QAM) and
    minimizes Gray violation errors at the boundary seams (Wing <-> Body).

    Args:
        order: Modulation order (must be an odd power of 2).

    Returns:
        Complex array of constellation points.
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
    gray_i_seq = _gray_code_np(n)
    gray_q_seq = _gray_code_np(m)

    # 2. Build Inverse Lookup Tables (Symbol -> Geometric Index)
    # This allows us to map the input Gray-coded symbols to their 2D grid coordinates (geo_i, geo_q).
    # We perform the "folding" in the Geometric domain, then map back to coordinates.
    # Use backend operations to maintain compatibility.
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
    # We remove 'n_shift' columns from each side (Wings) and stack them on top/bottom.
    # For 32-QAM (n=3, W=8): n_shift = 1. We move Col 0 and Col 7.
    # For 128-QAM (n=4, W=16): n_shift = 2. We move Cols 0,1 and 14,15.
    n_shift = 0
    if n >= 3:
        n_shift = 2 ** (n - 3)

    if n_shift == 0:
        # Fallback for purely rectangular cases (e.g. 8-QAM if it were supported)
        val_i = (-width + 1 + 2 * geo_i).astype(float)
        val_q = (-height + 1 + 2 * geo_q).astype(float)
        return val_i + 1j * val_q

    # 6. Identify Wings
    mask_left = geo_i < n_shift
    mask_right = geo_i >= (width - n_shift)

    # 7. Calculate Rotation/Translation
    # We map the Wing blocks (Height x n_shift) to the Caps (n_shift x Height)
    # Center the mapped block horizontally relative to the main body.
    center_offset = (width - height) // 2

    # Left Wing -> Top Cap
    # Rotation: New I = Old Q + Offset.
    rotated_i_left = geo_q + center_offset

    # New Q = Top Extension (Stacking upwards)
    # The inner-most wing column maps to the row just above the Body.
    rotated_q_left = height + (n_shift - 1 - geo_i)

    # Right Wing -> Bottom Cap
    # Rotation: New I = Old Q + Offset.
    rotated_i_right = geo_q + center_offset

    # New Q = Bottom Extension (Stacking downwards)
    # The inner-most wing column maps to the row just below the Body.
    rotated_q_right = -1 - (geo_i - (width - n_shift))

    # Apply Updates
    geo_i_final = geo_i.copy() if hasattr(geo_i, "copy") else geo_i
    geo_q_final = geo_q.copy() if hasattr(geo_q, "copy") else geo_q

    geo_i_final = np.where(mask_left, rotated_i_left, geo_i_final)
    geo_i_final = np.where(mask_right, rotated_i_right, geo_i_final)

    geo_q_final = np.where(mask_left, rotated_q_left, geo_q_final)
    geo_q_final = np.where(mask_right, rotated_q_right, geo_q_final)

    # Convert Final Geometric Indices to Values
    # Note: geo_q_final can be outside 0..H-1 range, which is intended.
    final_i_vals = (-width + 1 + 2 * geo_i_final).astype(float)
    final_q_vals = (-height + 1 + 2 * geo_q_final).astype(float)

    return final_i_vals + 1j * final_q_vals


def map_bits(bits: ArrayType, modulation: str, order: int) -> ArrayType:
    """
    Map a sequence of bits to constellation symbols.

    Args:
        bits: Input array of bits (0s and 1s).
        modulation: Modulation type ('psk', 'qam', 'ask').
        order: Modulation order.

    Returns:
        Array of complex symbols.
    """
    bits = ensure_on_backend(bits)
    xp = get_xp()

    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError(f"Order must be a power of 2, got {order}")

    if len(bits) % k != 0:
        raise ValueError(
            f"Number of bits ({len(bits)}) must be divisible by bits per symbol ({k})"
        )

    # Reshape bits into symbols (N/k, k)
    # We need to process this carefully to remain backend-agnostic efficiently
    # For now, we assume bits are 0/1 integers
    num_symbols = len(bits) // k

    # Pack bits into integer indices
    # We can reshape to (num_symbols, k) and multiply by powers of 2
    bits_reshaped = bits.reshape((num_symbols, k))

    # Powers of 2: [2^(k-1), ..., 2^0] for MSB first mapping
    powers = 2 ** xp.arange(k - 1, -1, -1, dtype=int)
    # Perform dot product
    indices = xp.sum(bits_reshaped * powers, axis=1)

    # Get constellation
    constellation = gray_constellation(modulation, order)

    # Map indices to points
    return constellation[indices]
