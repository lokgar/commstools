"""
Symbol mapping and constellation generation.

This module handles the mapping of bits to complex symbols and vice versa.
It supports:
- Gray coding and decoding.
- Constellation generation for PSK, QAM (Square, Cross, Rectangular, Star), and ASK.
- Bit-to-symbol mapping.
"""

import numpy as np

from .backend import ArrayType, dispatch
from .logger import logger


def gray_code(n: int) -> np.ndarray:
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


def gray_constellation(modulation: str, order: int) -> ArrayType:
    """
    Generate constellation points with Gray mapping.

    The returned array is indexed by the symbol value (0 to order-1).
    constellation[s] is the complex/float value for symbol s.

    Args:
        modulation: Modulation type ('psk', 'qam', 'ask').
        order: Modulation order.

    Returns:
        Array of constellation points (NumPy).
    """
    logger.debug(
        f"Generating Gray-coded constellation: modulation={modulation}, order={order}"
    )
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

    return result


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

    gray = gray_code(k)
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

    Args:
        order: Modulation order (must be 8).

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


def _gray_qam_cross(order: int) -> np.ndarray:
    """
    Generate Cross-QAM constellation (M = 2^(2k+1)) using a Quasi-Gray mapping (Numpy).

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

    return final_i_vals + 1j * final_q_vals


def map_bits(bits: ArrayType, modulation: str, order: int) -> ArrayType:
    """
    Map a sequence of bits to constellation symbols.

    Args:
        bits: Input array of bits (0s and 1s).
        modulation: Modulation type ('psk', 'qam', 'ask').
        order: Modulation order.

    Returns:
        Array of complex symbols on the same backend as bits.
    """
    logger.debug(f"Mapping bits to {modulation.upper()} {order}-level symbols.")
    bits, xp, _ = dispatch(bits)

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
    # We can reshape to (num_symbols, k) and multiply by powers of 2
    bits_reshaped = bits.reshape((num_symbols, k))

    # Powers of 2: [2^(k-1), ..., 2^0] for MSB first mapping
    powers = 2 ** xp.arange(k - 1, -1, -1, dtype=int)
    # Perform dot product
    indices = xp.sum(bits_reshaped * powers, axis=1)

    # Get constellation (returns NumPy)
    constellation = gray_constellation(modulation, order)

    # Ensure constellation is on the same backend as indices
    constellation = xp.asarray(constellation)

    # Map indices to points
    return constellation[indices]
