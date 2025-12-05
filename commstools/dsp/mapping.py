import numpy as np
from ..core.backend import get_backend, ArrayType


def gray_code(n: int) -> ArrayType:
    """
    Generate Gray code sequence of length 2^n.

    Args:
        n: Number of bits.

    Returns:
        Array of integers representing the Gray code sequence.
    """
    backend = get_backend()
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return backend.array([0], dtype=int)

    # We'll compute on CPU/numpy for simplicity and transfer to backend.
    gray = np.zeros(1 << n, dtype=int)
    for i in range(1, 1 << n):
        gray[i] = i ^ (i >> 1)

    return backend.asarray(gray)


# TODO: fix mapping for qam of 2^(2k+1) orders
def gray_constellation(modulation: str, order: int) -> ArrayType:
    """
    Generate constellation points with Gray mapping.

    The returned array is indexed by the symbol value (0 to order-1).
    constellation[s] is the complex/float value for symbol s.

    Args:
        modulation: Modulation type ('psk', 'qam', 'ask').
        order: Modulation order (e.g., 4, 16, 64).

    Returns:
        Array of constellation points on the active backend.
    """
    backend = get_backend()
    modulation = modulation.lower()

    if modulation == "psk":
        # M-PSK
        # Geometric points: phases 0 to 2pi (exclusive)
        # We want P[i] correspond to Gray[i]

        # Bits per symbol
        k = int(np.log2(order))
        if 2**k != order:
            raise ValueError(f"Order must be power of 2 for {modulation}")

        # Gray codes for k bits
        gray = gray_code(k)

        # BPSK Special Case: strictly real values
        if order == 2:
            points = backend.array([-1.0, 1.0], dtype=float)
            constellation = backend.zeros(order, dtype=float)
            constellation[gray] = points
            return constellation

        # Geometric points (phases)
        # Usually starts at 0 or offset. Standard M-PSK often starts at 0.
        phases = backend.arange(order) * 2 * backend.pi / order
        points = backend.exp(1j * phases)

        # Map: points[i] corresponds to symbol gray[i]
        # We need constellation[symbol] = point
        # So constellation[gray[i]] = points[i]
        constellation = backend.zeros(order, dtype=complex)
        constellation[gray] = points
        return constellation

    elif modulation == "ask":
        # M-ASK
        # Bipolar constellation centered at 0
        # Points: -M+1, -M+3, ..., M-1

        k = int(np.log2(order))
        if 2**k != order:
            raise ValueError(f"Order must be power of 2 for {modulation}")

        gray = gray_code(k)

        points = backend.linspace(-order + 1, order - 1, order)

        constellation = backend.zeros(order, dtype=float)
        constellation[gray] = points
        return constellation

    elif modulation == "qam":
        # M-QAM
        # Support both Square QAM (M=2^2k) and Cross-QAM (M=2^(2k+1))
        # Note: 8-QAM (Cross) is explicitly not supported as non-standard.

        k = int(np.log2(order))
        if 2**k != order:
            raise ValueError(f"Order must be power of 2 for {modulation}")

        # Check for 8-QAM
        if order == 8:
            raise ValueError("8-QAM is not supported.")

        # Determine grid dimensions
        # For Square QAM: n = m = k/2.
        # For Cross QAM: n = ceil(k/2), m = floor(k/2).
        # We start with a Rectangular QAM of size 2^n x 2^m.
        n = (k + 1) // 2
        m = k // 2

        # Generate Gray-coded ASK axes (reusing ASK logic)
        pam_i = gray_constellation("ask", 2**n)
        pam_q = gray_constellation("ask", 2**m)

        # Create Rectangular Grid
        # Using vectorization
        # Symbol S (k bits) -> (n bits for I, m bits for Q)
        # We assume high bits for I, low bits for Q for the initial Rectangular map
        s = backend.arange(order)
        mask_q = (1 << m) - 1

        idx_i = s >> m
        idx_q = s & mask_q

        i_vals = pam_i[idx_i]
        q_vals = pam_q[idx_q]

        # If Square QAM (k is even), n=m, we are done.
        if k % 2 == 0:
            return i_vals + 1j * q_vals

        # If Cross QAM (k is odd), n = m + 1.
        # We have a Rectangular constellation (Width > Height).
        # Width W = 2^n, Height H = 2^m.
        # We want to transform this into a "Cross" shape which is roughly square.
        # Transformation: Move outer columns to top/bottom caps.

        # Calculate parameters
        # Shift amount (number of columns to move from each side)
        # N_shift = 2^(n-3) for n >= 3. (e.g. 1 for 32-QAM n=3, 2 for 128-QAM n=4)

        n_shift = 0
        if n >= 3:
            n_shift = 2 ** (n - 3)

        # Current I-range max index (in ASK array) is 2^n - 1.
        # We remove N_shift columns from each side. Each column has width 2 in ASK values.
        # Threshold = I_max - Width_Removed.
        # Width_Removed = n_shift * 2.

        i_max = 2**n - 1
        i_cutoff = i_max - n_shift * 2

        # Points to move
        # We use numpy/backend conditionals

        # Identify points to move
        move_mask = backend.abs(i_vals) > i_cutoff

        # New positions
        # Standard Cross-QAM folding:
        # If I > Cutoff (Right side): Move to Top.
        # Folded I is Q_old.
        # Folded Q = sign(I) * (Q_max + 2 + (|I| - i_cutoff - 2))

        q_max = 2**m - 1

        # We need to perform the update.
        # Cast to float/complex if not already (ASK returns float)
        final_i = i_vals.copy() if hasattr(i_vals, "copy") else i_vals
        final_q = q_vals.copy() if hasattr(q_vals, "copy") else q_vals

        # Folded I is Q_old
        folded_i = q_vals

        # Folded Q
        abs_i = backend.abs(i_vals)
        sign_i = backend.where(i_vals > 0, 1.0, -1.0)

        folded_q = sign_i * (q_max + 2 + (abs_i - i_cutoff - 2))

        # Apply only where mask is true
        final_i = backend.where(move_mask, folded_i, i_vals)
        final_q = backend.where(move_mask, folded_q, q_vals)

        return final_i + 1j * final_q

    else:
        raise ValueError(f"Unsupported modulation type: {modulation}")
