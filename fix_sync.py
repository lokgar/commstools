import numpy as np
import pprint

# replicate the numba kernel logic for debugging
def cycle_slip_loop(phi_u, symmetry, history_length, threshold):
    two_pi = 2.0 * np.pi
    quantum = two_pi / float(symmetry)
    B = len(phi_u)

    buf_x = np.empty(history_length, dtype=np.float64)
    buf_y = np.empty(history_length, dtype=np.float64)
    buf_head = 0
    n_buf = 0

    Sx = 0.0; Sy = 0.0; Sxx = 0.0; Sxy = 0.0

    out = phi_u.copy()
    for b in range(B):
        x_b = float(b)
        y_b = out[b]

        if n_buf < 2:
            buf_x[buf_head % history_length] = x_b
            buf_y[buf_head % history_length] = y_b
            buf_head += 1
            n_buf += 1
            Sx += x_b; Sy += y_b; Sxx += x_b * x_b; Sxy += x_b * y_b
            continue

        n_f = float(n_buf)
        denom = n_f * Sxx - Sx * Sx
        if abs(denom) > 1e-30:
            slope = (n_f * Sxy - Sx * Sy) / denom
            intercept = (Sy - slope * Sx) / n_f
        else:
            slope = 0.0
            intercept = Sy / n_f
        phi_pred = slope * x_b + intercept

        diff = y_b - phi_pred
        k = round(diff / quantum)
        if abs(diff) > threshold and k != 0:
            out[b] -= float(k) * quantum
            y_b = out[b]

        if n_buf == history_length:
            old_idx = buf_head % history_length
            ox = buf_x[old_idx]; oy = buf_y[old_idx]
            Sx -= ox; Sy -= oy; Sxx -= ox * ox; Sxy -= ox * oy
            buf_x[old_idx] = x_b; buf_y[old_idx] = y_b
            Sx += x_b; Sy += y_b; Sxx += x_b * x_b; Sxy += x_b * y_b
            buf_head += 1
        else:
            idx = buf_head % history_length
            buf_x[idx] = x_b; buf_y[idx] = y_b
            Sx += x_b; Sy += y_b; Sxx += x_b * x_b; Sxy += x_b * y_b
            buf_head += 1
            n_buf += 1
    return out

phi_u = np.array([0.0, np.pi/2, np.pi/2, np.pi/2, np.pi/2])
print("Original phase:", phi_u)
print("Corrected:", cycle_slip_loop(phi_u, 4, 10, np.pi/4))
