import numpy as np
import pprint

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

        if n_buf == 0:
            buf_x[buf_head % history_length] = x_b
            buf_y[buf_head % history_length] = y_b
            buf_head += 1
            n_buf += 1
            Sx += x_b; Sy += y_b; Sxx += x_b * x_b; Sxy += x_b * y_b
            continue

        if n_buf < min(10, history_length):
            # Constant extrapolation
            phi_pred = buf_y[(buf_head - 1) % history_length]
        else:
            # Linear extrapolation
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

phi_u = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
print("Original phase:", phi_u)
print("Corrected:", cycle_slip_loop(phi_u, 4, 10, np.pi/4))
