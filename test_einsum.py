import numpy as xp

num_ch = 2
num_blocks = 10
N_fft = 16

Wk = xp.arange(N_fft * num_ch * num_ch, dtype=xp.complex64).reshape((N_fft, num_ch, num_ch))
Y = xp.arange(num_ch * num_blocks * N_fft, dtype=xp.complex64).reshape((num_ch, num_blocks, N_fft))

# Expected loop outcome
X_hat_f_loop = xp.zeros((num_ch, num_blocks, N_fft), dtype=xp.complex64)
for i in range(num_blocks):
    Y_block = Y[:, i, :] # (num_ch, N_fft)
    Y_t = Y_block.T[:, :, None] # (N_fft, num_ch, 1)
    X_hat_t = Wk @ Y_t # (N_fft, num_ch, 1)
    X_hat_f_loop[:, i, :] = X_hat_t[..., 0].T

X_hat_f_einsum = xp.einsum('k c j, j b k -> c b k', Wk, Y)
assert xp.allclose(X_hat_f_loop, X_hat_f_einsum)
print("SUCCESS EINSUM")
