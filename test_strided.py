import numpy as xp

num_ch = 2
N = 100
B = 10
N_fft = 16
L = N_fft - B + 1 # 7
samples = xp.arange(num_ch * N, dtype=xp.complex64).reshape(num_ch, N)

pad_left = L - 1
num_blocks = (N + B - 1) // B
pad_right = num_blocks * B - N
samples_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))

stride = samples_padded.strides
windows = xp.lib.stride_tricks.as_strided(
    samples_padded,
    shape=(num_ch, num_blocks, N_fft),
    strides=(stride[0], B * stride[1], stride[1])
)

Y = xp.fft.fft(windows, n=N_fft, axis=-1)
x_hat = xp.fft.ifft(Y, n=N_fft, axis=-1)

valid = x_hat[:, :, L-1:L-1+B]
out = valid.reshape(num_ch, -1)[:, :N]

# The ifft of fft introduces floating point errors, we need to compare using allclose properly
print(xp.max(xp.abs(out - samples)))
assert xp.allclose(out, samples, atol=1e-5)
