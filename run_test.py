import numpy as xp
import commstools.equalizers as equalizers

n = 256
channel = xp.array([1.0, 0.5], dtype=xp.complex64)

rng = xp.random.RandomState(42)
tx = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)

rx = xp.convolve(tx, channel, mode="full")[:n]
equalized = equalizers.zf_equalizer(rx, channel)
print("Max diff:", xp.max(xp.abs(equalized - tx)))

channel2 = xp.array([1.0, -0.9, 0.1], dtype=xp.complex64)
tx2 = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)
rx2 = xp.convolve(tx2, channel2, mode="full")[:n]
noise = 0.1 * (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)
rx2_noisy = rx2 + noise

zf_out = equalizers.zf_equalizer(rx2_noisy, channel2)
mmse_out = equalizers.zf_equalizer(rx2_noisy, channel2, noise_variance=0.01)

mse_zf = xp.mean(xp.abs(zf_out - tx2) ** 2)
mse_mmse = xp.mean(xp.abs(mmse_out - tx2) ** 2)

print("ZF MSE:", mse_zf)
print("MMSE MSE:", mse_mmse)

