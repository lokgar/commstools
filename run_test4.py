import numpy as xp
import commstools

n_symbols = 200
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

print("Original source symbols length:", sig.source_symbols.shape)
print("Samples length:", sig.samples.shape)

sig2 = sig.copy()
sig2.equalize(method="zf", channel_estimate=[1.0, 0])

print("After ZF equalize output y_hat length:", sig2.samples.shape)

