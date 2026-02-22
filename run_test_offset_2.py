import numpy as xp
import commstools

n_symbols = 20
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

for num_taps in [5, 11, 21]:
    sig2 = sig.copy()
    sig2.equalize(method="lms", num_train_symbols=10, num_taps=num_taps)
    print(f"LMS taps={num_taps}, output length={sig2.samples.shape}")

