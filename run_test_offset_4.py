import numpy as xp
import commstools

n_symbols = 20
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

for t_test in [1,2,3,4,5,6,7,8,9,10,11]:
    n_samples = sig.samples.shape[-1]
    stride = 2
    n_sym = (n_samples - t_test + 1) // stride
    print(f"taps={t_test}, expected={n_sym}")

