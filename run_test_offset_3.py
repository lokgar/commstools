import numpy as xp
import commstools

n_symbols = 20
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

for t_test in [1,2,3,4,5,6,7,8,9,10,11]:
    n_sym = sig.samples.shape[-1] // sps
    num_taps = t_test
    expected = n_sym - (num_taps - 1) // sps
    print(f"taps={t_test}, expected={expected}")

