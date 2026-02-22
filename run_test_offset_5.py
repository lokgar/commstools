import numpy as xp
import commstools

n_symbols = 20
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

for t_test in [11]:
    n_samples = sig.samples.shape[-1]
    stride = 2
    n_sym = n_samples // stride
    
    pad_len = t_test - 1
    
    samples_pad = xp.pad(sig.samples, ((0,0),(0, pad_len))) if sig.samples.ndim > 1 else xp.pad(sig.samples, (0, pad_len))
    n_sym_padded = (samples_pad.shape[-1] - t_test + 1) // stride
    print(f"taps={t_test}, expected={n_sym_padded}")

