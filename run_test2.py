import numpy as xp
import commstools

n_symbols = 200
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

print("Original source symbols length:", sig.source_symbols.shape)
print("Samples length:", sig.samples.shape)

sig.matched_filter()
print("After matched filter samples length:", sig.samples.shape)

sig2 = sig.copy()
sig2.equalize(method="lms", num_train_symbols=100, num_taps=11)

print("After LMS equalize symbols length:", sig2.samples.shape)
try:
    sig2.resolve_symbols()
    print("Resolved length:", sig2.resolved_symbols.shape)
except Exception as e:
    print("Resolve error:", e)

