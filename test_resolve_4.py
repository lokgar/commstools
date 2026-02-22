import numpy as xp
import commstools

n_symbols = 10000
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

sig.matched_filter()
print("Source length:", sig.source_symbols.shape)
print("Samples length:", sig.samples.shape)

sig.equalize(method="zf", channel_estimate=[1.0, 0])
sig.resolve_symbols()

print("Resolved length:", sig.resolved_symbols.shape)
try:
    print(sig.evm(discard_training=True))
    print("Done")
except Exception as e:
    print(f"EVM Exception: {e}")

