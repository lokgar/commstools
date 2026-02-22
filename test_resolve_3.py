import numpy as xp
import commstools

n_symbols = 20
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

sig.equalize(method="lms", num_train_symbols=10, num_taps=11)
sig.resolve_symbols()

print("Resolved length:", sig.resolved_symbols.shape)
try:
    print(sig.evm())
    print("Done")
except Exception as e:
    print(f"EVM Exception: {e}")

