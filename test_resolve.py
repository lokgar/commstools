import numpy as xp
import commstools

n_symbols = 20
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

print("Pre equalize:")
print("Source length:", sig.source_symbols.shape)
print("Samples length:", sig.samples.shape)

sig.equalize(method="lms", num_train_symbols=10, num_taps=11)

print("\nPost equalize:")
print("Source length:", sig.source_symbols.shape)
print("Samples length:", sig.samples.shape)
try:
    sig.evm()
except Exception as e:
    print(f"EVM Exception: {e}")

