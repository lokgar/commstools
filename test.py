from commstools import Signal, set_backend
import numpy as np

set_backend("gpu")

samples = np.random.randn(1000) + 1j * np.random.randn(1000)
sig = Signal(samples=samples, sampling_rate=1e6, symbol_rate=1e5)

print(sig.backend)
print(type(sig.samples))
# sig.plot_psd()
