import numpy as np
from commstools.dsp import sequences, waveforms
from commstools import SystemConfig, set_config

# Setup config
config = SystemConfig(sampling_rate=1e6, samples_per_symbol=4)
set_config(config)

# Generate bits
bits = sequences.prbs(length=10)
print(f"Bits: {bits}")

# Generate OOK Signal (Rect)
sig_rect = waveforms.ook(bits, pulse_type="rect")
print(f"Rect Signal shape: {sig_rect.samples.shape}")
print(f"Rect Signal samples (first 10): {sig_rect.samples[:10]}")

# Generate OOK Signal (Impulse)
sig_imp = waveforms.ook(bits, pulse_type="impulse")
print(f"Impulse Signal shape: {sig_imp.samples.shape}")
print(f"Impulse Signal samples (first 10): {sig_imp.samples[:10]}")
