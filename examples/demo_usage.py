import numpy as np

from commstools import Signal, set_backend
from commstools.dsp.sequences import prbs
from commstools.waveforms import ook

set_backend("jax")

# Generate a Pseudo-Random Binary Sequence (PRBS)
bits = prbs(length=100000, order=31, seed=4202460010)

# Define system parameters
symbol_rate = 50e6
sampling_rate = 400e6
sps = int(sampling_rate / symbol_rate)

sig_impulse = ook(bits, sampling_rate=sampling_rate, sps=sps, pulse_shape="none")

fig1, _ = sig_impulse.plot_signal(num_symbols=20)
fig2, _ = sig_impulse.plot_psd()

fig1.savefig("1.png")
fig2.savefig("2.png")
