import numpy as xp
import commstools

n_symbols = 20
sps = 2
sig = commstools.Signal.psk(symbol_rate=1e6, num_symbols=n_symbols, order=4, pulse_shape="rrc", sps=sps, seed=42)

def check_offset(num_taps):
    center = num_taps // 2
    if sps == 2 and center % 2 != 0:
        center -= 1
    return center

for t_test in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
    print(f"taps={t_test}, center={t_test // 2}, sps=2 -> offset={check_offset(t_test)}, delay in symbols={check_offset(t_test) // 2}")

