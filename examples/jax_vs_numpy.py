from time import time
import time

import numpy as np

import commstools.plotting as plotting
from commstools import get_backend, set_backend
from commstools.dsp.filtering import rrc_taps
from commstools.dsp.sequences import prbs
from commstools.waveforms import ook


def test(n=2**21):
    # Generate a Pseudo-Random Binary Sequence (PRBS)
    bits = prbs(length=n, order=31, seed=0x30F1CA55)

    # Define system parameters
    symbol_rate = 50e6
    sampling_rate = 400e6
    sps = int(sampling_rate / symbol_rate)
    rrc_rolloff = 0.7

    # Create shaped waveform
    sig_rrc = ook(
        bits,
        sampling_rate=sampling_rate,
        sps=sps,
        pulse_shape="rrc",
        filter_span=18,
        rrc_rolloff=rrc_rolloff,
    )

    # Matched filter
    rrc_filter = rrc_taps(sps=sps, rolloff=rrc_rolloff, span=18)

    # Some distortion
    sig_received = sig_rrc.update(
        sig_rrc.samples + 0.2 * np.random.randn(*sig_rrc.samples.shape)
    )

    #
    sig_matchedfilt = sig_received.matched_filter(
        pulse_taps=rrc_filter, taps_normalization="unity_gain", normalize_output=False
    )


if __name__ == "__main__":
    print("Testing numpy n=2^21...")
    set_backend("numpy")
    t = time.time()
    test()
    print()
    print("TIME NUMPY: ", time.time() - t)
    print()

    print("Testing jax n=2^21 1...")
    set_backend("jax")
    t = time.time()
    test()
    print()
    print("TIME JAX 1: ", time.time() - t)
    print()

    print("Testing jax n=2^21 2...")
    set_backend("jax")
    t = time.time()
    test()
    print()
    print("TIME JAX 2: ", time.time() - t)
    print()

    print("-" * 79)
    print("-" * 79)

    print("Testing numpy n=2^22...")
    set_backend("numpy")
    t = time.time()
    test(n=2**22)
    print()
    print("TIME NUMPY 1: ", time.time() - t)
    print()

    print("Testing jax n=2^22 1...")
    set_backend("jax")
    t = time.time()
    test(n=2**22)
    print()
    print("TIME JAX 1: ", time.time() - t)
    print()

    print("Testing jax n=2^22 2...")
    set_backend("jax")
    t = time.time()
    test(n=2**22)
    print()
    print("TIME JAX 2: ", time.time() - t)
    print()

    print("-" * 79)
    print("-" * 79)

    print("Testing numpy n=2^23...")
    set_backend("numpy")
    t = time.time()
    test(n=2**23)
    print()
    print("TIME NUMPY 1: ", time.time() - t)
    print()

    print("Testing jax n=2^23 1...")
    set_backend("jax")
    t = time.time()
    test(n=2**23)
    print()
    print("TIME JAX 1: ", time.time() - t)
    print()

    print("Testing jax n=2^23 2...")
    set_backend("jax")
    t = time.time()
    test(n=2**23)
    print()
    print("TIME JAX 2: ", time.time() - t)
    print()

    print("-" * 79)
    print("-" * 79)

    print("Testing numpy n=2^24...")
    set_backend("numpy")
    t = time.time()
    test(n=2**24)
    print()
    print("TIME NUMPY 1: ", time.time() - t)
    print()

    print("Testing jax n=2^24 1...")
    set_backend("jax")
    t = time.time()
    test(n=2**24)
    print()
    print("TIME JAX 1: ", time.time() - t)
    print()

    print("Testing jax n=2^24 2...")
    set_backend("jax")
    t = time.time()
    test(n=2**24)
    print()
    print("TIME JAX 2: ", time.time() - t)
    print()
