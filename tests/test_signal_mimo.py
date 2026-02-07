from commstools.core import Signal, SingleCarrierFrame


def test_signal_generate_mimo(backend_device, xp):
    # Test MIMO generation via factories
    sig = Signal.qam(order=4, num_symbols=100, sps=4, symbol_rate=1e6, num_streams=2)

    # Check shape: (num_streams, num_symbols * sps)
    expected_samples = 100 * 4
    assert sig.samples.shape == (2, expected_samples)
    assert sig.num_streams == 2
    assert sig.sps == 4.0

    # Check if streams are not identical (random seed should diverge or be handled)
    # With random_symbols(total_symbols), they should be different segments
    assert not xp.allclose(sig.samples[0], sig.samples[1])


def test_frame_mimo_generation(backend_device, xp):
    # Test Frame MIMO support
    frame = SingleCarrierFrame(
        payload_len=100,
        symbol_rate=1e6,
        num_streams=2,
        pilot_pattern="none",
        guard_type="zero",
        guard_len=10,
    )

    sig = frame.generate_sequence()
    # Length: 100 payload + 10 guard = 110 symbols
    assert sig.samples.shape == (2, 110)
    assert sig.num_streams == 2

    # Check guard interval (zeros)
    # Last 10 samples (axis=-1)
    assert xp.all(sig.samples[:, -10:] == 0)


def test_frame_mimo_pilots(backend_device, xp):
    # Test Frame with Pilots and MIMO
    frame = SingleCarrierFrame(
        payload_len=10,
        symbol_rate=1e6,
        num_streams=2,
        pilot_pattern="comb",
        pilot_period=2,
    )
    # len=10 payload. period=2 -> 1 pilot, 1 data.
    # data_per_period = 1.
    # total len = 10 data -> 10 periods -> 20 symbols.

    sig = frame.generate_sequence()
    assert sig.samples.shape == (2, 20)

    # Check mask and body
    mask, _ = frame._generate_pilot_mask()
    # Mask is 1D (time), applicable to all streams
    assert len(mask) == 20
    assert xp.sum(mask) == 10


def test_frame_mimo_preamble_broadcasting(backend_device, xp):
    # Test 1D preamble broadcasting to 2 streams
    preamble = xp.ones(10)
    frame = SingleCarrierFrame(
        payload_len=20, symbol_rate=1e6, num_streams=2, preamble=preamble
    )

    sig = frame.generate_sequence()
    # Total: 10 preamble + 20 payload = 30
    assert sig.samples.shape == (2, 30)

    # Check preamble on both streams
    # (Channels, Time)
    assert xp.allclose(sig.samples[0, :10], preamble)
    assert xp.allclose(sig.samples[1, :10], preamble)


def test_frame_mimo_waveform(backend_device, xp):
    # Test generate_waveform with MIMO
    frame = SingleCarrierFrame(payload_len=10, symbol_rate=1e6, num_streams=2)

    sig = frame.generate_waveform(sps=4, pulse_shape="rect")
    # 10 symbols * 4 sps = 40 samples
    assert sig.samples.shape == (2, 40)
    assert sig.sps == 4.0
