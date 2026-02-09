from commstools.core import SingleCarrierFrame


def test_sc_frame_none(backend_device, xp):
    frame = SingleCarrierFrame(payload_len=100, symbol_rate=1e6, pilot_pattern="none")
    sig = frame.generate_waveform(sps=1, pulse_shape="none")
    assert len(sig.samples) == 100
    assert sig.symbol_rate == 1e6
    assert sig.frame_info is not None
    assert sig.frame_info.payload_len == 100


def test_sc_frame_comb(backend_device, xp):
    # payload=10, period=4 -> data_per_period=3
    # 3 periods: 3*3=9 data. remainder=1.
    # total_length = 3*4 + 1 (data) + 1 (leading pilot) = 14
    # Mask [T, F, F, F, T, F, F, F, T, F, F, F, T, F] -> 4 pilots, 10 data
    frame = SingleCarrierFrame(
        payload_len=10, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
    )
    mask, length = frame._generate_pilot_mask()
    assert length == 14
    assert xp.sum(mask) == 4

    sig = frame.generate_waveform(sps=1, pulse_shape="none")
    assert len(sig.samples) == 14
    assert sig.frame_info.pilot_count == 4


def test_sc_frame_block(backend_device, xp):
    # payload=10, period=4, len=2 -> data_per_block=2
    # num_blocks = ceil(10/2) = 5
    # Mask [T, T, F, F] * 5 -> [T, T, F, F, T, T, F, F, T, T, F, F, T, T, F, F, T, T, F, F]
    # False indices: 2, 3, 6, 7, 10, 11, 14, 15, 18, 19
    # 10th False is at index 19.
    # Truncated length: 19 + 1 = 20.
    frame = SingleCarrierFrame(
        payload_len=10,
        symbol_rate=1e6,
        pilot_pattern="block",
        pilot_period=4,
        pilot_block_len=2,
    )
    mask, length = frame._generate_pilot_mask()
    assert length == 20
    assert xp.sum(mask) == 10

    sig = frame.generate_waveform(sps=1, pulse_shape="none")
    assert len(sig.samples) == 20


def test_sc_frame_guard_zero(backend_device, xp):
    frame = SingleCarrierFrame(
        payload_len=100, symbol_rate=1e6, guard_type="zero", guard_len=20
    )
    sig = frame.generate_waveform(sps=1, pulse_shape="none")
    assert len(sig.samples) == 120
    assert xp.all(sig.samples[-20:] == 0)
    assert sig.frame_info.guard_len == 20
    assert sig.frame_info.guard_type == "zero"


def test_sc_frame_guard_cp(backend_device, xp):
    frame = SingleCarrierFrame(
        payload_len=100, symbol_rate=1e6, guard_type="cp", guard_len=20
    )
    sig = frame.generate_waveform(sps=1, pulse_shape="none")
    assert len(sig.samples) == 120
    # CP should match the last 20 samples of the *original* body
    # Original body length is 100.
    # In generate_waveform for cp:
    # cp_slice = samples[..., -guard_len_samples:]
    # samples = xp.concatenate([cp_slice, samples], axis=-1)
    # So new structure: [CP (20), Body (100)]
    # CP is a copy of Body[-20:]
    # So sig.samples[:20] == sig.samples[-20:]
    assert xp.allclose(sig.samples[:20], sig.samples[-20:])
    assert sig.frame_info.guard_type == "cp"


def test_sc_frame_preamble(backend_device, xp):
    from commstools.core import Preamble

    # Create preamble with bit-first architecture
    # BPSK = 1 bit/symbol, so 50 bits -> 50 symbols
    preamble_bits = xp.array([0, 1] * 25)  # 50 bits for 50 BPSK symbols
    preamble = Preamble(bits=preamble_bits, modulation_scheme="PSK", modulation_order=2)
    frame = SingleCarrierFrame(payload_len=100, symbol_rate=1e6, preamble=preamble)
    sig = frame.generate_waveform(sps=1, pulse_shape="none")
    assert len(sig.samples) == 150  # 50 preamble + 100 payload
    # Verify preamble symbols match mapped bits
    assert xp.allclose(sig.samples[:50], preamble.symbols)
    # Verify FrameInfo
    assert sig.frame_info.preamble_len == 50
    assert sig.frame_info.preamble_mod_scheme == "PSK"


def test_sc_frame_bit_first(backend_device, xp):
    """Verify Frame preserves source bits (bit-first architecture)."""
    frame = SingleCarrierFrame(
        payload_len=100, symbol_rate=1e6, payload_mod_order=4, payload_seed=42
    )
    # Access bits and symbols
    bits = frame.payload_bits
    symbols = frame.payload_symbols

    # Bits should exist and have correct length (100 symbols * 2 bits/symbol)
    assert bits is not None
    assert bits.size == 200

    # Signal should have source_bits
    sig = frame.generate_waveform(sps=1, pulse_shape="none")
    assert sig.source_bits is not None


def test_preamble_to_waveform(backend_device, xp):
    """Verify Preamble.to_waveform() works correctly."""
    from commstools.core import Preamble

    preamble_bits = xp.array([0, 1] * 10)  # 20 bits -> 20 BPSK symbols
    preamble = Preamble(bits=preamble_bits, modulation_scheme="PSK", modulation_order=2)

    sig = preamble.to_waveform(sps=4, symbol_rate=1e6, pulse_shape="rrc")

    assert len(sig.samples) == 20 * 4  # 20 symbols * 4 sps
    assert sig.modulation_scheme == "PREAMBLE-PSK"
    assert sig.source_bits is not None
