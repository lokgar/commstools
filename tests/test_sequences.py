from commstools import utils


def test_random_bits(backend_device, xp):
    bits = utils.random_bits(100, seed=42)
    assert len(bits) == 100
    assert xp.all((bits == 0) | (bits == 1))
    assert isinstance(bits, xp.ndarray)
