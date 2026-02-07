from commstools import utils


def test_random_bits(backend_device, xp):
    # utils.random_bits automatically uses GPU if available
    bits = utils.random_bits(100, seed=42)
    assert len(bits) == 100
    assert xp.all((bits == 0) | (bits == 1))
    assert isinstance(bits, xp.ndarray)
