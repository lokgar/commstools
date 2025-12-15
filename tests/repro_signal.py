import numpy as np
import pytest
from pydantic import ValidationError
from commstools.signal import Signal


def test_signal_initialization():
    samples = np.array([1 + 1j, 2 + 2j])
    sig = Signal(samples=samples, sampling_rate=100.0, symbol_rate=10.0)
    assert sig.sampling_rate == 100.0
    assert sig.symbol_rate == 10.0
    assert sig.spectral_domain == "baseband"
    assert sig.domain is None


def test_signal_validation():
    samples = np.array([1 + 1j, 2 + 2j])

    # Missing sampling_rate
    with pytest.raises(ValidationError, match="Field required"):
        Signal(samples=samples, symbol_rate=10.0)

    # Missing symbol_rate
    with pytest.raises(ValidationError, match="Field required"):
        Signal(samples=samples, sampling_rate=100.0)

    # Invalid spectral_domain
    with pytest.raises(
        ValidationError, match="Input should be 'baseband' or 'passband'"
    ):
        Signal(
            samples=samples,
            sampling_rate=100.0,
            symbol_rate=10.0,
            spectral_domain="invalid",
        )

    # Passband missing domain
    with pytest.raises(
        ValueError, match="domain must be provided for passband signals"
    ):
        Signal(
            samples=samples,
            sampling_rate=100.0,
            symbol_rate=10.0,
            spectral_domain="passband",
        )

    # Passband invalid domain
    with pytest.raises(ValidationError, match="Input should be 'RF' or 'OPT'"):
        Signal(
            samples=samples,
            sampling_rate=100.0,
            symbol_rate=10.0,
            spectral_domain="passband",
            domain="invalid",
        )


def test_signal_properties():
    samples = np.zeros(100) + 1j
    sig = Signal(samples=samples, sampling_rate=100.0, symbol_rate=10.0)
    assert sig.duration == 1.0
    assert sig.sps == 10.0


def test_validation_on_assignment():
    samples = np.array([1 + 1j, 2 + 2j])
    sig = Signal(samples=samples, sampling_rate=100.0, symbol_rate=10.0)

    with pytest.raises(
        ValidationError, match="Input should be 'baseband' or 'passband'"
    ):
        sig.spectral_domain = "invalid"

    # Also test rate verification
    with pytest.raises(ValidationError):
        sig.sampling_rate = -1.0


if __name__ == "__main__":
    # check if pytest is installed, if not just run functions
    try:
        import pytest
        import sys

        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        print("Pytest not found, running manually")
        test_signal_initialization()
        test_signal_validation()
        test_signal_properties()
        test_validation_on_assignment()
        print("All tests passed!")
