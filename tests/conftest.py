import pytest
from commstools.core.backend import set_backend


@pytest.fixture(autouse=True)
def reset_backend():
    """Ensure backend is reset to numpy after each test."""
    set_backend("numpy")
    yield
    set_backend("numpy")
