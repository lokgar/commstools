from .core.signal import Signal
from .core.backend import set_backend, get_backend
from .plotting import apply_default_theme

__all__ = [
    "Signal",
    "set_backend",
    "get_backend",
    "apply_default_theme",
]

apply_default_theme()
