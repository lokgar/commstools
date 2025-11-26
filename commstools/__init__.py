from .core.signal import Signal
from .core.backend import set_backend, get_backend, jit
from .core.config import (
    SystemConfig,
    set_config,
    get_config,
    clear_config,
    require_config,
)

from .plotting import apply_default_theme

__all__ = [
    "Signal",
    "set_backend",
    "get_backend",
    "jit",
    "SystemConfig",
    "set_config",
    "get_config",
    "clear_config",
    "require_config",
    "apply_default_theme",
]

apply_default_theme()
