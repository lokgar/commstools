# temp
import warnings

from .signal import Signal
from .backend import set_backend, get_backend
from .plotting import apply_default_theme
from . import baseband

# Filter the specific warning message using a regular expression match
warnings.filterwarnings("ignore", message=".*cupyx.jit.rawkernel is experimental.*")

__all__ = [
    "Signal",
    "set_backend",
    "get_backend",
    "baseband",
]

apply_default_theme()
