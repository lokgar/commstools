# temp
import warnings

# Filter the specific warning message using a regular expression match
warnings.filterwarnings("ignore", message=".*cupyx.jit.rawkernel is experimental.*")

from .signal import Signal
from .backend import set_backend, get_backend
from .plotting import apply_default_theme
from . import waveforms

__all__ = [
    "Signal",
    "set_backend",
    "get_backend",
    "waveforms",
]

apply_default_theme()
