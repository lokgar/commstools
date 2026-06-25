"""
Core data structures for commstools.

Re-exports the primary containers so existing imports
(``from commstools.core import Signal, Preamble, SingleCarrierFrame``) keep
working after the split of the former monolithic ``core.py`` into a package.

``signal`` is imported first (the thin container, no leaf-module dependencies),
then ``frame`` (which imports ``Signal``).
"""

from .frame import Preamble, SingleCarrierFrame
from .signal import Signal

__all__ = ["Preamble", "Signal", "SingleCarrierFrame"]
