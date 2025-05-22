# commstools/rx/registry.py
import logging
import jax.numpy as jnp  # Assuming JAX is used for signals
from typing import Callable, Dict, Any, Tuple, TypeVar

# For Pydantic models used in type hints for config
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type variable for the specific config model
ConfigModelType = TypeVar("ConfigModelType", bound=BaseModel)

# Type hint for a registered DSP function
DSPFunctionType = Callable[
    [jnp.ndarray, ConfigModelType, Dict[str, Any]], Tuple[jnp.ndarray, Dict[str, Any]]
]

# RX DSP Function Registry
# Stores: {"block_type_str": processing_function}
_RX_DSP_FUNCTION_REGISTRY: Dict[str, DSPFunctionType] = {}  # type: ignore


def register_rx_dsp_function(block_type_str: str):
    """
    Decorator to register an RX DSP processing function.
    Args:
        block_type_str: The string identifier for the DSP block,
                        must match the 'block' Literal in the corresponding config model
                        from commstools.rx.config.block_parameters.
    """
    if not isinstance(block_type_str, str):
        raise TypeError(
            f"block_type_str for registering RX DSP function must be a string. Got: {block_type_str}"
        )

    def decorator(func: DSPFunctionType) -> DSPFunctionType:  # type: ignore
        if block_type_str in _RX_DSP_FUNCTION_REGISTRY:
            logger.warning(
                f"RX DSP function for block type '{block_type_str}' from function {func.__name__} "
                f"is already registered by {_RX_DSP_FUNCTION_REGISTRY[block_type_str].__name__}. Overwriting."
            )
        _RX_DSP_FUNCTION_REGISTRY[block_type_str] = func
        logger.debug(
            f"Registered RX DSP function: '{block_type_str}' -> {func.__name__}"
        )
        return func

    return decorator


def get_rx_dsp_function_map() -> Dict[str, DSPFunctionType]:  # type: ignore
    """Returns a copy of the RX DSP function registry."""
    if not _RX_DSP_FUNCTION_REGISTRY:
        logger.warning(
            "RX DSP Function Registry is empty. Ensure RX DSP function modules are imported in commstools/rx/__init__.py."
        )
    return _RX_DSP_FUNCTION_REGISTRY.copy()


# Basic logging configuration
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )

logger.debug("RX DSP function registry mechanism defined.")
