# commstools/rx/algorithms/resampling.py
import logging
import jax.numpy as jnp
from typing import Dict, Tuple

# Import the specific config model for this block
from commstools.rx.config.block import ResamplingConfig

# Import the decorator from the RX registry module
from commstools.rx.registry import register_rx_dsp_function

logger = logging.getLogger(__name__)


@register_rx_dsp_function("resampling")
def apply_resampling(
    signal: jnp.ndarray, config: ResamplingConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    """
    Applies resampling to the input signal.
    Placeholder for actual JAX-based resampling logic.
    """
    logger.info(
        f"Applying resampling with factor: {config.resample_factor} or target rate: {config.target_rate_hz} using filter: {config.filter_type}"
    )

    current_sample_rate_hz = signal_state.get("sample_rate_hz")
    if not current_sample_rate_hz:
        logger.error(
            "Current sample_rate_hz not found in signal_state, required for resampling."
        )
        # Or raise ValueError("sample_rate_hz missing from signal_state for resampling")
        return signal, signal_state  # Return unchanged if critical info is missing

    # --- Actual resampling logic using JAX would go here ---
    # This is a very simplified placeholder. Real resampling involves filter design and application.
    # For example, using jax.scipy.signal.resample or a custom polyphase filter.

    processed_signal = signal  # Placeholder
    new_sample_rate_hz = current_sample_rate_hz  # Default to current

    if config.target_rate_hz:
        if current_sample_rate_hz == config.target_rate_hz:
            logger.info(
                f"Target sample rate {config.target_rate_hz} Hz is same as current. No resampling needed."
            )
            return signal, signal_state
        # Actual resampling to target_rate_hz
        # factor = config.target_rate_hz / current_sample_rate_hz
        # processed_signal = jax_resample_function(signal, factor) # Replace with actual JAX resampling
        new_sample_rate_hz = config.target_rate_hz
        logger.info(
            f"Resampling from {current_sample_rate_hz} Hz to {new_sample_rate_hz} Hz."
        )

    elif config.resample_factor:
        if config.resample_factor == 1.0:
            logger.info("Resample factor is 1.0. No resampling needed.")
            return signal, signal_state
        # Actual resampling by factor
        # processed_signal = jax_resample_function(signal, config.resample_factor) # Replace
        new_sample_rate_hz = current_sample_rate_hz * config.resample_factor
        logger.info(
            f"Resampling by factor {config.resample_factor}. New rate: {new_sample_rate_hz} Hz."
        )

    else:  # Should not happen due to Pydantic validator in ResamplingConfig
        logger.error("Resampling configured without target_rate_hz or resample_factor.")
        return signal, signal_state

    # --- End of actual logic ---

    signal_state["sample_rate_hz"] = new_sample_rate_hz
    return processed_signal, signal_state
