# commstools/rx/dsp_chain.py (Refactored)
import logging
import jax.numpy as jnp
import os
from typing import Dict, Any, Tuple

# Configuration imports
from commstools.rx.config.chain import DSPChainConfig  # Main config for the chain

# Import the function map accessor from the RX registry module
from commstools.rx.registry import get_rx_dsp_function_map

logger = logging.getLogger(__name__)


# Helper functions (can remain here or move to a utils.py within commstools.rx or commstools.utils)
def _validate_signal_integrity(signal: jnp.ndarray, block_identifier: str):
    """
    Basic signal integrity checks (NaNs, Infs).
    """
    if jnp.any(jnp.isnan(signal)):
        logger.warning(f"NaNs detected in signal after block: {block_identifier}")
    if jnp.any(jnp.isinf(signal)):
        logger.warning(f"Infs detected in signal after block: {block_identifier}")


def _save_intermediate_signal(
    signal: jnp.ndarray, filename_base: str, output_dir_str: str
):
    """
    Saves the intermediate signal to a .npy file.
    """
    full_path = os.path.join(output_dir_str, f"{filename_base}.npy")
    try:
        # Ensure the directory for the file exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        jnp.save(full_path, signal)  # Use jnp.save for JAX arrays
        logger.info(f"Saved intermediate signal to {full_path}")
    except Exception as e:
        logger.error(f"Failed to save intermediate signal to {full_path}: {e}")


def run_dsp_chain(
    config: DSPChainConfig, initial_signal: jnp.ndarray
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Runs the configured RX DSP chain on an initial signal.
    Relies on RX DSP functions being registered in commstools.rx.registry.
    """
    current_signal = initial_signal
    signal_state: Dict[str, Any] = {}  # Initialize signal state

    # Populate initial signal_state from SystemConfig if available
    if config.system:
        if config.system.samplerate_adc_hz:
            signal_state["sample_rate_hz"] = config.system.samplerate_adc_hz
        if config.system.symbolrate_baud:
            signal_state["symbol_rate_baud"] = config.system.symbolrate_baud
        if config.system.center_frequency_hz:
            signal_state["center_frequency_hz"] = config.system.center_frequency_hz
        # Add other relevant system parameters to signal_state if needed

    logger.info(f"Starting RX DSP chain for job_id: {config.job_id or 'N/A'}")
    logger.debug(f"Initial signal state: {signal_state}")

    # Get the map of registered RX DSP functions
    rx_dsp_function_map = get_rx_dsp_function_map()
    if not rx_dsp_function_map:
        logger.error(
            "RX DSP function map is empty! No DSP functions seem to be registered. "
            "Check imports in commstools/rx/__init__.py."
        )
        raise RuntimeError(
            "RX DSP function map is empty. Cannot proceed with DSP chain."
        )

    for i, block_config_instance in enumerate(config.dsp_chain):
        # block_config_instance is an instance of one of the Pydantic models in RXDSPBlocksUnion
        block_type_str = block_config_instance.block  # Access the discriminator field
        block_identifier = (
            f"{i + 1}_{block_type_str}"  # Unique identifier for logging/saving
        )

        if not block_config_instance.enabled:
            logger.info(f"Skipping disabled block: {block_identifier}")
            continue

        logger.info(f"Running block: {block_identifier}")
        logger.debug(
            f"Block {block_identifier} configuration: {block_config_instance.model_dump_json(indent=2)}"
        )

        if block_type_str in rx_dsp_function_map:
            dsp_function = rx_dsp_function_map[block_type_str]
            try:
                current_signal, signal_state = dsp_function(
                    current_signal, block_config_instance, signal_state
                )
                logger.info(
                    f"Successfully processed block {block_identifier}. Updated signal state: {signal_state}"
                )
                _validate_signal_integrity(current_signal, block_identifier)
            except Exception as e:
                logger.error(
                    f"Error processing block {block_identifier}: {e}", exc_info=True
                )
                raise  # Re-raise to halt processing on error
        else:
            logger.error(
                f"RX DSP function for block type '{block_type_str}' (Block: {block_identifier}) not found in registry."
            )
            raise NotImplementedError(
                f"RX DSP function for block type '{block_type_str}' (Block: {block_identifier}) is not implemented or registered."
            )

        if config.save_intermediate:
            output_dir_str = str(
                config.output_dir
            )  # Ensure it's a string for os.path.join
            _save_intermediate_signal(
                current_signal, f"intermediate_{block_identifier}", output_dir_str
            )

    logger.info(
        f"RX DSP chain processing completed for job_id: {config.job_id or 'N/A'}"
    )
    return current_signal, signal_state


# Basic logging configuration
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )

logger.debug("RX DSP chain runner (run_dsp_chain) defined.")
