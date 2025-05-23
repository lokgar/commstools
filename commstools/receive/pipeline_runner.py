import logging
import os
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from commstools.receive.configs import ReceivePipelineConfig
from commstools.receive.dsp_functions import get_dsp_functions

logger = logging.getLogger(__name__)


# Helper functions (can remain here or move to a utils.py within commstools.rx or commstools.utils)
def _validate_signal_integrity(signal: jnp.ndarray, function_id: str):
    """
    Basic signal integrity checks (NaNs, Infs).
    """
    if jnp.any(jnp.isnan(signal)):
        logger.warning(f"NaNs detected in signal after function: {function_id}")
    if jnp.any(jnp.isinf(signal)):
        logger.warning(f"Infs detected in signal after function: {function_id}")


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


def run_receive_pipeline(
    config: ReceivePipelineConfig, initial_signal: jnp.ndarray
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Runs the configured receive DSP pipeline on an initial signal.
    Relies on DSP functions being registered in the function registry.
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

    logger.info(f"Starting DSP pipeline for job_id: {config.job_id or 'N/A'}")
    logger.debug(f"Initial signal state: {signal_state}")

    # Get the map of registered DSP functions
    dsp_functions = get_dsp_functions()
    if not dsp_functions:
        logger.error(
            "DSP function registry is empty! No DSP functions are registered. "
            "Check imports in commstools/receive/__init__.py."
        )
        raise RuntimeError(
            "DSP function registry is empty. Cannot proceed with pipeline."
        )

    for i, step_config in enumerate(config.pipeline):
        step_name = step_config.function
        step_id = f"{i + 1}_{step_name}"

        if not step_config.enabled:
            logger.info(f"Skipping disabled function: {step_id}")
            continue

        logger.info(f"Running function: {step_id}")
        logger.debug(
            f"Function {step_id} configuration: {step_config.model_dump_json(indent=2)}"
        )

        if step_name in dsp_functions:
            step_function = dsp_functions[step_name]
            try:
                current_signal, signal_state = step_function(
                    current_signal, step_config, signal_state
                )
                logger.info(
                    f"Successfully processed function {step_id}. Signal shape: {current_signal.shape}"
                )
                logger.debug(f"Updated signal state: {signal_state}")
                _validate_signal_integrity(current_signal, step_id)
            except Exception as e:
                logger.error(
                    f"Error processing function {step_id}: {e}",
                    exc_info=True,
                )
                raise  # Re-raise to halt processing on error
        else:
            logger.error(
                f"DSP function for function type '{step_name}' (Function: {step_id}) not found in registry."
            )
            raise NotImplementedError(
                f"DSP function for function type '{step_name}' (Function: {step_id}) is not implemented or registered."
            )

        if config.save_intermediate:
            output_dir_str = str(config.output_dir)
            _save_intermediate_signal(
                current_signal, f"intermediate_{step_id}", output_dir_str
            )

    logger.info(
        f"DSP pipeline processing completed for job_id: {config.job_id or 'N/A'}"
    )
    return current_signal, signal_state


logger.debug(
    "DSP functions and registry mechanism defined in commstools.receive.dsp_functions."
)
