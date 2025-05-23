import logging
from typing import Any, Callable, Dict, Tuple, TypeVar

import jax.numpy as jnp
from pydantic import BaseModel

from .configs import (
    CDCompensationConfig,
    EqualizerConfig,
    FrequencyCorrectionConfig,
    MetricsCalculationConfig,
    PhaseCorrectionConfig,
    ResamplingConfig,
    SymbolDecisionConfig,
    TimingRecoveryConfig,
    get_dsp_configs,
)

logger = logging.getLogger(__name__)

# Type variable for the specific config model
DSPConfig = TypeVar("DSPConfig", bound=BaseModel)

# Type hint for a registered DSP function
DSPFunction = Callable[
    [jnp.ndarray, DSPConfig, Dict[str, Any]], Tuple[jnp.ndarray, Dict[str, Any]]
]

_DSP_FUNCTION_REGISTRY: Dict[str, DSPFunction] = {}


# --- Decorator for registering DSP functions ---


def dsp_function(func: DSPFunction) -> DSPFunction:
    """
    Decorator to register a DSP processing function.

    The function name is automatically extracted from the function definition.
    The function name must match the 'function' Literal in the corresponding config model
    from commstools.receive.configs.

    Args:
        func: The DSP function to register

    Returns:
        The decorated function
    """
    dsp_function_name = func.__name__

    dsp_configs = get_dsp_configs()
    if dsp_function_name not in dsp_configs:
        available_functions = list(dsp_configs.keys())
        raise ValueError(
            f"DSP function '{dsp_function_name}' does not match any registered config model. "
            f"Available functions: {available_functions}. "
            f"Ensure the corresponding config class with 'function: Literal[\"{dsp_function_name}\"]' "
            f"is defined and decorated with @dsp_config in commstools.receive.configs."
        )

    if dsp_function_name in _DSP_FUNCTION_REGISTRY:
        logger.warning(
            f"DSP function '{dsp_function_name}' is already registered. Overwriting."
        )

    _DSP_FUNCTION_REGISTRY[dsp_function_name] = func
    logger.debug(f"Registered DSP function: '{dsp_function_name}' -> {func.__name__}")
    return func


def get_dsp_functions() -> Dict[str, DSPFunction]:
    """Returns a copy of the DSP function registry."""
    if not _DSP_FUNCTION_REGISTRY:
        logger.warning(
            "DSP Function Registry is empty. Ensure commstools.receive.dsp_functions module is imported in commstools/receive/__init__.py."
        )
    return _DSP_FUNCTION_REGISTRY.copy()


# --- DSP Function Implementations ---


@dsp_function
def resampling(
    signal: jnp.ndarray, config: ResamplingConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Applying resampling with factor: {config.resample_factor} or target rate: {config.target_rate_hz} using filter: {config.filter_type}"
    )
    current_sample_rate_hz = signal_state.get("sample_rate_hz")
    if not current_sample_rate_hz:
        logger.error("Current sample_rate_hz not found in signal_state for resampling.")
        return signal, signal_state
    processed_signal = signal
    new_sample_rate_hz = current_sample_rate_hz
    if config.target_rate_hz:
        if current_sample_rate_hz != config.target_rate_hz:
            logger.info(
                f"Resampling from {current_sample_rate_hz} Hz to {config.target_rate_hz} Hz."
            )
            # Placeholder: actual_resample_logic(signal, current_sample_rate_hz, config.target_rate_hz)
            new_sample_rate_hz = config.target_rate_hz
    elif config.resample_factor and config.resample_factor != 1.0:
        new_sample_rate_hz = current_sample_rate_hz * config.resample_factor
        logger.info(
            f"Resampling by factor {config.resample_factor}. New rate: {new_sample_rate_hz} Hz."
        )
        # Placeholder: actual_resample_logic(signal, factor=config.resample_factor)
    signal_state["sample_rate_hz"] = new_sample_rate_hz
    return processed_signal, signal_state


@dsp_function
def cd_compensation(
    signal: jnp.ndarray, config: CDCompensationConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Applying CD compensation: L={config.fiber_length_km}km, D={config.dispersion_ps_nm_km}ps/nm/km, lambda_c={config.center_wavelength_nm}nm."
    )
    # Placeholder for JAX-based CD compensation logic
    # Requires: signal_state.get('sample_rate_hz'), config.center_wavelength_nm (or signal_state.get('center_frequency_hz'))
    return signal, signal_state


@dsp_function
def frequency_correction(
    signal: jnp.ndarray, config: FrequencyCorrectionConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Applying frequency correction using method: {config.method}, exponent: {config.fft_exponent}, PLL BW: {config.pll_bandwidth_norm}"
    )
    # Placeholder for JAX-based frequency correction
    return signal, signal_state


@dsp_function
def timing_recovery(
    signal: jnp.ndarray, config: TimingRecoveryConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Applying timing recovery using method: {config.method}, SPS_in: {config.samples_per_symbol_in}, Loop BW: {config.loop_bandwidth_norm}"
    )
    # Placeholder for JAX-based timing recovery
    # This block would typically change the number of samples and might output 1 or 2 SpS.
    # It might also update 'sample_rate_hz' and add 'samples_per_symbol_out' to signal_state.
    return signal, signal_state


@dsp_function
def equalization(
    signal: jnp.ndarray, config: EqualizerConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Applying equalization: method={config.method}, taps={config.num_taps}, mode={config.mode}, step_size={config.step_size}"
    )
    # Placeholder for JAX-based equalization
    return signal, signal_state


@dsp_function
def phase_correction(
    signal: jnp.ndarray, config: PhaseCorrectionConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(f"Applying phase correction using method: {config.method}")
    # Placeholder for JAX-based phase correction
    return signal, signal_state


@dsp_function
def symbol_decision(
    signal: jnp.ndarray, config: SymbolDecisionConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(f"Applying symbol decision for modulation: {config.modulation_format}")
    # Placeholder for JAX-based symbol decision
    # Output might be decided symbols (e.g., integers) or soft bits.
    return signal, signal_state


@dsp_function
def metrics_calculation(
    signal: jnp.ndarray, config: MetricsCalculationConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(f"Calculating metrics: {config.metrics_to_calculate}")
    logger.info(
        f"Skipping {config.skip_symbols_start} symbols at start, {config.skip_symbols_end} at end"
    )
    # Placeholder for JAX-based metrics calculation
    # This function might not modify the signal but could add metrics to signal_state
    signal_state["calculated_metrics"] = {}
    for metric in config.metrics_to_calculate:
        logger.debug(f"Computing {metric}")
        # Placeholder: actual metric calculations would go here
        signal_state["calculated_metrics"][metric] = 0.0
    return signal, signal_state
