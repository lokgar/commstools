# commstools/rx/dsp.py
import logging
from typing import Any, Callable, Dict, Tuple, TypeVar

import jax.numpy as jnp

# For Pydantic models used in type hints for config
from pydantic import BaseModel

# Import all specific RX DSP Block Config models
from commstools.receive.config.dsp import (
    CDCompensationConfig,
    EqualizerConfig,
    FrequencyCorrectionConfig,
    MetricsCalculationConfig,
    # Add other specific config imports if new blocks are added
    PhaseCorrectionConfig,
    ResamplingConfig,
    SymbolDecisionConfig,
    TimingRecoveryConfig,
)

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
            "RX DSP Function Registry is empty. Ensure commstools.rx.dsp module is imported in commstools/rx/__init__.py."
        )
    return _RX_DSP_FUNCTION_REGISTRY.copy()


# --- RX DSP Function Implementations (Decorated) ---


@register_rx_dsp_function("resampling")
def apply_resampling(
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


@register_rx_dsp_function("cd_compensation")
def apply_cd_compensation(
    signal: jnp.ndarray, config: CDCompensationConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Applying CD compensation: L={config.fiber_length_km}km, D={config.dispersion_ps_nm_km}ps/nm/km, lambda_c={config.center_wavelength_nm}nm."
    )
    # Placeholder for JAX-based CD compensation logic
    # Requires: signal_state.get('sample_rate_hz'), config.center_wavelength_nm (or signal_state.get('center_frequency_hz'))
    return signal, signal_state


@register_rx_dsp_function("frequency_correction")
def apply_frequency_correction(
    signal: jnp.ndarray, config: FrequencyCorrectionConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Applying frequency correction using method: {config.method}, exponent: {config.fft_exponent}, PLL BW: {config.pll_bandwidth_norm}"
    )
    # Placeholder for JAX-based frequency correction
    return signal, signal_state


@register_rx_dsp_function("timing_recovery")
def apply_timing_recovery(
    signal: jnp.ndarray, config: TimingRecoveryConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Applying timing recovery using method: {config.method}, SPS_in: {config.samples_per_symbol_in}, Loop BW: {config.loop_bandwidth_norm}"
    )
    # Placeholder for JAX-based timing recovery
    # This block would typically change the number of samples and might output 1 or 2 SpS.
    # It might also update 'sample_rate_hz' and add 'samples_per_symbol_out' to signal_state.
    return signal, signal_state


@register_rx_dsp_function("equalization")
def apply_equalization(
    signal: jnp.ndarray, config: EqualizerConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Applying equalization: method={config.method}, taps={config.num_taps}, mode={config.mode}, step_size={config.step_size}"
    )
    # Placeholder for JAX-based equalization
    return signal, signal_state


@register_rx_dsp_function("phase_correction")
def apply_phase_correction(
    signal: jnp.ndarray, config: PhaseCorrectionConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(f"Applying phase correction using method: {config.method}")
    # Placeholder for JAX-based phase correction
    return signal, signal_state


@register_rx_dsp_function("symbol_decision")
def apply_symbol_decision(
    signal: jnp.ndarray, config: SymbolDecisionConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(f"Applying symbol decision for modulation: {config.modulation_format}")
    # Placeholder for JAX-based symbol decision
    # Output might be decided symbols (e.g., integers) or soft bits.
    return signal, signal_state


@register_rx_dsp_function("metrics_calculation")
def apply_metrics_calculation(
    signal: jnp.ndarray, config: MetricsCalculationConfig, signal_state: Dict
) -> Tuple[jnp.ndarray, Dict]:
    logger.info(
        f"Calculating metrics: {config.metrics_to_calculate}, skipping {config.skip_symbols_start} symbols."
    )
    # Placeholder for metrics calculation. This block might not modify the signal itself,
    # but rather populate signal_state with calculated metrics (e.g., signal_state['evm'], signal_state['ber']).
    # It would likely use functions from commstools.utils.metrics.
    # Example:
    # if "evm" in config.metrics_to_calculate:
    #   ref_symbols = ... # get reference symbols, possibly from signal_state or a loaded file
    #   evm = calculate_evm(signal_to_measure, ref_symbols) # from commstools.utils.metrics
    #   signal_state['evm'] = evm
    #   logger.info(f"Calculated EVM: {evm}")
    return signal, signal_state


# Basic logging configuration
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )

logger.debug("RX DSP functions and registry mechanism defined in commstools.rx.dsp.")
