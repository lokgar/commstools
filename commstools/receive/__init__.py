"""
Receive module.

This module provides the functionality for implementing
the receive pipeline for communication setups. It includes
the main pipeline runner, configuration classes for various
DSP functions, and utilities for managing DSP function
registrations and configurations.
"""

# Import DSP function implementations to trigger registration
from . import dsp_functions as _dsp_functions

# Import main pipeline components
from .pipeline_runner import run_receive_pipeline
from .configs import (
    ReceivePipelineConfig,
    BaseDSPConfig,
    DSPConfigUnion,
    get_dsp_configs,
    # Individual config classes
    ResamplingConfig,
    CDCompensationConfig,
    FrequencyCorrectionConfig,
    TimingRecoveryConfig,
    EqualizerConfig,
    PhaseCorrectionConfig,
    SymbolDecisionConfig,
    MetricsCalculationConfig,
)

# Import DSP function utilities
from .dsp_functions import dsp_function, get_dsp_functions

__all__ = [
    # Main pipeline
    "run_receive_pipeline",
    "ReceivePipelineConfig",
    # Configuration utilities
    "get_dsp_configs",
    "get_dsp_functions",
    "dsp_function",
    # Base config classes
    "BaseDSPConfig",
    "DSPConfigUnion",
    # Individual DSP config classes
    "ResamplingConfig",
    "CDCompensationConfig",
    "FrequencyCorrectionConfig",
    "TimingRecoveryConfig",
    "EqualizerConfig",
    "PhaseCorrectionConfig",
    "SymbolDecisionConfig",
    "MetricsCalculationConfig",
]
