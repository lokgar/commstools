import logging
from typing import Dict, List, Literal, Optional, Type, Union, Annotated, Any

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo

from commstools.system_config import SystemConfig

logger = logging.getLogger(__name__)

_DSP_CONFIG_REGISTRY: Dict[str, Type[BaseModel]] = {}


# --- Decorators for DSP Function Configuration Models ---


def dsp_config(cls: Type[BaseModel]):
    """
    Decorator to register a DSP function configuration Pydantic model.
    The model must have a 'function: Literal["function_name"] = "function_name"' field.
    """
    function_field: Optional[FieldInfo] = cls.model_fields.get("function")

    if not function_field:
        raise TypeError(
            f"Class {cls.__name__} must have a 'function' field to be registered as a DSP function config."
        )

    dsp_function_name = function_field.default

    if dsp_function_name is None or dsp_function_name is Ellipsis:
        literal_args = getattr(function_field.annotation, "__args__", None)
        if literal_args and len(literal_args) == 1 and isinstance(literal_args[0], str):
            dsp_function_name = literal_args[0]
        else:
            raise ValueError(
                f"Could not automatically determine dsp_function_name for {cls.__name__} from 'function' field. "
                f'Ensure \'function: Literal["unique_name"] = "unique_name"\' is defined with a default value.'
            )

    if not isinstance(dsp_function_name, str):
        raise TypeError(
            f"The 'function' field's resolved type string for {cls.__name__} must be a string. Got: {dsp_function_name}"
        )

    if dsp_function_name in _DSP_CONFIG_REGISTRY:
        logger.warning(
            f"DSP function config type '{dsp_function_name}' from class {cls.__name__} "
            f"is already registered by {_DSP_CONFIG_REGISTRY[dsp_function_name].__name__}. Overwriting."
        )

    _DSP_CONFIG_REGISTRY[dsp_function_name] = cls
    logger.debug(
        f"Registered DSP function config: '{dsp_function_name}' -> {cls.__name__}"
    )
    return cls


def get_dsp_configs() -> Dict[str, Type[BaseModel]]:
    """Returns a copy of the DSP function configuration registry."""
    return _DSP_CONFIG_REGISTRY.copy()


# --- Base DSP Function Configuration Model ---


class BaseDSPConfig(BaseModel):
    """Base model for all DSP function configurations."""

    function: str = Field(
        ..., description="The name of DSP function (e.g., 'resampling', 'equalization')"
    )
    enabled: bool = Field(
        True, description="Flag to enable or disable this DSP function in the chain"
    )


# --- Individual DSP Function Configuration Models ---


@dsp_config
class ResamplingConfig(BaseDSPConfig):
    function: Literal["resampling"] = "resampling"
    resample_factor: Optional[float] = Field(
        None,
        gt=0,
        description="Resampling factor (new_rate = old_rate * factor). Mutually exclusive with target_rate_hz.",
    )
    target_rate_hz: Optional[float] = Field(
        None,
        gt=0,
        description="Target sample rate (Hz). Mutually exclusive with resample_factor.",
    )
    filter_type: Literal["polyphase", "fft"] = Field(
        "polyphase", description="Type of filter to use for resampling."
    )

    @model_validator(mode="after")
    def check_resample_definition(self) -> "ResamplingConfig":
        if self.resample_factor is not None and self.target_rate_hz is not None:
            raise ValueError(
                "Provide either resample_factor or target_rate_hz for resampling, not both."
            )
        if self.resample_factor is None and self.target_rate_hz is None:
            raise ValueError(
                "Provide either resample_factor or target_rate_hz for resampling."
            )
        return self


@dsp_config
class CDCompensationConfig(BaseDSPConfig):
    function: Literal["cd_compensation"] = "cd_compensation"
    fiber_length_km: float = Field(
        ..., gt=0, description="Length of the fiber in kilometers."
    )
    dispersion_ps_nm_km: float = Field(
        ..., description="Dispersion parameter in ps/(nm*km)."
    )
    center_wavelength_nm: float = Field(
        ..., gt=0, description="Center wavelength of the optical signal in nanometers."
    )


@dsp_config
class FrequencyCorrectionConfig(BaseDSPConfig):
    function: Literal["frequency_correction"] = "frequency_correction"
    method: Literal["fft_based_v_v", "pll"] = Field(
        "fft_based_v_v", description="Method for frequency offset correction."
    )
    fft_exponent: Optional[int] = Field(
        4,  # Common default for QPSK/16QAM
        description="Exponent for Viterbi & Viterbi FFT method (e.g., 2 for BPSK, 4 for QPSK/16QAM).",
    )
    pll_bandwidth_norm: Optional[float] = Field(
        0.01,
        gt=0,
        description="Normalized loop bandwidth for PLL based frequency correction.",
    )


@dsp_config
class TimingRecoveryConfig(BaseDSPConfig):
    function: Literal["timing_recovery"] = "timing_recovery"
    method: Literal["gardner", "mueller_muller", "nda_early_late"] = Field(
        "gardner", description="Method for symbol timing recovery."
    )
    samples_per_symbol_in: Optional[int] = Field(
        None,
        gt=1,
        description="Expected samples per symbol at the input of this block. If None, might be inferred or default (e.g., 2).",
    )
    loop_bandwidth_norm: float = Field(
        0.01,
        gt=0,
        description="Normalized loop bandwidth for the timing recovery loop.",
    )


@dsp_config
class EqualizerConfig(BaseDSPConfig):
    function: Literal["equalization"] = "equalization"
    method: Literal["cma", "lms", "rls", "decision_directed_lms"] = Field(
        "lms", description="Equalization algorithm."
    )
    num_taps: int = Field(
        11, gt=0, description="Number of equalizer taps per polarization/channel."
    )
    step_size: Optional[float] = Field(
        1e-3, gt=0, description="Step size (mu) for LMS/CMA adaptive algorithms."
    )
    forgetting_factor: Optional[float] = Field(
        0.999, gt=0, lt=1, description="Forgetting factor (lambda) for RLS algorithm."
    )
    reference_constellation: Optional[List[complex]] = Field(
        None,
        description="Reference constellation points for decision-directed or training modes.",
    )
    num_iterations_train: Optional[int] = Field(
        None,
        gt=0,
        description="Number of symbols to use for initial training phase (if applicable).",
    )
    mode: Optional[Literal["training", "tracking", "blind"]] = Field(
        "blind", description="Operating mode of the equalizer."
    )


@dsp_config
class PhaseCorrectionConfig(BaseDSPConfig):
    function: Literal["phase_correction"] = "phase_correction"
    method: Literal["v_v_cpe", "blind_phase_search", "feedforward_pll"] = Field(
        "v_v_cpe", description="Method for carrier phase estimation/correction."
    )
    # Viterbi & Viterbi Carrier Phase Estimation (V&V CPE) specific
    v_v_avg_length: Optional[int] = Field(
        32, gt=0, description="Averaging length for V&V CPE."
    )
    v_v_exponent: Optional[int] = Field(
        4, description="Exponent for V&V CPE (e.g., 4 for QPSK/16QAM)."
    )
    # Blind Phase Search (BPS) specific
    bps_num_test_angles: Optional[int] = Field(
        32, gt=0, description="Number of test angles for BPS."
    )
    bps_block_length: Optional[int] = Field(
        64, gt=0, description="Block length for BPS averaging."
    )
    # PLL specific
    pll_bandwidth_norm_phase: Optional[float] = Field(
        0.001,
        gt=0,
        description="Normalized loop bandwidth for PLL based phase correction.",
    )


@dsp_config
class SymbolDecisionConfig(BaseDSPConfig):
    function: Literal["symbol_decision"] = "symbol_decision"
    modulation_format: Literal["qpsk", "16qam", "64qam", "bpsk"] = Field(
        ..., description="Modulation format for symbol decision."
    )


@dsp_config
class MetricsCalculationConfig(BaseDSPConfig):
    function: Literal["metrics_calculation"] = "metrics_calculation"
    metrics_to_calculate: List[Literal["evm", "ber", "ser", "snr"]] = Field(
        default_factory=lambda: ["evm"],
        description="List of metrics to calculate (e.g., EVM, BER, SER, SNR).",
    )
    skip_symbols_start: int = Field(
        0,
        ge=0,
        description="Number of symbols to skip at the start for metrics calculation.",
    )
    skip_symbols_end: int = Field(
        0,
        ge=0,
        description="Number of symbols to skip at the end for metrics calculation.",
    )
    reference_sequence_path: Optional[FilePath] = Field(
        None,
        description="Optional path to reference sequence file for BER/SER calculations.",
    )


# --- Main Pipeline Configuration Model ---


# First, define a forward reference for the Union type that will be created after all configs are registered
def _create_dsp_config_union():
    """Create the DSP config union after all configs are registered."""
    if not _DSP_CONFIG_REGISTRY:
        logger.warning(
            "DSP Function Config Registry is empty. "
            "DSPConfigUnion will fallback to Union[BaseDSPConfig]. Ensure DSP function config classes are defined and decorated."
        )
        return Annotated[Union[BaseDSPConfig], Field(discriminator="function")]
    else:
        # Only include the registered specific config classes, not BaseDSPConfig
        return Annotated[
            Union[tuple(_DSP_CONFIG_REGISTRY.values())], Field(discriminator="function")
        ]


# Create the union type
DSPConfigUnion = _create_dsp_config_union()


class ReceivePipelineConfig(BaseModel):
    """
    Defines the configuration for a receive DSP pipeline.
    This orchestrates the entire post-processing workflow.
    """

    job_id: Optional[str] = Field(
        None, description="Optional unique identifier for this processing job."
    )
    system: Optional[SystemConfig] = Field(
        None,
        description="System parameters (e.g., sample rates, symbol rate). Can sometimes be inferred from capture metadata if not provided.",
    )
    input_dtype: Literal[
        "complex64", "complex128", "float32_interleaved", "int16_interleaved", "npy"
    ] = Field("complex64", description="Data type and format of the input file.")
    input_file_path: FilePath = Field(
        ..., description="Path to the raw input signal file (e.g., captured waveform)."
    )
    associated_metadata_path: Optional[FilePath] = Field(
        None,
        description="Optional path to a YAML file containing metadata from generation or capture.",
    )
    output_dir: DirectoryPath = Field(
        ...,
        description="Directory where processed outputs (and intermediates) will be saved.",
    )

    # Now use the proper DSPConfigUnion type
    pipeline: List[DSPConfigUnion] = Field(
        default_factory=list,
        description="Sequence of DSP functions and their parameters. Order matters.",
    )

    save_intermediate: bool = Field(
        False, description="Option to save output after each enabled DSP step."
    )
    seed: Optional[int] = Field(
        None,
        description="Seed for any random processes in DSP (e.g., initial equalizer weights if randomized).",
    )

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: DirectoryPath) -> DirectoryPath:
        """Ensure the output directory exists."""
        if v:
            v.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory ensured: {v}")
        return v


# Example YAML structure for pipeline:
# pipeline:
#   - function: resampling  # This 'function' value is the discriminator
#     enabled: true
#     target_rate_hz: 64e9
#   - function: cd_compensation
#     fiber_length_km: 80
#     dispersion_ps_nm_km: 17.0
#     center_wavelength_nm: 1550.0
#   - function: equalization
#     method: lms
#     num_taps: 31
#     step_size: 5e-4


if __name__ == "__main__":
    print(get_dsp_configs())
