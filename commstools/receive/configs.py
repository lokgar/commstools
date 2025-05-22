import logging
from typing import Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field, model_validator
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)

# Registry for RX DSP block configuration Pydantic models
# Stores: {"block_type_str": ModelClass}
_RX_DSP_CONFIG_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_rx_dsp_block_config(cls: Type[BaseModel]):
    """
    Decorator to register an RX DSP block configuration Pydantic model.
    The model must have a 'block: Literal["block_type_name"] = "block_type_name"' field.
    Tailored for Pydantic V2.
    """
    if not hasattr(cls, "model_fields"):
        raise TypeError(
            f"Class {cls.__name__} does not appear to be a Pydantic V2 model (missing 'model_fields')."
        )

    block_field_info: Optional[FieldInfo] = cls.model_fields.get("block")

    if not block_field_info:
        raise TypeError(
            f"Class {cls.__name__} must have a 'block' field to be registered as an RX DSP block config."
        )

    block_type_str = block_field_info.default
    if block_type_str is None or block_type_str is Ellipsis:
        literal_args = getattr(block_field_info.annotation, "__args__", None)
        if literal_args and len(literal_args) == 1 and isinstance(literal_args[0], str):
            block_type_str = literal_args[0]
        else:
            raise ValueError(
                f"Could not automatically determine block_type_str for {cls.__name__} from 'block' field. "
                f'Ensure \'block: Literal["unique_name"] = "unique_name"\' is defined with a default value.'
            )

    if not isinstance(block_type_str, str):
        raise TypeError(
            f"The 'block' field's resolved type string for {cls.__name__} must be a string. Got: {block_type_str}"
        )

    if block_type_str in _RX_DSP_CONFIG_REGISTRY:
        logger.warning(
            f"RX DSP block config type '{block_type_str}' from class {cls.__name__} "
            f"is already registered by {_RX_DSP_CONFIG_REGISTRY[block_type_str].__name__}. Overwriting."
        )

    _RX_DSP_CONFIG_REGISTRY[block_type_str] = cls
    logger.debug(
        f"Registered RX DSP block config: '{block_type_str}' -> {cls.__name__}"
    )
    return cls


class RXDSPBlockConfigBase(BaseModel):  # Renamed from DSPBlockConfig to be RX specific
    """Base model for all RX DSP block configurations."""

    block: str = Field(
        ..., description="The type of DSP block (e.g., 'resampling', 'equalization')"
    )
    enabled: bool = Field(
        True, description="Flag to enable or disable this DSP block in the chain"
    )


# --- Definitions of individual RX DSP Block Configurations ---
@register_rx_dsp_block_config
class ResamplingConfig(RXDSPBlockConfigBase):
    block: Literal["resampling"] = "resampling"
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


@register_rx_dsp_block_config
class CDCompensationConfig(RXDSPBlockConfigBase):
    block: Literal["cd_compensation"] = "cd_compensation"
    fiber_length_km: float = Field(
        ..., gt=0, description="Length of the fiber in kilometers."
    )
    dispersion_ps_nm_km: float = Field(
        ..., description="Dispersion parameter in ps/(nm*km)."
    )
    center_wavelength_nm: float = Field(
        ..., gt=0, description="Center wavelength of the optical signal in nanometers."
    )


@register_rx_dsp_block_config
class FrequencyCorrectionConfig(RXDSPBlockConfigBase):
    block: Literal["frequency_correction"] = "frequency_correction"
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


@register_rx_dsp_block_config
class TimingRecoveryConfig(RXDSPBlockConfigBase):
    block: Literal["timing_recovery"] = "timing_recovery"
    method: Literal["gardner", "mueller_muller", "nda_early_late"] = Field(
        "gardner", description="Method for symbol timing recovery."
    )
    samples_per_symbol_in: Optional[int] = Field(  # Clarified name
        None,
        gt=1,
        description="Expected samples per symbol at the input of this block. If None, might be inferred or default (e.g., 2).",
    )
    loop_bandwidth_norm: float = Field(
        0.01,
        gt=0,
        description="Normalized loop bandwidth for the timing recovery loop.",
    )
    # Add other method-specific parameters as needed, e.g., for NDA Early/Late


@register_rx_dsp_block_config
class EqualizerConfig(RXDSPBlockConfigBase):
    block: Literal["equalization"] = "equalization"
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
    num_iterations_train: Optional[int] = Field(  # Added for training phase
        None,
        gt=0,
        description="Number of symbols to use for initial training phase (if applicable).",
    )
    mode: Optional[Literal["training", "tracking", "blind"]] = (
        Field(  # Added for equalizer mode
            "blind", description="Operating mode of the equalizer."
        )
    )


@register_rx_dsp_block_config
class PhaseCorrectionConfig(RXDSPBlockConfigBase):
    block: Literal["phase_correction"] = "phase_correction"
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


@register_rx_dsp_block_config
class SymbolDecisionConfig(RXDSPBlockConfigBase):
    block: Literal["symbol_decision"] = "symbol_decision"
    modulation_format: Literal["qpsk", "16qam", "64qam", "bpsk"] = Field(
        ..., description="Modulation format for symbol decision."
    )


@register_rx_dsp_block_config
class MetricsCalculationConfig(
    RXDSPBlockConfigBase
):  # Added based on example_dsp_chain.yaml
    block: Literal["metrics_calculation"] = "metrics_calculation"
    metrics_to_calculate: List[Literal["evm", "ber", "ser", "snr_est"]] = Field(
        ..., description="List of metrics to calculate."
    )
    reference_sequence_path: Optional[str] = Field(
        None,
        description="Path to the reference sequence (bits or symbols) for BER/SER calculation.",
    )
    skip_symbols_start: Optional[int] = Field(
        0,
        ge=0,
        description="Number of initial symbols to skip before calculating metrics (e.g., to exclude training/convergence).",
    )


# Dynamically create the Union of all registered RX DSP block configurations
if not _RX_DSP_CONFIG_REGISTRY:
    logger.warning(
        "RX DSP Block Config Registry is empty. "
        "RXDSPBlocksUnion will fallback to Union[RXDSPBlockConfigBase]. Ensure RX DSP block config classes are defined and decorated."
    )
    RXDSPBlocksUnion = Union[RXDSPBlockConfigBase]
else:
    RXDSPBlocksUnion = Union[tuple(_RX_DSP_CONFIG_REGISTRY.values())]  # type: ignore


def get_rx_dsp_config_registry() -> Dict[str, Type[BaseModel]]:
    """Returns a copy of the RX DSP block configuration registry."""
    return _RX_DSP_CONFIG_REGISTRY.copy()


from typing import Optional, List, Literal
from pydantic import BaseModel, Field, FilePath, DirectoryPath, field_validator
import logging

# Import the dynamically created Union of RX DSP block configurations
from .dsp_config import RXDSPBlocksUnion

# Import the shared SystemConfig
from commstools.system_config import SystemConfig

logger = logging.getLogger(__name__)


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

    # The 'discriminator' field tells Pydantic to use the 'block' field in each item
    # of the list to determine which specific Pydantic model (from RXDSPBlocksUnion) to use.
    dsp_chain: List[RXDSPBlocksUnion] = Field(  # type: ignore
        default_factory=list,
        discriminator="block",  # Crucial for Pydantic to correctly parse the Union types
        description="Sequence of RX DSP blocks and their parameters. Order matters.",
    )

    save_intermediate: bool = Field(
        False, description="Option to save output after each enabled DSP step."
    )
    seed: Optional[int] = Field(
        None,
        description="Seed for any random processes in DSP (e.g., initial equalizer weights if randomized).",
    )

    @field_validator("output_dir")
    @classmethod  # Pydantic V2 validators are class methods by default if first arg is cls
    def create_output_dir(cls, v: DirectoryPath) -> DirectoryPath:
        """Ensure the output directory exists."""
        if v:
            v.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory ensured: {v}")
        return v

    # Example YAML structure for dsp_chain:
    # dsp_chain:
    #   - block: resampling  # This 'block' value is the discriminator
    #     enabled: true
    #     target_rate_hz: 64e9
    #   - block: cd_compensation
    #     fiber_length_km: 80
    #     dispersion_ps_nm_km: 17.0
    #     center_wavelength_nm: 1550.0
    #   - block: equalization
    #     method: lms
    #     num_taps: 31
    #     step_size: 5e-4
