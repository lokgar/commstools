# commstools/rx/config/chain.py
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, FilePath, DirectoryPath, field_validator
import logging

# Import the dynamically created Union of RX DSP block configurations
from commstools.rx.config.block import RXDSPBlocksUnion

# Import the shared SystemConfig
from commstools.system_config import SystemConfig

logger = logging.getLogger(__name__)


class DSPChainConfig(BaseModel):
    """
    Defines the configuration for a digital signal processing (DSP) chain for the receiver.
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

    # Use the Discriminated Union for the DSP chain
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


# Basic logging configuration
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )

logger.debug("RX DSPChainConfig model defined.")
