# commstools/tx/config/waveform.py
from typing import Literal, Optional
from pydantic import BaseModel, Field, FilePath

# Import the shared SystemConfig
from commstools.system_config import SystemConfig
import logging

logger = logging.getLogger(__name__)


class SequenceConfig(BaseModel):
    """Defines the source of the data symbols/bits for the transmitter."""

    type: Literal["prbs", "random_bits", "file"] = Field(
        "prbs", description="Type of data sequence source."
    )
    prbs_order: Optional[int] = Field(
        15, gt=0, description="Order of the PRBS generator (if type is 'prbs')."
    )
    bits_filepath: Optional[FilePath] = Field(
        None, description="Path to a file containing bits (if type is 'file')."
    )
    num_symbols: int = Field(
        2**16, gt=0, description="Total number of symbols to generate for the sequence."
    )
    seed: Optional[int] = Field(  # Moved seed here for sequence generation
        None, description="Seed for random data generation (PRBS or random bits)."
    )


class ModulationConfig(BaseModel):
    """Defines the modulation format properties for the transmitter."""

    format: Literal["qpsk", "16qam", "64qam", "bpsk"] = Field(
        "qpsk", description="Modulation format."
    )
    # bits_per_symbol could be a computed property or added if needed for other logic
    # @property
    # def bits_per_symbol(self) -> int:
    #     return {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}[self.format]


class PulseShapeConfig(BaseModel):
    """Defines the pulse shaping filter properties for the transmitter."""

    type: Literal["rrc", "rc", "gaussian", "rect"] = Field(
        "rrc", description="Type of pulse shaping filter."
    )
    rolloff: Optional[float] = Field(
        None, ge=0, le=1, description="Rolloff factor for RRC/RC filters."
    )
    span_symbols: Optional[int] = Field(
        16, gt=0, description="Span of the filter in symbols."
    )
    bt_product: Optional[float] = Field(
        None, gt=0, description="BT product for Gaussian filters."
    )
    # Add validators to check required params per type if necessary


class WaveformConfig(BaseModel):
    """
    Defines the overall waveform generation properties for the transmitter.
    """

    job_id: Optional[str] = Field(
        None, description="Optional unique identifier for this waveform generation job."
    )
    system: SystemConfig  # Embed SystemConfig directly
    sequence: SequenceConfig
    modulation: ModulationConfig
    pulse_shape: PulseShapeConfig

    oversampling_factor: int = Field(
        4,
        gt=0,
        description="Oversampling factor relative to symbol rate for the generated waveform.",
    )
    output_dtype: Literal["complex64", "complex128", "float32_interleaved", "npy"] = (
        Field("npy", description="Output data type and format.")
    )
    # Seed for sequence generation is now part of SequenceConfig.
    # A global seed for other potential random aspects of TX could be here if needed.


# Basic logging configuration
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )

logger.debug("TX WaveformConfig models defined.")
