from typing import Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class SystemConfig(BaseModel):
    """
    Global system configuration parameters.
    These parameters are fundamental and may be used by TX, RX, and SIM components.
    """

    samplerate_dac_hz: Optional[float] = Field(
        None, gt=0, description="Sample rate of the DAC/AWG (Hz). Relevant for TX."
    )
    samplerate_adc_hz: Optional[float] = Field(
        None,
        gt=0,
        description="Sample rate of the ADC/Digitizer (Hz). Relevant for RX and some SIM scenarios.",
    )
    symbolrate_baud: float = Field(
        ...,  # Making this required as it's fundamental
        gt=0,
        description="Symbol rate of the signal (Baud). Crucial for most components.",
    )
    resolution_dac_bits: Optional[int] = Field(
        None,
        gt=0,
        description="Resolution of the DAC/AWG (bits). Relevant for TX quantization effects.",
    )
    resolution_adc_bits: Optional[int] = Field(
        None,
        gt=0,
        description="Resolution of the ADC/Digitizer (bits). Relevant for RX quantization effects.",
    )
    center_frequency_hz: Optional[float] = Field(
        None,
        description="Optical carrier center frequency (Hz) or RF center frequency. Used for frequency-dependent calculations (e.g., CD compensation).",
    )

    # Example of how this might be used in a top-level configuration YAML:
    # system:
    #   symbolrate_baud: 32e9
    #   samplerate_adc_hz: 92e9
    #   samplerate_dac_hz: 92e9
    #   center_frequency_hz: 193.1e12 # For optical C-band example


# Basic logging configuration (can be overridden by application-level config)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )

logger.debug("Global SystemConfig model defined.")
