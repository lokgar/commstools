from typing import Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class SystemConfig(BaseModel):
    """
    Global system configuration parameters.
    """

    samplerate_dac_hz: Optional[float] = Field(
        None, gt=0, description="Sample rate of the DAC/AWG (Hz)."
    )
    samplerate_adc_hz: Optional[float] = Field(
        None, gt=0, description="Sample rate of the ADC/Digitizer (Hz)."
    )
    symbolrate_baud: float = Field(
        ..., gt=0, description="Symbol rate of the signal (Baud)."
    )
    resolution_dac_bits: Optional[int] = Field(
        None, gt=0, description="Resolution of the DAC/AWG (bits)."
    )
    resolution_adc_bits: Optional[int] = Field(
        None, gt=0, description="Resolution of the ADC/Digitizer (bits)."
    )
    center_frequency_hz: Optional[float] = Field(
        None,
        description="Optical carrier center frequency (Hz) or RF center frequency.",
    )

    # Example of how this might be used in a top-level configuration YAML:
    # system:
    #   symbolrate_baud: 32e9
    #   samplerate_adc_hz: 92e9
    #   samplerate_dac_hz: 92e9
    #   center_frequency_hz: 193.1e12 # For optical C-band example


logger.debug("Global SystemConfig model defined.")
