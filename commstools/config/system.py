from typing import Optional

from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """Configuration parameters for the system."""

    samplerate_dac_hz: Optional[float] = Field(
        None, gt=0, description="Sample rate of the DAC/AWG (Hz)"
    )
    samplerate_adc_hz: Optional[float] = Field(
        None, gt=0, description="Sample rate of the ADC/Digitizer (Hz)"
    )
    resolution_dac_bits: Optional[int] = Field(
        None, gt=0, description="Resolution of the DAC/AWG (bits)"
    )
    resolution_adc_bits: Optional[int] = Field(
        None, gt=0, description="Resolution of the ADC/Digitizer (bits)"
    )
    symbolrate_baud: float = Field(
        None, gt=0, description="Symbol rate of the signal (Baud)"
    )
    center_frequency_hz: Optional[float] = Field(
        None, description="Optical carrier center frequency (Hz)"
    )
