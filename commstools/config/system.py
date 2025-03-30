from typing import Optional

from pydantic import BaseModel, Field


class SystemParams(BaseModel):
    """Main system parameters."""

    sample_rate_dac_hz: Optional[float] = Field(
        None, gt=0, description="Sample rate of the DAC/AWG (Hz)"
    )
    sample_rate_adc_hz: Optional[float] = Field(
        None, gt=0, description="Sample rate of the ADC/Digitizer (Hz)"
    )
    resolution_dac_bits: Optional[int] = Field(
        None, gt=0, description="Resolution of the DAC/AWG (bits)"
    )
    resolution_adc_bits: Optional[int] = Field(
        None, gt=0, description="Resolution of the ADC/Digitizer (bits)"
    )
    symbol_rate_baud: float = Field(
        None, gt=0, description="Symbol rate of the signal (Baud)"
    )
    center_frequency_hz: Optional[float] = Field(
        None, description="Optical carrier center frequency (Hz)"
    )
