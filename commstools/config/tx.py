from typing import Literal, Optional

from pydantic import BaseModel, Field, FilePath

from .system import SystemConfig


class SequenceConfig(BaseModel):
    """Defines the source of the data symbols/bits."""

    type: Literal["prbs", "random_bits", "file"] = "prbs"
    prbs_order: Optional[int] = Field(15, gt=0)
    bits_filepath: Optional[FilePath] = None
    # Add validators


class ModulationConfig(BaseModel):
    """Defines the modulation format properties."""

    format: Literal["qpsk", "16qam", "64qam", "bpsk"] = "qpsk"
    # Add bits_per_symbol derived field? Or calculate on use


class PulseShapeConfig(BaseModel):
    """Defines the pulse shaping filter properties."""

    type: Literal["rrc", "rc", "gaussian", "rect"] = "rrc"
    rolloff: Optional[float] = Field(None, ge=0, le=1)
    span_symbols: Optional[int] = Field(16, gt=0)
    bt_product: Optional[float] = Field(None, gt=0)
    # Add validators to check required params per type


class WaveformConfig(BaseModel):
    """Defines the waveform generation properties."""

    job_id: Optional[str] = Field(
        None, description="Optional unique identifier for this generation job"
    )
    system: SystemConfig
    sequence: SequenceConfig
    modulation: ModulationConfig
    pulse_shape: PulseShapeConfig
    num_symbols: int = Field(
        2**16, gt=0, description="Total number of symbols to generate"
    )
    oversampling_factor: int = Field(
        4, gt=0, description="Oversampling factor relative to symbol rate"
    )
    output_dtype: Literal["complex64", "complex128", "float32_interleaved", "npy"] = (
        "npy"
    )
    seed: Optional[int] = Field(
        None, description="Seed for random data generation (PRBS or random bits)"
    )
    # Add validators to check required params per type
