"""Global system configuration management for CommsTools.

This module provides a global configuration context that can be accessed
from anywhere in the codebase without explicit passing.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator, ConfigDict


class SystemConfig(BaseModel):
    """Central configuration for communication system parameters.

    This class holds all system-level parameters that can be accessed globally
    by Signal processing functions and sequence generators.
    """

    model_config = ConfigDict(extra="allow")

    # Physical Layer Parameters
    sampling_rate: float = Field(..., gt=0, description="Sampling rate in Hz")
    center_freq: float = Field(0.0, description="Center frequency in Hz")
    modulation_format: str = Field("QPSK", description="Modulation format")

    # Symbol-Level Parameters
    symbol_rate: Optional[float] = Field(None, description="Symbol rate in Hz")
    samples_per_symbol: Optional[int] = Field(None, description="Samples per symbol")

    # Channel Parameters
    snr_db: Optional[float] = Field(None, description="Signal-to-noise ratio in dB")
    phase_noise: Optional[float] = Field(None, description="Phase noise power")
    frequency_offset: Optional[float] = Field(
        None, description="Carrier frequency offset in Hz"
    )

    # DSP Parameters
    filter_roll_off: float = Field(
        0.35, ge=0, le=1, description="Pulse shaping filter roll-off factor"
    )
    equalizer_taps: int = Field(15, description="Number of equalizer taps")

    # Sequence Generation
    sequence_length: int = Field(
        128, description="Length of training/preamble sequences"
    )
    pilot_spacing: int = Field(8, description="Spacing between pilot symbols")

    # Extensibility for custom parameters
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Custom user-defined parameters"
    )

    @model_validator(mode="after")
    def compute_dependent_parameters(self) -> "SystemConfig":
        """Auto-compute dependent parameters."""
        if self.symbol_rate and self.samples_per_symbol is None:
            self.samples_per_symbol = int(self.sampling_rate / self.symbol_rate)
        elif self.samples_per_symbol and self.symbol_rate is None:
            self.symbol_rate = self.sampling_rate / self.samples_per_symbol
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            SystemConfig instance
        """
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle 'extra' field explicitly if present in YAML, otherwise Pydantic handles top-level extra fields via Config
        extra = data.pop("extra", {})
        config = cls(**data)
        config.extra.update(extra)
        return config

    def to_yaml(self, path: str):
        """Save configuration to YAML file.

        Args:
            path: Path where YAML file will be saved
        """
        import yaml

        # Convert model to dict, including extra fields
        data = self.model_dump()

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_signal_params(self) -> dict:
        """Extract parameters suitable for Signal constructor.

        Returns:
            Dictionary with sampling_rate, center_freq, modulation_format
        """
        return {
            "sampling_rate": self.sampling_rate,
            "center_freq": self.center_freq,
            "modulation_format": self.modulation_format,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter value, checking extra dict if not in main fields.

        Args:
            key: Parameter name
            default: Default value if key not found

        Returns:
            Parameter value or default
        """
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key, default)

    def set(self, key: str, value: Any):
        """Set a parameter value, using extra dict for custom parameters.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value


# ============================================================================
# Global Configuration Context
# ============================================================================

_global_config: Optional[SystemConfig] = None


def set_config(config: SystemConfig):
    """Set the global system configuration.

    Args:
        config: SystemConfig instance to use globally
    """
    global _global_config
    _global_config = config


def get_config() -> Optional[SystemConfig]:
    """Get the current global system configuration.

    Returns:
        Current SystemConfig instance, or None if not set
    """
    return _global_config


def clear_config():
    """Clear the global configuration."""
    global _global_config
    _global_config = None


def require_config() -> SystemConfig:
    """Get the current config, raising an error if not set.

    Returns:
        Current SystemConfig instance

    Raises:
        RuntimeError: If no config is currently set
    """
    config = get_config()
    if config is None:
        raise RuntimeError(
            "No system configuration is set. Please call set_config(config) first."
        )
    return config
