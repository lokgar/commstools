"""Global system configuration management for CommsTools.

This module provides a global configuration context that can be accessed
from anywhere in the codebase without explicit passing.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class SystemConfig:
    """Central configuration for communication system parameters.

    This class holds all system-level parameters that can be accessed globally
    by Signal processing functions and sequence generators.

    Attributes:
        # Physical Layer Parameters
        sampling_rate: Sampling rate in Hz
        center_freq: Center frequency in Hz
        modulation_format: Modulation format (e.g., 'QPSK', '16QAM', '64QAM')

        # Symbol-Level Parameters
        symbol_rate: Symbol rate in Hz (symbols/second)
        samples_per_symbol: Number of samples per symbol

        # Channel Parameters
        snr_db: Signal-to-noise ratio in dB
        phase_noise: Phase noise power
        frequency_offset: Carrier frequency offset in Hz

        # DSP Parameters
        filter_roll_off: Pulse shaping filter roll-off factor (0 to 1)
        equalizer_taps: Number of equalizer taps

        # Sequence Generation
        sequence_length: Length of training/preamble sequences
        pilot_spacing: Spacing between pilot symbols

        # Extensibility
        extra: Dictionary for custom user-defined parameters
    """

    # Physical Layer Parameters
    sampling_rate: float
    center_freq: float = 0.0
    modulation_format: str = "QPSK"

    # Symbol-Level Parameters
    symbol_rate: Optional[float] = None
    samples_per_symbol: Optional[int] = None

    # Channel Parameters
    snr_db: Optional[float] = None
    phase_noise: Optional[float] = None
    frequency_offset: Optional[float] = None

    # DSP Parameters
    filter_roll_off: float = 0.35
    equalizer_taps: int = 15

    # Sequence Generation
    sequence_length: int = 128
    pilot_spacing: int = 8

    # Extensibility for custom parameters
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and auto-compute dependent parameters."""
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        if self.filter_roll_off < 0 or self.filter_roll_off > 1:
            raise ValueError("filter_roll_off must be between 0 and 1")

        # Auto-compute dependent parameters
        if self.symbol_rate and self.samples_per_symbol is None:
            self.samples_per_symbol = int(self.sampling_rate / self.symbol_rate)
        elif self.samples_per_symbol and self.symbol_rate is None:
            self.symbol_rate = self.sampling_rate / self.samples_per_symbol

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            SystemConfig instance

        Example:
            >>> config = SystemConfig.from_yaml('sim_config.yaml')
        """
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Remove extra dict if it's in the YAML to avoid duplication
        extra = data.pop("extra", {})
        config = cls(**data)
        config.extra.update(extra)
        return config

    def to_yaml(self, path: str):
        """Save configuration to YAML file.

        Args:
            path: Path where YAML file will be saved

        Example:
            >>> config.to_yaml('sim_config.yaml')
        """
        import yaml

        # Convert dataclass to dict
        data = {
            "sampling_rate": self.sampling_rate,
            "center_freq": self.center_freq,
            "modulation_format": self.modulation_format,
            "symbol_rate": self.symbol_rate,
            "samples_per_symbol": self.samples_per_symbol,
            "snr_db": self.snr_db,
            "phase_noise": self.phase_noise,
            "frequency_offset": self.frequency_offset,
            "filter_roll_off": self.filter_roll_off,
            "equalizer_taps": self.equalizer_taps,
            "sequence_length": self.sequence_length,
            "pilot_spacing": self.pilot_spacing,
            "extra": self.extra,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_signal_params(self) -> dict:
        """Extract parameters suitable for Signal constructor.

        Returns:
            Dictionary with sampling_rate, center_freq, modulation_format

        Example:
            >>> sig = Signal(samples=data, **config.to_signal_params())
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

    Example:
        >>> config = SystemConfig(sampling_rate=1e6, modulation_format='16QAM')
        >>> set_config(config)
        >>> # Now all functions can access this config via get_config()
    """
    global _global_config
    _global_config = config


def get_config() -> Optional[SystemConfig]:
    """Get the current global system configuration.

    Returns:
        Current SystemConfig instance, or None if not set

    Example:
        >>> config = get_config()
        >>> if config:
        >>>     fs = config.sampling_rate
    """
    return _global_config


def clear_config():
    """Clear the global configuration.

    Example:
        >>> clear_config()  # Reset to no config
    """
    global _global_config
    _global_config = None


def require_config() -> SystemConfig:
    """Get the current config, raising an error if not set.

    This is useful for functions that absolutely need a config to operate.

    Returns:
        Current SystemConfig instance

    Raises:
        RuntimeError: If no config is currently set

    Example:
        >>> def my_function():
        >>>     config = require_config()  # Ensures config exists
        >>>     return config.sampling_rate
    """
    config = get_config()
    if config is None:
        raise RuntimeError(
            "No system configuration is set. Please call set_config(config) first."
        )
    return config
