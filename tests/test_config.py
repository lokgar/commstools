"""Tests for the SystemConfig functionality."""

import pytest
import numpy as np
from commstools import (
    Signal,
    SystemConfig,
    set_config,
    get_config,
    clear_config,
    require_config,
)
from commstools.dsp import generate_pilot_sequence, generate_training_signal, add_awgn


from pydantic import ValidationError


class TestSystemConfig:
    """Test SystemConfig creation and validation."""

    def test_basic_config_creation(self):
        """Test creating a basic config."""
        config = SystemConfig(
            sampling_rate=1e6, center_freq=2.4e9, modulation_format="QPSK"
        )

        assert config.sampling_rate == 1e6
        assert config.center_freq == 2.4e9
        assert config.modulation_format == "QPSK"

    def test_auto_compute_samples_per_symbol(self):
        """Test automatic computation of samples_per_symbol."""
        config = SystemConfig(sampling_rate=1e6, symbol_rate=250e3)

        assert config.samples_per_symbol == 4

    def test_auto_compute_symbol_rate(self):
        """Test automatic computation of symbol_rate."""
        config = SystemConfig(sampling_rate=1e6, samples_per_symbol=5)

        assert config.symbol_rate == 200e3

    def test_invalid_sampling_rate(self):
        """Test that invalid sampling rate raises error."""
        with pytest.raises(ValidationError):
            SystemConfig(sampling_rate=-1)

    def test_invalid_roll_off(self):
        """Test that invalid roll-off raises error."""
        with pytest.raises(ValidationError):
            SystemConfig(sampling_rate=1e6, filter_roll_off=1.5)

    def test_to_signal_params(self):
        """Test conversion to Signal parameters."""
        config = SystemConfig(
            sampling_rate=1e6, center_freq=2.4e9, modulation_format="16QAM"
        )

        params = config.to_signal_params()
        assert params["sampling_rate"] == 1e6
        assert params["center_freq"] == 2.4e9
        assert params["modulation_format"] == "16QAM"

    def test_extra_params(self):
        """Test custom extra parameters."""
        config = SystemConfig(
            sampling_rate=1e6, extra={"custom_param": 42, "another_one": "test"}
        )

        assert config.extra["custom_param"] == 42
        assert config.get("custom_param") == 42
        assert config.get("nonexistent", "default") == "default"

    def test_set_and_get_custom_params(self):
        """Test setting custom parameters."""
        config = SystemConfig(sampling_rate=1e6)
        config.set("my_param", 123)

        assert config.get("my_param") == 123


class TestGlobalConfigContext:
    """Test global config context management."""

    def setup_method(self):
        """Clear config before each test."""
        clear_config()

    def teardown_method(self):
        """Clear config after each test."""
        clear_config()

    def test_set_and_get_config(self):
        """Test setting and getting global config."""
        config = SystemConfig(sampling_rate=1e6)
        set_config(config)

        retrieved = get_config()
        assert retrieved is config
        assert retrieved.sampling_rate == 1e6

    def test_get_config_none_when_not_set(self):
        """Test that get_config returns None when not set."""
        assert get_config() is None

    def test_require_config_raises_when_not_set(self):
        """Test that require_config raises error when not set."""
        with pytest.raises(RuntimeError, match="No system configuration is set"):
            require_config()

    def test_require_config_returns_when_set(self):
        """Test that require_config returns config when set."""
        config = SystemConfig(sampling_rate=1e6)
        set_config(config)

        retrieved = require_config()
        assert retrieved is config

    def test_clear_config(self):
        """Test clearing global config."""
        config = SystemConfig(sampling_rate=1e6)
        set_config(config)
        assert get_config() is not None

        clear_config()
        assert get_config() is None


class TestSignalConfigIntegration:
    """Test Signal integration with config."""

    def setup_method(self):
        """Clear config before each test."""
        clear_config()

    def teardown_method(self):
        """Clear config after each test."""
        clear_config()

    def test_signal_from_config(self):
        """Test creating Signal from global config."""
        config = SystemConfig(
            sampling_rate=1e6, center_freq=2.4e9, modulation_format="16QAM"
        )
        set_config(config)

        samples = np.array([1.0, 2.0, 3.0])
        sig = Signal.from_config(samples=samples)

        assert sig.sampling_rate == 1e6
        assert sig.center_freq == 2.4e9
        assert sig.modulation_format == "16QAM"
        assert np.array_equal(sig.samples, samples)

    def test_signal_from_config_requires_config(self):
        """Test that from_config raises error when no config set."""
        samples = np.array([1.0, 2.0, 3.0])

        with pytest.raises(RuntimeError, match="No system configuration is set"):
            Signal.from_config(samples=samples)

    def test_traditional_signal_creation_still_works(self):
        """Test that traditional Signal creation still works without config."""
        samples = np.array([1.0, 2.0, 3.0])
        sig = Signal(samples=samples, sampling_rate=1e6, center_freq=2.4e9)

        assert sig.sampling_rate == 1e6
        assert sig.center_freq == 2.4e9


class TestDSPConfigIntegration:
    """Test DSP functions with config."""

    def setup_method(self):
        """Clear config before each test."""
        clear_config()

    def teardown_method(self):
        """Clear config after each test."""
        clear_config()

    def test_generate_pilot_sequence_with_config(self):
        """Test pilot generation using config."""
        config = SystemConfig(
            sampling_rate=1e6, sequence_length=256, modulation_format="QPSK"
        )
        set_config(config)

        pilots = generate_pilot_sequence()

        assert len(pilots) == 256
        assert isinstance(pilots, np.ndarray)

    def test_generate_pilot_sequence_without_config(self):
        """Test pilot generation works without config (uses defaults)."""
        pilots = generate_pilot_sequence()

        # Should use default length
        assert len(pilots) == 128

    def test_generate_pilot_sequence_override_config(self):
        """Test overriding config values explicitly."""
        config = SystemConfig(
            sampling_rate=1e6, sequence_length=256, modulation_format="QPSK"
        )
        set_config(config)

        # Override with explicit parameters
        pilots = generate_pilot_sequence(length=512, modulation="16QAM")

        assert len(pilots) == 512

    def test_generate_training_signal_with_config(self):
        """Test training signal generation with config."""
        config = SystemConfig(
            sampling_rate=1e6,
            samples_per_symbol=4,
            sequence_length=128,
            modulation_format="QPSK",
        )
        set_config(config)

        training = generate_training_signal()

        assert isinstance(training, Signal)
        assert training.sampling_rate == 1e6
        assert training.modulation_format == "QPSK"
        assert training.samples.shape[0] == 128 * 4  # symbols * samples_per_symbol

    def test_add_awgn_with_config(self):
        """Test AWGN using config SNR."""
        config = SystemConfig(sampling_rate=1e6, snr_db=20)
        set_config(config)

        samples = np.ones(100, dtype=complex)
        sig = Signal.from_config(samples=samples)

        noisy = add_awgn(sig)

        # Signal should have noise added
        assert not np.allclose(noisy.samples, sig.samples)
        # Metadata preserved
        assert noisy.sampling_rate == sig.sampling_rate

    def test_add_awgn_override_config(self):
        """Test overriding config SNR."""
        config = SystemConfig(sampling_rate=1e6, snr_db=20)
        set_config(config)

        samples = np.ones(100, dtype=complex)
        sig = Signal.from_config(samples=samples)

        # Override with explicit SNR
        noisy = add_awgn(sig, snr_db=30)

        assert not np.allclose(noisy.samples, sig.samples)


class TestConfigSaveLoad:
    """Test config save/load functionality."""

    def test_save_and_load_yaml(self, tmp_path):
        """Test saving and loading config from YAML."""
        config = SystemConfig(
            sampling_rate=10e6,
            center_freq=5.8e9,
            modulation_format="64QAM",
            symbol_rate=2.5e6,
            snr_db=28,
            filter_roll_off=0.22,
            sequence_length=512,
        )

        # Save
        config_file = tmp_path / "test_config.yaml"
        config.to_yaml(str(config_file))

        # Load
        loaded = SystemConfig.from_yaml(str(config_file))

        # Verify
        assert loaded.sampling_rate == config.sampling_rate
        assert loaded.center_freq == config.center_freq
        assert loaded.modulation_format == config.modulation_format
        assert loaded.symbol_rate == config.symbol_rate
        assert loaded.snr_db == config.snr_db
        assert loaded.filter_roll_off == config.filter_roll_off
        assert loaded.sequence_length == config.sequence_length
