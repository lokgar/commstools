#!/usr/bin/env python3
"""
Test script to verify ReceivePipelineConfig works properly with discriminated unions.
"""

import yaml
from pathlib import Path
from pprint import pprint


def test_get_dsp_configs():
    """Test that get_dsp_configs() works without errors."""
    print("=" * 60)
    print("Testing get_dsp_configs()...")

    from commstools.receive.configs import get_dsp_configs

    configs = get_dsp_configs()
    print(f"Found {len(configs)} DSP config types:")
    for name, cls in configs.items():
        print(f"  - {name}: {cls.__name__}")

    return configs


def test_create_config_programmatically():
    """Test creating ReceivePipelineConfig programmatically."""
    print("=" * 60)
    print("Testing programmatic config creation...")

    from commstools.receive.configs import (
        ReceivePipelineConfig,
        ResamplingConfig,
        EqualizerConfig,
        MetricsCalculationConfig,
    )
    import tempfile
    import os

    # Create temporary files and directories for testing
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
        dummy_input_path = tmp_file.name

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Create individual DSP configs
            resampling = ResamplingConfig(
                function="resampling", target_rate_hz=64e9, filter_type="polyphase"
            )

            equalizer = EqualizerConfig(
                function="equalization", method="lms", num_taps=41, step_size=5e-4
            )

            metrics = MetricsCalculationConfig(
                function="metrics_calculation", metrics_to_calculate=["evm", "ber"]
            )

            # Create the pipeline config
            config = ReceivePipelineConfig(
                job_id="test_job",
                input_file_path=dummy_input_path,
                output_dir=tmp_dir,
                pipeline=[resampling, equalizer, metrics],
            )

            print("✓ Successfully created ReceivePipelineConfig programmatically!")
            print(f"Pipeline has {len(config.pipeline)} steps:")
            for i, step in enumerate(config.pipeline):
                print(f"  {i + 1}. {step.function} ({type(step).__name__})")

            return config
        finally:
            # Clean up the temporary file
            os.unlink(dummy_input_path)


def test_yaml_loading():
    """Test loading config from YAML file."""
    print("=" * 60)
    print("Testing YAML config loading...")

    yaml_path = Path("examples/configs/example_receive_pipeline.yaml")

    if not yaml_path.exists():
        print(f"⚠️  YAML file not found: {yaml_path}")
        return None

    from commstools.receive.configs import ReceivePipelineConfig
    import tempfile
    import os

    # Load YAML data
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    print(f"Loaded YAML with {len(yaml_data.get('pipeline', []))} pipeline steps:")
    for step in yaml_data.get("pipeline", []):
        print(f"  - {step.get('function', 'unknown')}")

    # Create temporary files and directories for testing
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
        dummy_input_path = tmp_file.name

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_meta:
        dummy_meta_path = tmp_meta.name

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Update paths to use temporary files/directories
            yaml_data["input_file_path"] = dummy_input_path
            yaml_data["output_dir"] = tmp_dir
            if "associated_metadata_path" in yaml_data:
                yaml_data["associated_metadata_path"] = dummy_meta_path

            # Parse with Pydantic
            config = ReceivePipelineConfig(**yaml_data)
            print("✓ Successfully parsed YAML config!")
            print(f"Job ID: {config.job_id}")
            print(f"Pipeline steps: {len(config.pipeline)}")

            # Verify discriminated union worked correctly
            for i, step in enumerate(config.pipeline):
                print(f"  {i + 1}. {step.function} -> {type(step).__name__}")

            return config

        except Exception as e:
            print(f"❌ Error parsing YAML config: {e}")
            return None
        finally:
            # Clean up temporary files
            os.unlink(dummy_input_path)
            os.unlink(dummy_meta_path)


def test_discriminated_union_edge_cases():
    """Test edge cases for discriminated union."""
    print("=" * 60)
    print("Testing discriminated union edge cases...")

    from commstools.receive.configs import ReceivePipelineConfig
    import tempfile
    import os

    # Create temporary files for testing
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
        dummy_input_path = tmp_file.name

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Test with invalid function name
            test_data = {
                "input_file_path": dummy_input_path,
                "output_dir": tmp_dir,
                "pipeline": [{"function": "invalid_function_name", "enabled": True}],
            }

            try:
                config = ReceivePipelineConfig(**test_data)
                print("❌ Should have failed with invalid function name!")
            except Exception as e:
                print(f"✓ Correctly rejected invalid function name: {type(e).__name__}")

            # Test with missing function field
            test_data2 = {
                "input_file_path": dummy_input_path,
                "output_dir": tmp_dir,
                "pipeline": [{"enabled": True, "target_rate_hz": 64e9}],
            }

            try:
                config = ReceivePipelineConfig(**test_data2)
                print("❌ Should have failed with missing function field!")
            except Exception as e:
                print(
                    f"✓ Correctly rejected missing function field: {type(e).__name__}"
                )
        finally:
            # Clean up
            os.unlink(dummy_input_path)


def main():
    """Run all tests."""
    print("Testing ReceivePipelineConfig with Discriminated Unions")
    print("=" * 60)

    try:
        # Test 1: Basic registry functionality
        configs = test_get_dsp_configs()

        # Test 2: Programmatic creation
        prog_config = test_create_config_programmatically()

        # Test 3: YAML loading
        yaml_config = test_yaml_loading()

        # Test 4: Edge cases
        test_discriminated_union_edge_cases()

        print("=" * 60)
        print("🎉 All tests completed!")

        if prog_config and yaml_config:
            print("✓ Both programmatic and YAML config creation work!")
        elif prog_config:
            print("✓ Programmatic config creation works!")
        elif yaml_config:
            print("✓ YAML config loading works!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
