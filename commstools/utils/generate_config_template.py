import yaml
import argparse
import os
import sys

# Add src directory to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    # Import your models dynamically based on arguments or list them explicitly
    from commstools.transmit.config.waveform import WaveformParams
    from commstools.config.rx import DSPChainParams
    # Add other relevant models if needed
except ImportError as e:
    print(f"Error importing framework modules: {e}")
    sys.exit(1)

MODEL_MAP = {
    "tx": WaveformParams,
    "rx": DSPChainParams,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate template YAML config from Pydantic models."
    )
    parser.add_argument(
        "model_type",
        choices=MODEL_MAP.keys(),
        help="Type of config model (generation/processing).",
    )
    parser.add_argument("output_file", help="Path to save the template YAML file.")
    args = parser.parse_args()

    model_class = MODEL_MAP[args.model_type]

    # Create an instance with defaults (Pydantic handles this)
    # For models requiring nested models, you might need to provide minimal valid nested instances
    # Or better: use model_construct() for models with required fields that lack defaults
    # For simplicity, let's assume defaults or Optionals cover most fields for a template
    try:
        # Use construct() if some required fields lack defaults, providing dummy values if needed
        # Example: Needs SystemParams, which has required fields
        if model_class is WaveformParams:
            from commstools.transmit.config.waveform import (
                SystemParams,
                ModulationParams,
                PulseShapeParams,
            )

            dummy_sys = SystemParams(
                sample_rate_dac_hz=1, sample_rate_adc_hz=1, symbol_rate_baud=1
            )
            dummy_mod = ModulationParams()  # Uses defaults
            dummy_ps = PulseShapeParams()  # Uses defaults
            # Use keyword arguments for clarity
            template_instance = model_class(
                system=dummy_sys,
                modulation=dummy_mod,
                pulse_shape=dummy_ps,
                # Other required fields without defaults would need dummy values here
            )
        elif model_class is DSPChainParams:
            # DSPChainParams needs FilePath/DirectoryPath which might fail without actual paths
            # For a template, maybe use strings initially? Or accept they need manual editing.
            template_instance = model_class.construct(  # construct skips validation
                # system=None, # Example if system is optional
                input_file_path="INPUT_FILE_PATH_PLACEHOLDER",
                output_dir="OUTPUT_DIR_PLACEHOLDER",
                # other fields use defaults or are None
            )
        else:
            template_instance = (
                model_class()
            )  # Simpler case if no required sub-models/paths

        # Convert to dict, preserving None values which is good for templates
        template_dict = template_instance.dict()

    except Exception as e:
        print(f"Error creating template instance for {args.model_type}: {e}")
        print(
            "Hint: Models with required fields or nested required models might need manual instantiation logic here."
        )
        sys.exit(1)

    print(f"Generating template config for '{args.model_type}'...")
    try:
        with open(args.output_file, "w") as f:
            yaml.dump(
                template_dict, f, default_flow_style=False, sort_keys=False, indent=2
            )
        print(f"Template saved to: {args.output_file}")
        print("NOTE: Review the template and fill in required fields or placeholders.")
    except Exception as e:
        print(f"Error writing YAML file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
