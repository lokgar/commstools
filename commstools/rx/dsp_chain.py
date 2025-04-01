# Simplified concept for dsp_chain runner
from commstools.config.rx import DSPChainParams
from commstools.config.dsp import ResamplingParams, EqualizerParams


def run_dsp_chain(config: DSPChainParams, initial_signal):
    current_signal = initial_signal
    signal_state = {
        "sample_rate_hz": config.system.samplerate_adc_hz if config.system else None
    }  # Track context

    for i, block_config in enumerate(config.dsp_chain):
        if not block_config.enabled:
            print(
                f"Skipping disabled block {i + 1}: {block_config.type} (Name: {block_config.name})"
            )
            continue

        print(f"Running block {i + 1}: {block_config.type} (Name: {block_config.name})")

        # Use isinstance() or match block_config.type to call the correct function
        if isinstance(block_config, ResamplingParams):
            current_signal, signal_state = apply_resampling(
                current_signal, block_config, signal_state
            )
        elif isinstance(block_config, FrequencyCorrectionParams):
            current_signal, signal_state = apply_freq_correction(
                current_signal, block_config, signal_state
            )
        elif isinstance(block_config, TimingRecoveryParams):
            current_signal, signal_state = apply_timing_recovery(
                current_signal, block_config, signal_state
            )
        # ... add cases for all block types ...
        else:
            raise NotImplementedError(
                f"DSP block type '{block_config.type}' not implemented."
            )

        if config.save_intermediate:
            output_filename = f"{config.output_dir}/intermediate_{i + 1}_{block_config.type}{'_' + block_config.name if block_config.name else ''}.npy"
            save_signal(current_signal, output_filename)  # Your save function

    return current_signal  # Final processed signal
