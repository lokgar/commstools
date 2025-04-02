from typing import Literal, Optional, List, Dict, Any

from pydantic import BaseModel, Field, FilePath, DirectoryPath, field_validator

from .system import SystemConfig

from .dsp import DSPBlocks


class DSPChainConfig(BaseModel):
    """
    Defines the configuration for a digital signal processing (DSP) chain.

    This configuration model orchestrates the entire post-processing workflow,
    from reading input data to applying a sequence of DSP algorithms and saving
    the results.

    Attributes:
        job_id: Optional unique identifier for this processing job.
        system: Optional system parameters (sample rates, etc.). May be inferred
                from metadata if available.
        input_dtype: Specifies the data type and format of the input file.
        input_file_path: Path to the raw input signal file (e.g., captured waveform).
        associated_metadata_path: Optional path to a YAML file containing metadata
                                  from generation or capture (useful for context).
        output_dir: Directory where processed outputs (and intermediates) will be saved.
        dsp_chain: Defines the sequence of DSP operations to be applied.
                   This is the core of the processing configuration. See details below.
        save_intermediate: If True, the output signal after each *enabled* DSP block
                           in the chain will be saved to the output directory.
        seed: Optional seed for initializing any stochastic processes within the
              DSP blocks (e.g., initial equalizer tap weights if randomized).

    **DSP Chain (`dsp_chain`) Configuration Rationale:**

    Unlike configurations for processes like waveform generation (which might be
    defined by a single set of holistic parameters), a DSP workflow is inherently
    a **sequence of distinct steps**. The order of operations matters significantly,
    and users often need to experiment by adding, removing, reordering, or
    temporarily disabling specific blocks (e.g., comparing results with and
    without equalization).

    To model this flexibility directly and explicitly, the `dsp_chain` attribute
    is defined as a `List[Union[...]]` (specifically `List[DSPBlocks]`).
    Each item in this list represents a single DSP block configuration.

    *   **Structure:** It's a list (`[]` in YAML) where each element (`-`) is a
        dictionary defining one DSP block.
    *   **Order:** The order of elements in the list dictates the exact sequence
        in which the DSP operations will be applied to the signal.
    *   **Block Types:** Each block dictionary *must* include a `type` field (e.g.,
        `type: resample`, `type: equalize`). This `type` acts as a discriminator,
        telling the toolkit which specific block model (e.g., `ResamplingParams`,
        `EqualizerParams`) to use for validation and processing. Only types
        defined in `commstools.config.dsp_blocks.DSPBlocks` are valid.
    *   **Flexibility (Skipping Blocks):**
        1.  **Omission:** To completely skip a block for a run, simply do not
            include its dictionary in the `dsp_chain` list in your YAML file.
        2.  **Disabling:** To temporarily disable a block while keeping its
            configuration in the file (e.g., for easy comparison runs), include
            its dictionary but set its `enabled` field to `false`. The runner
            will explicitly skip blocks where `enabled` is false.
    *   **Validation:** Pydantic validates each block in the list against its
        specific model definition based on the `type` field, ensuring all necessary
        parameters for that block are present and correctly typed.

    **Example YAML Snippet:**

    ```yaml
    # ... other DSPChainParams fields ...
    dsp_chain:
      - type: resample            # First step
        target_rate_hz: 2.0e9
        filter_type: polyphase
      - type: frequency_correct   # Second step
        method: fft_based_v_v
        fft_exponent: 4
      - type: timing_recovery     # Third step - Temporarily disabled
        enabled: false
        method: gardner
        samples_per_symbol: 2
        loop_bandwidth_norm: 0.01
      - type: equalize            # Fourth step (equalization block is present)
        method: cma
        num_taps: 15
        step_size: 0.001
      # Phase correction step is omitted entirely for this run
      - type: symbol_decision     # Final step
        modulation_format: qpsk
    # ...
    ```

    This list-based approach provides maximum flexibility for defining and
    experimenting with different DSP workflows in a clear, structured, and
    validated manner, directly reflecting the sequential nature of the process.
    The processing engine iterates through this list, applying only the enabled
    blocks in the specified order.
    """

    job_id: Optional[str] = Field(
        None, description="Optional unique identifier for this processing job"
    )
    system: Optional[SystemConfig] = Field(
        None,
        description="System parameters (can sometimes be inferred from capture metadata)",
    )
    input_dtype: Literal[
        "complex64", "complex128", "float32_interleaved", "int16_interleaved", "npy"
    ] = "complex64"
    input_file_path: FilePath
    associated_metadata_path: Optional[FilePath] = Field(
        None, description="Path to generation/capture metadata YAML"
    )
    output_dir: DirectoryPath
    # Use the Discriminated Union for the DSP chain
    dsp_chain: List[DSPBlocks] = Field(
        default_factory=list,
        description="Sequence of DSP blocks and their parameters. Order matters.",
    )
    save_intermediate: bool = Field(
        False, description="Option to save output after each enabled DSP step"
    )
    seed: Optional[int] = Field(
        None, description="Seed for any random processes in DSP (e.g., initial weights)"
    )

    @field_validator("output_dir")
    def create_output_dir(cls, v):
        """Ensure the output directory exists."""
        if v:
            v.mkdir(parents=True, exist_ok=True)
        return v
