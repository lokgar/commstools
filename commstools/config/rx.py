from typing import Literal, Optional, List, Dict, Any

from pydantic import BaseModel, Field, FilePath, DirectoryPath, field_validator

from .system import SystemParams


class DSPChainParams(BaseModel):
    """Defines the DSP chain"""

    job_id: Optional[str] = Field(
        None, description="Optional unique identifier for this processing job"
    )
    system: Optional[SystemParams] = Field(
        None,
        description="System parameters (can sometimes be inferred from capture metadata)",
    )
    input_dtype: Literal[
        "complex64", "complex128", "float32_interleaved", "int16_interleaved", "npy"
    ] = "complex64"
    input_file_path: FilePath  # Use FilePath for validation
    associated_metadata_path: Optional[FilePath] = Field(
        None, description="Path to generation/capture metadata YAML"
    )
    output_dir: DirectoryPath  # Ensure output directory exists (Pydantic can create it)
    dsp_chain: List[Dict[str, Any]] = []  # The dynamic DSP chain definition
    save_intermediate: bool = Field(
        False, description="Option to save output after each DSP step"
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
