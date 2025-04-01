from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class DSPBlockParams(BaseModel):
    """Defines the DSP block properties."""

    block: str = Field(
        ..., description="The type of DSP block (e.g., 'filter', 'fft', 'demodulator')"
    )
    enabled: bool = Field(
        True, description="Flag to enable or disable this DSP block in the chain"
    )


class ResamplingParams(DSPBlockParams):
    """Parameters for resampling."""

    block: Literal["resampling"] = "resampling"
    resample_factor: Optional[float] = Field(
        None,
        gt=0,
        description="Resampling factor (new_rate = old_rate * factor). Mutually exclusive with target_rate_hz.",
    )
    target_rate_hz: Optional[float] = Field(
        None,
        gt=0,
        description="Target sample rate (Hz). Mutually exclusive with resample_factor.",
    )
    # Add other params like filter type ('polyphase', 'fft'), filter length, etc.
    filter_type: Literal["polyphase", "fft"] = "polyphase"

    # Add validator to ensure only one of factor/target_rate is set


class FrequencyCorrectionParams(DSPBlockParams):
    """Parameters for carrier frequency offset correction."""

    block: Literal["frequency_correction"] = "frequency_correction"
    method: Literal["fft_based_v_v", "pll"] = "fft_based_v_v"
    fft_exponent: Optional[int] = Field(
        2, description="Exponent for V&V FFT method (e.g., 2 for BPSK, 4 for QPSK)"
    )
    pll_bandwidth_norm: Optional[float] = Field(
        0.01, gt=0, description="Normalized loop bandwidth for PLL"
    )


class TimingRecoveryParams(DSPBlockParams):
    """Parameters for symbol timing recovery."""

    block: Literal["timing_recovery"] = "timing_recovery"
    method: Literal["gardner", "mueller_muller", "nda"] = "gardner"

    samples_per_symbol: Optional[int] = Field(
        None,
        gt=1,
        description="Expected samples per symbol at the input of this block. If None, might be inferred.",
    )
    loop_bandwidth_norm: float = Field(
        0.01, gt=0, description="Normalized loop bandwidth"
    )


class EqualizerParams(DSPBlockParams):
    """Parameters for equalization."""

    block: Literal["equalize"] = "equalization"  # Discriminator value
    method: Literal["cma", "lms", "rls", "decision_directed_lms"] = "cma"
    num_taps: int = Field(11, gt=0, description="Number of equalizer taps")
    step_size: Optional[float] = Field(
        1e-3, gt=0, description="Step size (mu) for LMS/CMA"
    )
    forgetting_factor: Optional[float] = Field(
        0.999, gt=0, lt=1, description="Forgetting factor (lambda) for RLS"
    )
    # Potentially add reference constellation, training sequence info if needed
    reference_constellation: Optional[List[complex]] = Field(
        None, description="Reference constellation points for some algorithms"
    )


class PhaseCorrectionParams(DSPBlockParams):
    """Parameters for carrier phase estimation/correction."""

    block: Literal["phase_correction"] = "phase_correction"  # Discriminator value
    method: Literal["v_v", "blind_phase_search", "pll"] = "v_v"
    # Add specific params, e.g., BPS block size, V&V exponent, PLL bandwidth


class SymbolDecisionParams(DSPBlockParams):
    """Parameters for making symbol decisions."""

    block: Literal["symbol_decision"] = "symbol_decision"  # Discriminator value
    modulation_format: Literal["qpsk", "16qam", "64qam", "bpsk"]  # Required


DSPBlocks = Union[
    ResamplingParams,
    FrequencyCorrectionParams,
    TimingRecoveryParams,
    EqualizerParams,
    PhaseCorrectionParams,
    SymbolDecisionParams,
    # Add other DSP blocks as needed
]
