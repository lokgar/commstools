"""
Channel impairments and signal degradation models.

This module provides routines for simulating physical layer impairments,
enabling the evaluation of receiver performance under realistic channel
conditions.

Functions
---------
add_awgn :
    Adds Additive White Gaussian Noise based on Es/N0.
"""

from typing import TYPE_CHECKING, Union

from .backend import ArrayType, dispatch
from .logger import logger

if TYPE_CHECKING:
    from .core import Signal


def add_awgn(
    signal: Union[ArrayType, "Signal"],
    esn0_db: float,
    sps: float = 1.0,
) -> Union[ArrayType, "Signal"]:
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal based on $E_s/N_0$.

    Uses the standard communications definition where $E_s/N_0$ is the ratio of
    symbol energy to noise spectral density. This accounts for oversampling
    so the specified $E_s/N_0$ matches what you'd measure in the signal bandwidth.

    Parameters
    ----------
    signal : array_like or Signal
        The input signal samples. Shape: (..., N_samples).
    esn0_db : float
        Symbol energy to noise spectral density ratio ($E_s/N_0$) in dB.
    sps : float, default 1.0
        Samples per symbol. For `Signal` objects, this is extracted
        automatically from the metadata.

    Returns
    -------
    array_like or Signal
        The noisy signal with the same type and backend as the input.

    Notes
    -----
    - For symbol-rate signals (sps=1), $E_s/N_0$ equals the sample-level SNR.
    - For oversampled signals, noise power is scaled by `sps` to maintain
      the correct $E_s/N_0$ in the signal bandwidth.
    - For complex signals, noise power is split equally between I and Q.

    Examples
    --------
    >>> sig = Signal.pam(order=4, num_symbols=1000, sps=4, symbol_rate=1e6)
    >>> noisy = add_awgn(sig, esn0_db=20)  # sps extracted from Signal
    """
    logger.info(f"Adding AWGN (Es/N0 target: {esn0_db:.2f} dB).")

    from .core import Signal

    # Extract samples and sps from Signal if applicable
    if isinstance(signal, Signal):
        samples = signal.samples
        sps = signal.sps  # Use actual sps from Signal
    else:
        samples = signal

    samples, xp, _ = dispatch(samples)

    # === Es/N0 to sample-level SNR conversion ===
    #
    # Es/N0 = Symbol Energy / Noise Spectral Density
    #
    # For oversampled signals:
    #   - Symbol energy Es = sps * (average sample power)  [sum over sps samples]
    #   - Noise power in full bandwidth = N0 * fs = N0 * (sps * symbol_rate)
    #   - Noise power per sample = N0 * symbol_rate
    #
    # Therefore:
    #   Es/N0 = (sps * P_sample) / (N0 * symbol_rate)
    #   Sample_SNR = P_sample / P_noise_per_sample = P_sample / (N0 * symbol_rate)
    #
    # Relationship: Sample_SNR = Es/N0 / sps
    #
    # Or equivalently: P_noise = P_signal * sps / Es_N0_linear

    signal_power = xp.mean(xp.abs(samples) ** 2)
    esn0_linear = 10 ** (esn0_db / 10)

    # Noise power accounting for oversampling
    if esn0_linear <= 1e-20:
        noise_power = signal_power * sps / 1e-20
    else:
        noise_power = signal_power * sps / esn0_linear

    # Handle complex signals (split power between I and Q)
    is_complex = xp.iscomplexobj(samples)

    if is_complex:
        noise_std_component = xp.sqrt(noise_power / 2)
        real_dtype = samples.real.dtype
        noise = xp.random.normal(0, noise_std_component, samples.shape).astype(
            real_dtype
        ) + 1j * xp.random.normal(0, noise_std_component, samples.shape).astype(
            real_dtype
        )
    else:
        noise_std = xp.sqrt(noise_power)
        noise = xp.random.normal(0, noise_std, samples.shape).astype(samples.dtype)

    noisy_samples = samples + noise

    if isinstance(signal, Signal):
        sig = signal.copy()
        sig.samples = noisy_samples
        return sig
    else:
        return noisy_samples
