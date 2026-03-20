"""
Channel impairments and signal degradation models.

This module provides routines for simulating physical layer impairments,
enabling the evaluation of receiver performance under realistic channel
conditions.

Functions
---------
apply_awgn :
    Adds Additive White Gaussian Noise based on Es/N0.
apply_pmd :
    Applies first-order Polarization Mode Dispersion to a dual-pol signal.
"""

from typing import Optional

from .backend import ArrayType, dispatch
from .logger import logger


def apply_awgn(
    samples: ArrayType,
    esn0_db: float,
    sps: float,
    seed: Optional[int] = None,
) -> ArrayType:
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal based on $E_s/N_0$.

    Uses the standard communications definition where $E_s/N_0$ is the ratio of
    symbol energy to noise spectral density. This accounts for oversampling
    so the specified $E_s/N_0$ matches what you'd measure in the signal bandwidth.

    Parameters
    ----------
    samples : array_like
        The input signal samples. Shape: (..., N_samples).
    esn0_db : float
        Symbol energy to noise spectral density ratio ($E_s/N_0$) in dB.
    sps : float
        Samples per symbol.
    seed : int, optional
        Random seed for reproducible noise generation. When ``None`` (default),
        the global RNG state is used.

    Returns
    -------
    array_like
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
    >>> noisy = apply_awgn(sig.samples, esn0_db=20, sps=sig.sps)
    """
    logger.info(f"Adding AWGN (Es/N0 target: {esn0_db:.2f} dB).")

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

    rng = xp.random.RandomState(seed) if seed is not None else xp.random
    if is_complex:
        noise_std_component = xp.sqrt(noise_power / 2)
        real_dtype = samples.real.dtype
        noise = rng.normal(0, noise_std_component, samples.shape).astype(
            real_dtype
        ) + 1j * rng.normal(0, noise_std_component, samples.shape).astype(
            real_dtype
        )
    else:
        noise_std = xp.sqrt(noise_power)
        noise = rng.normal(0, noise_std, samples.shape).astype(samples.dtype)

    noisy_samples = samples + noise

    return noisy_samples


def apply_pmd(
    samples: ArrayType,
    dgd: float,
    sampling_rate: float,
    theta: float = 0.0,
) -> ArrayType:
    """
    Applies a bulk rotation and first-order Polarization Mode Dispersion (PMD) to a dual-pol signal.

    Models an uncompensated channel segment as a frequency-dependent Jones matrix:

    .. math::

        H(f) = \\text{diag}(e^{-j\\pi f \\tau},\\;
        e^{+j\\pi f \\tau}) \\cdot R(\\theta)

    where :math:`\\tau` is the differential group delay (DGD) and
    :math:`\\theta` is the polarization rotation angle.

    The operation is fully vectorized in the frequency domain (no Python loops)
    and backend-agnostic (NumPy / CuPy).

    Parameters
    ----------
    samples : array_like
        Dual-polarization signal. Shape: ``(2, N_samples)``.
    dgd : float
        Differential group delay in seconds. Use ``0`` to apply
        pure rotation without DGD.
    sampling_rate : float
        Sampling rate in Hz.
    theta : float, default 0.0
        Polarization rotation angle in radians.

    Returns
    -------
    array_like
        PMD-distorted signal, same backend/shape as input.

    Raises
    ------
    ValueError
        If input is not 2-dimensional with first axis == 2.

    Examples
    --------
    >>> samples = sig.samples  # shape (2, N), dual-pol
    >>> distorted = apply_pmd(samples, dgd=5e-12, sig.sampling_rate, theta=np.pi/5)
    """
    logger.info(f"Applying PMD (DGD={dgd:.2e} s, theta={theta:.3f} rad).")

    samples, xp, _ = dispatch(samples)

    if samples.ndim != 2 or samples.shape[0] != 2:
        raise ValueError(
            f"apply_pmd requires dual-pol input with shape (2, N). "
            f"Got shape {samples.shape}."
        )

    N = samples.shape[1]
    freqs = xp.fft.fftfreq(N, d=1.0 / sampling_rate)

    c = float(xp.cos(theta))
    s = float(xp.sin(theta))
    R = xp.array([[c, -s], [s, c]], dtype=samples.dtype)

    phase = xp.pi * freqs * dgd
    D = xp.stack([xp.exp(-1j * phase), xp.exp(1j * phase)])

    S_F = xp.fft.fft(samples, axis=-1)

    # Apply PMD phase shift and a bulk SOP rotation
    out_F = (R @ S_F) * D

    result = xp.fft.ifft(out_F, axis=-1)

    # Preserve input dtype (ifft may produce complex128 from complex64 input)
    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return result
