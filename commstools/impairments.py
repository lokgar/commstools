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
apply_iq_imbalance :
    Applies IQ imbalance (amplitude and phase mismatch) to a complex signal.
"""

import math
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


def apply_iq_imbalance(
    samples: ArrayType,
    amplitude_imbalance_db: float,
    phase_imbalance_deg: float,
) -> ArrayType:
    """
    Applies IQ imbalance to a complex baseband signal.

    Models the widely linear mixing that occurs when the I and Q branches of a
    receiver have mismatched gain and/or non-orthogonal phase:

    .. math::

        r[n] = K_1 \\, s[n] + K_2 \\, s^*[n]

    where

    .. math::

        K_1 = \\frac{1 + g \\, e^{j\\phi}}{2}, \\quad
        K_2 = \\frac{1 - g \\, e^{-j\\phi}}{2}

    and :math:`g = 10^{A / 20}` is the I/Q amplitude ratio and :math:`\\phi`
    is the phase error in radians.

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    amplitude_imbalance_db : float
        Amplitude imbalance between I and Q branches in dB.  Positive values
        mean Q has higher gain than I.  Use ``0.0`` for no amplitude mismatch.
    phase_imbalance_deg : float
        Phase error between I and Q branches in degrees.  Use ``0.0`` for no
        phase mismatch.

    Returns
    -------
    array_like
        Imbalanced signal, same shape and dtype as input.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0)
    """
    logger.info(
        f"Applying IQ imbalance "
        f"(amplitude={amplitude_imbalance_db:.2f} dB, phase={phase_imbalance_deg:.2f} deg)."
    )

    samples, xp, _ = dispatch(samples)

    g = 10.0 ** (amplitude_imbalance_db / 20.0)
    phi = math.radians(phase_imbalance_deg)

    # Mixing coefficients: r = K1*s + K2*conj(s)
    K1 = complex(0.5 * (1.0 + g * math.cos(phi)),  0.5 * g * math.sin(phi))
    K2 = complex(0.5 * (1.0 - g * math.cos(phi)), -0.5 * g * math.sin(phi))

    result = K1 * samples + K2 * xp.conj(samples)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return result


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

    c = math.cos(theta)
    s = math.sin(theta)
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
