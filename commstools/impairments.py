"""
Channel impairments and signal degradation models.

This module provides routines for simulating physical layer impairments,
enabling the evaluation of receiver performance under realistic channel
conditions.

Functions
---------
apply_awgn :
    Adds Additive White Gaussian Noise based on Es/N0.
apply_phase_noise :
    Adds laser/oscillator phase noise modelled as a Wiener (random-walk) process.
apply_pmd :
    Applies first-order Polarization Mode Dispersion to a dual-pol signal.
apply_iq_imbalance :
    Applies IQ imbalance (amplitude and phase mismatch) to a complex signal.
apply_chromatic_dispersion :
    Applies chromatic dispersion (CD) to a signal in the frequency domain.
apply_polarization_mixing :
    Applies a static or time-varying polarization rotation (pure SOP mixing).
"""

import math
from typing import Optional, Union

import numpy as np

from .backend import ArrayType, dispatch
from .logger import logger


def apply_awgn(
    samples: ArrayType,
    sps: float,
    esn0_db: float,
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
        The input signal samples. Shape: (..., N_samples)
    sps : float
        Samples per symbol.
    esn0_db : float
        Symbol energy to noise spectral density ratio ($E_s/N_0$) in dB.
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
        ) + 1j * rng.normal(0, noise_std_component, samples.shape).astype(real_dtype)
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
    K1 = complex(0.5 * (1.0 + g * math.cos(phi)), 0.5 * g * math.sin(phi))
    K2 = complex(0.5 * (1.0 - g * math.cos(phi)), -0.5 * g * math.sin(phi))

    result = K1 * samples + K2 * xp.conj(samples)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return result


def apply_pmd(
    samples: ArrayType,
    sampling_rate: float,
    dgd: float,
    theta: float = 0.0,
) -> ArrayType:
    """
    Applies first-order Polarization Mode Dispersion (PMD) to a dual-pol signal.

    Models an uncompensated channel segment as a frequency-dependent Jones matrix:

    .. math::

        H(f) = R(+\\theta) \\cdot
        \\text{diag}(e^{-j\\pi f \\tau},\\; e^{+j\\pi f \\tau})
        \\cdot R(-\\theta)

    where :math:`\\tau` is the differential group delay (DGD),
    :math:`\\theta` is the PSP orientation angle, and :math:`R(\\theta)` is
    the 2×2 Jones rotation matrix.

    The DGD is applied in the *principal states of polarisation* (PSP) frame:
    :math:`R(-\\theta)` projects the signal onto the PSPs, the differential
    delay :math:`\\pm\\pi f\\tau` is applied to each PSP, then
    :math:`R(+\\theta)` rotates back to the lab frame.  The two PSPs
    experience equal and opposite group delays, giving a total differential
    delay of :math:`\\tau` seconds.

    :math:`\\theta` is **not** a separate bulk rotation — it is the PSP
    orientation angle that is intrinsic to the PMD model.  For a
    frequency-independent polarization rotation with no DGD use
    :func:`apply_polarization_mixing` instead.

    The operation is fully vectorised in the frequency domain and
    backend-agnostic (NumPy / CuPy).

    Parameters
    ----------
    samples : array_like
        Dual-polarization signal. Shape: ``(2, N_samples)``.
    sampling_rate : float
        Sampling rate in Hz.
    dgd : float
        Differential group delay :math:`\\tau` in seconds.
        Set to ``0`` to apply pure SOP rotation with no delay (equivalent
        to :func:`apply_polarization_mixing`).
    theta : float, default 0.0
        PSP orientation angle :math:`\\theta` in radians.  Determines how
        much energy couples between X and Y polarisations.
        :math:`\\theta = 0` → PSPs aligned with lab axes (no cross-coupling);
        :math:`\\theta = \\pi/4` → maximum coupling.

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
    >>> distorted = apply_pmd(samples, sig.sampling_rate, dgd=5e-12, theta=np.pi/5)
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
    # H(f) = R(+θ) · diag(D) · R(-θ)
    # R(-θ): rotate INTO the principal-state-of-polarisation (PSP) frame
    Rfwd = xp.array([[c, s], [-s, c]], dtype=samples.dtype)
    # R(+θ): rotate back to the lab frame
    Rinv = xp.array([[c, -s], [s, c]], dtype=samples.dtype)

    phase = xp.pi * freqs * dgd
    D = xp.stack([xp.exp(-1j * phase), xp.exp(1j * phase)])  # (2, N)

    S_F = xp.fft.fft(samples, axis=-1)  # (2, N)

    # Apply: rotate to PSP frame → DGD delay → rotate back to lab frame
    out_F = Rinv @ (D * (Rfwd @ S_F))

    result = xp.fft.ifft(out_F, axis=-1)

    # Preserve input dtype (ifft may produce complex128 from complex64 input)
    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return result


def apply_phase_noise(
    samples: ArrayType,
    sampling_rate: float,
    linewidth_hz: float,
    seed: Optional[int] = None,
    shared_lo: bool = False,
) -> ArrayType:
    r"""
    Adds laser / oscillator phase noise modelled as a Wiener (random-walk) process.

    Each sample is rotated by an accumulated phase drawn from a discrete Wiener
    process whose per-sample variance is set by the laser linewidth:

    .. math::

        \phi[n] = \sum_{k=0}^{n} \delta_k, \quad
        \delta_k \sim \mathcal{N}\!\left(0,\, 2\pi \Delta\nu / f_s\right)

    .. math::

        r[n] = s[n] \cdot e^{\,j\phi[n]}

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    sampling_rate : float
        Sampling rate in Hz.
    linewidth_hz : float
        Combined transmitter + receiver laser linewidth :math:`\Delta\nu` in Hz.
        Typical values: 100 kHz (narrow-linewidth laser) to 10 MHz (DFB).
    seed : int, optional
        Random seed for reproducible noise.
    shared_lo : bool, default False
        When ``False`` (default), each channel receives independent phase noise
        (separate oscillators / lasers per TX-RX path).
        When ``True``, a single phase noise trajectory is shared across all
        channels (common local oscillator in a coherent system).

    Returns
    -------
    array_like
        Phase-noise-impaired signal, same shape, dtype, and backend as input.

    Examples
    --------
    >>> noisy = apply_phase_noise(sig.samples, linewidth_hz=100e3,
    ...                           sampling_rate=sig.sampling_rate)
    """
    logger.info(
        f"Applying phase noise (linewidth={linewidth_hz:.3g} Hz, "
        f"shared_lo={shared_lo})."
    )

    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]  # (1, N)
    C, N = samples.shape

    variance_per_sample = 2.0 * math.pi * linewidth_hz / sampling_rate
    std_per_sample = math.sqrt(variance_per_sample)

    rng = xp.random.RandomState(seed) if seed is not None else xp.random

    if shared_lo:
        # One trajectory shared across all channels
        increments = rng.normal(0.0, std_per_sample, N).astype(xp.float64)
        phase = xp.cumsum(increments)  # (N,)
        result = samples * xp.exp(1j * phase[None, :])
    else:
        # Independent trajectory per channel
        increments = rng.normal(0.0, std_per_sample, (C, N)).astype(xp.float64)
        phase = xp.cumsum(increments, axis=-1)  # (C, N)
        result = samples * xp.exp(1j * phase)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    if was_1d:
        return result[0]
    return result


def apply_polarization_mixing(
    samples: ArrayType,
    theta: Union[float, ArrayType],
    drift_rate_rad_per_sym: float = 0.0,
) -> ArrayType:
    r"""
    Applies a static or time-varying polarization rotation (pure SOP mixing).

    Models a frequency-independent 2x2 Jones rotation matrix:

    .. math::

        \begin{bmatrix} E_x'[n] \\ E_y'[n] \end{bmatrix}
        =
        R(\theta[n])
        \begin{bmatrix} E_x[n] \\ E_y[n] \end{bmatrix},
        \quad
        R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\
                                     \sin\theta &  \cos\theta \end{bmatrix}

    Unlike :func:`apply_pmd`, there is no differential group delay — this is a
    pure bulk polarization rotation.  Useful for testing polarization-diverse
    receivers and modelling slow SOP drift.

    Parameters
    ----------
    samples : array_like
        Dual-polarization signal. Shape: ``(2, N_samples)``.
    theta : float or array_like of shape ``(N_samples,)``
        Rotation angle(s) in radians.

        * **Scalar** — static rotation: the same :math:`R(\theta)` is applied
          to every sample.
        * **Array of shape** ``(N,)`` — time-varying SOP: one angle per sample,
          applied sample-by-sample via vectorised broadcasting.

        When ``theta`` is a scalar and ``drift_rate_rad_per_sym != 0``, the
        trajectory is extended as a linear ramp:
        ``θ[n] = theta + drift_rate_rad_per_sym * n``.
    drift_rate_rad_per_sym : float, default 0.0
        Linear SOP drift rate in radians per sample.  Only used when ``theta``
        is a scalar.  Ignored when ``theta`` is an array.

    Returns
    -------
    array_like
        Rotated dual-polarization signal, same shape, dtype, and backend as
        input.

    Raises
    ------
    ValueError
        If input is not 2-dimensional with first axis == 2.

    Examples
    --------
    >>> # Static 45° rotation
    >>> rotated = apply_polarization_mixing(samples, theta=np.pi / 4)

    >>> # Slow linear SOP drift: 1 mrad per symbol
    >>> drifted = apply_polarization_mixing(samples, theta=0.0,
    ...                                     drift_rate_rad_per_sym=1e-3)
    """
    logger.info(
        f"Applying polarization mixing (theta={theta if np.ndim(theta) == 0 else 'array'}, "
        f"drift={drift_rate_rad_per_sym:.3g} rad/sym)."
    )

    samples, xp, _ = dispatch(samples)

    if samples.ndim != 2 or samples.shape[0] != 2:
        raise ValueError(
            f"apply_polarization_mixing requires dual-pol input with shape (2, N). "
            f"Got shape {samples.shape}."
        )

    N = samples.shape[1]

    # Build angle trajectory
    if np.ndim(theta) == 0:
        scalar_theta = float(theta)
        if drift_rate_rad_per_sym != 0.0:
            angles = (
                xp.arange(N, dtype=xp.float64) * drift_rate_rad_per_sym + scalar_theta
            )
        else:
            # Static: scalar path — avoid building (N,) array
            c = math.cos(scalar_theta)
            s = math.sin(scalar_theta)
            R = xp.array([[c, -s], [s, c]], dtype=samples.dtype)
            result = R @ samples
            if result.dtype != samples.dtype:
                result = result.astype(samples.dtype)
            return result
    else:
        angles = xp.asarray(theta, dtype=xp.float64)
        if angles.shape != (N,):
            raise ValueError(
                f"theta array must have shape (N,)={(N,)}, got {angles.shape}."
            )

    # Time-varying: vectorised per-sample rotation via broadcasting
    # R(θ[n]) applied to each column of samples
    cos_t = xp.cos(angles)  # (N,)
    sin_t = xp.sin(angles)  # (N,)

    Ex, Ey = samples[0], samples[1]
    Ex_out = cos_t * Ex - sin_t * Ey
    Ey_out = sin_t * Ex + cos_t * Ey

    result = xp.stack([Ex_out, Ey_out])  # (2, N)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return result


def apply_chromatic_dispersion(
    samples: ArrayType,
    sampling_rate: float,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float,
) -> ArrayType:
    r"""
    Applies chromatic dispersion (CD) to a signal in the frequency domain.

    Multiplies the signal spectrum by the CD transfer function:

    .. math::

        H_{\text{CD}}(f) = \exp\!\left[-\tfrac{j}{2}\beta_2 (2\pi f)^2 L\right]

    where

    .. math::

        \beta_2 = -\frac{D \lambda^2}{2\pi c}

    and :math:`D` is the dispersion parameter, :math:`\lambda` is the centre
    wavelength, :math:`c` is the speed of light, and :math:`L` is the fibre
    length.

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    sampling_rate : float
        Sampling rate in Hz.
    dispersion_ps_nm_km : float
        Fibre dispersion parameter :math:`D` in ps / (nm · km).
        Standard SMF-28: 17 ps/(nm·km) at 1550 nm.
    fiber_length_km : float
        Fibre span length in km.
    center_wavelength_nm : float
        Centre wavelength in nm (e.g. 1550 for C-band).

    Returns
    -------
    array_like
        CD-impaired signal, same shape, dtype, and backend as input.

    See Also
    --------
    commstools.filtering.compensate_chromatic_dispersion :
        Remove CD in the receiver (electronic dispersion compensation).

    Examples
    --------
    >>> distorted = apply_chromatic_dispersion(
    ...     sig.samples, dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0, sampling_rate=sig.sampling_rate)
    """
    logger.info(
        f"Applying CD (D={dispersion_ps_nm_km} ps/nm/km, "
        f"L={fiber_length_km} km, λ={center_wavelength_nm} nm)."
    )

    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]
    _, N = samples.shape

    # Convert to SI
    D = dispersion_ps_nm_km * 1e-12 / (1e-9 * 1e3)  # s / m²
    lam = center_wavelength_nm * 1e-9  # m
    c = 2.998e8  # m/s
    L = fiber_length_km * 1e3  # m
    beta2 = -(D * lam**2) / (2.0 * np.pi * c) * L  # s²  (β₂·L product)

    omega = 2.0 * np.pi * xp.fft.fftfreq(N, d=1.0 / sampling_rate)
    H = xp.exp(-1j * (beta2 / 2.0) * omega**2)

    S_F = xp.fft.fft(samples, axis=-1)
    out_F = S_F * H[None, :]
    result = xp.fft.ifft(out_F, axis=-1)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    if was_1d:
        return result[0]
    return result
