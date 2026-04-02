"""
Core signal processing abstractions and data containers.

This module defines the primary data structures used throughout the library.
It provides high-level abstractions for handling raw IQ samples, physical
layer metadata, and complex frame structures.

All core classes are built on Pydantic for robust validation and support
transparent backend switching between CPU (NumPy) and GPU (CuPy).

Classes
-------
Signal :
    The primary container for IQ samples and signal-centric metadata.
    Includes methods for filtering, resampling, and visualization.
Preamble :
    A structured container for frame synchronization sequences.
SingleCarrierFrame :
    A complex frame container supporting pilot patterns, guard intervals,
    and spatial multiplexing (MIMO).
"""

import types
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


from . import helpers
from .backend import (
    ArrayType,
    from_jax,
    get_array_module,
    get_scipy_module,
    is_cupy_available,
    to_device,
    to_jax,
)
from .logger import logger


class Signal(BaseModel):
    """
    Primary container for digital baseband or RF signals.

    The `Signal` class encapsulates complex-valued IQ samples along with the
    physical layer metadata (sampling rate, modulation, etc.) required for
    comprehensive Digital Signal Processing (DSP) pipelines. It supports
    seamless switching between CPU (NumPy) and GPU (CuPy) backends.

    Attributes
    ----------
    samples : array_like
        The complex IQ samples.
        Shape: (N_samples,) for SISO or (N_channels, N_samples) for MIMO.
        The last dimension is always assumed to be Time.
    sampling_rate : float
        Sampling frequency in Hertz (Hz). Must be > 0.
    symbol_rate : float
        Symbol frequency (Baud rate) in Hertz (Hz). Must be > 0.
    mod_scheme : str, optional
        Identifier for the modulation format (e.g., 'QPSK', '16QAM').
        For frame-generated signals this is ``None``; modulation is carried by
        ``frame.payload_mod_scheme`` instead.
    mod_order : int, optional
        The modulation order. Similar to `mod_scheme`, it might be `None`
        if multiple modes are present within a frame. See
        ``frame.payload_mod_order``.
    mod_unipolar : bool, optional
        If True, uses a unipolar constellation (e.g., 0 to M-1).
    mod_rz : bool, optional
        If True, uses Return-to-Zero (RZ) signaling.
    source_bits : array_like, optional
        The original binary data that generated the signal (full wire order).
        Populated by :meth:`Signal.generate` and the factory methods.
        For frame-generated signals this is ``None``; extract the payload
        segment via ``frame.get_structure_map()`` and construct a plain
        ``Signal`` with the relevant ``source_bits`` for per-segment metrics.
    source_symbols : array_like, optional
        The mapped constellation symbols before pulse shaping (full wire
        order). Same scoping note as ``source_bits``.
    ps_pmf : array_like of float, optional
        Maxwell-Boltzmann PMF of shape ``(M,)`` for PS-QAM signals.
        Set automatically by :meth:`Signal.psqam`.  When present, the
        normalization of ``source_symbols`` is skipped (PS symbols have
        intentionally lower average energy than uniform QAM), and
        :meth:`mi`, :meth:`gmi`, and :meth:`plot_constellation` use the
        non-uniform prior automatically.  ``None`` for all other modulations.
    pulse_shape : str, optional
        Name of the pulse shaping filter (e.g., ``'rrc'``, ``'rect'``,
        ``'gaussian'``).
    filter_span : int
        Span of the pulse-shaping filter in symbols.
    rrc_rolloff : float
        Roll-off factor for the Root-Raised Cosine (RRC) filter.
    rc_rolloff : float
        Roll-off factor for the Raised Cosine (RC) filter.
    gaussian_bt : float
        Bandwidth-Time (BT) product for Gaussian pulse shaping.
    smoothrect_bt : float
        BT product for SmoothRect shaping filters.
    spectral_domain : {"BASEBAND", "PASSBAND", "INTERMEDIATE"}
        The signal's current placement in the frequency spectrum.
    physical_domain : {"DIG", "RF", "OPT"}
        The physical transmission domain: ``'DIG'`` (Digital), ``'RF'``
        (Radio), ``'OPT'`` (Optical).
    center_frequency : float
        The carrier or center frequency in Hz.
    digital_frequency_offset : float
        Cumulative digital frequency shift applied to the signal in Hz.
    signal_type : {"Single-Carrier Frame", "OFDM Frame", "Preamble"}, optional
        Human-readable label for the signal structure. Informational only.
    frame : Frame, optional
        The frame that generated the signal.
    resolved_symbols : array_like, optional
        Symbols at 1 SPS, normalised to unit average power.
        Populated by :meth:`resolve_symbols`.  Call only on plain signals
        (non-frame); frame signals contain mixed preamble/pilot/payload that
        may have different modulations or gains — resolve after splitting.
    resolved_bits : array_like, optional
        Hard-decision bits demapped from ``resolved_symbols``.
        Populated by :meth:`demap_symbols_hard`.

    Notes
    -----
    **Frame-generated signals**: :meth:`SingleCarrierFrame.to_signal` sets
    ``self.frame`` but leaves ``source_symbols`` and
    ``source_bits`` as ``None``.  The receive workflow is:

    1. Run timing / FOE / CPR / equalization on the frame signal.
    2. Use ``frame.get_structure_map()`` to slice sample/symbol indices.
    3. Extract each segment and build a plain ``Signal`` with the appropriate
       ``source_symbols``/``source_bits`` before calling ``resolve_symbols()``,
       ``evm()``, ``ber()``, etc.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    samples: Any
    sampling_rate: float = Field(..., gt=0)
    symbol_rate: float = Field(..., gt=0)

    mod_scheme: Optional[str] = None
    mod_order: Optional[int] = None
    mod_unipolar: Optional[bool] = None
    mod_rz: Optional[bool] = None

    source_bits: Optional[Any] = None
    source_symbols: Optional[Any] = None
    ps_pmf: Optional[Any] = None  # (M,) PMF over constellation; set only for PS-QAM

    pulse_shape: Optional[str] = None
    filter_span: int = Field(default=10, ge=1)
    rrc_rolloff: float = Field(default=0.35, ge=0, le=1)
    rc_rolloff: float = Field(default=0.35, ge=0, le=1)
    gaussian_bt: float = Field(default=0.3, gt=0)
    smoothrect_bt: float = Field(default=1.0, gt=0)

    spectral_domain: Literal["BASEBAND", "PASSBAND", "INTERMEDIATE"] = "BASEBAND"
    physical_domain: Literal["DIG", "RF", "OPT"] = "DIG"

    center_frequency: float = Field(default=0, ge=0)
    digital_frequency_offset: float = Field(default=0)

    # Human-readable label for the signal structure
    signal_type: Optional[Literal["Single-Carrier Frame", "OFDM Frame", "Preamble"]] = (
        None
    )

    # Back-reference to the SingleCarrierFrame that generated this signal (set by
    # SingleCarrierFrame.to_signal()). Enables frame-aware convenience methods
    # (correct_timing, frame-aware equalizers) without requiring the caller to re-supply
    # the frame object.  Excluded from serialisation (numpy arrays inside frame
    # duplicate samples data and are not JSON-serialisable).
    frame: Optional[Any] = Field(default=None, exclude=True, repr=False)

    # Resolved data from processing (1 SPS, normalized — populated by resolve_symbols())
    resolved_symbols: Optional[Any] = Field(default=None, repr=False)
    resolved_bits: Optional[Any] = Field(default=None, repr=False)

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    @field_validator("samples", mode="before")
    @classmethod
    def validate_samples(cls, v: Any) -> Any:
        """
        Validates and coerces the samples input into a backend-compatible array.

        This validator ensures that the input is converted to a NumPy or CuPy array
        and enforces a (Channels, Time) shape convention for multidimensional inputs.

        Parameters
        ----------
        v : array_like
            Input samples (list, tuple, NumPy array, CuPy array, or JAX array).

        Returns
        -------
        array_like
            The validated samples as a NumPy or CuPy array.

        Raises
        ------
        ValueError
            If the input cannot be converted to a supported array type or has
            unsupported dimensions (> 2).

        Notes
        -----
        The library enforces a **Time-Last** convention: (N_channels, N_samples)
        or simply (N_samples,) for 1D signals. This aligns with C-contiguous
        memory layout which is generally more performant for time-axis operations.
        """
        arr = helpers.validate_array(v, name="samples")

        # Check shape conventions
        # We enforce Time-Last convention: (Channels, Time) or (Time,) for 1D.
        # This aligns better with C-contiguous operations on the time axis (last axis)
        # which is critical for CuPy performance/stability.

        if arr.ndim > 2:
            raise ValueError(
                f"Samples array has {arr.ndim} dimensions. "
                "Only 1D (SISO) or 2D (MIMO/Dual-Pol) arrays are supported."
            )

        if arr.ndim == 2:
            # Check dimensions to guess orientation
            s0, s1 = arr.shape
            # If dim0 (rows) > dim1 (cols) and dim0 >> 10, it's likely (Time, Channels)
            # We want (Channels, Time).
            if s0 > s1 and s0 > 32:  # Heuristic: Time dim usually > 32
                logger.warning(
                    f"Samples shape is {arr.shape}. Converting to Time-Last convention (N_channels={s1}, N_samples={s0}). "
                    "Please provide input as (N_channels, N_samples) for MIMO signals."
                )
                arr = arr.T  # Transpose to (Channels, Time)

            # If shape is (2, 2), ambiguous but assumes (Channels, Time)
            # If s1 > s0, likely already correct.

        return arr

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to handle metadata derivation and device placement.

        This method automatically derives `source_symbols` from `source_bits` if
        modulation parameters are present, and moves the signal samples to
        the GPU if a compatible device is available.
        """
        # Bit-first: derive symbols from bits if not provided
        if self.source_bits is not None and self.source_symbols is None:
            if self.mod_scheme and self.mod_order:
                from . import mapping

                self.source_symbols = mapping.map_bits(
                    self.source_bits,
                    self.mod_scheme,
                    self.mod_order,
                    self.mod_unipolar,
                )

        # Ensure source_symbols are normalized to unit average power for consistent metrics.
        # For MIMO (multichannel), we normalize per-stream (axis=-1) to ensure each stream
        # independently adheres to E_s=1, facilitating per-stream metric calculation.
        # Skip for PS-QAM: symbols are exact constellation points whose sample average
        # power is intentionally < 1 (MB weights inner points more). Scaling them would
        # break the correspondence with ps_pmf.
        if self.source_symbols is not None and self.ps_pmf is None:
            self.source_symbols = helpers.normalize(
                self.source_symbols, mode="average_power", axis=-1
            )

        # Default to GPU if available and supported
        if is_cupy_available():
            self.to("gpu")

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def print_info(self) -> None:
        """
        Prints a formatted summary of the signal's physical and digital properties.

        In Jupyter/IPython environments, this renders as an HTML table. In standard
        shells, it outputs a plain-text table via the logger.

        Sections
        --------
        **Signal** — always shown: type, waveform, rate, shape, backend.
        **Frame structure** — shown when ``frame`` carries frame metadata
        (preamble, payload, pilots, guard).
        **Reference data** — shows which symbol/bit arrays and frame object are
        attached (determines which of ``ber()``, ``evm()``
        can be called without extra arguments).
        """
        import pandas as pd
        from IPython import get_ipython
        from IPython.display import display

        # ── helpers ──────────────────────────────────────────────────────────
        def _yn(v) -> str:
            return "yes" if v is not None else "no"

        # Modulation: frame signals store it on the frame.
        frame = getattr(self, "frame", None)
        mod_scheme = self.mod_scheme or (getattr(frame, "payload_mod_scheme", None))
        mod_order = self.mod_order or (getattr(frame, "payload_mod_order", None))
        mod_unipolar = self.mod_unipolar or (
            getattr(frame, "payload_mod_unipolar", False)
        )
        mod_str = (
            f"{mod_scheme or 'None'} / {mod_order or 'None'}"
            f"{' (UNIPOL)' if mod_unipolar else ''}"
            f"{' (RZ)' if self.mod_rz else ''}"
        )
        bit_rate = (
            helpers.format_si(self.symbol_rate * np.log2(mod_order), "bps")
            if mod_order
            else "None"
        )

        # ── Section 1: Signal ─────────────────────────────────────────────
        sig_type_label = self.signal_type or "Signal"

        rows: list[tuple[str, str]] = [
            ("Signal type", sig_type_label),
            ("Spectral domain", self.spectral_domain),
            ("Physical domain", self.physical_domain),
            ("Modulation", mod_str),
            ("Symbol rate", helpers.format_si(self.symbol_rate, "Baud")),
            ("Bit rate", bit_rate),
            ("Sampling rate", helpers.format_si(self.sampling_rate, "Hz")),
            ("Samples per symbol", f"{self.sps:.2f}"),
            ("Pulse shape", self.pulse_shape.upper() if self.pulse_shape else "None"),
        ]

        if self.ps_pmf is not None and self.mod_order:
            _pmf = np.asarray(self.ps_pmf)
            _nz = _pmf > 0
            _h = float(-np.sum(_pmf[_nz] * np.log2(_pmf[_nz])))
            rows.append(
                ("PS entropy", f"{_h:.4f} b/sym  (max {np.log2(self.mod_order):.2f})")
            )

        rows += [
            ("Duration", helpers.format_si(self.duration, "s")),
            ("Center frequency", helpers.format_si(self.center_frequency, "Hz")),
            ("Freq. offset", helpers.format_si(self.digital_frequency_offset, "Hz")),
            ("Backend", self.backend.upper()),
            (
                "Configuration",
                "SISO" if self.num_streams == 1 else f"MIMO ({self.num_streams}x)",
            ),
            ("Samples shape", str(self.samples.shape)),
        ]

        # ── Section 2: Structure info (content varies by signal_type) ───────
        if self.signal_type == "Preamble" and self.frame is not None:
            preamble = getattr(self.frame, "preamble", None)
            if preamble is not None:
                rows.append(("─── Preamble info", ""))
                preamble_str = (
                    f"{preamble.sequence_type.upper()}  len={preamble.length}"
                )
                if preamble.sequence_type == "zc":
                    preamble_str += f"  root={preamble.root}"
                rows.append(("Sequence", preamble_str))

        elif self.signal_type == "Single-Carrier Frame" and self.frame is not None:
            frame = self.frame
            rows.append(("─── Frame structure", ""))

            if hasattr(frame, "preamble") and frame.preamble is not None:
                p = frame.preamble
                preamble_str = f"{p.sequence_type.upper()}  len={p.length}"
                if p.sequence_type == "zc":
                    preamble_str += f"  root={p.root}"
                rows.append(("Preamble", preamble_str))
            else:
                rows.append(("Preamble", "none"))

            if hasattr(frame, "payload_len") and frame.payload_len is not None:
                rows.append(("Payload length", f"{frame.payload_len} symbols"))

            pilot_pattern = getattr(frame, "pilot_pattern", "none")
            if pilot_pattern != "none":
                mask, _ = frame._generate_pilot_mask()
                pilot_count = (
                    int(np.sum(mask))
                    if hasattr(frame, "_generate_pilot_mask")
                    else None
                )
                pilot_period = getattr(frame, "pilot_period", None)
                pilot_gain_db = getattr(frame, "pilot_gain_db", None)

                pilot_str = (
                    f"{pilot_pattern}"
                    + (f"  count={pilot_count}" if pilot_count else "")
                    + (f"  period={pilot_period}" if pilot_period else "")
                    + (
                        f"  gain={pilot_gain_db} dB"
                        if pilot_gain_db is not None
                        else ""
                    )
                )
                rows.append(("Pilots", pilot_str))
            else:
                rows.append(("Pilots", "none"))

            if hasattr(frame, "guard_len") and frame.guard_len:
                rows.append(("Guard", f"{frame.guard_type}  len={frame.guard_len}"))
            else:
                rows.append(("Guard", "none"))

        # ── Section 3: Reference data ─────────────────────────────────────
        rows.append(("─── Reference data", ""))
        rows.append(("source_symbols", _yn(self.source_symbols)))
        rows.append(("source_bits", _yn(self.source_bits)))
        rows.append(("frame attached", _yn(self.frame)))

        # ── Section 4: Resolved data ──────────────────────────────────────
        rows.append(("─── Resolved data", ""))
        rows.append(("resolved_symbols", _yn(self.resolved_symbols)))
        rows.append(("resolved_bits", _yn(self.resolved_bits)))

        # ── Render ────────────────────────────────────────────────────────
        df = pd.DataFrame(rows, columns=["Property", "Value"])

        if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
            display(df.style.hide(axis="index"))
        else:
            logger.info("\n" + df.to_string(index=False))

    def copy(self) -> "Signal":
        """
        Creates a deep copy of the `Signal` instance.

        Returns
        -------
        Signal
            A new signal object with identical data and metadata.
        """
        return self.model_copy(deep=True)

    def to(self, device: str) -> "Signal":
        """
        Transfers signal data to the target device (CPU or GPU).

        Parameters
        ----------
        device : {"CPU", "GPU"}
            The target device. Case-insensitive.

        Returns
        -------
        Signal
            Returns self for method chaining.

        Raises
        ------
        ImportError
            If GPU is requested but CuPy is not installed/functional.
        """
        self.samples = to_device(self.samples, device)
        return self

    def export_samples_to_jax(self, device: Optional[str] = None) -> Any:
        """
        Exports the signal samples to a JAX array.

        Ensures zero-copy transfer to JAX when possible, preserving the
        device affinity of the underlying samples unless otherwise specified.

        Parameters
        ----------
        device : {"CPU", "GPU", "TPU"}, optional
            Target JAX device. If None, it targets the device matching the
            signal's current backend (CPU or GPU).

        Returns
        -------
        jax.Array
            JAX array containing signal samples.
            Shape: (N_channels, N_samples) or (N_samples,).
        """
        # If device is not explicitly requested, use the signal's backend
        target_device = device if device is not None else self.backend
        return to_jax(self.samples, device=target_device)

    def update_samples_from_jax(self, jax_array: Any) -> "Signal":
        """
        Updates signal samples from a JAX array.

        Converts the JAX array back to the signal's original backend (NumPy
        or CuPy) to maintain consistent state.

        Parameters
        ----------
        jax_array : jax.Array
            Input JAX array. Shape must match signal's expected shape.

        Returns
        -------
        Signal
            Returns self for method chaining.
        """
        original_backend = self.backend
        # Convert JAX array to backend-compatible array
        # from_jax will return NumPy (for CPU/TPU) or CuPy (for GPU)
        new_samples = from_jax(jax_array)

        # Ensure we move the data back to the original backend if it differs
        # (e.g., if signal was GPU but jax_array was on CPU/TPU)
        self.samples = to_device(new_samples, original_backend)

        return self

    def time_axis(self) -> ArrayType:
        """
        Generates the time vector associated with signal samples.

        Returns
        -------
        array_like
            Time axis in seconds, starting at 0.
            Shape: (N_samples,).
        """
        n_samples = self.samples.shape[-1]
        return self.xp.arange(0, n_samples) / self.sampling_rate

    def welch_psd(
        self,
        nperseg: int = 256,
        detrend: Optional[Union[str, bool]] = False,
        average: Optional[str] = "mean",
    ) -> Tuple[ArrayType, ArrayType]:
        """
        Compute the Power Spectral Density (PSD) using Welch's method.

        Parameters
        ----------
        nperseg : int, default 256
            Length of each segment for FFT.
        detrend : str or bool, default False
            Specifies how to detrend each segment (e.g., 'constant', 'linear').
        average : {"mean", "median"}, default "mean"
            Method to use for averaging overlapping segments.

        Returns
        -------
        f : array_like
            Array of sample frequencies. Shape: (N_freqs,).
        Pxx : array_like
            Power spectral density of the signal.
            Shape: (N_channels, N_freqs) or (N_freqs,).
        """
        from . import spectral

        return spectral.welch_psd(
            self.samples,
            sampling_rate=self.sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
            axis=-1,  # Explicitly specify time axis
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def xp(self) -> types.ModuleType:
        """
        Access the active array backend (NumPy or CuPy).

        This property allows for backend-agnostic code by returning the
        appropriate module based on where the samples currently reside.

        Returns
        -------
        module
            `numpy` if data is on CPU, `cupy` if on GPU.
        """
        return get_array_module(self.samples)

    @property
    def sp(self) -> types.ModuleType:
        """
        Access the signal processing module (`scipy` or `cupyx.scipy`).

        Returns
        -------
        module
            Appropriate signal processing library for the current backend.
        """
        return get_scipy_module(self.xp)

    @property
    def backend(self) -> str:
        """
        Returns the current computational backend name.

        Returns
        -------
        {"CPU", "GPU"}
            A string indicating the device location of samples.
        """
        return "GPU" if self.xp == cp else "CPU"

    @property
    def num_streams(self) -> int:
        """
        Returns the number of spatial or polarization streams.

        Returns
        -------
        int
            1 for SISO signals, N for MIMO/Dual-Pol signals.
        """
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        """
        Returns the total duration of the signal.

        Returns
        -------
        float
            Duration in seconds.
        """
        if self.samples.ndim == 1:
            return self.samples.shape[0] / self.sampling_rate
        return self.samples.shape[-1] / self.sampling_rate

    @property
    def sps(self) -> float:
        """
        Samples per symbol.

        Returns
        -------
        float
            Ratio of sampling rate to symbol rate.
        """
        return self.sampling_rate / self.symbol_rate

    @property
    def bits_per_symbol(self) -> Optional[int]:
        """
        Bits per symbol for the active modulation scheme.

        Returns
        -------
        int or None
            Calculated as ``log2(modulation_order)``.
        """
        if self.mod_order:
            return int(np.log2(self.mod_order))
        return None

    # -------------------------------------------------------------------------
    # Plotting and Visualization
    # -------------------------------------------------------------------------

    def plot_psd(
        self,
        nperseg: int = 128,
        detrend: Optional[Union[str, bool]] = False,
        average: Optional[str] = "mean",
        ax: Optional[Any] = None,
        title: Optional[str] = "Power Spectral Density",
        x_axis: Optional[str] = "frequency",
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots the Power Spectral Density (PSD) of the signal.

        Parameters
        ----------
        nperseg : int, default 128
            Length of each segment for Welch's method.
        detrend : str or bool, default False
            Specifies how to detrend each segment.
        average : {"mean", "median"}, default "mean"
            Method to use for averaging segments.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axis for plotting.
        title : str, default "Spectrum"
            Plot title.
        x_axis : {"frequency", "wavelength"}, default "frequency"
            The metric for the x-axis.
        show : bool, default False
            If True, calls `plt.show()` immediately.
        **kwargs : Any
            Additional arguments passed to `plotting.psd`.

        Returns
        -------
        figure, axis : tuple or None
            The matplotlib figure and axis objects, or None if `show=True`.
        """
        from . import plotting

        return plotting.psd(
            self.samples,
            sampling_rate=self.sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
            center_frequency=self.center_frequency,
            domain=self.physical_domain if self.physical_domain else "RF",
            x_axis=x_axis,
            ax=ax,
            title=title,
            show=show,
            **kwargs,
        )

    def plot_waveform(
        self,
        start_symbol: int = 0,
        num_symbols: int = None,
        ax: Optional[Any] = None,
        title: Optional[str] = "Waveform",
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots the time-domain waveform of the signal samples,
        starting from the specified symbol and for the specified number of symbols,
        so sps * num_symbols samples are plotted.

        Parameters
        ----------
        start_symbol : int, default 0
            The starting symbol to plot.
        num_symbols : int, optional
            Number of symbols to display. If None, plots the entire signal.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axis for plotting.
        title : str, default "Waveform"
            Plot title.
        show : bool, default False
            If True, calls `plt.show()` immediately.
        **kwargs : Any
            Additional arguments passed to `plotting.time_domain`.

        Returns
        -------
        figure, axis : tuple or None
            The matplotlib figure and axis objects, or None if `show=True`.
        """
        from . import plotting

        return plotting.time_domain(
            self.samples,
            sampling_rate=self.sampling_rate,
            start_symbol=start_symbol,
            num_symbols=num_symbols,
            sps=self.sps,
            ax=ax,
            title=title,
            show=show,
            **kwargs,
        )

    def plot_eye(
        self,
        ax: Optional[Any] = None,
        type: str = "hist",
        title: Optional[str] = "Eye Diagram",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots the eye diagram of the signal.

        The eye diagram is a critical tool for assessing ISI and timing
        jitter. This method supports both high-density histograms and
        classic trace-based line plots.

        Always plots from ``self.samples`` at ``self.sps`` samples per
        symbol.  To visualise a specific segment (e.g. payload symbols at
        1 SPS after equalization), build a new ``Signal`` from that data
        and call ``plot_eye()`` on it directly.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or list, optional
            Plotting axis. For complex signals, a list of two axes can be
            provided (for I and Q eyes).
        type : {"hist", "line"}, default "hist"
            ``"hist"``: 2D histogram density plot (recommended for noisy signals).
            ``"line"``: Overlaid individual signal traces.
        title : str, default "Eye Diagram"
            Plot title.
        vmin, vmax : float, optional
            Density scale limits for ``"hist"`` mode.
        show : bool, default False
            If True, calls ``plt.show()`` immediately.
        **kwargs : Any
            Additional parameters passed to the underlying plotting functions.

        Returns
        -------
        figure, axis : tuple or None
            The matplotlib figure and axis objects, or None if ``show=True``.
        """
        from . import plotting

        return plotting.eye_diagram(
            self.samples,
            ax=ax,
            sps=self.sps,
            type=type,
            title=title,
            vmin=vmin,
            vmax=vmax,
            show=show,
            **kwargs,
        )

    def plot_constellation(
        self,
        bins: int = 100,
        cmap: str = "inferno",
        overlay_ideal: bool = False,
        overlay_source: bool = False,
        ax: Optional[Any] = None,
        title: Optional[str] = "Constellation",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots the constellation diagram of the signal.

        Uses a 2D histogram density map of ``self.samples`` to visualise the
        distribution of received symbols, providing better insight into noise
        and impairments than traditional scatter plots.

        To visualise a specific segment (e.g. equalised payload symbols at
        1 SPS), build a new ``Signal`` from that data and call
        ``plot_constellation()`` on it directly.

        Parameters
        ----------
        bins : int, default 100
            Number of bins per axis for the 2D density histogram.
        cmap : str, default "inferno"
            Matplotlib colormap for the density visualization.
        overlay_ideal : bool, default False
            If True, overlays the ideal constellation points on the plot.
            Requires ``mod_scheme`` and ``mod_order`` to be set.
        overlay_source : bool, default False
            If True, overlays ``source_symbols`` as cyan scatter markers on
            top of the density plot.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axis for plotting.
        title : str, default "Constellation"
            Plot title.
        vmin, vmax : float, optional
            Density scale limits.
        show : bool, default False
            If True, calls ``plt.show()`` immediately.
        **kwargs : Any
            Additional arguments passed to ``plotting.constellation``.

        Returns
        -------
        figure, axis : tuple or None
            The matplotlib figure and axis objects, or None if ``show=True``.
        """
        from . import plotting

        result = plotting.constellation(
            self.samples,
            bins=bins,
            cmap=cmap,
            ax=ax,
            overlay_ideal=overlay_ideal,
            modulation=self.mod_scheme,
            order=self.mod_order,
            unipolar=self.mod_unipolar,
            pmf=self.ps_pmf,
            title=title,
            vmin=vmin,
            vmax=vmax,
            show=False,
            **kwargs,
        )

        if overlay_source and self.source_symbols is not None and result is not None:
            fig, axes = result
            src = to_device(self.source_symbols, "cpu")

            # PS-QAM: source_symbols are on the {s_m} grid (average power E_PS < 1)
            # but received samples normalise to unit power ({s_m/sqrt(E_PS)}).
            # Scale source symbols to match the received symbol scale.
            if (
                self.ps_pmf is not None
                and self.mod_scheme is not None
                and self.mod_order is not None
            ):
                from .mapping import gray_constellation as _gc_src

                _const_src = _gc_src(self.mod_scheme, self.mod_order)
                _pmf_src = np.asarray(self.ps_pmf, dtype=np.float64)
                _e_ps = float(np.dot(_pmf_src, np.abs(_const_src) ** 2))
                if _e_ps > 0 and _e_ps < 1.0 - 1e-6:
                    src = src / np.sqrt(_e_ps)

            def _scatter_source(ax, symbols):
                ax.scatter(
                    symbols.real,
                    symbols.imag,
                    c="lime",
                    edgecolors="dimgray",
                    linewidths=1.5,
                    s=30,
                    zorder=10,
                    marker="o",
                )

            if src.ndim > 1:
                ax_list = list(np.asarray(axes).flat)
                for ch in range(min(src.shape[0], len(ax_list))):
                    _scatter_source(ax_list[ch], src[ch])
            else:
                _scatter_source(axes, src)

        if show:
            import matplotlib.pyplot as plt

            plt.show()
            return None
        return result

    # -------------------------------------------------------------------------
    # Signal Processing Methods
    # -------------------------------------------------------------------------

    def upsample(self, factor: int, correct_power: bool = True) -> "Signal":
        """
        Upsamples the signal by an integer factor using polyphase filtering.

        This method increases the sampling rate by inserting zeros and
        applying an anti-imaging filter.

        Parameters
        ----------
        factor : int
            Upsampling factor (interpolation).
        correct_power : bool, default True
            If True, applies a deterministic amplitude gain of
            ``1 / sqrt(factor)`` after upsampling to restore the
            ``"symbol_power"`` invariant (``E[|x|²] = 1 / sps``).

            **Why this is needed**: ``resample_poly`` has unity DC gain, so
            for a bandlimited pulse-shaped signal it preserves average sample
            power.  After upsampling by *factor*, ``sps`` grows by *factor*
            but ``E[|x|²]`` stays at ``1 / sps_old``, violating
            ``E[|x|²] = 1 / sps_new``.  The correction factor restores the
            invariant without measuring signal statistics.

            Set to False when chaining custom gain stages or when the signal
            is not ``"symbol_power"`` normalized.

        Returns
        -------
        Signal
            self (modified in-place).
        """
        from . import multirate

        self.samples = multirate.upsample(self.samples, factor, axis=-1)
        self.sampling_rate = self.sampling_rate * factor
        if correct_power:
            self.samples = self.samples * (factor**-0.5)
        return self

    def decimate(
        self,
        factor: int,
        filter_type: str = "fir",
        correct_power: bool = True,
        **kwargs: Any,
    ) -> "Signal":
        """
        Decimates the signal by an integer factor using anti-aliasing filtering.

        Parameters
        ----------
        factor : int
            Decimation factor (e.g., 2 to reduce sample rate by half).
        filter_type : {"fir", "polyphase"}, default "fir"
            The type of decimation filter to use.
        correct_power : bool, default True
            If True, applies a deterministic amplitude gain of
            ``sqrt(factor)`` after decimation to restore the
            ``"symbol_power"`` invariant (``E[|x|²] = 1 / sps``).

            **Why this is needed**: ``resample_poly`` has unity DC gain, so
            for a bandlimited pulse-shaped signal it preserves average sample
            power.  After decimating by *factor*, ``sps`` shrinks by *factor*
            but ``E[|x|²]`` stays at ``1 / sps_old``, violating
            ``E[|x|²] = 1 / sps_new``.  The correction factor restores the
            invariant without measuring signal statistics.

            Set to False when chaining custom gain stages or when the signal
            is not ``"symbol_power"`` normalized.

        **kwargs : Any
            Additional parameters passed to the decimation algorithm.

        Returns
        -------
        Signal
            self (modified in-place).

        Notes
        -----
        This method applies an anti-aliasing filter before downsampling. If
        the signal has already been matched-filtered, you should likely use
        `decimate_to_symbol_rate` instead to avoid double-filtering.
        """
        from . import multirate

        self.samples = multirate.decimate(
            self.samples, factor, filter_type=filter_type, axis=-1, **kwargs
        )
        self.sampling_rate = self.sampling_rate / factor
        if correct_power:
            self.samples = self.samples * (factor**0.5)
        return self

    def decimate_to_symbol_rate(
        self, offset: int = 0, normalize: bool = True
    ) -> "Signal":
        """
        Extracts symbols from an oversampled signal via direct slicing.

        This method is the canonical way to recover symbols at $1 \text{ sps}$
        after matched filtering. It does not apply any additional filtering,
        ensuring that the matched filter remains the optimal receiver.

        Parameters
        ----------
        offset : int, default 0
            The sampling phase offset in samples. Adjust this to sample
            at the peak of the impulse response or the maximum eye opening.
        normalize : bool, default True
            If ``True``, normalizes the output to unit average power
            (``mean(|x|²) = 1``) after slicing.  This is almost always
            desirable because practical channels introduce gain uncertainty,
            timing residuals, and filter mismatch that shift symbol power
            away from the theoretical unit value, even after matched
            filtering.  Set to ``False`` only if you need to observe or
            measure the raw channel gain.

        Returns
        -------
        Signal
            self (modified in-place, sampling rate updated to symbol rate).

        Notes
        -----
        Use `resample` or `decimate` for general rate changes where
        anti-aliasing is required. Use this method ONLY when the signal
        is already filtered and aligned.
        """
        from . import multirate

        sps = int(self.sps)
        if sps <= 1:
            logger.info("Signal already at 1 sps, no downsampling needed.")
            return self
        self.samples = multirate.decimate_to_symbol_rate(
            self.samples, sps=sps, offset=offset, axis=-1
        )
        self.sampling_rate = self.symbol_rate
        if normalize:
            self.samples = helpers.normalize(self.samples, "average_power", axis=-1)
        return self

    def resample(
        self,
        up: Optional[int] = None,
        down: Optional[int] = None,
        sps_out: Optional[float] = None,
        correct_power: bool = True,
    ) -> "Signal":
        """
        Resamples the signal by a rational factor using polyphase filtering.

        This method is the canonical way to change the sampling rate of a
        signal while maintaining spectral integrity through anti-aliasing
        and anti-imaging filters.

        Parameters
        ----------
        up : int, optional
            Upsampling factor (interpolation).
        down : int, optional
            Downsampling factor (decimation).
        sps_out : float, optional
            Desired samples per symbol in the output signal. If provided,
            the required `up/down` factors are calculated automatically
            relative to the current `sps`.
        correct_power : bool, default True
            If True, applies a deterministic amplitude gain of
            ``sqrt(sps_before / sps_after)`` after resampling to restore the
            ``"symbol_power"`` invariant (``E[|x|²] = 1 / sps``).

            **Why this is needed**: ``resample_poly`` has unity DC gain, so
            for a bandlimited pulse-shaped signal it preserves average sample
            power.  After any rational rate change, ``sps`` changes but
            ``E[|x|²]`` stays at ``1 / sps_old``, violating
            ``E[|x|²] = 1 / sps_new``.  The correction factor is exact
            (not statistical) because signal power is known at creation.

            Set to False when chaining custom gain stages or when the signal
            is not ``"symbol_power"`` normalized.

        Returns
        -------
        Signal
            self (modified in-place, sampling rate updated).

        Notes
        -----
        .. warning::
            Do NOT use this method on a signal that has already been
            matched-filtered if the goal is to extract symbols at $1\\text{ sps}$.
            The polyphase filter's transition band will distort the
            optimally filtered pulse shape. Use `decimate_to_symbol_rate` instead.
        """
        from . import multirate

        sps_before = self.sps

        # If sps_out is provided, we use the current signal's sps as input sps
        sps_in = self.sps if sps_out is not None else None

        self.samples = multirate.resample(
            self.samples, up=up, down=down, sps_in=sps_in, sps_out=sps_out, axis=-1
        )

        # Update sampling rate
        if sps_out is not None:
            self.sampling_rate = sps_out * self.symbol_rate
        elif up is not None and down is not None:
            self.sampling_rate = self.sampling_rate * up / down

        if correct_power:
            self.samples = self.samples * (sps_before / self.sps) ** 0.5

        return self

    def shift_frequency(self, offset: float) -> "Signal":
        """
        Applies a frequency offset (mixing) to the signal.

        This method shifts the signal's spectrum in the frequency domain.
        Metadata is updated to track total digital frequency offset.

        Parameters
        ----------
        offset : float
            Desired frequency shift in Hz.

        Returns
        -------
        Signal
            self (modified in-place).
        """
        from . import spectral

        self.samples, actual_offset = spectral.shift_frequency(
            self.samples, offset, self.sampling_rate
        )
        self.digital_frequency_offset += actual_offset
        return self

    def fir_filter(self, taps: ArrayType) -> "Signal":
        """
        Applies a Finite Impulse Response (FIR) filter to the signal.

        Parameters
        ----------
        taps : array_like
            FIR filter coefficients. Shape: (N_taps,).

        Returns
        -------
        Signal
            self (modified in-place).
        """
        from . import filtering

        self.samples = filtering.fir_filter(self.samples, taps, axis=-1)
        return self

    def shaping_filter_taps(self) -> ArrayType:
        """
        Computes shaping filter taps based on signal metadata.

        Returns
        -------
        array_like
            Generated filter taps normalized according to pulse type.
            Resides on the same device as the signal samples.

        Raises
        -------
        ValueError
            If `pulse_shape` is missing or unsupported.
        """
        if not self.pulse_shape or self.pulse_shape == "none":
            raise ValueError("No pulse shape defined for this signal.")
        logger.info(f"Generating shaping filter taps (shape: {self.pulse_shape}).")
        from . import filtering

        # Determine pulse width based on modulation if RZ
        p_width = 0.5 if self.mod_rz else 1.0

        # Generate taps using filtering module (returns default numpy usually)
        if self.pulse_shape == "rect":
            taps = np.ones(int(self.sps * p_width))
        elif self.pulse_shape == "smoothrect":
            taps = filtering.smoothrect_taps(
                sps=self.sps,
                span=self.filter_span,
                bt=self.smoothrect_bt,
                pulse_width=p_width,
            )
        elif self.pulse_shape == "gaussian":
            taps = filtering.gaussian_taps(
                sps=self.sps,
                span=self.filter_span,
                bt=self.gaussian_bt,
            )
        elif self.pulse_shape == "rrc":
            taps = filtering.rrc_taps(
                sps=self.sps,
                span=self.filter_span,
                rolloff=self.rrc_rolloff,
            )
        elif self.pulse_shape == "rc":
            taps = filtering.rc_taps(
                sps=self.sps,
                span=self.filter_span,
                rolloff=self.rc_rolloff,
            )
        else:
            raise ValueError(f"Unknown pulse shape: {self.pulse_shape}")

        # Ensure taps are on the correct device
        return to_device(taps, self.backend)

    def matched_filter(
        self,
        taps=None,
        taps_normalization: str = "unit_energy",
    ) -> "Signal":
        """
        Applies a matched filter to the signal samples.

        When *taps* is not provided, the filter is derived automatically from
        the ``pulse_shape`` and related parameters stored in the signal metadata.

        Parameters
        ----------
        taps : array-like or None, default None
            Explicit filter taps to use as the matched filter.  If ``None``,
            taps are generated from the signal's ``pulse_shape`` metadata.
        taps_normalization : {"unit_energy", "unity_gain"}, default "unit_energy"
            Normalization strategy for the filter taps.

        Returns
        -------
        Signal
            self (modified in-place).
        """
        from . import filtering

        if taps is None:
            try:
                taps = self.shaping_filter_taps()
            except ValueError as e:
                logger.error(f"Cannot apply matched filter: {e}")
                return self
        self.samples = filtering.matched_filter(
            self.samples,
            taps,
            taps_normalization=taps_normalization,
            axis=-1,
        )
        return self

    # -------------------------------------------------------------------------
    # Generation Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        modulation: str,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: str = "none",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generates a generic baseband waveform with specified modulation.

        This is the primary factory method for creating synthetic signals.
        It follows a bit-first architecture: random bits are generated,
        mapped to symbols, upsampled, and pulse-shaped.

        Parameters
        ----------
        num_symbols : int
            Number of symbols to generate per stream.
        sps : float
            Samples per symbol.
        symbol_rate : float
            Symbol rate in symbols per second (Baud).
        modulation : {"psk", "qam", "ask"}
            The modulation scheme identifier.
        order : int
            Modulation order (e.g., 4, 16, 64).
        unipolar : bool, default False
            If True, uses a unipolar constellation.
        rz : bool, default False
            If True, uses Return-to-Zero signaling.
        pulse_shape : str, default "none"
            Pulse shaping filter type (e.g., 'rrc', 'rect').
        num_streams : int, default 1
            Number of independent streams (MIMO).
        seed : int, optional
            Seed for reproducible random generation.
        **kwargs : Any
            Additional filter parameters (e.g., `filter_span`, `rrc_rolloff`).

        Returns
        -------
        Signal
            A new `Signal` instance.

        Notes
        -----
        Symbols are ``complex64`` for PSK/QAM and ``float32`` for ASK/PAM.
        The generated samples are normalized to **unit symbol power (Es = 1)**
        via `filtering.shape_pulse`, meaning average sample power = 1/sps.
        This matches the convention expected by ``apply_awgn``:
        ``Es = mean_sample_power x sps = 1``, so Es/N0 calibration is exact for
        all pulse shapes without any per-pulse offset.
        Call ``resolve_symbols()`` to populate ``resolved_symbols`` at unit
        average power (Es = 1, 1 SPS) before demapping or computing metrics.
        """
        from . import filtering, mapping

        # Bit-first architecture: generate bits → map to symbols
        k = int(np.log2(order))  # bits per symbol
        total_symbols = num_symbols * num_streams
        total_bits = total_symbols * k

        # Generate source bits
        bits = helpers.random_bits(total_bits, seed=seed)

        # Map bits to symbols
        symbols_flat = mapping.map_bits(bits, modulation, order, unipolar)

        if num_streams > 1:
            # Shape: (Channels, Time)
            symbols = symbols_flat.reshape(num_streams, num_symbols)
            bits = bits.reshape(num_streams, num_symbols * k)
        else:
            symbols = symbols_flat

        if is_cupy_available():
            symbols = to_device(symbols, "gpu")
            bits = to_device(bits, "gpu")

        # Apply pulse shaping
        # shape_pulse defaults to axis=-1 (Time) which is correct for (C, T)
        samples = filtering.shape_pulse(
            symbols=symbols, sps=sps, pulse_shape=pulse_shape, rz=rz, **kwargs
        )

        return cls(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            mod_scheme=modulation.upper(),
            mod_order=order,
            mod_unipolar=unipolar,
            mod_rz=rz,
            source_bits=bits,
            source_symbols=symbols,
            pulse_shape=pulse_shape,
            **kwargs,
        )

    @classmethod
    def pam(
        cls,
        num_symbols: int,
        sps: int,
        symbol_rate: float,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: Literal["rect", "smoothrect"] = "rect",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generates a Pulse Amplitude Modulation (PAM) baseband waveform.

        Supports both NRZ (Non-Return-to-Zero) and RZ (Return-to-Zero)
        signaling, with configurable pulse shaping and bipolar/unipolar
        constellations.

        Parameters
        ----------
        num_symbols : int
            Total number of symbols to generate per stream.
        sps : int
            Samples per symbol. For RZ mode, this must be an even integer.
        symbol_rate : float
            Symbol rate in symbols per second (Baud).
        order : int
            Modulation order (e.g., 2, 4, 8).
        unipolar : bool, default False
            If True, uses a unipolar constellation starting from 0 (e.g., 0, 1).
            If False, uses a symmetric bipolar constellation (e.g., -1, +1).
        rz : bool, default False
            If True, uses Return-to-Zero signaling.
        pulse_shape : {"rect", "smoothrect"}, default "rect"
            Pulse shaping filter type. Default is "rect" for PAM.
        num_streams : int, default 1
            Number of independent streams (channels) to generate.
        seed : int, optional
            Random seed for reproducible bit and symbol generation.
        **kwargs : Any
            Additional parameters passed to the pulse shaping filter.

        Returns
        -------
        Signal
            A `Signal` object containing the generated PAM waveform.

        Notes
        -----
        The generated samples are normalized to **unit symbol power (Es = 1)**
        via `filtering.shape_pulse`, meaning average sample power = 1/sps.
        This matches the convention expected by ``apply_awgn``:
        ``Es = mean_sample_power x sps = 1``, so Es/N0 calibration is exact for
        all pulse shapes without any per-pulse offset.
        Call ``resolve_symbols()`` to populate ``resolved_symbols`` at unit
        average power (Es = 1, 1 SPS) before demapping or computing metrics.
        """
        if rz:
            if sps % 2 != 0:
                raise ValueError("For correct RZ duty cycle, `sps` must be even")

            allowed_rz_pulses = ["rect", "smoothrect"]
            if pulse_shape not in allowed_rz_pulses:
                raise ValueError(
                    f"Pulse shape '{pulse_shape}' is not allowed for RZ PAM. "
                    f"Allowed: {allowed_rz_pulses}"
                )

        return cls.generate(
            num_symbols=num_symbols,
            sps=sps,
            symbol_rate=symbol_rate,
            modulation="PAM",
            order=order,
            unipolar=unipolar,
            rz=rz,
            pulse_shape=pulse_shape,
            num_streams=num_streams,
            seed=seed,
            **kwargs,
        )

    @classmethod
    def psk(
        cls,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generates a Phase Shift Keying (PSK) baseband waveform.

        Parameters
        ----------
        num_symbols : int
            Total number of symbols to generate per stream.
        sps : float
            Samples per symbol.
        symbol_rate : float
            Symbol rate in symbols per second (Baud).
        order : int
            Modulation order (e.g., 2 for BPSK, 4 for QPSK, 8 for 8-PSK).
        unipolar : bool, default False
            If True, uses a unipolar constellation.
        rz : bool, default False
            If True, uses Return-to-Zero signaling.
        pulse_shape : str, default "rrc"
            Pulse shaping filter type.
        num_streams : int, default 1
            Number of independent streams (channels) to generate.
        seed : int, optional
            Random seed for bit and symbol generation.
        **kwargs : Any
            Additional parameters passed to `filtering.shape_pulse`.

        Returns
        -------
        Signal
            A `Signal` object containing the PSK waveform.

        Notes
        -----
        The generated samples are normalized to **unit symbol power (Es = 1)**
        via `filtering.shape_pulse`, meaning average sample power = 1/sps.
        This matches the convention expected by ``apply_awgn``:
        ``Es = mean_sample_power x sps = 1``, so Es/N0 calibration is exact for
        all pulse shapes without any per-pulse offset.
        Call ``resolve_symbols()`` to populate ``resolved_symbols`` at unit
        average power (Es = 1, 1 SPS) before demapping or computing metrics.
        """
        return cls.generate(
            modulation="psk",
            order=order,
            num_symbols=num_symbols,
            sps=sps,
            symbol_rate=symbol_rate,
            pulse_shape=pulse_shape,
            num_streams=num_streams,
            seed=seed,
            unipolar=unipolar,
            rz=rz,
            **kwargs,
        )

    @classmethod
    def qam(
        cls,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generates a Quadrature Amplitude Modulation (QAM) baseband waveform.

        Parameters
        ----------
        num_symbols : int
            Number of symbols to generate per stream.
        sps : float
            Samples per symbol.
        symbol_rate : float
            Symbol rate in symbols per second (Baud).
        order : int
            Modulation order (e.g., 16, 64, 256).
        unipolar : bool, default False
            If True, uses a unipolar constellation.
        rz : bool, default False
            If True, uses Return-to-Zero signaling.
        pulse_shape : str, default "rrc"
            Pulse shaping filter type.
        num_streams : int, default 1
            Number of MIMO streams.
        seed : int, optional
            Seed for random generation.
        **kwargs : Any
            Additional filter parameters.

        Returns
        -------
        Signal
            A `Signal` object containing the QAM waveform.

        Notes
        -----
        The generated samples are normalized to **unit symbol power (Es = 1)**
        via `filtering.shape_pulse`, meaning average sample power = 1/sps.
        This matches the convention expected by ``apply_awgn``:
        ``Es = mean_sample_power x sps = 1``, so Es/N0 calibration is exact for
        all pulse shapes without any per-pulse offset.
        Call ``resolve_symbols()`` to populate ``resolved_symbols`` at unit
        average power (Es = 1, 1 SPS) before demapping or computing metrics.
        """
        return cls.generate(
            modulation="qam",
            order=order,
            num_symbols=num_symbols,
            sps=sps,
            symbol_rate=symbol_rate,
            pulse_shape=pulse_shape,
            num_streams=num_streams,
            seed=seed,
            unipolar=unipolar,
            rz=rz,
            **kwargs,
        )

    @classmethod
    def psqam(
        cls,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        order: int,
        *,
        nu: Optional[float] = None,
        entropy: Optional[float] = None,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generates a Probabilistically Shaped QAM (PS-QAM) baseband waveform.

        Symbols are drawn from a Maxwell-Boltzmann (MB) distribution over the
        normalized QAM constellation, giving inner (low-energy) points higher
        probability. This recovers up to 1.53 dB shaping gain over uniform QAM.

        Exactly one of ``nu`` or ``entropy`` must be specified.

        Parameters
        ----------
        num_symbols : int
            Number of symbols to generate per stream.
        sps : float
            Samples per symbol.
        symbol_rate : float
            Symbol rate in symbols per second (Baud).
        order : int
            QAM modulation order (e.g. 16, 64, 256).
        nu : float, optional
            MB shaping parameter ``ν ≥ 0``. ``ν = 0`` is uniform QAM.
            Larger values apply stronger shaping (lower entropy, lower power).
        entropy : float, optional
            Target per-symbol entropy in bits, in the range ``(0, log₂(order)]``.
            ``optimal_nu`` is called to solve for the corresponding ``ν``.
        pulse_shape : str, default "rrc"
            Pulse shaping filter type.
        num_streams : int, default 1
            Number of independent streams (MIMO).
        seed : int, optional
            Random seed for reproducible symbol generation.
        **kwargs : Any
            Additional filter parameters (e.g. ``filter_span``, ``rrc_rolloff``).

        Returns
        -------
        Signal
            A ``Signal`` with ``mod_scheme="PS-QAM"``, ``ps_pmf`` set to the MB
            distribution, and both ``source_symbols`` and ``source_bits`` populated.

        Notes
        -----
        ``source_bits`` are obtained by hard-demapping the clean shaped symbols —
        they carry the non-uniform statistics of the MB distribution, not uniform
        bits. This is correct for uncoded physical-layer BER and GMI estimation
        but does not model a full coded PAS transmitter.

        The average symbol energy is below 1 by design (``E_ps[|s|²] < 1`` for
        ``ν > 0``). Pass ``pmf=signal.ps_pmf`` to :func:`~commstools.metrics.mi`
        and :func:`~commstools.mapping.compute_llr` to account for the non-uniform
        prior in capacity and soft-demapping computations.

        Examples
        --------
        Generate 64-PS-QAM at 6 bits/symbol effective rate::

            sig = Signal.psqam(10000, sps=4, symbol_rate=32e9, order=64, entropy=6.0)

        Generate with explicit shaping parameter::

            sig = Signal.psqam(10000, sps=4, symbol_rate=32e9, order=64, nu=0.3)
        """
        from . import filtering, mapping

        if (nu is None) == (entropy is None):
            raise ValueError("Exactly one of `nu` or `entropy` must be specified.")

        if entropy is not None:
            nu_val, _ = mapping.optimal_nu(order, entropy)
        else:
            nu_val = float(nu)
            if nu_val < 0:
                raise ValueError("`nu` must be non-negative.")

        pmf = mapping.maxwell_boltzmann(order, nu_val)
        k = int(np.log2(order))
        total_symbols = num_symbols * num_streams

        # Sample symbols from MB distribution (NumPy, CPU)
        symbols_flat = mapping.sample_ps_symbols(total_symbols, order, pmf, seed=seed)

        # Derive source bits by demapping noiseless shaped symbols (lossless)
        bits_flat = mapping.demap_symbols_hard(symbols_flat, "qam", order)

        if num_streams > 1:
            symbols = symbols_flat.reshape(num_streams, num_symbols)
            bits = bits_flat.reshape(num_streams, num_symbols * k)
        else:
            symbols = symbols_flat
            bits = bits_flat

        if is_cupy_available():
            symbols = to_device(symbols, "gpu")
            bits = to_device(bits, "gpu")

        samples = filtering.shape_pulse(
            symbols=symbols, sps=sps, pulse_shape=pulse_shape, **kwargs
        )

        return cls(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            mod_scheme="PS-QAM",
            mod_order=order,
            source_bits=bits,
            source_symbols=symbols,
            pulse_shape=pulse_shape,
            ps_pmf=pmf,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Resolving and Demapping Methods
    # -------------------------------------------------------------------------

    def resolve_symbols(self, offset: int = 0) -> None:
        """Decimate to symbol rate and populate ``resolved_symbols``.

        Decimates ``self.samples`` to 1 SPS and normalises to unit average
        power ($E_s = 1$), writing the result to ``self.resolved_symbols``.

        Parameters
        ----------
        offset : int, default 0
            Sample offset applied before decimation (integer only).
            Use after timing correction to select the optimal sampling instant.

        Notes
        -----
        The normalisation step absorbs any gain introduced by the channel,
        equaliser, or matched filter.  It does not change the constellation
        shape, only the scale.

        Designed for plain (non-frame) signals at a single modulation/gain.
        Frame signals contain preamble, pilots, and payload packed together,
        potentially with different modulations or pilot boosts.  Calling
        this on a frame signal is skipped with a warning; extract the
        desired segment, create a plain ``Signal``, then call
        ``resolve_symbols()`` on that.

        Raises
        ------
        ValueError
            If ``symbol_rate`` or ``sampling_rate`` are missing, or SPS is
            not a positive integer.
        """
        if self.signal_type is not None:
            logger.warning(
                "resolve_symbols() called on a frame-generated signal — skipping. "
                "Frame signals mix preamble, pilots, and payload segments that may "
                "have different modulations or gains. Extract the desired segment "
                "via frame.get_structure_map(), build a plain Signal, then call "
                "resolve_symbols() on that."
            )
            return

        sps = self.sps
        if sps is None:
            raise ValueError("Symbol rate or sampling rate missing.")

        if sps < 1:
            raise ValueError("Symbol rate must be >= 1.")

        if sps % 1 != 0:
            raise ValueError("Symbol rate must be an integer.")

        if sps == 1:
            logger.info("Signal already at 1 sps, no downsampling needed.")
            res = self.samples
        else:
            from . import multirate

            logger.info(
                f"SpS is not 1. Decimating to symbol rate with sps={sps} and offset={offset}."
            )
            res = multirate.decimate_to_symbol_rate(
                self.samples, sps=int(sps), offset=int(offset), axis=-1
            )

        self.resolved_symbols = helpers.normalize(res, "average_power", axis=-1)

    def demap_symbols_hard(
        self,
        **kwargs: Any,
    ) -> None:
        """Map resolved symbols to bits and store in ``self.resolved_bits``.

        Performs hard-decision demapping via minimum Euclidean distance and
        writes the result to ``self.resolved_bits``.  Call
        :meth:`resolve_symbols` first.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments forwarded to :func:`mapping.demap_symbols_hard`
            (e.g. ``unipolar``).

        Raises
        ------
        ValueError
            If modulation metadata is missing or :meth:`resolve_symbols` has
            not been called yet.
        """
        from .mapping import demap_symbols_hard

        if self.signal_type is not None:
            logger.warning(
                "demap_symbols_hard() called on a frame-generated signal — skipping. "
                "Extract the payload segment via frame.get_structure_map() and build "
                "a plain Signal before demapping."
            )
            return

        if self.mod_scheme is None or self.mod_order is None:
            raise ValueError("Modulation scheme and order required for demapping.")

        if self.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Please call `resolve_symbols()` "
                "first to decimate the signal to symbol rate."
            )

        bits = demap_symbols_hard(
            symbols=self.resolved_symbols,
            modulation=self.mod_scheme,
            order=self.mod_order,
            unipolar=self.mod_unipolar,
            **kwargs,
        )
        self.resolved_bits = bits

    def resolve_phase_ambiguity(self) -> None:
        """
        Resolves rotational phase ambiguity in ``resolved_symbols`` in place.

        After blind CPR (VV, BPS, Tikhonov) a global ``π/2`` (QAM) or
        ``2π/M`` (PSK) phase ambiguity remains per stream.  This method tests
        all symmetry rotations, scores each against ``source_symbols`` by SER,
        and overwrites ``resolved_symbols`` with the best rotation.

        For MIMO (``resolved_symbols`` shape ``(C, N)``), each channel is
        resolved independently — after MIMO equalisation different output
        streams may land on different ambiguity branches.

        The result is written to ``self.resolved_symbols`` in place.

        Raises
        ------
        ValueError
            If ``resolved_symbols``, ``source_symbols``, ``mod_scheme``, or
            ``mod_order`` are not populated.
        """
        from .sync import resolve_phase_ambiguity as _resolve

        if self.resolved_symbols is None:
            raise ValueError(
                "resolved_symbols is not set. Call resolve_symbols() or assign "
                "resolved_symbols directly before calling resolve_phase_ambiguity()."
            )
        if self.source_symbols is None:
            raise ValueError(
                "source_symbols is not set. Populate source_symbols (the known TX "
                "symbol sequence) before calling resolve_phase_ambiguity()."
            )
        if self.mod_scheme is None or self.mod_order is None:
            raise ValueError("mod_scheme and mod_order must be set.")

        self.resolved_symbols = _resolve(
            symbols=self.resolved_symbols,
            ref_symbols=self.source_symbols,
            modulation=self.mod_scheme,
            order=self.mod_order,
        )

    # -------------------------------------------------------------------------
    # Metrics Methods
    # -------------------------------------------------------------------------

    def evm(
        self,
        reference_symbols: Optional[ArrayType] = None,
        num_train_symbols: Optional[int] = None,
        *,
        mode: str = "data_aided",
        modulation: Optional[str] = None,
        order: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Computes the Error Vector Magnitude (EVM).

        Parameters
        ----------
        reference_symbols : array_like, optional
            Known transmitted symbols. Falls back to ``source_symbols`` when
            not provided.  Ignored when ``mode="blind"``.
        num_train_symbols : int, optional
            Number of symbols to discard from the front before computing the
            metric (e.g. equalizer training length).
        mode : {"data_aided", "blind"}, default "data_aided"
            Estimation mode (keyword-only).  ``"blind"`` uses ML hard
            decisions against the constellation as the reference — no
            knowledge of the transmitted sequence is required.
        modulation : str, optional
            Modulation type for blind mode (``"psk"``, ``"qam"``, ``"ask"``).
            Falls back to ``self.mod_scheme`` when not provided.
        order : int, optional
            Modulation order for blind mode.  Falls back to
            ``self.mod_order`` when not provided.

        Returns
        -------
        evm_percent : float
            EVM as a percentage.
        evm_db : float
            EVM in decibels (dB).

        Raises
        ------
        ValueError
            If ``resolved_symbols`` is not set, or required arguments for
            the chosen mode are missing.

        Notes
        -----
        For frame-generated signals, extract the payload segment manually
        via ``frame.get_structure_map()`` and create a plain ``Signal``
        with the appropriate ``source_symbols`` before calling this method.
        """
        from . import metrics

        if self.signal_type is not None:
            logger.warning(
                "evm() called on a frame-generated signal. For accurate per-segment "
                "metrics, extract the payload (or pilot) segment manually via "
                "frame.get_structure_map() and create a plain Signal first."
            )
            return

        if self.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Please call `resolve_symbols()` "
                "first to decimate the signal to symbol rate."
            )

        y = self.resolved_symbols
        trim = num_train_symbols if num_train_symbols is not None else 0

        if mode == "blind":
            mod = modulation or self.mod_scheme
            ord_ = order or self.mod_order
            if mod is None or ord_ is None:
                raise ValueError(
                    "mode='blind' requires modulation and order. "
                    "Pass them explicitly or ensure mod_scheme/mod_order are set on the Signal."
                )
            if trim > 0:
                logger.info(f"Discarding {trim} training symbols for EVM calculation.")
                y = y[..., trim:]
            return metrics.evm(y, mode="blind", modulation=mod, order=ord_)

        # data_aided
        ref = (
            reference_symbols if reference_symbols is not None else self.source_symbols
        )
        if ref is None:
            raise ValueError(
                "No reference available. Provide reference_symbols or ensure "
                "source_symbols is set."
            )
        r = ref
        if trim > 0:
            logger.info(f"Discarding {trim} training symbols for EVM calculation.")
            n = min(y.shape[-1], r.shape[-1])
            y = y[..., trim:n]
            r = r[..., trim:n]

        return metrics.evm(y, r)

    def snr(
        self,
        reference_symbols: Optional[ArrayType] = None,
        num_train_symbols: Optional[int] = None,
    ) -> float:
        """
        Estimates the Signal-to-Noise Ratio (SNR) using a Data-Aided method.

        This method calculates the ratio of reference signal power over
        the variance of the error vector.

        Parameters
        ----------
        reference_symbols : array_like, optional
            Known transmitted symbols. If None, falls back to
            ``source_symbols`` when available.
        num_train_symbols : int, optional
            Number of symbols to discard from the front before computing the
            metric (e.g. equalizer training length).

        Returns
        -------
        float
            Estimated SNR in decibels (dB).

        Raises
        ------
        ValueError
            If no reference symbols are available or ``resolved_symbols``
            has not been set (call ``resolve_symbols()`` first).

        Notes
        -----
        For frame-generated signals, extract the payload segment manually
        via ``frame.get_structure_map()`` and create a plain ``Signal``
        with the appropriate ``source_symbols`` before calling this method.
        """
        from . import metrics

        if self.signal_type is not None:
            logger.warning(
                "snr() called on a frame-generated signal. For accurate per-segment "
                "metrics, extract the payload (or pilot) segment manually via "
                "frame.get_structure_map() and create a plain Signal first."
            )
            return

        ref = (
            reference_symbols if reference_symbols is not None else self.source_symbols
        )
        if ref is None:
            raise ValueError(
                "No reference available. Provide reference_symbols or ensure "
                "source_symbols is set."
            )

        if self.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Please call `resolve_symbols()` "
                "first to decimate the signal to symbol rate."
            )
        y = self.resolved_symbols
        r = ref
        trim = num_train_symbols if num_train_symbols is not None else 0
        if trim > 0:
            logger.info(f"Discarding {trim} training symbols for SNR calculation.")
            n = min(y.shape[-1], r.shape[-1])
            y = y[..., trim:n]
            r = r[..., trim:n]

        return metrics.snr(y, r)

    def ber(
        self,
        reference_bits: Optional[ArrayType] = None,
        num_train_symbols: Optional[int] = None,
    ) -> Union[float, ArrayType]:
        """
        Computes the Bit Error Rate (BER).

        Compares decoded bits against the reference bit sequence.

        For MIMO signals, BER is computed independently per stream.

        Parameters
        ----------
        reference_bits : array_like, optional
            The original transmitted bits. If None, falls back to
            ``source_bits`` when available.
        num_train_symbols : int, optional
            Number of symbols to discard from the front before computing the
            metric.  Converted to bits internally via ``bits_per_symbol``.

        Returns
        -------
        float or ndarray
            BER as a ratio in [0, 1]. Scalar for SISO, array for MIMO.

        Raises
        ------
        ValueError
            If required data or modulation metadata is missing, or
            ``resolved_bits`` has not been set (call
            ``demap_symbols_hard()`` first).

        Notes
        -----
        For frame-generated signals, extract the payload segment manually
        via ``frame.get_structure_map()`` and create a plain ``Signal``
        with the appropriate ``source_bits`` before calling this method.
        """
        from . import metrics

        if self.signal_type is not None:
            logger.warning(
                "ber() called on a frame-generated signal. For accurate per-segment "
                "metrics, extract the payload (or pilot) segment manually via "
                "frame.get_structure_map() and create a plain Signal first."
            )
            return

        ref = reference_bits if reference_bits is not None else self.source_bits
        if ref is None:
            raise ValueError(
                "No reference bits available. Provide reference_bits or ensure "
                "source_bits is set."
            )

        if self.resolved_bits is None:
            raise ValueError(
                "No resolved bits available. Please call `demap_symbols_hard()` first."
            )

        y = self.resolved_bits
        r = ref
        trim = num_train_symbols if num_train_symbols is not None else 0
        if trim > 0 and self.mod_order is not None:
            logger.info(f"Discarding {trim} training symbols for BER calculation.")
            bps = self.bits_per_symbol
            bit_trim = trim * bps
            n = min(y.shape[-1], r.shape[-1])
            y = y[..., bit_trim:n]
            r = r[..., bit_trim:n]

        return metrics.ber(y, r)

    def ser(
        self,
        reference_symbols: Optional[ArrayType] = None,
        num_train_symbols: Optional[int] = None,
        *,
        modulation: Optional[str] = None,
        order: Optional[int] = None,
    ) -> Union[float, ArrayType]:
        """
        Computes the Symbol Error Rate (SER) using ML hard decisions.

        Each received symbol is decided to its nearest constellation point.
        The decision is then compared to the corresponding transmitted symbol.

        Parameters
        ----------
        reference_symbols : array_like, optional
            Known transmitted symbols. Falls back to ``source_symbols`` when
            not provided.
        num_train_symbols : int, optional
            Number of symbols to discard from the front before computing the
            metric.
        modulation : str, optional
            Modulation type (``"psk"``, ``"qam"``, ``"ask"``).
            Falls back to ``self.mod_scheme`` when not provided.
        order : int, optional
            Modulation order *M*. Falls back to ``self.mod_order`` when not
            provided.

        Returns
        -------
        float or ndarray
            SER as a ratio in ``[0, 1]``. Scalar for SISO, array for MIMO.

        Raises
        ------
        ValueError
            If ``resolved_symbols`` is not set, no reference is available,
            or modulation metadata is missing.

        Notes
        -----
        For frame-generated signals, extract the payload segment manually
        via ``frame.get_structure_map()`` and create a plain ``Signal``
        with the appropriate ``source_symbols`` before calling this method.
        """
        from . import metrics

        if self.signal_type is not None:
            logger.warning(
                "ser() called on a frame-generated signal. For accurate per-segment "
                "metrics, extract the payload (or pilot) segment manually via "
                "frame.get_structure_map() and create a plain Signal first."
            )
            return

        ref = (
            reference_symbols if reference_symbols is not None else self.source_symbols
        )
        if ref is None:
            raise ValueError(
                "No reference available. Provide reference_symbols or ensure "
                "source_symbols is set."
            )

        if self.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Please call `resolve_symbols()` "
                "first to decimate the signal to symbol rate."
            )

        mod = modulation or self.mod_scheme
        ord_ = order or self.mod_order
        if mod is None or ord_ is None:
            raise ValueError(
                "SER requires modulation and order. Pass them explicitly or "
                "ensure mod_scheme/mod_order are set on the Signal."
            )

        y = self.resolved_symbols
        r = ref
        trim = num_train_symbols if num_train_symbols is not None else 0
        if trim > 0:
            logger.info(f"Discarding {trim} training symbols for SER calculation.")
            n = min(y.shape[-1], r.shape[-1])
            y = y[..., trim:n]
            r = r[..., trim:n]

        return metrics.ser(y, r, mod, ord_)

    def mi(
        self,
        noise_var: float,
        *,
        pmf=None,
    ) -> float:
        """
        Computes Mutual Information (MI) under a Gaussian channel assumption.

        Delegates to :func:`commstools.metrics.mi`.  For PS-QAM signals,
        ``self.ps_pmf`` is passed automatically unless overridden via *pmf*.

        Parameters
        ----------
        noise_var : float
            Complex noise variance :math:`\\sigma^2` of the AWGN channel.
            For unit-power symbols at :math:`E_s/N_0` (dB):
            :math:`\\sigma^2 = 10^{-E_s/N_0 / 10}`.
        pmf : array-like, optional
            Symbol PMF of shape ``(M,)``.  Defaults to ``self.ps_pmf`` when
            set, otherwise uniform (``None``) is assumed.

        Returns
        -------
        float
            MI in bits per channel use (b/cu).

        Raises
        ------
        ValueError
            If ``resolved_symbols`` is not set or modulation metadata is
            missing.
        """
        from . import metrics

        if self.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Call `resolve_symbols()` first."
            )

        mod = self.mod_scheme
        ord_ = self.mod_order
        if mod is None or ord_ is None:
            raise ValueError(
                "MI requires modulation and order. Ensure mod_scheme/mod_order "
                "are set on the Signal."
            )

        effective_pmf = pmf if pmf is not None else self.ps_pmf

        resolved = self.resolved_symbols
        adj_noise_var = noise_var

        if effective_pmf is not None:
            # shape_pulse normalises samples to unit symbol power (E_s = 1).
            # For PS-QAM, source symbols have E_PS = Σ P(sₘ)|sₘ|² < 1 on the
            # normalised QAM grid, so shape_pulse scales them by c = 1/√E_PS.
            # resolved_symbols therefore lives on {c·sₘ}, not {sₘ}.
            # Rescale back to {sₘ} so distances against gray_constellation are
            # correct; noise_var scales by the same factor (c² = 1/E_PS).
            from .mapping import gray_constellation as _gc

            const = _gc(mod, ord_)
            pmf_arr = np.asarray(effective_pmf, dtype=np.float64)
            e_ps = float(np.dot(pmf_arr, np.abs(const) ** 2))
            if e_ps < 1.0 - 1e-6:
                xp = get_array_module(resolved)
                scale = xp.asarray(np.sqrt(e_ps), dtype=resolved.real.dtype)
                resolved = resolved * scale
                adj_noise_var = noise_var * e_ps

        return metrics.mi(resolved, mod, ord_, adj_noise_var, pmf=effective_pmf)

    def gmi(
        self,
        noise_var: float,
        *,
        pmf=None,
        method: str = "maxlog",
    ) -> float:
        """
        Computes Generalised Mutual Information (GMI) from per-bit LLRs.

        LLRs are computed via :func:`commstools.mapping.compute_llr` using
        ``self.ps_pmf`` automatically for PS-QAM signals.  GMI is then
        evaluated against ``self.source_bits``.

        Parameters
        ----------
        noise_var : float
            Complex noise variance :math:`\\sigma^2` of the AWGN channel.
        pmf : array-like, optional
            Symbol PMF of shape ``(M,)``.  Defaults to ``self.ps_pmf`` when
            set, otherwise uniform is assumed.
        method : {"maxlog", "exact"}, default "maxlog"
            LLR computation method passed to :func:`~commstools.mapping.compute_llr`.

        Returns
        -------
        float
            GMI in bits per channel use (b/cu).

        Raises
        ------
        ValueError
            If ``resolved_symbols`` or ``source_bits`` is not set, or
            modulation metadata is missing.
        """
        from . import metrics
        from .mapping import compute_llr
        from .backend import to_device

        if self.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Call `resolve_symbols()` first."
            )
        if self.source_bits is None:
            raise ValueError(
                "GMI requires source_bits. Ensure the Signal was created via a "
                "factory method (e.g. Signal.qam(), Signal.psqam())."
            )

        mod = self.mod_scheme
        ord_ = self.mod_order
        if mod is None or ord_ is None:
            raise ValueError(
                "GMI requires modulation and order. Ensure mod_scheme/mod_order "
                "are set on the Signal."
            )

        effective_pmf = pmf if pmf is not None else self.ps_pmf

        resolved = self.resolved_symbols
        adj_noise_var = noise_var

        if effective_pmf is not None:
            # Same scale correction as Signal.mi(): shape_pulse normalises to
            # unit symbol power, placing resolved_symbols on {c·sₘ} with
            # c = 1/√E_PS.  Rescale to {sₘ} before LLR computation so that
            # distances against gray_constellation are correct.
            from .mapping import gray_constellation as _gc

            const = _gc(mod, ord_)
            pmf_arr = np.asarray(effective_pmf, dtype=np.float64)
            e_ps = float(np.dot(pmf_arr, np.abs(const) ** 2))
            if e_ps < 1.0 - 1e-6:
                xp = get_array_module(resolved)
                scale = xp.asarray(np.sqrt(e_ps), dtype=resolved.real.dtype)
                resolved = resolved * scale
                adj_noise_var = noise_var * e_ps

        llrs = compute_llr(
            resolved,
            mod,
            ord_,
            adj_noise_var,
            method=method,
            pmf=effective_pmf,
            output="numpy",
        )
        src_bits = to_device(self.source_bits, "cpu")
        return metrics.gmi(llrs, src_bits)


class Preamble(BaseModel):
    """
    Structured container for frame synchronization sequences (preambles).

    Preambles are automatically generated based on the specified sequence type
    and length. Manual overrides for bits or symbols are not supported to
    ensure consistency within the processing pipeline.

    Attributes
    ----------
    sequence_type : {"barker", "zc"}, default "barker"
        The synchronization sequence algorithm.
    length : int
        Total length of the preamble in symbols.
        For "barker": length must be from the set {2, 3, 4, 5, 7, 11, 13}.
        For "zc": length must be a prime number.
    root : int, default 1
        ZC root index (only meaningful for ``sequence_type='zc'``).
        Must satisfy ``1 ≤ root < length``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    sequence_type: Literal["barker", "zc"] = "barker"
    length: int
    root: int = Field(
        default=1,
        ge=1,
        description="ZC root index.  Only meaningful for ``sequence_type='zc'``; "
        "ignored for Barker sequences.  Must satisfy ``1 ≤ root < length``; "
        "for prime ``length`` every root in this range yields a valid CAZAC sequence.",
    )
    num_streams: int = Field(
        default=1,
        ge=1,
        description="Number of TX streams.  For ZC preambles each stream gets a "
        "unique root derived via :func:`~commstools.helpers.zc_mimo_root`.  "
        "For Barker the same sequence is broadcast to all streams.",
    )

    # Internal state managed during post-init
    _symbols: Any = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to automate symbol generation and device placement.

        This ensures that standard sequences are generated correctly according
        to the requested sequence properties.

        For ``num_streams == 1`` the internal ``_symbols`` shape is ``(length,)``.
        For ``num_streams > 1`` it becomes ``(num_streams, length)``:
        - ZC: each row uses the unique root returned by
          :func:`~commstools.helpers.zc_mimo_root`.
        - Barker: the same sequence is tiled across all streams.
        """
        from . import sync

        stype = self.sequence_type.lower()

        if stype == "barker":
            # Barker symbols (-1, +1)
            base = sync.barker_sequence(self.length)
        elif stype in ("zc", "zadoff_chu"):
            # ZC complex symbols — use the named 'root' field directly.
            base = sync.zadoff_chu_sequence(self.length, root=self.root)
        else:
            base = None

        if base is not None and self.num_streams > 1:
            if stype in ("zc", "zadoff_chu"):
                rows = [
                    sync.zadoff_chu_sequence(
                        self.length,
                        root=helpers.zc_mimo_root(k, self.root, self.length),
                    )
                    for k in range(self.num_streams)
                ]
                self._symbols = np.stack(rows, axis=0)  # (num_streams, length)
            else:
                self._symbols = np.tile(base[None, :], (self.num_streams, 1))
        else:
            self._symbols = base

        # Move to GPU if available
        if is_cupy_available():
            if self._symbols is not None:
                self._symbols = to_device(self._symbols, "gpu")

            # Ensure consistent internal dtype (complex64)
            self._symbols = self._symbols.astype("complex64")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def symbols(self) -> Any:
        """The IQ symbols of the preamble."""
        return self._symbols

    @property
    def num_symbols(self) -> int:
        """Total number of symbols in the preamble."""
        return self.length

    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------

    def to_signal(
        self,
        sps: int,
        symbol_rate: float,
        pulse_shape: str = "rrc",
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        smoothrect_bt: float = 1.0,
        gaussian_bt: float = 0.3,
        **kwargs: Any,
    ) -> Signal:
        """
        Generates a shaped waveform from the preamble sequence.

        Parameters
        ----------
        sps : int
            Samples per symbol.
        symbol_rate : float
            Symbol rate in Hz.
        pulse_shape : str, default "rrc"
            The pulse shaping type to apply.
        filter_span : int, default 10
            Filter span in symbols.
        rrc_rolloff : float, default 0.35
            Roll-off factor for RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for RC filter.
        smoothrect_bt : float, default 1.0
            Bandwidth-Time product for Smooth Rect filter.
        gaussian_bt : float, default 0.3
            Bandwidth-Time product for Gaussian filter.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Signal
            A `Signal` object with the shaped preamble.
        """
        from .filtering import shape_pulse

        samples = shape_pulse(
            self.symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            smoothrect_bt=smoothrect_bt,
            gaussian_bt=gaussian_bt,
            **kwargs,
        )

        return Signal(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            mod_scheme=None,
            mod_order=None,
            source_symbols=None,
            pulse_shape=pulse_shape,
            signal_type="Preamble",
            **kwargs,
        )


class SingleCarrierFrame(BaseModel):
    """
    Represents a structured single-carrier frame with Preamble, Pilots, Payload, and Guard Interval.

    This class provides a high-level abstraction for constructing frames
    used in digital communication systems ($1/10/100\text{GbE}$, $5\text{G}$, etc.).
    It supports various pilot patterns for channel estimation and guard
    intervals for multi-path mitigation.

    Attributes
    ----------
    payload_len : int, default 1000
        Number of data symbols per spatial stream.
    payload_mod_scheme : str, default "PSK"
        Modulation for payload data (e.g., 'QAM').
    payload_mod_order : int, default 4
        Modulation order for payload (e.g., 16 for 16-QAM).
    payload_seed : int, default 42
        Seed for reproducible payload data generation.
    payload_nu : float, optional
        Maxwell-Boltzmann shaping parameter :math:`\\nu \\geq 0` for a
        probabilistically shaped QAM payload.  Mutually exclusive with
        ``payload_entropy``.  Requires ``payload_mod_scheme`` to contain
        ``"qam"`` (case-insensitive).  :math:`\\nu = 0` → uniform QAM.
    payload_entropy : float, optional
        Target entropy in bits per symbol for a PS-QAM payload.  The
        optimal :math:`\\nu` is solved numerically via
        :func:`~commstools.mapping.optimal_nu`.  Mutually exclusive with
        ``payload_nu``.  Same QAM-only constraint as ``payload_nu``.
    preamble : Preamble, optional
        Structured preamble for synchronization.  For MIMO with ZC sequences,
        each TX stream automatically receives a unique root via
        :func:`~commstools.helpers.zc_mimo_root`.
    pilot_pattern : {"none", "block", "comb"}, default "none"
        "none": No pilots.
        "block": A block of symbols at the start of the frame body.
        "comb": Single pilot symbols interleaved every `pilot_period`.
    pilot_period : int, default 0
        The period of pilot insertion in symbols.
    pilot_block_len : int, default 0
        Length of the pilot block (mode="block") in symbols.
    pilot_seed : int, default 1337
        Seed for pilot symbol generation.
    pilot_mod_scheme : str, default "PSK"
        Modulation for pilots.
    pilot_mod_order : int, default 4
        Modulation order for pilots.
    pilot_gain_db : float, default 0.0
        Pilot boosting in dB relative to the payload power.
    guard_type : {"zero", "cp"}, default "zero"
        "zero": Zero-padding at the end of the frame.
        "cp": Cyclic Prefix prepended to the frame.
    guard_len : int, default 0
        Length of the guard interval in symbols.
    num_streams : int, default 1
        Number of independent spatial streams (MIMO).

    Notes
    -----
    **PS-QAM payload**: set either ``payload_nu`` or ``payload_entropy`` (not
    both) together with a QAM ``payload_mod_scheme``.  The MB distribution is
    solved once and cached; access the resulting PMF via the read-only
    :attr:`payload_ps_pmf` property after the frame has been generated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    payload_len: int = Field(default=1000, gt=0)
    payload_seed: int = 42
    payload_mod_scheme: str = "PSK"
    payload_mod_order: int = Field(default=4, ge=1)
    payload_mod_unipolar: bool = False
    payload_nu: Optional[float] = Field(default=None, ge=0)
    payload_entropy: Optional[float] = Field(default=None, gt=0)

    preamble: Optional[Preamble] = None

    pilot_pattern: Literal["none", "block", "comb"] = "none"
    pilot_period: int = Field(default=0, ge=0)
    pilot_block_len: int = Field(default=0, ge=0)
    pilot_seed: int = 1337
    pilot_mod_scheme: str = "PSK"
    pilot_mod_order: int = Field(default=4, ge=1)
    pilot_mod_unipolar: bool = False
    pilot_gain_db: float = 0.0

    guard_type: Literal["zero", "cp"] = "zero"
    guard_len: int = Field(default=0, ge=0)

    num_streams: int = Field(default=1, ge=1)

    # Internal cache
    _payload_bits: Optional[Any] = PrivateAttr(default=None)
    _payload_symbols: Optional[Any] = PrivateAttr(default=None)
    _payload_ps_pmf: Optional[Any] = PrivateAttr(default=None)
    _pilot_bits: Optional[Any] = PrivateAttr(default=None)
    _pilot_symbols: Optional[Any] = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook.

        Validates that payload_len is evenly divisible by the per-period or
        per-block data count implied by the pilot parameters.  If not, snaps
        payload_len up to the next valid multiple and emits a warning so the
        frame structure always satisfies:
            num_pilot_periods == num_data_periods  (comb)
            num_pilot_blocks  == num_data_blocks   (block)
        """
        import math

        if self.pilot_pattern == "comb" and self.pilot_period > 1:
            data_per_period = self.pilot_period - 1
            if self.payload_len % data_per_period != 0:
                snapped = (
                    math.ceil(self.payload_len / data_per_period) * data_per_period
                )
                logger.warning(
                    f"SingleCarrierFrame (comb): payload_len={self.payload_len} is not "
                    f"divisible by data_per_period={data_per_period} "
                    f"(pilot_period={self.pilot_period}). "
                    f"Snapping payload_len {self.payload_len} → {snapped} so that "
                    f"num_pilot_periods == num_data_periods == {snapped // data_per_period}."
                )
                self.payload_len = snapped

        elif (
            self.pilot_pattern == "block"
            and self.pilot_period > self.pilot_block_len > 0
        ):
            data_per_block = self.pilot_period - self.pilot_block_len
            if self.payload_len % data_per_block != 0:
                snapped = math.ceil(self.payload_len / data_per_block) * data_per_block
                logger.warning(
                    f"SingleCarrierFrame (block): payload_len={self.payload_len} is not "
                    f"divisible by data_per_block={data_per_block} "
                    f"(pilot_period={self.pilot_period}, pilot_block_len={self.pilot_block_len}). "
                    f"Snapping payload_len {self.payload_len} → {snapped} so that "
                    f"num_pilot_blocks == num_data_blocks == {snapped // data_per_block}."
                )
                self.payload_len = snapped

    @model_validator(mode="after")
    def _check_psqam_fields(self) -> "SingleCarrierFrame":
        if self.payload_nu is not None and self.payload_entropy is not None:
            raise ValueError(
                "payload_nu and payload_entropy are mutually exclusive — specify one or neither."
            )
        if self.payload_nu is not None or self.payload_entropy is not None:
            if "qam" not in self.payload_mod_scheme.lower():
                raise ValueError(
                    f"payload_nu / payload_entropy require a QAM payload modulation, "
                    f"got payload_mod_scheme='{self.payload_mod_scheme}'."
                )
        return self

    @model_validator(mode="after")
    def _check_preamble_streams(self) -> "SingleCarrierFrame":
        if self.preamble is not None and self.preamble.num_streams > 1:
            if self.preamble.num_streams != self.num_streams:
                raise ValueError(
                    f"preamble.num_streams={self.preamble.num_streams} does not match "
                    f"frame.num_streams={self.num_streams}"
                )
        return self

    # -------------------------------------------------------------------------
    # Mask Generation and Internal Data Preparation Methods
    # -------------------------------------------------------------------------

    def _generate_pilot_mask(self) -> Tuple[ArrayType, int]:
        """
        Calculates the pilot placement mask and total frame length.

        Returns
        -------
        mask : array_like (bool)
            Boolean mask where True indicates a pilot symbol location.
        body_length : int
            Total number of symbols in the frame body (payload + pilots).
        """
        xp = cp if is_cupy_available() else np

        # No pilots: simple payload mapping
        if self.pilot_pattern == "none":
            body_length = self.payload_len
            mask = xp.zeros(body_length, dtype=bool)
            return mask, body_length

        # Comb pattern: single pilot every N symbols
        if self.pilot_pattern == "comb":
            if self.pilot_period <= 1:
                raise ValueError("pilot_period must be > 1 for 'comb' pattern.")
            data_per_period = self.pilot_period - 1
            num_full_periods = self.payload_len // data_per_period
            remainder = self.payload_len % data_per_period

            total_length = num_full_periods * self.pilot_period + remainder
            # If we have a remainder, we need one more pilot at the start of the partial period
            if remainder > 0:
                total_length += 1

            mask = xp.zeros(total_length, dtype=bool)
            mask[:: self.pilot_period] = True
            return mask, total_length

        # Block pattern: block of pilots followed by block of data
        if self.pilot_pattern == "block":
            if self.pilot_period <= self.pilot_block_len:
                raise ValueError(
                    "pilot_period must be > pilot_block_len for 'block' pattern."
                )
            data_per_block = self.pilot_period - self.pilot_block_len
            num_blocks = int(xp.ceil(self.payload_len / data_per_block))

            # Create a single block pattern [P P ... P D D ... D]
            block_pattern = xp.zeros(self.pilot_period, dtype=bool)
            block_pattern[: self.pilot_block_len] = True

            # Repeat the pattern for all blocks
            mask = xp.tile(block_pattern, num_blocks)

            # Truncation: Find the exact index where the required payload ends
            false_indices = xp.where(~mask)[0]
            last_idx = false_indices[self.payload_len - 1]
            mask = mask[: last_idx + 1]
            return mask, len(mask)

        return xp.zeros(self.payload_len, dtype=bool), self.payload_len

    def _ensure_payload_generated(self) -> None:
        """
        Generates and caches payload bits and symbols via the appropriate Signal factory.

        Dispatches to ``Signal.psqam``, ``Signal.qam``, ``Signal.psk``, or
        ``Signal.pam`` based on ``payload_mod_scheme`` and the PS parameters.
        Using factory methods as the single source of generation logic avoids
        duplicating bit/symbol generation code here.
        """
        if self._payload_bits is not None:
            return

        scheme = self.payload_mod_scheme.lower()
        is_ps = self.payload_nu is not None or self.payload_entropy is not None

        common = dict(
            num_symbols=self.payload_len,
            sps=1,
            symbol_rate=1.0,
            pulse_shape="none",
            num_streams=self.num_streams,
            seed=self.payload_seed,
        )

        if is_ps:
            sig = Signal.psqam(
                order=self.payload_mod_order,
                nu=self.payload_nu,
                entropy=self.payload_entropy,
                **common,
            )
            self._payload_ps_pmf = sig.ps_pmf
        elif "qam" in scheme:
            sig = Signal.qam(
                order=self.payload_mod_order,
                unipolar=self.payload_mod_unipolar,
                **common,
            )
        elif "psk" in scheme:
            sig = Signal.psk(
                order=self.payload_mod_order,
                **common,
            )
        elif "pam" in scheme or "ask" in scheme:
            sig = Signal.pam(
                order=self.payload_mod_order,
                unipolar=self.payload_mod_unipolar,
                **common,
            )
        else:
            sig = Signal.generate(
                modulation=self.payload_mod_scheme,
                order=self.payload_mod_order,
                unipolar=self.payload_mod_unipolar,
                **common,
            )

        self._payload_bits = sig.source_bits
        self._payload_symbols = sig.source_symbols

    def _ensure_pilot_generated(self) -> None:
        """
        Generates and caches pilot bits and symbols via the appropriate Signal factory.

        Pilots are always generated with a uniform distribution — PS on pilots
        would destroy the known-reference property required for channel estimation.
        """
        if self._pilot_bits is not None or self.pilot_pattern == "none":
            return

        xp = cp if is_cupy_available() else np
        mask, _ = self._generate_pilot_mask()
        pilot_count = int(xp.sum(mask))
        if pilot_count == 0:
            return

        scheme = self.pilot_mod_scheme.lower()

        common = dict(
            num_symbols=pilot_count,
            sps=1,
            symbol_rate=1.0,
            pulse_shape="none",
            num_streams=self.num_streams,
            seed=self.pilot_seed,
        )

        if "qam" in scheme:
            sig = Signal.qam(
                order=self.pilot_mod_order,
                unipolar=self.pilot_mod_unipolar,
                **common,
            )
        elif "psk" in scheme:
            sig = Signal.psk(
                order=self.pilot_mod_order,
                **common,
            )
        elif "pam" in scheme or "ask" in scheme:
            sig = Signal.pam(
                order=self.pilot_mod_order,
                unipolar=self.pilot_mod_unipolar,
                **common,
            )
        else:
            sig = Signal.generate(
                modulation=self.pilot_mod_scheme,
                order=self.pilot_mod_order,
                unipolar=self.pilot_mod_unipolar,
                **common,
            )

        self._pilot_bits = sig.source_bits
        self._pilot_symbols = sig.source_symbols

    # -------------------------------------------------------------------------
    # Properties for Accessing Payload and Pilot Data
    # -------------------------------------------------------------------------

    @property
    def payload_bits(self) -> ArrayType:
        """
        Returns the raw payload bits.

        Returns
        -------
        array_like
            Binary bits (0s and 1s).
        """
        self._ensure_payload_generated()
        return self._payload_bits

    @property
    def payload_symbols(self) -> ArrayType:
        """
        Returns the modulated payload symbols.

        Returns
        -------
        array_like
            IQ symbols.
        """
        self._ensure_payload_generated()
        return self._payload_symbols

    @property
    def payload_ps_pmf(self) -> Optional[Any]:
        """
        Returns the Maxwell-Boltzmann PMF used for PS-QAM payload generation.

        ``None`` for uniform (non-PS) payloads.  Pass this to
        :func:`~commstools.metrics.mi` and
        :func:`~commstools.mapping.compute_llr` after frame equalization
        to compute PS-aware capacity and soft-decision metrics.

        Returns
        -------
        np.ndarray or None
            PMF array of shape ``(payload_mod_order,)`` summing to 1, or ``None``.
        """
        self._ensure_payload_generated()
        return self._payload_ps_pmf

    @property
    def pilot_bits(self) -> Optional[ArrayType]:
        """
        Returns the raw pilot bits, if pilots are enabled.

        Returns
        -------
        array_like or None
            Binary bits if `pilot_pattern` is not "none".
        """
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._pilot_bits

    @property
    def pilot_symbols(self) -> Optional[ArrayType]:
        """
        Returns the modulated pilot symbols.

        Returns
        -------
        array_like or None
            IQ symbols if `pilot_pattern` is not "none".
        """
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._pilot_symbols

    @property
    def body_symbols(self) -> ArrayType:
        """
        Returns the interleaved payload and pilot symbols.

        WARNING: Pilot gain is applied if `pilot_gain_db` is not zero,
        so these are not "clean" symbols but scaled relatively.

        Returns
        -------
        array_like
            Determined by `pilot_pattern` and `pilot_period`.
        """
        xp = cp if is_cupy_available() else np
        mask, body_length = self._generate_pilot_mask()

        if self.num_streams > 1:
            # Shape: (Channels, Time)
            body = xp.zeros((self.num_streams, body_length), dtype="complex64")

            if self.pilot_pattern != "none":
                pilot_symbols = self.pilot_symbols
                # Apply pilot boosting/gain (dB to linear)
                if self.pilot_gain_db != 0.0:
                    pilot_symbols = pilot_symbols * (10 ** (self.pilot_gain_db / 20))

                body[:, mask] = pilot_symbols

            body[:, ~mask] = self.payload_symbols
        else:
            body = xp.zeros(body_length, dtype="complex64")
            if self.pilot_pattern != "none":
                pilot_symbols = self.pilot_symbols
                # Apply pilot boosting/gain (dB to linear)
                if self.pilot_gain_db != 0.0:
                    pilot_symbols = pilot_symbols * (10 ** (self.pilot_gain_db / 20))

                body[mask] = pilot_symbols
            body[~mask] = self.payload_symbols

        return body

    # -------------------------------------------------------------------------
    # Frame Structure Mapping
    # -------------------------------------------------------------------------

    def get_structure_map(
        self,
        unit: Literal["symbols", "samples"] = "symbols",
        sps: int = 1,
        include_preamble: bool = True,
    ) -> Dict[str, ArrayType]:
        """
        Generates boolean masks identifying the segments of the frame.

        Parameters
        ----------
        unit : {"symbols", "samples"}, default "symbols"
            The scale of the returned masks.
        sps : int, default 1
            Samples per symbol (required if unit="samples").
        include_preamble : bool, default True
            If True, returns masks for the full frame including preamble and
            guard intervals. If False, returns masks only for the segments
            after the preamble (and after CP removal if guard_type='cp').

        Returns
        -------
        dict
            Dictionary containing boolean masks for:
            - 'preamble' (only if include_preamble=True)
            - 'pilots'
            - 'payload'
            - 'guard' (only if include_preamble=True OR guard_type='zero')
        """
        xp = cp if is_cupy_available() else np
        mask, body_length = self._generate_pilot_mask()

        preamble_len = self.preamble.num_symbols if self.preamble else 0

        if include_preamble:
            total_len = preamble_len + body_length + self.guard_len

            preamble_bool = xp.zeros(total_len, dtype=bool)
            pilot_bool = xp.zeros(total_len, dtype=bool)
            payload_bool = xp.zeros(total_len, dtype=bool)
            guard_bool = xp.zeros(total_len, dtype=bool)

            if self.guard_type == "cp":
                g_slice = slice(0, self.guard_len)
                p_slice = slice(self.guard_len, self.guard_len + preamble_len)
                b_slice = slice(self.guard_len + preamble_len, total_len)
            else:
                p_slice = slice(0, preamble_len)
                b_slice = slice(preamble_len, preamble_len + body_length)
                g_slice = slice(preamble_len + body_length, total_len)

            if preamble_len > 0:
                preamble_bool[p_slice] = True

            pilot_bool[b_slice] = mask
            payload_bool[b_slice] = ~mask

            if self.guard_len > 0:
                guard_bool[g_slice] = True

            res = {
                "preamble": preamble_bool,
                "pilots": pilot_bool,
                "payload": payload_bool,
                "guard": guard_bool,
            }
        else:
            # Preamble removed.
            # If CP, guard is at the start and is typically removed with preamble.
            # If ZERO, guard is at the end and remains part of the signal.
            if self.guard_type == "cp":
                total_len = body_length
                pilot_bool = mask
                payload_bool = ~mask
                res = {
                    "pilots": pilot_bool,
                    "payload": payload_bool,
                }
            else:
                total_len = body_length + self.guard_len
                pilot_bool = xp.zeros(total_len, dtype=bool)
                payload_bool = xp.zeros(total_len, dtype=bool)
                guard_bool = xp.zeros(total_len, dtype=bool)

                b_slice = slice(0, body_length)
                g_slice = slice(body_length, total_len)

                pilot_bool[b_slice] = mask
                payload_bool[b_slice] = ~mask
                guard_bool[g_slice] = True

                res = {
                    "pilots": pilot_bool,
                    "payload": payload_bool,
                    "guard": guard_bool,
                }

        if unit == "samples":
            for k in res:
                res[k] = xp.repeat(res[k], int(sps))

        return res

    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------

    def to_signal(
        self,
        sps: int = 4,
        symbol_rate: float = 1e6,
        pulse_shape: str = "rrc",
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        smoothrect_bt: float = 1.0,
        gaussian_bt: float = 0.3,
        **kwargs: Any,
    ) -> Signal:
        """
        Generates a shaped, oversampled waveform from the frame description.

        This is the primary method for moving from a logical frame to
        physical IQ samples. It handles upsampling, pulse shaping,
        guard interval insertion, and metadata population.

        Parameters
        ----------
        sps : int, default 4
            Samples per symbol (oversampling factor).
        pulse_shape : str, default "rrc"
            Pulse shaping filter type.
        filter_span : int, default 10
            Filter span in symbols.
        rrc_rolloff : float, default 0.35
            Roll-off factor for RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for RC filter.
        smoothrect_bt : float, default 1.0
            Bandwidth-Time product for Smooth Rect filter.
        gaussian_bt : float, default 0.3
            Bandwidth-Time product for Gaussian filter.
        **kwargs : Any
            Additional parameters passed to the shaping filter.

        Returns
        -------
        Signal
            A `Signal` object containing the IQ samples and metadata.

        Notes
        -----
        Each section (preamble and body) is independently I/Q component peak-normalised
        so both occupy the full DAC range regardless of their modulation format.
        After concatenation the full frame is normalised to **unit symbol power
        (Es = 1)**, meaning average sample power = 1/sps.  This matches the
        convention used by ``shape_pulse`` and ``apply_awgn``.
        Pilot/payload power ratios set by `pilot_gain_db` are preserved throughout.
        """
        xp = cp if is_cupy_available() else np
        from .filtering import shape_pulse

        # 1. Shape Body (Payload + Pilots)
        body_symbols = self.body_symbols
        body_samples = shape_pulse(
            symbols=body_symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            smoothrect_bt=smoothrect_bt,
            gaussian_bt=gaussian_bt,
            **kwargs,
        )

        # Normalise body per-channel by max(peak_|I|, peak_|Q|) — a single scale
        # factor that brings the dominant component to 1.0 while preserving the I/Q
        # ratio.  Complex-envelope peak normalisation (used in the DSP chain) divides
        # by max(|sample|) instead, leaving components at ≤ 1/√2 ≈ 0.707 for square
        # QAM/PSK whose envelope peak sits at 45°.  Applied per-section (body and
        # preamble separately) so each segment uses the full DAC range regardless of
        # modulation type or constellation phase geometry.
        max_iq = xp.maximum(
            xp.max(xp.abs(body_samples.real), axis=-1, keepdims=True),
            xp.max(xp.abs(body_samples.imag), axis=-1, keepdims=True),
        )
        max_iq = xp.where(max_iq == 0, xp.ones_like(max_iq), max_iq)
        body_samples = body_samples / max_iq

        # 2. Shape Preamble (if present)
        if self.preamble is not None:
            # Use Preamble's to_signal for shaping to reuse logic,
            # but we only need the samples.
            # CRITICAL: Must use EXACT same shaping parameters as body.
            preamble_signal = self.preamble.to_signal(
                sps=sps,
                symbol_rate=symbol_rate,
                pulse_shape=pulse_shape,
                filter_span=filter_span,
                rrc_rolloff=rrc_rolloff,
                rc_rolloff=rc_rolloff,
                smoothrect_bt=smoothrect_bt,
                gaussian_bt=gaussian_bt,
                **kwargs,
            )
            preamble_samples = xp.asarray(preamble_signal.samples)
            # (L*sps,) for SISO  or  (num_streams, L*sps) for MIMO — shape driven by preamble.num_streams

            # I/Q peak normalisation — axis=-1, keepdims=True works for both 1-D and 2-D
            max_iq_p = xp.maximum(
                xp.max(xp.abs(preamble_samples.real), axis=-1, keepdims=True),
                xp.max(xp.abs(preamble_samples.imag), axis=-1, keepdims=True),
            )
            max_iq_p = xp.where(max_iq_p == 0, xp.ones_like(max_iq_p), max_iq_p)
            preamble_samples = preamble_samples / max_iq_p

            # Concatenate Preamble + Body
            samples = xp.concatenate([preamble_samples, body_samples], axis=-1)
        else:
            samples = body_samples

        # 3. Apply Guard Interval at sample level
        if self.guard_len > 0:
            guard_len_samples = int(self.guard_len * sps)
            if self.guard_type == "zero":
                if self.num_streams > 1:
                    zeros = xp.zeros(
                        (self.num_streams, guard_len_samples), dtype="complex64"
                    )
                else:
                    zeros = xp.zeros(guard_len_samples, dtype="complex64")
                samples = xp.concatenate([samples, zeros], axis=-1)
            elif self.guard_type == "cp":
                cp_slice = samples[..., -guard_len_samples:]
                samples = xp.concatenate([cp_slice, samples], axis=-1)

        # 4. Normalize assembled frame to unit average power.
        # Each section (preamble, body) was independently I/Q peak-normalised so
        # that both use the full DAC range irrespective of their modulation format.
        # After concatenation the sections may differ in average power, so a final
        # global normalization brings the frame to unit symbol power (Es = 1),
        # i.e. average sample power = 1/sps.  Pilot/payload power ratios within
        # the body are preserved because every section's samples are scaled by the
        # same factor.  Guard zeros remain zero after scaling.
        samples = helpers.normalize(samples, "symbol_power", sps=sps, axis=-1)

        return Signal(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            mod_scheme=None,
            mod_order=None,
            mod_unipolar=None,
            mod_rz=None,
            source_bits=None,  # extract via frame.get_structure_map() after equalization
            source_symbols=None,  # samples include full frame (preamble + body);
            # extract payload segment via frame.get_structure_map() explicitly.
            pulse_shape=pulse_shape,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            smoothrect_bt=smoothrect_bt,
            gaussian_bt=gaussian_bt,
            signal_type="Single-Carrier Frame",
            frame=self,
            **kwargs,
        )
