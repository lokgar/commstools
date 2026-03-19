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
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from .equalization import EqualizerResult

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

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

    # Private: cached equalizer result for post-hoc inspection
    _equalizer_result: Any = PrivateAttr(default=None)

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

        # Ensure source_symbols are normalized to unit average power for consistent metrics
        # Ensure source_symbols are normalized to unit average power for consistent metrics
        # For MIMO (multichannel), we normalize per-stream (axis=-1) to ensure each stream
        # independently adheres to E_s=1, facilitating per-stream metric calculation.
        if self.source_symbols is not None:
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
    def equalizer_result(self) -> Optional["EqualizerResult"]:
        """The result of the most recent equalizer call, or ``None``.

        Populated by :meth:`equalize`.  Gives
        access to all equalizer outputs after the fact::

            sig.equalize(method="rde", ...)
            taps   = sig.equalizer_result.weights          # final FIR taps
            error  = sig.equalizer_result.error            # complex error per symbol
            w_hist = sig.equalizer_result.weights_history  # tap trajectory (store_weights=True)

        Returns
        -------
        EqualizerResult or None
            ``None`` when no equalizer has been run yet.
        """
        return self._equalizer_result

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
        title: Optional[str] = "Spectrum",
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
            title=title,
            vmin=vmin,
            vmax=vmax,
            show=False,
            **kwargs,
        )

        if overlay_source and self.source_symbols is not None and result is not None:
            fig, axes = result
            src = to_device(self.source_symbols, "cpu")

            def _scatter_source(ax, symbols):
                ax.scatter(
                    symbols.real,
                    symbols.imag,
                    c="cyan",
                    edgecolors="black",
                    s=10,
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

    def plot_equalizer(
        self,
        smoothing: int = 50,
        ax=None,
        show: bool = False,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots equalizer diagnostics: convergence curve and tap weights.

        Requires ``equalize()`` to have been called first.

        Parameters
        ----------
        smoothing : int, default 50
            Moving-average window for the MSE convergence curve.
        ax : list of 2 Axes, optional
            Pre-existing axes ``[ax_convergence, ax_taps]``.
        show : bool, default False
            If True, calls ``plt.show()`` and returns None.

        Returns
        -------
        (fig, axes) or None
            Figure and axes array when ``show=False``, None otherwise.

        Raises
        ------
        ValueError
            If ``equalize()`` has not been called on this signal.
        """
        if self._equalizer_result is None:
            raise ValueError("No equalizer result available. Call equalize() first.")
        from . import plotting

        return plotting.equalizer_result(
            self._equalizer_result,
            smoothing=smoothing,
            ax=ax,
            show=show,
        )

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

    def correct_timing(
        self,
        preamble=None,
        mode: str = "slice",
        debug_plot: bool = False,
        **kwargs,
    ) -> Tuple["ArrayType", "ArrayType"]:
        """
        Estimates and corrects timing offset in-place.

        Wraps :func:`~commstools.sync.estimate_timing` +
        :func:`~commstools.sync.correct_timing`. If ``preamble``
        is not provided and the signal was generated from a
        :class:`SingleCarrierFrame` (via :meth:`SingleCarrierFrame.to_signal`),
        the preamble is resolved automatically from the attached frame.

        Parameters
        ----------
        preamble : Preamble or array_like, optional
            Known preamble sequence. Forwarded to
            :func:`~commstools.sync.estimate_timing`.
            Falls back to ``self.frame.preamble`` when ``None`` and a
            frame is attached.
        mode : {'slice', 'zero', 'circular'}, default 'slice'
            Boundary handling after coarse correction.

            * ``'slice'``: Discard leading samples; output is shorter.
            * ``'zero'``: Shift left, fill tail with zeros.
            * ``'circular'``: Roll (wrap-around).

        debug_plot : bool, default False
            Forwarded to :func:`~commstools.sync.estimate_timing`.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`~commstools.sync.estimate_timing`.

        Returns
        -------
        tuple of (coarse_offsets, fractional_offsets)
            Per-channel integer and fractional timing estimates (before
            correction is applied).

        Raises
        ------
        ValueError
            If no preamble can be resolved.
        """
        from . import sync

        resolved_preamble = preamble
        if resolved_preamble is None:
            if hasattr(self.frame, "preamble") and self.frame.preamble is not None:
                resolved_preamble = self.frame.preamble
            elif self.signal_type == "Preamble" and self.frame is not None:
                resolved_preamble = self.frame
            else:
                raise ValueError(
                    "A preamble is required for timing estimation. Pass preamble=..., or "
                    "generate the signal via SingleCarrierFrame.to_signal() with a preamble."
                )

        coarse, fractional = sync.estimate_timing(
            self,
            preamble=resolved_preamble,
            debug_plot=debug_plot,
            **kwargs,
        )
        self.samples = sync.correct_timing(self.samples, coarse, fractional, mode=mode)
        return coarse, fractional

    def correct_frequency_offset(
        self,
        method: str = "mth_power",
        pilot_indices: Optional[ArrayType] = None,
        pilot_values: Optional[ArrayType] = None,
        debug_plot: bool = False,
        **kwargs,
    ) -> float:
        """
        Estimates and corrects the carrier frequency offset in-place.

        Estimates the frequency offset using the chosen algorithm, applies the
        correction to ``self.samples`` via complex mixing, and returns the
        estimated offset for inspection or logging.

        Parameters
        ----------
        method : {'mth_power', 'differential', 'pilots'}, default 'mth_power'
            Frequency offset estimation algorithm.

            * ``'mth_power'``: Blind M-th power spectral method. Requires
              ``mod_scheme`` and ``mod_order`` to be set on the Signal.
              Keyword args: ``search_range``, ``nfft``.
            * ``'differential'``: Blind or data-aided differential
              auto-correlation (Kay's estimator).  Requires ``mod_scheme``
              and ``mod_order`` when used blind.  Pass ``ref_signal=`` kwarg
              with the known reference waveform for data-aided estimation.
              Keyword args: ``ref_signal``, ``weighted``.
            * ``'pilots'``: Scattered-pilot phase slope estimator. Requires
              ``pilot_indices`` and ``pilot_values`` arguments.

        pilot_indices : array_like of int, optional
            Sample indices of pilot positions (sorted, unique). Required
            for ``method='pilots'``.
        pilot_values : array_like, optional
            Known transmitted pilot symbols at the corresponding indices.
            Required for ``method='pilots'``.
        debug_plot : bool, default False
            If True, produces diagnostic plots inside the estimator.
        **kwargs
            Additional algorithm-specific parameters forwarded to the
            underlying ``sync.estimate_frequency_offset_*`` function.

        Returns
        -------
        float
            Estimated frequency offset in Hz (positive = signal shifted up).
            The correction ``-offset`` has already been applied to
            ``self.samples``.

        Notes
        -----
        ``digital_frequency_offset`` is **not** updated — that attribute
        tracks intentional digital frequency shifts applied by the TX/RX
        chain, not recovered carrier offsets.

        The preamble-based FOE method has been removed.  To achieve fine
        frequency estimation using a preamble waveform, strip the preamble
        from ``self.samples`` first (after :meth:`correct_timing`), then
        use ``method='differential'`` with ``ref_signal=preamble_samples``
        for data-aided Kay ML estimation::

            # strip preamble (already done by correct_timing)
            preamble_sig = frame.preamble.to_signal(sps=int(sig.sps), ...)
            sig.correct_frequency_offset(
                method="differential",
                ref_signal=preamble_sig.samples,
            )
        """
        from . import sync

        if method == "mth_power":
            if self.mod_scheme is None or self.mod_order is None:
                raise ValueError(
                    "mod_scheme and mod_order must be set on the Signal for "
                    "the 'mth_power' method."
                )
            fo = sync.estimate_frequency_offset_mth_power(
                self.samples,
                fs=self.sampling_rate,
                modulation=self.mod_scheme,
                order=self.mod_order,
                debug_plot=debug_plot,
                **kwargs,
            )
        elif method == "differential":
            fo = sync.estimate_frequency_offset_differential(
                self.samples,
                fs=self.sampling_rate,
                modulation=self.mod_scheme,
                order=self.mod_order,
                debug_plot=debug_plot,
                **kwargs,
            )
        elif method == "pilots":
            if pilot_indices is None or pilot_values is None:
                raise ValueError(
                    "correct_frequency_offset('pilots') requires both "
                    "pilot_indices and pilot_values arguments."
                )
            fo = sync.estimate_frequency_offset_pilots(
                self.samples,
                pilot_indices=pilot_indices,
                pilot_values=pilot_values,
                fs=self.sampling_rate,
                debug_plot=debug_plot,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown FOE method: {method!r}. "
                "Choose from 'mth_power', 'differential', 'pilots'."
            )

        self.samples = sync.correct_frequency_offset(
            self.samples, fo, self.sampling_rate
        )
        logger.info(f"FOE ({method}): estimated {fo:.2f} Hz, correction applied.")
        return fo

    def recover_carrier_phase(
        self,
        method: str = "viterbi_viterbi",
        pilot_indices: Optional[ArrayType] = None,
        pilot_values: Optional[ArrayType] = None,
        interpolation: str = "linear",
        debug_plot: bool = False,
        **kwargs,
    ) -> "ArrayType":
        """
        Estimates and corrects carrier phase in-place.

        Estimates the per-symbol phase using the chosen algorithm, applies the
        correction to ``self.samples``, and returns the phase vector for
        inspection or further processing.

        Parameters
        ----------
        method : {'viterbi_viterbi', 'bps', 'pilots'}, default 'viterbi_viterbi'
            Carrier phase recovery algorithm.

            * ``'viterbi_viterbi'``: Blind M-th power block estimator for
              PSK and QAM. Requires ``mod_scheme`` and ``mod_order`` to be
              set on the Signal.  Keyword args: ``block_size``.
            * ``'bps'``: Blind Phase Search for QAM constellations (Pfau
              et al.).  Requires ``mod_scheme`` and ``mod_order``.
              Keyword args: ``num_test_phases``, ``block_size``.
            * ``'dd_pll'``: Decision-Directed Phase-Locked Loop (DD-PLL).
              Requires ``mod_scheme`` and ``mod_order``.
              Keyword args: ``mu``, ``beta``, ``phase_init``.
            * ``'pilots'``: Pilot-aided phase estimation with interpolation.
              Requires ``pilot_indices`` and ``pilot_values``.

        pilot_indices : array_like of int, optional
            Symbol indices of pilot positions (sorted, unique). Required
            when ``method='pilots'``.
        pilot_values : array_like, optional
            Known transmitted pilot symbols at the corresponding indices.
            Required when ``method='pilots'``.
        interpolation : str, default 'linear'
            Interpolation strategy for the pilot-aided estimator.
            Passed directly to :func:`sync.recover_carrier_phase_pilots`.
        debug_plot : bool, default False
            If True, produces diagnostic plots inside the estimator.
        **kwargs
            Additional algorithm-specific parameters forwarded to the
            underlying ``sync.recover_carrier_phase_*`` function.

        Returns
        -------
        array_like
            Per-symbol phase estimate in radians, shape ``(N,)`` or
            ``(C, N)``.  The correction has already been applied to
            ``self.samples``.

        Notes
        -----
        Assumes ``self.samples`` is already at 1 SPS (i.e. after
        :meth:`decimate_to_symbol_rate` or :meth:`equalize` with ``sps``
        internally decimated).  Applying CPR to an oversampled signal will
        give poor results.
        """
        from . import sync

        if method == "viterbi_viterbi":
            if self.mod_scheme is None or self.mod_order is None:
                raise ValueError(
                    "mod_scheme and mod_order must be set on the Signal for "
                    "the 'viterbi_viterbi' method."
                )
            phase = sync.recover_carrier_phase_viterbi_viterbi(
                self.samples,
                modulation=self.mod_scheme,
                order=self.mod_order,
                debug_plot=debug_plot,
                **kwargs,
            )
        elif method == "bps":
            if self.mod_scheme is None or self.mod_order is None:
                raise ValueError(
                    "mod_scheme and mod_order must be set on the Signal for "
                    "the 'bps' method."
                )
            phase = sync.recover_carrier_phase_bps(
                self.samples,
                modulation=self.mod_scheme,
                order=self.mod_order,
                debug_plot=debug_plot,
                **kwargs,
            )
        elif method == "pilots":
            if pilot_indices is None or pilot_values is None:
                raise ValueError(
                    "recover_carrier_phase('pilots') requires both "
                    "pilot_indices and pilot_values arguments."
                )
            phase = sync.recover_carrier_phase_pilots(
                self.samples,
                pilot_indices=pilot_indices,
                pilot_values=pilot_values,
                interpolation=interpolation,
                debug_plot=debug_plot,
                **kwargs,
            )
        elif method == "dd_pll":
            phase = sync.recover_carrier_phase_decision_directed(
                self.samples,
                debug_plot=debug_plot,
                modulation=self.mod_scheme,
                order=self.mod_order,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown CPR method: {method!r}. "
                "Choose from 'viterbi_viterbi', 'bps', 'dd_pll', 'pilots'."
            )

        self.samples = sync.correct_carrier_phase(self.samples, phase)
        logger.info(f"CPR ({method}): phase correction applied.")
        return phase

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

    def equalize(
        self,
        method: str = "lms",
        # ── common to all adaptive equalization ─────────────────────────────
        num_taps: int = 21,
        step_size: float = 0.01,
        store_weights: bool = False,
        center_tap: Optional[int] = None,
        device: Optional[str] = "cpu",
        backend: Optional[str] = "numba",
        # ── training control ───────────────────────────────────────────────
        training_symbols: Optional[ArrayType] = None,
        num_train_symbols: Optional[int] = None,
        # ── RLS-specific ───────────────────────────────────────────────────
        forgetting_factor: float = 0.99,
        delta: float = 0.01,
        leakage: float = 0.0,
        # ── ZF/MMSE-specific ───────────────────────────────────────────────
        channel_estimate: Optional[ArrayType] = None,
        noise_variance: float = 0.0,
        # ── pilot-aided hybrid (cma / rde only) ────────────────────────────
        pilot_ref: Optional[ArrayType] = None,
        pilot_mask: Optional[ArrayType] = None,
        # ── warm-start ────────────────────────────────────────────────────
        w_init: Optional[ArrayType] = None,
        debug_plot: bool = False,
    ) -> "Signal":
        """
        Apply adaptive or block equalization to the signal samples in-place.

        Signal metadata (``sps``, ``mod_scheme``, ``mod_order``, ``mod_unipolar``,
        ``source_symbols``) is read automatically — you only need to supply the
        algorithm-specific tuning parameters. After this call:

        - ``signal.samples`` contains the equalized complex symbols at symbol
          rate (1 SPS for adaptive methods).
        - ``signal.equalizer_result`` holds the full
          :class:`~commstools.equalization.EqualizerResult` (adaptive methods only).

        Algorithm overview
        ------------------

        +-----------+-------------------------+------------------+-----------------------------------+
        | method    | Training required       | Phase ambiguity  | Convergence                       |
        +===========+=========================+==================+===================================+
        | ``"lms"`` | data-aided → DD         | resolved         | moderate (NLMS)                   |
        +-----------+-------------------------+------------------+-----------------------------------+
        | ``"rls"`` | data-aided → DD         | resolved         | fast (optimal LS)                 |
        +-----------+-------------------------+------------------+-----------------------------------+
        | ``"cma"`` | blind (no training)     | phase-ambiguous  | slow; best for PSK/low-order QAM  |
        +-----------+-------------------------+------------------+-----------------------------------+
        | ``"rde"`` | blind (no training)     | phase-ambiguous  | slow; best for high-order QAM     |
        +-----------+-------------------------+------------------+-----------------------------------+
        | ``"zf"``  | channel estimate needed | resolved         | instant (block)                   |
        +-----------+-------------------------+------------------+-----------------------------------+

        For ``"lms"`` and ``"rls"``, training symbols are taken from
        ``signal.source_symbols`` by default. Pass ``training_symbols`` to
        override with an external sequence (e.g. a received preamble from
        a separate detector). After exhausting the training sequence, the
        equalizer transitions to decision-directed (DD) mode using the
        modulation constellation stored in the signal's metadata.

        CMA ignores ``training_symbols`` entirely — it is a fully blind
        algorithm. It recovers the channel up to a **phase ambiguity**, so a
        carrier-phase recovery step (Viterbi-Viterbi, pilot-aided, etc.) is
        typically needed afterwards.

        Parameters
        ----------
        method : {"lms", "rls", "cma", "rde", "zf"}, default "lms"
            Equalization algorithm. See the table above for a summary.
        num_taps : int, default 21
            Number of FIR taps in the equalizer filter (per polyphase arm for
            fractionally-spaced equalization). Use at least ``4 * sps`` taps.
            More taps allow deeper ISI compensation but slow convergence.
            *Applies to: lms, rls, cma.*
        step_size : float, default 0.01
            NLMS step size (mu) for LMS, or fixed gradient step for CMA.

            - **LMS**: normalized step in ``(0, 2)``; see
              :func:`~commstools.equalization.lms` for the stability derivation.
              Larger values converge faster but increase steady-state
              misadjustment. Typical: 0.01-0.1.
            - **CMA / RDE**: fixed step on the non-convex Godard surface; must
              be kept small (1e-5 to 1e-3) for stability. Unlike LMS, there is
              no input-power normalization.

            *Applies to: lms, cma, rde.*
        store_weights : bool, default False
            If ``True``, the full tap-weight trajectory is stored in
            ``signal.equalizer_result.weights_history``, enabling
            convergence analysis and debugging. Incurs extra memory cost
            proportional to ``num_symbols x num_taps``.
            *Applies to: lms, rls, cma, rde.*
        center_tap : int, optional
            Index of the center (decision-delay) tap. Defaults to
            ``num_taps // 2``. Adjust when channel delay is asymmetric
            (e.g., heavy pre-cursor ISI). *Applies to: lms, rls, cma, rde.*
        device : {"cpu", "gpu"}, optional
            Force JAX computation to run on the specified device regardless
            of where the input samples reside. Default is "cpu".
            Ignored when ``backend="numba"``.
            *Applies to: lms, rls, cma.*
        backend : {"numba", "jax"}, default "numba"
            Computational backend for the equalization algorithm.
        training_symbols : array_like, optional
            External training sequence to use instead of
            ``signal.source_symbols``. Shape ``(N_train,)`` for SISO or
            ``(C, N_train)`` for MIMO. Useful when the signal contains only
            a received payload and training symbols come from a preamble
            detector or a known pilot frame.
            *Applies to: lms, rls. Ignored by cma and zf.*
        num_train_symbols : int, optional
            Caps the number of data-aided symbols. After this many symbols,
            the equalizer switches to blind decision-directed mode even if
            more training symbols are available. When ``None``, the entire
            training array (or ``source_symbols``) is used.
            *Applies to: lms, rls.*
        forgetting_factor : float, default 0.99
            RLS exponential forgetting factor (λ). Range ``(0, 1]``.
            Values near 1.0 give long-term memory (suitable for slowly
            time-varying channels); values near 0.9 track fast variations
            at the cost of higher estimation noise. *Applies to: rls.*
        delta : float, default 0.01
            RLS initialisation regularisation. The inverse correlation matrix
            is initialised as ``P = (1/delta) x I``. Larger values impose a
            stronger prior toward zero weights and slow initial convergence;
            smaller values allow faster start-up at the risk of early
            numerical instability on ill-conditioned channels.
            *Applies to: rls.*
        leakage : float, default 0.0
            Diagonal loading coefficient for Leaky RLS. At every step,
            ``leakage x I`` is added to the P matrix after the rank-1
            downdate, flooring its minimum eigenvalue and preventing
            null-subspace modes (noise-only bands in T/2-spaced signals)
            from being amplified into the equalizer weights. Use ``0.0``
            for standard RLS; for fractionally-spaced data start with
            ``1e-4`` and tune upwards until EVM stops degrading.
            *Applies to: rls.*
        channel_estimate : array_like
            Known channel impulse response used to construct the frequency-
            domain equalizer. Shape ``(L,)`` for SISO or ``(C, C, L)`` for
            MIMO (``H[rx, tx, delay]``). **Required** when ``method="zf"``;
            ignored by all adaptive methods. *Applies to: zf.*
        noise_variance : float, default 0.0
            Noise power σ² for MMSE regularisation in the ZF equalizer.
            ``0.0`` gives a pure Zero-Forcing inversion (may amplify noise
            severely at spectral notches); any positive value yields an MMSE
            solution that trades residual ISI for reduced noise enhancement.
            Rule of thumb: set to the estimated noise variance of the
            received signal (e.g. ``10 ** (-SNR_dB / 10)``).
            *Applies to: zf.*
        pilot_ref : array_like, optional
            Dense pilot reference array of shape ``(C, N_sym)`` or
            ``(N_sym,)`` for SISO, built by
            :func:`~commstools.equalization.build_pilot_ref`.  When supplied
            together with ``pilot_mask``, the equalizer switches into
            **hybrid pilot-aided mode**: DA-LMS error at pilot positions,
            blind CMA/RDE error elsewhere.  *Applies to: cma, rde.*
        pilot_mask : array_like, optional
            Per-symbol uint8 mask of shape ``(N_sym,)``: ``1`` at pilot
            positions, ``0`` elsewhere.  Built by
            :func:`~commstools.equalization.build_pilot_ref`.
            *Applies to: cma, rde.*
        w_init : array_like, optional
            Initial tap weights for warm-starting.  Shape
            ``(C, C, num_taps)`` complex64, or the SISO short-hand
            ``(num_taps,)`` as returned by ``EqualizerResult.weights``.
            Useful for handing off pre-converged weights from a preamble
            LMS stage to the body CMA/RDE pass.
            *Applies to: lms, rls, cma, rde.*

        Returns
        -------
        Signal
            ``self``, enabling method chaining. Key side-effects:

            - ``signal.samples`` — equalized symbols (1 SPS after adaptive
              methods; unchanged rate for ZF).
            - ``signal.equalizer_result`` — full
              :class:`~commstools.equalization.EqualizerResult` including
              ``y_hat``, ``weights``, ``error``, ``num_train_symbols``,
              and optionally ``weights_history``. *Not set for ZF.*

        Raises
        ------
        ValueError
            If ``method`` is ``"lms"``, ``"rls"``, ``"cma"``, or ``"rde"`` and
            the signal is not at 2 SPS. Resample to 2 SPS first.
        ValueError
            If ``method="zf"`` and ``channel_estimate`` is ``None``.
        ValueError
            If ``method`` is not one of the four supported algorithms.

        Examples
        --------
        Data-aided LMS with decision-directed tail, then BER:

        >>> sig = Signal.qam(symbol_rate=28e9, num_symbols=10_000, order=16,
        ...                  pulse_shape="rrc", sps=2)
        >>> sig.equalize(method="lms", num_taps=31, step_size=0.05,
        ...              num_train_symbols=500)
        >>> ber = sig.ber(discard_training=True)

        Blind CMA for QPSK polarization demux, followed by Viterbi-Viterbi
        carrier recovery (not shown):

        >>> rx_mimo = ...  # (2, N) received samples at 2 SPS
        >>> sig = Signal(samples=rx_mimo, sampling_rate=2*Rb, symbol_rate=Rb,
        ...              mod_scheme="psk", mod_order=4)
        >>> sig.equalize(method="cma", num_taps=21, step_size=5e-4)

        ZF channel inversion with MMSE regularisation:

        >>> h = estimate_channel(rx, pilot_syms)  # (L,) impulse response
        >>> sig.equalize(method="zf", channel_estimate=h,
        ...              noise_variance=1e-2)
        """
        from . import equalization

        sps = int(self.sps)

        if method not in ["zf", "rls"] and sps != 2:
            raise ValueError(
                f"Signal is at {sps} SPS. Adaptive equalization require 2 SPS "
                f"(T/2-spaced input) — resample first."
            )

        train = (
            training_symbols if training_symbols is not None else self.source_symbols
        )

        if train is None and method in ("lms", "rls"):
            logger.warning(
                f"{method.upper()}: no training_symbols provided and signal has no "
                "source_symbols — running in pure decision-directed (DD) mode from "
                "symbol 0. Pass training_symbols=... to equalize() for data-aided "
                "convergence. For frame signals, extract payload symbols via "
                "frame.get_structure_map() first."
            )

        if method == "lms":
            result = equalization.lms(
                self.samples,
                training_symbols=train,
                sps=sps,
                num_taps=num_taps,
                step_size=step_size,
                store_weights=store_weights,
                center_tap=center_tap,
                device=device,
                num_train_symbols=num_train_symbols,
                modulation=self.mod_scheme,
                order=self.mod_order,
                unipolar=self.mod_unipolar,
                backend=backend,
                w_init=w_init,
                debug_plot=debug_plot,
            )
        elif method == "rls":
            result = equalization.rls(
                self.samples,
                training_symbols=train,
                sps=sps,
                num_taps=num_taps,
                forgetting_factor=forgetting_factor,
                delta=delta,
                leakage=leakage,
                store_weights=store_weights,
                center_tap=center_tap,
                device=device,
                num_train_symbols=num_train_symbols,
                modulation=self.mod_scheme,
                order=self.mod_order,
                unipolar=self.mod_unipolar,
                backend=backend,
                w_init=w_init,
                debug_plot=debug_plot,
            )
        elif method == "cma":
            result = equalization.cma(
                self.samples,
                sps=sps,
                num_taps=num_taps,
                step_size=step_size,
                store_weights=store_weights,
                center_tap=center_tap,
                device=device,
                modulation=self.mod_scheme,
                order=self.mod_order,
                unipolar=self.mod_unipolar,
                backend=backend,
                w_init=w_init,
                pilot_ref=pilot_ref,
                pilot_mask=pilot_mask,
                debug_plot=debug_plot,
            )
        elif method == "rde":
            result = equalization.rde(
                self.samples,
                sps=sps,
                num_taps=num_taps,
                step_size=step_size,
                store_weights=store_weights,
                center_tap=center_tap,
                device=device,
                modulation=self.mod_scheme,
                order=self.mod_order,
                unipolar=self.mod_unipolar,
                backend=backend,
                w_init=w_init,
                pilot_ref=pilot_ref,
                pilot_mask=pilot_mask,
                debug_plot=debug_plot,
            )
        elif method == "zf":
            if channel_estimate is None:
                raise ValueError(
                    "method='zf' requires channel_estimate to be provided."
                )
            self.samples = equalization.zf_equalizer(
                self.samples,
                channel_estimate=channel_estimate,
                noise_variance=noise_variance,
                debug_plot=debug_plot,
            )
            return self
        else:
            raise ValueError(
                f"Unknown equalization method '{method}'. "
                f"Choose from: 'lms', 'rls', 'cma', 'rde', 'zf'."
            )

        self.samples = result.y_hat
        self._equalizer_result = result
        # RLS truncates the last num_taps//2 symbols from y_hat to remove the
        # terminal artifact zone (windows that overlap right zero-padding).
        # Trim the reference arrays now so Signal state is self-consistent:
        # source_symbols and source_bits will match y_hat length from this point on.
        if method == "rls":
            tail_trim = num_taps // 2
            if tail_trim > 0:
                if self.source_symbols is not None:
                    self.source_symbols = self.source_symbols[..., :-tail_trim]
                if self.source_bits is not None and self.mod_order is not None:
                    bit_trim = tail_trim * self.bits_per_symbol
                    self.source_bits = self.source_bits[..., :-bit_trim]
        self.sampling_rate = self.symbol_rate
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
        ``Es = mean_sample_power × sps = 1``, so Es/N0 calibration is exact for
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
        ``Es = mean_sample_power × sps = 1``, so Es/N0 calibration is exact for
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

    # -------------------------------------------------------------------------
    # Metrics Methods
    # -------------------------------------------------------------------------

    def evm(
        self,
        reference_symbols: Optional[ArrayType] = None,
        discard_training: bool = True,
        num_train_symbols: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Computes the Error Vector Magnitude (EVM).

        EVM is a measure of the difference between the received symbols and
        the ideal reference symbols.

        Parameters
        ----------
        reference_symbols : array_like, optional
            Known transmitted symbols. If None, falls back to
            ``source_symbols`` when available.
        discard_training : bool, default True
            If True, discards the initial DA-training symbols before
            computing the metric.
        num_train_symbols : int, optional
            Explicit number of symbols to discard from the front before
            computing the metric.  When provided, overrides both
            ``discard_training`` and the internally recorded training count.

        Returns
        -------
        evm_percent : float
            EVM expressed as a percentage of the average symbol power.
        evm_db : float
            EVM expressed in decibels (dB).

        Raises
        -------
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
                "evm() called on a frame-generated signal. For accurate per-segment "
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
        trim = (
            num_train_symbols
            if num_train_symbols is not None
            else (
                getattr(self._equalizer_result, "num_train_symbols", 0)
                if discard_training
                else 0
            )
        )
        if trim > 0:
            logger.info(f"Discarding {trim} training symbols for EVM calculation.")
            n = min(y.shape[-1], r.shape[-1])
            y = y[..., trim:n]
            r = r[..., trim:n]

        return metrics.evm(y, r)

    def snr(
        self,
        reference_symbols: Optional[ArrayType] = None,
        discard_training: bool = True,
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
        discard_training : bool, default True
            If True, discards the initial DA-training symbols before
            computing the metric.
        num_train_symbols : int, optional
            Explicit number of symbols to discard from the front before
            computing the metric.  When provided, overrides both
            ``discard_training`` and the internally recorded training count.

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
        trim = (
            num_train_symbols
            if num_train_symbols is not None
            else (
                getattr(self._equalizer_result, "num_train_symbols", 0)
                if discard_training
                else 0
            )
        )
        if trim > 0:
            logger.info(f"Discarding {trim} training symbols for SNR calculation.")
            n = min(y.shape[-1], r.shape[-1])
            y = y[..., trim:n]
            r = r[..., trim:n]

        return metrics.snr(y, r)

    def ber(
        self,
        reference_bits: Optional[ArrayType] = None,
        discard_training: bool = True,
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
        discard_training : bool, default True
            If True, discards the initial DA-training symbols (converted to
            bits) before computing the metric.
        num_train_symbols : int, optional
            Explicit number of symbols to discard from the front before
            computing the metric.  Converted to bits internally via
            ``bits_per_symbol``.  When provided, overrides both
            ``discard_training`` and the internally recorded training count.

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
        trim = (
            num_train_symbols
            if num_train_symbols is not None
            else (
                getattr(self._equalizer_result, "num_train_symbols", 0)
                if discard_training
                else 0
            )
        )
        if trim > 0 and self.mod_order is not None:
            logger.info(f"Discarding {trim} training symbols for BER calculation.")
            bps = self.bits_per_symbol
            bit_trim = trim * bps
            n = min(y.shape[-1], r.shape[-1])
            y = y[..., bit_trim:n]
            r = r[..., bit_trim:n]

        return metrics.ber(y, r)


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
        """
        from . import sync

        stype = self.sequence_type.lower()

        if stype == "barker":
            # Barker symbols (-1, +1)
            self._symbols = sync.barker_sequence(self.length)

        elif stype in ("zc", "zadoff_chu"):
            # ZC complex symbols — use the named 'root' field directly.
            self._symbols = sync.zadoff_chu_sequence(self.length, root=self.root)

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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    payload_len: int = Field(default=1000, gt=0)
    payload_seed: int = 42
    payload_mod_scheme: str = "PSK"
    payload_mod_order: int = Field(default=4, ge=1)
    payload_mod_unipolar: bool = False

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
        Generates and caches payload bits and symbols.

        This internal method ensures that the bit-first representations are available
        and correctly mapped to the target modulation scheme.
        """
        if self._payload_bits is not None:
            return

        from . import mapping

        k = int(np.log2(self.payload_mod_order))
        total_symbols = self.payload_len * self.num_streams
        bits = helpers.random_bits(total_symbols * k, seed=self.payload_seed)
        symbols = mapping.map_bits(
            bits=bits,
            modulation=self.payload_mod_scheme,
            order=self.payload_mod_order,
            unipolar=self.payload_mod_unipolar,
        )

        if self.num_streams > 1:
            bits = bits.reshape(self.num_streams, self.payload_len * k)
            symbols = symbols.reshape(self.num_streams, self.payload_len)

        self._payload_bits = bits
        self._payload_symbols = symbols

    def _ensure_pilot_generated(self) -> None:
        """
        Generates and caches pilot bits and symbols.

        This internal method ensuring pilots are generated with the correct
        seed and modulation before frame assembly.
        """
        if self._pilot_bits is not None or self.pilot_pattern == "none":
            return

        from . import mapping

        xp = cp if is_cupy_available() else np
        mask, _ = self._generate_pilot_mask()
        pilot_count = int(xp.sum(mask))
        if pilot_count == 0:
            return

        k = int(np.log2(self.pilot_mod_order))
        total_pilots = pilot_count * self.num_streams
        bits = helpers.random_bits(total_pilots * k, seed=self.pilot_seed)
        symbols = mapping.map_bits(
            bits=bits,
            modulation=self.pilot_mod_scheme,
            order=self.pilot_mod_order,
            unipolar=self.pilot_mod_unipolar,
        )

        if self.num_streams > 1:
            bits = bits.reshape(self.num_streams, pilot_count * k)
            symbols = symbols.reshape(self.num_streams, pilot_count)

        self._pilot_bits = bits
        self._pilot_symbols = symbols

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
            preamble_samples = preamble_signal.samples

            # Same single-scale normalisation for preamble (1D here; axis=-1 == global).
            max_iq_p = xp.maximum(
                xp.max(xp.abs(preamble_samples.real), axis=-1, keepdims=True),
                xp.max(xp.abs(preamble_samples.imag), axis=-1, keepdims=True),
            )
            max_iq_p = xp.where(max_iq_p == 0, xp.ones_like(max_iq_p), max_iq_p)
            preamble_samples = preamble_samples / max_iq_p

            # Handle MIMO preamble structure.
            # ZC preambles: generate a unique root per TX stream for near-orthogonal
            # simultaneous transmission — all streams transmit at the same time and
            # each RX channel sees a mixture that can be timed independently.
            # Non-ZC preambles: broadcast the single shaped waveform to all streams.
            if self.num_streams > 1:
                if self.preamble.sequence_type == "zc":
                    stream_waveforms = []
                    for k in range(self.num_streams):
                        root_k = helpers.zc_mimo_root(
                            k, self.preamble.root, self.preamble.length
                        )
                        pk_sig = Preamble(
                            sequence_type="zc",
                            length=self.preamble.length,
                            root=root_k,
                        ).to_signal(
                            sps=sps,
                            symbol_rate=symbol_rate,
                            pulse_shape=pulse_shape,
                            filter_span=filter_span,
                            rrc_rolloff=rrc_rolloff,
                            rc_rolloff=rc_rolloff,
                            smoothrect_bt=smoothrect_bt,
                            gaussian_bt=gaussian_bt,
                        )
                        pk_samples = pk_sig.samples
                        max_iq_k = xp.maximum(
                            xp.max(xp.abs(pk_samples.real), keepdims=True),
                            xp.max(xp.abs(pk_samples.imag), keepdims=True),
                        )
                        max_iq_k = xp.where(
                            max_iq_k == 0, xp.ones_like(max_iq_k), max_iq_k
                        )
                        stream_waveforms.append(pk_samples / max_iq_k)
                    preamble_samples = xp.stack(stream_waveforms, axis=0)  # (C, L*sps)
                else:
                    preamble_samples = xp.tile(
                        preamble_samples[None, :], (self.num_streams, 1)
                    )

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
