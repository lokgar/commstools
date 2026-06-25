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
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


from .. import helpers
from ..backend import (
    ArrayType,
    from_jax,
    get_array_module,
    get_scipy_module,
    is_cupy_available,
    to_device,
    to_jax,
)
from ..logger import logger


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
        Populated by ``Signal.generate`` and the factory methods.
        For frame-generated signals this is ``None``; extract the payload
        segment via ``frame.get_structure_map()`` and construct a plain
        ``Signal`` with the relevant ``source_bits`` for per-segment metrics.
    source_symbols : array_like, optional
        The mapped constellation symbols before pulse shaping (full wire
        order). Same scoping note as ``source_bits``.
    ps_pmf : array_like of float, optional
        Maxwell-Boltzmann PMF of shape ``(M,)`` for PS-QAM signals.
        Set automatically by ``Signal.psqam``.  When present, the
        normalization of ``source_symbols`` is skipped (PS symbols have
        intentionally lower average energy than uniform QAM), and
        ``mi``, ``gmi``, and ``plot_constellation`` use the
        non-uniform prior automatically.  ``None`` for all other modulations.
    ps_nu : float, optional
        Maxwell-Boltzmann shaping parameter ν ≥ 0.  Set automatically
        alongside ``ps_pmf`` by ``Signal.psqam`` and
        ``SingleCarrierFrame.to_signal``.  ν = 0 is uniform QAM (never
        stored; ``ps_nu`` is ``None`` for non-PS signals).  When the
        signal was specified via ``entropy``, ν is the numerically solved
        value returned by ``mapping.optimal_nu``.
    pulse_shape : str, optional
        Name of the pulse shaping filter (e.g., ``'rrc'``, ``'rect'``,
        ``'gaussian'``).
    filter_span : int
        Span of the pulse-shaping filter in symbols.
    rrc_rolloff : float
        Roll-off factor for the Root-Raised Cosine (RRC) filter.
    rc_rolloff : float
        Roll-off factor for the Raised Cosine (RC) filter.
    duty_cycle : float
        Pulse width in symbol periods. Meaning depends on pulse shape:
        ``rect``/``smoothrect`` — on-time fraction (incl. ramps);
        ``gaussian`` — FWHM. For NRZ signals this is always 1.0 internally;
        only meaningful when ``mod_rz=True``. Stored so
        ``generate_shaping_taps()`` can reconstruct the correct taps.
    rise_time : float
        Edge transition duration in symbol periods for ``rect`` and
        ``smoothrect``. For ``rect``: linear ramp duration (flat top =
        ``duty_cycle - 2 * rise_time``). For ``smoothrect``: 10%-90%
        erf-edge duration. Ignored for all other pulse shapes.
    spectral_domain : {"BASEBAND", "PASSBAND", "INTERMEDIATE"}
        The signal's current placement in the frequency spectrum.
    physical_domain : {"DIG", "RF", "OPT"}
        The physical transmission domain: ``'DIG'`` (Digital), ``'RF'``
        (Radio), ``'OPT'`` (Optical).
    center_frequency : float
        The carrier or center frequency in Hz.
    digital_frequency_offset : float
        Cumulative digital frequency shift applied to the signal in Hz.
    pilot_tone_frequency : numpy.ndarray
        Per-channel pilot-tone frequencies in Hz, as a 1-D ``float64`` array
        (one entry per channel; length 1 for a SISO / shared tone).  Any scalar
        or sequence assigned is coerced to this array form, so the field is
        handled uniformly like the other array fields.  Set by
        ``add_pilot_tone``; distinct per-channel tones enable e.g. tone-based
        polarization demultiplexing.
    pilot_tone_power_ratio_db : numpy.ndarray
        Per-channel pilot-to-signal power ratio (PSR) in dB of the added
        tone(s), in the same 1-D ``float64`` array form as
        ``pilot_tone_frequency``.  Set by ``add_pilot_tone``; travels with the
        signal through save/load.
    signal_type : {"Single-Carrier Frame", "OFDM Frame", "Preamble"}, optional
        Human-readable label for the signal structure. Informational only.
    frame : Frame, optional
        The frame that generated the signal.
    resolved_symbols : array_like, optional
        Symbols at 1 SPS, normalised to unit average power.
        Populated by ``resolve_symbols()``.  Call only on plain signals
        (non-frame); frame signals contain mixed preamble/pilot/payload that
        may have different modulations or gains — resolve after splitting.
    resolved_bits : array_like, optional
        Hard-decision bits demapped from ``resolved_symbols``.
        Populated by ``demap_symbols_hard()``.

    Notes
    -----
    **Frame-generated signals**: ``SingleCarrierFrame.to_signal`` sets
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

    mod_scheme: str | None = None
    mod_order: int | None = None
    mod_unipolar: bool | None = None
    mod_rz: bool | None = None

    source_bits: Any | None = None
    source_symbols: Any | None = None
    ps_pmf: Any | None = None  # (M,) PMF over constellation; set only for PS-QAM
    ps_nu: float | None = None  # MB shaping parameter ν; set only for PS-QAM

    pulse_shape: str | None = None
    filter_span: int = Field(default=10, ge=1)
    rrc_rolloff: float = Field(default=0.35, ge=0, le=1)
    rc_rolloff: float = Field(default=0.35, ge=0, le=1)
    duty_cycle: float = Field(default=1.0, gt=0, le=1)
    rise_time: float = Field(default=0.0, ge=0)

    spectral_domain: Literal["BASEBAND", "PASSBAND", "INTERMEDIATE"] = "BASEBAND"
    physical_domain: Literal["DIG", "RF", "OPT"] = "DIG"

    center_frequency: float = Field(default=0, ge=0)
    digital_frequency_offset: float | None = None
    pilot_tone_frequency: Any | None = None
    pilot_tone_power_ratio_db: Any | None = None

    # Human-readable label for the signal structure
    signal_type: Literal["Single-Carrier Frame", "OFDM Frame", "Preamble"] | None = None

    # Back-reference to the SingleCarrierFrame that generated this signal (set by
    # SingleCarrierFrame.to_signal()). Enables frame-aware convenience methods
    # (correct_timing, frame-aware equalizers) without requiring the caller to re-supply
    # the frame object.  Excluded from serialisation (numpy arrays inside frame
    # duplicate samples data and are not JSON-serialisable).
    frame: Any | None = Field(default=None, exclude=True, repr=False)

    # Resolved data from processing (1 SPS, normalized — populated by resolve_symbols())
    resolved_symbols: Any | None = Field(default=None, repr=False)
    resolved_bits: Any | None = Field(default=None, repr=False)

    # -------------------------------------------------------------------------
    # Validators and Post-Initialization Hooks
    # -------------------------------------------------------------------------

    @field_validator("pilot_tone_frequency", "pilot_tone_power_ratio_db", mode="before")
    @classmethod
    def _coerce_pilot_field(cls, v: Any) -> Any:
        """Coerce pilot metadata to a 1-D per-channel ``float64`` array.

        ``None`` passes through; anything else (scalar, sequence, or array)
        becomes a 1-D ``float64`` ``np.ndarray`` — one value per channel — so the
        field is handled uniformly like the other array fields (always an array,
        never a scalar or list).  A scalar becomes a length-1 array.
        """
        if v is None:
            return None
        return np.asarray(v, dtype=np.float64).reshape(-1)

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
                from .. import mapping

                self.source_symbols = mapping.map_bits(
                    self.source_bits,
                    self.mod_scheme,
                    self.mod_order,
                    unipolar=self.mod_unipolar or False,
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

        if self.ps_pmf is not None and mod_order:
            rows.append(
                (
                    "PS shaping (ν)",
                    f"{self.ps_nu:.4f}" if self.ps_nu is not None else "unknown",
                )
            )

        rows += [
            ("Duration", helpers.format_si(self.duration, "s")),
            ("Center frequency", helpers.format_si(self.center_frequency, "Hz")),
        ]

        if self.digital_frequency_offset is not None:
            rows.append(
                (
                    "Frequency offset",
                    helpers.format_si(self.digital_frequency_offset, "Hz"),
                )
            )

        def _pilot_row(value, fmt) -> str:
            # value is a 1-D per-channel array (validator-coerced).
            return ", ".join(fmt(float(x)) for x in value)

        if self.pilot_tone_frequency is not None:
            rows.append(
                (
                    "Pilot tone frequency",
                    _pilot_row(
                        self.pilot_tone_frequency,
                        lambda x: helpers.format_si(x, "Hz"),
                    ),
                )
            )

        if self.pilot_tone_power_ratio_db is not None:
            rows.append(
                (
                    "Pilot tone power",
                    _pilot_row(self.pilot_tone_power_ratio_db, lambda x: f"{x:.1f} dB"),
                )
            )

        rows += [
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
        rows.append(("ps_pmf", _yn(self.ps_pmf)))
        rows.append(("frame attached", _yn(self.frame)))

        # ── Section 4: Resolved data ──────────────────────────────────────
        rows.append(("─── Resolved data", ""))
        rows.append(("resolved_symbols", _yn(self.resolved_symbols)))
        rows.append(("resolved_bits", _yn(self.resolved_bits)))

        # ── Render ────────────────────────────────────────────────────────
        # Rich HTML table in Jupyter; plain-text table everywhere else. IPython
        # is an optional dependency (the ``notebook`` extra) — degrade
        # gracefully to the plain-text path when it is not installed.
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display

            ipy = get_ipython()
            in_kernel = ipy is not None and "IPKernelApp" in ipy.config
        except ImportError:
            in_kernel = False

        if in_kernel:
            body = "".join(
                f"<tr><td><b>{prop}</b></td><td>{val}</td></tr>" for prop, val in rows
            )
            display(HTML(f"<table>{body}</table>"))
        else:
            width = max(len(prop) for prop, _ in rows)
            text = "\n".join(f"{prop.ljust(width)}  {val}" for prop, val in rows)
            logger.info("\n" + text)

    def copy(self) -> "Signal":  # type: ignore[override]
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

    def export_samples_to_jax(self, device: str | None = None) -> Any:
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
    def bits_per_symbol(self) -> int | None:
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
    # Generation Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        num_symbols: int,
        sps: int,
        symbol_rate: float,
        modulation: str,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: str = "none",
        num_streams: int = 1,
        seed: int | None = None,
        duty_cycle: float = 1.0,
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        rise_time: float = 0.0,
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
            Pulse shaping filter type (e.g., ``'rrc'``, ``'rect'``).
        num_streams : int, default 1
            Number of independent streams (MIMO).
        seed : int, optional
            Seed for reproducible random generation.
        duty_cycle : float, default 1.0
            Fraction of the symbol period occupied by the pulse (rect/smoothrect).
            Overridden to 0.5 when ``rz=True``.
        filter_span : int, default 10
            Filter span in symbols for smoothrect/gaussian/rrc/rc/sinc.
        rrc_rolloff : float, default 0.35
            Roll-off factor for the RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for the RC filter.
        rise_time : float, default 0.22
            10%-90% edge transition duration in symbol periods for smoothrect.
        duty_cycle : float, default 1.0
            FWHM of the Gaussian pulse in symbol periods.

        Returns
        -------
        Signal
            A new `Signal` instance.

        Notes
        -----
        Samples are normalized to unit symbol power (Es = 1, average sample power = 1/sps).
        Call ``resolve_symbols()`` before demapping or computing metrics.
        """
        from .. import filtering, mapping

        if sps != int(sps) or sps < 1:
            logger.warning(
                f"sps={sps!r} is not a positive integer. "
                "Non-integer sps is valid for captured/resampled signals but not for "
                "generation: resample_poly requires an integer upsampling factor, so "
                "the sample buffer would not match the stored sampling_rate metadata. "
                "To generate at a fractional sps, generate at an integer sps then call "
                "Signal.resample(up=..., down=...)."
            )
            raise ValueError(
                f"sps must be a positive integer for signal generation, got {sps!r}."
            )
        sps = int(sps)

        # When rz=True and the caller hasn't specified a custom duty_cycle,
        # default to 50% (canonical RZ). Explicit duty_cycle values are preserved.
        if rz and duty_cycle == 1.0:
            duty_cycle = 0.5

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
            symbols=symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            rz=rz,
            duty_cycle=duty_cycle,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            rise_time=rise_time,
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
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            rise_time=rise_time,
            duty_cycle=duty_cycle,
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
        seed: int | None = None,
        duty_cycle: float = 1.0,
        filter_span: int = 10,
        rise_time: float = 0.0,
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
        duty_cycle : float, default 1.0
            Fraction of the symbol period occupied by the pulse. Overridden to
            0.5 when ``rz=True``.
        filter_span : int, default 10
            Filter span in symbols (smoothrect only).
        rise_time : float, default 0.22
            10%-90% edge transition duration in symbol periods (smoothrect only).

        Returns
        -------
        Signal
            A `Signal` object containing the generated PAM waveform.

        Notes
        -----
        Samples are normalized to unit symbol power (Es = 1, average sample power = 1/sps).
        Call ``resolve_symbols()`` before demapping or computing metrics.
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
            filter_span=filter_span,
            rise_time=rise_time,
            duty_cycle=duty_cycle,
        )

    @classmethod
    def psk(
        cls,
        num_symbols: int,
        sps: int,
        symbol_rate: float,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: int | None = None,
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        rise_time: float = 0.0,
        duty_cycle: float = 1.0,
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
        duty_cycle : float, default 1.0
            Fraction of the symbol period occupied by the pulse (rect/smoothrect).
            Only meaningful when ``rz=True``.
        filter_span : int, default 10
            Filter span in symbols.
        rrc_rolloff : float, default 0.35
            Roll-off factor for the RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for the RC filter.
        rise_time : float, default 0.22
            10%-90% edge transition duration in symbol periods (smoothrect only).
        duty_cycle : float, default 1.0
            FWHM of the Gaussian pulse in symbol periods (gaussian only).

        Returns
        -------
        Signal
            A `Signal` object containing the PSK waveform.

        Notes
        -----
        Samples are normalized to unit symbol power (Es = 1, average sample power = 1/sps).
        Call ``resolve_symbols()`` before demapping or computing metrics.
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
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            rise_time=rise_time,
            duty_cycle=duty_cycle,
        )

    @classmethod
    def qam(
        cls,
        num_symbols: int,
        sps: int,
        symbol_rate: float,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: int | None = None,
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        rise_time: float = 0.0,
        duty_cycle: float = 1.0,
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
        duty_cycle : float, default 1.0
            Fraction of the symbol period occupied by the pulse (rect/smoothrect).
            Only meaningful when ``rz=True``.
        filter_span : int, default 10
            Filter span in symbols.
        rrc_rolloff : float, default 0.35
            Roll-off factor for the RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for the RC filter.
        rise_time : float, default 0.22
            10%-90% edge transition duration in symbol periods (smoothrect only).
        duty_cycle : float, default 1.0
            FWHM of the Gaussian pulse in symbol periods (gaussian only).

        Returns
        -------
        Signal
            A `Signal` object containing the QAM waveform.

        Notes
        -----
        Samples are normalized to unit symbol power (Es = 1, average sample power = 1/sps).
        Call ``resolve_symbols()`` before demapping or computing metrics.
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
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            rise_time=rise_time,
            duty_cycle=duty_cycle,
        )

    @classmethod
    def psqam(
        cls,
        num_symbols: int,
        sps: int,
        symbol_rate: float,
        order: int,
        *,
        nu: float | None = None,
        entropy: float | None = None,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: int | None = None,
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        duty_cycle: float = 1.0,
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
            MB shaping parameter nu >= 0. nu = 0 is uniform QAM.
            Larger values apply stronger shaping (lower entropy, lower power).
        entropy : float, optional
            Target per-symbol entropy in bits, in the range (0, log2(order)].
            optimal_nu is called to solve for the corresponding nu.
        pulse_shape : str, default "rrc"
            Pulse shaping filter type.
        num_streams : int, default 1
            Number of independent streams (MIMO).
        seed : int, optional
            Random seed for reproducible symbol generation.
        filter_span : int, default 10
            Filter span in symbols.
        rrc_rolloff : float, default 0.35
            Roll-off factor for the RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for the RC filter.

        Returns
        -------
        Signal
            A ``Signal`` with ``mod_scheme="PS-QAM"``, ``ps_pmf`` set to the MB
            distribution, and both ``source_symbols`` and ``source_bits`` populated.

        Notes
        -----
        ``source_bits`` carry the non-uniform MB statistics (correct for BER/GMI
        estimation, not a full coded PAS transmitter). Average symbol energy is
        below 1 for nu > 0; pass ``pmf=signal.ps_pmf`` to ``metrics.mi`` and
        ``compute_llr`` for correct soft-demapping.

        Examples
        --------
        >>> sig = Signal.psqam(10000, sps=4, symbol_rate=32e9, order=64, entropy=6.0)
        >>> sig = Signal.psqam(10000, sps=4, symbol_rate=32e9, order=64, nu=0.3)
        """
        from .. import filtering, mapping

        if sps != int(sps) or sps < 1:
            logger.warning(
                f"sps={sps!r} is not a positive integer. "
                "Non-integer sps is valid for captured/resampled signals but not for "
                "generation: resample_poly requires an integer upsampling factor, so "
                "the sample buffer would not match the stored sampling_rate metadata. "
                "To generate at a fractional sps, generate at an integer sps then call "
                "Signal.resample(up=..., down=...)."
            )
            raise ValueError(
                f"sps must be a positive integer for signal generation, got {sps!r}."
            )
        sps = int(sps)

        if (nu is None) == (entropy is None):
            raise ValueError("Exactly one of `nu` or `entropy` must be specified.")

        if entropy is not None:
            nu_val, _ = mapping.optimal_nu(order, entropy)
        else:
            assert nu is not None
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
            symbols=symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            duty_cycle=duty_cycle,
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
            ps_nu=nu_val,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            duty_cycle=duty_cycle,
        )

    # -------------------------------------------------------------------------
    # Resolving and Demapping Methods
    # -------------------------------------------------------------------------

    def resolve_symbols(self, offset: int = 0) -> None:
        """Decimate to symbol rate and populate ``resolved_symbols``.

        Decimates ``self.samples`` to 1 SPS and normalises to unit average
        power (Es = 1), writing the result to ``self.resolved_symbols``.

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
            from .. import multirate

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
        ``resolve_symbols()`` first.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments forwarded to ``mapping.demap_symbols_hard``
            (e.g. ``unipolar``).

        Raises
        ------
        ValueError
            If modulation metadata is missing or ``resolve_symbols()`` has
            not been called yet.
        """
        from ..mapping import demap_symbols_hard

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

        # Auto-forward ps_pmf for PS-QAM: receive-path symbols at unit avg
        # power live on ``{s_m/sqrt(E_PS)}``; demap_symbols_hard rescales
        # them to the ``{s_m}`` grid expected by ``gray_constellation``.
        kwargs.setdefault("pmf", self.ps_pmf)
        bits = demap_symbols_hard(
            symbols=self.resolved_symbols,
            modulation=self.mod_scheme,
            order=self.mod_order,
            unipolar=self.mod_unipolar or False,
            **kwargs,
        )
        self.resolved_bits = bits

    def resolve_phase_ambiguity(self) -> None:
        """
        Resolves rotational phase ambiguity in ``resolved_symbols`` in place.

        After blind CPR (VV, BPS, Tikhonov) a global pi/2 (QAM) or
        2pi/M (PSK) phase ambiguity remains per stream.  This method tests
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
        from ..recovery import resolve_phase_ambiguity as _resolve

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
            pmf=self.ps_pmf,
        )

    def resolve_channel_permutation(self, num_skip_symbols: int = 0) -> None:
        """Resolve a polarization (channel) permutation in ``resolved_symbols``.

        A MIMO equalizer may emit the polarizations in swapped output order; per
        channel metrics would then score a valid demux as random.  This matches
        each output stream to the ``source_symbols`` stream it carries and
        reorders ``resolved_symbols`` in place to reference order.

        Run **before** ``resolve_phase_ambiguity()`` (it is rotation
        invariant) and before SER/BER.  No-op for SISO or a converged
        data-aided equalizer; the fix for blind MIMO whose output order is
        arbitrary.

        Parameters
        ----------
        num_skip_symbols : int, default 0
            Leading symbols excluded from the correlation scoring.

        Raises
        ------
        ValueError
            If ``resolved_symbols`` or ``source_symbols`` are not populated.
        """
        from ..recovery import resolve_channel_permutation as _resolve_perm

        if self.resolved_symbols is None:
            raise ValueError(
                "resolved_symbols is not set. Call resolve_symbols() or assign "
                "resolved_symbols directly before calling "
                "resolve_channel_permutation()."
            )
        if self.source_symbols is None:
            raise ValueError(
                "source_symbols is not set. Populate source_symbols (the known TX "
                "symbol sequence) before calling resolve_channel_permutation()."
            )

        self.resolved_symbols = _resolve_perm(
            self.resolved_symbols,
            self.source_symbols,
            num_skip_symbols=num_skip_symbols,
        )
