"""Analog input task configuration for NI-DAQmx devices.

Provides the ``AITask`` class for programmatic creation and management
of analog input acquisition tasks.  Supports accelerometer (IEPE),
force (IEPE), and voltage channels with optional linear custom scales.

Architecture
------------
Direct delegation: the nidaqmx Task is created immediately in the
constructor. :meth:`add_channel` delegates straight to
``task.ai_channels.add_ai_*_chan()``.  The nidaqmx Task object is the
single source of truth; no intermediate channel dict is maintained.

Notes
-----
nidaqmx is an optional dependency.  If absent, ``_NIDAQMX_AVAILABLE``
is ``False`` and construction raises ``RuntimeError``.
"""

from __future__ import annotations

import pathlib
import warnings
from datetime import datetime

import numpy as np

from .base_task import _SAMPLE_MODE_ATTR, BaseTask
from .utils import UNITS, UNITS_REVERSE, _require_nidaqmx

try:
    import nidaqmx
    from nidaqmx import constants

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False

# String-to-enum-attribute mapping for analog trigger slopes, resolved
# against the live ``constants`` namespace at call time.
_SLOPE_ATTR: dict[str, str] = {"rising": "RISING", "falling": "FALLING"}


class AITask(BaseTask):
    """Programmatic analog input task for NI-DAQmx devices.

    The nidaqmx hardware task is created immediately at construction.
    Channels are added via :meth:`add_channel` which delegates directly
    to the nidaqmx task.  Call :meth:`configure` to set up timing,
    then :meth:`start` to begin acquisition.

    Parameters
    ----------
    task_name : str
        Unique name for this task.  Must not collide with an existing
        task in NI MAX.
    sample_rate : float
        Sampling rate in Hz.

    Raises
    ------
    ValueError
        If ``task_name`` already exists in NI MAX.
    RuntimeError
        If nidaqmx is not installed.

    Examples
    --------
    >>> task = AITask("vibration_test", sample_rate=25600)
    >>> task.add_channel("accel_x", device='Dev1', channel_ind=0,
    ...                  sensitivity=100.0, sensitivity_units="mV/g",
    ...                  units="g")
    >>> task.start()
    """

    _channel_attr = "ai_channels"
    _channel_type_label = "AI"

    def __init__(
        self,
        task_name: str,
        sample_rate: float,
    ) -> None:
        _require_nidaqmx()

        self.task_name = task_name
        self.sample_rate = sample_rate

        # Device discovery
        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [d.name for d in system.devices]
        self.device_product_type: list[str] = [
            d.product_type for d in system.devices
        ]

        # Guard against duplicate task names before allocating a handle
        if task_name in system.tasks.task_names:
            raise ValueError(
                f"Task name '{task_name}' already exists in NI MAX. "
                "Choose a different name or delete the existing task first."
            )

        self.sample_mode = constants.AcquisitionType.CONTINUOUS

        # Synchronized-acquisition settings (introspection only — the live
        # nidaqmx task remains the source of truth). Set by configure().
        self.sample_mode_str: str = "continuous"
        self.samples_per_channel: int | None = None
        self.clock_source: str | None = None

        # Create the nidaqmx task immediately — it is the single source of truth
        self.task = nidaqmx.task.Task(new_task_name=task_name)

        # Ownership flag: True when this AITask created the nidaqmx.Task,
        # False when wrapping an externally-provided task via from_task().
        self._owns_task = True

    # -- Channel configuration -----------------------------------------------

    def add_channel(
        self,
        channel_name: str,
        device: str,
        channel_ind: int,
        sensitivity: float | None = None,
        sensitivity_units: str | None = None,
        units: str | None = None,
        scale: float | tuple[float, float] | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        """Add an analog input channel to this task.

        The channel is configured immediately on the underlying nidaqmx
        task.  Channel type is determined from the units constant's
        ``__objclass__.__name__`` (``AccelUnits``, ``ForceUnits``, or
        ``VoltageUnits``).  Providing a *scale* forces the voltage path
        regardless of units.

        Parameters
        ----------
        channel_name : str
            Unique name for this channel.
        device : str
            NI-DAQmx device name string (e.g. ``'Dev1'``, ``'cDAQ1Mod1'``,
            ``'SimDev1'``).  Must be a non-empty string.  The driver
            validates the device name at channel-creation time and raises
            ``DaqError`` if the device does not exist.
        channel_ind : int
            Physical analog-input channel number on the device.
        sensitivity : float, optional
            Sensor sensitivity.  Required for accel/force channels
            when *scale* is not provided.
        sensitivity_units : str, optional
            Key into the ``UNITS`` dict for sensor sensitivity units
            (e.g. ``'mV/g'``, ``'mV/N'``).
        units : str, optional
            Key into the ``UNITS`` dict for output measurement units
            (e.g. ``'g'``, ``'N'``, ``'V'``).  Always required.
        scale : float or tuple[float, float], optional
            Linear custom scale.  Float → slope with y_intercept=0.
            Tuple → ``(slope, y_intercept)``.  When given, *sensitivity*
            and *sensitivity_units* are not required.
        min_val : float, optional
            Minimum expected value.  ``0.0`` is a valid value.
        max_val : float, optional
            Maximum expected value.  ``0.0`` is a valid value.

        Raises
        ------
        ValueError
            Duplicate channel name, duplicate physical channel, empty
            device string, invalid units, missing sensitivity, or missing
            units.
        TypeError
            Invalid *scale* type.
        RuntimeError
            If this task was created via :meth:`from_task` (channels must
            be configured on the nidaqmx task before wrapping).
        """
        # -- Ownership check ------------------------------------------------
        if not self._owns_task:
            raise RuntimeError(
                "Cannot add channels to an externally-provided task. "
                "Configure channels on the nidaqmx.Task before calling "
                "from_task()."
            )

        # -- Basic validation -----------------------------------------------
        if units is None:
            raise ValueError(
                "units must be specified. "
                f"Valid units: {list(UNITS.keys())}"
            )

        # Duplicate name detection: check what nidaqmx already knows about
        if channel_name in self.task.channel_names:
            raise ValueError(
                f"Channel name '{channel_name}' already exists in this task."
            )

        if not device or not isinstance(device, str):
            raise ValueError("device must be a non-empty string")

        # Duplicate physical channel detection: iterate the live task channels
        physical_channel = f"{device}/ai{channel_ind}"
        for ch in self.task.ai_channels:
            if ch.physical_channel.name == physical_channel:
                raise ValueError(
                    f"Physical channel ai{channel_ind} on device "
                    f"'{device}' is already in use."
                )

        # -- Scale type validation ------------------------------------------
        if scale is not None and not isinstance(scale, (int, float, tuple)):
            raise TypeError(
                f"scale must be a float or tuple, got {type(scale).__name__}."
            )

        # -- Units / sensitivity validation (skip when scale given) ---------
        if scale is None:
            if sensitivity_units is not None and sensitivity_units not in UNITS:
                raise ValueError(
                    f"Invalid sensitivity_units: '{sensitivity_units}'. "
                    f"Valid sensitivity_units: {list(UNITS.keys())}"
                )
            if units not in UNITS:
                raise ValueError(
                    f"Invalid units: '{units}'. "
                    f"Valid units: {list(UNITS.keys())}"
                )

        # -- Resolve nidaqmx constants from UNITS dict ----------------------
        resolved_units = UNITS[units] if units in UNITS else units
        resolved_sens_units = (
            UNITS[sensitivity_units]
            if sensitivity_units is not None and sensitivity_units in UNITS
            else sensitivity_units
        )

        # -- Determine channel mode from the units constant's enum class ----
        # When a custom scale is provided, force the voltage path — nidaqmx
        # requires FROM_CUSTOM_SCALE on a VoltageAIChannel.
        if scale is not None:
            mode = "VoltageUnits"
        elif hasattr(resolved_units, "__objclass__"):
            mode = resolved_units.__objclass__.__name__
        else:
            mode = "VoltageUnits"

        # -- Sensitivity required for non-voltage, non-scale channels -------
        if scale is None and mode != "VoltageUnits":
            if sensitivity is None:
                raise ValueError(
                    "sensitivity must be specified for non-voltage "
                    "channels when no scale is provided."
                )
            if sensitivity_units is None:
                raise ValueError(
                    "sensitivity_units must be specified for non-voltage "
                    "channels when no scale is provided."
                )

        # -- Create linear custom scale in NI MAX when requested ------------
        custom_scale_name = ""
        if scale is not None:
            if isinstance(scale, tuple):
                slope, y_intercept = float(scale[0]), float(scale[1])
            else:
                slope, y_intercept = float(scale), 0.0

            # units_str is the original string, used as the scaled output unit
            scale_obj = nidaqmx.Scale.create_lin_scale(
                f"{channel_name}_scale",
                slope=slope,
                y_intercept=y_intercept,
                pre_scaled_units=constants.VoltageUnits.VOLTS,
                scaled_units=units,
            )
            custom_scale_name = scale_obj.name

        # -- Build the options dict shared by all channel types -------------
        options: dict[str, Any] = {
            "physical_channel": physical_channel,
            "name_to_assign_to_channel": channel_name,
            "terminal_config": constants.TerminalConfiguration.DEFAULT,
        }

        # Use is not None so that 0.0 is correctly forwarded (LDAQ bug fix)
        if min_val is not None:
            options["min_val"] = min_val
        if max_val is not None:
            options["max_val"] = max_val

        # -- Dispatch to the correct nidaqmx channel factory ---------------
        if mode == "AccelUnits":
            options["sensitivity"] = sensitivity
            options["sensitivity_units"] = resolved_sens_units
            options["units"] = resolved_units
            self.task.ai_channels.add_ai_accel_chan(**options)

        elif mode == "ForceUnits":
            options["sensitivity"] = sensitivity
            options["sensitivity_units"] = resolved_sens_units
            options["units"] = resolved_units
            self.task.ai_channels.add_ai_force_iepe_chan(**options)

        else:
            # VoltageUnits path — plain voltage or custom-scale channel
            if custom_scale_name:
                options["units"] = constants.VoltageUnits.FROM_CUSTOM_SCALE
                options["custom_scale_name"] = custom_scale_name
            else:
                options["units"] = resolved_units
            self.task.ai_channels.add_ai_voltage_chan(**options)


    # -- Task lifecycle ------------------------------------------------------

    def configure(
        self,
        *,
        sample_mode: str = "continuous",
        samples_per_channel: int | None = None,
        clock_source: str | None = None,
    ) -> None:
        """Configure sample-clock timing for acquisition.

        Sets up the sample-clock timing on the nidaqmx task and validates
        that the driver accepted the requested sample rate.  Call
        :meth:`start` afterwards to begin acquisition.  Defaults reproduce
        the classic behavior: continuous acquisition from the onboard clock.

        Parameters
        ----------
        sample_mode : str, optional
            ``'continuous'`` (default) or ``'finite'``.  Finite tasks
            acquire exactly *samples_per_channel* samples and then stop —
            the mode used for hardware-triggered synchronized acquisition.
        samples_per_channel : int, optional
            Number of samples per channel.  Required (> 0) for
            ``'finite'`` mode.  With ``'continuous'`` mode it is an
            optional buffer-size hint (nidaqmx semantics).
        clock_source : str, optional
            Terminal name of an external sample clock (e.g.
            ``'/cDAQ1Mod1/ai/SampleClock'``).  ``None`` (default) uses the
            onboard clock.  When set, the rate-coercion check is skipped —
            the reported rate is whatever the external clock provides, so
            *sample_rate* must match the master task's rate (it sizes
            buffers and timeouts).

        Raises
        ------
        ValueError
            If *sample_mode* is not ``'continuous'``/``'finite'``; if
            ``'finite'`` is requested without a positive integer
            *samples_per_channel*; if *clock_source* is not a non-empty
            string; or if the driver coerces the sample rate while using
            the onboard clock.
        RuntimeError
            If this task was created via :meth:`from_task` (timing must
            be configured on the nidaqmx task before wrapping).
        """
        self._check_start_preconditions()

        if sample_mode not in _SAMPLE_MODE_ATTR:
            raise ValueError(
                f"sample_mode must be one of {list(_SAMPLE_MODE_ATTR)}, "
                f"got {sample_mode!r}."
            )
        if samples_per_channel is not None and (
            not isinstance(samples_per_channel, int)
            or isinstance(samples_per_channel, bool)
            or samples_per_channel <= 0
        ):
            raise ValueError(
                f"samples_per_channel must be a positive integer, "
                f"got {samples_per_channel!r}."
            )
        if sample_mode == "finite" and samples_per_channel is None:
            raise ValueError(
                "samples_per_channel must be specified (int > 0) when "
                "sample_mode='finite'."
            )
        if clock_source is not None and (
            not isinstance(clock_source, str) or not clock_source
        ):
            raise ValueError(
                f"clock_source must be a non-empty terminal name string "
                f"(e.g. '/cDAQ1Mod1/ai/SampleClock') or None, "
                f"got {clock_source!r}."
            )

        acq_type = getattr(
            constants.AcquisitionType, _SAMPLE_MODE_ATTR[sample_mode]
        )

        timing_kwargs: dict[str, Any] = {
            "rate": self.sample_rate,
            "sample_mode": acq_type,
        }
        if samples_per_channel is not None:
            timing_kwargs["samps_per_chan"] = samples_per_channel
        if clock_source is not None:
            timing_kwargs["source"] = clock_source

        self.task.timing.cfg_samp_clk_timing(**timing_kwargs)

        # Store for introspection — the live task remains the source of truth
        self.sample_mode = acq_type
        self.sample_mode_str = sample_mode
        self.samples_per_channel = samples_per_channel
        self.clock_source = clock_source

        # Rate-coercion check applies only to the onboard clock: with an
        # external clock_source the reported rate is whatever the exported
        # clock provides (validating it is the master task's job).
        if clock_source is None:
            actual_rate = float(self.task.timing.samp_clk_rate)
            requested_rate = float(self.sample_rate)
            if actual_rate != requested_rate:
                raise ValueError(
                    f"Sample rate {requested_rate} Hz is not supported by this "
                    f"device. The driver coerced it to {actual_rate} Hz. "
                    "Use a rate that the device supports."
                )

    def set_analog_start_trigger(
        self,
        source: str,
        slope: str = "rising",
        level: float = 0.0,
    ) -> None:
        """Configure an analog edge start trigger for this task.

        Arms the task to wait for the analog signal on *source* to cross
        *level* with the given *slope* before starting the acquisition.
        Configure the trigger before calling :meth:`start`.

        Parameters
        ----------
        source : str
            Name of the analog trigger source (e.g. ``'APFI0'`` or an
            analog input channel like ``'Dev1/ai0'``).
        slope : str, optional
            Trigger slope: ``'rising'`` (default) or ``'falling'``.
        level : float, optional
            Threshold level in the units of the trigger source signal,
            by default ``0.0``.  Coerced to ``float``.

        Raises
        ------
        ValueError
            If *source* is not a non-empty string, or *slope* is not
            ``'rising'`` or ``'falling'``.
        RuntimeError
            If this task was created via :meth:`from_task` (externally
            owned — configure triggers on the nidaqmx.Task directly).
        """
        if not self._owns_task:
            raise RuntimeError(
                "Cannot configure a trigger on an externally-provided task. "
                "Configure the nidaqmx.Task directly or pass an "
                "already-configured task to from_task()."
            )
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"source must be a non-empty trigger source name string "
                f"(e.g. 'APFI0'), got {source!r}."
            )
        if slope not in _SLOPE_ATTR:
            raise ValueError(
                f"slope must be one of {list(_SLOPE_ATTR)}, got {slope!r}."
            )

        self.task.triggers.start_trigger.cfg_anlg_edge_start_trig(
            trigger_source=source,
            trigger_slope=getattr(constants.Slope, _SLOPE_ATTR[slope]),
            trigger_level=float(level),
        )

    def acquire(self, n_samples: int | None = None) -> np.ndarray:
        """Read samples from the hardware buffer.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples per channel to read.  If provided, the call
            **blocks** until exactly *n_samples* are available — suitable
            for scripts and notebooks.  If ``None`` (default), drains every
            sample currently in the buffer without blocking (``READ_ALL_
            AVAILABLE``) — suitable for acquisition loops.

            On a **finite** task (``configure(sample_mode='finite', ...)``)
            ``acquire()`` with the default ``None`` blocks until the finite
            acquisition completes and returns all
            ``samples_per_channel`` samples — this is the synchronization
            mechanism for hardware-triggered multi-task acquisition.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(n_samples, n_channels)``.  For single-
            channel tasks nidaqmx returns a 1-D list; this method reshapes
            it to ``(n_samples, 1)`` so callers always receive a consistent
            shape.
        """
        count = -1 if n_samples is None else n_samples
        raw = self.task.read(number_of_samples_per_channel=count)
        data = np.array(raw)

        if data.ndim == 1:
            # Single-channel: nidaqmx returns a flat list → (n_samples, 1)
            data = data.reshape(-1, 1)
        else:
            # Multi-channel: nidaqmx returns (n_channels, n_samples) → transpose
            data = data.T

        return data  # shape: (n_samples, n_channels)

    def save(self, clear_task: bool = True) -> None:
        """Save the task to NI MAX.

        The task always exists in the direct-delegation architecture, so
        this method calls ``task.save()`` directly without auto-initiating.
        After saving, optionally closes the task.

        Parameters
        ----------
        clear_task : bool, optional
            If ``True`` (default), call :meth:`clear_task` after saving.
        """
        self.task.save(overwrite_existing_task=True)

        if clear_task:
            self.clear_task()

    # -- TOML config persistence --------------------------------------------

    def save_config(self, path: str | pathlib.Path) -> None:
        """Serialise the task configuration to a TOML file.

        Reads channel information directly from ``self.task.ai_channels``
        (the nidaqmx Task is the single source of truth).  Writes a
        human-readable TOML file that can be loaded back with
        :meth:`from_config` to recreate the same task on any compatible
        hardware.  Device names are replaced by short aliases in
        ``[devices]`` so that changing chassis enumeration only requires
        editing one line per module.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path.  The file is created or overwritten.

        Notes
        -----
        TOML is generated with simple string formatting — no third-party
        library is required for writing.  ``min_val`` / ``max_val`` are
        always written (read from ``channel.ai_rng_low`` /
        ``channel.ai_rng_high``).  For custom-scale channels, ``units``
        is omitted (the scale's output unit string is not recoverable
        from the nidaqmx channel object).

        Examples
        --------
        >>> task.save_config("/tmp/vibration.toml")
        """
        # Build device alias map from unique device names in channel order
        device_names_seen: list[str] = []
        for ch in self.task.ai_channels:
            dev = ch.physical_channel.name.rsplit("/", 1)[0]
            if dev not in device_names_seen:
                device_names_seen.append(dev)
        device_to_alias: dict[str, str] = {
            name: f"dev{i}" for i, name in enumerate(device_names_seen)
        }

        lines: list[str] = []

        # Header comment with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"# Generated by nidaqwrapper 0.1.0 on {timestamp}")
        lines.append("")

        # [task] section
        lines.append("[task]")
        lines.append(f'name = "{self.task_name}"')
        lines.append(f"sample_rate = {self.sample_rate}")
        lines.append('type = "input"')
        lines.append("")

        # [devices] section
        lines.append("[devices]")
        for dev_name, alias in device_to_alias.items():
            try:
                dev_idx = self.device_list.index(dev_name)
                product_type = self.device_product_type[dev_idx]
                lines.append(f'{alias} = "{dev_name}"  # {product_type}')
            except (ValueError, IndexError):
                lines.append(f'{alias} = "{dev_name}"')
        lines.append("")

        # [[channels]] entries — read all info from the live task channels
        for ch in self.task.ai_channels:
            phys = ch.physical_channel.name
            dev_name = phys.rsplit("/", 1)[0]
            ch_spec = phys.rsplit("/", 1)[1]
            channel_ind = int(ch_spec.lstrip("ai"))
            alias = device_to_alias[dev_name]
            meas_type = ch.ai_meas_type

            lines.append("[[channels]]")
            lines.append(f'name = "{ch.name}"')
            lines.append(f'device = "{alias}"')
            lines.append(f"channel = {channel_ind}")

            if meas_type == constants.UsageTypeAI.ACCELERATION_ACCELEROMETER_CURRENT_INPUT:
                lines.append(f"sensitivity = {ch.ai_accel_sensitivity}")
                sens_units_str = UNITS_REVERSE.get(
                    ch.ai_accel_sensitivity_units, str(ch.ai_accel_sensitivity_units)
                )
                lines.append(f'sensitivity_units = "{sens_units_str}"')
                units_str = UNITS_REVERSE.get(ch.ai_accel_units, str(ch.ai_accel_units))
                lines.append(f'units = "{units_str}"')

            elif meas_type == constants.UsageTypeAI.FORCE_IEPE_SENSOR:
                lines.append(f"sensitivity = {ch.ai_force_iepe_sensor_sensitivity}")
                sens_units_str = UNITS_REVERSE.get(
                    ch.ai_force_iepe_sensor_sensitivity_units,
                    str(ch.ai_force_iepe_sensor_sensitivity_units),
                )
                lines.append(f'sensitivity_units = "{sens_units_str}"')
                units_str = UNITS_REVERSE.get(ch.ai_force_units, str(ch.ai_force_units))
                lines.append(f'units = "{units_str}"')

            else:
                # Voltage path (plain voltage or custom scale)
                scale_name = ch.ai_custom_scale.name
                if scale_name:
                    slope = ch.ai_custom_scale.lin_slope
                    y_int = ch.ai_custom_scale.lin_y_intercept
                    lines.append(f"scale = [{slope}, {y_int}]")
                else:
                    units_str = UNITS_REVERSE.get(
                        ch.ai_voltage_units, str(ch.ai_voltage_units)
                    )
                    lines.append(f'units = "{units_str}"')

            lines.append(f"min_val = {ch.ai_rng_low}")
            lines.append(f"max_val = {ch.ai_rng_high}")
            lines.append("")

        pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_config(cls, path: str | pathlib.Path) -> AITask:
        """Create an :class:`AITask` from a TOML configuration file.

        Reads the TOML file produced by :meth:`save_config`, constructs
        a new task, and calls :meth:`add_channel` for every ``[[channels]]``
        entry.  Device aliases are resolved to system device indices.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file.

        Returns
        -------
        AITask
            A fully configured task (channels added, not yet started).

        Raises
        ------
        ValueError
            If the ``[task]`` or ``[devices]`` section is absent, or if a
            channel references an unknown device alias.
        tomllib.TOMLDecodeError
            On syntactically invalid TOML (propagated from the parser).

        Examples
        --------
        >>> task = AITask.from_config("/tmp/vibration.toml")
        >>> task.start()
        """
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(path, "rb") as fh:
            data = tomllib.load(fh)

        if "task" not in data:
            raise ValueError(
                "TOML file is missing required [task] section."
            )
        if "devices" not in data:
            raise ValueError(
                "TOML file is missing required [devices] section."
            )

        task_section = data["task"]
        alias_to_name: dict[str, str] = data["devices"]

        task = cls(task_section["name"], sample_rate=task_section["sample_rate"])

        for ch in data.get("channels", []):
            alias = ch["device"]
            if alias not in alias_to_name:
                raise ValueError(
                    f"Channel '{ch['name']}' references unknown device alias "
                    f"'{alias}'. Available aliases: {list(alias_to_name)}"
                )

            device_name = alias_to_name[alias]

            # TOML arrays become Python lists; convert to tuple for add_channel()
            raw_scale = ch.get("scale")
            scale: float | tuple[float, float] | None
            if raw_scale is not None:
                scale = (float(raw_scale[0]), float(raw_scale[1]))
            else:
                scale = None

            task.add_channel(
                channel_name=ch["name"],
                device=device_name,
                channel_ind=ch["channel"],
                sensitivity=ch.get("sensitivity"),
                sensitivity_units=ch.get("sensitivity_units"),
                units=ch.get("units"),
                scale=scale,
                min_val=ch.get("min_val"),
                max_val=ch.get("max_val"),
            )

        return task

    @classmethod
    def from_task(
        cls, task: nidaqmx.task.Task, take_ownership: bool = False
    ) -> AITask:
        """Wrap a pre-created ``nidaqmx.task.Task`` object.

        Creates an :class:`AITask` instance that wraps an existing nidaqmx
        task instead of creating its own.  This provides an escape hatch for
        advanced users who need to configure task properties not exposed by
        the wrapper API.

        By default, the wrapper does NOT own the task: :meth:`add_channel`,
        :meth:`configure`, and :meth:`start` are blocked, and
        :meth:`clear_task` / ``__exit__`` will NOT close the task.
        Pass ``take_ownership=True`` to transfer ownership — the wrapper will
        then permit all mutating methods and will close the task on
        :meth:`clear_task`.

        Parameters
        ----------
        task : nidaqmx.task.Task
            A pre-created nidaqmx Task object with at least one AI channel.
        take_ownership : bool, optional
            If ``True``, the wrapper takes ownership of the task and all
            mutating methods (add_channel, configure, start, clear_task)
            are permitted.  The task will be closed when :meth:`clear_task`
            is called.  Default is ``False`` (original behaviour: task is
            not owned, mutating methods raise ``RuntimeError``).

        Returns
        -------
        AITask
            An :class:`AITask` instance wrapping the provided task.

        Raises
        ------
        ValueError
            If the task has no AI channels.

        Warnings
        --------
        If the task is already running, a warning is issued.

        Examples
        --------
        >>> import nidaqmx
        >>> raw_task = nidaqmx.task.Task("external")
        >>> raw_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        >>> raw_task.timing.cfg_samp_clk_timing(rate=25600)
        >>> wrapped = AITask.from_task(raw_task)
        >>> data = wrapped.acquire()
        >>> raw_task.close()  # Caller must close the task
        """
        _require_nidaqmx()

        # Validation: task must have at least one AI channel
        if len(task.ai_channels) == 0:
            raise ValueError("Task has no AI channels.")

        # Check if task is already running and warn
        try:
            if not task.is_task_done():
                warnings.warn(
                    "Task is already running.",
                    stacklevel=2,
                )
        except Exception:
            # Some task states may not support is_task_done(); ignore
            pass

        # Create instance without calling __init__ (use object.__new__)
        instance = object.__new__(cls)

        # Populate all instance attributes by reading from the live task
        instance.device_list = [d.name for d in task.devices]
        instance.device_product_type = [d.product_type for d in task.devices]

        instance.task = task
        instance.task_name = task.name
        instance.sample_rate = float(task.timing.samp_clk_rate)
        instance.sample_mode = task.timing.samp_quant_samp_mode

        # Synchronized-acquisition attributes — derived from the live task;
        # failures fall back to safe defaults with a warning.
        instance.sample_mode_str = "continuous"
        instance.samples_per_channel = None
        instance.clock_source = None
        try:
            if instance.sample_mode == constants.AcquisitionType.FINITE:
                instance.sample_mode_str = "finite"
                instance.samples_per_channel = int(
                    task.timing.samp_quant_samp_per_chan
                )
        except Exception as exc:
            warnings.warn(
                f"Could not read finite samples_per_channel from task: {exc}",
                stacklevel=2,
            )
        try:
            clk_src = task.timing.samp_clk_src
            instance.clock_source = clk_src if clk_src else None
        except Exception as exc:
            warnings.warn(
                f"Could not read sample clock source from task: {exc}",
                stacklevel=2,
            )

        # Ownership flag: controls whether mutating methods are permitted
        # and whether clear_task() closes the underlying nidaqmx task.
        instance._owns_task = take_ownership

        return instance

