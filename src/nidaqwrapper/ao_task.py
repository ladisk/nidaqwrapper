"""AOTask — programmatic analog output task configuration.

Provides the :class:`AOTask` class for creating and managing NI-DAQmx
analog output tasks.  Channels are added programmatically (not from NI MAX),
and the output buffer supports continuous regeneration.

Architecture
------------
Direct delegation: the nidaqmx Task is created immediately in the
constructor. :meth:`add_channel` delegates straight to
``task.ao_channels.add_ao_voltage_chan()``.  The nidaqmx Task object is the
single source of truth; no intermediate channel dict is maintained.

Data Format
-----------
The public API accepts signal data in ``(n_samples, n_channels)`` format.
Internally, nidaqmx expects ``(n_channels, n_samples)``, so :meth:`generate`
transposes 2-D arrays automatically.  1-D arrays (single channel) are passed
through unchanged.

Examples
--------
>>> task = AOTask("sig_gen", sample_rate=10000)
>>> task.add_channel("ao_0", device='Dev1', channel_ind=0)
>>> task.start()
>>> task.generate(signal_array)
>>> task.clear_task()

Or as a context manager::

    with AOTask("sig_gen", 10000) as task:
        task.add_channel("ao_0", device='Dev1', channel_ind=0)
        task.start()
        task.generate(signal_array)
"""

from __future__ import annotations

import pathlib
import warnings
from datetime import datetime
from typing import Any

import numpy as np

try:
    import nidaqmx
    from nidaqmx import constants
    from nidaqmx.errors import DaqError

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False
    DaqError = Exception  # type: ignore[misc,assignment]

from .base_task import BaseTask


class AOTask(BaseTask):
    """Programmatic analog output task for NI-DAQmx devices.

    The nidaqmx hardware task is created immediately at construction.
    Channels are added via :meth:`add_channel` which delegates directly
    to the nidaqmx task.  Call :meth:`configure` to set up timing,
    then :meth:`start` to begin output generation.

    Parameters
    ----------
    task_name : str
        Unique name for the output task.  Must not collide with tasks
        already saved in NI MAX.
    sample_rate : float
        Output sample rate in Hz.
    samples_per_channel : int, optional
        Buffer size per channel.  Defaults to ``5 * int(sample_rate)``
        (5 seconds of buffer).

    Raises
    ------
    ValueError
        If ``task_name`` already exists in NI MAX.
    """

    _channel_attr = "ao_channels"
    _channel_type_label = "AO"

    def __init__(
        self,
        task_name: str,
        sample_rate: float,
        samples_per_channel: int | None = None,
    ) -> None:
        self.task_name = task_name
        self.sample_rate = sample_rate

        if samples_per_channel is None:
            self.samples_per_channel = 5 * int(sample_rate)
        else:
            self.samples_per_channel = int(samples_per_channel)

        self.sample_mode = constants.AcquisitionType.CONTINUOUS

        # External sample clock terminal (introspection only — the live
        # nidaqmx task remains the source of truth). Set by configure().
        self.clock_source: str | None = None

        # Discover connected devices
        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [dev.name for dev in system.devices]
        self.device_product_type: list[str] = [
            dev.product_type for dev in system.devices
        ]

        # Reject duplicate task names in NI MAX before allocating a handle
        if task_name in system.tasks.task_names:
            raise ValueError(
                f"Task '{task_name}' already exists in NI MAX. "
                "Choose a different name."
            )

        # Create the nidaqmx task immediately — it is the single source of truth
        self.task = nidaqmx.task.Task(new_task_name=task_name)

        # Track ownership — True when we created the task, False when wrapping external
        self._owns_task = True

    # -- Channel configuration -----------------------------------------------

    def add_channel(
        self,
        channel_name: str,
        device: str,
        channel_ind: int,
        min_val: float = -10.0,
        max_val: float = 10.0,
    ) -> None:
        """Add an analog output voltage channel to the task.

        The channel is configured immediately on the underlying nidaqmx task.

        Parameters
        ----------
        channel_name : str
            Logical name for the channel.
        device : str
            NI-DAQmx device name string (e.g. ``'Dev1'``, ``'cDAQ1Mod1'``,
            ``'SimDev1'``).  Must be a non-empty string.  The driver
            validates the device name at channel-creation time and raises
            ``DaqError`` if the device does not exist.
        channel_ind : int
            AO channel number on the device (e.g. 0 for ``ao0``).
        min_val : float, optional
            Minimum output voltage, by default -10.0.
        max_val : float, optional
            Maximum output voltage, by default 10.0.

        Raises
        ------
        ValueError
            If ``channel_name`` is a duplicate, the ``(device, channel_ind)``
            pair is already used, or ``device`` is an empty string.
        RuntimeError
            If this task wraps an externally-provided nidaqmx.Task
            (created via :meth:`from_task`).
        nidaqmx.errors.DaqError
            If the driver rejects the channel configuration.  After the
            channel is added, sample-clock timing is configured with the
            task's current settings and the driver validates the whole
            task (``TASK_VERIFY``), so invalid parameters (e.g. an
            out-of-range *min_val*/*max_val*, DaqError -200077) raise
            here instead of at a later operation.

        Notes
        -----
        NI-DAQmx cannot remove channels from a task.  If ``add_channel()``
        raises after the driver accepted the channel but validation
        failed, the task still contains the invalid channel — recreate
        the task (``clear_task()`` and build a new one) before retrying.
        """
        # Block channel addition for externally-provided tasks
        if not self._owns_task:
            raise RuntimeError(
                "Cannot add channels to an externally-provided task. "
                "Configure channels on the nidaqmx.Task before calling from_task()."
            )

        # Duplicate name detection: check what nidaqmx already knows about
        if channel_name in self.task.channel_names:
            raise ValueError(
                f"Channel with duplicate name '{channel_name}' already exists."
            )

        # Validate device is a non-empty string
        if not device or not isinstance(device, str):
            raise ValueError("device must be a non-empty string")

        physical_channel = f"{device}/ao{channel_ind}"

        # Duplicate physical channel detection is delegated to the driver:
        # it natively rejects duplicate physical AO channels at add time
        # with DaqError -200371, mapped here to the documented ValueError.
        # (No channel iteration — iterating channel objects triggers
        # implicit verification, which raises -201087 on untimed tasks on
        # delta-sigma devices; GH issue #5.)
        try:
            self.task.ao_channels.add_ao_voltage_chan(
                physical_channel=physical_channel,
                name_to_assign_to_channel=channel_name,
                min_val=min_val,
                max_val=max_val,
            )
        except DaqError as exc:
            if getattr(exc, "error_code", None) == -200371:
                raise ValueError(
                    f"Physical channel ao{channel_ind} on device "
                    f"'{device}' is already in use."
                ) from exc
            raise

        # -- Eager validation (GH issues #5, #6) -----------------------------
        # Configure sample-clock timing with the task's current stored
        # settings, then force driver validation.  Timing first: verifying
        # an untimed task would itself raise -201087 on delta-sigma devices.
        # Validation errors (e.g. -200077 out-of-range limits) surface here,
        # on the offending add_channel() call.  DaqErrors propagate
        # unchanged.  Regeneration mode remains a configure() concern.
        self._apply_timing()
        self.task.control(constants.TaskMode.TASK_VERIFY)

    def _apply_timing(self) -> None:
        """Apply the stored sample-clock timing settings to the live task.

        Shared by :meth:`configure` and :meth:`add_channel` so the timing
        call cannot diverge between the two paths.  Reads the stored
        attributes (``sample_rate``, ``samples_per_channel``,
        ``clock_source``); AO output is always continuous.
        """
        timing_kwargs: dict[str, Any] = {
            "rate": self.sample_rate,
            "sample_mode": constants.AcquisitionType.CONTINUOUS,
            "samps_per_chan": self.samples_per_channel,
        }
        if self.clock_source is not None:
            timing_kwargs["source"] = self.clock_source

        self.task.timing.cfg_samp_clk_timing(**timing_kwargs)

    # -- Task lifecycle -------------------------------------------------------

    def configure(self, *, clock_source: str | None = None) -> None:
        """Configure sample-clock timing for continuous output generation.

        Sets up the sample-clock timing on the nidaqmx task, enables
        buffer regeneration, and validates that the driver accepted the
        requested sample rate.  Call :meth:`start` afterwards to begin
        generation.

        Parameters
        ----------
        clock_source : str, optional
            Terminal name of an external sample clock (e.g.
            ``'/Dev1/ai/SampleClock'``).  ``None`` (default) uses the
            onboard clock.  When set, the rate-coercion check is skipped —
            the reported rate is whatever the external clock provides, so
            *sample_rate* must match the master task's rate (it sizes
            buffers and timeouts).

        Raises
        ------
        ValueError
            If no channels have been added; if *clock_source* is not a
            non-empty string; or if the hardware driver coerces the
            sample rate while using the onboard clock (some devices only
            support discrete rates).
        RuntimeError
            If this task wraps an externally-provided nidaqmx.Task
            (created via :meth:`from_task`).
        """
        self._check_start_preconditions()

        if clock_source is not None and (
            not isinstance(clock_source, str) or not clock_source
        ):
            raise ValueError(
                f"clock_source must be a non-empty terminal name string "
                f"(e.g. '/Dev1/ai/SampleClock') or None, got {clock_source!r}."
            )

        # Store for introspection — the live task remains the source of truth
        self.clock_source = clock_source

        self._apply_timing()

        self.task._out_stream.regen_mode = constants.RegenerationMode.ALLOW_REGENERATION

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

    # -- Signal generation ---------------------------------------------------

    def generate(self, signal: np.ndarray) -> None:
        """Write signal data to the output task.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal data in public format:

            - ``(n_samples, n_channels)`` — multi-channel 2-D array
            - ``(n_samples,)`` — single-channel 1-D array
            - ``(n_samples, 1)`` — single-channel 2-D array

            2-D arrays are transposed to ``(n_channels, n_samples)``
            internally before writing.  A C-contiguous copy is made to
            satisfy the nidaqmx C layer requirement (``data.T`` returns
            Fortran-order).
        """
        if signal.ndim == 2:
            data = np.ascontiguousarray(signal.T)
            # nidaqmx requires a 1-D array for single-channel tasks;
            # a (1, N) 2-D array triggers a channel-count validation error.
            if data.shape[0] == 1:
                data = data[0]
        else:
            data = signal

        self.task.write(data, auto_start=True)

    # -- TOML config persistence ---------------------------------------------

    def save_config(self, path: str | pathlib.Path) -> None:
        """Serialise the task configuration to a TOML file.

        Reads channel information directly from ``self.task.ao_channels``
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
        always written (read from ``channel.ao_min`` / ``channel.ao_max``).

        Examples
        --------
        >>> task.save_config("/tmp/signal_gen.toml")
        """
        # Build device alias map from unique device names in channel order
        device_names_seen: list[str] = []
        for ch in self.task.ao_channels:
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
        lines.append(f"samples_per_channel = {self.samples_per_channel}")
        lines.append('type = "output"')
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
        for ch in self.task.ao_channels:
            phys = ch.physical_channel.name
            dev_name = phys.rsplit("/", 1)[0]
            ch_spec = phys.rsplit("/", 1)[1]
            channel_ind = int(ch_spec.lstrip("ao"))
            alias = device_to_alias[dev_name]

            lines.append("[[channels]]")
            lines.append(f'name = "{ch.name}"')
            lines.append(f'device = "{alias}"')
            lines.append(f"channel = {channel_ind}")
            lines.append(f"min_val = {ch.ao_min}")
            lines.append(f"max_val = {ch.ao_max}")
            lines.append("")

        pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_config(cls, path: str | pathlib.Path) -> AOTask:
        """Create an :class:`AOTask` from a TOML configuration file.

        Reads the TOML file produced by :meth:`save_config`, constructs
        a new task, and calls :meth:`add_channel` for every ``[[channels]]``
        entry.  Device aliases are resolved to system device indices.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file.

        Returns
        -------
        AOTask
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
        >>> task = AOTask.from_config("/tmp/signal_gen.toml")
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

        samples_per_channel = task_section.get("samples_per_channel")

        task = cls(
            task_section["name"],
            sample_rate=task_section["sample_rate"],
            samples_per_channel=samples_per_channel,
        )

        for ch in data.get("channels", []):
            alias = ch["device"]
            if alias not in alias_to_name:
                raise ValueError(
                    f"Channel '{ch['name']}' references unknown device alias "
                    f"'{alias}'. Available aliases: {list(alias_to_name)}"
                )

            device_name = alias_to_name[alias]

            task.add_channel(
                channel_name=ch["name"],
                device=device_name,
                channel_ind=ch["channel"],
                min_val=ch.get("min_val", -10.0),
                max_val=ch.get("max_val", 10.0),
            )

        return task

    @classmethod
    def from_task(
        cls, task: nidaqmx.task.Task, take_ownership: bool = False
    ) -> AOTask:
        """Wrap a pre-created nidaqmx.Task object.

        This provides an escape hatch for advanced users who need to configure
        task properties not exposed by the wrapper API.  The task must already
        have AO channels configured.

        Parameters
        ----------
        task : nidaqmx.task.Task
            An existing nidaqmx Task object with AO channels configured.
            Timing configuration and task state are preserved.
        take_ownership : bool, optional
            If ``True``, the wrapper takes ownership of the task and all
            mutating methods (:meth:`add_channel`, :meth:`configure`,
            :meth:`start`, :meth:`clear_task`) are permitted.  The task
            will be closed when :meth:`clear_task` is called.  Default is
            ``False`` (original behaviour: task is not owned, mutating
            methods raise ``RuntimeError``).

        Returns
        -------
        AOTask
            A wrapper instance that delegates to the provided task.

        Raises
        ------
        ValueError
            If ``task.ao_channels`` is empty (no AO channels configured).

        Warnings
        --------
        If the task is already running, a warning is issued.

        Notes
        -----
        When wrapping an external task with ``take_ownership=False``:

        - :meth:`add_channel` is blocked and raises ``RuntimeError``
        - :meth:`configure` and :meth:`start` are blocked and raise ``RuntimeError``
        - :meth:`clear_task` and ``__exit__`` do NOT close the task
        - The caller remains responsible for calling ``task.close()``

        Examples
        --------
        >>> import nidaqmx
        >>> task = nidaqmx.Task("my_task")
        >>> task.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        >>> wrapper = AOTask.from_task(task)
        >>> wrapper.generate(signal_data)
        >>> task.close()  # Caller must close
        """
        # Validate that the task has AO channels
        if not task.ao_channels or len(task.ao_channels) == 0:
            raise ValueError("Task has no AO channels.")

        # Warn if task is already running
        try:
            # Check if task is running by accessing is_task_done()
            # A task that hasn't started will raise or return True
            if hasattr(task, 'is_task_done') and not task.is_task_done():
                warnings.warn("Task is already running.", stacklevel=2)
        except Exception:
            # If we can't determine state, assume not running
            pass

        # Create instance without calling __init__
        instance = object.__new__(cls)

        # Populate attributes from the live task
        instance.task = task
        instance.task_name = task.name

        # Reading samp_clk_rate forces implicit task verification; on an
        # untimed task on a device that rejects on-demand timing
        # (delta-sigma) this raises DaqError -201087 (GH issue #5).
        try:
            instance.sample_rate = task.timing.samp_clk_rate
        except DaqError as exc:
            if getattr(exc, "error_code", None) == -201087:
                instance.sample_rate = None
                warnings.warn(
                    "Could not read the sample rate from the task: no "
                    "sample-clock timing is configured and the device "
                    "rejects on-demand timing (DaqError -201087). "
                    "sample_rate is set to None — call configure() after "
                    "setting sample_rate, or configure timing on the "
                    "nidaqmx task before wrapping.",
                    stacklevel=2,
                )
            else:
                raise
        instance.samples_per_channel = task.timing.samp_quant_samp_per_chan
        instance.sample_mode = task.timing.samp_quant_samp_mode

        # External clock terminal — derived from the live task; failures
        # fall back to the safe default (None) with a warning.
        instance.clock_source = None
        try:
            clk_src = task.timing.samp_clk_src
            instance.clock_source = clk_src if clk_src else None
        except Exception as exc:
            warnings.warn(
                f"Could not read sample clock source from task: {exc}",
                stacklevel=2,
            )

        # Derive device info from the task itself
        instance.device_list = [dev.name for dev in task.devices]
        instance.device_product_type = [dev.product_type for dev in task.devices]

        # Set ownership: True transfers full control, False preserves external ownership
        instance._owns_task = take_ownership

        return instance
