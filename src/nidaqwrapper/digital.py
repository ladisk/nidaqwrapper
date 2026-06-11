"""Digital I/O classes for NI-DAQmx digital input and output.

Provides ``DITask`` and ``DOTask`` classes supporting both
on-demand (single-sample) and clocked (continuous/buffered) operation modes.
Digital channels use NI line specification strings rather than device/channel
indices.

Architecture
------------
Direct delegation: the nidaqmx Task is created immediately in the constructor.
:meth:`add_channel` delegates straight to ``task.di_channels.add_di_chan()``
(or ``do_channels.add_do_chan()``). The nidaqmx Task object is the single
source of truth; no intermediate channel dict is maintained.

Notes
-----
This is a new module — no LDAQ/OpenEOL source to consolidate. The design
follows the patterns from AITask and AOTask for consistency.
"""

from __future__ import annotations

import pathlib
import warnings
from datetime import datetime

import numpy as np

try:
    import nidaqmx
    from nidaqmx import constants

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False

from .base_task import _SAMPLE_MODE_ATTR, BaseTask


def _expand_port_to_line_range(lines: str) -> str:
    """Expand a port-only spec to an explicit line range.

    ``CHAN_PER_LINE`` requires line-level specifications.  When the user
    passes a port-level spec (e.g. ``'Dev1/port0'``), this helper queries
    the device to determine how many lines the port has and returns an
    explicit line range string (e.g. ``'Dev1/port0/line0:7'``).

    If the spec already contains ``'/line'`` it is returned unchanged.
    """
    if "/line" in lines:
        return lines

    # Port-only spec — query device for line count
    # Extract device name (everything before the first '/')
    parts = lines.split("/")
    dev_name = parts[0]

    system = nidaqmx.system.System.local()
    dev = system.devices[dev_name]

    # Find all DI lines belonging to this port
    port_lines = [
        l.name for l in dev.di_lines if l.name.startswith(lines + "/")
    ]
    if not port_lines:
        # Try DO lines (for DOTask)
        port_lines = [
            l.name for l in dev.do_lines if l.name.startswith(lines + "/")
        ]

    if not port_lines:
        return lines  # No expansion possible, let nidaqmx handle the error

    n_lines = len(port_lines)
    return f"{lines}/line0:{n_lines - 1}"


class DITask(BaseTask):
    """Programmatic digital input task supporting on-demand and clocked modes.

    The nidaqmx hardware task is created immediately at construction.
    Channels are added via :meth:`add_channel` which delegates directly to the
    nidaqmx task. Call :meth:`configure` to set up timing, then :meth:`start`
    to begin acquisition.

    Parameters
    ----------
    task_name : str
        Unique name for this digital input task.
    sample_rate : float or None, optional
        Sample rate in Hz. If ``None`` (default), the task operates in
        on-demand mode (single-sample reads). If a float is provided,
        the task operates in clocked (continuous) mode.

    Raises
    ------
    ValueError
        If ``task_name`` already exists in NI MAX.

    Examples
    --------
    On-demand (single-sample) usage:

    >>> di = DITask(task_name='switches')
    >>> di.add_channel('sw1', lines='Dev1/port0/line0:3')
    >>> di.configure()
    >>> data = di.read()
    >>> di.clear_task()

    Clocked (continuous) usage:

    >>> di = DITask(task_name='fast_di', sample_rate=1000)
    >>> di.add_channel('signals', lines='Dev1/port0/line0:7')
    >>> di.configure()
    >>> di.start()
    >>> data = di.acquire()
    >>> di.clear_task()
    """

    _channel_attr = "di_channels"
    _channel_type_label = "DI"

    def __init__(self, task_name: str, sample_rate: float | None = None) -> None:
        self.task_name = task_name
        self.sample_rate = sample_rate
        self.mode: str = "on_demand" if sample_rate is None else "clocked"

        # Synchronized-acquisition settings (introspection only — the live
        # nidaqmx task remains the source of truth). Set by configure().
        self.sample_mode_str: str = "continuous"
        self.samples_per_channel: int | None = None
        self.clock_source: str | None = None

        # Discover connected devices
        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [dev.name for dev in system.devices]

        # Check for duplicate task name in NI MAX before allocating a handle
        existing_tasks = system.tasks.task_names
        if task_name in existing_tasks:
            raise ValueError(
                f"Task name '{task_name}' already exists in NI MAX. "
                "Choose a unique name."
            )

        # Create the nidaqmx task immediately — it is the single source of truth
        self.task = nidaqmx.task.Task(new_task_name=task_name)

        # Track ownership — False when task is externally provided
        self._owns_task: bool = True

    # -- Channel configuration -----------------------------------------------

    def add_channel(self, channel_name: str, lines: str) -> None:
        """Add a digital input channel by line specification.

        Delegates directly to ``task.di_channels.add_di_chan()`` on the
        underlying nidaqmx task.

        Parameters
        ----------
        channel_name : str
            Unique name for this channel.
        lines : str
            NI-DAQmx line specification (e.g. ``'Dev1/port0/line0'``,
            ``'Dev1/port0/line0:3'``, ``'Dev1/port0'``).

        Raises
        ------
        ValueError
            If ``channel_name`` is already used or ``lines`` are already
            assigned to another channel.
        RuntimeError
            If this task was created via :meth:`from_task` (externally provided).
        """
        if not self._owns_task:
            raise RuntimeError(
                "Cannot add channels to an externally-provided task. "
                "Configure channels on the nidaqmx.Task before calling from_task()."
            )

        # Duplicate name detection: check what nidaqmx already knows about
        if channel_name in self.task.channel_names:
            raise ValueError(
                f"Channel name '{channel_name}' already exists. "
                "Use a unique name."
            )

        # Expand port-only spec to explicit line range before duplicate check
        expanded_lines = _expand_port_to_line_range(lines)

        # Duplicate lines detection: iterate the live task channels
        for ch in self.task.di_channels:
            if ch.physical_channel.name == expanded_lines:
                raise ValueError(
                    f"Lines '{lines}' are already assigned to channel "
                    f"'{ch.name}'. Each channel must use unique lines."
                )

        self.task.di_channels.add_di_chan(
            lines=expanded_lines,
            name_to_assign_to_lines=channel_name,
            line_grouping=constants.LineGrouping.CHAN_PER_LINE,
        )

    # -- Task lifecycle -------------------------------------------------------

    def configure(
        self,
        *,
        sample_mode: str = "continuous",
        samples_per_channel: int | None = None,
        clock_source: str | None = None,
    ) -> None:
        """Configure timing for digital input.

        In clocked mode, configures sample-clock timing.  In on-demand
        mode this is a no-op (no timing to configure) and passing any of
        the synchronization parameters raises ``RuntimeError``.  Call
        :meth:`start` afterwards to begin acquisition.  Defaults reproduce
        the classic behavior: continuous acquisition from the onboard clock.

        Parameters
        ----------
        sample_mode : str, optional
            ``'continuous'`` (default) or ``'finite'``.  Finite tasks
            acquire exactly *samples_per_channel* samples and then stop —
            the mode used for hardware-triggered synchronized acquisition.
            Clocked mode only.
        samples_per_channel : int, optional
            Number of samples per channel.  Required (> 0) for
            ``'finite'`` mode.  With ``'continuous'`` mode it is an
            optional buffer-size hint (nidaqmx semantics).  Clocked mode
            only.
        clock_source : str, optional
            Terminal name of an external sample clock (e.g.
            ``'/cDAQ1Mod1/ai/SampleClock'``).  ``None`` (default) uses the
            onboard clock.  Clocked mode only.

        Raises
        ------
        ValueError
            If no channels have been added; if *sample_mode* is not
            ``'continuous'``/``'finite'``; if ``'finite'`` is requested
            without a positive integer *samples_per_channel*; or if
            *clock_source* is not a non-empty string.
        RuntimeError
            If this task was created via :meth:`from_task` (externally
            provided), or if any synchronization parameter is passed in
            on-demand mode.
        """
        self._check_start_preconditions()

        if self.mode != "clocked":
            if (
                sample_mode != "continuous"
                or samples_per_channel is not None
                or clock_source is not None
            ):
                raise RuntimeError(
                    "Synchronization parameters (sample_mode, "
                    "samples_per_channel, clock_source) require clocked "
                    "mode. Create the DITask with a sample_rate to enable "
                    "clocked acquisition."
                )
            return  # on-demand: nothing to configure

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

        timing_kwargs: dict[str, Any] = {
            "rate": self.sample_rate,
            "sample_mode": getattr(
                constants.AcquisitionType, _SAMPLE_MODE_ATTR[sample_mode]
            ),
        }
        if samples_per_channel is not None:
            timing_kwargs["samps_per_chan"] = samples_per_channel
        if clock_source is not None:
            timing_kwargs["source"] = clock_source

        self.task.timing.cfg_samp_clk_timing(**timing_kwargs)

        # Store for introspection — the live task remains the source of truth
        self.sample_mode_str = sample_mode
        self.samples_per_channel = samples_per_channel
        self.clock_source = clock_source

    # -- Data acquisition ----------------------------------------------------

    def read(self) -> np.ndarray:
        """Read a single sample from all digital input lines (on-demand).

        Returns
        -------
        numpy.ndarray
            Array of bool-like values, one per line.
        """
        data = self.task.read()

        if isinstance(data, bool):
            return np.array([data])
        return np.array(data)

    def acquire(self, n_samples: int | None = None) -> np.ndarray:
        """Read samples from the input buffer (clocked mode).

        Parameters
        ----------
        n_samples : int, optional
            Number of samples per channel to read.  If provided, the call
            **blocks** until exactly *n_samples* are available — suitable
            for scripts and notebooks.  If ``None`` (default), drains every
            sample currently in the buffer without blocking
            (``READ_ALL_AVAILABLE``) — suitable for acquisition loops.

            On a **finite** task (``configure(sample_mode='finite', ...)``)
            ``acquire()`` with the default ``None`` blocks until the finite
            acquisition completes and returns all
            ``samples_per_channel`` samples — this is the synchronization
            mechanism for hardware-triggered multi-task acquisition.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_lines)``.

        Raises
        ------
        RuntimeError
            If called in on-demand mode.
        """
        if self.mode != "clocked":
            raise RuntimeError(
                "acquire() requires clocked mode. "
                "Create DITask with a sample_rate to enable "
                "continuous reading."
            )

        count = constants.READ_ALL_AVAILABLE if n_samples is None else n_samples
        data = self.task.read(number_of_samples_per_channel=count)

        if not data:
            return np.array([]).reshape(0, 0)

        arr = np.array(data)

        # nidaqmx returns (n_lines, n_samples) for multi-line;
        # single-line returns a flat list → reshape to (n_samples, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr.T

    # -- TOML config persistence ---------------------------------------------

    def save_config(self, path: str | pathlib.Path) -> None:
        """Serialise the task configuration to a TOML file.

        Reads channel information directly from ``self.task.di_channels``
        (the nidaqmx Task is the single source of truth).  Writes a
        human-readable TOML file that can be loaded back with
        :meth:`from_config` to recreate the same task on any compatible
        hardware.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path. The file is created or overwritten.

        Notes
        -----
        TOML is generated with simple string formatting — no third-party
        library is required for writing. The ``sample_rate`` key is only
        written for clocked-mode tasks; on-demand tasks omit it entirely
        so that ``from_config`` correctly restores the mode.  The
        ``lines`` value for each channel is read from
        ``channel.physical_channel.name``.
        """
        lines: list[str] = []

        # Header comment with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"# Generated by nidaqwrapper 0.1.0 on {timestamp}")
        lines.append("")

        # [task] section
        lines.append("[task]")
        lines.append(f'name = "{self.task_name}"')
        lines.append('type = "digital_input"')
        if self.sample_rate is not None:
            lines.append(f"sample_rate = {self.sample_rate}")
        lines.append("")

        # [[channels]] entries — read from live task channels
        for ch in self.task.di_channels:
            lines.append("[[channels]]")
            lines.append(f'name = "{ch.name}"')
            lines.append(f'lines = "{ch.physical_channel.name}"')
            lines.append("")

        pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_config(cls, path: str | pathlib.Path) -> DITask:
        """Create a :class:`DITask` from a TOML configuration file.

        Reads the TOML file produced by :meth:`save_config`, constructs a new
        task, and calls :meth:`add_channel` for every ``[[channels]]`` entry.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file produced by :meth:`save_config`.

        Returns
        -------
        DITask
            A fully configured task (channels added, not yet started).

        Raises
        ------
        ValueError
            If the ``[task]`` section is absent.
        tomllib.TOMLDecodeError
            On syntactically invalid TOML (propagated from the parser).
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

        task_section = data["task"]
        sample_rate = task_section.get("sample_rate", None)

        task = cls(task_section["name"], sample_rate=sample_rate)

        for ch in data.get("channels", []):
            task.add_channel(channel_name=ch["name"], lines=ch["lines"])

        return task

    @classmethod
    def from_task(
        cls, task: nidaqmx.task.Task, take_ownership: bool = False
    ) -> DITask:
        """Wrap an externally-created nidaqmx.Task in a DITask.

        This classmethod provides an escape hatch for advanced users who need
        to configure task properties not exposed by the wrapper API. The task
        must already have DI channels configured before calling this method.

        Parameters
        ----------
        task : nidaqmx.task.Task
            A pre-configured nidaqmx.Task with at least one DI channel.
        take_ownership : bool, optional
            If ``True``, the wrapper takes ownership of the task and all
            mutating methods (:meth:`add_channel`, :meth:`configure`,
            :meth:`start`, :meth:`clear_task`) are permitted.  The task
            will be closed when :meth:`clear_task` is called.  Default is
            ``False`` (original behaviour: task is not owned, mutating
            methods raise ``RuntimeError``).

        Returns
        -------
        DITask
            A DITask instance wrapping the provided task.

        Raises
        ------
        ValueError
            If the task has no DI channels configured.

        Warnings
        --------
        Emits a UserWarning if the task is already running.

        Notes
        -----
        When created via this method with ``take_ownership=False``:

        - :meth:`add_channel`, :meth:`configure`, and :meth:`start` will raise RuntimeError
        - :meth:`clear_task` and :meth:`__exit__` will NOT close the task
        - The caller retains full ownership and must manage the task lifecycle

        Examples
        --------
        >>> import nidaqmx
        >>> task = nidaqmx.Task()
        >>> task.di_channels.add_di_chan("Dev1/port0/line0:3")
        >>> task.timing.cfg_samp_clk_timing(rate=1000)
        >>> di = DITask.from_task(task)
        >>> data = di.acquire()
        >>> task.close()  # Caller must close
        """
        if len(task.di_channels) == 0:
            raise ValueError("Task has no DI channels.")

        # Warn if task is already running (check if task.is_task_done() exists)
        try:
            if hasattr(task, "is_task_done") and not task.is_task_done():
                warnings.warn("Task is already running.", stacklevel=2)
        except Exception:
            # Suppress errors from checking task state
            pass

        # Create instance without calling __init__ (bypass constructor checks)
        instance = object.__new__(cls)

        # Populate attributes from the live task
        instance.task = task
        instance.task_name = task.name
        # channel_list and number_of_ch are properties that read from task

        # Detect mode from timing configuration
        try:
            sample_rate = task.timing.samp_clk_rate
            if sample_rate and sample_rate > 0:
                instance.sample_rate = sample_rate
                instance.mode = "clocked"
            else:
                instance.sample_rate = None
                instance.mode = "on_demand"
        except Exception:
            # No timing configured — assume on-demand
            instance.sample_rate = None
            instance.mode = "on_demand"

        # Synchronized-acquisition attributes: safe defaults only.  Deriving
        # them from an external task is in scope for AITask.from_task() only.
        instance.sample_mode_str = "continuous"
        instance.samples_per_channel = None
        instance.clock_source = None

        # Derive device info from the task itself
        instance.device_list = [dev.name for dev in task.devices]

        # Set ownership: True transfers full control, False preserves external ownership
        instance._owns_task = take_ownership

        return instance


class DOTask(BaseTask):
    """Programmatic digital output task supporting on-demand and clocked modes.

    The nidaqmx hardware task is created immediately at construction.
    Channels are added via :meth:`add_channel` which delegates directly to the
    nidaqmx task. Call :meth:`configure` to set up timing, then :meth:`start`
    to begin output.

    Parameters
    ----------
    task_name : str
        Unique name for this digital output task.
    sample_rate : float or None, optional
        Sample rate in Hz. If ``None`` (default), the task operates in
        on-demand mode (single-sample writes). If a float is provided,
        the task operates in clocked (continuous) mode.

    Raises
    ------
    ValueError
        If ``task_name`` already exists in NI MAX.

    Examples
    --------
    On-demand (single-sample) usage:

    >>> do = DOTask(task_name='leds')
    >>> do.add_channel('led_1', lines='Dev1/port1/line0')
    >>> do.configure()
    >>> do.write(True)
    >>> do.clear_task()

    Clocked (continuous) usage:

    >>> do = DOTask(task_name='pattern_gen', sample_rate=1000)
    >>> do.add_channel('lines', lines='Dev1/port1/line0:3')
    >>> do.configure()
    >>> do.start()
    >>> do.write_continuous(data)
    >>> do.clear_task()
    """

    _channel_attr = "do_channels"
    _channel_type_label = "DO"

    def __init__(self, task_name: str, sample_rate: float | None = None) -> None:
        self.task_name = task_name
        self.sample_rate = sample_rate
        self.mode: str = "on_demand" if sample_rate is None else "clocked"

        system = nidaqmx.system.System.local()
        self.device_list: list[str] = [dev.name for dev in system.devices]

        existing_tasks = system.tasks.task_names
        if task_name in existing_tasks:
            raise ValueError(
                f"Task name '{task_name}' already exists in NI MAX. "
                "Choose a unique name."
            )

        # Create the nidaqmx task immediately — it is the single source of truth
        self.task = nidaqmx.task.Task(new_task_name=task_name)

        # Track ownership — False when task is externally provided
        self._owns_task: bool = True

    # -- Channel configuration -----------------------------------------------

    def add_channel(self, channel_name: str, lines: str) -> None:
        """Add a digital output channel by line specification.

        Delegates directly to ``task.do_channels.add_do_chan()`` on the
        underlying nidaqmx task.

        Parameters
        ----------
        channel_name : str
            Unique name for this channel.
        lines : str
            NI-DAQmx line specification (e.g. ``'Dev1/port1/line0'``,
            ``'Dev1/port1/line0:7'``, ``'Dev1/port1'``).

        Raises
        ------
        ValueError
            If ``channel_name`` is already used or ``lines`` are already
            assigned to another channel.
        RuntimeError
            If this task was created via :meth:`from_task` (externally provided).
        """
        if not self._owns_task:
            raise RuntimeError(
                "Cannot add channels to an externally-provided task. "
                "Configure channels on the nidaqmx.Task before calling from_task()."
            )

        # Duplicate name detection: check what nidaqmx already knows about
        if channel_name in self.task.channel_names:
            raise ValueError(
                f"Channel name '{channel_name}' already exists. "
                "Use a unique name."
            )

        # Expand port-only spec to explicit line range before duplicate check
        expanded_lines = _expand_port_to_line_range(lines)

        # Duplicate lines detection: iterate the live task channels
        for ch in self.task.do_channels:
            if ch.physical_channel.name == expanded_lines:
                raise ValueError(
                    f"Lines '{lines}' are already assigned to channel "
                    f"'{ch.name}'. Each channel must use unique lines."
                )

        self.task.do_channels.add_do_chan(
            lines=expanded_lines,
            name_to_assign_to_lines=channel_name,
            line_grouping=constants.LineGrouping.CHAN_PER_LINE,
        )

    # -- Task lifecycle -------------------------------------------------------

    def configure(self) -> None:
        """Configure timing for digital output.

        In clocked mode, configures sample-clock timing for continuous
        generation.  In on-demand mode, this is a no-op (no timing to
        configure).  Call :meth:`start` afterwards to begin generation.

        Raises
        ------
        ValueError
            If no channels have been added to the task.
        RuntimeError
            If this task was created via :meth:`from_task` (externally provided).
        """
        self._check_start_preconditions()

        if self.mode == "clocked":
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=constants.AcquisitionType.CONTINUOUS,
            )

    # -- Signal output -------------------------------------------------------

    def write(self, data: bool | int | list | np.ndarray) -> None:
        """Write a single sample to digital output lines (on-demand).

        Parameters
        ----------
        data : bool, int, list, or numpy.ndarray
            Data to write. For a single line, pass a bool or int.
            For multiple lines, pass a list or array of values.
        """
        if isinstance(data, np.ndarray):
            data = [bool(v) for v in data]
        elif isinstance(data, list):
            data = [bool(v) for v in data]
        elif isinstance(data, int) and not isinstance(data, bool):
            data = bool(data)
        self.task.write(data)

    def write_continuous(self, data: np.ndarray) -> None:
        """Write buffered data to digital output lines (clocked mode).

        Parameters
        ----------
        data : numpy.ndarray
            For multi-line: shape ``(n_samples, n_lines)``, transposed
            internally to ``(n_lines, n_samples)`` for nidaqmx.
            For single-line: 1D array of shape ``(n_samples,)``.

        Raises
        ------
        RuntimeError
            If called in on-demand mode.
        """
        if self.mode != "clocked":
            raise RuntimeError(
                "write_continuous() requires clocked mode. "
                "Create DOTask with a sample_rate to enable "
                "continuous writing."
            )

        if data.ndim == 2:
            # Transpose from (n_samples, n_lines) to (n_lines, n_samples)
            transposed = data.T
            write_data = [[bool(v) for v in row] for row in transposed]
        else:
            write_data = [bool(v) for v in data]

        self.task.write(write_data, auto_start=True)

    # -- TOML config persistence ---------------------------------------------

    def save_config(self, path: str | pathlib.Path) -> None:
        """Serialise the task configuration to a TOML file.

        Reads channel information directly from ``self.task.do_channels``
        (the nidaqmx Task is the single source of truth).  Writes a
        human-readable TOML file that can be loaded back with
        :meth:`from_config` to recreate the same task on any compatible
        hardware.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path. The file is created or overwritten.

        Notes
        -----
        TOML is generated with simple string formatting — no third-party
        library is required for writing. The ``sample_rate`` key is only
        written for clocked-mode tasks; on-demand tasks omit it entirely
        so that ``from_config`` correctly restores the mode.  The
        ``lines`` value for each channel is read from
        ``channel.physical_channel.name``.
        """
        lines: list[str] = []

        # Header comment with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"# Generated by nidaqwrapper 0.1.0 on {timestamp}")
        lines.append("")

        # [task] section
        lines.append("[task]")
        lines.append(f'name = "{self.task_name}"')
        lines.append('type = "digital_output"')
        if self.sample_rate is not None:
            lines.append(f"sample_rate = {self.sample_rate}")
        lines.append("")

        # [[channels]] entries — read from live task channels
        for ch in self.task.do_channels:
            lines.append("[[channels]]")
            lines.append(f'name = "{ch.name}"')
            lines.append(f'lines = "{ch.physical_channel.name}"')
            lines.append("")

        pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_config(cls, path: str | pathlib.Path) -> DOTask:
        """Create a :class:`DOTask` from a TOML configuration file.

        Reads the TOML file produced by :meth:`save_config`, constructs a new
        task, and calls :meth:`add_channel` for every ``[[channels]]`` entry.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file produced by :meth:`save_config`.

        Returns
        -------
        DOTask
            A fully configured task (channels added, not yet started).

        Raises
        ------
        ValueError
            If the ``[task]`` section is absent.
        tomllib.TOMLDecodeError
            On syntactically invalid TOML (propagated from the parser).
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

        task_section = data["task"]
        sample_rate = task_section.get("sample_rate", None)

        task = cls(task_section["name"], sample_rate=sample_rate)

        for ch in data.get("channels", []):
            task.add_channel(channel_name=ch["name"], lines=ch["lines"])

        return task

    @classmethod
    def from_task(
        cls, task: nidaqmx.task.Task, take_ownership: bool = False
    ) -> DOTask:
        """Wrap an externally-created nidaqmx.Task in a DOTask.

        This classmethod provides an escape hatch for advanced users who need
        to configure task properties not exposed by the wrapper API. The task
        must already have DO channels configured before calling this method.

        Parameters
        ----------
        task : nidaqmx.task.Task
            A pre-configured nidaqmx.Task with at least one DO channel.
        take_ownership : bool, optional
            If ``True``, the wrapper takes ownership of the task and all
            mutating methods (:meth:`add_channel`, :meth:`configure`,
            :meth:`start`, :meth:`clear_task`) are permitted.  The task
            will be closed when :meth:`clear_task` is called.  Default is
            ``False`` (original behaviour: task is not owned, mutating
            methods raise ``RuntimeError``).

        Returns
        -------
        DOTask
            A DOTask instance wrapping the provided task.

        Raises
        ------
        ValueError
            If the task has no DO channels configured.

        Warnings
        --------
        Emits a UserWarning if the task is already running.

        Notes
        -----
        When created via this method with ``take_ownership=False``:

        - :meth:`add_channel`, :meth:`configure`, and :meth:`start` will raise RuntimeError
        - :meth:`clear_task` and :meth:`__exit__` will NOT close the task
        - The caller retains full ownership and must manage the task lifecycle

        Examples
        --------
        >>> import nidaqmx
        >>> task = nidaqmx.Task()
        >>> task.do_channels.add_do_chan("Dev1/port1/line0:3")
        >>> task.timing.cfg_samp_clk_timing(rate=1000)
        >>> do = DOTask.from_task(task)
        >>> do.write_continuous(data)
        >>> task.close()  # Caller must close
        """
        if len(task.do_channels) == 0:
            raise ValueError("Task has no DO channels.")

        # Warn if task is already running
        try:
            if hasattr(task, "is_task_done") and not task.is_task_done():
                warnings.warn("Task is already running.", stacklevel=2)
        except Exception:
            # Suppress errors from checking task state
            pass

        # Create instance without calling __init__
        instance = object.__new__(cls)

        # Populate attributes from the live task
        instance.task = task
        instance.task_name = task.name
        # channel_list and number_of_ch are properties that read from task

        # Detect mode from timing configuration
        try:
            sample_rate = task.timing.samp_clk_rate
            if sample_rate and sample_rate > 0:
                instance.sample_rate = sample_rate
                instance.mode = "clocked"
            else:
                instance.sample_rate = None
                instance.mode = "on_demand"
        except Exception:
            # No timing configured — assume on-demand
            instance.sample_rate = None
            instance.mode = "on_demand"

        # Derive device info from the task itself
        instance.device_list = [dev.name for dev in task.devices]

        # Set ownership: True transfers full control, False preserves external ownership
        instance._owns_task = take_ownership

        return instance
