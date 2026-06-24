"""Multi-task synchronized NI-DAQmx acquisition.

Provides the :class:`MultiHandler` class, which consolidates OpenEOL's
advanced implementation with four bug fixes applied.  It supports
synchronizing multiple NI-DAQmx tasks via hardware triggers (FINITE mode)
or a single-task software trigger using pyTrigger (CONTINUOUS mode).

Bug Fixes Applied (relative to OpenEOL base_advanced.py)
---------------------------------------------------------
1. Error codes compared as integers, not strings (was ``== "-200088"``).
2. ``trigger_type`` defaults to ``'software'`` in ``__init__`` — avoids
   ``AttributeError`` in ``_validate_acquisition_mode`` when no hardware
   triggers are present.
3. ``task.start()`` called before the read loop in
   ``acquire_with_software_trigger`` — OpenEOL omitted this, causing reads
   from a task that was never started.
4. ``configure()`` uses ``None`` default arguments instead of mutable ``[]``
   defaults, preventing shared-state bugs across repeated calls.

Notes
-----
MultiHandler is **not** a subclass of DAQHandler (NFR-8.2).  It is an
independent class that operates on pre-configured nidaqmx tasks.
"""

from __future__ import annotations

import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import numpy as np

from .utils import get_connected_devices, get_task_by_name
from .ai_task import AITask
from .ao_task import AOTask

try:
    import nidaqmx
    from nidaqmx.constants import READ_ALL_AVAILABLE
    from nidaqmx.errors import DaqError

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False
    DaqError = Exception  # type: ignore[misc,assignment]

try:
    from pyTrigger import pyTrigger

    _PYTRIGGER_AVAILABLE = True
except ImportError:
    _PYTRIGGER_AVAILABLE = False
    pyTrigger = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Error codes (integers — bug fix #1)
# ---------------------------------------------------------------------------

_DAQ_ERROR_INVALID_TASK = -200088
_DAQ_ERROR_NO_CHANNELS = -200478


class MultiHandler:
    """Multi-task synchronized NI-DAQmx acquisition.

    Supports multiple input and output tasks with hardware trigger
    synchronization (FINITE acquisition mode) or single-task software
    triggering via pyTrigger (CONTINUOUS acquisition mode).

    Parameters
    ----------
    None
        All configuration is done via :meth:`configure`.

    Attributes
    ----------
    input_tasks : list
        Resolved ``nidaqmx.task.Task`` objects for analog input.
    output_tasks : list
        Resolved ``nidaqmx.task.Task`` objects for analog output.
    trigger_type : str
        Either ``'software'`` or ``'hardware'``.  Defaults to ``'software'``
        (bug fix #2 — OpenEOL omitted this default).
    required_devices : set
        Device names that must be present for :meth:`ping` to succeed.
    _configured : bool
        True after a successful :meth:`configure` call.
    _connected : bool
        True after a successful :meth:`connect` call.

    Examples
    --------
    >>> adv = MultiHandler()
    >>> adv.configure(input_tasks=[task1, task2])
    True
    >>> adv.connect()
    True
    >>> adv.set_trigger(n_samples=25600, trigger_channel=0, trigger_level=0.5)
    >>> data = adv.acquire()
    """

    def __init__(self) -> None:
        self.input_tasks: list = []
        self.output_tasks: list = []
        self.trigger_type: str = "software"  # Bug fix #2
        self.required_devices: set = set()
        self._configured: bool = False
        self._connected: bool = False
        self._lock: threading.RLock = threading.RLock()
        self.trigger = None
        self._trigger_is_set: bool = False
        self.input_channels: list | None = None
        self.input_sample_rate: float | None = None
        self._hw_trigger_start_fn: Callable | None = None
        self._hw_trigger_support_fn: Callable | None = None
        self._executor: ThreadPoolExecutor | None = None
        # Guards lazy executor creation only — never held during
        # acquisition, so acquire(blocking=False) cannot block on it.
        self._executor_lock: threading.Lock = threading.Lock()
        self._connect_called: bool = False
        self._acquire_running: bool = False

    # -----------------------------------------------------------------------
    # Public lifecycle
    # -----------------------------------------------------------------------

    def configure(
        self,
        input_tasks: list | None = None,  # Bug fix #4 — no mutable default
        output_tasks: list | None = None,
    ) -> bool:
        """Validate and resolve task lists for synchronized acquisition.

        Runs the full validation pipeline sequentially.  Stores resolved
        task lists on the instance only when all validation passes.

        Parameters
        ----------
        input_tasks : list or None, optional
            Input (acquisition) tasks.  Each element may be a
            ``nidaqmx.task.Task``, :class:`AITask`, :class:`AOTask`,
            or ``str`` (task name in NI MAX).  Default is an empty list.
        output_tasks : list or None, optional
            Output (generation) tasks.  Same accepted types as
            *input_tasks*.  Default is an empty list.

        Returns
        -------
        bool
            ``True`` when all validation passes and tasks are stored.
            ``False`` when any validation step fails.

        Raises
        ------
        TypeError
            If *input_tasks* or *output_tasks* are not lists, or contain
            objects of unsupported types.
        """
        # Bug fix #4 — create fresh lists, never share mutable defaults
        if input_tasks is None:
            input_tasks = []
        if output_tasks is None:
            output_tasks = []

        success = True

        # Step 1: type-check before any resolution
        success &= self._validate_types(input_tasks, output_tasks)
        if not success:
            return False

        # Steps 2-3: resolve all task representations to nidaqmx.task.Task
        resolved_input = self._resolve_tasks(input_tasks)
        resolved_output = self._resolve_tasks(output_tasks)

        # Steps 4-5: are resolved tasks valid and open?
        success &= self._validate_validity(resolved_input)
        success &= self._validate_validity(resolved_output)

        # Steps 6-7: all tasks within a group share the same sample rate?
        success &= self._validate_sample_rates(resolved_input)
        success &= self._validate_sample_rates(resolved_output)

        # Steps 8-9: clock configuration identical within each group?
        success &= self._validate_timing(resolved_input)
        success &= self._validate_timing(resolved_output)

        # Steps 10-11: triggers consistent? sets self.trigger_type
        success &= self._validate_triggers(resolved_input)
        success &= self._validate_triggers(resolved_output)

        # Steps 12-13: acquisition mode compatible with trigger type?
        success &= self._validate_acquisition_mode(resolved_input)
        success &= self._validate_acquisition_mode(resolved_output)

        if not success:
            return False

        # Store copies so each configure() call produces new list objects
        # (prevents cross-call state sharing — bug fix #4 intent)
        self.input_tasks = list(resolved_input)
        self.output_tasks = list(resolved_output)

        # Cache channel and rate metadata from the first input task
        if resolved_input:
            self.input_channels = resolved_input[0].channel_names
            self.input_sample_rate = resolved_input[0].timing.samp_clk_rate

        self._define_required_devices()
        self._configured = True
        return True

    def connect(self) -> bool:
        """Connect to NI hardware and verify all required devices are present.

        Discovers device requirements from all configured tasks, then calls
        :meth:`ping` to confirm physical availability.

        Returns
        -------
        bool
            ``True`` when all required devices respond.  ``False`` otherwise.
        """
        with self._lock:
            self._connect_called = True
            self._define_required_devices()
            result = self.ping()
            if result:
                self._connected = True
            return result

    def disconnect(self) -> bool:
        """Close all input and output tasks and release hardware resources.

        Safe to call multiple times (idempotent).

        Returns
        -------
        bool
            Always ``True``.
        """
        with self._lock:
            for task in self.input_tasks:
                try:
                    task.close()
                except Exception as exc:
                    warnings.warn(str(exc), stacklevel=2)

            for task in self.output_tasks:
                try:
                    task.close()
                except Exception as exc:
                    warnings.warn(str(exc), stacklevel=2)

            self._connected = False
            return True

    def ping(self) -> bool:
        """Check that all required NI-DAQmx devices are currently connected.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            ``True`` when every device in :attr:`required_devices` is found
            in the connected-device set.  ``True`` also when
            :attr:`required_devices` is empty.
        """
        connected = get_connected_devices()
        if self.required_devices.issubset(connected):
            return True
        return False

    def check_state(self) -> str:
        """Check connection state with auto-reconnection.

        Mirrors ``DAQHandler.check_state()``: when the connection appears
        lost, a single :meth:`connect` retry is attempted.

        Returns
        -------
        str
            One of ``'connected'``, ``'reconnected'``,
            ``'connection lost'``, or ``'disconnected'`` (when
            :meth:`connect` was never called).
        """
        with self._lock:
            if not self._connected:
                if self._connect_called:
                    if self.connect():
                        return "reconnected"
                    return "connection lost"
                return "disconnected"

            # Connected — verify with ping
            if self.ping():
                return "connected"

            # Ping failed — attempt reconnect
            if self.connect():
                return "reconnected"

            return "connection lost"

    def is_running(self) -> bool:
        """Return whether any configured task is currently running.

        Queries ``is_task_done()`` on every input and output task.  The
        query is intentionally lock-free so it can be called from a
        watchdog thread while an acquisition (which holds the RLock) is in
        flight.

        Returns
        -------
        bool
            ``True`` when any task reports ``is_task_done() == False``.
            ``False`` when there are no tasks, all tasks are done, or every
            query fails (a failing task warns and counts as not running).
        """
        running = False
        for task in self.input_tasks + self.output_tasks:
            try:
                if not task.is_task_done():
                    running = True
            except DaqError as exc:
                warnings.warn(
                    f"is_task_done() failed — treating task as not "
                    f"running: {exc}",
                    stacklevel=2,
                )
        return running

    def get_device_info(self) -> dict:
        """Return per-task channel names and sample rates.

        Returns
        -------
        dict
            ``{'input': {task_name: {'channel_names': list[str],
            'sample_rate': float}}, 'output': {...}}``.  The ``'input'`` /
            ``'output'`` keys are present only when the corresponding task
            list is non-empty; with no tasks an empty dict is returned.
        """
        def _per_task_info(tasks: list) -> dict:
            return {
                task.name: {
                    "channel_names": list(task.channel_names),
                    "sample_rate": float(task.timing.samp_clk_rate),
                }
                for task in tasks
            }

        info: dict = {}
        if self.input_tasks:
            info["input"] = _per_task_info(self.input_tasks)
        if self.output_tasks:
            info["output"] = _per_task_info(self.output_tasks)
        return info

    # -----------------------------------------------------------------------
    # Acquisition
    # -----------------------------------------------------------------------

    def acquire(
        self,
        custom_mode: object | None = None,
        blocking: bool = True,
    ) -> dict | np.ndarray | Future:
        """Acquire data, dispatching to hardware or software trigger path.

        Parameters
        ----------
        custom_mode : object, optional
            Forwarded to :meth:`acquire_with_hardware_trigger`, where it is
            passed to the registered ``support_function`` hook.  Ignored
            (with a warning) on the software-trigger path — hooks are a
            hardware-trigger concept.
        blocking : bool, optional
            If ``True`` (default), block until acquisition completes.
            If ``False``, submit the acquisition to a single-worker
            executor and return a :class:`~concurrent.futures.Future`.

        Returns
        -------
        dict or numpy.ndarray or Future
            Result from :meth:`acquire_with_hardware_trigger` (dict) or
            :meth:`acquire_with_software_trigger` (dict or ndarray), or a
            Future wrapping either when *blocking* is ``False``.

        Notes
        -----
        Do not block on the returned Future while holding this handler's
        methods in the same thread that runs the acquisition loop — the
        RLock is acquired inside the worker (same contract as
        ``DAQHandler``).
        """
        if blocking:
            return self._acquire_impl(custom_mode)
        return self._get_executor().submit(self._acquire_impl, custom_mode)

    def _acquire_impl(self, custom_mode: object | None = None) -> dict | np.ndarray:
        """Synchronous acquisition implementation.

        Called directly for blocking acquire, or submitted to the executor
        for non-blocking.  The RLock is acquired here — inside the worker
        thread for the non-blocking path — to ensure thread safety during
        hardware access.

        Parameters
        ----------
        custom_mode : object, optional
            See :meth:`acquire`.

        Returns
        -------
        dict or numpy.ndarray
            Acquired data.
        """
        with self._lock:
            if self.trigger_type == "hardware":
                return self.acquire_with_hardware_trigger(custom_mode=custom_mode)
            if custom_mode is not None:
                warnings.warn(
                    "custom_mode is ignored for software-triggered "
                    "acquisition — hardware-trigger hooks only apply to the "
                    "hardware-trigger path.",
                    stacklevel=2,
                )
            return self.acquire_with_software_trigger()

    def _get_executor(self) -> ThreadPoolExecutor:
        """Return the single-worker executor, creating it lazily.

        Creation is guarded by a dedicated lock (not the RLock) so that
        concurrent ``acquire(blocking=False)`` calls cannot race to create
        two executors, and so this call never blocks on an in-flight
        acquisition.

        Returns
        -------
        concurrent.futures.ThreadPoolExecutor
            Executor used for non-blocking acquisitions.
        """
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1)
            return self._executor

    def __del__(self) -> None:
        """Best-effort executor shutdown on garbage collection."""
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
        except Exception:
            pass

    def set_hardware_trigger_functions(
        self,
        start_function: Callable | None = None,
        support_function: Callable | None = None,
    ) -> None:
        """Register hook callables for hardware-triggered acquisition.

        Hooks are used by :meth:`acquire_with_hardware_trigger`:
        ``support_function(custom_mode)`` runs before the input tasks are
        armed (only when ``custom_mode`` is provided); ``start_function()``
        runs after all input tasks are armed, to fire the physical trigger
        line (e.g. via PLC or PFI).

        Parameters
        ----------
        start_function : callable or None, optional
            Called with no arguments after all input tasks are armed.
            ``None`` (default) clears the hook.
        support_function : callable or None, optional
            Called with ``custom_mode`` before arming, when ``custom_mode``
            is not ``None``.  ``None`` (default) clears the hook.

        Raises
        ------
        TypeError
            If either argument is neither callable nor ``None``.  No hook
            state changes in that case.
        """
        # Validate both arguments before assigning either, so a TypeError
        # leaves the hook state unchanged.
        if start_function is not None and not callable(start_function):
            raise TypeError(
                f"start_function must be callable or None, "
                f"got {type(start_function).__name__}."
            )
        if support_function is not None and not callable(support_function):
            raise TypeError(
                f"support_function must be callable or None, "
                f"got {type(support_function).__name__}."
            )
        self._hw_trigger_start_fn = start_function
        self._hw_trigger_support_fn = support_function

    def acquire_with_hardware_trigger(
        self, custom_mode: object | None = None
    ) -> dict:
        """Acquire finite data from all input tasks using a hardware trigger.

        The registered ``support_function`` hook (if any) is called with
        *custom_mode* before arming, when *custom_mode* is not ``None``.
        All tasks are then started (they arm and wait for the trigger), the
        registered ``start_function`` hook (if any) fires the physical
        trigger line, data is read from each task, and all started tasks
        are stopped — also when a hook or read raises.

        Parameters
        ----------
        custom_mode : object, optional
            Mode identifier forwarded to the ``support_function`` hook
            (e.g. selects a test procedure on an EOL bench).  When ``None``
            (default), the support hook is skipped.

        Returns
        -------
        dict
            Nested dict with structure
            ``{task_name: {channel_name: numpy.ndarray}}``.
            Each array contains the samples for one channel.

        Notes
        -----
        The read uses :data:`READ_ALL_AVAILABLE` because FINITE-mode tasks
        have a known sample count configured via ``samps_per_chan``.
        Single-channel tasks return 1-D data from nidaqmx; this method
        normalises all outputs to 1-D NumPy arrays keyed by channel name.
        Failures while stopping tasks during cleanup are reported via
        :func:`warnings.warn` so they cannot mask acquired data or the
        original exception.
        """
        acquired_data: dict = {}

        # Support hook runs before arming — selects e.g. a bench test mode.
        # Exact OpenEOL semantics: only when the hook is set AND a mode is
        # given.  No tasks are armed yet, so a hook failure leaks nothing.
        if self._hw_trigger_support_fn is not None and custom_mode is not None:
            self._hw_trigger_support_fn(custom_mode)

        started: list = []
        try:
            # Start all tasks before reading — they arm and wait for the
            # trigger.
            for task in self.input_tasks:
                task.start()
                started.append(task)

            # Start hook fires the physical trigger line once all tasks are
            # armed (only if registered).
            if self._hw_trigger_start_fn is not None:
                self._hw_trigger_start_fn()

            for task in self.input_tasks:
                raw = task.read(number_of_samples_per_channel=READ_ALL_AVAILABLE)
                channel_names = task.channel_names
                channel_data: dict = {}

                raw_array = np.array(raw)
                if raw_array.ndim == 1:
                    # Single-channel: nidaqmx returns a 1-D list
                    channel_data[channel_names[0]] = raw_array
                else:
                    # Multi-channel: shape is (n_channels, n_samples)
                    for channel_name, signal in zip(channel_names, raw_array):
                        channel_data[channel_name] = np.array(signal)

                acquired_data[task.name] = channel_data
        finally:
            # Stop every task that was started — also on hook/read failure
            # (OpenEOL leaked armed tasks here).  Stop failures warn so they
            # cannot mask the data or the original exception.
            for task in started:
                try:
                    task.stop()
                except Exception as exc:
                    warnings.warn(str(exc), stacklevel=2)

        return acquired_data

    def acquire_with_software_trigger(self, return_dict: bool = True) -> dict | np.ndarray:
        """Acquire data using pyTrigger on a single input task.

        Reads data in a loop, feeding it to the pyTrigger ring buffer until
        the trigger condition is satisfied.

        Parameters
        ----------
        return_dict : bool, optional
            When ``True`` (default), return a dict mapping each channel name
            to its data array, plus a ``'time'`` key.
            When ``False``, return the raw 2-D NumPy array from
            ``trigger.get_data()``.

        Returns
        -------
        dict or numpy.ndarray
            Dict with ``{channel_name: ndarray, 'time': ndarray}`` when
            *return_dict* is ``True``.  Otherwise a 2-D ndarray of shape
            ``(n_samples, n_channels)``.

        Raises
        ------
        ValueError
            If more than one input task is configured (software triggering
            is single-task only — reads are sequential and cannot synchronise
            multiple tasks).

        Notes
        -----
        The polling loop can be aborted from another thread via
        :meth:`stop_acquisition`.  An aborted acquisition stops the input
        task and returns the current trigger ring-buffer contents —
        possibly untriggered or partial data.
        """
        if len(self.input_tasks) > 1:
            raise ValueError(
                "Software trigger can only be used with one input task. "
                f"Got {len(self.input_tasks)} tasks. Use hardware triggers "
                "for multi-task acquisition."
            )

        if self.trigger is None:
            raise RuntimeError(
                "set_trigger() must be called before acquire(). "
                "Configure the trigger parameters first."
            )

        task = self.input_tasks[0]

        self._reset_trigger()

        # Reset the abort flag at the start of every acquisition so a stale
        # False value (left by a previous stop_acquisition() call) cannot
        # abort this run before it starts.
        self._acquire_running = True

        # Bug fix #3 — start the task before attempting any reads.
        # OpenEOL's base_advanced.py omitted this call, so reads were issued
        # against a task that was never started.
        task.start()

        # Flush stale samples from the hardware buffer before acquisition.
        task.read(number_of_samples_per_channel=READ_ALL_AVAILABLE, timeout=0.5)

        # Main acquisition loop — read until pyTrigger reports done or
        # stop_acquisition() clears the abort flag from another thread.
        while not self.trigger.finished and self._acquire_running:
            raw_data = np.array(
                task.read(number_of_samples_per_channel=READ_ALL_AVAILABLE, timeout=0.5)
            )
            if raw_data.ndim == 1:
                # Single channel: reshape to (1, n_samples) then transpose
                raw_data = raw_data[np.newaxis, :]

            # pyTrigger expects (n_samples, n_channels)
            data = raw_data.T
            self.trigger.add_data(data)

        self._acquire_running = False
        time.sleep(0.05)
        task.stop()

        if return_dict:
            data_arr = self.trigger.get_data()
            result: dict = {}
            channels = self.input_channels or task.channel_names
            for i, channel_name in enumerate(channels):
                result[channel_name] = data_arr[:, i]
            result["time"] = (
                np.arange(data_arr.shape[0]) / self.input_sample_rate
            )
            return result

        return self.trigger.get_data()

    def stop_acquisition(self) -> None:
        """Cooperatively abort an in-flight software-triggered acquisition.

        Clears the abort flag checked by the polling loop in
        :meth:`acquire_with_software_trigger`.  The aborted acquisition
        stops the input task and returns the current trigger ring-buffer
        contents (possibly untriggered or partial data).  Calling this with
        no acquisition in flight is a no-op.

        Notes
        -----
        Hardware-triggered acquisitions use blocking driver reads and are
        **not** affected by this method.
        """
        # Deliberately does NOT take self._lock: the acquisition loop holds
        # the RLock for the entire acquisition, so taking it here (from the
        # aborting thread) would deadlock until the acquisition finished —
        # defeating the purpose.  A plain bool store is atomic under the
        # GIL, so the flag itself needs no locking.
        self._acquire_running = False

    # -----------------------------------------------------------------------
    # Trigger management
    # -----------------------------------------------------------------------

    def set_trigger(
        self,
        n_samples: int,
        trigger_channel: int,
        trigger_level: float,
        trigger_type: str = "abs",
        presamples: int = 0,
    ) -> None:
        """Configure the pyTrigger for software-triggered acquisition.

        Parameters
        ----------
        n_samples : int
            Number of samples to acquire after the trigger event.
        trigger_channel : int
            Index of the channel to monitor for the trigger condition.
        trigger_level : float
            Threshold level that activates the trigger.
        trigger_type : str, optional
            Trigger detection mode.  Passed directly to ``pyTrigger``.
            Default is ``'abs'``.
        presamples : int, optional
            Number of samples to retain before the trigger event.
            Default is 0.

        Raises
        ------
        ValueError
            If no input tasks are configured (channel count cannot be
            determined).
        """
        if not self.input_tasks:
            raise ValueError(
                "No input tasks configured. Call configure() with at least "
                "one input task before setting a trigger."
            )

        n_channels = len(self.input_tasks[0].channel_names)
        sample_rate = self.input_tasks[0].timing.samp_clk_rate

        self.trigger = pyTrigger(
            n_samples,
            n_channels,
            trigger_channel=trigger_channel,
            trigger_level=trigger_level,
            trigger_type=trigger_type,
            presamples=presamples,
        )
        self._trigger_is_set = True

    def _reset_trigger(self) -> None:
        """Reset the pyTrigger instance to its initial state.

        Called at the start of each software-triggered acquisition so that
        previously accumulated data does not bleed into the new measurement.

        pyTrigger does not provide a ``reset()`` method, so each attribute
        is restored manually (same pattern as ``DAQHandler._reset_trigger``).
        """
        if self.trigger is not None:
            self.trigger.ringbuff.clear()
            self.trigger.triggered = False
            self.trigger.rows_left = self.trigger.rows
            self.trigger.finished = False
            self.trigger.first_data = True

    # -----------------------------------------------------------------------
    # Device management
    # -----------------------------------------------------------------------

    def _define_required_devices(self) -> None:
        """Collect device names from all configured tasks into :attr:`required_devices`.

        Iterates over both input and output tasks, extracting device names
        from each task's ``.devices`` attribute and storing the union in
        :attr:`required_devices`.
        """
        self.required_devices = set()
        for task in self.input_tasks:
            self.required_devices.update(self._get_task_devices(task))
        for task in self.output_tasks:
            self.required_devices.update(self._get_task_devices(task))

    def _get_task_devices(self, task) -> set:
        """Return the set of device names used by a nidaqmx task.

        Parameters
        ----------
        task : nidaqmx.task.Task
            The task to inspect.

        Returns
        -------
        set
            Device name strings extracted from ``task.devices``.
        """
        return {dev.name for dev in task.devices}

    # -----------------------------------------------------------------------
    # Validation pipeline (all operate on nidaqmx.task.Task only)
    # -----------------------------------------------------------------------

    def _validate_types(
        self, input_tasks: list, output_tasks: list
    ) -> bool:
        """Validate that task lists contain only supported task types.

        Parameters
        ----------
        input_tasks : list
            Candidate input tasks.
        output_tasks : list
            Candidate output tasks.

        Returns
        -------
        bool
            ``True`` when both lists are valid.

        Raises
        ------
        TypeError
            If either argument is not a list, or if any element is not a
            ``str``, :class:`AITask`, :class:`AOTask`, or
            ``nidaqmx.task.Task``.
        """
        if not isinstance(input_tasks, list):
            raise TypeError(
                f"input_tasks must be a list, got {type(input_tasks).__name__}."
            )
        if not isinstance(output_tasks, list):
            raise TypeError(
                f"output_tasks must be a list, got {type(output_tasks).__name__}."
            )

        for task_list, label in ((input_tasks, "input"), (output_tasks, "output")):
            for task in task_list:
                if not self._is_valid_task_type(task):
                    raise TypeError(
                        f"All {label}_tasks must be nidaqmx.task.Task, AITask, "
                        f"AOTask, or str. Got {type(task).__name__}."
                    )

        return True

    def _is_valid_task_type(self, task: object) -> bool:
        """Return True when *task* is one of the four supported types.

        Parameters
        ----------
        task : object
            Object to inspect.

        Returns
        -------
        bool
        """
        if isinstance(task, (str, AITask, AOTask)):
            return True
        # nidaqmx may not be installed; guard the isinstance check
        if _NIDAQMX_AVAILABLE and isinstance(task, nidaqmx.task.Task):
            return True
        return False

    def _resolve_tasks(self, tasks: list) -> list:
        """Convert mixed task representations to ``nidaqmx.task.Task`` objects.

        Accepts four types and resolves each:

        - ``str``           → :func:`get_task_by_name`
        - :class:`AITask`   → ``ni_task.configure()``
        - :class:`AOTask` → ``ni_task_out.configure()``
        - ``nidaqmx.task.Task`` → passed through unchanged

        Parameters
        ----------
        tasks : list
            Mixed list of task representations.

        Returns
        -------
        list
            List of resolved ``nidaqmx.task.Task`` objects.

        Raises
        ------
        TypeError
            If any element is not one of the four supported types.
        """
        resolved = []
        for task in tasks:
            if isinstance(task, str):
                resolved.append(get_task_by_name(task))

            elif isinstance(task, AITask):
                task.configure()
                resolved.append(task.task)

            elif isinstance(task, AOTask):
                task.configure()
                resolved.append(task.task)

            elif _NIDAQMX_AVAILABLE and isinstance(task, nidaqmx.task.Task):
                resolved.append(task)

            else:
                raise TypeError(
                    f"Task must be nidaqmx.task.Task, AITask, AOTask, "
                    f"or str. Got {type(task).__name__}."
                )

        return resolved

    def _validate_validity(self, tasks: list) -> bool:
        """Check that each task is open and has at least one channel.

        Parameters
        ----------
        tasks : list
            Resolved ``nidaqmx.task.Task`` objects.

        Returns
        -------
        bool
            ``True`` when every task passes both checks.  ``False`` when a
            task is invalid (error code ``-200088``) or has no channels
            (error code ``-200478``).

        Raises
        ------
        DaqError
            Any DaqError with an unrecognised error code is re-raised.

        Notes
        -----
        Error codes are compared as **integers** (bug fix #1).  OpenEOL
        compared them as strings (``== "-200088"``), which always evaluated
        to ``False``.
        """
        for task in tasks:
            # Check the task handle is still valid
            try:
                task.is_task_done()
            except DaqError as exc:
                # Bug fix #1 — integer comparison, not string
                if exc.error_code == _DAQ_ERROR_INVALID_TASK:
                    return False
                raise

            # Check the task has at least one channel
            try:
                _ = task.channel_names
            except DaqError as exc:
                if exc.error_code == _DAQ_ERROR_NO_CHANNELS:
                    return False
                raise

        return True

    def _validate_sample_rates(self, tasks: list) -> bool:
        """Verify all tasks share the same sample clock rate.

        Parameters
        ----------
        tasks : list
            Resolved ``nidaqmx.task.Task`` objects.

        Returns
        -------
        bool
            ``True`` when all tasks use the same rate, or when fewer than
            two tasks are present.
        """
        if len(tasks) <= 1:
            return True

        rates = {task.timing.samp_clk_rate for task in tasks}
        if len(rates) > 1:
            return False

        return True

    def _validate_timing(self, tasks: list) -> bool:
        """Verify all tasks share compatible timing settings (FR-5.5, relaxed).

        Sample-clock rate and samples-per-channel must be equal across all
        tasks — mismatches there corrupt synchronized data and fail
        validation.  Sample-clock *source* readback strings are **not**
        required to match: readbacks are not canonical (a master task on the
        onboard clock reads back its timebase terminal, e.g.
        ``/Dev1/ai/SampleClockTimebase``, while a slave configured with the
        exported terminal reads back ``/Dev1/ai/SampleClock``), so string
        equality rejects exactly the master/slave and cross-device
        configurations this handler exists to support.  Differing sources
        emit a single :class:`UserWarning` listing each task's readback.

        Parameters
        ----------
        tasks : list
            Resolved ``nidaqmx.task.Task`` objects.

        Returns
        -------
        bool
            ``True`` when rates and samples-per-channel are equal (clock
            sources may differ — see warning), or when fewer than two tasks
            are present.  ``False`` when rates or samples-per-channel
            differ.
        """
        if len(tasks) <= 1:
            return True

        rates = {task.timing.samp_clk_rate for task in tasks}
        samples = {task.timing.samp_quant_samp_per_chan for task in tasks}
        if len(rates) > 1 or len(samples) > 1:
            return False

        # The sample-clock source readback is best-effort: it only feeds the
        # mismatch warning below. On real hardware ``samp_clk_src`` is only
        # gettable while a task is reserved/committed/running — an un-reserved
        # task raises DaqError -200983 (invisible on simulated devices, which
        # permit the read). Hard validation (rate, samples) is already done, so
        # an unreadable source degrades to "skip the warning", never an error.
        try:
            sources = [task.timing.samp_clk_src for task in tasks]
        except DaqError:
            sources = []
        if len(set(sources)) > 1:
            detail = ", ".join(
                f"{task.name!r}: {source!r}"
                for task, source in zip(tasks, sources)
            )
            warnings.warn(
                f"Tasks report different sample-clock sources ({detail}). "
                "Verify the clock wiring — identical-source validation is "
                "not possible from readback strings (a master reads back "
                "its timebase terminal, a slave the exported sample-clock "
                "terminal).",
                stacklevel=2,
            )

        return True

    def _validate_triggers(self, tasks: list) -> bool:
        """Check that all tasks share the same start trigger.

        Sets :attr:`trigger_type` to ``'hardware'`` when hardware triggers
        are found, or ``'software'`` when no triggers are configured (bug
        fix #2 — OpenEOL only set ``'hardware'`` and left ``trigger_type``
        undefined for the software path).

        Parameters
        ----------
        tasks : list
            Resolved ``nidaqmx.task.Task`` objects.

        Returns
        -------
        bool
            ``True`` when all tasks share the same trigger type and source,
            or when fewer than two tasks are present.  ``False`` when trigger
            types or sources differ across tasks.
        """
        if len(tasks) <= 1:
            return True

        # Collect trigger type names across all tasks
        trig_type_names = {
            task.triggers.start_trigger.trig_type.name for task in tasks
        }

        if len(trig_type_names) > 1:
            return False

        trig_type_name = trig_type_names.pop()

        if trig_type_name == "NONE":
            # No hardware trigger — software mode
            self.trigger_type = "software"
            return True

        # Validate that all tasks share the same trigger source
        if trig_type_name == "DIGITAL_EDGE":
            sources = {
                task.triggers.start_trigger.dig_edge_src for task in tasks
            }
        elif trig_type_name == "ANALOG_EDGE":
            sources = {
                task.triggers.start_trigger.anlg_edge_src for task in tasks
            }
        else:
            return False

        if len(sources) > 1:
            return False

        self.trigger_type = "hardware"
        return True

    def _validate_acquisition_mode(self, tasks: list) -> bool:
        """Verify that acquisition mode is compatible with the trigger type.

        FINITE mode requires a hardware trigger; CONTINUOUS mode requires
        a software trigger.  Mixed modes are rejected.

        Parameters
        ----------
        tasks : list
            Resolved ``nidaqmx.task.Task`` objects.

        Returns
        -------
        bool
            ``True`` when mode and trigger type are compatible, or when
            *tasks* is empty.  ``False`` on mismatch.

        Notes
        -----
        Accesses ``self.trigger_type``, which must be set before this
        method is called (guaranteed because :meth:`_validate_triggers`
        runs first in the pipeline).
        """
        if not tasks:
            return True

        modes = {task.timing.samp_quant_samp_mode.name for task in tasks}

        if len(modes) > 1:
            return False

        mode = modes.pop().lower()

        if mode == "finite":
            if self.trigger_type == "hardware":
                return True
            return False

        if mode == "continuous":
            if self.trigger_type == "software":
                return True
            return False

        return False
