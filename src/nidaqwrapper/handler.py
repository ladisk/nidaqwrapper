"""High-level single-task NI-DAQmx interface.

Provides :class:`DAQHandler`, which consolidates OpenEOL's
``IONationalInstruments`` lifecycle, LDAQ's ``NIAcquisition`` / ``NIGeneration``
patterns, and pyTrigger-based software triggering into a single, unified class.

Supports NI MAX task name strings and programmatic :class:`AITask` /
:class:`AOTask` objects.  Thread-safe via per-instance
:class:`threading.RLock`.

Examples
--------
Standalone quick-start::

    wrapper = DAQHandler(task_in='MyInputTask')
    wrapper.connect()
    wrapper.set_trigger(n_samples=1000, trigger_channel=0, trigger_level=0.5)
    data = wrapper.acquire()
    wrapper.disconnect()

Framework integration (LDAQ / OpenEOL)::

    wrapper = DAQHandler()
    wrapper.configure(task_in=ni_task_obj, task_out='OutputTask')
    wrapper.connect()
    ...
    wrapper.disconnect()
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Union

import warnings

import numpy as np

from .digital import DITask, DOTask
from .ai_task import AITask
from .ao_task import AOTask
from .utils import get_connected_devices, get_task_by_name

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


class DAQHandler:
    """High-level single-task NI-DAQmx interface.

    Supports NI MAX task names (strings), programmatic
    :class:`AITask` / :class:`AOTask` objects, and raw
    ``nidaqmx.task.Task`` objects.  Raw tasks are automatically wrapped
    via the appropriate ``from_task()`` classmethod.  Provides a
    ``configure → connect → acquire/generate → disconnect`` lifecycle
    with software-triggered acquisition via pyTrigger, continuous signal
    generation, single-sample I/O, auto-reconnection, and thread safety.

    Parameters
    ----------
    task_in : str, AITask, or nidaqmx.task.Task, optional
        Input task — NI MAX name string, :class:`AITask` instance, or
        raw ``nidaqmx.task.Task`` object.
    task_out : str, AOTask, or nidaqmx.task.Task, optional
        Output task — NI MAX name string, :class:`AOTask` instance, or
        raw ``nidaqmx.task.Task`` object.
    **kwargs
        Additional keyword arguments forwarded to :meth:`configure`.
        Includes ``acquisition_sleep`` and ``post_trigger_delay``.
    """

    def __init__(
        self,
        task_in: str | AITask | None = None,
        task_out: str | AOTask | None = None,
        **kwargs: Any,
    ) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1)

        # State flags
        self._configured = False
        self._connected = False
        self._connect_called = False
        self._trigger_is_set = False
        self._state = "disconnected"

        # Task references (active nidaqmx task handles)
        self._task_in: Any | None = None
        self._task_out: Any | None = None
        self._task_in_obj_active: Any | None = None  # active AITask obj
        self._task_out_obj_active: Any | None = None  # active AOTask obj

        # Type flags
        self._task_in_is_str = False
        self._task_in_is_obj = False
        self._task_out_is_str = False
        self._task_out_is_obj = False

        # Stored configuration for reconnection
        self._task_in_name: str | None = None
        self._task_in_obj: AITask | None = None
        self._task_in_name_str: str | None = None
        self._task_in_sample_rate: float | None = None
        self._task_out_name: str | None = None
        self._task_out_obj: AOTask | None = None
        self._task_out_name_str: str | None = None
        self._task_out_sample_rate: float | None = None

        # Metadata (populated on connect)
        self._channel_names_in: list[str] = []
        self._channel_names_out: list[str] = []
        self._sample_rate_in: float | None = None
        self._sample_rate_out: float | None = None
        self._n_channels_in: int = 0
        self._n_channels_out: int = 0
        self._required_devices: set[str] = set()

        # Digital task state
        self._task_digital_in_is_str = False
        self._task_digital_in_is_obj = False
        self._task_digital_in_name: str | None = None
        self._task_digital_in_obj: DITask | None = None
        self._task_digital_in: Any | None = None  # active DITask after connect
        self._task_digital_out_is_str = False
        self._task_digital_out_is_obj = False
        self._task_digital_out_name: str | None = None
        self._task_digital_out_obj: DOTask | None = None
        self._task_digital_out: Any | None = None  # active DOTask after connect

        # Runtime flags
        self._acquire_running = False
        self._generation_running = False

        # Timing parameters (overridable via kwargs or configure)
        self.acquisition_sleep: float = kwargs.get("acquisition_sleep", 0.01)
        self.post_trigger_delay: float = kwargs.get("post_trigger_delay", 0.05)

        # If task kwargs provided, call configure (but NOT connect)
        _has_tasks = (
            task_in is not None
            or task_out is not None
            or kwargs.get("task_digital_in") is not None
            or kwargs.get("task_digital_out") is not None
        )
        if _has_tasks:
            self.configure(task_in=task_in, task_out=task_out, **kwargs)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        task_in: str | AITask | None = None,
        task_out: str | AOTask | None = None,
        task_digital_in: str | DITask | None = None,
        task_digital_out: str | DOTask | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure input, output, and digital tasks.

        Accepts NI MAX task name strings, :class:`AITask` /
        :class:`AOTask` / :class:`DITask` / :class:`DOTask` objects,
        or raw ``nidaqmx.task.Task`` objects.  Raw tasks are wrapped
        automatically via the appropriate ``from_task()`` classmethod.
        Resets all internal state so the wrapper can be reconfigured
        without creating a new instance.

        Parameters
        ----------
        task_in : str, AITask, or nidaqmx.task.Task, optional
            Analog input task specification.
        task_out : str, AOTask, or nidaqmx.task.Task, optional
            Analog output task specification.
        task_digital_in : str, DITask, or nidaqmx.task.Task, optional
            Digital input task — object or NI MAX name string.
        task_digital_out : str, DOTask, or nidaqmx.task.Task, optional
            Digital output task — object or NI MAX name string.
        **kwargs
            ``acquisition_sleep``, ``post_trigger_delay``.
        """
        # Reset state
        self._trigger_is_set = False
        self._acquire_running = False
        self._generation_running = False
        self._task_in = None
        self._task_out = None
        self._task_in_obj_active = None
        self._task_out_obj_active = None
        self._task_digital_in = None
        self._task_digital_out = None

        # Timing parameters
        if "acquisition_sleep" in kwargs:
            self.acquisition_sleep = kwargs["acquisition_sleep"]
        if "post_trigger_delay" in kwargs:
            self.post_trigger_delay = kwargs["post_trigger_delay"]

        # -- Input task ------------------------------------------------
        self._task_in_is_str = False
        self._task_in_is_obj = False
        self._task_in_name = None
        self._task_in_obj = None
        self._task_in_name_str = None
        self._task_in_sample_rate = None

        if isinstance(task_in, str):
            self._task_in_is_str = True
            self._task_in_name = task_in
        elif isinstance(task_in, AITask):
            self._task_in_is_obj = True
            self._task_in_obj = task_in
            self._task_in_name_str = task_in.task_name
            self._task_in_sample_rate = task_in.sample_rate
        elif task_in is not None:
            # Check if it's a raw nidaqmx.task.Task object
            is_raw_task = False
            if _NIDAQMX_AVAILABLE:
                try:
                    is_raw_task = isinstance(task_in, nidaqmx.task.Task)
                except TypeError:
                    # nidaqmx.task.Task is not a valid type (e.g., mocked or unavailable)
                    pass

            if is_raw_task:
                # Wrap via AITask.from_task()
                wrapped_task = AITask.from_task(task_in)
                self._task_in_is_obj = True
                self._task_in_obj = wrapped_task
                self._task_in_name_str = wrapped_task.task_name
                self._task_in_sample_rate = wrapped_task.sample_rate
            else:
                raise TypeError(
                    f"task_in must be a string, AITask, or nidaqmx.task.Task, "
                    f"got {type(task_in).__name__}"
                )

        # -- Output task -----------------------------------------------
        self._task_out_is_str = False
        self._task_out_is_obj = False
        self._task_out_name = None
        self._task_out_obj = None
        self._task_out_name_str = None
        self._task_out_sample_rate = None

        if isinstance(task_out, str):
            self._task_out_is_str = True
            self._task_out_name = task_out
        elif isinstance(task_out, AOTask):
            self._task_out_is_obj = True
            self._task_out_obj = task_out
            self._task_out_name_str = task_out.task_name
            self._task_out_sample_rate = task_out.sample_rate
        elif task_out is not None:
            # Check if it's a raw nidaqmx.task.Task object
            is_raw_task = False
            if _NIDAQMX_AVAILABLE:
                try:
                    is_raw_task = isinstance(task_out, nidaqmx.task.Task)
                except TypeError:
                    # nidaqmx.task.Task is not a valid type (e.g., mocked or unavailable)
                    pass

            if is_raw_task:
                # Wrap via AOTask.from_task()
                wrapped_task = AOTask.from_task(task_out)
                self._task_out_is_obj = True
                self._task_out_obj = wrapped_task
                self._task_out_name_str = wrapped_task.task_name
                self._task_out_sample_rate = wrapped_task.sample_rate
            else:
                raise TypeError(
                    f"task_out must be a string, AOTask, or nidaqmx.task.Task, "
                    f"got {type(task_out).__name__}"
                )

        # -- Digital input task ----------------------------------------
        self._task_digital_in_is_str = False
        self._task_digital_in_is_obj = False
        self._task_digital_in_name = None
        self._task_digital_in_obj = None

        if isinstance(task_digital_in, str):
            self._task_digital_in_is_str = True
            self._task_digital_in_name = task_digital_in
        elif isinstance(task_digital_in, DITask):
            self._task_digital_in_is_obj = True
            self._task_digital_in_obj = task_digital_in
        elif task_digital_in is not None:
            # Check if it's a raw nidaqmx.task.Task object
            is_raw_task = False
            if _NIDAQMX_AVAILABLE:
                try:
                    is_raw_task = isinstance(task_digital_in, nidaqmx.task.Task)
                except TypeError:
                    # nidaqmx.task.Task is not a valid type (e.g., mocked or unavailable)
                    pass

            if is_raw_task:
                # Wrap via DITask.from_task()
                wrapped_task = DITask.from_task(task_digital_in)
                self._task_digital_in_is_obj = True
                self._task_digital_in_obj = wrapped_task
            else:
                raise TypeError(
                    f"task_digital_in must be a string, DITask, or nidaqmx.task.Task, "
                    f"got {type(task_digital_in).__name__}"
                )

        # -- Digital output task ---------------------------------------
        self._task_digital_out_is_str = False
        self._task_digital_out_is_obj = False
        self._task_digital_out_name = None
        self._task_digital_out_obj = None

        if isinstance(task_digital_out, str):
            self._task_digital_out_is_str = True
            self._task_digital_out_name = task_digital_out
        elif isinstance(task_digital_out, DOTask):
            self._task_digital_out_is_obj = True
            self._task_digital_out_obj = task_digital_out
        elif task_digital_out is not None:
            # Check if it's a raw nidaqmx.task.Task object
            is_raw_task = False
            if _NIDAQMX_AVAILABLE:
                try:
                    is_raw_task = isinstance(task_digital_out, nidaqmx.task.Task)
                except TypeError:
                    # nidaqmx.task.Task is not a valid type (e.g., mocked or unavailable)
                    pass

            if is_raw_task:
                # Wrap via DOTask.from_task()
                wrapped_task = DOTask.from_task(task_digital_out)
                self._task_digital_out_is_obj = True
                self._task_digital_out_obj = wrapped_task
            else:
                raise TypeError(
                    f"task_digital_out must be a string, DOTask, or nidaqmx.task.Task, "
                    f"got {type(task_digital_out).__name__}"
                )

        # Validate at least one task provided
        has_any = (
            self._task_in_is_str
            or self._task_in_is_obj
            or self._task_out_is_str
            or self._task_out_is_obj
            or self._task_digital_in_is_str
            or self._task_digital_in_is_obj
            or self._task_digital_out_is_str
            or self._task_digital_out_is_obj
        )
        if not has_any:
            raise ValueError(
                "At least one task must be provided (task_in, task_out, "
                "task_digital_in, or task_digital_out)."
            )

        self._configured = True

    # ------------------------------------------------------------------
    # Lifecycle: connect / disconnect
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to NI hardware by loading/creating tasks.

        For NI MAX name strings, loads the saved task via
        :func:`get_task_by_name`.  For :class:`AITask` objects, calls
        ``start()`` to create the underlying hardware task.

        Returns
        -------
        bool
            ``True`` if connection succeeded and :meth:`ping` passes,
            ``False`` otherwise.
        """
        with self._lock:
            self._connect_called = True

            try:
                # Close any existing tasks first
                self._close_task_in()
                self._close_task_out()

                # -- Input task ----------------------------------------
                if self._task_in_is_str:
                    loaded = get_task_by_name(self._task_in_name)
                    if loaded is None:
                        self._connected = False
                        return False
                    self._task_in = loaded
                    self._extract_input_metadata_from_nidaqmx(loaded)
                    # Wrap in AITask so read_all_available() can use it
                    try:
                        loaded.start()
                        self._task_in_obj_active = AITask.from_task(loaded)
                    except Exception:
                        pass  # Wrapping optional — basic connect still works

                elif self._task_in_is_obj:
                    ni_task = self._task_in_obj
                    # Only configure if we own the task (not externally-provided)
                    if ni_task._owns_task:
                        ni_task.configure()
                    self._task_in = ni_task.task
                    self._task_in_obj_active = ni_task
                    self._extract_input_metadata_from_ni_task(ni_task)

                # -- Output task ---------------------------------------
                if self._task_out_is_str:
                    loaded = get_task_by_name(self._task_out_name)
                    if loaded is None:
                        self._connected = False
                        return False
                    self._task_out = loaded
                    self._extract_output_metadata_from_nidaqmx(loaded)

                elif self._task_out_is_obj:
                    ni_task_out = self._task_out_obj
                    # Only configure if we own the task (not externally-provided)
                    if ni_task_out._owns_task:
                        ni_task_out.configure()
                    self._task_out = ni_task_out.task
                    self._task_out_obj_active = ni_task_out
                    self._extract_output_metadata_from_ni_task_out(ni_task_out)

                # -- Digital input task ---------------------------------
                self._close_digital_in()
                if self._task_digital_in_is_str:
                    loaded = get_task_by_name(self._task_digital_in_name)
                    if loaded is not None:
                        self._task_digital_in = loaded

                elif self._task_digital_in_is_obj:
                    try:
                        self._task_digital_in_obj.configure()
                        self._task_digital_in = self._task_digital_in_obj
                    except Exception as exc:
                        warnings.warn(str(exc), stacklevel=2)

                # -- Digital output task --------------------------------
                self._close_digital_out()
                if self._task_digital_out_is_str:
                    loaded = get_task_by_name(self._task_digital_out_name)
                    if loaded is not None:
                        self._task_digital_out = loaded

                elif self._task_digital_out_is_obj:
                    try:
                        self._task_digital_out_obj.configure()
                        self._task_digital_out = self._task_digital_out_obj
                    except Exception as exc:
                        warnings.warn(str(exc), stacklevel=2)

                # Build required devices set
                self._required_devices = set()
                if self._task_in is not None:
                    for dev in self._task_in.devices:
                        self._required_devices.add(dev.name)
                if self._task_out is not None:
                    for dev in self._task_out.devices:
                        self._required_devices.add(dev.name)

                # Verify connectivity
                if self.ping():
                    self._connected = True
                    self._state = "connected"
                    return True
                else:
                    # Digital-only configs have no required_devices
                    # and will fail ping — still mark as connected
                    has_digital = (
                        self._task_digital_in is not None
                        or self._task_digital_out is not None
                    )
                    has_analog = (
                        self._task_in is not None
                        or self._task_out is not None
                    )
                    if has_digital and not has_analog:
                        self._connected = True
                        self._state = "connected"
                        return True
                    self._connected = False
                    return False

            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
                self._connected = False
                return False

    def disconnect(self) -> bool:
        """Disconnect from NI hardware, closing all tasks.

        Idempotent — safe to call multiple times or when never connected.

        Returns
        -------
        bool
            Always ``True``.
        """
        with self._lock:
            # Stop any running generation safely
            if self._generation_running:
                try:
                    self._stop_generation_impl()
                except Exception as exc:
                    warnings.warn(str(exc), stacklevel=2)

            self._close_task_in()
            self._close_task_out()
            self._close_digital_in()
            self._close_digital_out()
            self._connected = False
            self._state = "disconnected"
            return True

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    def set_trigger(
        self,
        n_samples: int,
        trigger_channel: int,
        trigger_level: float,
        trigger_type: str = "abs",
        presamples: int = 0,
    ) -> None:
        """Set up software triggering via pyTrigger.

        Parameters
        ----------
        n_samples : int
            Number of samples to acquire after trigger.
        trigger_channel : int
            Channel index used for trigger detection.
        trigger_level : float
            Trigger level in the channel's native units.
        trigger_type : str, optional
            Trigger type: ``'abs'``, ``'up'``, or ``'down'``.
            Default is ``'abs'``.
        presamples : int, optional
            Number of pre-trigger samples. Default is ``0``.

        Raises
        ------
        ValueError
            If no input task is configured.
        """
        if not (self._task_in_is_str or self._task_in_is_obj):
            raise ValueError(
                "No input task is configured. Call configure() with task_in first."
            )

        self.trigger = pyTrigger(
            rows=n_samples,
            channels=self._n_channels_in,
            trigger_channel=trigger_channel,
            trigger_level=trigger_level,
            trigger_type=trigger_type,
            presamples=presamples,
        )
        self._trigger_is_set = True

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def acquire(
        self,
        return_dict: bool = False,
        blocking: bool = True,
    ) -> np.ndarray | dict | Future:
        """Perform software-triggered acquisition.

        Requires :meth:`set_trigger` to have been called first.

        Parameters
        ----------
        return_dict : bool, optional
            If ``True``, return a dict with channel names as keys and a
            ``'time'`` key.  Default is ``False`` (numpy array).
        blocking : bool, optional
            If ``True`` (default), block until acquisition completes.
            If ``False``, return a :class:`~concurrent.futures.Future`.

        Returns
        -------
        numpy.ndarray or dict or Future
            Acquired data in ``(n_samples, n_channels)`` format, or a
            dict, or a Future wrapping either.

        Raises
        ------
        ValueError
            If no input task is configured.
        RuntimeError
            If :meth:`set_trigger` has not been called.
        """
        if not (self._task_in_is_str or self._task_in_is_obj):
            raise ValueError(
                "No input task is configured. Call configure() with task_in first."
            )

        if not self._trigger_is_set:
            raise RuntimeError(
                "set_trigger() must be called before acquire(). "
                "Configure a trigger with set_trigger(n_samples, trigger_channel, trigger_level)."
            )

        if blocking:
            return self._acquire_impl(return_dict)
        else:
            return self._executor.submit(self._acquire_impl, return_dict)

    def _acquire_impl(self, return_dict: bool = False) -> np.ndarray | dict:
        """Execute the triggered acquisition loop.

        Internal method — called directly for blocking acquire, or
        submitted to the executor for non-blocking.

        Parameters
        ----------
        return_dict : bool
            Whether to return a dict with channel names.

        Returns
        -------
        numpy.ndarray or dict
            Acquired data.  When aborted via :meth:`stop_acquisition`, the
            current trigger ring-buffer contents are returned — possibly
            untriggered or partial data.
        """
        with self._lock:
            # Reset trigger state for fresh acquisition
            self._reset_trigger()

            # Start the input task
            self._task_in.start()

            # Flush buffer (discard first read — driver init artifact)
            try:
                self._task_in.read(READ_ALL_AVAILABLE, timeout=0.5)
            except Exception:
                pass  # Flush may return empty; ignore

            # Main acquisition loop — runs until the trigger fires or
            # stop_acquisition() clears the abort flag from another thread.
            # The flag is reset True here (under the lock) so a stale False
            # value from a previous abort cannot skip this acquisition.
            self._acquire_running = True
            while not self.trigger.finished and self._acquire_running:
                raw_data = np.array(
                    self._task_in.read(READ_ALL_AVAILABLE, timeout=0.5)
                )

                # Single-channel: nidaqmx returns 1D → reshape to (1, n_samples)
                if raw_data.ndim == 1:
                    raw_data = raw_data[None, :]

                # Transpose to (n_samples, n_channels) for pyTrigger
                data = raw_data.T
                self.trigger.add_data(data)

                if not self.trigger.finished:
                    time.sleep(self.acquisition_sleep)
            self._acquire_running = False

            # Post-trigger delay
            time.sleep(self.post_trigger_delay)

            # Stop the input task
            self._task_in.stop()

            # Get triggered data
            if return_dict:
                data_arr = self.trigger.get_data()
                result: dict[str, Any] = {}
                for i, ch_name in enumerate(self._channel_names_in):
                    result[ch_name] = data_arr[:, i]
                result["time"] = np.arange(data_arr.shape[0]) / self._sample_rate_in
                return result
            else:
                return self.trigger.get_data()

    def _reset_trigger(self) -> None:
        """Reset the pyTrigger instance for a fresh acquisition."""
        self.trigger.ringbuff.clear()
        self.trigger.triggered = False
        self.trigger.rows_left = self.trigger.rows
        self.trigger.finished = False
        self.trigger.first_data = True

    def stop_acquisition(self) -> None:
        """Cooperatively abort an in-flight triggered acquisition.

        Clears the abort flag checked by the polling loop in
        :meth:`_acquire_impl`.  The aborted acquisition stops the input
        task and returns the current trigger ring-buffer contents
        (possibly untriggered or partial data).  Calling this with no
        acquisition in flight is a no-op.
        """
        # Deliberately does NOT take self._lock: the acquisition loop holds
        # the RLock for the entire acquisition, so taking it here (from the
        # aborting thread) would deadlock until the acquisition finished —
        # defeating the purpose.  A plain bool store is atomic under the
        # GIL, so the flag itself needs no locking.
        self._acquire_running = False

    # ------------------------------------------------------------------
    # Read (LDAQ integration + single sample)
    # ------------------------------------------------------------------

    def read_all_available(self, n_samples: int | None = None) -> np.ndarray:
        """Read samples without triggering (LDAQ pattern).

        Calls ``acquire()`` on the active input task and transposes
        the result to ``(n_samples, n_channels)`` public format.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples per channel to read.  If provided, the call
            **blocks** until exactly *n_samples* are available — suitable
            for scripts and notebooks.  If ``None`` (default), drains every
            sample currently in the buffer without blocking
            (``READ_ALL_AVAILABLE``) — suitable for acquisition loops.

        Returns
        -------
        numpy.ndarray
            Data in ``(n_samples, n_channels)`` format.  Returns
            ``(0, n_channels)`` if no data is available.
        """
        with self._lock:
            if self._task_in_obj_active is None:
                raise ValueError(
                    "No active input task. Call connect() first."
                )
            raw = self._task_in_obj_active.acquire(n_samples=n_samples)
            return raw

    def read(self) -> np.ndarray:
        """Read a single sample from each input channel.

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(n_channels,)``.

        Raises
        ------
        ValueError
            If no input task is configured, or if a continuous acquisition
            is currently in progress.
        """
        with self._lock:
            if self._task_in is None:
                raise ValueError(
                    "No input task is configured. Call configure() and connect() first."
                )
            if self._acquire_running:
                raise ValueError(
                    "Cannot call read() while a continuous acquisition is running. "
                    "Wait for acquire() to complete or use read_all_available()."
                )
            data = np.array(self._task_in.read(1))
            # nidaqmx may return a scalar for single-channel tasks, producing
            # a 0-D array.  Reshape to (1,) so callers always receive 1-D output.
            if data.ndim == 0:
                data = data.reshape((1,))
            elif data.ndim == 2:
                data = data.squeeze(axis=1)
            return data

    # ------------------------------------------------------------------
    # Digital I/O
    # ------------------------------------------------------------------

    def read_digital(self) -> np.ndarray:
        """Read the current state of all configured digital input lines.

        Delegates to the stored :class:`DITask` task's ``read()``
        method for on-demand reading.

        Returns
        -------
        numpy.ndarray
            Array of bool-like values, one per line.

        Raises
        ------
        RuntimeError
            If no digital input task is configured or not connected.
        """
        with self._lock:
            if not (self._task_digital_in_is_str or self._task_digital_in_is_obj):
                raise RuntimeError(
                    "No digital input task configured. "
                    "Call configure(task_digital_in=...) first."
                )
            if self._task_digital_in is None:
                raise RuntimeError(
                    "Digital input task is configured but not connected. "
                    "Call connect() first."
                )
            return self._task_digital_in.read()

    def write_digital(
        self, data: bool | int | list | np.ndarray
    ) -> None:
        """Write values to all configured digital output lines.

        Delegates to the stored :class:`DOTask` task's ``write()``
        method for on-demand writing.

        Parameters
        ----------
        data : bool, int, list, or numpy.ndarray
            Data to write.  Single bool/int for single-line, list or
            array for multi-line.

        Raises
        ------
        RuntimeError
            If no digital output task is configured or not connected.
        """
        with self._lock:
            if not (self._task_digital_out_is_str or self._task_digital_out_is_obj):
                raise RuntimeError(
                    "No digital output task configured. "
                    "Call configure(task_digital_out=...) first."
                )
            if self._task_digital_out is None:
                raise RuntimeError(
                    "Digital output task is configured but not connected. "
                    "Call connect() first."
                )
            self._task_digital_out.write(data)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, data: np.ndarray, overwrite: bool = True) -> None:
        """Start continuous signal generation.

        Parameters
        ----------
        data : numpy.ndarray
            Signal data in ``(n_samples, n_channels)`` or ``(n_samples,)``
            format.
        overwrite : bool, optional
            If ``True`` (default), stop any current generation first.
            If ``False``, raise if generation is running.

        Raises
        ------
        ValueError
            If no output task configured, shape mismatch, or generation
            running with ``overwrite=False``.
        """
        with self._lock:
            if self._task_out is None:
                raise ValueError(
                    "No output task is configured. Call configure() with task_out first."
                )

            # Validate shape
            if data.ndim == 2 and data.shape[1] != self._n_channels_out:
                raise ValueError(
                    f"Data has {data.shape[1]} columns but {self._n_channels_out} "
                    f"output channels are configured."
                )

            if self._generation_running:
                if overwrite:
                    self._task_out.stop()
                    time.sleep(0.01)
                else:
                    raise ValueError(
                        "Output generation is currently running and overwrite=False."
                    )

            # Transpose for nidaqmx: (n_samples, n_channels) → (n_channels, n_samples)
            if data.ndim == 2:
                write_data = np.ascontiguousarray(data.T)
            else:
                # 1D single-channel — ensure C-contiguous for nidaqmx C layer
                write_data = np.ascontiguousarray(data)

            self._task_out.out_stream.output_buf_size = data.shape[0]
            self._task_out.write(write_data, auto_start=True)
            self._generation_running = True

    def write(
        self,
        data: float | int | list | np.ndarray,
        overwrite: bool = True,
    ) -> None:
        """Write a single sample to each output channel.

        Handles nidaqmx's 2-sample minimum buffer requirement by
        duplicating the data to 2 samples internally.

        Parameters
        ----------
        data : float, int, list, or numpy.ndarray
            Single sample per channel.  Float/int for single-channel,
            array of shape ``(n_channels,)`` for multi-channel.
        overwrite : bool, optional
            If ``True`` (default), stop current generation first.

        Raises
        ------
        ValueError
            If shape doesn't match channel count, or generation running
            with ``overwrite=False``.
        """
        with self._lock:
            if self._task_out is None:
                raise ValueError(
                    "No output task is configured. Call configure() with task_out first."
                )

            # Normalize input
            if isinstance(data, (float, int)):
                data = np.array([data], dtype=float)
            elif isinstance(data, list):
                data = np.array(data, dtype=float)

            # Validate shape
            if data.shape[0] != self._n_channels_out:
                raise ValueError(
                    f"Data has {data.shape[0]} values but {self._n_channels_out} "
                    f"output channels are configured."
                )

            if self._generation_running:
                if overwrite:
                    self._task_out.stop()
                    time.sleep(0.01)
                else:
                    raise ValueError(
                        "Output generation is currently running and overwrite=False."
                    )

            # nidaqmx requires at least 2 samples
            if self._n_channels_out == 1:
                write_data = np.concatenate((data, data))
            else:
                # Stack to (n_channels, 2) then make C-contiguous
                write_data = np.ascontiguousarray(
                    np.array([data, data]).T
                )

            self._task_out.out_stream.output_buf_size = 2
            self._task_out.write(write_data, auto_start=True)
            try:
                self._task_out.stop()
            except Exception:
                # Some devices (e.g. NI 9260) may report DAC underrun
                # (-200018) when the 2-sample buffer drains before stop().
                # This is benign for single-sample DC output.
                pass

    def stop_generation(self) -> None:
        """Stop signal generation and write zeros to all output channels.

        Follows the safe shutdown pattern: stop the task, then write
        zeros so the output doesn't hold the last voltage.
        """
        with self._lock:
            if self._task_out is None:
                return  # No-op if no output task

            self._stop_generation_impl()

    def _stop_generation_impl(self) -> None:
        """Internal stop generation — caller must hold lock."""
        self._generation_running = False
        self._task_out.stop()
        # Write zeros through the write() path (handles 2-sample minimum)
        self.write(np.zeros(self._n_channels_out))

    # ------------------------------------------------------------------
    # Connectivity: ping / check_state
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Test if all required NI devices are connected.

        Returns
        -------
        bool
            ``True`` if all required devices are in the connected set.
        """
        with self._lock:
            if not self._configured:
                return False

            if not self._required_devices:
                return False

            connected = get_connected_devices()
            for device in self._required_devices:
                if device not in connected:
                    return False
            return True

    def check_state(self) -> str:
        """Check connection state with auto-reconnection.

        Returns
        -------
        str
            One of ``'connected'``, ``'reconnected'``,
            ``'connection lost'``, or ``'disconnected'``.
        """
        with self._lock:
            if not self._connected:
                if self._connect_called:
                    if self.connect():
                        self._state = "reconnected"
                        return "reconnected"
                    self._state = "connection lost"
                    return "connection lost"
                self._state = "disconnected"
                return "disconnected"

            # Connected — verify with ping
            if self.ping():
                self._state = "connected"
                return "connected"

            # Ping failed — attempt reconnect
            if self.connect():
                self._state = "reconnected"
                return "reconnected"

            self._state = "connection lost"
            return "connection lost"

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Return whether any active task is currently running.

        Queries ``is_task_done()`` on the active analog input/output
        nidaqmx tasks and the digital input/output tasks.  The query is
        intentionally lock-free so it can be called from a watchdog thread
        while an acquisition (which holds the RLock) is in flight.

        Returns
        -------
        bool
            ``True`` when any active task reports ``is_task_done() ==
            False``.  ``False`` when not connected, when there are no
            active tasks, when all tasks are done, or when every query
            fails (a failing task warns and counts as not running).
        """
        if not self._connected:
            return False

        candidates: list[Any] = []
        if self._task_in is not None:
            candidates.append(self._task_in)
        if self._task_out is not None:
            candidates.append(self._task_out)
        for digital in (self._task_digital_in, self._task_digital_out):
            if digital is None:
                continue
            # DITask/DOTask wrappers expose the raw nidaqmx task as .task;
            # tasks loaded from NI MAX name strings are raw already.
            inner = getattr(digital, "task", None)
            candidates.append(inner if inner is not None else digital)

        running = False
        for task in candidates:
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
        """Return device/task metadata.

        Returns
        -------
        dict
            Keys ``'input'`` and/or ``'output'``, each containing
            ``'channel_names'`` and ``'sample_rate'``.
        """
        info: dict[str, Any] = {}
        if self._task_in_is_str or self._task_in_is_obj:
            if self._channel_names_in:
                info["input"] = {
                    "channel_names": self._channel_names_in,
                    "sample_rate": self._sample_rate_in,
                }
        if self._task_out_is_str or self._task_out_is_obj:
            if self._channel_names_out:
                info["output"] = {
                    "channel_names": self._channel_names_out,
                    "sample_rate": self._sample_rate_out,
                }
        return info

    def get_sample_rate(self) -> float:
        """Return the input task's sample rate.

        Returns
        -------
        float
            Sample rate in Hz.

        Raises
        ------
        ValueError
            If no input task is configured.
        """
        if self._sample_rate_in is None:
            raise ValueError(
                "No input task configured. Cannot determine sample rate."
            )
        return self._sample_rate_in

    def get_channel_names(self) -> list[str]:
        """Return input channel names.

        Returns
        -------
        list[str]
            Channel name strings, or empty list if no input task.
        """
        return list(self._channel_names_in)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def clear_buffer(self) -> None:
        """Drain and discard all buffered input data.

        Prevents buffer overflow by reading and discarding all samples
        currently in the hardware FIFO.
        """
        with self._lock:
            if self._task_in_obj_active is None:
                return  # No-op
            self._task_in_obj_active.acquire()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> DAQHandler:
        """Enter the runtime context."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit the runtime context, disconnecting from hardware."""
        self.disconnect()

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        try:
            self.disconnect()
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_task_in(self) -> None:
        """Close the active input task (nidaqmx handle)."""
        if self._task_in is not None:
            try:
                self._task_in.close()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self._task_in = None

        if self._task_in_obj_active is not None:
            try:
                self._task_in_obj_active.clear_task()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self._task_in_obj_active = None

    def _close_task_out(self) -> None:
        """Close the active output task (nidaqmx handle)."""
        if self._task_out is not None:
            try:
                self._task_out.close()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self._task_out = None

        if self._task_out_obj_active is not None:
            try:
                self._task_out_obj_active.clear_task()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self._task_out_obj_active = None

    def _close_digital_in(self) -> None:
        """Close the active digital input task."""
        if self._task_digital_in is not None:
            try:
                self._task_digital_in.clear_task()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self._task_digital_in = None

    def _close_digital_out(self) -> None:
        """Close the active digital output task."""
        if self._task_digital_out is not None:
            try:
                self._task_digital_out.clear_task()
            except Exception as exc:
                warnings.warn(str(exc), stacklevel=2)
            self._task_digital_out = None

    def _extract_input_metadata_from_nidaqmx(self, task: Any) -> None:
        """Extract metadata from a loaded nidaqmx.Task (NI MAX)."""
        self._channel_names_in = list(task.channel_names)
        self._n_channels_in = task.number_of_channels
        self._sample_rate_in = float(task.timing.samp_clk_rate)

    def _extract_input_metadata_from_ni_task(self, ni_task: AITask) -> None:
        """Extract metadata from a programmatic AITask object."""
        self._channel_names_in = ni_task.channel_list
        self._n_channels_in = ni_task.number_of_ch
        self._sample_rate_in = ni_task.sample_rate

    def _extract_output_metadata_from_nidaqmx(self, task: Any) -> None:
        """Extract metadata from a loaded output nidaqmx.Task."""
        self._channel_names_out = list(task.channel_names)
        self._n_channels_out = task.number_of_channels
        self._sample_rate_out = float(task.timing.samp_clk_rate)

    def _extract_output_metadata_from_ni_task_out(
        self, ni_task_out: AOTask
    ) -> None:
        """Extract metadata from a programmatic AOTask object."""
        self._channel_names_out = ni_task_out.channel_list
        self._n_channels_out = ni_task_out.number_of_ch
        self._sample_rate_out = ni_task_out.sample_rate
