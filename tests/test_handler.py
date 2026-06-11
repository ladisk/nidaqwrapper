"""Unit tests for DAQHandler — high-level single-task NI-DAQmx interface.

Tests are fully mocked — nidaqmx is NOT required. Each test patches
only the specific dependencies needed, keeping tests isolated and fast.

Test groups mirror the task groups in openspec/changes/wrapper/tasks.md.
"""

from __future__ import annotations

import threading
import time
import warnings
from concurrent.futures import Future
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_ai_task(
    task_name="TestInput",
    sample_rate=25600.0,
):
    """Create a mock AITask object that behaves like the real one.

    After the direct-delegation refactor, AITask objects are stored by
    reference — no deep-copy of channels, no settings_file stored on
    the wrapper.  The mock reflects that simplified surface area.
    """
    task = MagicMock()
    task.task_name = task_name
    task.sample_rate = sample_rate
    task.channel_list = ["ch0", "ch1"]
    task.number_of_ch = 2
    task.device_list = ["cDAQ1Mod1"]
    task.configure = MagicMock()

    # Mock the underlying nidaqmx task (set after configure())
    inner_task = MagicMock()
    inner_task.channel_names = ["ch0", "ch1"]
    inner_task.number_of_channels = 2
    inner_task.devices = [MagicMock(name="cDAQ1Mod1")]
    inner_task.devices[0].name = "cDAQ1Mod1"
    inner_task.timing = MagicMock()
    inner_task.timing.samp_clk_rate = sample_rate
    task.task = inner_task

    return task


def _make_mock_ao_task(
    task_name="TestOutput",
    sample_rate=10000.0,
):
    """Create a mock AOTask object.

    After the direct-delegation refactor, AOTask objects are stored
    by reference. Metadata is read via channel_list/number_of_ch properties
    instead of a .channels dict.
    """
    task = MagicMock()
    task.task_name = task_name
    task.sample_rate = sample_rate
    task.channel_list = ["ao0"]
    task.number_of_ch = 1
    task.device_list = ["cDAQ1Mod2"]
    task.configure = MagicMock()

    inner_task = MagicMock()
    inner_task.channel_names = ["ao0"]
    inner_task.number_of_channels = 1
    inner_task.devices = [MagicMock(name="cDAQ1Mod2")]
    inner_task.devices[0].name = "cDAQ1Mod2"
    inner_task.timing = MagicMock()
    inner_task.timing.samp_clk_rate = sample_rate
    task.task = inner_task

    return task


class _FakeDITask:
    """Fake DITask for testing isinstance() checks.

    MagicMock() instances each get unique metaclasses, making
    ``isinstance(mock2, type(mock1))`` fail.  Using a concrete class
    ensures all instances share the same type.

    After the direct-delegation refactor, connect() calls configure() instead
    of start() on digital task objects.
    """

    def __init__(self, task_name="di_test", sample_rate=None):
        self.task_name = task_name
        self.sample_rate = sample_rate
        self.mode = "on_demand" if sample_rate is None else "clocked"
        self.task = None
        self.configure = MagicMock()
        self.read = MagicMock(return_value=np.array([True, False]))
        self.clear_task = MagicMock()


class _FakeDOTask:
    """Fake DOTask for testing isinstance() checks.

    After the direct-delegation refactor, connect() calls configure() instead
    of start() on digital task objects.
    """

    def __init__(self, task_name="do_test", sample_rate=None):
        self.task_name = task_name
        self.sample_rate = sample_rate
        self.mode = "on_demand" if sample_rate is None else "clocked"
        self.task = None
        self.configure = MagicMock()
        self.write = MagicMock()
        self.clear_task = MagicMock()


def _make_mock_digital_input(task_name="di_test", sample_rate=None):
    """Create a fake DITask object."""
    return _FakeDITask(task_name=task_name, sample_rate=sample_rate)


def _make_mock_digital_output(task_name="do_test", sample_rate=None):
    """Create a fake DOTask object."""
    return _FakeDOTask(task_name=task_name, sample_rate=sample_rate)


def _make_mock_nidaqmx_task(channel_names=None, sample_rate=25600.0, device_names=None):
    """Create a mock nidaqmx.task.Task (loaded from NI MAX)."""
    task = MagicMock()
    task.channel_names = channel_names or ["ch0", "ch1"]
    task.number_of_channels = len(task.channel_names)

    devices = []
    for name in (device_names or ["cDAQ1Mod1"]):
        d = MagicMock()
        d.name = name
        devices.append(d)
    task.devices = devices

    task.timing = MagicMock()
    task.timing.samp_clk_rate = sample_rate
    task.is_task_done.return_value = True

    return task


# ---------------------------------------------------------------------------
# Module-level patches
# ---------------------------------------------------------------------------

@pytest.fixture
def wrapper_module():
    """Import the wrapper module with nidaqmx mocked.

    Carefully saves and restores ALL affected sys.modules entries so
    that other test files (test_utils, test_digital, etc.) are not
    polluted by the mock nidaqmx references.
    """
    import sys

    # Modules that will be mocked or indirectly affected
    _MOCK_TARGETS = [
        "nidaqmx", "nidaqmx.constants", "nidaqmx.system",
        "nidaqmx.task", "nidaqmx.errors",
        "pyTrigger",
    ]
    _NIDAQWRAPPER_MODULES = [
        "nidaqwrapper", "nidaqwrapper.handler", "nidaqwrapper.utils",
        "nidaqwrapper.ai_task", "nidaqwrapper.ao_task",
        "nidaqwrapper.digital",
    ]

    # Save current state
    saved = {}
    for mod_name in _MOCK_TARGETS + _NIDAQWRAPPER_MODULES:
        saved[mod_name] = sys.modules.get(mod_name)

    # Mock nidaqmx
    mock_nidaqmx = MagicMock()
    mock_nidaqmx.constants.READ_ALL_AVAILABLE = -1
    mock_nidaqmx.constants.AcquisitionType.CONTINUOUS = "CONTINUOUS"

    # Real exception class so `from nidaqmx.errors import DaqError` yields
    # something usable in `except` clauses (a MagicMock would raise
    # "catching classes that do not inherit from BaseException").
    mock_nidaqmx.errors.DaqError = type("DaqError", (Exception,), {})

    for mod_name in _MOCK_TARGETS:
        if mod_name == "nidaqmx":
            sys.modules[mod_name] = mock_nidaqmx
        elif mod_name == "nidaqmx.errors":
            sys.modules[mod_name] = mock_nidaqmx.errors
        elif mod_name == "pyTrigger":
            sys.modules[mod_name] = MagicMock()
        else:
            sys.modules[mod_name] = MagicMock()

    # Remove cached nidaqwrapper modules so they reimport with mocked nidaqmx
    for mod_name in _NIDAQWRAPPER_MODULES:
        sys.modules.pop(mod_name, None)

    import nidaqwrapper.handler as wrapper_mod

    yield wrapper_mod

    # Restore ALL modules to their pre-fixture state
    for mod_name, original in saved.items():
        if original is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original


@pytest.fixture
def DAQHandler(wrapper_module):
    """Return the DAQHandler class."""
    return wrapper_module.DAQHandler


# ===================================================================
# 1. Constructor and Dual Initialization
# ===================================================================

class TestConstructor:
    """Task group 1: Constructor and dual initialization."""

    def test_empty_constructor_state(self, DAQHandler):
        """1.1 Empty constructor sets correct default state."""
        w = DAQHandler()
        assert w._configured is False
        assert w._connected is False
        assert w._trigger_is_set is False
        assert isinstance(w._lock, type(threading.RLock()))

    def test_empty_constructor_timing_defaults(self, DAQHandler):
        """1.1 Default timing parameters."""
        w = DAQHandler()
        assert w.acquisition_sleep == 0.01
        assert w.post_trigger_delay == 0.05

    def test_constructor_with_task_in_calls_configure(self, DAQHandler):
        """1.2 Constructor with task_in kwarg calls configure."""
        w = DAQHandler(task_in="MyTask")
        assert w._configured is True
        assert w._task_in_name == "MyTask"

    def test_constructor_with_task_out_calls_configure(self, DAQHandler):
        """1.2 Constructor with task_out kwarg calls configure."""
        w = DAQHandler(task_out="MyOutput")
        assert w._configured is True

    def test_constructor_with_both_tasks(self, DAQHandler):
        """1.2 Constructor with both task_in and task_out."""
        w = DAQHandler(task_in="Input", task_out="Output")
        assert w._configured is True
        assert w._task_in_name == "Input"
        assert w._task_out_name == "Output"

    def test_constructor_custom_timing(self, DAQHandler):
        """1.3 Constructor with custom timing kwargs."""
        w = DAQHandler(acquisition_sleep=0.005, post_trigger_delay=0.1)
        assert w.acquisition_sleep == 0.005
        assert w.post_trigger_delay == 0.1

    def test_constructor_does_not_call_connect(self, DAQHandler):
        """Constructor should NOT call connect() even with kwargs."""
        w = DAQHandler(task_in="MyTask")
        assert w._connected is False
        assert w._connect_called is False


# ===================================================================
# 2. configure() with NI MAX Task Name Strings
# ===================================================================

class TestConfigureStrings:
    """Task group 2: configure() with NI MAX task name strings."""

    def test_configure_input_string(self, DAQHandler):
        """2.1 configure(task_in=string) stores name and sets flag."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        assert w._task_in_name == "InputTask"
        assert w._task_in_is_str is True
        assert w._task_in_is_obj is False
        assert w._configured is True

    def test_configure_output_string(self, DAQHandler):
        """2.2 configure(task_out=string) stores output task name."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        assert w._task_out_name == "OutputTask"
        assert w._task_out_is_str is True

    def test_configure_both_strings(self, DAQHandler):
        """2.3 configure() with both input and output strings."""
        w = DAQHandler()
        w.configure(task_in="Input", task_out="Output")
        assert w._task_in_name == "Input"
        assert w._task_out_name == "Output"
        assert w._configured is True

    def test_configure_resets_state(self, DAQHandler):
        """2.4 Calling configure() twice resets previous state."""
        w = DAQHandler()
        w.configure(task_in="First")
        w._trigger_is_set = True
        w.configure(task_in="Second")
        assert w._task_in_name == "Second"
        assert w._trigger_is_set is False

    def test_configure_timing_kwargs(self, DAQHandler):
        """2.5 configure() with timing kwargs overrides defaults."""
        w = DAQHandler()
        w.configure(task_in="Task", acquisition_sleep=0.02, post_trigger_delay=0.1)
        assert w.acquisition_sleep == 0.02
        assert w.post_trigger_delay == 0.1


# ===================================================================
# 3. configure() with AITask/AOTask Objects
# ===================================================================

class TestConfigureObjects:
    """Task group 3: configure() with AITask/AOTask objects."""

    def test_configure_ni_task_object(self, DAQHandler, wrapper_module):
        """3.1 configure(task_in=AITask) stores object and extracts config."""
        mock_task = _make_mock_ai_task()

        with patch.object(wrapper_module, "AITask", new=type(mock_task)):
            w = DAQHandler()
            w.configure(task_in=mock_task)

        assert w._task_in_obj is mock_task
        assert w._task_in_is_obj is True
        assert w._task_in_is_str is False
        assert w._task_in_name_str == "TestInput"
        assert w._task_in_sample_rate == 25600.0
        # Direct-delegation: object is stored by reference, no deep copy

    def test_configure_ao_task_object(self, DAQHandler, wrapper_module):
        """3.2 configure(task_out=AOTask) stores output object."""
        mock_task_out = _make_mock_ao_task()

        with patch.object(wrapper_module, "AOTask", new=type(mock_task_out)):
            w = DAQHandler()
            w.configure(task_out=mock_task_out)

        assert w._task_out_obj is mock_task_out
        assert w._task_out_is_obj is True

    def test_configure_mixed_types(self, DAQHandler, wrapper_module):
        """3.3 configure() with mixed types: AITask + string output."""
        mock_task = _make_mock_ai_task()

        with patch.object(wrapper_module, "AITask", new=type(mock_task)):
            w = DAQHandler()
            w.configure(task_in=mock_task, task_out="OutputTask")

        assert w._task_in_is_obj is True
        assert w._task_out_is_str is True

    def test_configure_stores_reference_without_deep_copy(
        self, DAQHandler, wrapper_module
    ):
        """3.4 configure() stores the AITask reference directly (no deep copy)."""
        mock_task = _make_mock_ai_task()

        with patch.object(wrapper_module, "AITask", new=type(mock_task)):
            w = DAQHandler()
            w.configure(task_in=mock_task)

        # Direct-delegation: stored object IS the same reference
        assert w._task_in_obj is mock_task

    def test_configure_no_deep_copy_attrs(self, DAQHandler, wrapper_module):
        """3.5 configure() does not create _task_in_channels or _task_in_settings_file."""
        mock_task = _make_mock_ai_task()

        with patch.object(wrapper_module, "AITask", new=type(mock_task)):
            w = DAQHandler()
            w.configure(task_in=mock_task)

        # These attributes were removed as part of the direct-delegation refactor
        assert not hasattr(w, "_task_in_channels") or w._task_in_channels is None
        assert (
            not hasattr(w, "_task_in_settings_file")
            or w._task_in_settings_file is None
        )

    def test_configure_no_deep_copy_output_attrs(self, DAQHandler, wrapper_module):
        """3.6 configure() does not create _task_out_channels for output tasks."""
        mock_task_out = _make_mock_ao_task()

        with patch.object(wrapper_module, "AOTask", new=type(mock_task_out)):
            w = DAQHandler()
            w.configure(task_out=mock_task_out)

        assert not hasattr(w, "_task_out_channels") or w._task_out_channels is None


# ===================================================================
# 4. connect() and disconnect() Lifecycle
# ===================================================================

class TestConnectDisconnect:
    """Task group 4: connect() and disconnect() lifecycle."""

    def test_connect_with_string_loads_task(self, DAQHandler, wrapper_module):
        """4.1 connect() with NI MAX string calls get_task_by_name."""
        w = DAQHandler()
        w.configure(task_in="InputTask")

        mock_loaded_task = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded_task) as mock_get, \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            result = w.connect()

        mock_get.assert_called_once_with("InputTask")
        assert result is True
        assert w._connected is True

    def test_connect_extracts_metadata_from_ni_max_task(self, DAQHandler, wrapper_module):
        """4.1 connect() extracts channel names, sample rate, devices."""
        w = DAQHandler()
        w.configure(task_in="InputTask")

        mock_loaded_task = _make_mock_nidaqmx_task(
            channel_names=["accel_x", "accel_y"],
            sample_rate=51200.0,
            device_names=["cDAQ1Mod1"],
        )
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded_task), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        assert w._channel_names_in == ["accel_x", "accel_y"]
        assert w._sample_rate_in == 51200.0
        assert "cDAQ1Mod1" in w._required_devices

    def test_connect_with_ni_task_object_calls_configure(self, DAQHandler, wrapper_module):
        """4.2 connect() with AITask object calls configure() on the stored reference.

        After the direct-delegation refactor, connect() calls configure() directly
        on the stored AITask object — no recreation, no constructor call.
        """
        mock_task = _make_mock_ai_task()

        with patch.object(wrapper_module, "AITask", new=type(mock_task)):
            w = DAQHandler()
            w.configure(task_in=mock_task)

        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        # configure() must be called — Pitfall #5: wrapper must not auto-start
        mock_task.configure.assert_called_once()

    def test_connect_closes_previous_tasks(self, DAQHandler, wrapper_module):
        """4.3 connect() closes previous tasks before opening new ones."""
        w = DAQHandler()
        w.configure(task_in="InputTask")

        mock_first_task = _make_mock_nidaqmx_task()
        mock_second_task = _make_mock_nidaqmx_task()

        with patch.object(wrapper_module, "get_task_by_name", side_effect=[mock_first_task, mock_second_task]), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()
            # First task is now the active _task_in
            w.connect()  # Second connect should close first task

        mock_first_task.close.assert_called_once()

    def test_connect_returns_false_on_failure(self, DAQHandler, wrapper_module):
        """4.4 connect() returns False when task load fails."""
        w = DAQHandler()
        w.configure(task_in="BadTask")

        with patch.object(wrapper_module, "get_task_by_name", side_effect=KeyError("Not found")):
            result = w.connect()

        assert result is False
        assert w._connected is False

    def test_connect_returns_false_when_ping_fails(self, DAQHandler, wrapper_module):
        """4.4 connect() returns False when ping fails."""
        w = DAQHandler()
        w.configure(task_in="InputTask")

        mock_loaded_task = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded_task), \
             patch.object(wrapper_module, "get_connected_devices", return_value=set()):
            result = w.connect()

        assert result is False

    def test_connect_with_obj_calls_configure_directly(self, DAQHandler, wrapper_module):
        """4.5 connect() calls configure() directly on stored AITask — no recreation."""
        mock_task = _make_mock_ai_task()

        with patch.object(wrapper_module, "AITask", new=type(mock_task)):
            w = DAQHandler()
            w.configure(task_in=mock_task)

        # No AITask constructor patch needed: recreation was removed
        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        # The stored reference itself receives configure(), not a new instance
        mock_task.configure.assert_called_once()
        assert w._task_in_obj is mock_task

    def test_connect_with_ao_task_object_calls_configure(
        self, DAQHandler, wrapper_module
    ):
        """4.x connect() calls configure() on stored AOTask object."""
        mock_task_out = _make_mock_ao_task()

        with patch.object(wrapper_module, "AOTask", new=type(mock_task_out)):
            w = DAQHandler()
            w.configure(task_out=mock_task_out)

        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod2"}):
            w.connect()

        # Pitfall #5: wrapper must not auto-start output tasks either
        mock_task_out.configure.assert_called_once()

    def test_disconnect_closes_input_task(self, DAQHandler, wrapper_module):
        """4.6 disconnect() closes input task."""
        w = DAQHandler()
        w.configure(task_in="InputTask")

        mock_loaded = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        w.disconnect()
        mock_loaded.close.assert_called()
        assert w._connected is False

    def test_disconnect_closes_output_task(self, DAQHandler, wrapper_module):
        """4.7 disconnect() closes output task."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")

        mock_loaded = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        w.disconnect()
        mock_loaded.close.assert_called()

    def test_connect_with_both_input_and_output_tasks(self, DAQHandler, wrapper_module):
        """4.x connect() loads both input and output NI MAX tasks simultaneously."""
        w = DAQHandler()
        w.configure(task_in="InputTask", task_out="OutputTask")

        mock_input = _make_mock_nidaqmx_task(
            channel_names=["ch0", "ch1"],
            sample_rate=25600.0,
            device_names=["cDAQ1Mod1"],
        )
        mock_output = _make_mock_nidaqmx_task(
            channel_names=["ao0"],
            sample_rate=10000.0,
            device_names=["cDAQ1Mod2"],
        )

        def _side_effect(name):
            return mock_input if name == "InputTask" else mock_output

        with patch.object(wrapper_module, "get_task_by_name", side_effect=_side_effect), \
             patch.object(wrapper_module, "get_connected_devices",
                          return_value={"cDAQ1Mod1", "cDAQ1Mod2"}):
            result = w.connect()

        assert result is True
        assert w._connected is True
        assert w._task_in is mock_input
        assert w._task_out is mock_output
        assert w._channel_names_in == ["ch0", "ch1"]
        assert w._channel_names_out == ["ao0"]

    def test_disconnect_idempotent(self, DAQHandler):
        """4.8 Calling disconnect() twice does not raise."""
        w = DAQHandler()
        result1 = w.disconnect()
        result2 = w.disconnect()
        assert result1 is True
        assert result2 is True

    def test_disconnect_when_never_connected(self, DAQHandler):
        """4.9 disconnect() when no tasks were ever connected."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        result = w.disconnect()
        assert result is True


# ===================================================================
# 5. set_trigger() with pyTrigger
# ===================================================================

class TestSetTrigger:
    """Task group 5: set_trigger() with pyTrigger."""

    def test_set_trigger_creates_pytrigger(self, DAQHandler, wrapper_module):
        """5.1 set_trigger() creates pyTrigger with correct params."""
        w = DAQHandler()
        w.configure(task_in="InputTask")

        # Simulate being connected with 2 channels
        w._connected = True
        w._n_channels_in = 2
        w._channel_names_in = ["ch0", "ch1"]

        mock_pytrigger_cls = MagicMock()
        with patch.object(wrapper_module, "pyTrigger", mock_pytrigger_cls):
            w.set_trigger(
                n_samples=1000,
                trigger_channel=0,
                trigger_level=0.5,
                trigger_type="abs",
                presamples=100,
            )

        mock_pytrigger_cls.assert_called_once_with(
            rows=1000,
            channels=2,
            trigger_channel=0,
            trigger_level=0.5,
            trigger_type="abs",
            presamples=100,
        )
        assert w._trigger_is_set is True

    def test_set_trigger_defaults(self, DAQHandler, wrapper_module):
        """5.2 set_trigger() with defaults: trigger_type='abs', presamples=0."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2
        w._channel_names_in = ["ch0", "ch1"]

        mock_pytrigger_cls = MagicMock()
        with patch.object(wrapper_module, "pyTrigger", mock_pytrigger_cls):
            w.set_trigger(n_samples=500, trigger_channel=0, trigger_level=1.0)

        mock_pytrigger_cls.assert_called_once_with(
            rows=500,
            channels=2,
            trigger_channel=0,
            trigger_level=1.0,
            trigger_type="abs",
            presamples=0,
        )

    def test_set_trigger_no_input_task_raises(self, DAQHandler):
        """5.3 set_trigger() raises ValueError with no input task."""
        w = DAQHandler()
        with pytest.raises(ValueError, match="input task"):
            w.set_trigger(n_samples=100, trigger_channel=0, trigger_level=0.5)

    def test_set_trigger_sets_flag(self, DAQHandler, wrapper_module):
        """5.4 set_trigger() sets _trigger_is_set=True."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 1
        w._channel_names_in = ["ch0"]

        with patch.object(wrapper_module, "pyTrigger", MagicMock()):
            w.set_trigger(n_samples=100, trigger_channel=0, trigger_level=0.5)

        assert w._trigger_is_set is True


# ===================================================================
# 6. acquire() RuntimeError Guard
# ===================================================================

class TestAcquireGuard:
    """Task group 6: acquire() RuntimeError guard."""

    def test_acquire_raises_without_trigger(self, DAQHandler):
        """6.1 acquire() raises RuntimeError when trigger not set."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2

        with pytest.raises(RuntimeError, match="set_trigger"):
            w.acquire()

    def test_acquire_raises_without_input_task(self, DAQHandler):
        """6.2 acquire() raises ValueError when no input task configured."""
        w = DAQHandler()
        with pytest.raises(ValueError, match="input task"):
            w.acquire()


# ===================================================================
# 7. acquire() Triggered Loop
# ===================================================================

class TestAcquireLoop:
    """Task group 7: acquire() triggered acquisition loop."""

    def _setup_acquire(self, DAQHandler, wrapper_module, n_channels=2):
        """Helper: set up a wrapper ready for acquire().

        Uses add_data side effect to set trigger.finished = True on first call,
        simulating immediate trigger completion.
        """
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._connect_called = True
        w._n_channels_in = n_channels
        w._channel_names_in = [f"ch{i}" for i in range(n_channels)]
        w._sample_rate_in = 25600.0
        w.acquisition_sleep = 0  # speed up tests
        w.post_trigger_delay = 0

        # Mock the underlying nidaqmx task
        mock_inner_task = MagicMock()
        if n_channels == 1:
            mock_inner_task.read.return_value = [0.0] * 100
        else:
            mock_inner_task.read.return_value = [[0.0] * 100] * n_channels
        w._task_in = mock_inner_task

        # Mock trigger — add_data side effect finishes trigger on first call
        mock_trigger = MagicMock()
        mock_trigger.finished = False
        mock_trigger.rows = 1000

        def _add_data_finishes(data):
            mock_trigger.finished = True

        mock_trigger.add_data.side_effect = _add_data_finishes
        mock_trigger.get_data.return_value = np.zeros((1000, n_channels))
        w.trigger = mock_trigger
        w._trigger_is_set = True

        return w, mock_inner_task, mock_trigger

    def test_acquire_starts_task(self, DAQHandler, wrapper_module):
        """7.1 acquire() starts the input task."""
        w, mock_task, _ = self._setup_acquire(DAQHandler, wrapper_module)
        w.acquire()
        mock_task.start.assert_called_once()

    def test_acquire_flushes_buffer(self, DAQHandler, wrapper_module):
        """7.2 acquire() flushes buffer (first read discarded)."""
        w, mock_task, _ = self._setup_acquire(DAQHandler, wrapper_module)
        w.acquire()
        # First read is the flush + at least one loop read = >= 2 total reads
        assert mock_task.read.call_count >= 2

    def test_acquire_loop_until_trigger_finished(self, DAQHandler, wrapper_module):
        """7.3 acquire() reads in loop until trigger.finished."""
        w, mock_task, mock_trigger = self._setup_acquire(DAQHandler, wrapper_module)

        # Override: trigger finishes after 3 add_data calls
        call_count = [0]
        def side_effect(data):
            call_count[0] += 1
            if call_count[0] >= 3:
                mock_trigger.finished = True

        mock_trigger.add_data.side_effect = side_effect
        w.acquire()
        assert call_count[0] >= 3

    def test_acquire_single_channel_reshape(self, DAQHandler, wrapper_module):
        """7.4 acquire() reshapes single-channel 1D data to 2D."""
        w, _, mock_trigger = self._setup_acquire(DAQHandler, wrapper_module, n_channels=1)
        mock_trigger.get_data.return_value = np.zeros((1000, 1))

        # Capture what add_data receives
        added_data = []
        def capture(data):
            added_data.append(data.copy())
            mock_trigger.finished = True
        mock_trigger.add_data.side_effect = capture

        result = w.acquire()
        assert result.shape == (1000, 1)
        # add_data should have received (n_samples, 1) shaped data
        assert added_data[0].ndim == 2
        assert added_data[0].shape[1] == 1

    def test_acquire_multi_channel_transpose(self, DAQHandler, wrapper_module):
        """7.5 acquire() transposes multi-channel 2D data for pyTrigger."""
        w, mock_task, mock_trigger = self._setup_acquire(DAQHandler, wrapper_module)

        added_data = []
        def capture(data):
            added_data.append(data.copy())
            mock_trigger.finished = True
        mock_trigger.add_data.side_effect = capture
        mock_task.read.return_value = [list(range(100)), list(range(100, 200))]

        w.acquire()

        # add_data should have received (n_samples, n_channels) format
        assert len(added_data) > 0
        assert added_data[0].shape == (100, 2)

    def test_acquire_post_trigger_delay(self, DAQHandler, wrapper_module):
        """7.6 acquire() sleeps for post_trigger_delay after trigger finishes."""
        w, _, _ = self._setup_acquire(DAQHandler, wrapper_module)
        w.post_trigger_delay = 0.123

        with patch.object(wrapper_module.time, "sleep") as mock_sleep:
            w.acquire()

        # The post-trigger sleep must be called with the configured delay.
        # (acquisition_sleep may also be called, so check any call matches)
        sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
        assert 0.123 in sleep_values, (
            f"Expected time.sleep(0.123) for post_trigger_delay; "
            f"actual sleep calls: {sleep_values}"
        )

    def test_acquire_stops_task(self, DAQHandler, wrapper_module):
        """7.7 acquire() stops the input task after acquisition."""
        w, mock_task, _ = self._setup_acquire(DAQHandler, wrapper_module)
        w.acquire()
        mock_task.stop.assert_called_once()

    def test_acquire_returns_correct_shape(self, DAQHandler, wrapper_module):
        """7.8 acquire() returns (n_samples, n_channels) array."""
        w, _, mock_trigger = self._setup_acquire(DAQHandler, wrapper_module)
        expected = np.random.randn(1000, 2)
        mock_trigger.get_data.return_value = expected

        result = w.acquire()
        assert result.shape == (1000, 2)
        np.testing.assert_array_equal(result, expected)

    def test_acquire_resets_trigger_on_second_call(self, DAQHandler, wrapper_module):
        """7.9 acquire() resets trigger on second call."""
        w, _, mock_trigger = self._setup_acquire(DAQHandler, wrapper_module)

        # First acquire
        w.acquire()
        # Reset side effect for second call
        mock_trigger.finished = False
        mock_trigger.add_data.side_effect = lambda data: setattr(mock_trigger, 'finished', True)

        w.acquire()
        # ringbuff.clear should be called at least twice (once per _reset_trigger)
        assert mock_trigger.ringbuff.clear.call_count >= 2


# ===================================================================
# 8. acquire() with return_dict=True
# ===================================================================

class TestAcquireReturnDict:
    """Task group 8: acquire() with return_dict=True."""

    def _setup_acquire_dict(self, DAQHandler, wrapper_module):
        """Helper: set up wrapper for dict-return acquire."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._connect_called = True
        w._n_channels_in = 2
        w._channel_names_in = ["accel_x", "accel_y"]
        w._sample_rate_in = 1000.0
        w.acquisition_sleep = 0
        w.post_trigger_delay = 0

        mock_inner_task = MagicMock()
        mock_inner_task.read.return_value = [[0.0] * 50, [0.0] * 50]
        w._task_in = mock_inner_task

        mock_trigger = MagicMock()
        mock_trigger.finished = False
        mock_trigger.rows = 100
        mock_trigger.add_data.side_effect = lambda data: setattr(mock_trigger, 'finished', True)
        w.trigger = mock_trigger
        w._trigger_is_set = True

        return w, mock_inner_task, mock_trigger

    def test_return_dict_has_channel_keys(self, DAQHandler, wrapper_module):
        """8.1 acquire(return_dict=True) returns dict with channel names."""
        w, _, mock_trigger = self._setup_acquire_dict(DAQHandler, wrapper_module)
        mock_trigger.get_data.return_value = np.random.randn(100, 2)

        result = w.acquire(return_dict=True)

        assert isinstance(result, dict)
        assert "accel_x" in result
        assert "accel_y" in result

    def test_return_dict_has_time_key(self, DAQHandler, wrapper_module):
        """8.2 Dict includes 'time' key with correct values."""
        w, _, mock_trigger = self._setup_acquire_dict(DAQHandler, wrapper_module)
        mock_trigger.get_data.return_value = np.random.randn(100, 2)

        result = w.acquire(return_dict=True)

        assert "time" in result
        expected_time = np.arange(100) / 1000.0
        np.testing.assert_array_almost_equal(result["time"], expected_time)

    def test_return_dict_channel_values_1d(self, DAQHandler, wrapper_module):
        """8.3 Each channel value is a 1D numpy array."""
        w, _, mock_trigger = self._setup_acquire_dict(DAQHandler, wrapper_module)
        data = np.random.randn(100, 2)
        mock_trigger.get_data.return_value = data

        result = w.acquire(return_dict=True)

        assert result["accel_x"].ndim == 1
        assert len(result["accel_x"]) == 100
        np.testing.assert_array_equal(result["accel_x"], data[:, 0])


# ===================================================================
# 9. acquire() with blocking=False (Future)
# ===================================================================

class TestAcquireNonBlocking:
    """Task group 9: acquire() with blocking=False."""

    def _setup_nb(self, DAQHandler, wrapper_module):
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._connect_called = True
        w._n_channels_in = 2
        w._channel_names_in = ["ch0", "ch1"]
        w._sample_rate_in = 25600.0
        w.acquisition_sleep = 0
        w.post_trigger_delay = 0

        mock_inner_task = MagicMock()
        mock_inner_task.read.return_value = [[0.0] * 50, [0.0] * 50]
        w._task_in = mock_inner_task

        mock_trigger = MagicMock()
        mock_trigger.finished = False
        mock_trigger.rows = 100
        mock_trigger.add_data.side_effect = lambda data: setattr(mock_trigger, 'finished', True)
        mock_trigger.get_data.return_value = np.zeros((100, 2))
        w.trigger = mock_trigger
        w._trigger_is_set = True

        return w, mock_inner_task, mock_trigger

    def test_blocking_true_returns_data(self, DAQHandler, wrapper_module):
        """9.1 acquire(blocking=True) returns data directly."""
        w, _, _ = self._setup_nb(DAQHandler, wrapper_module)
        result = w.acquire(blocking=True)
        assert isinstance(result, np.ndarray)

    def test_blocking_false_returns_future(self, DAQHandler, wrapper_module):
        """9.2 acquire(blocking=False) returns Future."""
        w, _, _ = self._setup_nb(DAQHandler, wrapper_module)
        result = w.acquire(blocking=False)
        assert isinstance(result, Future)

    def test_future_result_returns_data(self, DAQHandler, wrapper_module):
        """9.3 future.result() returns acquired data."""
        w, _, _ = self._setup_nb(DAQHandler, wrapper_module)
        future = w.acquire(blocking=False)
        data = future.result(timeout=5)
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 2)

    def test_future_done_after_completion(self, DAQHandler, wrapper_module):
        """9.4 future.done() returns True after acquisition completes."""
        w, _, _ = self._setup_nb(DAQHandler, wrapper_module)
        future = w.acquire(blocking=False)
        # Wait for acquisition to finish before checking done()
        future.result(timeout=5)
        assert future.done() is True


# ===================================================================
# 10. acquire() for LDAQ Integration
# ===================================================================

class TestReadAllAvailable:
    """Task group 10: acquire() for LDAQ integration."""

    def test_acquire_transposes(self, DAQHandler, wrapper_module):
        """10.1 read_all_available() returns (n_samples, n_channels) without double-transpose."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True

        # AITask.acquire() now returns (n_samples, n_channels) — 3 samples, 2 channels
        mock_ni_task = MagicMock()
        mock_ni_task.acquire.return_value = np.array([[1, 4], [2, 5], [3, 6]])
        w._task_in_obj_active = mock_ni_task

        result = w.read_all_available()
        assert result.shape == (3, 2)  # (n_samples, n_channels)
        # Verify data passes through without double-transposition
        np.testing.assert_array_equal(result, np.array([[1, 4], [2, 5], [3, 6]]))

    def test_acquire_empty(self, DAQHandler, wrapper_module):
        """10.2 read_all_available() returns empty array when no data."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2

        mock_ni_task = MagicMock()
        # AITask.acquire() returns (0, 2) for empty buffer with 2 channels
        mock_ni_task.acquire.return_value = np.empty((0, 2))
        w._task_in_obj_active = mock_ni_task

        result = w.read_all_available()
        assert result.shape == (0, 2)

    def test_acquire_no_trigger(self, DAQHandler, wrapper_module):
        """10.3 read_all_available() does NOT use pyTrigger."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True

        mock_ni_task = MagicMock()
        # AITask.acquire() returns (n_samples, n_channels) — 2 samples, 2 channels
        mock_ni_task.acquire.return_value = np.array([[1, 3], [2, 4]])
        w._task_in_obj_active = mock_ni_task

        # No trigger should be involved
        w._trigger_is_set = False
        result = w.read_all_available()
        assert result.shape == (2, 2)

    def test_acquire_single_channel(self, DAQHandler, wrapper_module):
        """10.4 read_all_available() passes through single-channel (n_samples, 1)."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 1

        mock_ni_task = MagicMock()
        # AITask.acquire() returns (n_samples, 1) for single channel
        mock_ni_task.acquire.return_value = np.array([[1], [2], [3], [4]])
        w._task_in_obj_active = mock_ni_task

        result = w.read_all_available()
        assert result.shape == (4, 1)


# ===================================================================
# 11. read() Single Sample
# ===================================================================

class TestRead:
    """Task group 11: read() single sample."""

    def test_read_returns_single_sample(self, DAQHandler, wrapper_module):
        """11.1 read() returns (n_channels,) array."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2

        mock_task = MagicMock()
        mock_task.read.return_value = [1.5, 2.5]
        w._task_in = mock_task

        result = w.read()
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, [1.5, 2.5])

    def test_read_raises_during_continuous_acquisition(self, DAQHandler, wrapper_module):
        """11.2 read() raises ValueError when continuous acquisition is running."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 2

        mock_task = MagicMock()
        mock_task.read.return_value = [1.5, 2.5]
        w._task_in = mock_task

        # Simulate a continuous acquisition in progress
        w._acquire_running = True

        with pytest.raises(ValueError, match="acquisition"):
            w.read()

    def test_read_no_input_task_raises(self, DAQHandler):
        """11.3 read() raises ValueError when no input task configured."""
        w = DAQHandler()
        with pytest.raises(ValueError, match="input task"):
            w.read()

    def test_read_single_channel_scalar_returns_1d(self, DAQHandler, wrapper_module):
        """11.4 read() reshapes 0-D scalar result to (1,) for single-channel tasks.

        nidaqmx may return a bare float for a single-channel task when 1 sample
        is requested. np.array(scalar) produces a 0-D array; this must be
        reshaped to (1,) so callers always receive a 1-D array.
        """
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._n_channels_in = 1

        mock_task = MagicMock()
        # Simulate nidaqmx returning a bare float (0-D after np.array())
        mock_task.read.return_value = 3.14
        w._task_in = mock_task

        result = w.read()
        assert result.ndim == 1
        assert result.shape == (1,)
        np.testing.assert_almost_equal(result[0], 3.14)


# ===================================================================
# 12. generate() Continuous with Buffer Management
# ===================================================================

class TestGenerate:
    """Task group 12: generate() continuous generation."""

    def test_generate_2d_data(self, DAQHandler, wrapper_module):
        """12.1 generate() with 2D data: transposes, writes, and is C-contiguous.

        Pitfall #1: data.T produces a Fortran-order view; nidaqmx C layer
        requires C-contiguous memory. np.ascontiguousarray() is mandatory.
        """
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._task_out_is_str = True
        w._n_channels_out = 2
        w._generation_running = False

        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        w.generate(data)

        mock_task.write.assert_called_once()
        written = mock_task.write.call_args[0][0]
        assert written.shape == (2, 3)  # transposed to (n_channels, n_samples)
        assert written.flags["C_CONTIGUOUS"], (
            "Written array must be C-contiguous (np.ascontiguousarray required "
            "after .T to avoid nidaqmx C layer memory errors — see Pitfall #1)"
        )

    def test_generate_1d_single_channel(self, DAQHandler, wrapper_module):
        """12.2 generate() with 1D data for single channel."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._task_out_is_str = True
        w._n_channels_out = 1
        w._generation_running = False

        data = np.array([1.0, 2.0, 3.0])
        w.generate(data)

        mock_task.write.assert_called_once()

    def test_generate_overwrite_stops_first(self, DAQHandler, wrapper_module):
        """12.3 generate(overwrite=True) stops current output first."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._task_out_is_str = True
        w._n_channels_out = 1
        w._generation_running = True

        data = np.array([1.0, 2.0, 3.0])
        w.generate(data, overwrite=True)

        mock_task.stop.assert_called()

    def test_generate_no_overwrite_raises_when_running(self, DAQHandler, wrapper_module):
        """12.4 generate(overwrite=False) raises when output is running."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = False
        w._task_out = mock_task
        w._n_channels_out = 1
        w._generation_running = True

        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="running"):
            w.generate(data, overwrite=False)

    def test_generate_no_output_task_raises(self, DAQHandler):
        """12.5 generate() raises ValueError with no output task."""
        w = DAQHandler()
        with pytest.raises(ValueError, match="output task"):
            w.generate(np.array([1.0]))

    def test_generate_shape_mismatch_raises(self, DAQHandler, wrapper_module):
        """12.6 generate() validates data shape vs channel count."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = False

        # 3 columns but only 2 output channels
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="channels"):
            w.generate(data)


# ===================================================================
# 13. write() Single Sample with 2-Sample Minimum
# ===================================================================

class TestWrite:
    """Task group 13: write() single sample."""

    def test_write_single_float(self, DAQHandler, wrapper_module):
        """13.1 write(float) for single-channel: duplicated to 2 samples."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._n_channels_out = 1
        w._generation_running = False

        w.write(2.5)

        mock_task.write.assert_called_once()
        written = mock_task.write.call_args[0][0]
        # Should be 2 samples for 1 channel
        assert len(written) == 2

    def test_write_multi_channel(self, DAQHandler, wrapper_module):
        """13.2 write(array) for multi-channel: duplicated to 2 rows, C-contiguous.

        Pitfall #1: stacking + .T produces a Fortran-order view; the written
        array must be C-contiguous for the nidaqmx C layer.
        """
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = False

        w.write(np.array([1.0, 2.0]))

        mock_task.write.assert_called_once()
        written = mock_task.write.call_args[0][0]
        assert written.shape == (2, 2)  # (n_channels, 2_samples)
        assert written.flags["C_CONTIGUOUS"], (
            "Written array must be C-contiguous (np.ascontiguousarray required "
            "after .T — see Pitfall #1)"
        )

    def test_write_overwrite_stops_first(self, DAQHandler, wrapper_module):
        """13.3 write(overwrite=True) stops current generation."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._n_channels_out = 1
        w._generation_running = True

        w.write(1.0, overwrite=True)
        mock_task.stop.assert_called()

    def test_write_shape_mismatch_raises(self, DAQHandler, wrapper_module):
        """13.4 write() raises ValueError for shape mismatch."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = False

        with pytest.raises(ValueError, match="channels"):
            w.write(np.array([1.0, 2.0, 3.0]))  # 3 values, 2 channels

    def test_write_no_overwrite_raises_when_running(self, DAQHandler, wrapper_module):
        """13.5 write() raises ValueError when running and overwrite=False."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = False
        w._task_out = mock_task
        w._n_channels_out = 1
        w._generation_running = True

        with pytest.raises(ValueError, match="running"):
            w.write(1.0, overwrite=False)


# ===================================================================
# 14. stop_generation() with Safe Zeros
# ===================================================================

class TestStopGeneration:
    """Task group 14: stop_generation() with safe zeros."""

    def test_stop_generation_stops_task(self, DAQHandler, wrapper_module):
        """14.1 stop_generation() stops the output task."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = True

        w.stop_generation()
        mock_task.stop.assert_called()

    def test_stop_generation_writes_zeros(self, DAQHandler, wrapper_module):
        """14.2 stop_generation() writes zeros to all channels before stopping."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 2
        w._task_out = mock_task
        w._n_channels_out = 2
        w._generation_running = True

        w.stop_generation()

        mock_task.write.assert_called()
        written = mock_task.write.call_args[0][0]
        # The safe-shutdown zeros write must contain only zero values.
        assert np.all(written == 0.0), (
            f"stop_generation() must write all-zero data; got: {written}"
        )

    def test_stop_generation_no_output_task(self, DAQHandler):
        """14.3 stop_generation() with no output task is no-op."""
        w = DAQHandler()
        # Should not raise
        w.stop_generation()


# ===================================================================
# 15. ping() and check_state()
# ===================================================================

class TestPingAndCheckState:
    """Task group 15: ping() and check_state() with auto-reconnection."""

    def test_ping_true_when_devices_present(self, DAQHandler, wrapper_module):
        """15.1 ping() returns True when required devices are connected."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._required_devices = {"cDAQ1Mod1"}

        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1", "cDAQ1Mod2"}):
            assert w.ping() is True

    def test_ping_false_when_device_missing(self, DAQHandler, wrapper_module):
        """15.2 ping() returns False when device is missing."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._required_devices = {"cDAQ1Mod1"}

        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod2"}):
            assert w.ping() is False

    def test_ping_false_when_not_configured(self, DAQHandler):
        """15.3 ping() returns False when not configured."""
        w = DAQHandler()
        assert w.ping() is False

    def test_check_state_connected(self, DAQHandler, wrapper_module):
        """15.4 check_state() returns 'connected' when all OK."""
        w = DAQHandler()
        w._connected = True
        w._configured = True
        w._required_devices = {"cDAQ1Mod1"}

        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            assert w.check_state() == "connected"

    def test_check_state_reconnected(self, DAQHandler, wrapper_module):
        """15.5 check_state() returns 'reconnected' on successful reconnect."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = False
        w._connect_called = True

        # Make connect() succeed
        mock_loaded = _make_mock_nidaqmx_task()
        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            result = w.check_state()

        assert result == "reconnected"

    def test_check_state_connection_lost(self, DAQHandler, wrapper_module):
        """15.6 check_state() returns 'connection lost' on failed reconnect."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = False
        w._connect_called = True

        # Make connect() fail
        with patch.object(wrapper_module, "get_task_by_name", side_effect=KeyError("No")):
            result = w.check_state()

        assert result == "connection lost"

    def test_check_state_disconnected(self, DAQHandler):
        """15.7 check_state() returns 'disconnected' when never connected."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        assert w.check_state() == "disconnected"


# ===================================================================
# 16. Introspection Methods
# ===================================================================

class TestIntrospection:
    """Task group 16: get_device_info, get_sample_rate, get_channel_names."""

    def test_get_device_info_input(self, DAQHandler):
        """16.1 get_device_info() with input task."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._channel_names_in = ["ch0", "ch1"]
        w._sample_rate_in = 25600.0

        info = w.get_device_info()
        assert "input" in info
        assert info["input"]["channel_names"] == ["ch0", "ch1"]
        assert info["input"]["sample_rate"] == 25600.0

    def test_get_device_info_both_tasks(self, DAQHandler):
        """16.2 get_device_info() with both input and output."""
        w = DAQHandler()
        w.configure(task_in="Input", task_out="Output")
        w._connected = True
        w._channel_names_in = ["ch0"]
        w._sample_rate_in = 25600.0
        w._channel_names_out = ["ao0"]
        w._sample_rate_out = 10000.0

        info = w.get_device_info()
        assert "input" in info
        assert "output" in info

    def test_get_device_info_no_tasks(self, DAQHandler):
        """16.3 get_device_info() with no tasks returns empty dict."""
        w = DAQHandler()
        info = w.get_device_info()
        assert info == {}

    def test_get_sample_rate(self, DAQHandler):
        """16.4 get_sample_rate() returns input sample rate."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._sample_rate_in = 51200.0
        assert w.get_sample_rate() == 51200.0

    def test_get_sample_rate_no_input_raises(self, DAQHandler):
        """16.5 get_sample_rate() raises ValueError with no input task."""
        w = DAQHandler()
        with pytest.raises(ValueError, match="input"):
            w.get_sample_rate()

    def test_get_channel_names(self, DAQHandler):
        """16.6 get_channel_names() returns channel name list."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._channel_names_in = ["accel_x", "accel_y"]
        assert w.get_channel_names() == ["accel_x", "accel_y"]

    def test_get_channel_names_no_input(self, DAQHandler):
        """16.7 get_channel_names() with no input returns empty list."""
        w = DAQHandler()
        assert w.get_channel_names() == []


# ===================================================================
# 17. clear_buffer()
# ===================================================================

class TestClearBuffer:
    """Task group 17: clear_buffer()."""

    def test_clear_buffer_reads_and_discards(self, DAQHandler, wrapper_module):
        """17.1 clear_buffer() reads and discards all buffered data."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True

        mock_ni_task = MagicMock()
        w._task_in_obj_active = mock_ni_task

        w.clear_buffer()
        mock_ni_task.acquire.assert_called_once()

    def test_clear_buffer_no_input_task(self, DAQHandler):
        """17.3 clear_buffer() with no input task is no-op."""
        w = DAQHandler()
        # Should not raise
        w.clear_buffer()


# ===================================================================
# 18. Context Manager
# ===================================================================

class TestContextManager:
    """Task group 18: Context manager protocol."""

    def test_enter_returns_self(self, DAQHandler):
        """18.1 __enter__() returns self."""
        w = DAQHandler()
        assert w.__enter__() is w

    def test_exit_calls_disconnect(self, DAQHandler):
        """18.2 __exit__() calls disconnect()."""
        w = DAQHandler()
        w.disconnect = MagicMock()
        w.__exit__(None, None, None)
        w.disconnect.assert_called_once()

    def test_exit_does_not_suppress_exceptions(self, DAQHandler):
        """18.3 __exit__() returns None (does not suppress exceptions)."""
        w = DAQHandler()
        result = w.__exit__(ValueError, ValueError("test"), None)
        assert result is None or result is False

    def test_context_manager_cleanup(self, DAQHandler):
        """18.4 Context manager calls disconnect on exit."""
        w = DAQHandler()
        w.disconnect = MagicMock()
        with w:
            pass
        w.disconnect.assert_called_once()

    def test_context_manager_cleanup_on_exception(self, DAQHandler):
        """18.4 Context manager calls disconnect even on exception."""
        w = DAQHandler()
        w.disconnect = MagicMock()
        with pytest.raises(RuntimeError):
            with w:
                raise RuntimeError("test")
        w.disconnect.assert_called_once()

    def test_del_best_effort(self, DAQHandler):
        """18.5 __del__() suppresses exceptions."""
        w = DAQHandler()
        w.disconnect = MagicMock(side_effect=RuntimeError("fail"))
        # __del__ should not raise
        w.__del__()


# ===================================================================
# 19. RLock Thread Safety
# ===================================================================

class TestRLockThreadSafety:
    """Task group 19: RLock thread safety."""

    def test_rlock_created_in_init(self, DAQHandler):
        """19.1 RLock is created in __init__."""
        w = DAQHandler()
        assert hasattr(w, "_lock")
        assert isinstance(w._lock, type(threading.RLock()))

    def test_connect_acquires_lock(self, DAQHandler, wrapper_module):
        """19.4 connect() acquires lock."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        original_lock = w._lock
        w._lock = MagicMock(wraps=original_lock)

        with patch.object(wrapper_module, "get_task_by_name", side_effect=KeyError("No")):
            w.connect()

        w._lock.__enter__.assert_called()

    def test_acquire_acquires_lock(self, DAQHandler, wrapper_module):
        """19.2 acquire() acquires the RLock during execution."""
        # Use the same _setup_acquire pattern from TestAcquireLoop to avoid hanging.
        setup = TestAcquireLoop()
        w, _, _ = setup._setup_acquire(DAQHandler, wrapper_module)

        original_lock = w._lock
        w._lock = MagicMock(wraps=original_lock)

        w.acquire()

        w._lock.__enter__.assert_called()

    def test_generate_acquires_lock(self, DAQHandler, wrapper_module):
        """19.3 generate() acquires the RLock during execution."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True

        mock_task = MagicMock()
        mock_task.is_task_done.return_value = True
        mock_task.number_of_channels = 1
        w._task_out = mock_task
        w._task_out_is_str = True
        w._n_channels_out = 1
        w._generation_running = False

        original_lock = w._lock
        w._lock = MagicMock(wraps=original_lock)

        w.generate(np.array([1.0, 2.0, 3.0]))

        w._lock.__enter__.assert_called()

    def test_rlock_reentrancy(self, DAQHandler):
        """19.5 RLock allows reentrant acquisition (no deadlock)."""
        w = DAQHandler()
        # Simulate nested lock usage
        with w._lock:
            with w._lock:
                pass  # Should not deadlock


# ===================================================================
# 20. Configurable Timing Parameters
# ===================================================================

class TestTimingParameters:
    """Task group 20: Configurable timing parameters."""

    def test_default_timing(self, DAQHandler):
        """20.1 Default timing values."""
        w = DAQHandler()
        assert w.acquisition_sleep == 0.01
        assert w.post_trigger_delay == 0.05

    def test_timing_via_configure(self, DAQHandler):
        """20.2 Custom timing via configure()."""
        w = DAQHandler()
        w.configure(task_in="TestTask", acquisition_sleep=0.002, post_trigger_delay=0.2)
        assert w.acquisition_sleep == 0.002
        assert w.post_trigger_delay == 0.2

    def test_timing_via_constructor(self, DAQHandler):
        """20.3 Custom timing via constructor kwargs."""
        w = DAQHandler(acquisition_sleep=0.003, post_trigger_delay=0.15)
        assert w.acquisition_sleep == 0.003
        assert w.post_trigger_delay == 0.15


# ===================================================================
# 21. configure() with Digital Task Parameters (FR-6.10a)
# ===================================================================

class TestConfigureDigital:
    """Task group 21: configure() with digital task parameters."""

    def test_configure_digital_input_object(self, DAQHandler, wrapper_module):
        """1.1 configure(task_digital_in=DITask(...)) stores the object."""
        mock_di = _make_mock_digital_input()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di)

        assert w._task_digital_in_obj is mock_di
        assert w._task_digital_in_is_obj is True
        assert w._task_digital_in_is_str is False
        assert w._configured is True

    def test_configure_digital_output_object(self, DAQHandler, wrapper_module):
        """1.2 configure(task_digital_out=DOTask(...)) stores the object."""
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_out=mock_do)

        assert w._task_digital_out_obj is mock_do
        assert w._task_digital_out_is_obj is True
        assert w._task_digital_out_is_str is False
        assert w._configured is True

    def test_configure_digital_input_string(self, DAQHandler):
        """1.3 configure(task_digital_in=string) stores the name."""
        w = DAQHandler()
        w.configure(task_digital_in="MyDITask")

        assert w._task_digital_in_name == "MyDITask"
        assert w._task_digital_in_is_str is True
        assert w._task_digital_in_is_obj is False
        assert w._configured is True

    def test_configure_digital_output_string(self, DAQHandler):
        """1.4 configure(task_digital_out=string) stores the name."""
        w = DAQHandler()
        w.configure(task_digital_out="MyDOTask")

        assert w._task_digital_out_name == "MyDOTask"
        assert w._task_digital_out_is_str is True
        assert w._task_digital_out_is_obj is False
        assert w._configured is True

    def test_configure_mixed_analog_and_digital(self, DAQHandler, wrapper_module):
        """1.5 configure() with analog + both digital stores all independently."""
        mock_di = _make_mock_digital_input()
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask), \
             patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(
                task_in="AnalogIn",
                task_out="AnalogOut",
                task_digital_in=mock_di,
                task_digital_out=mock_do,
            )

        assert w._task_in_name == "AnalogIn"
        assert w._task_in_is_str is True
        assert w._task_out_name == "AnalogOut"
        assert w._task_out_is_str is True
        assert w._task_digital_in_obj is mock_di
        assert w._task_digital_in_is_obj is True
        assert w._task_digital_out_obj is mock_do
        assert w._task_digital_out_is_obj is True

    def test_configure_digital_only_no_analog(self, DAQHandler, wrapper_module):
        """1.6 configure() with only digital task — succeeds, analog methods raise."""
        mock_di = _make_mock_digital_input()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di)

        assert w._configured is True
        # Analog-specific methods should raise
        with pytest.raises(ValueError, match="input task"):
            w.set_trigger(n_samples=100, trigger_channel=0, trigger_level=0.5)
        with pytest.raises(ValueError, match="input task"):
            w.acquire()
        with pytest.raises(ValueError, match="output task"):
            w.generate(np.array([1.0]))

    def test_configure_replaces_previous_digital(self, DAQHandler, wrapper_module):
        """1.7 Calling configure() with different digital tasks replaces previous."""
        mock_di1 = _make_mock_digital_input(task_name="di_first")
        mock_di2 = _make_mock_digital_input(task_name="di_second")
        mock_do1 = _make_mock_digital_output(task_name="do_first")
        mock_do2 = _make_mock_digital_output(task_name="do_second")

        with patch.object(wrapper_module, "DITask", new=_FakeDITask), \
             patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di1, task_digital_out=mock_do1)
            assert w._task_digital_in_obj is mock_di1
            assert w._task_digital_out_obj is mock_do1
            w.configure(task_digital_in=mock_di2, task_digital_out=mock_do2)
            assert w._task_digital_in_obj is mock_di2
            assert w._task_digital_out_obj is mock_do2

    def test_configure_digital_invalid_type_raises(self, DAQHandler):
        """1.x configure() with invalid type for digital param raises TypeError."""
        w = DAQHandler()
        with pytest.raises(TypeError, match="task_digital_in"):
            w.configure(task_digital_in=42)
        with pytest.raises(TypeError, match="task_digital_out"):
            w.configure(task_digital_out=42)

    def test_configure_no_tasks_raises(self, DAQHandler):
        """1.x configure() with no tasks at all raises ValueError."""
        w = DAQHandler()
        with pytest.raises(ValueError, match="[Aa]t least one"):
            w.configure()


# ===================================================================
# 22. read_digital() Method (FR-6.10b)
# ===================================================================

class TestReadDigital:
    """Task group 22: read_digital() method."""

    def test_read_digital_delegates_to_digital_input(self, DAQHandler, wrapper_module):
        """3.1 read_digital() delegates to DITask.read() and returns result."""
        mock_di = _make_mock_digital_input()
        expected = np.array([True, False, True])
        mock_di.read.return_value = expected

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di)

        # Simulate connected state
        w._task_digital_in = mock_di

        result = w.read_digital()
        mock_di.read.assert_called_once()
        np.testing.assert_array_equal(result, expected)

    def test_read_digital_no_task_raises(self, DAQHandler):
        """3.2 read_digital() with no digital input task raises RuntimeError."""
        w = DAQHandler()
        with pytest.raises(RuntimeError, match="digital input task"):
            w.read_digital()

    def test_read_digital_not_connected_raises(self, DAQHandler, wrapper_module):
        """3.3 read_digital() when configured but not connected raises error."""
        mock_di = _make_mock_digital_input()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di)

        # _task_digital_in is None (not connected)
        with pytest.raises(RuntimeError, match="[Cc]onnect"):
            w.read_digital()

    def test_read_digital_acquires_lock(self, DAQHandler, wrapper_module):
        """3.4 read_digital() acquires the RLock during execution."""
        mock_di = _make_mock_digital_input()
        mock_di.read.return_value = np.array([True])

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di)

        w._task_digital_in = mock_di
        original_lock = w._lock
        w._lock = MagicMock(wraps=original_lock)

        w.read_digital()
        w._lock.__enter__.assert_called()


# ===================================================================
# 23. write_digital() Method (FR-6.10c)
# ===================================================================

class TestWriteDigital:
    """Task group 23: write_digital() method."""

    def test_write_digital_delegates_to_digital_output(self, DAQHandler, wrapper_module):
        """5.1 write_digital(data) delegates to DOTask.write(data)."""
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_out=mock_do)

        w._task_digital_out = mock_do

        w.write_digital(True)
        mock_do.write.assert_called_once_with(True)

    def test_write_digital_no_task_raises(self, DAQHandler):
        """5.2 write_digital() with no digital output task raises RuntimeError."""
        w = DAQHandler()
        with pytest.raises(RuntimeError, match="digital output task"):
            w.write_digital(True)

    def test_write_digital_not_connected_raises(self, DAQHandler, wrapper_module):
        """5.3 write_digital() when configured but not connected raises error."""
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_out=mock_do)

        # _task_digital_out is None (not connected)
        with pytest.raises(RuntimeError, match="[Cc]onnect"):
            w.write_digital(True)

    def test_write_digital_acquires_lock(self, DAQHandler, wrapper_module):
        """5.4 write_digital() acquires the RLock during execution."""
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_out=mock_do)

        w._task_digital_out = mock_do
        original_lock = w._lock
        w._lock = MagicMock(wraps=original_lock)

        w.write_digital(True)
        w._lock.__enter__.assert_called()

    def test_write_digital_single_bool(self, DAQHandler, wrapper_module):
        """5.5 write_digital(True) with single boolean value."""
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_out=mock_do)

        w._task_digital_out = mock_do

        w.write_digital(True)
        mock_do.write.assert_called_once_with(True)

    def test_write_digital_list(self, DAQHandler, wrapper_module):
        """5.6 write_digital([True, False, True]) with list for multi-line."""
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_out=mock_do)

        w._task_digital_out = mock_do

        data = [True, False, True]
        w.write_digital(data)
        mock_do.write.assert_called_once_with(data)


# ===================================================================
# 24. Digital Tasks in connect()/disconnect() Lifecycle (FR-6.10d)
# ===================================================================

class TestDigitalLifecycle:
    """Task group 24: Digital tasks in connect()/disconnect() lifecycle."""

    def test_connect_configures_digital_input(self, DAQHandler, wrapper_module):
        """7.1 connect() calls configure() on configured DITask task."""
        mock_di = _make_mock_digital_input()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di)

        with patch.object(wrapper_module, "get_connected_devices", return_value=set()):
            w.connect()

        mock_di.configure.assert_called_once()
        assert w._task_digital_in is mock_di

    def test_connect_configures_digital_output(self, DAQHandler, wrapper_module):
        """7.2 connect() calls configure() on configured DOTask task."""
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_out=mock_do)

        with patch.object(wrapper_module, "get_connected_devices", return_value=set()):
            w.connect()

        mock_do.configure.assert_called_once()
        assert w._task_digital_out is mock_do

    def test_connect_starts_both_analog_and_digital(self, DAQHandler, wrapper_module):
        """7.3 connect() with both analog and digital tasks starts all."""
        mock_di = _make_mock_digital_input()
        mock_do = _make_mock_digital_output()
        mock_loaded_task = _make_mock_nidaqmx_task()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask), \
             patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(
                task_in="AnalogIn",
                task_digital_in=mock_di,
                task_digital_out=mock_do,
            )

        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded_task), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            result = w.connect()

        assert result is True
        mock_di.configure.assert_called_once()
        mock_do.configure.assert_called_once()

    def test_disconnect_clears_digital_input(self, DAQHandler, wrapper_module):
        """7.4 disconnect() calls clear_task() on active DITask task."""
        mock_di = _make_mock_digital_input()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di)

        # Simulate connected state
        w._task_digital_in = mock_di

        w.disconnect()
        mock_di.clear_task.assert_called_once()
        assert w._task_digital_in is None

    def test_disconnect_clears_digital_output(self, DAQHandler, wrapper_module):
        """7.5 disconnect() calls clear_task() on active DOTask task."""
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_out=mock_do)

        w._task_digital_out = mock_do

        w.disconnect()
        mock_do.clear_task.assert_called_once()
        assert w._task_digital_out is None

    def test_disconnect_idempotent_with_digital(self, DAQHandler, wrapper_module):
        """7.6 disconnect() called multiple times does not raise."""
        mock_di = _make_mock_digital_input()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di)

        w._task_digital_in = mock_di

        result1 = w.disconnect()
        result2 = w.disconnect()
        assert result1 is True
        assert result2 is True

    def test_connect_with_string_digital_task(self, DAQHandler, wrapper_module):
        """7.7 connect() with string digital task name loads from NI MAX."""
        w = DAQHandler()
        w.configure(task_digital_in="MyDigitalTask")

        mock_loaded = _make_mock_nidaqmx_task()

        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            w.connect()

        assert w._task_digital_in is mock_loaded

    def test_connect_digital_configure_failure_does_not_prevent_analog(
        self, DAQHandler, wrapper_module
    ):
        """7.8 Digital configure() failure doesn't prevent analog connect()."""
        mock_di = _make_mock_digital_input()
        mock_di.configure.side_effect = RuntimeError("digital configure failed")
        mock_loaded_task = _make_mock_nidaqmx_task()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_in="AnalogIn", task_digital_in=mock_di)

        with patch.object(wrapper_module, "get_task_by_name", return_value=mock_loaded_task), \
             patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod1"}):
            result = w.connect()

        # Analog succeeds despite digital failure
        assert result is True
        assert w._task_in is not None
        # Digital was NOT set (start failed)
        assert w._task_digital_in is None

    def test_disconnect_digital_failure_does_not_prevent_analog_cleanup(
        self, DAQHandler, wrapper_module
    ):
        """8.5 Digital task cleanup failure doesn't prevent analog cleanup."""
        mock_di = _make_mock_digital_input()
        mock_di.clear_task.side_effect = RuntimeError("digital cleanup failed")

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_in="AnalogIn", task_digital_in=mock_di)

        # Simulate connected state
        mock_analog_task = _make_mock_nidaqmx_task()
        w._task_in = mock_analog_task
        w._task_digital_in = mock_di
        w._connected = True

        # disconnect should not raise despite digital failure
        result = w.disconnect()
        assert result is True
        mock_analog_task.close.assert_called()


# ===================================================================
# 25. Digital Tasks in Context Manager (FR-6.10e)
# ===================================================================

class TestDigitalContextManager:
    """Task group 25: Digital tasks in context manager."""

    def test_context_manager_cleans_up_digital_on_normal_exit(
        self, DAQHandler, wrapper_module
    ):
        """9.1 Context manager normal exit cleans up digital tasks."""
        mock_di = _make_mock_digital_input()
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask), \
             patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di, task_digital_out=mock_do)

        w._task_digital_in = mock_di
        w._task_digital_out = mock_do

        with w:
            pass  # normal exit

        mock_di.clear_task.assert_called_once()
        mock_do.clear_task.assert_called_once()

    def test_context_manager_cleans_up_digital_on_exception(
        self, DAQHandler, wrapper_module
    ):
        """9.2 Context manager exception exit still cleans up digital tasks."""
        mock_di = _make_mock_digital_input()

        with patch.object(wrapper_module, "DITask", new=_FakeDITask):
            w = DAQHandler()
            w.configure(task_digital_in=mock_di)

        w._task_digital_in = mock_di

        with pytest.raises(RuntimeError):
            with w:
                raise RuntimeError("test exception")

        mock_di.clear_task.assert_called_once()

    def test_context_manager_digital_only(self, DAQHandler, wrapper_module):
        """9.3 Context manager with only digital tasks (no analog) cleans up."""
        mock_do = _make_mock_digital_output()

        with patch.object(wrapper_module, "DOTask", new=_FakeDOTask):
            w = DAQHandler()
            w.configure(task_digital_out=mock_do)

        w._task_digital_out = mock_do

        with w:
            pass

        mock_do.clear_task.assert_called_once()
        assert w._task_digital_out is None


# ===================================================================
# 26. Direct-Delegation Guard Tests
# ===================================================================

class TestDirectDelegationGuards:
    """Guard tests: removed attributes and methods must not exist on DAQHandler.

    These tests ensure that the pre-refactor recreation helpers are gone
    and that no dead code re-appears during future edits.
    """

    def test_no_recreate_ni_task_in_method(self, DAQHandler):
        """_recreate_ni_task_in() was removed in the direct-delegation refactor."""
        w = DAQHandler()
        assert not hasattr(w, "_recreate_ni_task_in")

    def test_no_recreate_ni_task_out_method(self, DAQHandler):
        """_recreate_ni_task_out() was removed in the direct-delegation refactor."""
        w = DAQHandler()
        assert not hasattr(w, "_recreate_ni_task_out")


# ===================================================================
# 27. Output Metadata via channel_list / number_of_ch
# ===================================================================

class TestOutputMetadataDelegation:
    """_extract_output_metadata_from_ni_task_out() uses channel_list and
    number_of_ch properties — NOT the removed .channels dict.
    """

    def test_extract_output_metadata_uses_channel_list(
        self, DAQHandler, wrapper_module
    ):
        """connect() populates _channel_names_out and _n_channels_out via
        channel_list/number_of_ch on the stored AOTask reference.
        """
        mock_task_out = _make_mock_ao_task()
        mock_task_out.channel_list = ["ao0", "ao1"]
        mock_task_out.number_of_ch = 2
        mock_task_out.sample_rate = 10000.0

        # Poison .channels so any access raises — ensures implementation
        # uses channel_list/number_of_ch, not the removed .channels dict
        type(mock_task_out).channels = PropertyMock(
            side_effect=AttributeError("channels dict was removed")
        )

        with patch.object(wrapper_module, "AOTask", new=type(mock_task_out)):
            w = DAQHandler()
            w.configure(task_out=mock_task_out)

        with patch.object(wrapper_module, "get_connected_devices", return_value={"cDAQ1Mod2"}):
            w.connect()

        assert w._channel_names_out == ["ao0", "ao1"]
        assert w._n_channels_out == 2
        assert w._sample_rate_out == 10000.0


# ===================================================================
# Task group: Raw nidaqmx.Task Support
# ===================================================================

class TestConfigureRawTask:
    """Tests for configure() with raw nidaqmx.task.Task objects."""

    def test_configure_raw_ai_task(self, DAQHandler, wrapper_module):
        """configure(task_in=raw_nidaqmx_task) wraps via AITask.from_task()."""
        # Create a raw task mock with a distinct type
        class FakeNidaqmxTask:
            def __init__(self):
                self.name = "RawAITask"
                self.timing = MagicMock()
                self.timing.samp_clk_rate = 25600.0
                self.ai_channels = [MagicMock(), MagicMock()]
                self.channel_names = ["ai0", "ai1"]
                self.number_of_channels = 2

        raw_task = FakeNidaqmxTask()

        wrapped_ai_task = _make_mock_ai_task(task_name="RawAITask", sample_rate=25600.0)
        wrapped_ai_task._owns_task = False

        # Mock AITask class to capture from_task call
        real_ai_task = wrapper_module.AITask
        mock_from_task = MagicMock(return_value=wrapped_ai_task)

        with patch.object(wrapper_module.AITask, "from_task", mock_from_task), \
             patch.object(wrapper_module, "_NIDAQMX_AVAILABLE", True), \
             patch.object(wrapper_module, "nidaqmx") as mock_nidaqmx:
            mock_nidaqmx.task.Task = FakeNidaqmxTask
            w = DAQHandler()
            w.configure(task_in=raw_task)

        # Verify from_task was called with raw task
        mock_from_task.assert_called_once_with(raw_task)

        # Verify the wrapped task is stored
        assert w._task_in_obj is wrapped_ai_task
        assert w._task_in_is_obj is True
        assert w._task_in_is_str is False
        assert w._configured is True

    def test_configure_raw_ao_task(self, DAQHandler, wrapper_module):
        """configure(task_out=raw_nidaqmx_task) wraps via AOTask.from_task()."""
        class FakeNidaqmxTask:
            def __init__(self):
                self.name = "RawAOTask"
                self.timing = MagicMock()
                self.timing.samp_clk_rate = 10000.0
                self.ao_channels = [MagicMock()]
                self.channel_names = ["ao0"]
                self.number_of_channels = 1

        raw_task = FakeNidaqmxTask()

        wrapped_ao_task = _make_mock_ao_task(task_name="RawAOTask", sample_rate=10000.0)
        wrapped_ao_task._owns_task = False

        mock_from_task = MagicMock(return_value=wrapped_ao_task)

        with patch.object(wrapper_module.AOTask, "from_task", mock_from_task), \
             patch.object(wrapper_module, "_NIDAQMX_AVAILABLE", True), \
             patch.object(wrapper_module, "nidaqmx") as mock_nidaqmx:
            mock_nidaqmx.task.Task = FakeNidaqmxTask
            w = DAQHandler()
            w.configure(task_out=raw_task)

        mock_from_task.assert_called_once_with(raw_task)
        assert w._task_out_obj is wrapped_ao_task
        assert w._task_out_is_obj is True
        assert w._task_out_is_str is False

    def test_configure_raw_di_task(self, DAQHandler, wrapper_module):
        """configure(task_digital_in=raw_task) wraps via DITask.from_task()."""
        class FakeNidaqmxTask:
            def __init__(self):
                self.name = "RawDITask"
                self.di_channels = [MagicMock()]

        raw_task = FakeNidaqmxTask()

        mock_di_task = _make_mock_digital_input(task_name="RawDITask")
        mock_di_task._owns_task = False

        mock_from_task = MagicMock(return_value=mock_di_task)

        with patch.object(wrapper_module.DITask, "from_task", mock_from_task), \
             patch.object(wrapper_module, "_NIDAQMX_AVAILABLE", True), \
             patch.object(wrapper_module, "nidaqmx") as mock_nidaqmx:
            mock_nidaqmx.task.Task = FakeNidaqmxTask
            w = DAQHandler()
            w.configure(task_digital_in=raw_task)

        mock_from_task.assert_called_once_with(raw_task)
        assert w._task_digital_in_obj is mock_di_task
        assert w._task_digital_in_is_obj is True
        assert w._task_digital_in_is_str is False

    def test_configure_raw_do_task(self, DAQHandler, wrapper_module):
        """configure(task_digital_out=raw_task) wraps via DOTask.from_task()."""
        class FakeNidaqmxTask:
            def __init__(self):
                self.name = "RawDOTask"
                self.do_channels = [MagicMock()]

        raw_task = FakeNidaqmxTask()

        mock_do_task = _make_mock_digital_output(task_name="RawDOTask")
        mock_do_task._owns_task = False

        mock_from_task = MagicMock(return_value=mock_do_task)

        with patch.object(wrapper_module.DOTask, "from_task", mock_from_task), \
             patch.object(wrapper_module, "_NIDAQMX_AVAILABLE", True), \
             patch.object(wrapper_module, "nidaqmx") as mock_nidaqmx:
            mock_nidaqmx.task.Task = FakeNidaqmxTask
            w = DAQHandler()
            w.configure(task_digital_out=raw_task)

        mock_from_task.assert_called_once_with(raw_task)
        assert w._task_digital_out_obj is mock_do_task
        assert w._task_digital_out_is_obj is True
        assert w._task_digital_out_is_str is False

    def test_configure_mixed_raw_and_wrapper_tasks(self, DAQHandler, wrapper_module):
        """configure() with both raw task and wrapper task in same call."""
        class FakeNidaqmxTask:
            def __init__(self):
                self.name = "RawAI"
                self.timing = MagicMock()
                self.timing.samp_clk_rate = 25600.0
                self.ai_channels = [MagicMock()]
                self.channel_names = ["ai0"]
                self.number_of_channels = 1

        raw_ai_task = FakeNidaqmxTask()

        wrapped_ai_task = _make_mock_ai_task(task_name="RawAI", sample_rate=25600.0)
        wrapped_ai_task._owns_task = False

        mock_ao_task = _make_mock_ao_task()

        mock_from_task = MagicMock(return_value=wrapped_ai_task)

        with patch.object(wrapper_module.AITask, "from_task", mock_from_task), \
             patch.object(wrapper_module, "AOTask", new=type(mock_ao_task)), \
             patch.object(wrapper_module, "_NIDAQMX_AVAILABLE", True), \
             patch.object(wrapper_module, "nidaqmx") as mock_nidaqmx:
            mock_nidaqmx.task.Task = FakeNidaqmxTask
            w = DAQHandler()
            w.configure(task_in=raw_ai_task, task_out=mock_ao_task)

        # AI task should be wrapped from raw
        mock_from_task.assert_called_once_with(raw_ai_task)
        assert w._task_in_obj is wrapped_ai_task
        assert w._task_in_is_obj is True

        # AO task should be stored directly (already wrapped)
        assert w._task_out_obj is mock_ao_task
        assert w._task_out_is_obj is True

    def test_configure_raw_task_ownership_not_closed_on_disconnect(
        self, DAQHandler, wrapper_module
    ):
        """disconnect() does not close raw tasks (warns instead)."""
        class FakeNidaqmxTask:
            def __init__(self):
                self.name = "RawTask"
                self.timing = MagicMock()
                self.timing.samp_clk_rate = 25600.0
                self.ai_channels = [MagicMock()]
                self.channel_names = ["ai0"]
                self.number_of_channels = 1

        raw_task = FakeNidaqmxTask()

        wrapped_ai_task = _make_mock_ai_task(task_name="RawTask", sample_rate=25600.0)
        wrapped_ai_task._owns_task = False
        wrapped_ai_task.clear_task = MagicMock()

        mock_from_task = MagicMock(return_value=wrapped_ai_task)

        with patch.object(wrapper_module.AITask, "from_task", mock_from_task), \
             patch.object(wrapper_module, "_NIDAQMX_AVAILABLE", True), \
             patch.object(wrapper_module, "nidaqmx") as mock_nidaqmx:
            mock_nidaqmx.task.Task = FakeNidaqmxTask
            w = DAQHandler()
            w.configure(task_in=raw_task)
            # Don't actually connect to avoid complex mock setup
            w._task_in_obj_active = wrapped_ai_task

        # Disconnect should call clear_task on the wrapper
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            w.disconnect()

        # clear_task was called on the wrapper
        wrapped_ai_task.clear_task.assert_called_once()

        # If clear_task issues a warning (because _owns_task=False), it should be captured
        # But we don't test the warning itself here — that's the wrapper class's responsibility

    def test_configure_raw_task_string_mixed(self, DAQHandler, wrapper_module):
        """configure() with raw task and string task in same call."""
        class FakeNidaqmxTask:
            def __init__(self):
                self.name = "RawAI"
                self.timing = MagicMock()
                self.timing.samp_clk_rate = 25600.0
                self.ai_channels = [MagicMock()]
                self.channel_names = ["ai0"]
                self.number_of_channels = 1

        raw_ai_task = FakeNidaqmxTask()

        wrapped_ai_task = _make_mock_ai_task(task_name="RawAI", sample_rate=25600.0)
        wrapped_ai_task._owns_task = False

        mock_from_task = MagicMock(return_value=wrapped_ai_task)

        with patch.object(wrapper_module.AITask, "from_task", mock_from_task), \
             patch.object(wrapper_module, "_NIDAQMX_AVAILABLE", True), \
             patch.object(wrapper_module, "nidaqmx") as mock_nidaqmx:
            mock_nidaqmx.task.Task = FakeNidaqmxTask
            w = DAQHandler()
            w.configure(task_in=raw_ai_task, task_out="OutputTaskString")

        # AI task wrapped from raw
        assert w._task_in_obj is wrapped_ai_task
        assert w._task_in_is_obj is True

        # AO task stored as string
        assert w._task_out_name == "OutputTaskString"
        assert w._task_out_is_str is True


# ===================================================================
# Change multi-handler-advanced-acquisition
# DAQHandler: is_running() and stop_acquisition()
# ===================================================================

class TestIsRunning:
    """Group 4 — DAQHandler.is_running()."""

    def test_not_connected_returns_false(self, DAQHandler):
        """is_running() returns False when not connected."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        assert w.is_running() is False

    def test_no_tasks_returns_false(self, DAQHandler):
        """is_running() returns False when connected but no active tasks."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        assert w.is_running() is False

    def test_input_task_running_returns_true(self, DAQHandler):
        """Running analog input task → True."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        mock_task = MagicMock()
        mock_task.is_task_done.return_value = False
        w._task_in = mock_task
        assert w.is_running() is True

    def test_output_task_running_returns_true(self, DAQHandler):
        """Running analog output task → True."""
        w = DAQHandler()
        w.configure(task_out="OutputTask")
        w._connected = True
        mock_task = MagicMock()
        mock_task.is_task_done.return_value = False
        w._task_out = mock_task
        assert w.is_running() is True

    def test_all_tasks_done_returns_false(self, DAQHandler):
        """All tasks report done → False."""
        w = DAQHandler()
        w.configure(task_in="In", task_out="Out")
        w._connected = True
        done_in = MagicMock()
        done_in.is_task_done.return_value = True
        done_out = MagicMock()
        done_out.is_task_done.return_value = True
        w._task_in = done_in
        w._task_out = done_out
        assert w.is_running() is False

    def test_digital_task_running_returns_true(self, DAQHandler):
        """Running digital input task (DITask wrapper) → True."""
        w = DAQHandler()
        w.configure(task_digital_in="DigInTask")
        w._connected = True
        # Active digital task is a DITask wrapper exposing .task
        fake = _make_mock_digital_input()
        fake.task = MagicMock()
        fake.task.is_task_done.return_value = False
        w._task_digital_in = fake
        assert w.is_running() is True

    def test_daqerror_warns_and_returns_false(self, DAQHandler, wrapper_module):
        """DaqError during query → warning, task treated as not running."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        mock_task = MagicMock()
        mock_task.is_task_done.side_effect = wrapper_module.DaqError("task gone")
        w._task_in = mock_task

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = w.is_running()

        assert result is False
        assert any("task gone" in str(x.message) for x in caught)

    def test_daqerror_on_one_task_other_running_returns_true(
        self, DAQHandler, wrapper_module
    ):
        """DaqError on one task does not hide another running task."""
        w = DAQHandler()
        w.configure(task_in="In", task_out="Out")
        w._connected = True
        broken = MagicMock()
        broken.is_task_done.side_effect = wrapper_module.DaqError("invalid")
        running = MagicMock()
        running.is_task_done.return_value = False
        w._task_in = broken
        w._task_out = running

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert w.is_running() is True


class TestStopAcquisition:
    """Group 4 — DAQHandler.stop_acquisition() cooperative abort."""

    def _setup_never_finishing(self, DAQHandler):
        """Set up a handler whose trigger never fires."""
        w = DAQHandler()
        w.configure(task_in="InputTask")
        w._connected = True
        w._connect_called = True
        w._n_channels_in = 2
        w._channel_names_in = ["ch0", "ch1"]
        w._sample_rate_in = 25600.0
        w.acquisition_sleep = 0.001
        w.post_trigger_delay = 0

        read_started = threading.Event()
        mock_task = MagicMock()

        def read_side_effect(*args, **kwargs):
            read_started.set()
            return [[0.0] * 10, [0.0] * 10]

        mock_task.read.side_effect = read_side_effect
        w._task_in = mock_task

        mock_trigger = MagicMock()
        mock_trigger.finished = False  # never fires
        mock_trigger.rows = 100
        ring = np.arange(200, dtype=float).reshape(100, 2)
        mock_trigger.get_data.return_value = ring
        w.trigger = mock_trigger
        w._trigger_is_set = True

        return w, mock_task, mock_trigger, ring, read_started

    def test_noop_when_idle(self, DAQHandler):
        """stop_acquisition() with no acquisition in flight is a no-op."""
        w = DAQHandler()
        w.stop_acquisition()  # must not raise
        assert w._acquire_running is False

    def test_stop_acquisition_does_not_take_lock(self, DAQHandler):
        """stop_acquisition() must not block on the RLock (deadlock avoidance)."""
        w = DAQHandler()
        locked = threading.Event()
        release = threading.Event()

        def hold_lock():
            with w._lock:
                locked.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        assert locked.wait(timeout=5)

        w._acquire_running = True
        stopper = threading.Thread(target=w.stop_acquisition)
        stopper.start()
        stopper.join(timeout=1.0)
        still_blocked = stopper.is_alive()
        release.set()
        holder.join(timeout=5)
        stopper.join(timeout=5)

        assert not still_blocked, "stop_acquisition() blocked on the RLock"
        assert w._acquire_running is False

    def test_abort_nonblocking_acquire(self, DAQHandler):
        """Abort: loop exits, task stopped, Future resolves with ring buffer."""
        w, mock_task, _, ring, read_started = self._setup_never_finishing(DAQHandler)

        future = w.acquire(blocking=False)
        assert read_started.wait(timeout=5), "acquisition loop never started"
        w.stop_acquisition()
        result = future.result(timeout=5)

        mock_task.stop.assert_called_once()
        np.testing.assert_array_equal(result, ring)
        assert w._acquire_running is False

    def test_flag_reset_at_start_of_each_acquire(self, DAQHandler):
        """A stale False flag (from a prior abort) does not skip the next acquire."""
        w, mock_task, mock_trigger, _, _ = self._setup_never_finishing(DAQHandler)
        mock_trigger.add_data.side_effect = (
            lambda data: setattr(mock_trigger, "finished", True)
        )

        w._acquire_running = False  # stale flag from a previous abort
        result = w.acquire()

        assert mock_trigger.add_data.call_count >= 1, "loop never ran"
        assert isinstance(result, np.ndarray)
