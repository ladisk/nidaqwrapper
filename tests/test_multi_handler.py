"""Tests for MultiHandler — multi-task synchronized acquisition.

Covers all 16 task groups from the OpenSpec change:
1. Constructor defaults
2-3. configure() with task types + _resolve_tasks()
4. _validate_types()
5. _validate_validity() with integer error codes (bug fix)
6. _validate_sample_rates()
7. _validate_timing()
8. _validate_triggers() with trigger_type default (bug fix)
9. _validate_acquisition_mode()
10. acquire_with_hardware_trigger()
11. acquire_with_software_trigger() with task.start() fix (bug fix)
12. acquire() dispatch
13. connect() / disconnect()
14. set_trigger()
15. ping() / device management
"""

from __future__ import annotations

import threading
import warnings
from unittest.mock import MagicMock, patch, PropertyMock, call

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: build a mock nidaqmx.task.Task with configurable attributes
# ---------------------------------------------------------------------------

def _make_nidaqmx_task(
    name="TestTask",
    channel_names=None,
    samp_clk_rate=25600.0,
    samp_clk_src="OnboardClock",
    samp_quant_samp_per_chan=25600,
    trig_type_name="NONE",
    dig_edge_src="",
    anlg_edge_src="",
    samp_quant_samp_mode_name="CONTINUOUS",
    devices=None,
):
    """Create a mock nidaqmx.task.Task with timing/trigger attributes."""
    task = MagicMock()
    task.name = name
    task.channel_names = channel_names or ["cDAQ1Mod1/ai0"]

    # Timing
    task.timing.samp_clk_rate = samp_clk_rate
    task.timing.samp_clk_src = samp_clk_src
    task.timing.samp_quant_samp_per_chan = samp_quant_samp_per_chan
    task.timing.samp_quant_samp_mode.name = samp_quant_samp_mode_name

    # Triggers
    task.triggers.start_trigger.trig_type.name = trig_type_name
    task.triggers.start_trigger.dig_edge_src = dig_edge_src
    task.triggers.start_trigger.anlg_edge_src = anlg_edge_src

    # Devices
    if devices is None:
        dev = MagicMock()
        dev.name = "cDAQ1Mod1"
        task.devices = [dev]
    else:
        task.devices = devices

    return task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nidaqmx_mock():
    """Set up the module-level nidaqmx mock for import in multi_handler.py."""
    mock_nidaqmx = MagicMock()
    mock_nidaqmx.task.Task = MagicMock
    mock_nidaqmx.constants.READ_ALL_AVAILABLE = -1
    mock_nidaqmx.errors.DaqError = type("DaqError", (Exception,), {})
    return mock_nidaqmx


@pytest.fixture
def advanced_module(nidaqmx_mock):
    """Import the multi_handler module with mocked nidaqmx dependencies.

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
        "nidaqwrapper", "nidaqwrapper.multi_handler", "nidaqwrapper.handler",
        "nidaqwrapper.utils", "nidaqwrapper.ai_task",
        "nidaqwrapper.ao_task", "nidaqwrapper.digital",
    ]

    # Save current state of all modules that will be touched
    saved = {}
    for mod_name in _MOCK_TARGETS + _NIDAQWRAPPER_MODULES:
        saved[mod_name] = sys.modules.get(mod_name)

    # Install mocks — reuse the nidaqmx_mock fixture object so that
    # all submodule references (nidaqmx.constants, etc.) stay consistent
    sys.modules["nidaqmx"] = nidaqmx_mock
    sys.modules["nidaqmx.task"] = nidaqmx_mock.task
    sys.modules["nidaqmx.constants"] = nidaqmx_mock.constants
    sys.modules["nidaqmx.errors"] = nidaqmx_mock.errors
    sys.modules["nidaqmx.system"] = nidaqmx_mock.system
    sys.modules["pyTrigger"] = MagicMock()

    # Remove cached nidaqwrapper modules so they reimport with mocked nidaqmx
    for mod_name in _NIDAQWRAPPER_MODULES:
        sys.modules.pop(mod_name, None)

    from nidaqwrapper import multi_handler

    yield multi_handler

    # Restore ALL modules to their pre-fixture state
    for mod_name, original in saved.items():
        if original is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original


@pytest.fixture
def MultiHandler(advanced_module):
    """Return the MultiHandler class."""
    return advanced_module.MultiHandler


@pytest.fixture
def adv(MultiHandler):
    """Return a fresh MultiHandler instance."""
    return MultiHandler()


# ===================================================================
# Group 1: MultiHandler Constructor
# ===================================================================


class TestConstructor:
    """Task group 1 — constructor defaults and RLock."""

    def test_input_tasks_default_empty(self, adv):
        """1.1 input_tasks defaults to empty list."""
        assert adv.input_tasks == []

    def test_output_tasks_default_empty(self, adv):
        """1.1 output_tasks defaults to empty list."""
        assert adv.output_tasks == []

    def test_trigger_type_default_software(self, adv):
        """1.1/Bug fix #2: trigger_type defaults to 'software'."""
        assert adv.trigger_type == "software"

    def test_required_devices_default_empty_set(self, adv):
        """1.1 required_devices defaults to empty set."""
        assert adv.required_devices == set()

    def test_configured_default_false(self, adv):
        """1.1 _configured defaults to False."""
        assert adv._configured is False

    def test_connected_default_false(self, adv):
        """1.1 _connected defaults to False."""
        assert adv._connected is False

    def test_rlock_created(self, adv):
        """1.1 RLock is created as threading.RLock."""
        assert isinstance(adv._lock, type(threading.RLock()))

    def test_trigger_attribute_exists_at_init(self, adv):
        """Bug fix #2 regression: trigger_type is set in __init__, never raises AttributeError."""
        assert hasattr(adv, "trigger_type")


# ===================================================================
# Group 2-3: configure() + _resolve_tasks()
# ===================================================================


class TestConfigureDefaults:
    """Task group 2.1 — configure() with None defaults (bug fix #4)."""

    def test_configure_none_defaults_not_mutable(self, adv):
        """2.1/Bug fix #4: configure(None, None) creates fresh empty lists, not shared mutable defaults."""
        # Call configure twice — if mutable default, lists would be shared
        with patch.object(adv, "_validate_types", return_value=True):
            with patch.object(adv, "_resolve_tasks", return_value=[]):
                with patch.object(adv, "_validate_validity", return_value=True):
                    with patch.object(adv, "_validate_sample_rates", return_value=True):
                        with patch.object(adv, "_validate_timing", return_value=True):
                            with patch.object(adv, "_validate_triggers", return_value=True):
                                with patch.object(adv, "_validate_acquisition_mode", return_value=True):
                                    with patch.object(adv, "_define_required_devices"):
                                        adv.configure()
                                        first_input = adv.input_tasks
                                        adv.configure()
                                        second_input = adv.input_tasks
                                        # Must be different list objects
                                        assert first_input is not second_input

    def test_configure_default_input_tasks_empty(self, adv):
        """2.1 Default input_tasks is empty list."""
        with patch.object(adv, "_validate_types", return_value=True):
            with patch.object(adv, "_resolve_tasks", return_value=[]):
                with patch.object(adv, "_validate_validity", return_value=True):
                    with patch.object(adv, "_validate_sample_rates", return_value=True):
                        with patch.object(adv, "_validate_timing", return_value=True):
                            with patch.object(adv, "_validate_triggers", return_value=True):
                                with patch.object(adv, "_validate_acquisition_mode", return_value=True):
                                    with patch.object(adv, "_define_required_devices"):
                                        result = adv.configure()
                                        assert result is True


class TestConfigureWithNidaqmxTask:
    """Task group 2.2 — configure() with nidaqmx.task.Task objects."""

    def test_nidaqmx_task_stored(self, adv, nidaqmx_mock):
        """2.2 nidaqmx.task.Task objects are passed through resolution and stored."""
        mock_task = _make_nidaqmx_task("InputTask")
        # Make mock_task pass isinstance check
        nidaqmx_mock.task.Task = type(mock_task)

        with patch.object(adv, "_validate_types", return_value=True):
            with patch.object(adv, "_resolve_tasks", return_value=[mock_task]) as resolve_mock:
                with patch.object(adv, "_validate_validity", return_value=True):
                    with patch.object(adv, "_validate_sample_rates", return_value=True):
                        with patch.object(adv, "_validate_timing", return_value=True):
                            with patch.object(adv, "_validate_triggers", return_value=True):
                                with patch.object(adv, "_validate_acquisition_mode", return_value=True):
                                    with patch.object(adv, "_define_required_devices"):
                                        adv.configure(input_tasks=[mock_task])
                                        assert adv.input_tasks == [mock_task]


class TestConfigureWithAITask:
    """Task group 2.3 — configure() with AITask objects."""

    def test_nitask_resolved_via_configure(self, adv, advanced_module):
        """2.3 AITask objects are resolved: configure() called, underlying task extracted."""
        mock_ni_task = MagicMock(spec=advanced_module.AITask)
        underlying = _make_nidaqmx_task("ResolvedTask")
        mock_ni_task.task = underlying

        result = adv._resolve_tasks([mock_ni_task])
        mock_ni_task.configure.assert_called_once()
        assert result == [underlying]


class TestConfigureWithAOTask:
    """Task group 2.4 — configure() with AOTask objects."""

    def test_nitaskoutput_resolved_via_configure(self, adv, advanced_module):
        """2.4 AOTask objects are resolved: configure() called, underlying task extracted."""
        mock_ni_task_out = MagicMock(spec=advanced_module.AOTask)
        underlying = _make_nidaqmx_task("OutputResolvedTask")
        mock_ni_task_out.task = underlying

        result = adv._resolve_tasks([mock_ni_task_out])
        mock_ni_task_out.configure.assert_called_once()
        assert result == [underlying]


class TestConfigureWithString:
    """Task group 2.5 — configure() with string task names."""

    def test_string_resolved_via_get_task_by_name(self, adv, advanced_module):
        """2.5 String task names are resolved via get_task_by_name()."""
        loaded_task = _make_nidaqmx_task("LoadedTask")
        with patch.object(advanced_module, "get_task_by_name", return_value=loaded_task) as mock_get:
            result = adv._resolve_tasks(["MyTask"])
            mock_get.assert_called_once_with("MyTask")
            assert result == [loaded_task]


class TestConfigureMixed:
    """Task group 2.6 — mixed types in same list."""

    def test_mixed_types_resolved(self, adv, advanced_module):
        """2.6 Mixed nidaqmx.Task and string in same list resolved correctly."""
        direct_task = _make_nidaqmx_task("DirectTask")
        loaded_task = _make_nidaqmx_task("LoadedTask")

        # Make direct_task an instance of nidaqmx.task.Task mock
        with patch.object(advanced_module, "get_task_by_name", return_value=loaded_task):
            with patch.object(advanced_module, "nidaqmx") as mock_nidaqmx:
                mock_nidaqmx.task.Task = type(direct_task)
                # We need to test _resolve_tasks handles multiple types
                # Use AITask mock for one, string for another
                ni_task_mock = MagicMock(spec=advanced_module.AITask)
                ni_task_mock.task = direct_task
                result = adv._resolve_tasks([ni_task_mock, "MyTask"])
                assert result == [direct_task, loaded_task]


class TestConfigureValidationResult:
    """Task groups 2.7-2.9 — configure() validation flow."""

    def test_configure_returns_true_on_full_validation_success(self, adv):
        """2.7 configure() returns True when all validators pass."""
        task = _make_nidaqmx_task()
        with patch.object(adv, "_validate_types", return_value=True):
            with patch.object(adv, "_resolve_tasks", return_value=[task]):
                with patch.object(adv, "_validate_validity", return_value=True):
                    with patch.object(adv, "_validate_sample_rates", return_value=True):
                        with patch.object(adv, "_validate_timing", return_value=True):
                            with patch.object(adv, "_validate_triggers", return_value=True):
                                with patch.object(adv, "_validate_acquisition_mode", return_value=True):
                                    with patch.object(adv, "_define_required_devices"):
                                        result = adv.configure(input_tasks=[task])
                                        assert result is True

    def test_configure_returns_false_when_validation_fails(self, adv):
        """2.8 configure() returns False when any validation fails."""
        task = _make_nidaqmx_task()
        with patch.object(adv, "_validate_types", return_value=True):
            with patch.object(adv, "_resolve_tasks", return_value=[task]):
                with patch.object(adv, "_validate_validity", return_value=False):
                    with patch.object(adv, "_validate_sample_rates", return_value=True):
                        with patch.object(adv, "_validate_timing", return_value=True):
                            with patch.object(adv, "_validate_triggers", return_value=True):
                                with patch.object(adv, "_validate_acquisition_mode", return_value=True):
                                    with patch.object(adv, "_define_required_devices"):
                                        result = adv.configure(input_tasks=[task])
                                        assert result is False

    def test_configure_sets_configured_true_on_success(self, adv):
        """2.9 configure() sets _configured=True on success."""
        with patch.object(adv, "_validate_types", return_value=True):
            with patch.object(adv, "_resolve_tasks", return_value=[]):
                with patch.object(adv, "_validate_validity", return_value=True):
                    with patch.object(adv, "_validate_sample_rates", return_value=True):
                        with patch.object(adv, "_validate_timing", return_value=True):
                            with patch.object(adv, "_validate_triggers", return_value=True):
                                with patch.object(adv, "_validate_acquisition_mode", return_value=True):
                                    with patch.object(adv, "_define_required_devices"):
                                        adv.configure()
                                        assert adv._configured is True

    def test_configure_does_not_set_configured_on_failure(self, adv):
        """2.8 _configured stays False when configure fails."""
        with patch.object(adv, "_validate_types", return_value=False):
            adv.configure()
            assert adv._configured is False


# ===================================================================
# Group 3: _resolve_tasks()
# ===================================================================


class TestResolveTasks:
    """Task group 3 — _resolve_tasks() conversion logic."""

    def test_resolve_string_calls_get_task_by_name(self, adv, advanced_module):
        """3.1 String resolved via get_task_by_name()."""
        loaded = _make_nidaqmx_task("Loaded")
        with patch.object(advanced_module, "get_task_by_name", return_value=loaded):
            result = adv._resolve_tasks(["SomeName"])
            assert result == [loaded]

    def test_resolve_nitask_calls_configure(self, adv, advanced_module):
        """3.2 AITask always calls configure() to set up timing before extraction."""
        ni_task = MagicMock(spec=advanced_module.AITask)
        underlying = _make_nidaqmx_task()
        ni_task.task = underlying  # task exists from direct-delegation __init__

        result = adv._resolve_tasks([ni_task])
        ni_task.configure.assert_called_once()
        assert result == [underlying]

    def test_resolve_nitaskoutput_calls_configure(self, adv, advanced_module):
        """3.3 AOTask always calls configure() to set up timing before extraction."""
        ni_task_out = MagicMock(spec=advanced_module.AOTask)
        underlying = _make_nidaqmx_task()
        ni_task_out.task = underlying  # task exists from direct-delegation __init__

        result = adv._resolve_tasks([ni_task_out])
        ni_task_out.configure.assert_called_once()
        assert result == [underlying]

    def test_resolve_nidaqmx_task_passthrough(self, adv, advanced_module):
        """3.4 nidaqmx.task.Task objects pass through unchanged."""
        mock_task = _make_nidaqmx_task()
        with patch.object(advanced_module, "nidaqmx") as mock_nidaqmx:
            mock_nidaqmx.task.Task = type(mock_task)
            result = adv._resolve_tasks([mock_task])
            assert result == [mock_task]

    def test_resolve_invalid_type_raises_typeerror(self, adv):
        """3.5 Invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Task must be"):
            adv._resolve_tasks([42])

    def test_resolve_dict_raises_typeerror(self, adv):
        """3.5 Dict raises TypeError."""
        with pytest.raises(TypeError, match="Task must be"):
            adv._resolve_tasks([{"name": "bad"}])


# ===================================================================
# Group 4: _validate_types()
# ===================================================================


class TestValidateTypes:
    """Task group 4 — _validate_types() type checking."""

    def test_valid_nidaqmx_tasks(self, adv, advanced_module):
        """4.1 Valid list of nidaqmx.task.Task objects returns True."""
        task = _make_nidaqmx_task()
        with patch.object(advanced_module, "nidaqmx") as mock_nidaqmx:
            mock_nidaqmx.task.Task = type(task)
            assert adv._validate_types([task], []) is True

    def test_valid_string_tasks(self, adv):
        """4.2 Valid list of strings returns True."""
        assert adv._validate_types(["Task1", "Task2"], []) is True

    def test_valid_nitask_objects(self, adv, advanced_module):
        """4.3 Valid list of AITask objects returns True."""
        ni_task = MagicMock(spec=advanced_module.AITask)
        assert adv._validate_types([ni_task], []) is True

    def test_valid_nitaskoutput_objects(self, adv, advanced_module):
        """4.4 Valid list of AOTask objects returns True."""
        ni_task_out = MagicMock(spec=advanced_module.AOTask)
        assert adv._validate_types([], [ni_task_out]) is True

    def test_input_not_list_raises_typeerror(self, adv):
        """4.5 Non-list input_tasks raises TypeError."""
        with pytest.raises(TypeError, match="input_tasks must be a list"):
            adv._validate_types("not_a_list", [])

    def test_output_not_list_raises_typeerror(self, adv):
        """4.6 Non-list output_tasks raises TypeError."""
        with pytest.raises(TypeError, match="output_tasks must be a list"):
            adv._validate_types([], "not_a_list")

    def test_invalid_type_in_list_raises_typeerror(self, adv):
        """4.7 Invalid type (int) in list raises TypeError."""
        with pytest.raises(TypeError, match="must be"):
            adv._validate_types([42], [])

    def test_empty_lists_returns_true(self, adv):
        """4.8 Empty lists return True."""
        assert adv._validate_types([], []) is True


# ===================================================================
# Group 5: _validate_validity()
# ===================================================================


class TestValidateValidity:
    """Task group 5 — _validate_validity() with integer error codes (bug fix #1)."""

    def test_valid_open_tasks_return_true(self, adv):
        """5.1 Valid open tasks return True."""
        task = _make_nidaqmx_task()
        task.is_task_done.return_value = False
        assert adv._validate_validity([task]) is True

    def test_invalid_task_error_code_integer(self, adv, advanced_module):
        """5.2/Bug fix #1: DaqError with error_code -200088 as INTEGER detected."""
        task = _make_nidaqmx_task()
        DaqError = advanced_module.DaqError
        error = DaqError("Invalid task")
        error.error_code = -200088  # INTEGER, not string
        task.is_task_done.side_effect = error
        assert adv._validate_validity([task]) is False

    def test_no_channels_error_code_integer(self, adv, advanced_module):
        """5.3 DaqError with error_code -200478 as INTEGER detected for no channels."""
        task = _make_nidaqmx_task()
        DaqError = advanced_module.DaqError
        error = DaqError("No channels")
        error.error_code = -200478  # INTEGER
        task.is_task_done.return_value = False
        # channel_names access raises
        type(task).channel_names = PropertyMock(side_effect=error)
        assert adv._validate_validity([task]) is False

    def test_empty_list_returns_true(self, adv):
        """5.4 Empty list returns True."""
        assert adv._validate_validity([]) is True

    def test_string_error_code_not_caught_regression(self, adv, advanced_module):
        """5.5 Regression test: string error code "-200088" must NOT match
        the integer comparison — verifies we compare as integer, not string.
        If the code used string comparison, this string would match; with
        correct integer comparison, this unknown-typed error is re-raised."""
        task = _make_nidaqmx_task()
        DaqError = advanced_module.DaqError
        error = DaqError("Invalid task")
        error.error_code = "-200088"  # STRING — must not match integer -200088
        task.is_task_done.side_effect = error
        # With correct integer comparison, this string error_code won't match
        # -200088, so the error should be re-raised (not caught as "invalid task").
        with pytest.raises(type(error)):
            adv._validate_validity([task])

    def test_other_daqerror_reraised(self, adv, advanced_module):
        """5.5 DaqError with unrecognized error_code is re-raised."""
        task = _make_nidaqmx_task()
        DaqError = advanced_module.DaqError
        error = DaqError("Unknown error")
        error.error_code = -999999
        task.is_task_done.side_effect = error
        with pytest.raises(type(error)):
            adv._validate_validity([task])


# ===================================================================
# Group 6: _validate_sample_rates()
# ===================================================================


class TestValidateSampleRates:
    """Task group 6 — _validate_sample_rates()."""

    def test_same_rate_returns_true(self, adv):
        """6.1 Two tasks at same rate (25600 Hz) return True."""
        t1 = _make_nidaqmx_task(samp_clk_rate=25600.0)
        t2 = _make_nidaqmx_task(samp_clk_rate=25600.0)
        assert adv._validate_sample_rates([t1, t2]) is True

    def test_different_rates_returns_false(self, adv):
        """6.2 Two tasks at different rates return False."""
        t1 = _make_nidaqmx_task(samp_clk_rate=25600.0)
        t2 = _make_nidaqmx_task(samp_clk_rate=51200.0)
        assert adv._validate_sample_rates([t1, t2]) is False

    def test_single_task_returns_true(self, adv):
        """6.3 Single task always returns True."""
        t1 = _make_nidaqmx_task(samp_clk_rate=25600.0)
        assert adv._validate_sample_rates([t1]) is True

    def test_empty_list_returns_true(self, adv):
        """6.4 Empty list returns True."""
        assert adv._validate_sample_rates([]) is True

    def test_three_tasks_same_rate(self, adv):
        """6.5 Three tasks at same rate return True."""
        tasks = [_make_nidaqmx_task(samp_clk_rate=10000.0) for _ in range(3)]
        assert adv._validate_sample_rates(tasks) is True


# ===================================================================
# Group 7: _validate_timing()
# ===================================================================


class TestValidateTiming:
    """Task group 7 — _validate_timing().

    FR-5.5 (relaxed by sync-validation-simulated-coverage): rate and
    samples-per-channel mismatches fail; clock-source readback mismatches
    warn and pass (readback strings are not canonical — a master reads back
    its timebase terminal, a slave the exported sample-clock terminal).
    """

    def test_matching_timing_returns_true_without_warning(self, adv):
        """7.1 Matching clock_source, clock_rate, samples_per_channel returns True, no warning."""
        t1 = _make_nidaqmx_task(samp_clk_src="OnboardClock", samp_clk_rate=25600, samp_quant_samp_per_chan=25600)
        t2 = _make_nidaqmx_task(samp_clk_src="OnboardClock", samp_clk_rate=25600, samp_quant_samp_per_chan=25600)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert adv._validate_timing([t1, t2]) is True

    def test_mismatching_clock_source_warns_and_passes(self, adv):
        """7.2 Mismatching samp_clk_src readbacks emit UserWarning naming
        each task and its source, and return True (master/slave pattern)."""
        t1 = _make_nidaqmx_task(
            name="MasterAI",
            samp_clk_src="/SimDev1/ai/SampleClockTimebase",
            samp_clk_rate=25600,
            samp_quant_samp_per_chan=25600,
        )
        t2 = _make_nidaqmx_task(
            name="SlaveDI",
            samp_clk_src="/SimDev1/ai/SampleClock",
            samp_clk_rate=25600,
            samp_quant_samp_per_chan=25600,
        )
        with pytest.warns(UserWarning) as record:
            assert adv._validate_timing([t1, t2]) is True
        assert len(record) == 1, "exactly one warning expected"
        msg = str(record[0].message)
        # repr-delimited pairs: the closing quote makes the assertion exact
        # ('/SimDev1/ai/SampleClock' is otherwise a substring of
        # '/SimDev1/ai/SampleClockTimebase')
        assert "'MasterAI': '/SimDev1/ai/SampleClockTimebase'" in msg
        assert "'SlaveDI': '/SimDev1/ai/SampleClock'" in msg

    def test_mismatching_source_and_rate_returns_false(self, adv):
        """7.2b Source mismatch combined with rate mismatch still hard-fails:
        the relaxation must not let a warning short-circuit the rate check."""
        t1 = _make_nidaqmx_task(
            samp_clk_src="/SimDev1/ai/SampleClockTimebase", samp_clk_rate=25600.0
        )
        t2 = _make_nidaqmx_task(
            samp_clk_src="/SimDev2/ai/SampleClock", samp_clk_rate=51200.0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            assert adv._validate_timing([t1, t2]) is False

    def test_mismatching_rate_returns_false(self, adv):
        """7.3 Mismatching samp_clk_rate returns False (hard failure, no pass)."""
        t1 = _make_nidaqmx_task(samp_clk_rate=25600.0)
        t2 = _make_nidaqmx_task(samp_clk_rate=51200.0)
        assert adv._validate_timing([t1, t2]) is False

    def test_mismatching_samples_per_channel_returns_false(self, adv):
        """7.4 Mismatching samples_per_channel returns False (hard failure)."""
        t1 = _make_nidaqmx_task(samp_quant_samp_per_chan=25600)
        t2 = _make_nidaqmx_task(samp_quant_samp_per_chan=51200)
        assert adv._validate_timing([t1, t2]) is False

    def test_single_task_returns_true(self, adv):
        """7.5 Single task returns True."""
        t1 = _make_nidaqmx_task()
        assert adv._validate_timing([t1]) is True

    def test_empty_list_returns_true(self, adv):
        """7.6 Empty list returns True."""
        assert adv._validate_timing([]) is True


# ===================================================================
# Group 8: _validate_triggers()
# ===================================================================


class TestValidateTriggers:
    """Task group 8 — _validate_triggers() with trigger_type default (bug fix #2)."""

    def test_no_trigger_returns_true_software(self, adv):
        """8.1 No trigger (NONE type) returns True and trigger_type='software'."""
        t1 = _make_nidaqmx_task(trig_type_name="NONE")
        t2 = _make_nidaqmx_task(trig_type_name="NONE")
        assert adv._validate_triggers([t1, t2]) is True
        assert adv.trigger_type == "software"

    def test_consistent_digital_edge_returns_true_hardware(self, adv):
        """8.2 Consistent digital_edge triggers return True and trigger_type='hardware'."""
        t1 = _make_nidaqmx_task(trig_type_name="DIGITAL_EDGE", dig_edge_src="/cDAQ1/PFI0")
        t2 = _make_nidaqmx_task(trig_type_name="DIGITAL_EDGE", dig_edge_src="/cDAQ1/PFI0")
        assert adv._validate_triggers([t1, t2]) is True
        assert adv.trigger_type == "hardware"

    def test_consistent_analog_edge_returns_true_hardware(self, adv):
        """8.3 Consistent analog_edge triggers return True and trigger_type='hardware'."""
        t1 = _make_nidaqmx_task(trig_type_name="ANALOG_EDGE", anlg_edge_src="/cDAQ1Mod1/ai0")
        t2 = _make_nidaqmx_task(trig_type_name="ANALOG_EDGE", anlg_edge_src="/cDAQ1Mod1/ai0")
        assert adv._validate_triggers([t1, t2]) is True
        assert adv.trigger_type == "hardware"

    def test_mixed_trigger_types_returns_false(self, adv):
        """8.4 Mixed trigger types (NONE and DIGITAL_EDGE) returns False."""
        t1 = _make_nidaqmx_task(trig_type_name="NONE")
        t2 = _make_nidaqmx_task(trig_type_name="DIGITAL_EDGE", dig_edge_src="/cDAQ1/PFI0")
        assert adv._validate_triggers([t1, t2]) is False

    def test_same_type_different_sources_returns_false(self, adv):
        """8.5 Same type but different sources returns False."""
        t1 = _make_nidaqmx_task(trig_type_name="DIGITAL_EDGE", dig_edge_src="/cDAQ1/PFI0")
        t2 = _make_nidaqmx_task(trig_type_name="DIGITAL_EDGE", dig_edge_src="/cDAQ1/PFI1")
        assert adv._validate_triggers([t1, t2]) is False

    def test_single_task_returns_true(self, adv):
        """8.6 Single task returns True."""
        t1 = _make_nidaqmx_task(trig_type_name="DIGITAL_EDGE", dig_edge_src="/cDAQ1/PFI0")
        assert adv._validate_triggers([t1]) is True

    def test_empty_list_returns_true(self, adv):
        """8.7 Empty list returns True."""
        assert adv._validate_triggers([]) is True

    def test_trigger_type_defaults_software_regression(self, adv):
        """8.8 Bug fix #2 regression: trigger_type defaults to 'software' when no hardware triggers."""
        t1 = _make_nidaqmx_task(trig_type_name="NONE")
        t2 = _make_nidaqmx_task(trig_type_name="NONE")
        adv._validate_triggers([t1, t2])
        # Must be 'software', not raise AttributeError
        assert adv.trigger_type == "software"


# ===================================================================
# Group 9: _validate_acquisition_mode()
# ===================================================================


class TestValidateAcquisitionMode:
    """Task group 9 — _validate_acquisition_mode()."""

    def test_finite_hardware_returns_true(self, adv):
        """9.1 FINITE mode + trigger_type='hardware' returns True."""
        adv.trigger_type = "hardware"
        t1 = _make_nidaqmx_task(samp_quant_samp_mode_name="FINITE")
        assert adv._validate_acquisition_mode([t1]) is True

    def test_finite_software_returns_false(self, adv):
        """9.2 FINITE mode + trigger_type='software' returns False."""
        adv.trigger_type = "software"
        t1 = _make_nidaqmx_task(samp_quant_samp_mode_name="FINITE")
        assert adv._validate_acquisition_mode([t1]) is False

    def test_continuous_software_returns_true(self, adv):
        """9.3 CONTINUOUS mode + trigger_type='software' returns True."""
        adv.trigger_type = "software"
        t1 = _make_nidaqmx_task(samp_quant_samp_mode_name="CONTINUOUS")
        assert adv._validate_acquisition_mode([t1]) is True

    def test_continuous_hardware_returns_false(self, adv):
        """9.4 CONTINUOUS mode + trigger_type='hardware' returns False."""
        adv.trigger_type = "hardware"
        t1 = _make_nidaqmx_task(samp_quant_samp_mode_name="CONTINUOUS")
        assert adv._validate_acquisition_mode([t1]) is False

    def test_mixed_modes_returns_false(self, adv):
        """9.5 Mixed modes (FINITE and CONTINUOUS) returns False."""
        adv.trigger_type = "hardware"
        t1 = _make_nidaqmx_task(samp_quant_samp_mode_name="FINITE")
        t2 = _make_nidaqmx_task(samp_quant_samp_mode_name="CONTINUOUS")
        assert adv._validate_acquisition_mode([t1, t2]) is False

    def test_empty_list_returns_true(self, adv):
        """9.6 Empty list returns True."""
        assert adv._validate_acquisition_mode([]) is True


# ===================================================================
# Group 10: acquire_with_hardware_trigger()
# ===================================================================


class TestAcquireWithHardwareTrigger:
    """Task group 10 — acquire_with_hardware_trigger()."""

    def test_starts_all_tasks_before_reading(self, adv):
        """10.1 All input tasks started before any reads."""
        t1 = _make_nidaqmx_task("Task1", channel_names=["ch0"])
        t2 = _make_nidaqmx_task("Task2", channel_names=["ch1"])

        # Track call order
        call_order = []
        t1.start.side_effect = lambda: call_order.append("start_t1")
        t2.start.side_effect = lambda: call_order.append("start_t2")
        t1.read.side_effect = lambda **kw: (call_order.append("read_t1"), [[1.0, 2.0]])[1]
        t2.read.side_effect = lambda **kw: (call_order.append("read_t2"), [[3.0, 4.0]])[1]

        adv.input_tasks = [t1, t2]
        adv.acquire_with_hardware_trigger()

        # Both starts must come before any reads
        start_indices = [call_order.index("start_t1"), call_order.index("start_t2")]
        read_indices = [call_order.index("read_t1"), call_order.index("read_t2")]
        assert max(start_indices) < min(read_indices)

    def test_reads_all_available(self, adv):
        """10.2 Reads READ_ALL_AVAILABLE from each task."""
        t1 = _make_nidaqmx_task("Task1", channel_names=["ch0"])
        t1.read.return_value = [[1.0, 2.0]]
        adv.input_tasks = [t1]
        adv.acquire_with_hardware_trigger()
        t1.read.assert_called_once()
        # Check that READ_ALL_AVAILABLE was passed
        _, kwargs = t1.read.call_args
        assert kwargs.get("number_of_samples_per_channel") == -1

    def test_stops_all_tasks_after_reading(self, adv):
        """10.3 All input tasks stopped after reading."""
        t1 = _make_nidaqmx_task("Task1", channel_names=["ch0"])
        t1.read.return_value = [[1.0, 2.0]]
        adv.input_tasks = [t1]
        adv.acquire_with_hardware_trigger()
        t1.stop.assert_called_once()

    def test_returns_nested_dict(self, adv):
        """10.4 Returns nested dict {task_name: {channel_name: numpy_array}}."""
        t1 = _make_nidaqmx_task("Voltage", channel_names=["ch0", "ch1"])
        t1.read.return_value = [[1.0, 2.0], [3.0, 4.0]]
        adv.input_tasks = [t1]

        result = adv.acquire_with_hardware_trigger()

        assert "Voltage" in result
        assert "ch0" in result["Voltage"]
        assert "ch1" in result["Voltage"]
        np.testing.assert_array_equal(result["Voltage"]["ch0"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result["Voltage"]["ch1"], np.array([3.0, 4.0]))

    def test_multiple_tasks_multiple_channels(self, adv):
        """10.5 Multiple tasks each having multiple channels."""
        t1 = _make_nidaqmx_task("VoltageTask", channel_names=["v0", "v1"])
        t1.read.return_value = [[1.0, 2.0], [3.0, 4.0]]
        t2 = _make_nidaqmx_task("AccelTask", channel_names=["a0"])
        t2.read.return_value = [5.0, 6.0]  # single channel = 1D

        adv.input_tasks = [t1, t2]
        result = adv.acquire_with_hardware_trigger()

        assert "VoltageTask" in result
        assert "AccelTask" in result
        assert len(result["VoltageTask"]) == 2
        assert len(result["AccelTask"]) == 1

    def test_single_task_single_channel(self, adv):
        """10.6 Single task single channel."""
        t1 = _make_nidaqmx_task("Single", channel_names=["ch0"])
        t1.read.return_value = [1.0, 2.0, 3.0]  # single channel = 1D
        adv.input_tasks = [t1]

        result = adv.acquire_with_hardware_trigger()
        assert "Single" in result
        assert "ch0" in result["Single"]
        np.testing.assert_array_equal(result["Single"]["ch0"], np.array([1.0, 2.0, 3.0]))


# ===================================================================
# Group 11: acquire_with_software_trigger()
# ===================================================================


class TestAcquireWithSoftwareTrigger:
    """Task group 11 — acquire_with_software_trigger() with task.start() fix (bug fix #3)."""

    def _setup_trigger(self, adv, n_channels=1, sample_rate=25600.0):
        """Helper to set up a mock pyTrigger on the instance."""
        trigger = MagicMock()
        # _reset_trigger() accesses these attributes directly
        trigger.ringbuff = MagicMock()
        trigger.rows = 5000
        trigger.rows_left = 5000
        trigger.triggered = False
        trigger.first_data = True
        # PropertyMock for finished: getter returns False then True (after 2 calls),
        # setter accepts the value without error (needed by _reset_trigger).
        call_count = [0]
        _finished_val = [False]
        def check_finished(*args):
            if args:
                # setter call from _reset_trigger
                _finished_val[0] = args[0]
                call_count[0] = 0
                return None
            call_count[0] += 1
            return call_count[0] >= 2
        type(trigger).finished = PropertyMock(side_effect=check_finished)
        trigger.get_data.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        adv.trigger = trigger
        adv._trigger_is_set = True
        adv.input_sample_rate = sample_rate
        return trigger

    def test_single_task_succeeds(self, adv):
        """11.1 Single task software trigger succeeds and returns data."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0"])
        task.read.return_value = [1.0, 2.0, 3.0]
        adv.input_tasks = [task]
        adv.input_channels = ["ch0"]
        trigger = self._setup_trigger(adv)

        result = adv.acquire_with_software_trigger(return_dict=False)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0

    def test_multiple_tasks_raises_valueerror(self, adv):
        """11.2 >1 task raises ValueError."""
        t1 = _make_nidaqmx_task("T1")
        t2 = _make_nidaqmx_task("T2")
        adv.input_tasks = [t1, t2]

        with pytest.raises(ValueError, match="Software trigger can only be used with one"):
            adv.acquire_with_software_trigger()

    def test_task_start_called_before_read(self, adv):
        """11.3 Bug fix #3 regression: task.start() called BEFORE read loop."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0"])
        adv.input_tasks = [task]
        adv.input_channels = ["ch0"]
        trigger = self._setup_trigger(adv)

        call_order = []
        task.start.side_effect = lambda: call_order.append("start")
        original_read = task.read
        task.read.side_effect = lambda *a, **kw: (call_order.append("read"), [1.0, 2.0])[1]

        adv.acquire_with_software_trigger(return_dict=False)

        assert "start" in call_order
        start_idx = call_order.index("start")
        read_idx = call_order.index("read")
        assert start_idx < read_idx, "task.start() must be called before task.read()"

    def test_flush_buffer_before_acquisition(self, adv):
        """11.4 Flush buffer before acquisition loop."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0"])
        adv.input_tasks = [task]
        adv.input_channels = ["ch0"]
        trigger = self._setup_trigger(adv)

        read_calls = []
        task.read.side_effect = lambda *a, **kw: (read_calls.append(kw), [1.0, 2.0])[1]

        adv.acquire_with_software_trigger(return_dict=False)

        # First read should be flush (READ_ALL_AVAILABLE with timeout)
        assert len(read_calls) >= 2  # at least flush + one loop read

    def test_reads_until_trigger_finished(self, adv):
        """11.5 Reads in loop until trigger.finished."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0"])
        task.read.return_value = [1.0, 2.0]
        adv.input_tasks = [task]
        adv.input_channels = ["ch0"]
        trigger = self._setup_trigger(adv)

        adv.acquire_with_software_trigger(return_dict=False)

        # trigger.add_data should have been called in the loop
        assert trigger.add_data.call_count >= 1

    def test_single_channel_1d_reshaped_to_2d(self, adv):
        """11.6 Single-channel 1D data reshaped to 2D for trigger."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0"])
        task.read.return_value = [1.0, 2.0, 3.0]  # 1D for single channel
        adv.input_tasks = [task]
        adv.input_channels = ["ch0"]
        trigger = self._setup_trigger(adv)

        adv.acquire_with_software_trigger(return_dict=False)

        # Verify that data added to trigger was 2D
        for call_args in trigger.add_data.call_args_list:
            data = call_args[0][0]
            assert data.ndim == 2

    def test_stops_task_after_acquisition(self, adv):
        """11.7 Task stopped after acquisition."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0"])
        task.read.return_value = [1.0, 2.0]
        adv.input_tasks = [task]
        adv.input_channels = ["ch0"]
        trigger = self._setup_trigger(adv)

        adv.acquire_with_software_trigger(return_dict=False)
        task.stop.assert_called_once()

    def test_return_dict_true(self, adv):
        """11.8 return_dict=True returns dict with channel names and 'time' key."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0", "ch1"])
        task.read.return_value = [[1.0, 2.0], [3.0, 4.0]]
        adv.input_tasks = [task]
        adv.input_channels = ["ch0", "ch1"]
        adv.input_sample_rate = 1000.0

        trigger = self._setup_trigger(adv, n_channels=2)
        trigger.get_data.return_value = np.array([[1.0, 3.0], [2.0, 4.0]])

        result = adv.acquire_with_software_trigger(return_dict=True)
        assert isinstance(result, dict)
        assert "ch0" in result
        assert "ch1" in result
        assert "time" in result

    def test_return_dict_false(self, adv):
        """11.9 return_dict=False returns numpy array."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0"])
        task.read.return_value = [1.0, 2.0]
        adv.input_tasks = [task]
        adv.input_channels = ["ch0"]

        trigger = self._setup_trigger(adv)
        trigger.get_data.return_value = np.array([[1.0], [2.0]])

        result = adv.acquire_with_software_trigger(return_dict=False)
        assert isinstance(result, np.ndarray)


# ===================================================================
# Group 12: acquire() Dispatch
# ===================================================================


class TestAcquireDispatch:
    """Task group 12 — acquire() dispatch logic."""

    def test_hardware_trigger_dispatches(self, adv):
        """12.1 trigger_type='hardware' calls acquire_with_hardware_trigger()."""
        adv.trigger_type = "hardware"
        adv.input_tasks = [_make_nidaqmx_task()]
        with patch.object(adv, "acquire_with_hardware_trigger", return_value={}) as mock_hw:
            adv.acquire()
            mock_hw.assert_called_once()

    def test_software_trigger_dispatches(self, adv):
        """12.2 trigger_type='software' calls acquire_with_software_trigger()."""
        adv.trigger_type = "software"
        adv.input_tasks = [_make_nidaqmx_task()]
        with patch.object(adv, "acquire_with_software_trigger", return_value=np.array([])) as mock_sw:
            adv.acquire()
            mock_sw.assert_called_once()

    def test_default_software_dispatches(self, adv):
        """12.3 Default trigger_type='software' calls acquire_with_software_trigger()."""
        # trigger_type defaults to 'software' from constructor
        adv.input_tasks = [_make_nidaqmx_task()]
        with patch.object(adv, "acquire_with_software_trigger", return_value=np.array([])) as mock_sw:
            adv.acquire()
            mock_sw.assert_called_once()


# ===================================================================
# Group 13: connect() / disconnect()
# ===================================================================


class TestConnectDisconnect:
    """Task group 13 — connect() and disconnect()."""

    def test_connect_calls_ping(self, adv):
        """13.1 connect() calls ping() and returns its result."""
        with patch.object(adv, "_define_required_devices"):
            with patch.object(adv, "ping", return_value=True) as mock_ping:
                result = adv.connect()
                mock_ping.assert_called_once()
                assert result is True

    def test_connect_sets_connected_true(self, adv):
        """13.1b connect() sets _connected=True on successful ping."""
        with patch.object(adv, "_define_required_devices"):
            with patch.object(adv, "ping", return_value=True):
                adv.connect()
                assert adv._connected is True

    def test_connect_returns_false_when_ping_fails(self, adv):
        """13.1c connect() returns False and _connected stays False when ping fails."""
        with patch.object(adv, "_define_required_devices"):
            with patch.object(adv, "ping", return_value=False):
                result = adv.connect()
                assert result is False
                assert adv._connected is False

    def test_connect_defines_required_devices(self, adv):
        """13.2 connect() defines required devices from all tasks."""
        with patch.object(adv, "_define_required_devices") as mock_define:
            with patch.object(adv, "ping", return_value=True):
                adv.connect()
                mock_define.assert_called_once()

    def test_disconnect_closes_input_tasks(self, adv):
        """13.3 disconnect() closes all input tasks."""
        t1 = _make_nidaqmx_task("In1")
        t2 = _make_nidaqmx_task("In2")
        adv.input_tasks = [t1, t2]
        adv.output_tasks = []
        adv._connected = True
        adv.disconnect()
        t1.close.assert_called_once()
        t2.close.assert_called_once()

    def test_disconnect_closes_output_tasks(self, adv):
        """13.4 disconnect() closes all output tasks."""
        t_out = _make_nidaqmx_task("Out1")
        adv.input_tasks = []
        adv.output_tasks = [t_out]
        adv._connected = True
        adv.disconnect()
        t_out.close.assert_called_once()

    def test_disconnect_idempotent(self, adv):
        """13.5 Calling disconnect() twice does not raise."""
        adv.input_tasks = []
        adv.output_tasks = []
        adv.disconnect()
        adv.disconnect()  # Should not raise

    def test_disconnect_sets_connected_false(self, adv):
        """13.6 disconnect() sets _connected=False."""
        adv.input_tasks = []
        adv.output_tasks = []
        adv._connected = True
        adv.disconnect()
        assert adv._connected is False

    def test_disconnect_input_exception_warns_not_propagated(self, adv):
        """13.7 disconnect() emits warning when input task close() raises."""
        import warnings

        t1 = _make_nidaqmx_task("In1")
        t1.close.side_effect = RuntimeError("cleanup error")
        adv.input_tasks = [t1]
        adv.output_tasks = []
        adv._connected = True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adv.disconnect()

        assert len(w) >= 1
        assert "cleanup error" in str(w[0].message)
        assert adv._connected is False

    def test_disconnect_output_exception_warns_not_propagated(self, adv):
        """13.8 disconnect() emits warning when output task close() raises."""
        import warnings

        t_out = _make_nidaqmx_task("Out1")
        t_out.close.side_effect = RuntimeError("output cleanup error")
        adv.input_tasks = []
        adv.output_tasks = [t_out]
        adv._connected = True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adv.disconnect()

        assert len(w) >= 1
        assert "output cleanup error" in str(w[0].message)
        assert adv._connected is False


# ===================================================================
# Group 14: set_trigger()
# ===================================================================


class TestSetTrigger:
    """Task group 14 — set_trigger() for software trigger."""

    def test_creates_pytrigger_instance(self, adv, advanced_module):
        """14.1 set_trigger() creates pyTrigger instance."""
        task = _make_nidaqmx_task("Task1", channel_names=["ch0", "ch1"])
        task.timing.samp_clk_rate = 25600.0
        adv.input_tasks = [task]

        adv.set_trigger(
            n_samples=25600,
            trigger_channel=0,
            trigger_level=0.5,
        )
        assert adv._trigger_is_set is True

    def test_uses_channel_count_from_first_task(self, adv, advanced_module):
        """14.2 Uses channel count from first input task."""
        task = _make_nidaqmx_task("Task1", channel_names=["ch0", "ch1", "ch2"])
        task.timing.samp_clk_rate = 25600.0
        adv.input_tasks = [task]

        with patch.object(advanced_module, "pyTrigger") as mock_trigger_cls:
            adv.set_trigger(
                n_samples=25600,
                trigger_channel=0,
                trigger_level=0.5,
            )
            # pyTrigger should receive the number of channels
            call_args = mock_trigger_cls.call_args
            # Check that n_channels=3 was passed
            assert 3 in call_args[0] or call_args[1].get("rows", None) == 3

    def test_raises_when_no_input_tasks(self, adv):
        """14.3 Raises ValueError when no input tasks."""
        adv.input_tasks = []
        with pytest.raises(ValueError, match="[Nn]o input task"):
            adv.set_trigger(n_samples=25600, trigger_channel=0, trigger_level=0.5)


# ===================================================================
# Group 15: ping() and Device Management
# ===================================================================


class TestPingAndDevices:
    """Task group 15 — ping(), _define_required_devices(), _get_task_devices()."""

    def test_ping_returns_true_all_devices_present(self, adv, advanced_module):
        """15.1/15.2 ping() returns True when all devices present."""
        adv.required_devices = {"cDAQ1Mod1", "cDAQ1Mod2"}
        with patch.object(
            advanced_module, "get_connected_devices",
            return_value={"cDAQ1Mod1", "cDAQ1Mod2", "cDAQ1Mod3"}
        ):
            assert adv.ping() is True

    def test_ping_returns_false_device_missing(self, adv, advanced_module):
        """15.3 ping() returns False when device missing."""
        adv.required_devices = {"cDAQ1Mod1", "cDAQ1Mod2"}
        with patch.object(
            advanced_module, "get_connected_devices",
            return_value={"cDAQ1Mod1"}
        ):
            assert adv.ping() is False

    def test_define_required_devices_collects_all(self, adv):
        """15.4 _define_required_devices() collects devices from all tasks."""
        dev1 = MagicMock()
        dev1.name = "cDAQ1Mod1"
        dev2 = MagicMock()
        dev2.name = "cDAQ1Mod2"

        t_in = _make_nidaqmx_task(devices=[dev1])
        t_out = _make_nidaqmx_task(devices=[dev2])
        adv.input_tasks = [t_in]
        adv.output_tasks = [t_out]

        adv._define_required_devices()
        assert adv.required_devices == {"cDAQ1Mod1", "cDAQ1Mod2"}

    def test_get_task_devices_returns_set(self, adv):
        """15.5 _get_task_devices() returns set of device names."""
        dev1 = MagicMock()
        dev1.name = "cDAQ1Mod1"
        dev2 = MagicMock()
        dev2.name = "cDAQ1Mod2"
        task = _make_nidaqmx_task(devices=[dev1, dev2])

        result = adv._get_task_devices(task)
        assert result == {"cDAQ1Mod1", "cDAQ1Mod2"}

    def test_ping_empty_required_devices(self, adv, advanced_module):
        """15.2 ping() with empty required_devices returns True."""
        adv.required_devices = set()
        with patch.object(
            advanced_module, "get_connected_devices",
            return_value=set()
        ):
            assert adv.ping() is True


# ===================================================================
# Group 16: Thread Safety
# ===================================================================


class TestThreadSafety:
    """Thread safety — RLock on hardware-accessing methods."""

    def test_acquire_uses_lock(self, adv):
        """acquire() uses the RLock."""
        adv.trigger_type = "hardware"
        adv.input_tasks = [_make_nidaqmx_task()]
        adv.input_tasks[0].read.return_value = [[1.0]]

        # Replace lock with a tracking mock (no spec — dunder methods need
        # direct attribute assignment which spec'd mocks block)
        lock = MagicMock()
        lock.__enter__ = MagicMock(return_value=None)
        lock.__exit__ = MagicMock(return_value=False)
        adv._lock = lock

        adv.acquire()
        lock.__enter__.assert_called()


# ===================================================================
# Change multi-handler-advanced-acquisition
# Group 1: Hardware trigger hooks
# ===================================================================


class TestSetHardwareTriggerFunctions:
    """Group 1 — set_hardware_trigger_functions() registration."""

    def test_hooks_default_none_at_init(self, adv):
        """1.1 Both hook attributes default to None."""
        assert adv._hw_trigger_start_fn is None
        assert adv._hw_trigger_support_fn is None

    def test_register_both_hooks(self, adv):
        """1.1 Both callables are stored."""
        def start_fn():
            pass

        def support_fn(mode):
            pass

        adv.set_hardware_trigger_functions(
            start_function=start_fn, support_function=support_fn
        )
        assert adv._hw_trigger_start_fn is start_fn
        assert adv._hw_trigger_support_fn is support_fn

    def test_register_start_only(self, adv):
        """1.1 Registering only start_function leaves support hook None."""
        def start_fn():
            pass

        adv.set_hardware_trigger_functions(start_function=start_fn)
        assert adv._hw_trigger_start_fn is start_fn
        assert adv._hw_trigger_support_fn is None

    def test_clear_hooks_with_no_args(self, adv):
        """1.1 Calling with no arguments clears previously registered hooks."""
        adv.set_hardware_trigger_functions(
            start_function=lambda: None, support_function=lambda m: None
        )
        adv.set_hardware_trigger_functions()
        assert adv._hw_trigger_start_fn is None
        assert adv._hw_trigger_support_fn is None

    def test_non_callable_start_raises_typeerror(self, adv):
        """1.1 Non-callable start_function raises TypeError."""
        with pytest.raises(TypeError, match="start_function"):
            adv.set_hardware_trigger_functions(start_function=42)

    def test_non_callable_support_raises_typeerror(self, adv):
        """1.1 Non-callable support_function raises TypeError."""
        with pytest.raises(TypeError, match="support_function"):
            adv.set_hardware_trigger_functions(support_function="not-callable")

    def test_typeerror_leaves_hook_state_unchanged(self, adv):
        """1.1 On TypeError, no hook state changes (spec scenario)."""
        def start_fn():
            pass

        def support_fn(mode):
            pass

        adv.set_hardware_trigger_functions(
            start_function=start_fn, support_function=support_fn
        )

        # Bad first argument — neither hook may change
        with pytest.raises(TypeError):
            adv.set_hardware_trigger_functions(
                start_function=1, support_function=lambda m: None
            )
        assert adv._hw_trigger_start_fn is start_fn
        assert adv._hw_trigger_support_fn is support_fn

        # Bad second argument — neither hook may change (validate-then-assign)
        with pytest.raises(TypeError):
            adv.set_hardware_trigger_functions(
                start_function=lambda: None, support_function=2
            )
        assert adv._hw_trigger_start_fn is start_fn
        assert adv._hw_trigger_support_fn is support_fn


class TestHardwareTriggerHookInvocation:
    """Group 1 — hook invocation order in acquire_with_hardware_trigger()."""

    def _make_ordered_tasks(self, adv, call_order):
        """Create two input tasks that record start/read/stop call order."""
        t1 = _make_nidaqmx_task("Task1", channel_names=["ch0"])
        t2 = _make_nidaqmx_task("Task2", channel_names=["ch1"])
        t1.start.side_effect = lambda: call_order.append("start_t1")
        t2.start.side_effect = lambda: call_order.append("start_t2")
        t1.read.side_effect = lambda **kw: (call_order.append("read_t1"), [1.0, 2.0])[1]
        t2.read.side_effect = lambda **kw: (call_order.append("read_t2"), [3.0, 4.0])[1]
        t1.stop.side_effect = lambda: call_order.append("stop_t1")
        t2.stop.side_effect = lambda: call_order.append("stop_t2")
        adv.input_tasks = [t1, t2]
        return t1, t2

    def test_support_called_before_arming(self, adv):
        """1.2 support_function(custom_mode) runs before any task.start()."""
        call_order = []
        self._make_ordered_tasks(adv, call_order)
        adv.set_hardware_trigger_functions(
            support_function=lambda mode: call_order.append(f"support:{mode}")
        )

        adv.acquire_with_hardware_trigger(custom_mode="modeA")

        assert call_order[0] == "support:modeA"
        assert call_order.index("support:modeA") < call_order.index("start_t1")

    def test_support_skipped_without_custom_mode(self, adv):
        """1.2 support_function not called when custom_mode is None."""
        call_order = []
        self._make_ordered_tasks(adv, call_order)
        support = MagicMock()
        start = MagicMock()
        adv.set_hardware_trigger_functions(
            start_function=start, support_function=support
        )

        adv.acquire_with_hardware_trigger()

        support.assert_not_called()
        start.assert_called_once_with()

    def test_support_skipped_when_unset_with_custom_mode(self, adv):
        """1.2 custom_mode with no support hook set runs without error."""
        call_order = []
        self._make_ordered_tasks(adv, call_order)

        result = adv.acquire_with_hardware_trigger(custom_mode="modeB")

        assert "Task1" in result and "Task2" in result

    def test_start_fn_called_after_arming_before_reads(self, adv):
        """1.2 start_function() runs after every task.start(), before reads."""
        call_order = []
        self._make_ordered_tasks(adv, call_order)
        adv.set_hardware_trigger_functions(
            start_function=lambda: call_order.append("start_fn")
        )

        adv.acquire_with_hardware_trigger()

        start_fn_idx = call_order.index("start_fn")
        assert call_order.index("start_t1") < start_fn_idx
        assert call_order.index("start_t2") < start_fn_idx
        assert start_fn_idx < call_order.index("read_t1")
        assert start_fn_idx < call_order.index("read_t2")

    def test_full_hook_sequence(self, adv):
        """1.2 Full sequence: support → arm all → start_fn → read all → stop all."""
        call_order = []
        self._make_ordered_tasks(adv, call_order)
        adv.set_hardware_trigger_functions(
            start_function=lambda: call_order.append("start_fn"),
            support_function=lambda mode: call_order.append("support"),
        )

        result = adv.acquire_with_hardware_trigger(custom_mode="modeA")

        assert call_order == [
            "support",
            "start_t1", "start_t2",
            "start_fn",
            "read_t1", "read_t2",
            "stop_t1", "stop_t2",
        ]
        assert "Task1" in result and "Task2" in result

    def test_start_fn_failure_stops_armed_tasks(self, adv):
        """1.2 start_function raising → all armed tasks stopped, exception propagates."""
        call_order = []
        t1, t2 = self._make_ordered_tasks(adv, call_order)

        def failing_start():
            raise RuntimeError("PLC trigger failed")

        adv.set_hardware_trigger_functions(start_function=failing_start)

        with pytest.raises(RuntimeError, match="PLC trigger failed"):
            adv.acquire_with_hardware_trigger()

        t1.stop.assert_called_once()
        t2.stop.assert_called_once()

    def test_support_failure_propagates_no_tasks_started(self, adv):
        """1.2 support_function raising before arming → no task started."""
        call_order = []
        t1, t2 = self._make_ordered_tasks(adv, call_order)

        def failing_support(mode):
            raise RuntimeError("mode selection failed")

        adv.set_hardware_trigger_functions(support_function=failing_support)

        with pytest.raises(RuntimeError, match="mode selection failed"):
            adv.acquire_with_hardware_trigger(custom_mode="modeA")

        t1.start.assert_not_called()
        t2.start.assert_not_called()
        t1.stop.assert_not_called()
        t2.stop.assert_not_called()

    def test_read_failure_stops_started_tasks(self, adv):
        """1.2 Read raising mid-acquisition → already-started tasks stopped."""
        call_order = []
        t1, t2 = self._make_ordered_tasks(adv, call_order)
        t1.read.side_effect = RuntimeError("read failed")

        with pytest.raises(RuntimeError, match="read failed"):
            adv.acquire_with_hardware_trigger()

        t1.stop.assert_called_once()
        t2.stop.assert_called_once()

    def test_stop_failure_during_cleanup_warns(self, adv):
        """1.2 task.stop() failure during cleanup warns, does not mask data."""
        import warnings as warnings_mod

        t1 = _make_nidaqmx_task("Task1", channel_names=["ch0"])
        t1.read.return_value = [1.0, 2.0]
        t1.stop.side_effect = RuntimeError("stop failed")
        adv.input_tasks = [t1]

        with warnings_mod.catch_warnings(record=True) as w:
            warnings_mod.simplefilter("always")
            result = adv.acquire_with_hardware_trigger()

        assert "Task1" in result
        assert any("stop failed" in str(x.message) for x in w)


class TestAcquireCustomModePassThrough:
    """Group 1 — acquire(custom_mode=...) pass-through and software-path warning."""

    def test_acquire_passes_custom_mode_to_hardware_path(self, adv):
        """1.4 acquire(custom_mode=...) forwards to acquire_with_hardware_trigger."""
        adv.trigger_type = "hardware"
        adv.input_tasks = [_make_nidaqmx_task()]
        with patch.object(
            adv, "acquire_with_hardware_trigger", return_value={}
        ) as mock_hw:
            adv.acquire(custom_mode="modeA")
            mock_hw.assert_called_once_with(custom_mode="modeA")

    def test_acquire_software_path_warns_and_ignores_custom_mode(self, adv):
        """1.4 Software path warns when custom_mode provided and ignores it."""
        import warnings as warnings_mod

        adv.trigger_type = "software"
        adv.input_tasks = [_make_nidaqmx_task()]
        with patch.object(
            adv, "acquire_with_software_trigger", return_value=np.array([])
        ) as mock_sw:
            with warnings_mod.catch_warnings(record=True) as w:
                warnings_mod.simplefilter("always")
                adv.acquire(custom_mode="modeA")

            mock_sw.assert_called_once()
            assert any("custom_mode" in str(x.message) for x in w)

    def test_acquire_software_path_no_warning_without_custom_mode(self, adv):
        """1.4 Software path emits no custom_mode warning when omitted."""
        import warnings as warnings_mod

        adv.trigger_type = "software"
        adv.input_tasks = [_make_nidaqmx_task()]
        with patch.object(
            adv, "acquire_with_software_trigger", return_value=np.array([])
        ):
            with warnings_mod.catch_warnings(record=True) as w:
                warnings_mod.simplefilter("always")
                adv.acquire()

            assert not any("custom_mode" in str(x.message) for x in w)


# ===================================================================
# Change multi-handler-advanced-acquisition
# Group 2: Non-blocking acquire (Future)
# ===================================================================


class TestMultiHandlerNonBlocking:
    """Group 2 — acquire(blocking=False) returns a Future."""

    def _setup_hardware(self, adv):
        """Configure a hardware-trigger handler with one deterministic task."""
        task = _make_nidaqmx_task("HWTask", channel_names=["ch0", "ch1"])
        task.read.return_value = [[1.0, 2.0], [3.0, 4.0]]
        adv.input_tasks = [task]
        adv.trigger_type = "hardware"
        return task

    def test_blocking_true_returns_data_directly(self, adv):
        """2.1 acquire(blocking=True) returns the data dict directly."""
        self._setup_hardware(adv)
        result = adv.acquire(blocking=True)
        assert isinstance(result, dict)
        assert "HWTask" in result

    def test_blocking_false_returns_future(self, adv):
        """2.1 acquire(blocking=False) returns a Future immediately."""
        from concurrent.futures import Future

        self._setup_hardware(adv)
        future = adv.acquire(blocking=False)
        assert isinstance(future, Future)
        future.result(timeout=5)

    def test_future_result_matches_blocking_result(self, adv):
        """2.1 future.result() yields the same structure as blocking acquire."""
        self._setup_hardware(adv)

        blocking_result = adv.acquire(blocking=True)
        future = adv.acquire(blocking=False)
        nb_result = future.result(timeout=5)

        assert set(nb_result.keys()) == set(blocking_result.keys())
        for task_name in blocking_result:
            assert set(nb_result[task_name]) == set(blocking_result[task_name])
            for ch in blocking_result[task_name]:
                np.testing.assert_array_equal(
                    nb_result[task_name][ch], blocking_result[task_name][ch]
                )

    def test_executor_lazily_created(self, adv):
        """2.1 Executor is None at init and created on first non-blocking acquire."""
        assert adv._executor is None

        self._setup_hardware(adv)
        adv.acquire(blocking=True)
        assert adv._executor is None  # blocking acquire must not create it

        future = adv.acquire(blocking=False)
        future.result(timeout=5)
        assert adv._executor is not None

    def test_executor_reused_across_calls(self, adv):
        """2.1 The same single-worker executor is reused for later calls."""
        self._setup_hardware(adv)
        f1 = adv.acquire(blocking=False)
        f1.result(timeout=5)
        executor = adv._executor
        f2 = adv.acquire(blocking=False)
        f2.result(timeout=5)
        assert adv._executor is executor

    def test_del_shuts_down_executor(self, adv):
        """2.1 __del__ shuts down a created executor."""
        mock_executor = MagicMock()
        adv._executor = mock_executor
        adv.__del__()
        mock_executor.shutdown.assert_called_once_with(wait=False)

    def test_del_without_executor_does_not_raise(self, adv):
        """2.1 __del__ is safe when the executor was never created."""
        assert adv._executor is None
        adv.__del__()  # must not raise

    def test_nonblocking_acquires_lock_in_worker(self, adv):
        """2.2 The RLock is acquired inside the submitted callable."""
        self._setup_hardware(adv)

        lock = MagicMock()
        lock.__enter__ = MagicMock(return_value=None)
        lock.__exit__ = MagicMock(return_value=False)
        adv._lock = lock

        future = adv.acquire(blocking=False)
        # The Future is returned before/independently of lock acquisition;
        # the lock must be taken by the worker thread.
        future.result(timeout=5)
        lock.__enter__.assert_called()


# ===================================================================
# Change multi-handler-advanced-acquisition
# Group 3: State/health introspection
# ===================================================================


class TestMultiHandlerCheckState:
    """Group 3 — check_state() with auto-reconnect."""

    def test_connect_sets_connect_called_flag(self, adv):
        """3.1 connect() sets the _connect_called flag."""
        assert adv._connect_called is False
        with patch.object(adv, "_define_required_devices"):
            with patch.object(adv, "ping", return_value=True):
                adv.connect()
        assert adv._connect_called is True

    def test_connect_called_flag_set_even_when_connect_fails(self, adv):
        """3.1 _connect_called is set even when connect() fails."""
        with patch.object(adv, "_define_required_devices"):
            with patch.object(adv, "ping", return_value=False):
                adv.connect()
        assert adv._connect_called is True

    def test_check_state_disconnected_when_never_connected(self, adv):
        """3.1 check_state() returns 'disconnected' before connect()."""
        assert adv.check_state() == "disconnected"

    def test_check_state_connected_when_ping_passes(self, adv):
        """3.1 check_state() returns 'connected' when connected and ping OK."""
        adv._connected = True
        adv._connect_called = True
        with patch.object(adv, "ping", return_value=True):
            assert adv.check_state() == "connected"

    def test_check_state_reconnected_after_lost_connection(self, adv):
        """3.1 check_state() retries connect() and returns 'reconnected'."""
        adv._connected = False
        adv._connect_called = True
        with patch.object(adv, "connect", return_value=True) as mock_connect:
            assert adv.check_state() == "reconnected"
            mock_connect.assert_called_once()

    def test_check_state_connection_lost_when_reconnect_fails(self, adv):
        """3.1 check_state() returns 'connection lost' when reconnect fails."""
        adv._connected = False
        adv._connect_called = True
        with patch.object(adv, "connect", return_value=False):
            assert adv.check_state() == "connection lost"

    def test_check_state_reconnects_when_connected_but_ping_fails(self, adv):
        """3.1 Connected handler with failing ping attempts reconnect."""
        adv._connected = True
        adv._connect_called = True
        with patch.object(adv, "ping", return_value=False):
            with patch.object(adv, "connect", return_value=True):
                assert adv.check_state() == "reconnected"

    def test_check_state_connection_lost_when_connected_ping_and_reconnect_fail(self, adv):
        """3.1 Connected handler: ping fails and reconnect fails → 'connection lost'."""
        adv._connected = True
        adv._connect_called = True
        with patch.object(adv, "ping", return_value=False):
            with patch.object(adv, "connect", return_value=False):
                assert adv.check_state() == "connection lost"


class TestMultiHandlerIsRunning:
    """Group 3 — is_running() across input and output tasks."""

    def test_no_tasks_returns_false(self, adv):
        """3.2 No tasks → False."""
        assert adv.is_running() is False

    def test_running_input_task_returns_true(self, adv):
        """3.2 An input task with is_task_done()==False → True."""
        task = _make_nidaqmx_task("In1")
        task.is_task_done.return_value = False
        adv.input_tasks = [task]
        assert adv.is_running() is True

    def test_running_output_task_returns_true(self, adv):
        """3.2 An output task with is_task_done()==False → True."""
        task = _make_nidaqmx_task("Out1")
        task.is_task_done.return_value = True
        out_task = _make_nidaqmx_task("Out2")
        out_task.is_task_done.return_value = False
        adv.input_tasks = [task]
        adv.output_tasks = [out_task]
        assert adv.is_running() is True

    def test_all_tasks_done_returns_false(self, adv):
        """3.2 All tasks done → False."""
        t1 = _make_nidaqmx_task("In1")
        t1.is_task_done.return_value = True
        t2 = _make_nidaqmx_task("Out1")
        t2.is_task_done.return_value = True
        adv.input_tasks = [t1]
        adv.output_tasks = [t2]
        assert adv.is_running() is False

    def test_daqerror_warns_and_treats_task_as_not_running(self, adv, advanced_module):
        """3.2 DaqError during query → warning, task treated as not running."""
        import warnings as warnings_mod

        task = _make_nidaqmx_task("Broken")
        error = advanced_module.DaqError("task closed")
        error.error_code = -200088
        task.is_task_done.side_effect = error
        adv.input_tasks = [task]

        with warnings_mod.catch_warnings(record=True) as w:
            warnings_mod.simplefilter("always")
            result = adv.is_running()

        assert result is False
        assert any("task closed" in str(x.message) for x in w)

    def test_daqerror_on_one_task_other_running_returns_true(self, adv, advanced_module):
        """3.2 DaqError on one task does not hide another running task."""
        import warnings as warnings_mod

        broken = _make_nidaqmx_task("Broken")
        error = advanced_module.DaqError("invalid task")
        error.error_code = -200088
        broken.is_task_done.side_effect = error

        running = _make_nidaqmx_task("Running")
        running.is_task_done.return_value = False

        adv.input_tasks = [broken, running]

        with warnings_mod.catch_warnings(record=True):
            warnings_mod.simplefilter("always")
            assert adv.is_running() is True


class TestMultiHandlerGetDeviceInfo:
    """Group 3 — get_device_info() nested per-task metadata."""

    def test_no_tasks_returns_empty_dict(self, adv):
        """3.2 No tasks → empty dict."""
        assert adv.get_device_info() == {}

    def test_input_tasks_only(self, adv):
        """3.2 Input tasks only → 'input' key only, nested per task."""
        task = _make_nidaqmx_task(
            "InTask", channel_names=["ch0", "ch1"], samp_clk_rate=25600.0
        )
        adv.input_tasks = [task]

        info = adv.get_device_info()
        assert set(info.keys()) == {"input"}
        assert info["input"]["InTask"]["channel_names"] == ["ch0", "ch1"]
        assert info["input"]["InTask"]["sample_rate"] == 25600.0
        assert isinstance(info["input"]["InTask"]["sample_rate"], float)

    def test_output_tasks_only(self, adv):
        """3.2 Output tasks only → 'output' key only."""
        task = _make_nidaqmx_task(
            "OutTask", channel_names=["ao0"], samp_clk_rate=10000.0
        )
        adv.output_tasks = [task]

        info = adv.get_device_info()
        assert set(info.keys()) == {"output"}
        assert info["output"]["OutTask"]["channel_names"] == ["ao0"]
        assert info["output"]["OutTask"]["sample_rate"] == 10000.0

    def test_input_and_output_tasks(self, adv):
        """3.2 Both task lists populated → both keys, one entry per task."""
        in1 = _make_nidaqmx_task("In1", channel_names=["a0"])
        in2 = _make_nidaqmx_task("In2", channel_names=["a1", "a2"])
        out1 = _make_nidaqmx_task("Out1", channel_names=["ao0"])
        adv.input_tasks = [in1, in2]
        adv.output_tasks = [out1]

        info = adv.get_device_info()
        assert set(info.keys()) == {"input", "output"}
        assert set(info["input"].keys()) == {"In1", "In2"}
        assert set(info["output"].keys()) == {"Out1"}
        assert info["input"]["In2"]["channel_names"] == ["a1", "a2"]


# ===================================================================
# Change multi-handler-advanced-acquisition
# Group 4: Acquisition abort (stop_acquisition)
# ===================================================================


class TestMultiHandlerStopAcquisition:
    """Group 4 — stop_acquisition() cooperative abort."""

    def test_acquire_running_defaults_false(self, adv):
        """4.1 _acquire_running defaults to False."""
        assert adv._acquire_running is False

    def test_noop_when_idle(self, adv):
        """4.1 stop_acquisition() with no acquisition in flight is a no-op."""
        adv.stop_acquisition()  # must not raise
        assert adv._acquire_running is False

    def test_stop_acquisition_does_not_take_lock(self, adv):
        """4.1 stop_acquisition() must not block on the RLock (deadlock avoidance)."""
        locked = threading.Event()
        release = threading.Event()

        def hold_lock():
            with adv._lock:
                locked.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        assert locked.wait(timeout=5)

        adv._acquire_running = True
        stopper = threading.Thread(target=adv.stop_acquisition)
        stopper.start()
        stopper.join(timeout=1.0)
        still_blocked = stopper.is_alive()
        release.set()
        holder.join(timeout=5)
        stopper.join(timeout=5)

        assert not still_blocked, "stop_acquisition() blocked on the RLock"
        assert adv._acquire_running is False

    def test_abort_software_polling_loop(self, adv):
        """4.1 Abort: loop exits, task stopped, ring-buffer contents returned."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0"])
        read_started = threading.Event()

        def read_side_effect(**kw):
            read_started.set()
            return [0.0] * 10

        task.read.side_effect = read_side_effect
        adv.input_tasks = [task]
        adv.input_channels = ["ch0"]
        adv.input_sample_rate = 1000.0
        adv.trigger_type = "software"

        trigger = MagicMock()
        trigger.ringbuff = MagicMock()
        trigger.rows = 100
        trigger.rows_left = 100
        trigger.finished = False  # never fires
        ring = np.arange(10, dtype=float).reshape(10, 1)
        trigger.get_data.return_value = ring
        adv.trigger = trigger
        adv._trigger_is_set = True

        future = adv.acquire(blocking=False)
        assert read_started.wait(timeout=5), "acquisition loop never started"
        adv.stop_acquisition()
        result = future.result(timeout=5)

        task.stop.assert_called_once()
        assert isinstance(result, dict)
        np.testing.assert_array_equal(result["ch0"], ring[:, 0])
        assert "time" in result

    def test_flag_reset_true_at_start_of_each_acquire(self, adv):
        """4.1 A stale False flag (from a prior abort) is reset on the next acquire."""
        task = _make_nidaqmx_task("SWTask", channel_names=["ch0"])
        task.read.return_value = [0.0] * 10
        adv.input_tasks = [task]
        adv.input_channels = ["ch0"]
        adv.input_sample_rate = 1000.0

        trigger = MagicMock()
        trigger.ringbuff = MagicMock()
        trigger.rows = 100
        trigger.rows_left = 100
        trigger.finished = False
        trigger.add_data.side_effect = lambda data: setattr(trigger, "finished", True)
        trigger.get_data.return_value = np.zeros((100, 1))
        adv.trigger = trigger
        adv._trigger_is_set = True

        adv._acquire_running = False  # stale flag from a previous abort
        result = adv.acquire_with_software_trigger(return_dict=False)

        assert trigger.add_data.call_count >= 1, "loop never ran — flag not reset"
        assert isinstance(result, np.ndarray)

    def test_abort_does_not_affect_hardware_path(self, adv):
        """4.1 Hardware-trigger acquisition ignores the abort flag (blocking reads)."""
        task = _make_nidaqmx_task("HWTask", channel_names=["ch0"])
        task.read.return_value = [1.0, 2.0]
        adv.input_tasks = [task]
        adv.trigger_type = "hardware"

        adv._acquire_running = False  # would abort the software loop
        result = adv.acquire()
        assert "HWTask" in result
        task.read.assert_called_once()
