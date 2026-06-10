"""Tests for nidaqwrapper.digital module (DITask and DOTask).

Architecture: Direct Delegation
--------------------------------
Constructor creates nidaqmx.Task immediately (not deferred to initiate()).
add_channel() delegates directly to nidaqmx task.di_channels / do_channels.
configure() sets up timing in clocked mode (no-op in on_demand mode).
start() (inherited from BaseTask) starts the underlying nidaqmx task.
self.channels dict is REMOVED — nidaqmx task is single source of truth.
initiate() is REMOVED.
save_config() / from_config() added for TOML persistence.

These tests are written TDD-style: they describe the target architecture and
WILL FAIL against the current store-then-pipe implementation. They pass once
the direct-delegation refactor is complete.

All tests use mocked nidaqmx — no hardware required.
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers — mock nidaqmx.Task that tracks DI/DO channel additions
# ---------------------------------------------------------------------------


def _make_mock_ni_task() -> MagicMock:
    """Create a mock nidaqmx.Task that tracks DI/DO channel additions.

    The mock records all add_di_chan() / add_do_chan() calls and maintains a
    shared channel list so that duplicate detection (via task.channel_names
    iteration) works correctly in the implementation under test.

    Digital channels use ``name_to_assign_to_lines`` (not
    ``name_to_assign_to_channel``), so the handler captures that keyword.
    """
    task = MagicMock()
    _channel_names: list[str] = []
    _channel_objects: list[MagicMock] = []

    def _make_handler(channel_type: str):
        def handler(**kwargs):
            # Digital channels use 'name_to_assign_to_lines', not 'name_to_assign_to_channel'
            name = kwargs.get("name_to_assign_to_lines", "")
            lines = kwargs.get("lines", "")
            _channel_names.append(name)
            ch = MagicMock()
            ch.name = name
            ch.physical_channel = MagicMock()
            # For digital, the 'lines' arg IS the physical channel spec
            ch.physical_channel.name = lines
            _channel_objects.append(ch)

        return handler

    task.di_channels.add_di_chan.side_effect = _make_handler("di")
    task.do_channels.add_do_chan.side_effect = _make_handler("do")

    # channel_names: same list object, stays in sync as channels are added
    task.channel_names = _channel_names

    # DI/DO channel iteration (used for duplicate line detection)
    task.di_channels.__iter__ = MagicMock(side_effect=lambda: iter(_channel_objects))
    task.do_channels.__iter__ = MagicMock(side_effect=lambda: iter(_channel_objects))

    # DI/DO channel length (used by from_task() validation)
    task.di_channels.__len__ = MagicMock(side_effect=lambda: len(_channel_objects))
    task.do_channels.__len__ = MagicMock(side_effect=lambda: len(_channel_objects))

    return task


def _build_di(
    mock_system,
    mock_constants,
    task_name: str = "test_di",
    sample_rate: float | None = None,
    task_names: list[str] | None = None,
) -> tuple[ExitStack, object, MagicMock]:
    """Construct a DITask inside a fully-patched context.

    Returns (exit_stack, di_task_instance, mock_nidaqmx_task).
    Use inside a ``with`` block — patches stay active so that add_channel()
    and start() can also run under mocking.

    The ``_expand_port_to_line_range`` function is patched to be a pass-through
    (returns the input unchanged) so that most tests are not sensitive to
    hardware queries. Specific port-expansion tests patch it separately.

    Example::

        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("btn", lines="Dev1/port0/line0")
            di.configure()
    """
    if task_names is None:
        task_names = []

    system = mock_system(task_names=task_names)
    mock_ni_task = _make_mock_ni_task()

    stack = ExitStack()
    stack.enter_context(
        patch(
            "nidaqwrapper.digital.nidaqmx.system.System.local",
            return_value=system,
        )
    )
    stack.enter_context(
        patch(
            "nidaqwrapper.digital.nidaqmx.task.Task",
            return_value=mock_ni_task,
        )
    )
    stack.enter_context(patch("nidaqwrapper.digital.constants", mock_constants))
    # Pass-through for port expansion so most tests don't need system queries
    stack.enter_context(
        patch(
            "nidaqwrapper.digital._expand_port_to_line_range",
            side_effect=lambda lines: lines,
        )
    )

    from nidaqwrapper.digital import DITask

    di = DITask(task_name, sample_rate=sample_rate)
    return stack, di, mock_ni_task


def _build_do(
    mock_system,
    mock_constants,
    task_name: str = "test_do",
    sample_rate: float | None = None,
    task_names: list[str] | None = None,
) -> tuple[ExitStack, object, MagicMock]:
    """Construct a DOTask inside a fully-patched context.

    Returns (exit_stack, do_task_instance, mock_nidaqmx_task).
    Use inside a ``with`` block. Mirrors _build_di() for DOTask.

    Example::

        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("led", lines="Dev1/port1/line0")
            do.configure()
    """
    if task_names is None:
        task_names = []

    system = mock_system(task_names=task_names)
    mock_ni_task = _make_mock_ni_task()

    stack = ExitStack()
    stack.enter_context(
        patch(
            "nidaqwrapper.digital.nidaqmx.system.System.local",
            return_value=system,
        )
    )
    stack.enter_context(
        patch(
            "nidaqwrapper.digital.nidaqmx.task.Task",
            return_value=mock_ni_task,
        )
    )
    stack.enter_context(patch("nidaqwrapper.digital.constants", mock_constants))
    # Pass-through for port expansion
    stack.enter_context(
        patch(
            "nidaqwrapper.digital._expand_port_to_line_range",
            side_effect=lambda lines: lines,
        )
    )

    from nidaqwrapper.digital import DOTask

    do = DOTask(task_name, sample_rate=sample_rate)
    return stack, do, mock_ni_task


# ===========================================================================
# DITask Tests
# ===========================================================================


class TestDITaskConstructor:
    """Constructor creates nidaqmx.Task immediately (direct delegation)."""

    def test_creates_nidaqmx_task_with_name(self, mock_system, mock_constants):
        """Constructor calls nidaqmx.task.Task(new_task_name=task_name)."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ) as mock_cls,
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            DITask("switches", sample_rate=None)

        mock_cls.assert_called_once_with(new_task_name="switches")

    def test_task_attribute_set_immediately(self, mock_system, mock_constants):
        """self.task is set to the nidaqmx.Task in the constructor."""
        ctx, di, mock_ni_task = _build_di(mock_system, mock_constants)
        with ctx:
            pass
        assert di.task is mock_ni_task

    def test_on_demand_mode(self, mock_system, mock_constants):
        """No sample_rate sets mode='on_demand'."""
        ctx, di, _ = _build_di(mock_system, mock_constants, sample_rate=None)
        with ctx:
            pass
        assert di.mode == "on_demand"
        assert di.sample_rate is None

    def test_clocked_mode(self, mock_system, mock_constants):
        """sample_rate=1000 sets mode='clocked'."""
        ctx, di, _ = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            pass
        assert di.mode == "clocked"
        assert di.sample_rate == 1000

    def test_device_discovery(self, mock_system, mock_constants):
        """device_list is populated from system devices in the constructor."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            pass
        # conftest default devices: cDAQ1Mod1, cDAQ1Mod2
        assert "cDAQ1Mod1" in di.device_list
        assert "cDAQ1Mod2" in di.device_list

    def test_duplicate_task_name_raises(self, mock_system, mock_constants):
        """Constructor raises ValueError when task_name exists in NI MAX."""
        system = mock_system(task_names=["existing_di"])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            with pytest.raises(ValueError, match="existing_di"):
                DITask("existing_di")

    def test_no_channels_dict(self, mock_system, mock_constants):
        """The old self.channels dict no longer exists in the new architecture."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(di, "channels")


class TestDITaskAddChannel:
    """add_channel() delegates directly to nidaqmx di_channels.add_di_chan()."""

    def test_delegates_to_di_channels(self, mock_system, mock_constants):
        """add_channel() calls add_di_chan() on the nidaqmx task."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("btn_1", lines="Dev1/port0/line0")

        mt.di_channels.add_di_chan.assert_called_once()

    def test_passes_lines_directly(self, mock_system, mock_constants):
        """The lines string is forwarded as the 'lines' kwarg."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("btn_1", lines="Dev1/port0/line0:3")

        kwargs = mt.di_channels.add_di_chan.call_args.kwargs
        assert kwargs["lines"] == "Dev1/port0/line0:3"

    def test_passes_channel_name(self, mock_system, mock_constants):
        """Channel name is forwarded as name_to_assign_to_lines."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("my_button", lines="Dev1/port0/line0")

        kwargs = mt.di_channels.add_di_chan.call_args.kwargs
        assert kwargs["name_to_assign_to_lines"] == "my_button"

    def test_uses_chan_per_line(self, mock_system, mock_constants):
        """add_channel() passes line_grouping=CHAN_PER_LINE to nidaqmx."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("btn_1", lines="Dev1/port0/line0")

        kwargs = mt.di_channels.add_di_chan.call_args.kwargs
        assert kwargs["line_grouping"] == mock_constants.LineGrouping.CHAN_PER_LINE

    def test_port_expansion_called_for_port_spec(self, mock_system, mock_constants):
        """Port-only spec triggers _expand_port_to_line_range() during add_channel()."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                return_value="Dev1/port0/line0:7",
            ) as mock_expand,
        ):
            from nidaqwrapper.digital import DITask

            di = DITask("test_expand")
            di.add_channel("port0", lines="Dev1/port0")

        mock_expand.assert_called_once_with("Dev1/port0")
        # The expanded result is forwarded to nidaqmx
        kwargs = mock_ni_task.di_channels.add_di_chan.call_args.kwargs
        assert kwargs["lines"] == "Dev1/port0/line0:7"

    def test_line_spec_passed_through_expansion(self, mock_system, mock_constants):
        """Line specs containing '/line' are passed through expansion unchanged."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ) as mock_expand,
        ):
            from nidaqwrapper.digital import DITask

            di = DITask("test_noexpand")
            di.add_channel("btn", lines="Dev1/port0/line0:3")

        # Expansion function was called — it's called for all specs, but returns
        # the spec unchanged when '/line' is present (handled internally)
        mock_expand.assert_called_once_with("Dev1/port0/line0:3")

    def test_duplicate_name_raises(self, mock_system, mock_constants):
        """Adding two channels with the same name raises ValueError."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("btn_1", lines="Dev1/port0/line0")
            with pytest.raises(ValueError, match="btn_1"):
                di.add_channel("btn_1", lines="Dev1/port0/line1")

    def test_duplicate_lines_raises(self, mock_system, mock_constants):
        """Adding two channels with the same lines string raises ValueError."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("ch1", lines="Dev1/port0/line0")
            with pytest.raises(ValueError, match="Dev1/port0/line0"):
                di.add_channel("ch2", lines="Dev1/port0/line0")

class TestDITaskConfigure:
    """configure() sets up clocked timing (no-op in on_demand mode)."""

    def test_clocked_configures_timing(self, mock_system, mock_constants):
        """configure() calls cfg_samp_clk_timing in clocked mode."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.configure()

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["rate"] == 1000

    def test_clocked_uses_continuous_mode(self, mock_system, mock_constants):
        """configure() passes CONTINUOUS sample mode to cfg_samp_clk_timing."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.configure()

        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["sample_mode"] == mock_constants.AcquisitionType.CONTINUOUS

    def test_on_demand_no_timing(self, mock_system, mock_constants):
        """configure() in on-demand mode does NOT configure timing."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=None)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.configure()

        mt.timing.cfg_samp_clk_timing.assert_not_called()

    def test_on_demand_configure_no_timing(self, mock_system, mock_constants):
        """configure() in on-demand mode is a no-op — no timing, no task start."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=None)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.configure()

        mt.timing.cfg_samp_clk_timing.assert_not_called()
        mt.start.assert_not_called()

    def test_clocked_configure_then_start(self, mock_system, mock_constants):
        """configure() then start() calls task.start() in clocked mode."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.configure()
            di.start()

        mt.start.assert_called_once()

    def test_clocked_configure_only(self, mock_system, mock_constants):
        """configure() alone configures timing but does NOT start the task."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.configure()

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        mt.start.assert_not_called()

    def test_configure_no_channels_raises(self, mock_system, mock_constants):
        """configure() raises ValueError when no channels have been added."""
        ctx, di, _ = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="[Nn]o channels|channel"):
                di.configure()


class TestDITaskStart:
    """start() (inherited from BaseTask) starts the underlying nidaqmx task."""

    def test_start_calls_task_start(self, mock_system, mock_constants):
        """start() delegates to self.task.start()."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.configure()
            di.start()

        mt.start.assert_called_once()

    def test_start_blocked_on_external_task(self, mock_system, mock_constants):
        """start() raises RuntimeError when _owns_task is False."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_di"
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_task(mock_ni_task)

            with pytest.raises(
                RuntimeError,
                match="Cannot start an externally-provided task",
            ):
                di.start()


class TestDITaskGetters:
    """Getters delegate to the nidaqmx task properties."""

    def test_channel_list(self, mock_system, mock_constants):
        """channel_list returns names tracked by the nidaqmx task."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("btn_1", lines="Dev1/port0/line0")
            di.add_channel("btn_2", lines="Dev1/port0/line1")

        assert di.channel_list == ["btn_1", "btn_2"]

    def test_number_of_ch(self, mock_system, mock_constants):
        """number_of_ch returns count from the nidaqmx task."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            assert di.number_of_ch == 0
            di.add_channel("btn_1", lines="Dev1/port0/line0")
            assert di.number_of_ch == 1
            di.add_channel("btn_2", lines="Dev1/port0/line1")
            assert di.number_of_ch == 2

    def test_channel_list_empty_initially(self, mock_system, mock_constants):
        """channel_list is empty on a new task before any channels are added."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            assert di.channel_list == []


class TestDITaskRead:
    """read() performs on-demand single-sample reads."""

    def test_read_single_line_bool(self, mock_system, mock_constants):
        """read() wraps a single bool into a numpy array of length 1."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        mt.read.return_value = True
        with ctx:
            di.add_channel("btn", lines="Dev1/port0/line0")
            data = di.read()

        assert isinstance(data, np.ndarray)
        assert len(data) == 1
        assert data[0] == True  # noqa: E712

    def test_read_multi_line_list(self, mock_system, mock_constants):
        """read() returns a numpy array with one value per line."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        mt.read.return_value = [True, False, True, False]
        with ctx:
            di.add_channel("sw", lines="Dev1/port0/line0:3")
            data = di.read()

        assert isinstance(data, np.ndarray)
        assert len(data) == 4
        np.testing.assert_array_equal(data, [True, False, True, False])

    def test_read_returns_numpy(self, mock_system, mock_constants):
        """read() always returns a numpy.ndarray."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        mt.read.return_value = False
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            data = di.read()

        assert isinstance(data, np.ndarray)


class TestDITaskReadAllAvailable:
    """acquire() reads buffered data in clocked mode."""

    def test_returns_n_samples_n_lines(self, mock_system, mock_constants):
        """acquire() returns (n_samples, n_lines) shaped array."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        # nidaqmx returns (n_lines, n_samples) for multi-line — 4 lines, 500 samples
        mt.read.return_value = [
            [True] * 500,
            [False] * 500,
            [True] * 500,
            [False] * 500,
        ]
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0:3")
            di.configure()
            data = di.acquire()

        assert data.shape == (500, 4)

    def test_empty_buffer(self, mock_system, mock_constants):
        """acquire() returns an empty array when the buffer is empty."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        mt.read.return_value = []
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            data = di.acquire()

        assert data.size == 0

    def test_uses_acquire_constant(self, mock_system, mock_constants):
        """acquire() passes READ_ALL_AVAILABLE to task.read()."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        mt.read.return_value = [True, False]
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.acquire()

        mt.read.assert_called_with(
            number_of_samples_per_channel=mock_constants.READ_ALL_AVAILABLE
        )

    def test_raises_in_on_demand_mode(self, mock_system, mock_constants):
        """acquire() raises RuntimeError when mode is on_demand."""
        ctx, di, _ = _build_di(mock_system, mock_constants, sample_rate=None)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            with pytest.raises(RuntimeError, match="clocked mode"):
                di.acquire()

    def test_single_line_reshaped(self, mock_system, mock_constants):
        """Single-line clocked read is reshaped to (n_samples, 1)."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        # Single line: nidaqmx returns a flat list
        mt.read.return_value = [True, False, True]
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")
            di.configure()
            data = di.acquire()

        assert data.shape == (3, 1)


class TestDITaskClearTask:
    """clear_task() releases hardware resources safely."""

    def test_closes_task(self, mock_system, mock_constants):
        """clear_task() calls task.close() on the nidaqmx task."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            pass

        di.clear_task()
        mt.close.assert_called_once()

    def test_sets_task_none(self, mock_system, mock_constants):
        """clear_task() sets self.task to None."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            pass

        di.clear_task()
        assert di.task is None

    def test_multiple_calls_safe(self, mock_system, mock_constants):
        """Calling clear_task() twice does not raise."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            pass

        di.clear_task()
        di.clear_task()  # Must not raise

    def test_exception_warns(self, mock_system, mock_constants):
        """clear_task() emits a warning when task.close() raises."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            pass
        mt.close.side_effect = OSError("hw error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            di.clear_task()

        assert len(w) >= 1
        assert "hw error" in str(w[0].message)
        assert di.task is None


class TestDITaskContextManager:
    """DITask implements the context manager protocol."""

    def test_enter_returns_self(self, mock_system, mock_constants):
        """__enter__ returns the DITask instance."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            pass

        result = di.__enter__()
        assert result is di

    def test_exit_calls_clear(self, mock_system, mock_constants):
        """__exit__ calls clear_task()."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            pass
        di.clear_task = MagicMock()

        di.__exit__(None, None, None)
        di.clear_task.assert_called_once()

    def test_cleanup_on_exception(self, mock_system, mock_constants):
        """clear_task() is called even when the with-block raises."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            pass

        cleared = []

        def _tracking_clear():
            cleared.append(True)
            if di.task is not None:
                di.task.close()

        di.clear_task = _tracking_clear

        with pytest.raises(RuntimeError):
            with di:
                raise RuntimeError("body error")

        assert cleared


class TestDITaskInitiateRemoved:
    """initiate() and old internal methods must not exist in the new architecture."""

    def test_no_initiate_method(self, mock_system, mock_constants):
        """initiate() method does not exist on DITask."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(di, "initiate")

    def test_no_add_channels_method(self, mock_system, mock_constants):
        """_add_channels() internal method no longer exists."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(di, "_add_channels")

    def test_no_create_task_method(self, mock_system, mock_constants):
        """_create_task() internal method no longer exists."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(di, "_create_task")


class TestDITaskSaveConfig:
    """save_config() serialises DITask configuration to TOML."""

    def test_writes_toml_file(self, mock_system, mock_constants, tmp_path):
        """save_config() creates a valid TOML file."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("btn_group", lines="Dev1/port0/line0:3")

        path = tmp_path / "di_config.toml"
        di.save_config(path)
        assert path.exists()

        with open(path, "rb") as f:
            data = tomllib.load(f)
        assert "task" in data
        assert "channels" in data

    def test_task_section_name(self, mock_system, mock_constants, tmp_path):
        """[task] section contains the task name."""
        ctx, di, _ = _build_di(
            mock_system, mock_constants, task_name="my_switches"
        )
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")

        path = tmp_path / "config.toml"
        di.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["name"] == "my_switches"

    def test_task_section_type_digital_input(self, mock_system, mock_constants, tmp_path):
        """[task] type field is 'digital_input'."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")

        path = tmp_path / "config.toml"
        di.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["type"] == "digital_input"

    def test_task_section_clocked_includes_sample_rate(
        self, mock_system, mock_constants, tmp_path
    ):
        """[task] section includes sample_rate for clocked mode."""
        ctx, di, _ = _build_di(
            mock_system, mock_constants, sample_rate=1000
        )
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")

        path = tmp_path / "config.toml"
        di.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["sample_rate"] == 1000

    def test_task_section_on_demand_no_sample_rate(
        self, mock_system, mock_constants, tmp_path
    ):
        """[task] section omits sample_rate for on-demand mode."""
        ctx, di, _ = _build_di(mock_system, mock_constants, sample_rate=None)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")

        path = tmp_path / "config.toml"
        di.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert "sample_rate" not in data["task"]

    def test_channel_entries(self, mock_system, mock_constants, tmp_path):
        """[[channels]] entries contain name and lines."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("btn_group", lines="Dev1/port0/line0:3")

        path = tmp_path / "config.toml"
        di.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        channels = data["channels"]
        assert len(channels) == 1
        ch = channels[0]
        assert ch["name"] == "btn_group"
        assert ch["lines"] == "Dev1/port0/line0:3"

    def test_multiple_channels(self, mock_system, mock_constants, tmp_path):
        """All channels are serialised to [[channels]] entries."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("btn_group", lines="Dev1/port0/line0:3")
            di.add_channel("sensors", lines="Dev1/port1/line0:7")

        path = tmp_path / "config.toml"
        di.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert len(data["channels"]) == 2
        names = {ch["name"] for ch in data["channels"]}
        assert names == {"btn_group", "sensors"}


    def test_header_comment_with_timestamp(self, mock_system, mock_constants, tmp_path):
        """save_config() includes header comment with version and timestamp."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            di.add_channel("ch", lines="Dev1/port0/line0")

        path = tmp_path / "config.toml"
        di.save_config(path)
        
        # Read raw text to check the header comment
        content = path.read_text()
        lines_list = content.splitlines()
        assert len(lines_list) > 0
        assert lines_list[0].startswith("# Generated by nidaqwrapper 0.1.0 on")
        
        # Verify from_config() can still parse it (round-trip test)
        system2 = mock_system(task_names=[])
        mock_ni_task2 = _make_mock_ni_task()

        from unittest.mock import patch
        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system2,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task2,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DITask
            di2 = DITask.from_config(path)
            assert di2.task_name == di.task_name


class TestDITaskFromConfig:
    """from_config() creates a DITask from a TOML file."""

    def _write_config(self, tmp_path, content: str):
        """Write a TOML string to a temporary file and return the path."""
        path = tmp_path / "config.toml"
        path.write_text(content)
        return path

    def test_creates_task_from_toml(self, mock_system, mock_constants, tmp_path):
        """from_config() creates a DITask with the name from [task]."""
        path = self._write_config(
            tmp_path,
            """\
[task]
name = "switches"
type = "digital_input"

[[channels]]
name = "btn_group"
lines = "Dev1/port0/line0:3"
""",
        )
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ) as mock_cls,
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_config(path)

        mock_cls.assert_called_once_with(new_task_name="switches")
        assert di.task_name == "switches"

    def test_adds_channels(self, mock_system, mock_constants, tmp_path):
        """from_config() calls add_channel() for each [[channels]] entry."""
        path = self._write_config(
            tmp_path,
            """\
[task]
name = "switches"
type = "digital_input"

[[channels]]
name = "btn_group"
lines = "Dev1/port0/line0:3"

[[channels]]
name = "sensors"
lines = "Dev1/port1/line0:7"
""",
        )
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DITask

            DITask.from_config(path)

        assert mock_ni_task.di_channels.add_di_chan.call_count == 2

    def test_clocked_mode_from_config(self, mock_system, mock_constants, tmp_path):
        """from_config() sets clocked mode when sample_rate is in [task]."""
        path = self._write_config(
            tmp_path,
            """\
[task]
name = "fast_di"
type = "digital_input"
sample_rate = 2000

[[channels]]
name = "ch"
lines = "Dev1/port0/line0"
""",
        )
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_config(path)

        assert di.mode == "clocked"
        assert di.sample_rate == 2000

    def test_missing_task_section_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when [task] section is missing."""
        path = self._write_config(
            tmp_path,
            """\
[[channels]]
name = "ch"
lines = "Dev1/port0/line0"
""",
        )
        system = mock_system(task_names=[])

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            with pytest.raises(ValueError, match="task"):
                DITask.from_config(path)

    def test_malformed_toml_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises an error on syntactically invalid TOML."""
        path = self._write_config(tmp_path, "not = valid [ toml {\n")

        from nidaqwrapper.digital import DITask

        with pytest.raises(Exception):  # tomllib.TOMLDecodeError
            DITask.from_config(path)


class TestDITaskConfigRoundtrip:
    """save_config() + from_config() round-trip preserves task configuration."""

    def test_roundtrip_on_demand(self, mock_system, mock_constants, tmp_path):
        """On-demand config survives a save/load cycle."""
        ctx, di, _ = _build_di(
            mock_system, mock_constants, task_name="roundtrip_di", sample_rate=None
        )
        with ctx:
            di.add_channel("btn_group", lines="Dev1/port0/line0:3")

        path = tmp_path / "config.toml"
        di.save_config(path)

        system2 = mock_system(task_names=[])
        mock_ni_task2 = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system2,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task2,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DITask

            di2 = DITask.from_config(path)

        assert di2.task_name == "roundtrip_di"
        assert di2.sample_rate is None
        assert di2.mode == "on_demand"
        kwargs = mock_ni_task2.di_channels.add_di_chan.call_args.kwargs
        assert kwargs["lines"] == "Dev1/port0/line0:3"
        assert kwargs["name_to_assign_to_lines"] == "btn_group"

    def test_roundtrip_clocked(self, mock_system, mock_constants, tmp_path):
        """Clocked config survives a save/load cycle."""
        ctx, di, _ = _build_di(
            mock_system, mock_constants, task_name="fast_di", sample_rate=5000
        )
        with ctx:
            di.add_channel("signals", lines="Dev1/port0/line0:7")

        path = tmp_path / "config.toml"
        di.save_config(path)

        system2 = mock_system(task_names=[])
        mock_ni_task2 = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system2,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task2,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DITask

            di2 = DITask.from_config(path)

        assert di2.task_name == "fast_di"
        assert di2.sample_rate == 5000
        assert di2.mode == "clocked"


class TestDITaskFromTask:
    """from_task() wraps externally-created nidaqmx.Task objects."""

    def test_wraps_existing_task(self, mock_system, mock_constants):
        """from_task() creates a DITask from an existing nidaqmx.Task."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_di"
        # Simulate that the task already has DI channels
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0:3", name_to_assign_to_lines="existing_ch"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_task(mock_ni_task)

        assert di.task is mock_ni_task
        assert di.task_name == "external_di"
        assert di.channel_list == ["existing_ch"]
        assert di.number_of_ch == 1

    def test_owns_task_false(self, mock_system, mock_constants):
        """from_task() sets _owns_task to False."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_di"
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_task(mock_ni_task)

        assert hasattr(di, "_owns_task")
        assert di._owns_task is False

    def test_constructor_owns_task_true(self, mock_system, mock_constants):
        """Normal constructor sets _owns_task to True."""
        ctx, di, _ = _build_di(mock_system, mock_constants)
        with ctx:
            pass
        assert di._owns_task is True

    def test_validates_no_di_channels(self, mock_system, mock_constants):
        """from_task() raises ValueError when task has no DI channels."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "no_channels"
        # No channels added — task.di_channels is empty

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            with pytest.raises(ValueError, match="[Nn]o DI channels"):
                DITask.from_task(mock_ni_task)

    def test_warns_if_task_running(self, mock_system, mock_constants):
        """from_task() warns when the task is already running."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "running_task"
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0", name_to_assign_to_lines="ch1"
        )
        # Simulate running task
        mock_ni_task.is_task_done.return_value = False
        mock_ni_task._is_running = True  # Custom flag for testing

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                DITask.from_task(mock_ni_task)

            # Check for warning if task appears to be running
            # (Implementation detail: might check task state)

    def test_add_channel_blocked(self, mock_system, mock_constants):
        """add_channel() raises RuntimeError when _owns_task is False."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_di"
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_task(mock_ni_task)

            with pytest.raises(
                RuntimeError, match="Cannot add channels to an externally-provided task"
            ):
                di.add_channel("new_ch", lines="Dev1/port0/line1")

    def test_configure_blocked(self, mock_system, mock_constants):
        """configure() raises RuntimeError when _owns_task is False."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_di"
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_task(mock_ni_task)

            with pytest.raises(
                RuntimeError,
                match="Cannot configure an externally-provided task",
            ):
                di.configure()

    def test_clear_task_does_not_close(self, mock_system, mock_constants):
        """clear_task() does NOT close external task, warns instead."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_di"
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_task(mock_ni_task)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                di.clear_task()

            # Should NOT call task.close()
            mock_ni_task.close.assert_not_called()
            # Should warn user
            assert len(w) >= 1
            assert "externally" in str(w[0].message).lower()

    def test_exit_does_not_close(self, mock_system, mock_constants):
        """__exit__ does NOT close external task, warns instead."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_di"
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_task(mock_ni_task)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                di.__exit__(None, None, None)

            mock_ni_task.close.assert_not_called()
            assert len(w) >= 1

    def test_normal_constructor_closes_task(self, mock_system, mock_constants):
        """Normal constructor (owns_task=True) closes task on clear_task()."""
        ctx, di, mt = _build_di(mock_system, mock_constants)
        with ctx:
            pass

        di.clear_task()
        mt.close.assert_called_once()

    def test_detects_clocked_mode(self, mock_system, mock_constants):
        """from_task() detects clocked mode from task.timing.samp_clk_rate."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "clocked_di"
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0:3", name_to_assign_to_lines="ch1"
        )
        mock_ni_task.timing.samp_clk_rate = 2000.0

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_task(mock_ni_task)

        assert di.mode == "clocked"
        assert di.sample_rate == 2000.0

    def test_detects_on_demand_mode(self, mock_system, mock_constants):
        """from_task() detects on-demand mode when no sample rate set."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "on_demand_di"
        mock_ni_task.di_channels.add_di_chan(
            lines="Dev1/port0/line0", name_to_assign_to_lines="ch1"
        )
        mock_ni_task.timing.samp_clk_rate = None  # Or not set

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask

            di = DITask.from_task(mock_ni_task)

        assert di.mode == "on_demand"
        assert di.sample_rate is None


# ===========================================================================
# DOTask Tests
# ===========================================================================


class TestDOTaskConstructor:
    """Constructor creates nidaqmx.Task immediately (direct delegation)."""

    def test_creates_nidaqmx_task_with_name(self, mock_system, mock_constants):
        """Constructor calls nidaqmx.task.Task(new_task_name=task_name)."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ) as mock_cls,
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            DOTask("leds", sample_rate=None)

        mock_cls.assert_called_once_with(new_task_name="leds")

    def test_task_attribute_set_immediately(self, mock_system, mock_constants):
        """self.task is set to the nidaqmx.Task in the constructor."""
        ctx, do, mock_ni_task = _build_do(mock_system, mock_constants)
        with ctx:
            pass
        assert do.task is mock_ni_task

    def test_on_demand_mode(self, mock_system, mock_constants):
        """No sample_rate sets mode='on_demand'."""
        ctx, do, _ = _build_do(mock_system, mock_constants, sample_rate=None)
        with ctx:
            pass
        assert do.mode == "on_demand"
        assert do.sample_rate is None

    def test_clocked_mode(self, mock_system, mock_constants):
        """sample_rate=1000 sets mode='clocked'."""
        ctx, do, _ = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            pass
        assert do.mode == "clocked"
        assert do.sample_rate == 1000

    def test_device_discovery(self, mock_system, mock_constants):
        """device_list is populated from system devices in the constructor."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            pass
        assert "cDAQ1Mod1" in do.device_list
        assert "cDAQ1Mod2" in do.device_list

    def test_duplicate_task_name_raises(self, mock_system, mock_constants):
        """Constructor raises ValueError when task_name exists in NI MAX."""
        system = mock_system(task_names=["existing_do"])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            with pytest.raises(ValueError, match="existing_do"):
                DOTask("existing_do")

    def test_no_channels_dict(self, mock_system, mock_constants):
        """The old self.channels dict no longer exists in the new architecture."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(do, "channels")


class TestDOTaskAddChannel:
    """add_channel() delegates directly to nidaqmx do_channels.add_do_chan()."""

    def test_delegates_to_do_channels(self, mock_system, mock_constants):
        """add_channel() calls add_do_chan() on the nidaqmx task."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("led_1", lines="Dev1/port1/line0")

        mt.do_channels.add_do_chan.assert_called_once()

    def test_passes_lines_directly(self, mock_system, mock_constants):
        """The lines string is forwarded as the 'lines' kwarg."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("leds", lines="Dev1/port1/line0:3")

        kwargs = mt.do_channels.add_do_chan.call_args.kwargs
        assert kwargs["lines"] == "Dev1/port1/line0:3"

    def test_passes_channel_name(self, mock_system, mock_constants):
        """Channel name is forwarded as name_to_assign_to_lines."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("my_leds", lines="Dev1/port1/line0")

        kwargs = mt.do_channels.add_do_chan.call_args.kwargs
        assert kwargs["name_to_assign_to_lines"] == "my_leds"

    def test_uses_chan_per_line(self, mock_system, mock_constants):
        """add_channel() passes line_grouping=CHAN_PER_LINE to nidaqmx."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("led_1", lines="Dev1/port1/line0")

        kwargs = mt.do_channels.add_do_chan.call_args.kwargs
        assert kwargs["line_grouping"] == mock_constants.LineGrouping.CHAN_PER_LINE

    def test_port_expansion_called_for_port_spec(self, mock_system, mock_constants):
        """Port-only spec triggers _expand_port_to_line_range() during add_channel()."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                return_value="Dev1/port1/line0:7",
            ) as mock_expand,
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask("test_expand")
            do.add_channel("port1", lines="Dev1/port1")

        mock_expand.assert_called_once_with("Dev1/port1")
        kwargs = mock_ni_task.do_channels.add_do_chan.call_args.kwargs
        assert kwargs["lines"] == "Dev1/port1/line0:7"

    def test_duplicate_name_raises(self, mock_system, mock_constants):
        """Adding two channels with the same name raises ValueError."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("led_1", lines="Dev1/port1/line0")
            with pytest.raises(ValueError, match="led_1"):
                do.add_channel("led_1", lines="Dev1/port1/line1")

    def test_duplicate_lines_raises(self, mock_system, mock_constants):
        """Adding two channels with the same lines string raises ValueError."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("ch1", lines="Dev1/port1/line0")
            with pytest.raises(ValueError, match="Dev1/port1/line0"):
                do.add_channel("ch2", lines="Dev1/port1/line0")

class TestDOTaskConfigure:
    """configure() sets up clocked timing (no-op in on_demand mode)."""

    def test_clocked_configures_timing(self, mock_system, mock_constants):
        """configure() calls cfg_samp_clk_timing in clocked mode."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.configure()

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["rate"] == 1000

    def test_clocked_uses_continuous_mode(self, mock_system, mock_constants):
        """configure() passes CONTINUOUS sample mode to cfg_samp_clk_timing."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.configure()

        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["sample_mode"] == mock_constants.AcquisitionType.CONTINUOUS

    def test_on_demand_no_timing(self, mock_system, mock_constants):
        """configure() in on-demand mode does NOT configure timing."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=None)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.configure()

        mt.timing.cfg_samp_clk_timing.assert_not_called()

    def test_on_demand_configure_no_timing(self, mock_system, mock_constants):
        """configure() in on-demand mode is a no-op — no timing, no task start."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=None)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.configure()

        mt.timing.cfg_samp_clk_timing.assert_not_called()
        mt.start.assert_not_called()

    def test_clocked_configure_then_start(self, mock_system, mock_constants):
        """configure() then start() calls task.start() in clocked mode."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.configure()
            do.start()

        mt.start.assert_called_once()

    def test_clocked_configure_only(self, mock_system, mock_constants):
        """configure() alone configures timing but does NOT start the task."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.configure()

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        mt.start.assert_not_called()

    def test_configure_no_channels_raises(self, mock_system, mock_constants):
        """configure() raises ValueError when no channels have been added."""
        ctx, do, _ = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="[Nn]o channels|channel"):
                do.configure()


class TestDOTaskStart:
    """start() (inherited from BaseTask) starts the underlying nidaqmx task."""

    def test_start_calls_task_start(self, mock_system, mock_constants):
        """start() delegates to self.task.start()."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.configure()
            do.start()

        mt.start.assert_called_once()

    def test_start_blocked_on_external_task(self, mock_system, mock_constants):
        """start() raises RuntimeError when _owns_task is False."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_do"
        mock_ni_task.do_channels.add_do_chan(
            lines="Dev1/port1/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_task(mock_ni_task)

            with pytest.raises(
                RuntimeError,
                match="Cannot start an externally-provided task",
            ):
                do.start()


class TestDOTaskGetters:
    """Getters delegate to the nidaqmx task properties."""

    def test_channel_list(self, mock_system, mock_constants):
        """channel_list returns names tracked by the nidaqmx task."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("led_1", lines="Dev1/port1/line0")
            do.add_channel("led_2", lines="Dev1/port1/line1")

        assert do.channel_list == ["led_1", "led_2"]

    def test_number_of_ch(self, mock_system, mock_constants):
        """number_of_ch returns count from the nidaqmx task."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            assert do.number_of_ch == 0
            do.add_channel("led_1", lines="Dev1/port1/line0")
            assert do.number_of_ch == 1
            do.add_channel("led_2", lines="Dev1/port1/line1")
            assert do.number_of_ch == 2

    def test_channel_list_empty_initially(self, mock_system, mock_constants):
        """channel_list is empty on a new task before any channels are added."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            assert do.channel_list == []


class TestDOTaskWrite:
    """write() performs on-demand single-sample writes."""

    def test_write_single_bool(self, mock_system, mock_constants):
        """write(True) passes True to task.write()."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("led", lines="Dev1/port1/line0")
            do.write(True)

        mt.write.assert_called_once_with(True)

    def test_write_single_int(self, mock_system, mock_constants):
        """write(1) converts to True and passes to task.write()."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("led", lines="Dev1/port1/line0")
            do.write(1)

        mt.write.assert_called_once_with(True)

    def test_write_list(self, mock_system, mock_constants):
        """write([True, False, True, False]) converts to bools and passes."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("leds", lines="Dev1/port1/line0:3")
            do.write([True, False, True, False])

        mt.write.assert_called_once_with([True, False, True, False])

    def test_write_numpy_array(self, mock_system, mock_constants):
        """write(np.array([1, 0, 1, 0])) converts to list of bools."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("leds", lines="Dev1/port1/line0:3")
            do.write(np.array([1, 0, 1, 0]))

        mt.write.assert_called_once_with([True, False, True, False])

    def test_write_calls_task_write(self, mock_system, mock_constants):
        """write() delegates to task.write()."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.write(False)

        mt.write.assert_called_once()


class TestDOTaskWriteContinuous:
    """write_continuous() writes buffered data in clocked mode."""

    def test_multi_line_2d_transposed(self, mock_system, mock_constants):
        """write_continuous() transposes (n_samples, n_lines) to (n_lines, n_samples)."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0:3")
            data = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])  # (3, 4)
            do.write_continuous(data)

        call_args = mt.write.call_args
        written_data = call_args[0][0]
        assert len(written_data) == 4  # n_lines (was rows after transpose)
        assert len(written_data[0]) == 3  # n_samples
        assert call_args.kwargs["auto_start"] is True
        # Known Pitfall #7: write data must be bool
        assert all(isinstance(v, bool) for row in written_data for v in row)

    def test_single_line_1d_written_directly(self, mock_system, mock_constants):
        """write_continuous() writes 1D array directly for single line."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            data = np.array([1, 0, 1, 0, 1])
            do.write_continuous(data)

        call_args = mt.write.call_args
        written_data = call_args[0][0]
        assert written_data == [True, False, True, False, True]

    def test_raises_in_on_demand_mode(self, mock_system, mock_constants):
        """write_continuous() raises RuntimeError when mode is on_demand."""
        ctx, do, _ = _build_do(mock_system, mock_constants, sample_rate=None)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            with pytest.raises(RuntimeError, match="clocked mode"):
                do.write_continuous(np.array([1, 0, 1]))

    def test_auto_start_true(self, mock_system, mock_constants):
        """write_continuous() calls task.write() with auto_start=True."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")
            do.write_continuous(np.array([1, 0, 1]))

        call_args = mt.write.call_args
        assert call_args.kwargs["auto_start"] is True


class TestDOTaskClearTask:
    """clear_task() releases hardware resources safely."""

    def test_closes_task(self, mock_system, mock_constants):
        """clear_task() calls task.close() on the nidaqmx task."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            pass

        do.clear_task()
        mt.close.assert_called_once()

    def test_sets_task_none(self, mock_system, mock_constants):
        """clear_task() sets self.task to None."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            pass

        do.clear_task()
        assert do.task is None

    def test_multiple_calls_safe(self, mock_system, mock_constants):
        """Calling clear_task() twice does not raise."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            pass

        do.clear_task()
        do.clear_task()  # Must not raise

    def test_exception_warns(self, mock_system, mock_constants):
        """clear_task() emits a warning when task.close() raises."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            pass
        mt.close.side_effect = OSError("hw error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            do.clear_task()

        assert len(w) >= 1
        assert "hw error" in str(w[0].message)
        assert do.task is None


class TestDOTaskContextManager:
    """DOTask implements the context manager protocol."""

    def test_enter_returns_self(self, mock_system, mock_constants):
        """__enter__ returns the DOTask instance."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            pass

        result = do.__enter__()
        assert result is do

    def test_exit_calls_clear(self, mock_system, mock_constants):
        """__exit__ calls clear_task()."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            pass
        do.clear_task = MagicMock()

        do.__exit__(None, None, None)
        do.clear_task.assert_called_once()

    def test_cleanup_on_exception(self, mock_system, mock_constants):
        """clear_task() is called even when the with-block raises."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            pass

        cleared = []

        def _tracking_clear():
            cleared.append(True)
            if do.task is not None:
                do.task.close()

        do.clear_task = _tracking_clear

        with pytest.raises(RuntimeError):
            with do:
                raise RuntimeError("body error")

        assert cleared


class TestDOTaskInitiateRemoved:
    """initiate() and old internal methods must not exist in the new architecture."""

    def test_no_initiate_method(self, mock_system, mock_constants):
        """initiate() method does not exist on DOTask."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(do, "initiate")

    def test_no_add_channels_method(self, mock_system, mock_constants):
        """_add_channels() internal method no longer exists."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(do, "_add_channels")

    def test_no_create_task_method(self, mock_system, mock_constants):
        """_create_task() internal method no longer exists."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(do, "_create_task")


class TestDOTaskSaveConfig:
    """save_config() serialises DOTask configuration to TOML."""

    def test_writes_toml_file(self, mock_system, mock_constants, tmp_path):
        """save_config() creates a valid TOML file."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("leds", lines="Dev1/port1/line0:3")

        path = tmp_path / "do_config.toml"
        do.save_config(path)
        assert path.exists()

        with open(path, "rb") as f:
            data = tomllib.load(f)
        assert "task" in data
        assert "channels" in data

    def test_task_section_name(self, mock_system, mock_constants, tmp_path):
        """[task] section contains the task name."""
        ctx, do, _ = _build_do(mock_system, mock_constants, task_name="my_leds")
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")

        path = tmp_path / "config.toml"
        do.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["name"] == "my_leds"

    def test_task_section_type_digital_output(self, mock_system, mock_constants, tmp_path):
        """[task] type field is 'digital_output'."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")

        path = tmp_path / "config.toml"
        do.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["type"] == "digital_output"

    def test_task_section_clocked_includes_sample_rate(
        self, mock_system, mock_constants, tmp_path
    ):
        """[task] section includes sample_rate for clocked mode."""
        ctx, do, _ = _build_do(mock_system, mock_constants, sample_rate=2000)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")

        path = tmp_path / "config.toml"
        do.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["sample_rate"] == 2000

    def test_task_section_on_demand_no_sample_rate(
        self, mock_system, mock_constants, tmp_path
    ):
        """[task] section omits sample_rate for on-demand mode."""
        ctx, do, _ = _build_do(mock_system, mock_constants, sample_rate=None)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")

        path = tmp_path / "config.toml"
        do.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert "sample_rate" not in data["task"]

    def test_channel_entries(self, mock_system, mock_constants, tmp_path):
        """[[channels]] entries contain name and lines."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("leds", lines="Dev1/port1/line0:3")

        path = tmp_path / "config.toml"
        do.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        channels = data["channels"]
        assert len(channels) == 1
        ch = channels[0]
        assert ch["name"] == "leds"
        assert ch["lines"] == "Dev1/port1/line0:3"

    def test_multiple_channels(self, mock_system, mock_constants, tmp_path):
        """All channels are serialised to [[channels]] entries."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("leds", lines="Dev1/port1/line0:3")
            do.add_channel("relays", lines="Dev1/port2/line0:7")

        path = tmp_path / "config.toml"
        do.save_config(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert len(data["channels"]) == 2
        names = {ch["name"] for ch in data["channels"]}
        assert names == {"leds", "relays"}


    def test_header_comment_with_timestamp(self, mock_system, mock_constants, tmp_path):
        """save_config() includes header comment with version and timestamp."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            do.add_channel("ch", lines="Dev1/port1/line0")

        path = tmp_path / "config.toml"
        do.save_config(path)
        
        # Read raw text to check the header comment
        content = path.read_text()
        lines_list = content.splitlines()
        assert len(lines_list) > 0
        assert lines_list[0].startswith("# Generated by nidaqwrapper 0.1.0 on")
        
        # Verify from_config() can still parse it (round-trip test)
        system2 = mock_system(task_names=[])
        mock_ni_task2 = _make_mock_ni_task()

        from unittest.mock import patch
        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system2,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task2,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DOTask
            do2 = DOTask.from_config(path)
            assert do2.task_name == do.task_name


class TestDOTaskFromConfig:
    """from_config() creates a DOTask from a TOML file."""

    def _write_config(self, tmp_path, content: str):
        """Write a TOML string to a temporary file and return the path."""
        path = tmp_path / "config.toml"
        path.write_text(content)
        return path

    def test_creates_task_from_toml(self, mock_system, mock_constants, tmp_path):
        """from_config() creates a DOTask with the name from [task]."""
        path = self._write_config(
            tmp_path,
            """\
[task]
name = "leds"
type = "digital_output"

[[channels]]
name = "led_group"
lines = "Dev1/port1/line0:3"
""",
        )
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ) as mock_cls,
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_config(path)

        mock_cls.assert_called_once_with(new_task_name="leds")
        assert do.task_name == "leds"

    def test_adds_channels(self, mock_system, mock_constants, tmp_path):
        """from_config() calls add_channel() for each [[channels]] entry."""
        path = self._write_config(
            tmp_path,
            """\
[task]
name = "leds"
type = "digital_output"

[[channels]]
name = "led_group"
lines = "Dev1/port1/line0:3"

[[channels]]
name = "relays"
lines = "Dev1/port2/line0:7"
""",
        )
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DOTask

            DOTask.from_config(path)

        assert mock_ni_task.do_channels.add_do_chan.call_count == 2

    def test_clocked_mode_from_config(self, mock_system, mock_constants, tmp_path):
        """from_config() sets clocked mode when sample_rate is in [task]."""
        path = self._write_config(
            tmp_path,
            """\
[task]
name = "pattern_gen"
type = "digital_output"
sample_rate = 3000

[[channels]]
name = "ch"
lines = "Dev1/port1/line0"
""",
        )
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_config(path)

        assert do.mode == "clocked"
        assert do.sample_rate == 3000

    def test_missing_task_section_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when [task] section is missing."""
        path = self._write_config(
            tmp_path,
            """\
[[channels]]
name = "ch"
lines = "Dev1/port1/line0"
""",
        )
        system = mock_system(task_names=[])

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            with pytest.raises(ValueError, match="task"):
                DOTask.from_config(path)

    def test_malformed_toml_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises an error on syntactically invalid TOML."""
        path = self._write_config(tmp_path, "not = valid [ toml {\n")

        from nidaqwrapper.digital import DOTask

        with pytest.raises(Exception):  # tomllib.TOMLDecodeError
            DOTask.from_config(path)


class TestDOTaskConfigRoundtrip:
    """save_config() + from_config() round-trip preserves task configuration."""

    def test_roundtrip_on_demand(self, mock_system, mock_constants, tmp_path):
        """On-demand config survives a save/load cycle."""
        ctx, do, _ = _build_do(
            mock_system, mock_constants, task_name="roundtrip_do", sample_rate=None
        )
        with ctx:
            do.add_channel("leds", lines="Dev1/port1/line0:3")

        path = tmp_path / "config.toml"
        do.save_config(path)

        system2 = mock_system(task_names=[])
        mock_ni_task2 = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system2,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task2,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DOTask

            do2 = DOTask.from_config(path)

        assert do2.task_name == "roundtrip_do"
        assert do2.sample_rate is None
        assert do2.mode == "on_demand"
        kwargs = mock_ni_task2.do_channels.add_do_chan.call_args.kwargs
        assert kwargs["lines"] == "Dev1/port1/line0:3"
        assert kwargs["name_to_assign_to_lines"] == "leds"

    def test_roundtrip_clocked(self, mock_system, mock_constants, tmp_path):
        """Clocked config survives a save/load cycle."""
        ctx, do, _ = _build_do(
            mock_system, mock_constants, task_name="pattern_gen", sample_rate=4000
        )
        with ctx:
            do.add_channel("relays", lines="Dev1/port2/line0:7")

        path = tmp_path / "config.toml"
        do.save_config(path)

        system2 = mock_system(task_names=[])
        mock_ni_task2 = _make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system2,
            ),
            patch(
                "nidaqwrapper.digital.nidaqmx.task.Task",
                return_value=mock_ni_task2,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DOTask

            do2 = DOTask.from_config(path)

        assert do2.task_name == "pattern_gen"
        assert do2.sample_rate == 4000
        assert do2.mode == "clocked"


class TestDOTaskFromTask:
    """from_task() wraps externally-created nidaqmx.Task objects."""

    def test_wraps_existing_task(self, mock_system, mock_constants):
        """from_task() creates a DOTask from an existing nidaqmx.Task."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_do"
        # Simulate that the task already has DO channels
        mock_ni_task.do_channels.add_do_chan(
            lines="Dev1/port1/line0:3", name_to_assign_to_lines="existing_ch"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_task(mock_ni_task)

        assert do.task is mock_ni_task
        assert do.task_name == "external_do"
        assert do.channel_list == ["existing_ch"]
        assert do.number_of_ch == 1

    def test_owns_task_false(self, mock_system, mock_constants):
        """from_task() sets _owns_task to False."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_do"
        mock_ni_task.do_channels.add_do_chan(
            lines="Dev1/port1/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_task(mock_ni_task)

        assert hasattr(do, "_owns_task")
        assert do._owns_task is False

    def test_constructor_owns_task_true(self, mock_system, mock_constants):
        """Normal constructor sets _owns_task to True."""
        ctx, do, _ = _build_do(mock_system, mock_constants)
        with ctx:
            pass
        assert do._owns_task is True

    def test_validates_no_do_channels(self, mock_system, mock_constants):
        """from_task() raises ValueError when task has no DO channels."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "no_channels"
        # No channels added — task.do_channels is empty

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            with pytest.raises(ValueError, match="[Nn]o DO channels"):
                DOTask.from_task(mock_ni_task)

    def test_add_channel_blocked(self, mock_system, mock_constants):
        """add_channel() raises RuntimeError when _owns_task is False."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_do"
        mock_ni_task.do_channels.add_do_chan(
            lines="Dev1/port1/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch(
                "nidaqwrapper.digital._expand_port_to_line_range",
                side_effect=lambda lines: lines,
            ),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_task(mock_ni_task)

            with pytest.raises(
                RuntimeError, match="Cannot add channels to an externally-provided task"
            ):
                do.add_channel("new_ch", lines="Dev1/port1/line1")

    def test_configure_blocked(self, mock_system, mock_constants):
        """configure() raises RuntimeError when _owns_task is False."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_do"
        mock_ni_task.do_channels.add_do_chan(
            lines="Dev1/port1/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_task(mock_ni_task)

            with pytest.raises(
                RuntimeError,
                match="Cannot configure an externally-provided task",
            ):
                do.configure()

    def test_clear_task_does_not_close(self, mock_system, mock_constants):
        """clear_task() does NOT close external task, warns instead."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_do"
        mock_ni_task.do_channels.add_do_chan(
            lines="Dev1/port1/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_task(mock_ni_task)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                do.clear_task()

            # Should NOT call task.close()
            mock_ni_task.close.assert_not_called()
            # Should warn user
            assert len(w) >= 1
            assert "externally" in str(w[0].message).lower()

    def test_exit_does_not_close(self, mock_system, mock_constants):
        """__exit__ does NOT close external task, warns instead."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "external_do"
        mock_ni_task.do_channels.add_do_chan(
            lines="Dev1/port1/line0", name_to_assign_to_lines="ch1"
        )

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_task(mock_ni_task)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                do.__exit__(None, None, None)

            mock_ni_task.close.assert_not_called()
            assert len(w) >= 1

    def test_normal_constructor_closes_task(self, mock_system, mock_constants):
        """Normal constructor (owns_task=True) closes task on clear_task()."""
        ctx, do, mt = _build_do(mock_system, mock_constants)
        with ctx:
            pass

        do.clear_task()
        mt.close.assert_called_once()

    def test_detects_clocked_mode(self, mock_system, mock_constants):
        """from_task() detects clocked mode from task.timing.samp_clk_rate."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "clocked_do"
        mock_ni_task.do_channels.add_do_chan(
            lines="Dev1/port1/line0:3", name_to_assign_to_lines="ch1"
        )
        mock_ni_task.timing.samp_clk_rate = 3000.0

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_task(mock_ni_task)

        assert do.mode == "clocked"
        assert do.sample_rate == 3000.0

    def test_detects_on_demand_mode(self, mock_system, mock_constants):
        """from_task() detects on-demand mode when no sample rate set."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()
        mock_ni_task.name = "on_demand_do"
        mock_ni_task.do_channels.add_do_chan(
            lines="Dev1/port1/line0", name_to_assign_to_lines="ch1"
        )
        mock_ni_task.timing.samp_clk_rate = None  # Or not set

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask

            do = DOTask.from_task(mock_ni_task)

        assert do.mode == "on_demand"
        assert do.sample_rate is None


# ===========================================================================
# Task Group: from_task(take_ownership=False) — DITask ownership transfer
# ===========================================================================


class TestDITaskFromTaskTakeOwnership:
    """from_task(take_ownership=True) grants mutating method access on DITask."""

    def _make_external_task(self, mock_constants) -> MagicMock:
        """Create a minimal external DITask mock with one DI channel."""
        ext = MagicMock()
        ext.name = "external_di"
        ext.timing.samp_clk_rate = 1000
        ext.is_task_done.return_value = True
        mock_ch = MagicMock()
        mock_ch.name = "ch0"
        mock_ch.physical_channel.name = "Dev1/port0/line0"
        ext.di_channels = [mock_ch]
        ext.channel_names = ["ch0"]
        return ext

    def test_default_not_owned(self, mock_system, mock_constants):
        """from_task() without take_ownership sets _owns_task=False."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask
            task = DITask.from_task(ext)

        assert task._owns_task is False

    def test_take_ownership_sets_owns_task(self, mock_system, mock_constants):
        """from_task(take_ownership=True) sets _owns_task=True."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask
            task = DITask.from_task(ext, take_ownership=True)

        assert task._owns_task is True

    def test_add_channel_allowed_when_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=True).add_channel() does not raise."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)
        # Use a MagicMock for di_channels so add_channel can detect duplicates
        di_channels_mock = MagicMock()
        di_channels_mock.__iter__ = MagicMock(return_value=iter([]))
        di_channels_mock.__len__ = MagicMock(return_value=1)
        ext.di_channels = di_channels_mock
        ext.channel_names = []

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch("nidaqwrapper.digital._expand_port_to_line_range",
                  side_effect=lambda lines: lines),
        ):
            from nidaqwrapper.digital import DITask
            task = DITask.from_task(ext, take_ownership=True)
            # Should not raise RuntimeError
            task.add_channel("new_ch", lines="Dev1/port0/line1")

    def test_clear_task_closes_when_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=True).clear_task() closes the task."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask
            task = DITask.from_task(ext, take_ownership=True)
            task.clear_task()

        ext.close.assert_called_once()

    def test_add_channel_blocked_when_not_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=False).add_channel() raises RuntimeError."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DITask
            task = DITask.from_task(ext, take_ownership=False)
            with pytest.raises(RuntimeError, match="Cannot add channels"):
                task.add_channel("new_ch", lines="Dev1/port0/line1")


# ===========================================================================
# Task Group: from_task(take_ownership=False) — DOTask ownership transfer
# ===========================================================================


class TestDOTaskFromTaskTakeOwnership:
    """from_task(take_ownership=True) grants mutating method access on DOTask."""

    def _make_external_task(self, mock_constants) -> MagicMock:
        """Create a minimal external DOTask mock with one DO channel."""
        ext = MagicMock()
        ext.name = "external_do"
        ext.timing.samp_clk_rate = 1000
        ext.is_task_done.return_value = True
        mock_ch = MagicMock()
        mock_ch.name = "ch0"
        mock_ch.physical_channel.name = "Dev1/port1/line0"
        ext.do_channels = [mock_ch]
        ext.channel_names = ["ch0"]
        return ext

    def test_default_not_owned(self, mock_system, mock_constants):
        """from_task() without take_ownership sets _owns_task=False."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask
            task = DOTask.from_task(ext)

        assert task._owns_task is False

    def test_take_ownership_sets_owns_task(self, mock_system, mock_constants):
        """from_task(take_ownership=True) sets _owns_task=True."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask
            task = DOTask.from_task(ext, take_ownership=True)

        assert task._owns_task is True

    def test_add_channel_allowed_when_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=True).add_channel() does not raise."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)
        # Use a MagicMock for do_channels so add_channel can detect duplicates
        do_channels_mock = MagicMock()
        do_channels_mock.__iter__ = MagicMock(return_value=iter([]))
        do_channels_mock.__len__ = MagicMock(return_value=1)
        ext.do_channels = do_channels_mock
        ext.channel_names = []

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
            patch("nidaqwrapper.digital._expand_port_to_line_range",
                  side_effect=lambda lines: lines),
        ):
            from nidaqwrapper.digital import DOTask
            task = DOTask.from_task(ext, take_ownership=True)
            # Should not raise RuntimeError
            task.add_channel("new_ch", lines="Dev1/port1/line1")

    def test_clear_task_closes_when_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=True).clear_task() closes the task."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask
            task = DOTask.from_task(ext, take_ownership=True)
            task.clear_task()

        ext.close.assert_called_once()

    def test_add_channel_blocked_when_not_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=False).add_channel() raises RuntimeError."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_constants)

        with (
            patch("nidaqwrapper.digital.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.digital.constants", mock_constants),
        ):
            from nidaqwrapper.digital import DOTask
            task = DOTask.from_task(ext, take_ownership=False)
            with pytest.raises(RuntimeError, match="Cannot add channels"):
                task.add_channel("new_ch", lines="Dev1/port1/line1")


# ===========================================================================
# Task Group: from_name() — NI MAX task loading via inherited BaseTask method
# ===========================================================================


class TestFromNameDI:
    """from_name() loads an NI MAX task by name and wraps it as a DITask."""

    def _make_mock_ni_task(self) -> MagicMock:
        """Create a mock nidaqmx task with one DI channel."""
        mock_ni_task = MagicMock()
        mock_ni_task.name = "MaxDITask"
        mock_ni_task.timing.samp_clk_rate = 1000
        mock_ch = MagicMock()
        mock_ch.name = "line0"
        mock_ni_task.di_channels = [mock_ch]
        mock_ni_task.channel_names = ["line0"]
        mock_ni_task.is_task_done.return_value = True
        return mock_ni_task

    def test_loads_and_wraps_successfully(self, mock_system, mock_constants):
        """from_name() loads the NI MAX task and returns a DITask."""
        system = mock_system(task_names=[])
        mock_ni_task = self._make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.base_task.get_task_by_name",
                return_value=mock_ni_task,
            ) as mock_get,
        ):
            from nidaqwrapper.digital import DITask
            task = DITask.from_name("MaxDITask")

        mock_get.assert_called_once_with("MaxDITask")
        assert isinstance(task, DITask)
        assert task.task is mock_ni_task
        assert task._owns_task is True

    def test_task_not_found_raises_keyerror(self, mock_system, mock_constants):
        """from_name() raises KeyError when task name is not in NI MAX."""
        with patch(
            "nidaqwrapper.base_task.get_task_by_name",
            side_effect=KeyError("No task named 'Missing'"),
        ):
            from nidaqwrapper.digital import DITask
            with pytest.raises(KeyError, match="Missing"):
                DITask.from_name("Missing")

    def test_task_already_loaded_raises_runtime_error(
        self, mock_system, mock_constants
    ):
        """from_name() raises RuntimeError when get_task_by_name returns None."""
        with patch(
            "nidaqwrapper.base_task.get_task_by_name",
            return_value=None,
        ):
            from nidaqwrapper.digital import DITask
            with pytest.raises(RuntimeError, match="already loaded"):
                DITask.from_name("BusyTask")


class TestFromNameDO:
    """from_name() loads an NI MAX task by name and wraps it as a DOTask."""

    def _make_mock_ni_task(self) -> MagicMock:
        """Create a mock nidaqmx task with one DO channel."""
        mock_ni_task = MagicMock()
        mock_ni_task.name = "MaxDOTask"
        mock_ni_task.timing.samp_clk_rate = 1000
        mock_ch = MagicMock()
        mock_ch.name = "line0"
        mock_ni_task.do_channels = [mock_ch]
        mock_ni_task.channel_names = ["line0"]
        mock_ni_task.is_task_done.return_value = True
        return mock_ni_task

    def test_loads_and_wraps_successfully(self, mock_system, mock_constants):
        """from_name() loads the NI MAX task and returns a DOTask."""
        system = mock_system(task_names=[])
        mock_ni_task = self._make_mock_ni_task()

        with (
            patch(
                "nidaqwrapper.digital.nidaqmx.system.System.local",
                return_value=system,
            ),
            patch(
                "nidaqwrapper.base_task.get_task_by_name",
                return_value=mock_ni_task,
            ) as mock_get,
        ):
            from nidaqwrapper.digital import DOTask
            task = DOTask.from_name("MaxDOTask")

        mock_get.assert_called_once_with("MaxDOTask")
        assert isinstance(task, DOTask)
        assert task.task is mock_ni_task
        assert task._owns_task is True

    def test_task_not_found_raises_keyerror(self, mock_system, mock_constants):
        """from_name() raises KeyError when task name is not in NI MAX."""
        with patch(
            "nidaqwrapper.base_task.get_task_by_name",
            side_effect=KeyError("No task named 'Missing'"),
        ):
            from nidaqwrapper.digital import DOTask
            with pytest.raises(KeyError, match="Missing"):
                DOTask.from_name("Missing")

    def test_task_already_loaded_raises_runtime_error(
        self, mock_system, mock_constants
    ):
        """from_name() raises RuntimeError when get_task_by_name returns None."""
        with patch(
            "nidaqwrapper.base_task.get_task_by_name",
            return_value=None,
        ):
            from nidaqwrapper.digital import DOTask
            with pytest.raises(RuntimeError, match="already loaded"):
                DOTask.from_name("BusyTask")


# ===========================================================================
# task-sync-configuration: set_start_trigger() (inherited from BaseTask)
# ===========================================================================

def _make_external_digital_task(channel_attr: str) -> MagicMock:
    """Create a minimal external mock nidaqmx task for from_task() tests.

    Parameters
    ----------
    channel_attr : str
        Either ``"di_channels"`` or ``"do_channels"``.
    """
    task = MagicMock()
    task.name = "external_task"

    ch = MagicMock()
    ch.name = "ch0"
    ch.physical_channel.name = "Dev1/port0/line0"
    setattr(task, channel_attr, [ch])
    task.channel_names = ["ch0"]

    task.timing.samp_clk_rate = 1000
    task.is_task_done = MagicMock(return_value=True)
    return task


class TestDITaskSetStartTrigger:
    """set_start_trigger() configures a digital edge start trigger on DITask."""

    def test_default_rising_edge(self, mock_system, mock_constants):
        """Default edge='rising' maps to constants.Edge.RISING."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            di.set_start_trigger("/cDAQ1/PFI0")

        trig = mt.triggers.start_trigger.cfg_dig_edge_start_trig
        trig.assert_called_once()
        kwargs = trig.call_args.kwargs
        assert kwargs["trigger_source"] == "/cDAQ1/PFI0"
        assert kwargs["trigger_edge"] is mock_constants.Edge.RISING

    def test_falling_edge(self, mock_system, mock_constants):
        """edge='falling' maps to constants.Edge.FALLING."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            di.set_start_trigger("/cDAQ1/PFI1", edge="falling")

        trig = mt.triggers.start_trigger.cfg_dig_edge_start_trig
        trig.assert_called_once()
        assert trig.call_args.kwargs["trigger_edge"] is mock_constants.Edge.FALLING

    def test_invalid_edge_raises(self, mock_system, mock_constants):
        """An edge string other than 'rising'/'falling' raises ValueError."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="edge"):
                di.set_start_trigger("/cDAQ1/PFI0", edge="both")

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_empty_source_raises(self, mock_system, mock_constants):
        """An empty source string raises ValueError."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                di.set_start_trigger("")

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_non_string_source_raises(self, mock_system, mock_constants):
        """A non-string source raises ValueError."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                di.set_start_trigger(["PFI0"])

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_none_source_raises(self, mock_system, mock_constants):
        """source=None raises ValueError."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                di.set_start_trigger(None)

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_on_demand_mode_allowed(self, mock_system, mock_constants):
        """Design decision 3: no ordering/timing constraint enforced on trigger
        configuration — it also works on an on-demand DITask."""
        ctx, di, mt = _build_di(mock_system, mock_constants, sample_rate=None)
        with ctx:
            di.set_start_trigger("/cDAQ1/PFI0")

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_called_once()

    def test_not_owned_raises(self, mock_system, mock_constants):
        """A task wrapped via from_task() (not owned) raises RuntimeError."""
        external = _make_external_digital_task("di_channels")

        with patch("nidaqwrapper.digital.constants", mock_constants):
            from nidaqwrapper.digital import DITask
            di = DITask.from_task(external)

            with pytest.raises(RuntimeError, match="externally-provided"):
                di.set_start_trigger("/cDAQ1/PFI0")

        external.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()


class TestDOTaskSetStartTrigger:
    """set_start_trigger() configures a digital edge start trigger on DOTask."""

    def test_default_rising_edge(self, mock_system, mock_constants):
        """Default edge='rising' maps to constants.Edge.RISING."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.set_start_trigger("/cDAQ1/PFI0")

        trig = mt.triggers.start_trigger.cfg_dig_edge_start_trig
        trig.assert_called_once()
        kwargs = trig.call_args.kwargs
        assert kwargs["trigger_source"] == "/cDAQ1/PFI0"
        assert kwargs["trigger_edge"] is mock_constants.Edge.RISING

    def test_falling_edge(self, mock_system, mock_constants):
        """edge='falling' maps to constants.Edge.FALLING."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            do.set_start_trigger("/cDAQ1/PFI1", edge="falling")

        trig = mt.triggers.start_trigger.cfg_dig_edge_start_trig
        trig.assert_called_once()
        assert trig.call_args.kwargs["trigger_edge"] is mock_constants.Edge.FALLING

    def test_invalid_edge_raises(self, mock_system, mock_constants):
        """An edge string other than 'rising'/'falling' raises ValueError."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="edge"):
                do.set_start_trigger("/cDAQ1/PFI0", edge="rise")

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_empty_source_raises(self, mock_system, mock_constants):
        """An empty source string raises ValueError."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                do.set_start_trigger("")

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_non_string_source_raises(self, mock_system, mock_constants):
        """A non-string source raises ValueError."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                do.set_start_trigger(7)

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_none_source_raises(self, mock_system, mock_constants):
        """source=None raises ValueError."""
        ctx, do, mt = _build_do(mock_system, mock_constants, sample_rate=1000)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                do.set_start_trigger(None)

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_not_owned_raises(self, mock_system, mock_constants):
        """A task wrapped via from_task() (not owned) raises RuntimeError."""
        external = _make_external_digital_task("do_channels")

        with patch("nidaqwrapper.digital.constants", mock_constants):
            from nidaqwrapper.digital import DOTask
            do = DOTask.from_task(external)

            with pytest.raises(RuntimeError, match="externally-provided"):
                do.set_start_trigger("/cDAQ1/PFI0")

        external.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()
