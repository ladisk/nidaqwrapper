"""Tests for nidaqwrapper.ai_task module (AITask class).

Architecture: Direct Delegation
-------------------------------
Constructor creates nidaqmx.Task immediately.
add_channel() delegates to nidaqmx task.ai_channels.add_ai_*() directly.
configure() configures timing. start() starts the task (inherited from BaseTask).
Getters read from nidaqmx task properties.

All tests use mocked nidaqmx — no hardware required.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — mock UNITS dict with __objclass__ for channel dispatch
# ---------------------------------------------------------------------------

def _make_mock_units():
    """Create a mock UNITS dict whose values have __objclass__ for dispatch."""
    units = {}

    def _make_const(objclass_name, value):
        c = MagicMock()
        c.__objclass__ = type(objclass_name, (), {"__name__": objclass_name})
        c._value_ = value
        c.__repr__ = lambda self: value
        return c

    units["mV/g"] = _make_const("AccelSensitivityUnits", "MILLIVOLTS_PER_G")
    units["mV/m/s**2"] = _make_const(
        "AccelSensitivityUnits", "MILLIVOLTS_PER_METERS_PER_SECOND_SQUARED"
    )
    units["g"] = _make_const("AccelUnits", "ACCEL_G")
    units["m/s**2"] = _make_const("AccelUnits", "METERS_PER_SECOND_SQUARED")
    units["mV/N"] = _make_const(
        "ForceIEPESensorSensitivityUnits", "MILLIVOLTS_PER_NEWTON"
    )
    units["N"] = _make_const("ForceUnits", "NEWTONS")
    units["V"] = _make_const("VoltageUnits", "VOLTS")
    return units


MOCK_UNITS = _make_mock_units()


# ---------------------------------------------------------------------------
# Helpers — mock nidaqmx.Task that tracks channel additions
# ---------------------------------------------------------------------------

def _make_mock_ni_task(samp_clk_rate=25600, usage_type_constants=None):
    """Create a mock nidaqmx.Task that tracks channel additions.

    The mock records all add_ai_*_chan() calls and maintains a channel list
    so that duplicate detection (via task.channel_names and ai_channels
    iteration) works correctly in the implementation under test.

    Each channel object is populated with the attributes that the new
    save_config() reads directly from self.task.ai_channels:
    - ai_meas_type, ai_accel_sensitivity, ai_accel_sensitivity_units,
      ai_accel_units (for accel channels)
    - ai_force_iepe_sensor_sensitivity, ai_force_iepe_sensor_sensitivity_units,
      ai_force_units (for force channels)
    - ai_voltage_units (for voltage channels)
    - ai_custom_scale.name, ai_custom_scale.lin_slope, ai_custom_scale.lin_y_intercept
    - ai_rng_low, ai_rng_high (always present)

    Parameters
    ----------
    samp_clk_rate : float
        Mock sample clock rate for timing assertions.
    usage_type_constants : object, optional
        Object whose ``UsageTypeAI`` attribute provides the ``ai_meas_type``
        sentinel values stored on channel mocks.  When provided (typically
        ``mock_constants`` from the fixture), the values match whatever
        ``save_config()`` sees as ``constants.UsageTypeAI.*`` at call time,
        so branch comparisons work correctly.  When ``None``, falls back to
        real nidaqmx constants (suitable for tests that do not call
        ``save_config()``).
    """
    # Determine which UsageTypeAI constants to store on channel mocks.
    # Must match what save_config() sees as constants.UsageTypeAI.* at
    # the time it runs (either real or patched mock_constants).
    if usage_type_constants is not None:
        _usage = usage_type_constants
    else:
        from nidaqmx import constants as _real_constants
        _usage = _real_constants

    task = MagicMock()
    _channel_names = []
    _channel_objects = []

    def _make_accel_handler():
        def handler(**kwargs):
            name = kwargs.get("name_to_assign_to_channel", "")
            phys = kwargs.get("physical_channel", "")
            _channel_names.append(name)
            ch = MagicMock()
            ch.name = name
            ch.physical_channel = MagicMock()
            ch.physical_channel.name = phys
            # Populate AI channel attributes for save_config()
            ch.ai_meas_type = _usage.UsageTypeAI.ACCELERATION_ACCELEROMETER_CURRENT_INPUT
            ch.ai_accel_sensitivity = kwargs.get("sensitivity", 100.0)
            ch.ai_accel_sensitivity_units = kwargs.get("sensitivity_units")
            ch.ai_accel_units = kwargs.get("units")
            ch.ai_rng_low = kwargs.get("min_val", -5.0)
            ch.ai_rng_high = kwargs.get("max_val", 5.0)
            # No custom scale for accel channels
            ch.ai_custom_scale = MagicMock()
            ch.ai_custom_scale.name = ""
            _channel_objects.append(ch)
        return handler

    def _make_force_handler():
        def handler(**kwargs):
            name = kwargs.get("name_to_assign_to_channel", "")
            phys = kwargs.get("physical_channel", "")
            _channel_names.append(name)
            ch = MagicMock()
            ch.name = name
            ch.physical_channel = MagicMock()
            ch.physical_channel.name = phys
            ch.ai_meas_type = _usage.UsageTypeAI.FORCE_IEPE_SENSOR
            ch.ai_force_iepe_sensor_sensitivity = kwargs.get("sensitivity", 22.5)
            ch.ai_force_iepe_sensor_sensitivity_units = kwargs.get("sensitivity_units")
            ch.ai_force_units = kwargs.get("units")
            ch.ai_rng_low = kwargs.get("min_val", -5.0)
            ch.ai_rng_high = kwargs.get("max_val", 5.0)
            ch.ai_custom_scale = MagicMock()
            ch.ai_custom_scale.name = ""
            _channel_objects.append(ch)
        return handler

    def _make_voltage_handler():
        def handler(**kwargs):
            name = kwargs.get("name_to_assign_to_channel", "")
            phys = kwargs.get("physical_channel", "")
            _channel_names.append(name)
            ch = MagicMock()
            ch.name = name
            ch.physical_channel = MagicMock()
            ch.physical_channel.name = phys
            ch.ai_meas_type = _usage.UsageTypeAI.VOLTAGE
            ch.ai_voltage_units = kwargs.get("units")
            ch.ai_rng_low = kwargs.get("min_val", -5.0)
            ch.ai_rng_high = kwargs.get("max_val", 5.0)
            # Custom scale handling: if custom_scale_name kwarg present, set name
            custom_scale_name = kwargs.get("custom_scale_name", "")
            ch.ai_custom_scale = MagicMock()
            ch.ai_custom_scale.name = custom_scale_name
            if custom_scale_name:
                # Populated by Scale.create_lin_scale mock in test
                ch.ai_custom_scale.lin_slope = 0.0
                ch.ai_custom_scale.lin_y_intercept = 0.0
            _channel_objects.append(ch)
        return handler

    task.ai_channels.add_ai_accel_chan.side_effect = _make_accel_handler()
    task.ai_channels.add_ai_force_iepe_chan.side_effect = _make_force_handler()
    task.ai_channels.add_ai_voltage_chan.side_effect = _make_voltage_handler()

    # channel_names: same list object, stays in sync as channels are added
    task.channel_names = _channel_names

    # ai_channels iteration (for physical channel duplicate detection and save_config)
    task.ai_channels.__iter__ = MagicMock(
        side_effect=lambda: iter(_channel_objects)
    )

    # Expose the internal channel objects list so tests can set attributes
    # after add_channel() (e.g. to set custom_scale.lin_slope)
    task._channel_objects = _channel_objects

    # Timing (for configure() tests)
    task._timing.samp_clk_rate = samp_clk_rate
    task.timing.samp_clk_rate = samp_clk_rate

    return task


# Build MOCK_UNITS_REVERSE for patching UNITS_REVERSE in save_config tests
MOCK_UNITS_REVERSE = {v: k for k, v in MOCK_UNITS.items()}


def _build(mock_system, mock_constants, sample_rate=25600, samp_clk_rate=None,
           task_names=None, task_name="test"):
    """Construct an AITask inside a fully-patched context.

    Returns (ai_task_instance, mock_nidaqmx_task, patch_context_manager).
    Use inside a ``with`` block — patches stay active so that add_channel()
    and configure() can also run under mocking.

    Example::

        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(...)
    """
    from contextlib import ExitStack

    if task_names is None:
        task_names = []
    if samp_clk_rate is None:
        samp_clk_rate = sample_rate

    system = mock_system(task_names=task_names)
    mock_ni_task = _make_mock_ni_task(
        samp_clk_rate=samp_clk_rate,
        usage_type_constants=mock_constants,
    )

    stack = ExitStack()
    stack.enter_context(
        patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
              return_value=system)
    )
    stack.enter_context(
        patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
              return_value=mock_ni_task)
    )
    stack.enter_context(
        patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS)
    )
    stack.enter_context(
        patch("nidaqwrapper.ai_task.UNITS_REVERSE", MOCK_UNITS_REVERSE)
    )
    stack.enter_context(
        patch("nidaqwrapper.ai_task.constants", mock_constants)
    )

    from nidaqwrapper.ai_task import AITask
    ni_task = AITask(task_name, sample_rate=sample_rate)

    return stack, ni_task, mock_ni_task


# ===========================================================================
# Task Group 1: AITask Constructor
# ===========================================================================

class TestAITaskConstructor:
    """Constructor creates nidaqmx.Task immediately (direct delegation)."""

    def test_creates_nidaqmx_task_with_name(self, mock_system, mock_constants):
        """Constructor calls nidaqmx.task.Task(new_task_name=task_name)."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task) as mock_cls,
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            AITask("vibration_test", sample_rate=25600)

        mock_cls.assert_called_once_with(new_task_name="vibration_test")

    def test_task_attribute_set_immediately(self, mock_system, mock_constants):
        """self.task is set to the nidaqmx.Task in the constructor."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask("test", sample_rate=25600)

        assert task.task is mock_ni_task

    def test_task_name_stored(self, mock_system, mock_constants):
        """task_name is stored on the instance."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.task_name == "test"

    def test_sample_rate_stored(self, mock_system, mock_constants):
        """sample_rate is stored on the instance."""
        ctx, task, _ = _build(mock_system, mock_constants, sample_rate=51200)
        with ctx:
            pass
        assert task.sample_rate == 51200

    def test_device_list_populated(self, mock_system, mock_constants):
        """device_list contains device name strings from the system."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.device_list == ["cDAQ1Mod1", "cDAQ1Mod2"]

    def test_device_product_type_populated(self, mock_system, mock_constants):
        """device_product_type contains product type strings."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.device_product_type == ["NI 9234", "NI 9263"]

    def test_duplicate_task_name_raises_valueerror(self, mock_system, mock_constants):
        """Constructor raises ValueError when task_name already exists in NI MAX."""
        system = mock_system(task_names=["existing_task"])

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(ValueError, match="already"):
                AITask("existing_task", sample_rate=25600)

    def test_no_channels_dict(self, mock_system, mock_constants):
        """The old self.channels dict no longer exists."""
        ctx, task, _ = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "channels")

    def test_no_settings_file_parameter(self, mock_system, mock_constants):
        """Constructor no longer accepts settings_file parameter."""
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(TypeError):
                AITask("test", sample_rate=25600, settings_file="foo.xlsx")


# ===========================================================================
# Task Group 1: add_channel() — Accelerometer channels
# ===========================================================================

class TestAddChannelAccel:
    """add_channel() with accelerometer-type sensors delegates to nidaqmx."""

    def test_calls_add_ai_accel_chan(self, mock_system, mock_constants):
        """Accelerometer channel calls add_ai_accel_chan() on the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        mt.ai_channels.add_ai_accel_chan.assert_called_once()

    def test_passes_physical_channel(self, mock_system, mock_constants):
        """Physical channel string includes device name and ai index."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=2,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod1/ai2"

    def test_passes_channel_name(self, mock_system, mock_constants):
        """Channel name is forwarded as name_to_assign_to_channel."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["name_to_assign_to_channel"] == "accel_x"

    def test_passes_sensitivity(self, mock_system, mock_constants):
        """Sensitivity value and units are forwarded to nidaqmx."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["sensitivity"] == 100.0
        assert kwargs["sensitivity_units"] == MOCK_UNITS["mV/g"]
        assert kwargs["units"] == MOCK_UNITS["g"]

    def test_passes_terminal_config_default(self, mock_system, mock_constants):
        """Terminal config is set to DEFAULT."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["terminal_config"] == mock_constants.TerminalConfiguration.DEFAULT

    def test_ms2_units(self, mock_system, mock_constants):
        """Accelerometer channel accepts mV/m/s**2 sensitivity and m/s**2 units."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_z", device="cDAQ1Mod1", channel_ind=2,
                sensitivity=10.204, sensitivity_units="mV/m/s**2", units="m/s**2",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["sensitivity_units"] == MOCK_UNITS["mV/m/s**2"]
        assert kwargs["units"] == MOCK_UNITS["m/s**2"]

    def test_min_val_zero_forwarded(self, mock_system, mock_constants):
        """min_val=0.0 is forwarded (not treated as falsy — LDAQ bug fix)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=0.0, max_val=50.0,
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["min_val"] == 0.0
        assert kwargs["max_val"] == 50.0

    def test_custom_min_max(self, mock_system, mock_constants):
        """Custom min_val and max_val are forwarded to nidaqmx."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=-50.0, max_val=50.0,
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["min_val"] == -50.0
        assert kwargs["max_val"] == 50.0

    def test_min_max_omitted_when_none(self, mock_system, mock_constants):
        """When min_val/max_val are None, they are NOT in the kwargs."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert "min_val" not in kwargs
        assert "max_val" not in kwargs

    def test_second_device(self, mock_system, mock_constants):
        """Channel on a second device uses that device's name."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod2", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        kwargs = mt.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod2/ai0"


# ===========================================================================
# Task Group 1: add_channel() — Force channels
# ===========================================================================

class TestAddChannelForce:
    """add_channel() with force (IEPE) sensors delegates to add_ai_force_iepe_chan."""

    def test_calls_add_ai_force_iepe_chan(self, mock_system, mock_constants):
        """Force channel calls add_ai_force_iepe_chan()."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "force_1", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=22.5, sensitivity_units="mV/N", units="N",
            )

        mt.ai_channels.add_ai_force_iepe_chan.assert_called_once()
        assert not mt.ai_channels.add_ai_accel_chan.called
        assert not mt.ai_channels.add_ai_voltage_chan.called

    def test_passes_sensitivity(self, mock_system, mock_constants):
        """Force channel sensitivity and units are forwarded."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "force_1", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=22.5, sensitivity_units="mV/N", units="N",
            )

        kwargs = mt.ai_channels.add_ai_force_iepe_chan.call_args.kwargs
        assert kwargs["sensitivity"] == 22.5
        assert kwargs["sensitivity_units"] == MOCK_UNITS["mV/N"]
        assert kwargs["units"] == MOCK_UNITS["N"]


# ===========================================================================
# Task Group 1: add_channel() — Voltage channels
# ===========================================================================

class TestAddChannelVoltage:
    """add_channel() with voltage channels delegates to add_ai_voltage_chan."""

    def test_calls_add_ai_voltage_chan(self, mock_system, mock_constants):
        """Plain voltage channel calls add_ai_voltage_chan()."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "voltage_1", device="cDAQ1Mod1", channel_ind=0, units="V",
            )

        mt.ai_channels.add_ai_voltage_chan.assert_called_once()
        assert not mt.ai_channels.add_ai_accel_chan.called
        assert not mt.ai_channels.add_ai_force_iepe_chan.called

    def test_no_sensitivity_required(self, mock_system, mock_constants):
        """Voltage channels don't need sensitivity or sensitivity_units."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "voltage_1", device="cDAQ1Mod1", channel_ind=0, units="V",
            )

        kwargs = mt.ai_channels.add_ai_voltage_chan.call_args.kwargs
        assert "sensitivity" not in kwargs
        assert "sensitivity_units" not in kwargs

    def test_float_custom_scale(self, mock_system, mock_constants):
        """Float scale creates a linear custom scale and uses FROM_CUSTOM_SCALE."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with patch("nidaqwrapper.ai_task.nidaqmx.Scale.create_lin_scale") as mock_scale:
                mock_scale.return_value.name = "voltage_1_scale"
                task.add_channel(
                    "voltage_1", device="cDAQ1Mod1", channel_ind=0,
                    units="V", scale=2500.0,
                )

        mock_scale.assert_called_once()
        scale_kwargs = mock_scale.call_args.kwargs
        assert scale_kwargs["slope"] == 2500.0
        assert scale_kwargs["y_intercept"] == 0

        chan_kwargs = mt.ai_channels.add_ai_voltage_chan.call_args.kwargs
        assert chan_kwargs["units"] == mock_constants.VoltageUnits.FROM_CUSTOM_SCALE
        assert chan_kwargs["custom_scale_name"] == "voltage_1_scale"

    def test_tuple_custom_scale(self, mock_system, mock_constants):
        """Tuple scale passes both slope and y_intercept."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with patch("nidaqwrapper.ai_task.nidaqmx.Scale.create_lin_scale") as mock_scale:
                mock_scale.return_value.name = "voltage_1_scale"
                task.add_channel(
                    "voltage_1", device="cDAQ1Mod1", channel_ind=0,
                    units="V", scale=(2500.0, -100.0),
                )

        scale_kwargs = mock_scale.call_args.kwargs
        assert scale_kwargs["slope"] == 2500.0
        assert scale_kwargs["y_intercept"] == -100.0

    def test_custom_scale_dispatches_to_voltage(self, mock_system, mock_constants):
        """Channel with scale always uses add_ai_voltage_chan regardless of units."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with patch("nidaqwrapper.ai_task.nidaqmx.Scale.create_lin_scale") as mock_scale:
                mock_scale.return_value.name = "ch_scale"
                task.add_channel(
                    "ch", device="cDAQ1Mod1", channel_ind=0,
                    units="V", scale=2500.0,
                )

        mt.ai_channels.add_ai_voltage_chan.assert_called_once()
        assert not mt.ai_channels.add_ai_accel_chan.called


# ===========================================================================
# Task Group 1: add_channel() — Validation
# ===========================================================================

class TestChannelValidation:
    """add_channel() input validation and error handling."""

    def test_duplicate_channel_name_raises(self, mock_system, mock_constants):
        """Adding a second channel with the same name raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            with pytest.raises(ValueError, match="accel_x"):
                task.add_channel(
                    "accel_x", device="cDAQ1Mod1", channel_ind=1,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )

    def test_duplicate_physical_channel_raises(self, mock_system, mock_constants):
        """Adding two channels for the same (device, channel) raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            with pytest.raises(ValueError, match="already"):
                task.add_channel(
                    "accel_y", device="cDAQ1Mod1", channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )

    def test_same_channel_on_different_device_ok(self, mock_system, mock_constants):
        """Same channel_ind on different devices is allowed."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.add_channel(
                "accel_y", device="cDAQ1Mod2", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        assert mt.ai_channels.add_ai_accel_chan.call_count == 2

    def test_reject_empty_device_string(self, mock_system, mock_constants):
        """Empty device string raises ValueError with clear message."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="device must be a non-empty string"):
                task.add_channel(
                    "accel_x", device="", channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )

    def test_missing_units_raises(self, mock_system, mock_constants):
        """add_channel() without units raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises((ValueError, TypeError)):
                task.add_channel(
                    "accel_x", device="cDAQ1Mod1", channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g",
                )

    def test_invalid_units_raises(self, mock_system, mock_constants):
        """Unrecognised units string raises ValueError when no scale given."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="units"):
                task.add_channel(
                    "accel_x", device="cDAQ1Mod1", channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g",
                    units="furlongs_per_fortnight",
                )

    def test_invalid_sensitivity_units_raises(self, mock_system, mock_constants):
        """Unrecognised sensitivity_units raises ValueError when no scale given."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="sensitivity_units"):
                task.add_channel(
                    "accel_x", device="cDAQ1Mod1", channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/parsec",
                    units="g",
                )

    def test_missing_sensitivity_for_accel_raises(self, mock_system, mock_constants):
        """Non-voltage channel without sensitivity raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="sensitivity"):
                task.add_channel(
                    "accel_x", device="cDAQ1Mod1", channel_ind=0,
                    sensitivity_units="mV/g", units="g",
                )

    def test_invalid_scale_type_raises(self, mock_system, mock_constants):
        """Scale of unsupported type raises TypeError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(TypeError, match="scale"):
                task.add_channel(
                    "voltage_1", device="cDAQ1Mod1", channel_ind=0,
                    units="V", scale="2500",
                )

    def test_multiple_channels_added(self, mock_system, mock_constants):
        """Multiple channels of different types can be added to one task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with patch("nidaqwrapper.ai_task.nidaqmx.Scale.create_lin_scale") as mock_scale:
                mock_scale.return_value.name = "v_scale"
                task.add_channel(
                    "accel_x", device="cDAQ1Mod1", channel_ind=0,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )
                task.add_channel(
                    "force_1", device="cDAQ1Mod1", channel_ind=1,
                    sensitivity=22.5, sensitivity_units="mV/N", units="N",
                )
                task.add_channel(
                    "voltage_1", device="cDAQ1Mod1", channel_ind=2,
                    units="V",
                )

        assert mt.ai_channels.add_ai_accel_chan.called
        assert mt.ai_channels.add_ai_force_iepe_chan.called
        assert mt.ai_channels.add_ai_voltage_chan.called


# ===========================================================================
# fix-gh-issues-5-8: eager validation in add_channel() (issues #5, #6)
# ===========================================================================

class TestAddChannelEagerValidation:
    """add_channel() configures timing and forces TASK_VERIFY after the add.

    Fixes GH issues #5 (-201087 on delta-sigma/IEPE modules: subsequent
    channel-attribute reads require configured timing) and #6 (deferred
    property validation: errors must surface on the offending line).
    """

    def test_add_cfg_verify_called_in_order(self, mock_system, mock_constants):
        """Call order is: add channel -> cfg_samp_clk_timing -> TASK_VERIFY."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            order = []
            orig_add = mt.ai_channels.add_ai_voltage_chan.side_effect

            def _tracking_add(**kwargs):
                order.append("add")
                return orig_add(**kwargs)

            mt.ai_channels.add_ai_voltage_chan.side_effect = _tracking_add
            mt.timing.cfg_samp_clk_timing.side_effect = (
                lambda *a, **kw: order.append("cfg")
            )
            mt.control.side_effect = lambda *a, **kw: order.append("verify")

            task.add_channel("v0", device="cDAQ1Mod1", channel_ind=0, units="V")

        assert order == ["add", "cfg", "verify"]

    def test_cfg_uses_stored_default_settings(self, mock_system, mock_constants):
        """Eager timing config uses the stored defaults (continuous, onboard)."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=25600)
        with ctx:
            task.add_channel("v0", device="cDAQ1Mod1", channel_ind=0, units="V")

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["rate"] == 25600
        assert kwargs["sample_mode"] is mock_constants.AcquisitionType.CONTINUOUS
        assert "samps_per_chan" not in kwargs
        assert "source" not in kwargs

    def test_verify_uses_task_verify_constant(self, mock_system, mock_constants):
        """task.control() is called with TaskMode.TASK_VERIFY."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel("v0", device="cDAQ1Mod1", channel_ind=0, units="V")

        mt.control.assert_called_once_with(mock_constants.TaskMode.TASK_VERIFY)

    def test_cfg_uses_stored_settings_after_configure(self, mock_system,
                                                      mock_constants):
        """A later add_channel() re-applies the task's CURRENT stored settings."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=25600)
        with ctx:
            task.add_channel("v0", device="cDAQ1Mod1", channel_ind=0, units="V")
            task.configure(sample_mode="finite", samples_per_channel=1000,
                           clock_source="/cDAQ1Mod1/ai/SampleClock")
            task.add_channel("v1", device="cDAQ1Mod1", channel_ind=1, units="V")

        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["rate"] == 25600
        assert kwargs["sample_mode"] is mock_constants.AcquisitionType.FINITE
        assert kwargs["samps_per_chan"] == 1000
        assert kwargs["source"] == "/cDAQ1Mod1/ai/SampleClock"

    def test_daqerror_from_cfg_propagates(self, mock_system, mock_constants):
        """A DaqError raised by cfg_samp_clk_timing propagates unchanged."""
        from nidaqmx.errors import DaqError

        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            mt.timing.cfg_samp_clk_timing.side_effect = DaqError(
                "timing rejected", -201087
            )
            with pytest.raises(DaqError):
                task.add_channel("v0", device="cDAQ1Mod1", channel_ind=0,
                                 units="V")

    def test_daqerror_from_verify_propagates(self, mock_system, mock_constants):
        """A DaqError raised by TASK_VERIFY propagates unchanged (issue #6)."""
        from nidaqmx.errors import DaqError

        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            mt.control.side_effect = DaqError("requested value not supported",
                                              -200077)
            with pytest.raises(DaqError) as exc_info:
                task.add_channel("v0", device="cDAQ1Mod1", channel_ind=0,
                                 units="V", min_val=-1e5, max_val=1e5)

        assert exc_info.value.error_code == -200077

    def test_no_rate_coercion_check_in_add_channel(self, mock_system,
                                                   mock_constants):
        """add_channel() does NOT raise on rate coercion (delta-sigma coerces
        legally); the strict check stays in configure() only."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25000)
        with ctx:
            task.add_channel("v0", device="cDAQ1Mod1", channel_ind=0,
                             units="V")  # no raise


class TestAddChannelDuplicateGuard:
    """The duplicate pre-check iteration is guarded against -201087 (issue #5)."""

    def test_iteration_201087_warns_and_skips(self, mock_system, mock_constants):
        """DaqError -201087 during iteration -> warning, pre-check skipped,
        the channel is still added."""
        from nidaqmx.errors import DaqError

        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            mt.ai_channels.__iter__ = MagicMock(
                side_effect=DaqError("on-demand not supported", -201087)
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task.add_channel("v0", device="cDAQ1Mod1", channel_ind=0,
                                 units="V")

        assert any("pre-check" in str(x.message) for x in w)
        mt.ai_channels.add_ai_voltage_chan.assert_called_once()

    def test_iteration_other_daqerror_reraises(self, mock_system, mock_constants):
        """Any other DaqError from the iteration is re-raised; no channel added."""
        from nidaqmx.errors import DaqError

        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            mt.ai_channels.__iter__ = MagicMock(
                side_effect=DaqError("device gone", -50103)
            )
            with pytest.raises(DaqError) as exc_info:
                task.add_channel("v0", device="cDAQ1Mod1", channel_ind=0,
                                 units="V")

        assert exc_info.value.error_code == -50103
        mt.ai_channels.add_ai_voltage_chan.assert_not_called()


class TestFromTaskTimingGuard:
    """from_task() tolerates untimed tasks on delta-sigma devices (issue #5)."""

    # All timing attributes from_task() reads — on a real untimed
    # delta-sigma task, EVERY one of these raises -201087 (implicit
    # verification), so the mock must raise from all of them.
    _TIMING_ATTRS = (
        "samp_clk_rate",
        "samp_quant_samp_mode",
        "samp_quant_samp_per_chan",
        "samp_clk_src",
    )

    def _make_external_task(self, mock_constants, rate_error=None,
                            timing_error=None):
        from unittest.mock import PropertyMock

        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True
        if timing_error is not None:
            # ALL timing reads raise (real untimed delta-sigma behavior)
            for attr in self._TIMING_ATTRS:
                setattr(
                    type(mock_ni_task.timing), attr,
                    PropertyMock(side_effect=timing_error),
                )
        elif rate_error is not None:
            type(mock_ni_task.timing).samp_clk_rate = PropertyMock(
                side_effect=rate_error
            )
        else:
            mock_ni_task.timing.samp_clk_rate = 25600
            mock_ni_task.timing.samp_quant_samp_mode = (
                mock_constants.AcquisitionType.CONTINUOUS
            )
            mock_ni_task.timing.samp_clk_src = ""
        return mock_ni_task

    def test_all_timing_reads_201087_safe_defaults_single_warning(
            self, mock_system, mock_constants):
        """All timing reads raising -201087 -> wrapper created with safe
        defaults (mirroring __init__) and exactly one warning."""
        from nidaqmx.errors import DaqError

        external = self._make_external_task(
            mock_constants,
            timing_error=DaqError("timing not configured", -201087),
        )

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task = AITask.from_task(external)

        assert task.sample_rate is None
        assert task.sample_mode == mock_constants.AcquisitionType.CONTINUOUS
        assert task.sample_mode_str == "continuous"
        assert task.samples_per_channel is None
        assert task.clock_source is None
        assert len(w) == 1
        assert "sample rate" in str(w[0].message).lower()
        assert "-201087" in str(w[0].message)

    def test_samp_clk_rate_201087_sets_none_and_warns(self, mock_system,
                                                      mock_constants):
        """DaqError -201087 reading samp_clk_rate -> sample_rate=None + warning."""
        from nidaqmx.errors import DaqError

        external = self._make_external_task(
            mock_constants,
            rate_error=DaqError("timing not configured", -201087),
        )

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task = AITask.from_task(external)

        assert task.sample_rate is None
        assert task.sample_mode_str == "continuous"
        assert task.samples_per_channel is None
        assert task.clock_source is None
        assert any("sample rate" in str(x.message).lower() for x in w)

    def test_samp_clk_rate_other_daqerror_propagates(self, mock_system,
                                                     mock_constants):
        """Any other DaqError reading samp_clk_rate propagates unchanged."""
        from nidaqmx.errors import DaqError

        external = self._make_external_task(
            mock_constants, rate_error=DaqError("device gone", -50103)
        )

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(DaqError) as exc_info:
                AITask.from_task(external)

        assert exc_info.value.error_code == -50103

    def test_normal_task_timing_reads_unchanged_no_warning(self, mock_system,
                                                           mock_constants):
        """A normally-timed task is read without warnings; values intact."""
        external = self._make_external_task(mock_constants)

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task = AITask.from_task(external)

        assert task.sample_rate == 25600.0
        assert task.sample_mode == mock_constants.AcquisitionType.CONTINUOUS
        assert task.sample_mode_str == "continuous"
        assert task.samples_per_channel is None
        assert task.clock_source is None
        assert len(w) == 0


class TestApplyTimingSampleRateGuard:
    """_apply_timing() refuses to run with sample_rate=None (review F2)."""

    def test_sample_rate_none_raises_runtime_error(self, mock_system,
                                                   mock_constants):
        """sample_rate=None -> explicit RuntimeError, not an opaque
        ctypes error from cfg_samp_clk_timing(rate=None)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.sample_rate = None
            with pytest.raises(RuntimeError, match="sample_rate is not set"):
                task._apply_timing()

        mt.timing.cfg_samp_clk_timing.assert_not_called()


class TestConfigureStateConsistency:
    """configure() persists timing attributes only after the driver
    accepts the timing (review F3)."""

    def test_rejected_timing_leaves_stored_state_unchanged(self, mock_system,
                                                           mock_constants):
        """cfg_samp_clk_timing raising in configure() -> stored
        sample_mode_str/samples_per_channel/clock_source keep their
        prior values (add_channel() must not replay rejected settings)."""
        from nidaqmx.errors import DaqError

        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=25600)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            prior = (task.sample_mode, task.sample_mode_str,
                     task.samples_per_channel, task.clock_source)

            mt.timing.cfg_samp_clk_timing.side_effect = DaqError(
                "timing rejected", -200077
            )
            with pytest.raises(DaqError):
                task.configure(
                    sample_mode="finite",
                    samples_per_channel=1000,
                    clock_source="/cDAQ1Mod1/ai/SampleClock",
                )

        assert (task.sample_mode, task.sample_mode_str,
                task.samples_per_channel, task.clock_source) == prior

    def test_accepted_timing_persists_new_state(self, mock_system,
                                                mock_constants):
        """When the driver accepts the timing, configure() stores the
        new attribute values (unchanged behavior)."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=25600)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.configure(
                sample_mode="finite",
                samples_per_channel=1000,
                clock_source="/cDAQ1Mod1/ai/SampleClock",
            )

        assert task.sample_mode_str == "finite"
        assert task.samples_per_channel == 1000
        assert task.clock_source == "/cDAQ1Mod1/ai/SampleClock"


# ===========================================================================
# Task Group 2: configure() — configures timing (formerly start())
# ===========================================================================

class TestConfigure:
    """configure() sets up timing on the nidaqmx task without starting it."""

    def test_configures_timing(self, mock_system, mock_constants):
        """configure() calls cfg_samp_clk_timing with correct rate and CONTINUOUS mode."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=25600)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            # add_channel() applies timing eagerly; isolate configure()'s call
            mt.timing.cfg_samp_clk_timing.reset_mock()
            task.configure()

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["rate"] == 25600
        assert kwargs["sample_mode"] == mock_constants.AcquisitionType.CONTINUOUS

    def test_validates_sample_rate_pass(self, mock_system, mock_constants):
        """configure() succeeds when actual rate matches requested rate."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25600)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.configure()  # Should not raise

    def test_validates_sample_rate_fail(self, mock_system, mock_constants):
        """configure() raises ValueError when driver coerces the rate."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25000)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            with pytest.raises(ValueError, match="[Ss]ample.?[Rr]ate|rate"):
                task.configure()

    def test_does_not_start_task(self, mock_system, mock_constants):
        """configure() does NOT call task.start() on the underlying nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.configure()

        mt.start.assert_not_called()

    def test_configure_no_channels_raises(self, mock_system, mock_constants):
        """configure() raises ValueError when no channels have been added."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="Cannot configure: no channels"):
                task.configure()

    def test_rate_validation_failure_does_not_close_task(self, mock_system, mock_constants):
        """On rate mismatch, task handle remains valid (clear_task() not called).

        Unlike old initiate() which created then destroyed the task on failure,
        configure() merely configures timing on an existing task. The task handle
        survives a rate validation failure.
        """
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25000)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            with pytest.raises(ValueError):
                task.configure()

        # task.close() should NOT have been called — the task still exists
        mt.close.assert_not_called()


# ===========================================================================
# Task Group 2: start() — BaseTask.start() delegates to nidaqmx task.start()
# ===========================================================================

class TestBaseTaskStart:
    """start() is inherited from BaseTask and delegates to self.task.start()."""

    def test_start_calls_task_start(self, mock_system, mock_constants):
        """start() delegates to self.task.start() on the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.configure()
            task.start()

        mt.start.assert_called_once()

    def test_start_blocked_on_external_task(self, mock_system, mock_constants):
        """start() raises RuntimeError when _owns_task is False.

        An externally-provided task (from_task()) is not owned by the wrapper,
        so start() must refuse to start it to avoid conflicting with the
        external caller's lifecycle management.
        """
        system = mock_system(task_names=[])

        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ni_task.timing.samp_quant_samp_mode = "CONTINUOUS"

        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

            with pytest.raises(RuntimeError, match="Cannot start"):
                task.start()


# ===========================================================================
# Task Group 2: Getters — read from nidaqmx
# ===========================================================================

class TestGetters:
    """Getters delegate to nidaqmx task properties."""

    def test_channel_list_from_nidaqmx(self, mock_system, mock_constants):
        """channel_list returns names from the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.add_channel(
                "accel_y", device="cDAQ1Mod1", channel_ind=1,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

        assert task.channel_list == ["accel_x", "accel_y"]

    def test_number_of_ch(self, mock_system, mock_constants):
        """number_of_ch returns count from the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            assert task.number_of_ch == 0

            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            assert task.number_of_ch == 1

            task.add_channel(
                "accel_y", device="cDAQ1Mod1", channel_ind=1,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            assert task.number_of_ch == 2

    def test_channel_list_empty_initially(self, mock_system, mock_constants):
        """channel_list is empty on a new task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert task.channel_list == []

    def test_sample_rate_from_attribute(self, mock_system, mock_constants):
        """sample_rate property returns the stored sample rate."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=51200)
        with ctx:
            pass
        assert task.sample_rate == 51200


# ===========================================================================
# Task Group 2: from_settings() removed
# ===========================================================================

class TestSettingsRemoved:
    """All CSV/Excel settings file support is removed."""

    def test_no_from_settings_method(self, mock_system, mock_constants):
        """from_settings() method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "from_settings")

    def test_no_read_settings_file_method(self, mock_system, mock_constants):
        """_read_settings_file() method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_read_settings_file")

    def test_no_lookup_serial_nr_method(self, mock_system, mock_constants):
        """_lookup_serial_nr() method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_lookup_serial_nr")

    def test_no_settings_attribute(self, mock_system, mock_constants):
        """settings attribute no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "settings")


# ===========================================================================
# Existing: acquire() — minimal changes
# ===========================================================================

class TestAcquireBase:
    """acquire() reads all available samples from the hardware buffer."""

    def test_multi_channel(self, mock_system, mock_constants):
        """Multi-channel square case (3 channels x 3 samples) returns (3, 3)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]

        result = task.acquire()

        assert isinstance(result, np.ndarray)
        # Square case: (n_channels, n_samples) == (n_samples, n_channels) == (3, 3)
        assert result.shape == (3, 3)
        # After transpose, columns are channels: result[:, 0] = channel 0's samples
        np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])

    def test_single_channel_reshaped(self, mock_system, mock_constants):
        """Single-channel 1D result is reshaped to (n_samples, 1)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = [1.0, 2.0, 3.0, 4.0]

        result = task.acquire()

        assert result.ndim == 2
        assert result.shape == (4, 1)

    def test_empty_buffer(self, mock_system, mock_constants):
        """Empty buffer returns an empty array without raising."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = []

        result = task.acquire()
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_calls_acquire(self, mock_system, mock_constants):
        """task.read() is called with number_of_samples_per_channel=-1."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = [[0.0], [0.0]]

        task.acquire()

        mt.read.assert_called_once_with(number_of_samples_per_channel=-1)

    def test_multi_channel_non_square(self, mock_system, mock_constants):
        """Multi-channel non-square: 2 channels x 5 samples returns (5, 2)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = [
            [10.0, 20.0, 30.0, 40.0, 50.0],  # channel 0
            [1.0, 2.0, 3.0, 4.0, 5.0],        # channel 1
        ]

        result = task.acquire()

        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 2)

    def test_single_channel_shape(self, mock_system, mock_constants):
        """Single-channel explicitly returns (n_samples, 1), not (1, n_samples)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.read.return_value = [5.0, 6.0, 7.0, 8.0]

        result = task.acquire()

        assert result.shape == (4, 1)
        assert result.shape[0] == 4   # samples dimension
        assert result.shape[1] == 1   # channels dimension

    def test_multi_channel_first_column_is_first_channel(self, mock_system, mock_constants):
        """Data orientation: result[:, 0] contains the first channel's samples."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        ch0_data = [10.0, 20.0, 30.0, 40.0, 50.0]
        ch1_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mt.read.return_value = [ch0_data, ch1_data]

        result = task.acquire()

        # result[:, 0] must be channel 0's samples (not sample 0 across channels)
        np.testing.assert_array_equal(result[:, 0], ch0_data)
        np.testing.assert_array_equal(result[:, 1], ch1_data)


# ===========================================================================
# Existing: clear_task()
# ===========================================================================

class TestClearTask:
    """clear_task() releases hardware resources."""

    def test_calls_close(self, mock_system, mock_constants):
        """clear_task() calls task.close()."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.clear_task()
        mt.close.assert_called_once()

    def test_sets_task_none(self, mock_system, mock_constants):
        """clear_task() sets self.task to None."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.clear_task()
        assert task.task is None

    def test_multiple_calls_no_error(self, mock_system, mock_constants):
        """Calling clear_task() twice raises no exception."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.clear_task()
        task.clear_task()  # Must not raise

    def test_exception_warns_not_propagated(self, mock_system, mock_constants):
        """clear_task() emits warning when task.close() raises."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.close.side_effect = RuntimeError("close failed")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.clear_task()

        assert len(w) >= 1
        assert "close failed" in str(w[0].message)
        assert task.task is None


# ===========================================================================
# Existing: save()
# ===========================================================================

class TestSave:
    """save() persists the task to NI MAX."""

    def test_calls_nidaqmx_save(self, mock_system, mock_constants):
        """save() calls task.save(overwrite_existing_task=True)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        task.save()
        mt.save.assert_called_once_with(overwrite_existing_task=True)

    def test_clears_task_by_default(self, mock_system, mock_constants):
        """save() calls clear_task() after saving when clear_task=True."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        task.clear_task = MagicMock()

        task.save()
        task.clear_task.assert_called_once()

    def test_does_not_clear_when_false(self, mock_system, mock_constants):
        """save(clear_task=False) saves but does not close."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        task.clear_task = MagicMock()

        task.save(clear_task=False)
        mt.save.assert_called_once()
        task.clear_task.assert_not_called()

    def test_no_auto_initiate(self, mock_system, mock_constants):
        """save() does NOT auto-initiate (task always exists in new arch)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        # Task always exists — save should work directly
        task.save()
        mt.save.assert_called_once()

    def test_save_blocked_on_external_task(self, mock_system, mock_constants):
        """save() raises RuntimeError when _owns_task is False (fix-gh-issues-5-8).

        An externally-provided task (from_task) is not owned by the wrapper,
        so save() must refuse to persist it — the external caller controls
        the task lifecycle and NI MAX persistence.
        """
        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

            with pytest.raises(RuntimeError, match="[Cc]annot save"):
                task.save()

        mock_ni_task.save.assert_not_called()


# ===========================================================================
# fix-gh-issues-5-8: stop() — BaseTask.stop() delegates to nidaqmx task.stop()
# ===========================================================================

class TestStop:
    """stop() is inherited from BaseTask and delegates to self.task.stop()."""

    def test_stop_calls_task_stop(self, mock_system, mock_constants):
        """stop() delegates to self.task.stop() on the nidaqmx task."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.configure()
            task.start()
            task.stop()

        mt.stop.assert_called_once()

    def test_stop_does_not_close_task(self, mock_system, mock_constants):
        """stop() leaves the task handle open (unlike clear_task())."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.configure()
            task.start()
            task.stop()

        mt.close.assert_not_called()
        assert task.task is mt

    def test_stop_blocked_on_external_task(self, mock_system, mock_constants):
        """stop() raises RuntimeError when _owns_task is False."""
        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

            with pytest.raises(RuntimeError, match="[Cc]annot stop"):
                task.stop()

        mock_ni_task.stop.assert_not_called()


# ===========================================================================
# Existing: Context Manager
# ===========================================================================

class TestContextManager:
    """AITask __enter__/__exit__ (context manager protocol)."""

    def test_enter_returns_self(self, mock_system, mock_constants):
        """__enter__ returns the AITask instance."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        result = task.__enter__()
        assert result is task

    def test_exit_calls_clear_task(self, mock_system, mock_constants):
        """__exit__ calls clear_task()."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        task.clear_task = MagicMock()

        task.__exit__(None, None, None)
        task.clear_task.assert_called_once()

    def test_exception_still_clears(self, mock_system, mock_constants):
        """Exception in with-block still triggers cleanup."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        cleared = []
        original_clear = task.clear_task

        def _tracking_clear():
            cleared.append(True)
            if task.task is not None:
                task.task.close()

        task.clear_task = _tracking_clear

        with pytest.raises(ValueError):
            with task:
                raise ValueError("body error")

        assert cleared

    def test_cleanup_exception_warns(self, mock_system, mock_constants):
        """Cleanup exception emits warning, not raised."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        mt.close.side_effect = RuntimeError("cleanup error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.__exit__(None, None, None)

        assert len(w) >= 1
        assert "cleanup error" in str(w[0].message)


# ===========================================================================
# Task Group 2: initiate() removed
# ===========================================================================

class TestInitiateRemoved:
    """initiate() method no longer exists."""

    def test_no_initiate_method(self, mock_system, mock_constants):
        """initiate() method does not exist on AITask."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "initiate")

    def test_no_add_channels_method(self, mock_system, mock_constants):
        """_add_channels() internal method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_add_channels")

    def test_no_add_channel_private_method(self, mock_system, mock_constants):
        """_add_channel() private method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_add_channel")

    def test_no_setup_task_method(self, mock_system, mock_constants):
        """_setup_task() internal method no longer exists."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass
        assert not hasattr(task, "_setup_task")


# ===========================================================================
# Task Group 3: TOML config save/load
# ===========================================================================

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


class TestSaveConfig:
    """save_config() serialises the task configuration to TOML.

    Note: save_config() is called INSIDE the ``with ctx:`` block so that
    UNITS_REVERSE (patched in _build) is active when save_config() runs.
    The new save_config() reads from self.task.ai_channels and uses
    UNITS_REVERSE to convert nidaqmx constants back to unit strings.
    """

    def test_writes_toml_file(self, mock_system, mock_constants, tmp_path):
        """save_config() creates a file that can be parsed as TOML."""
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.save_config(path)

        assert path.exists()
        with open(path, "rb") as f:
            data = tomllib.load(f)
        assert "task" in data
        assert "devices" in data
        assert "channels" in data

    def test_task_section(self, mock_system, mock_constants, tmp_path):
        """[task] section contains name, sample_rate, and type='input'."""
        ctx, task, mt = _build(mock_system, mock_constants, sample_rate=51200)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.save_config(path)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        assert data["task"]["name"] == "test"
        assert data["task"]["sample_rate"] == 51200
        assert data["task"]["type"] == "input"

    def test_devices_section(self, mock_system, mock_constants, tmp_path):
        """[devices] section contains unique device aliases for used devices."""
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.add_channel(
                "accel_y", device="cDAQ1Mod2", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.save_config(path)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Each used device gets an alias in the [devices] section
        devices = data["devices"]
        assert len(devices) == 2
        device_names = set(devices.values())
        assert "cDAQ1Mod1" in device_names
        assert "cDAQ1Mod2" in device_names

    def test_channel_entries(self, mock_system, mock_constants, tmp_path):
        """[[channels]] entries contain name, device alias, channel, units, etc."""
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=2,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=-50.0, max_val=50.0,
            )
            task.save_config(path)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        channels = data["channels"]
        assert len(channels) == 1
        ch = channels[0]
        assert ch["name"] == "accel_x"
        assert ch["channel"] == 2
        assert ch["sensitivity"] == 100.0
        assert ch["sensitivity_units"] == "mV/g"
        assert ch["units"] == "g"
        assert ch["min_val"] == -50.0
        assert ch["max_val"] == 50.0
        # Device alias must reference a key in [devices]
        assert ch["device"] in data["devices"]

    def test_force_channel(self, mock_system, mock_constants, tmp_path):
        """Force/IEPE channel is saved correctly in TOML."""
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel(
                "force_1", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=22.5, sensitivity_units="mV/N", units="N",
            )
            task.save_config(path)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert ch["name"] == "force_1"
        assert ch["sensitivity"] == 22.5
        assert ch["sensitivity_units"] == "mV/N"
        assert ch["units"] == "N"

    def test_min_val_zero_preserved(self, mock_system, mock_constants, tmp_path):
        """min_val=0.0 is saved in TOML (not treated as falsy)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=0.0, max_val=50.0,
            )
            task.save_config(path)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert ch["min_val"] == 0.0
        assert ch["max_val"] == 50.0

    def test_voltage_channel_no_sensitivity(self, mock_system, mock_constants, tmp_path):
        """Voltage channels omit sensitivity/sensitivity_units from TOML."""
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel("v1", device="cDAQ1Mod1", channel_ind=0, units="V")
            task.save_config(path)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert ch["units"] == "V"
        assert "sensitivity" not in ch
        assert "sensitivity_units" not in ch

    def test_scale_channel(self, mock_system, mock_constants, tmp_path):
        """Channel with custom scale saves scale value in TOML, omits units."""
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            with patch("nidaqwrapper.ai_task.nidaqmx.Scale.create_lin_scale") as ms:
                ms.return_value.name = "v1_scale"
                task.add_channel(
                    "v1", device="cDAQ1Mod1", channel_ind=0,
                    units="V", scale=(2500.0, -100.0),
                )
            # Set the custom scale slope/y_intercept on the channel mock
            # (save_config reads these from ch.ai_custom_scale)
            mt._channel_objects[0].ai_custom_scale.lin_slope = 2500.0
            mt._channel_objects[0].ai_custom_scale.lin_y_intercept = -100.0
            task.save_config(path)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        assert ch["scale"] == [2500.0, -100.0]
        # units is omitted for custom scale channels
        assert "units" not in ch

    def test_min_max_always_written(self, mock_system, mock_constants, tmp_path):
        """min_val/max_val are always written in TOML (read from nidaqmx channel).

        The new save_config() reads ai_rng_low/ai_rng_high from the nidaqmx
        channel object, which always has values after channel creation.
        When add_channel() is called without explicit min/max, the mock
        defaults to -5.0 and 5.0.
        """
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.save_config(path)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        ch = data["channels"][0]
        # min_val and max_val are always present (read from mock ai_rng_low/high)
        assert "min_val" in ch
        assert "max_val" in ch

    def test_header_comment_with_timestamp(self, mock_system, mock_constants, tmp_path):
        """save_config() includes header comment with version and timestamp."""
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.save_config(path)

        content = path.read_text()
        lines = content.splitlines()
        assert len(lines) > 0
        assert lines[0].startswith("# Generated by nidaqwrapper 0.1.0 on")

        # Verify from_config() can still parse it (round-trip)
        system2 = mock_system(task_names=[])
        mock_ni_task2 = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system2),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task2),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task2 = AITask.from_config(path)
            assert task2.task_name == "test"

    def test_device_product_type_comments(self, mock_system, mock_constants, tmp_path):
        """save_config() annotates device lines with product type comments."""
        ctx, task, mt = _build(mock_system, mock_constants)
        path = tmp_path / "config.toml"
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.save_config(path)

        content = path.read_text()
        assert "# NI 9234" in content


class TestFromConfig:
    """from_config() creates an AITask from a TOML file."""

    def _write_config(self, tmp_path, content):
        """Write a TOML string to a temp file and return the path."""
        path = tmp_path / "config.toml"
        path.write_text(content)
        return path

    def test_creates_task_with_name(self, mock_system, mock_constants, tmp_path):
        """from_config() creates a task with the name from [task] section."""
        path = self._write_config(tmp_path, """\
[task]
name = "vibration_test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task) as mock_cls,
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_config(path)

        mock_cls.assert_called_once_with(new_task_name="vibration_test")
        assert task.sample_rate == 25600

    def test_resolves_device_alias(self, mock_system, mock_constants, tmp_path):
        """from_config() resolves device alias to device index."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod2 = "cDAQ1Mod2"

[[channels]]
name = "accel_x"
device = "mod2"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_config(path)

        # cDAQ1Mod2 is device="cDAQ1Mod2", so physical channel should be cDAQ1Mod2/ai0
        kwargs = mock_ni_task.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "cDAQ1Mod2/ai0"

    def test_multi_device_channels(self, mock_system, mock_constants, tmp_path):
        """from_config() handles channels on different devices."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"
mod2 = "cDAQ1Mod2"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"

[[channels]]
name = "accel_y"
device = "mod2"
channel = 1
sensitivity = 50.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_config(path)

        assert mock_ni_task.ai_channels.add_ai_accel_chan.call_count == 2
        calls = mock_ni_task.ai_channels.add_ai_accel_chan.call_args_list
        phys_channels = {c.kwargs["physical_channel"] for c in calls}
        assert "cDAQ1Mod1/ai0" in phys_channels
        assert "cDAQ1Mod2/ai1" in phys_channels

    def test_force_channel_from_config(self, mock_system, mock_constants, tmp_path):
        """from_config() handles force/IEPE channels."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "force_1"
device = "mod1"
channel = 0
sensitivity = 22.5
sensitivity_units = "mV/N"
units = "N"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_config(path)

        mock_ni_task.ai_channels.add_ai_force_iepe_chan.assert_called_once()
        kwargs = mock_ni_task.ai_channels.add_ai_force_iepe_chan.call_args.kwargs
        assert kwargs["sensitivity"] == 22.5

    def test_min_val_zero_from_config(self, mock_system, mock_constants, tmp_path):
        """from_config() preserves min_val=0.0 (not treated as falsy)."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
min_val = 0.0
max_val = 50.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_config(path)

        kwargs = mock_ni_task.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["min_val"] == 0.0
        assert kwargs["max_val"] == 50.0

    def test_voltage_channel_from_config(self, mock_system, mock_constants, tmp_path):
        """from_config() handles voltage channels (no sensitivity)."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "v1"
device = "mod1"
channel = 0
units = "V"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_config(path)

        mock_ni_task.ai_channels.add_ai_voltage_chan.assert_called_once()

    def test_channel_with_scale(self, mock_system, mock_constants, tmp_path):
        """from_config() handles channels with scale parameter."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "v1"
device = "mod1"
channel = 0
units = "V"
scale = [2500.0, -100.0]
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
            patch("nidaqwrapper.ai_task.nidaqmx.Scale.create_lin_scale") as ms,
        ):
            ms.return_value.name = "v1_scale"
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_config(path)

        ms.assert_called_once()
        scale_kwargs = ms.call_args.kwargs
        assert scale_kwargs["slope"] == 2500.0
        assert scale_kwargs["y_intercept"] == -100.0

    def test_channel_with_min_max(self, mock_system, mock_constants, tmp_path):
        """from_config() forwards min_val/max_val from TOML."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
min_val = -50.0
max_val = 50.0
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_config(path)

        kwargs = mock_ni_task.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["min_val"] == -50.0
        assert kwargs["max_val"] == 50.0

    def test_invalid_device_alias_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when channel references unknown alias."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "nonexistent_module"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(ValueError, match="alias|device"):
                AITask.from_config(path)

    def test_device_alias_passed_directly_to_add_channel(self, mock_system, mock_constants, tmp_path):
        """from_config() passes device name directly to add_channel (no pre-validation).

        Per design decision 3: from_config() no longer validates device names
        against the system device list. The device name from the [devices]
        alias is passed directly to add_channel(device=...). If the device
        does not exist, nidaqmx raises DaqError at channel-creation time.
        """
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[devices]
mod1 = "NonExistentDevice"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
sensitivity = 100.0
sensitivity_units = "mV/g"
units = "g"
""")
        system = mock_system(task_names=[])
        mock_ni_task = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            # from_config() does NOT pre-validate the device name;
            # it passes "NonExistentDevice" directly to add_channel().
            # The mock task accepts any device name, so no error is raised here.
            task = AITask.from_config(path)

        # Verify add_channel was called with device="NonExistentDevice"
        kwargs = mock_ni_task.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["physical_channel"] == "NonExistentDevice/ai0"

    def test_missing_task_section_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when [task] section is missing."""
        path = self._write_config(tmp_path, """\
[devices]
mod1 = "cDAQ1Mod1"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
units = "g"
""")
        system = mock_system(task_names=[])

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(ValueError, match="task"):
                AITask.from_config(path)

    def test_missing_devices_section_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises ValueError when [devices] section is missing."""
        path = self._write_config(tmp_path, """\
[task]
name = "test"
sample_rate = 25600
type = "input"

[[channels]]
name = "accel_x"
device = "mod1"
channel = 0
units = "g"
""")
        system = mock_system(task_names=[])

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(ValueError, match="devices"):
                AITask.from_config(path)

    def test_malformed_toml_raises(self, mock_system, mock_constants, tmp_path):
        """from_config() raises an error on syntactically invalid TOML."""
        path = self._write_config(tmp_path, "not = valid [ toml {\n")

        from nidaqwrapper.ai_task import AITask
        with pytest.raises(Exception):  # tomllib.TOMLDecodeError
            AITask.from_config(path)


class TestConfigRoundtrip:
    """save_config → from_config produces equivalent task configuration."""

    def test_roundtrip_accel(self, mock_system, mock_constants, tmp_path):
        """Accel channel survives a save/load roundtrip."""
        # Create and configure original task; save_config must run inside ctx
        # so that constants and UNITS_REVERSE are patched consistently with
        # the channel mock objects created by add_channel().
        path = tmp_path / "config.toml"
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
                min_val=-50.0, max_val=50.0,
            )
            task.save_config(path)

        # Load back via from_config
        system = mock_system(task_names=[])
        mock_ni_task2 = _make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.nidaqmx.task.Task",
                  return_value=mock_ni_task2) as mock_cls,
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            loaded = AITask.from_config(path)

        # Verify the loaded task matches the original
        mock_cls.assert_called_once_with(new_task_name="test")
        assert loaded.sample_rate == 25600
        mock_ni_task2.ai_channels.add_ai_accel_chan.assert_called_once()
        kwargs = mock_ni_task2.ai_channels.add_ai_accel_chan.call_args.kwargs
        assert kwargs["name_to_assign_to_channel"] == "accel_x"
        assert kwargs["physical_channel"] == "cDAQ1Mod1/ai0"
        assert kwargs["sensitivity"] == 100.0
        assert kwargs["min_val"] == -50.0
        assert kwargs["max_val"] == 50.0


# ===========================================================================
# Task Group 4: from_task() — External task injection
# ===========================================================================

class TestFromTask:
    """from_task() wraps a pre-created nidaqmx.Task for external configuration."""

    def test_creates_instance_without_init(self, mock_system, mock_constants):
        """from_task() creates an AITask instance without calling __init__."""
        system = mock_system(task_names=[])

        # Create a mock nidaqmx task with one AI channel
        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 51200
        mock_ni_task.timing.samp_quant_samp_mode = "CONTINUOUS"

        # Mock AI channel
        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]

        # Check if task is running (is_task_done property)
        mock_ni_task.is_task_done.return_value = True

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

        # Verify instance was created and has the task
        assert isinstance(task, AITask)
        assert task.task is mock_ni_task
        assert task._owns_task is False

    def test_populates_properties_from_task(self, mock_system, mock_constants):
        """from_task() reads and populates all instance attributes from the task."""
        system = mock_system(task_names=[])

        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 51200
        mock_ni_task.timing.samp_quant_samp_mode = mock_constants.AcquisitionType.CONTINUOUS

        mock_ch1 = MagicMock()
        mock_ch1.name = "accel_x"
        mock_ch2 = MagicMock()
        mock_ch2.name = "accel_y"
        mock_ni_task.ai_channels = [mock_ch1, mock_ch2]
        mock_ni_task.channel_names = ["accel_x", "accel_y"]
        mock_ni_task.is_task_done.return_value = True

        mock_dev1 = MagicMock()
        mock_dev1.name = "cDAQ1Mod1"
        mock_dev1.product_type = "NI 9234"
        mock_dev2 = MagicMock()
        mock_dev2.name = "cDAQ1Mod2"
        mock_dev2.product_type = "NI 9263"
        mock_ni_task.devices = [mock_dev1, mock_dev2]

        with (
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

        assert task.task_name == "external_task"
        assert task.sample_rate == 51200
        assert task.channel_list == ["accel_x", "accel_y"]
        assert task.number_of_ch == 2
        assert task.device_list == ["cDAQ1Mod1", "cDAQ1Mod2"]
        assert task.device_product_type == ["NI 9234", "NI 9263"]
        assert task.sample_mode == mock_constants.AcquisitionType.CONTINUOUS

    def test_validation_no_ai_channels_raises(self, mock_system, mock_constants):
        """from_task() raises ValueError when task has no AI channels."""
        system = mock_system(task_names=[])

        mock_ni_task = MagicMock()
        mock_ni_task.name = "empty_task"
        mock_ni_task.ai_channels = []

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(ValueError, match="no AI channels"):
                AITask.from_task(mock_ni_task)

    def test_warns_when_task_already_running(self, mock_system, mock_constants):
        """from_task() warns when wrapping a task that is already running."""
        system = mock_system(task_names=[])

        mock_ni_task = MagicMock()
        mock_ni_task.name = "running_task"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ni_task.timing.samp_quant_samp_mode = "CONTINUOUS"

        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]

        # Task is running (is_task_done returns False)
        mock_ni_task.is_task_done.return_value = False

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task = AITask.from_task(mock_ni_task)

            assert len(w) >= 1
            assert "already running" in str(w[0].message).lower()

    def test_add_channel_blocked_raises(self, mock_system, mock_constants):
        """add_channel() raises RuntimeError on externally-provided task."""
        system = mock_system(task_names=[])

        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ni_task.timing.samp_quant_samp_mode = "CONTINUOUS"

        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

            with pytest.raises(RuntimeError, match="Cannot add channels"):
                task.add_channel(
                    "new_ch", device="cDAQ1Mod1", channel_ind=1,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )

    def test_configure_blocked_raises(self, mock_system, mock_constants):
        """configure() raises RuntimeError on externally-provided task."""
        system = mock_system(task_names=[])

        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ni_task.timing.samp_quant_samp_mode = "CONTINUOUS"

        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

            with pytest.raises(RuntimeError, match="Cannot configure"):
                task.configure()

    def test_clear_task_does_not_close_warns(self, mock_system, mock_constants):
        """clear_task() does NOT close external task but warns."""
        system = mock_system(task_names=[])

        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ni_task.timing.samp_quant_samp_mode = "CONTINUOUS"

        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task.clear_task()

            # Should warn but NOT call task.close()
            mock_ni_task.close.assert_not_called()
            assert len(w) >= 1
            assert "externally" in str(w[0].message).lower()

    def test_exit_does_not_close_warns(self, mock_system, mock_constants):
        """__exit__ does NOT close external task but warns."""
        system = mock_system(task_names=[])

        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ni_task.timing.samp_quant_samp_mode = "CONTINUOUS"

        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task.__exit__(None, None, None)

            # Should warn but NOT call task.close()
            mock_ni_task.close.assert_not_called()
            assert len(w) >= 1
            assert "externally" in str(w[0].message).lower()

    def test_normal_constructor_owns_task(self, mock_system, mock_constants):
        """Normal constructor sets _owns_task=True and clear_task() closes."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        # Verify _owns_task is True
        assert task._owns_task is True

        # clear_task() should close the task
        task.clear_task()
        mt.close.assert_called_once()


class TestFromTaskTakeOwnership:
    """from_task(task, take_ownership=True) makes the wrapper own the task."""

    def _make_external_task(self, mock_system, mock_constants):
        """Create a mock nidaqmx.Task with one AI channel."""
        mock_ni_task = MagicMock()
        mock_ni_task.name = "external_task"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ni_task.timing.samp_quant_samp_mode = mock_constants.AcquisitionType.CONTINUOUS
        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ch.physical_channel = MagicMock()
        mock_ch.physical_channel.name = "Dev1/ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True
        return mock_ni_task

    def test_default_not_owned(self, mock_system, mock_constants):
        """from_task(task) with default take_ownership=False sets _owns_task=False."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_system, mock_constants)

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(ext)

        assert task._owns_task is False

    def test_take_ownership_sets_owns_task(self, mock_system, mock_constants):
        """from_task(task, take_ownership=True) sets _owns_task=True."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_system, mock_constants)

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(ext, take_ownership=True)

        assert task._owns_task is True

    def test_add_channel_allowed_when_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=True).add_channel() does not raise."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_system, mock_constants)
        # Set up ai_channels as a MagicMock that supports iteration
        ai_channels_mock = MagicMock()
        ai_channels_mock.__iter__ = MagicMock(return_value=iter([]))
        ai_channels_mock.__len__ = MagicMock(return_value=1)
        ext.ai_channels = ai_channels_mock
        ext.channel_names = []

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.UNITS_REVERSE", MOCK_UNITS_REVERSE),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(ext, take_ownership=True)
            # Should not raise RuntimeError
            task.add_channel(
                "new_ch", device="cDAQ1Mod1", channel_ind=1,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )

    def test_configure_allowed_when_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=True).configure() does not raise RuntimeError."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_system, mock_constants)
        # Set up timing mock to return requested rate unchanged
        ext.timing.samp_clk_rate = 25600

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(ext, take_ownership=True)
            # configure() should not raise RuntimeError (ownership check passes)
            task.configure()

    def test_clear_task_closes_when_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=True).clear_task() calls task.close()."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_system, mock_constants)

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(ext, take_ownership=True)
            task.clear_task()

        ext.close.assert_called_once()

    def test_add_channel_blocked_when_not_owned(self, mock_system, mock_constants):
        """from_task(task, take_ownership=False).add_channel() raises RuntimeError."""
        system = mock_system(task_names=[])
        ext = self._make_external_task(mock_system, mock_constants)

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(ext, take_ownership=False)

            with pytest.raises(RuntimeError, match="Cannot add channels"):
                task.add_channel(
                    "new_ch", device="cDAQ1Mod1", channel_ind=1,
                    sensitivity=100.0, sensitivity_units="mV/g", units="g",
                )


class TestFromName:
    """from_name() loads an NI MAX task by name and wraps it as an AITask."""

    def _make_mock_ni_task(self):
        """Create a mock nidaqmx task with one AI channel."""
        mock_ni_task = MagicMock()
        mock_ni_task.name = "MaxTask"
        mock_ni_task.timing.samp_clk_rate = 25600
        mock_ni_task.timing.samp_quant_samp_mode = "CONTINUOUS"
        mock_ch = MagicMock()
        mock_ch.name = "ai0"
        mock_ni_task.ai_channels = [mock_ch]
        mock_ni_task.channel_names = ["ai0"]
        mock_ni_task.is_task_done.return_value = True
        return mock_ni_task

    def test_loads_and_wraps_successfully(self, mock_system, mock_constants):
        """from_name() loads the NI MAX task and returns an AITask."""
        system = mock_system(task_names=[])
        mock_ni_task = self._make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.base_task.get_task_by_name",
                  return_value=mock_ni_task) as mock_get,
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_name("MaxTask")

        mock_get.assert_called_once_with("MaxTask")
        assert isinstance(task, AITask)
        assert task.task is mock_ni_task
        assert task.task_name == "MaxTask"

    def test_owns_task(self, mock_system, mock_constants):
        """from_name() sets _owns_task=True so cleanup closes the task."""
        system = mock_system(task_names=[])
        mock_ni_task = self._make_mock_ni_task()

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.base_task.get_task_by_name",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_name("MaxTask")

        assert task._owns_task is True

    def test_task_not_found_raises_keyerror(self, mock_system, mock_constants):
        """from_name() raises KeyError when task name not in NI MAX."""
        with (
            patch("nidaqwrapper.base_task.get_task_by_name",
                  side_effect=KeyError("No task named 'Missing'")),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(KeyError, match="Missing"):
                AITask.from_name("Missing")

    def test_task_already_loaded_raises_runtime_error(
        self, mock_system, mock_constants
    ):
        """from_name() raises RuntimeError when get_task_by_name returns None."""
        with (
            patch("nidaqwrapper.base_task.get_task_by_name",
                  return_value=None),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(RuntimeError, match="already loaded"):
                AITask.from_name("BusyTask")

    def test_device_disconnected_raises_connection_error(
        self, mock_system, mock_constants
    ):
        """from_name() propagates ConnectionError from get_task_by_name."""
        with (
            patch("nidaqwrapper.base_task.get_task_by_name",
                  side_effect=ConnectionError("Device disconnected")),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(ConnectionError, match="disconnected"):
                AITask.from_name("BadDevice")

    def test_no_ai_channels_raises_value_error(self, mock_system, mock_constants):
        """from_name() raises ValueError when loaded task has no AI channels."""
        system = mock_system(task_names=[])
        mock_ni_task = MagicMock()
        mock_ni_task.name = "ao_only"
        mock_ni_task.ai_channels = []  # No AI channels

        with (
            patch("nidaqwrapper.ai_task.nidaqmx.system.System.local",
                  return_value=system),
            patch("nidaqwrapper.base_task.get_task_by_name",
                  return_value=mock_ni_task),
            patch("nidaqwrapper.ai_task.UNITS", MOCK_UNITS),
            patch("nidaqwrapper.ai_task.constants", mock_constants),
        ):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(ValueError, match="no AI channels"):
                AITask.from_name("ao_only")

    def test_raises_without_nidaqmx(self):
        """from_name() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.ai_task import AITask
            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                AITask.from_name("AnyTask")


# ===========================================================================
# task-sync-configuration: set_start_trigger() (inherited from BaseTask)
# ===========================================================================

def _make_external_ai_task(mock_constants):
    """Create a minimal external mock nidaqmx task for from_task() tests."""
    mock_ni_task = MagicMock()
    mock_ni_task.name = "external_task"
    mock_ni_task.timing.samp_clk_rate = 25600
    mock_ni_task.timing.samp_quant_samp_mode = mock_constants.AcquisitionType.CONTINUOUS
    mock_ni_task.timing.samp_clk_src = ""

    mock_ch = MagicMock()
    mock_ch.name = "ai0"
    mock_ni_task.ai_channels = [mock_ch]
    mock_ni_task.channel_names = ["ai0"]
    mock_ni_task.is_task_done.return_value = True

    # Pitfall 11: from_task() populates device info from task.devices
    mock_dev = MagicMock()
    mock_dev.name = "cDAQ1Mod1"
    mock_dev.product_type = "NI 9234"
    mock_ni_task.devices = [mock_dev]
    return mock_ni_task


class TestSetStartTrigger:
    """set_start_trigger() configures a digital edge start trigger."""

    def test_default_rising_edge(self, mock_system, mock_constants):
        """Default edge='rising' maps to constants.Edge.RISING."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.set_start_trigger("/Dev1/PFI0")

        trig = mt.triggers.start_trigger.cfg_dig_edge_start_trig
        trig.assert_called_once()
        kwargs = trig.call_args.kwargs
        assert kwargs["trigger_source"] == "/Dev1/PFI0"
        assert kwargs["trigger_edge"] is mock_constants.Edge.RISING

    def test_falling_edge(self, mock_system, mock_constants):
        """edge='falling' maps to constants.Edge.FALLING."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.set_start_trigger("/Dev1/PFI1", edge="falling")

        trig = mt.triggers.start_trigger.cfg_dig_edge_start_trig
        trig.assert_called_once()
        kwargs = trig.call_args.kwargs
        assert kwargs["trigger_source"] == "/Dev1/PFI1"
        assert kwargs["trigger_edge"] is mock_constants.Edge.FALLING

    def test_invalid_edge_raises(self, mock_system, mock_constants):
        """An edge string other than 'rising'/'falling' raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="edge"):
                task.set_start_trigger("/Dev1/PFI0", edge="both")

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_empty_source_raises(self, mock_system, mock_constants):
        """An empty source string raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                task.set_start_trigger("")

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_non_string_source_raises(self, mock_system, mock_constants):
        """A non-string source raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                task.set_start_trigger(123)

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_none_source_raises(self, mock_system, mock_constants):
        """source=None raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                task.set_start_trigger(None)

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_not_owned_raises(self, mock_system, mock_constants):
        """A task wrapped via from_task() (not owned) raises RuntimeError."""
        mock_ni_task = _make_external_ai_task(mock_constants)

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

            with pytest.raises(RuntimeError, match="externally-provided"):
                task.set_start_trigger("/Dev1/PFI0")

        mock_ni_task.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

    def test_works_without_channels(self, mock_system, mock_constants):
        """Design decision 3: no ordering constraint enforced — trigger config
        is independent of channels/timing, so no channel gate applies."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            # No add_channel() call — trigger configuration must still work
            task.set_start_trigger("/Dev1/PFI0")

        mt.triggers.start_trigger.cfg_dig_edge_start_trig.assert_called_once()


# ===========================================================================
# task-sync-configuration: configure() sync extensions
# ===========================================================================

class TestConfigureSyncExtensions:
    """configure(sample_mode=, samples_per_channel=, clock_source=) extensions."""

    def _add_channel(self, task):
        task.add_channel(
            "accel_x", device="cDAQ1Mod1", channel_ind=0,
            sensitivity=100.0, sensitivity_units="mV/g", units="g",
        )
        # add_channel() applies timing eagerly (fix-gh-issues-5-8); reset
        # the timing mock so each test asserts configure()'s calls only.
        task.task.timing.cfg_samp_clk_timing.reset_mock()

    def test_finite_mode_maps_to_enum(self, mock_system, mock_constants):
        """sample_mode='finite' maps to AcquisitionType.FINITE with samps_per_chan."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            task.configure(sample_mode="finite", samples_per_channel=1000)

        mt.timing.cfg_samp_clk_timing.assert_called_once()
        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["sample_mode"] is mock_constants.AcquisitionType.FINITE
        assert kwargs["samps_per_chan"] == 1000
        assert kwargs["rate"] == 25600

    def test_finite_without_samples_per_channel_raises(self, mock_system, mock_constants):
        """sample_mode='finite' without samples_per_channel raises ValueError;
        task timing is not changed."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            with pytest.raises(ValueError, match="samples_per_channel"):
                task.configure(sample_mode="finite")

        mt.timing.cfg_samp_clk_timing.assert_not_called()

    def test_invalid_sample_mode_raises(self, mock_system, mock_constants):
        """An unknown sample_mode string raises ValueError; timing unchanged."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            with pytest.raises(ValueError, match="sample_mode"):
                task.configure(sample_mode="single")

        mt.timing.cfg_samp_clk_timing.assert_not_called()

    # True is included deliberately: isinstance(True, int) holds in Python,
    # so bools must be rejected explicitly by the implementation.
    @pytest.mark.parametrize("bad_count", [0, -5, 2.5, "100", True])
    def test_invalid_samples_per_channel_raises(self, mock_system, mock_constants,
                                                bad_count):
        """samples_per_channel must be a positive int (bool excluded)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            with pytest.raises(ValueError, match="samples_per_channel"):
                task.configure(sample_mode="finite", samples_per_channel=bad_count)

        mt.timing.cfg_samp_clk_timing.assert_not_called()

    def test_clock_source_passed_as_source(self, mock_system, mock_constants):
        """A non-empty clock_source is forwarded as source= to cfg_samp_clk_timing."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            task.configure(clock_source="/cDAQ1Mod1/ai/SampleClock")

        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["source"] == "/cDAQ1Mod1/ai/SampleClock"

    def test_default_call_shape_unchanged(self, mock_system, mock_constants):
        """configure() with defaults omits source and samps_per_chan entirely."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            task.configure()

        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert "source" not in kwargs
        assert "samps_per_chan" not in kwargs
        assert kwargs["sample_mode"] is mock_constants.AcquisitionType.CONTINUOUS

    def test_external_clock_skips_rate_coercion_check(self, mock_system, mock_constants):
        """With clock_source set, a differing reported rate does NOT raise."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25000)
        with ctx:
            self._add_channel(task)
            task.configure(clock_source="/cDAQ1Mod1/ai/SampleClock")  # no raise

    def test_internal_clock_keeps_rate_coercion_check(self, mock_system, mock_constants):
        """Without clock_source the rate-coercion ValueError stays active."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25000)
        with ctx:
            self._add_channel(task)
            with pytest.raises(ValueError, match="rate"):
                task.configure(sample_mode="finite", samples_per_channel=100)

    def test_continuous_with_samples_per_channel_allowed(self, mock_system,
                                                         mock_constants):
        """samples_per_channel with 'continuous' is a buffer-size hint (allowed)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            task.configure(samples_per_channel=50000)

        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["sample_mode"] is mock_constants.AcquisitionType.CONTINUOUS
        assert kwargs["samps_per_chan"] == 50000

    def test_all_parameters_compose(self, mock_system, mock_constants):
        """All three extensions compose in a single configure() call."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            task.configure(sample_mode="continuous", samples_per_channel=5000,
                           clock_source="/Dev1/ai/SampleClock")

        kwargs = mt.timing.cfg_samp_clk_timing.call_args.kwargs
        assert kwargs["sample_mode"] is mock_constants.AcquisitionType.CONTINUOUS
        assert kwargs["samps_per_chan"] == 5000
        assert kwargs["source"] == "/Dev1/ai/SampleClock"

    @pytest.mark.parametrize("bad_source", ["", 42, ["/Dev1/PFI0"]])
    def test_invalid_clock_source_raises(self, mock_system, mock_constants,
                                         bad_source):
        """clock_source must be a non-empty string (or None)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            with pytest.raises(ValueError, match="clock_source"):
                task.configure(clock_source=bad_source)

        mt.timing.cfg_samp_clk_timing.assert_not_called()

    def test_stores_introspection_attributes(self, mock_system, mock_constants):
        """configure() stores sample_mode_str, samples_per_channel, clock_source."""
        ctx, task, mt = _build(mock_system, mock_constants,
                               sample_rate=25600, samp_clk_rate=25000)
        with ctx:
            self._add_channel(task)
            task.configure(sample_mode="finite", samples_per_channel=500,
                           clock_source="/cDAQ1Mod1/ai/SampleClock")

        assert task.sample_mode_str == "finite"
        assert task.samples_per_channel == 500
        assert task.clock_source == "/cDAQ1Mod1/ai/SampleClock"

    def test_default_attributes_after_configure(self, mock_system, mock_constants):
        """Plain configure() stores the default attribute values."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            self._add_channel(task)
            task.configure()

        assert task.sample_mode_str == "continuous"
        assert task.samples_per_channel is None
        assert task.clock_source is None

    def test_attributes_initialized_at_construction(self, mock_system, mock_constants):
        """New attributes exist with defaults before configure() is called."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            pass

        assert task.sample_mode_str == "continuous"
        assert task.samples_per_channel is None
        assert task.clock_source is None


# ===========================================================================
# task-sync-configuration: set_analog_start_trigger()
# ===========================================================================

class TestSetAnalogStartTrigger:
    """set_analog_start_trigger() configures an analog edge start trigger."""

    def test_default_rising_slope_level_zero(self, mock_system, mock_constants):
        """Defaults: slope='rising' -> Slope.RISING, level 0.0."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.set_analog_start_trigger("APFI0")

        trig = mt.triggers.start_trigger.cfg_anlg_edge_start_trig
        trig.assert_called_once()
        kwargs = trig.call_args.kwargs
        assert kwargs["trigger_source"] == "APFI0"
        assert kwargs["trigger_slope"] is mock_constants.Slope.RISING
        assert kwargs["trigger_level"] == 0.0

    def test_falling_slope_with_level(self, mock_system, mock_constants):
        """slope='falling' maps to Slope.FALLING; level is forwarded."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.set_analog_start_trigger("APFI0", slope="falling", level=2.5)

        kwargs = mt.triggers.start_trigger.cfg_anlg_edge_start_trig.call_args.kwargs
        assert kwargs["trigger_slope"] is mock_constants.Slope.FALLING
        assert kwargs["trigger_level"] == 2.5

    def test_level_coerced_to_float(self, mock_system, mock_constants):
        """An int level is coerced to float before the driver call."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.set_analog_start_trigger("APFI0", level=3)

        level = mt.triggers.start_trigger.cfg_anlg_edge_start_trig.call_args.kwargs[
            "trigger_level"
        ]
        assert isinstance(level, float)
        assert level == 3.0

    def test_invalid_slope_raises(self, mock_system, mock_constants):
        """A slope other than 'rising'/'falling' raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="slope"):
                task.set_analog_start_trigger("APFI0", slope="sideways")

        mt.triggers.start_trigger.cfg_anlg_edge_start_trig.assert_not_called()

    @pytest.mark.parametrize("bad_source", ["", 0, None])
    def test_invalid_source_raises(self, mock_system, mock_constants, bad_source):
        """Empty/non-string source raises ValueError."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            with pytest.raises(ValueError, match="source"):
                task.set_analog_start_trigger(bad_source)

        mt.triggers.start_trigger.cfg_anlg_edge_start_trig.assert_not_called()

    def test_not_owned_raises(self, mock_system, mock_constants):
        """A task wrapped via from_task() (not owned) raises RuntimeError."""
        mock_ni_task = _make_external_ai_task(mock_constants)

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

            with pytest.raises(RuntimeError, match="externally-provided"):
                task.set_analog_start_trigger("APFI0")

        mock_ni_task.triggers.start_trigger.cfg_anlg_edge_start_trig.assert_not_called()


# ===========================================================================
# task-sync-configuration: finite blocking-read contract for acquire()
# ===========================================================================

class TestFiniteAcquireContract:
    """acquire() on a finite task reads with READ_ALL_AVAILABLE (-1), which in
    nidaqmx blocks until the finite acquisition completes — the hardware-trigger
    synchronization mechanism."""

    def test_finite_acquire_reads_all_available(self, mock_system, mock_constants):
        """After finite configure, acquire() calls read(-1) and returns
        (n_samples, n_channels)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.add_channel(
                "accel_y", device="cDAQ1Mod1", channel_ind=1,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.configure(sample_mode="finite", samples_per_channel=4)

        # nidaqmx returns (n_channels, n_samples) — full finite acquisition
        mt.read.return_value = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]

        result = task.acquire()

        mt.read.assert_called_once_with(number_of_samples_per_channel=-1)
        assert result.shape == (4, 2)

    def test_finite_acquire_single_channel(self, mock_system, mock_constants):
        """Finite single-channel acquisition returns (n_samples, 1)."""
        ctx, task, mt = _build(mock_system, mock_constants)
        with ctx:
            task.add_channel(
                "accel_x", device="cDAQ1Mod1", channel_ind=0,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            task.configure(sample_mode="finite", samples_per_channel=4)

        mt.read.return_value = [1.0, 2.0, 3.0, 4.0]

        result = task.acquire()

        mt.read.assert_called_once_with(number_of_samples_per_channel=-1)
        assert result.shape == (4, 1)


# ===========================================================================
# task-sync-configuration: from_task() populates sync attributes
# ===========================================================================

class TestFromTaskSyncAttributes:
    """from_task() derives sample_mode_str / samples_per_channel / clock_source."""

    def test_finite_external_task(self, mock_system, mock_constants):
        """A FINITE external task yields 'finite' + samples count + clock source."""
        mock_ni_task = _make_external_ai_task(mock_constants)
        mock_ni_task.timing.samp_quant_samp_mode = mock_constants.AcquisitionType.FINITE
        mock_ni_task.timing.samp_quant_samp_per_chan = 1000
        mock_ni_task.timing.samp_clk_src = "/cDAQ1Mod1/ai/SampleClock"

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

        assert task.sample_mode_str == "finite"
        assert task.samples_per_channel == 1000
        assert task.clock_source == "/cDAQ1Mod1/ai/SampleClock"

    def test_continuous_external_task(self, mock_system, mock_constants):
        """A CONTINUOUS external task yields 'continuous' and no samples count."""
        mock_ni_task = _make_external_ai_task(mock_constants)
        mock_ni_task.timing.samp_clk_src = "OnboardClock"

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

        assert task.sample_mode_str == "continuous"
        assert task.samples_per_channel is None
        assert task.clock_source == "OnboardClock"

    def test_empty_clock_source_maps_to_none(self, mock_system, mock_constants):
        """An empty samp_clk_src string is stored as None."""
        mock_ni_task = _make_external_ai_task(mock_constants)
        mock_ni_task.timing.samp_clk_src = ""

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

        assert task.clock_source is None

    def test_unexpected_sample_mode_defaults_to_continuous(self, mock_system,
                                                           mock_constants):
        """An unrecognized samp_quant_samp_mode value yields the safe default."""
        mock_ni_task = _make_external_ai_task(mock_constants)
        mock_ni_task.timing.samp_quant_samp_mode = "HW_TIMED_SINGLE_POINT"

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            task = AITask.from_task(mock_ni_task)

        assert task.sample_mode_str == "continuous"
        assert task.samples_per_channel is None

    def test_samples_per_channel_failure_warns_and_defaults(self, mock_system,
                                                            mock_constants):
        """A failing samp_quant_samp_per_chan read warns and stores None."""
        from unittest.mock import PropertyMock

        mock_ni_task = _make_external_ai_task(mock_constants)
        mock_ni_task.timing.samp_quant_samp_mode = mock_constants.AcquisitionType.FINITE
        type(mock_ni_task.timing).samp_quant_samp_per_chan = PropertyMock(
            side_effect=RuntimeError("driver error")
        )

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task = AITask.from_task(mock_ni_task)

        assert task.samples_per_channel is None
        assert any("samples" in str(x.message).lower() or
                   "finite" in str(x.message).lower() for x in w)

    def test_clock_source_failure_warns_and_defaults(self, mock_system,
                                                     mock_constants):
        """A failing samp_clk_src read warns and stores None."""
        from unittest.mock import PropertyMock

        mock_ni_task = _make_external_ai_task(mock_constants)
        type(mock_ni_task.timing).samp_clk_src = PropertyMock(
            side_effect=RuntimeError("driver error")
        )

        with patch("nidaqwrapper.ai_task.constants", mock_constants):
            from nidaqwrapper.ai_task import AITask
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task = AITask.from_task(mock_ni_task)

        assert task.clock_source is None
        assert any("clock" in str(x.message).lower() for x in w)
