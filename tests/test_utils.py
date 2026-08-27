"""Tests for nidaqwrapper.utils module.

Notes
-----
nidaqmx 1.4.1 (installed) does not have
``AccelSensitivityUnits.M_VOLTS_PER_METERS_PER_SECOND_SQUARED``.
The enum only contains ``MILLIVOLTS_PER_G`` and ``VOLTS_PER_G``.

The intended bug fix (mapping ``mV/m/s**2`` to a dedicated m/s² sensitivity
constant) cannot be implemented until NI ships that constant in nidaqmx.
Until then, ``mV/m/s**2`` maps to ``MILLIVOLTS_PER_G`` — matching LDAQ behaviour
— with a TODO marking the correct fix for a future nidaqmx release.

See agent note: .claude-notes/agent-notes/units-nidaqmx-compat.md
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestUNITS:
    """Tests for the UNITS dictionary."""

    def test_units_has_exactly_seven_keys(self):
        """UNITS contains exactly 7 unit mappings."""
        from nidaqwrapper.utils import UNITS

        assert len(UNITS) == 7

    def test_units_keys(self):
        """UNITS contains all expected string keys."""
        from nidaqwrapper.utils import UNITS

        expected_keys = {"mV/g", "mV/m/s**2", "g", "m/s**2", "mV/N", "N", "V"}
        assert set(UNITS.keys()) == expected_keys

    def test_units_mV_per_g(self):
        """mV/g maps to AccelSensitivityUnits.MILLIVOLTS_PER_G."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["mV/g"] == constants.AccelSensitivityUnits.MILLIVOLTS_PER_G

    def test_units_mV_per_ms2_is_accel_sensitivity_unit(self):
        """mV/m/s**2 maps to an AccelSensitivityUnits member.

        Notes
        -----
        nidaqmx 1.4.1 has no ``M_VOLTS_PER_METERS_PER_SECOND_SQUARED`` constant.
        The intended fix (distinct m/s² sensitivity unit) is deferred until NI
        ships the constant.  This test ensures the key exists and resolves to
        a valid ``AccelSensitivityUnits`` member, not some arbitrary value.
        """
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["mV/m/s**2"] in list(constants.AccelSensitivityUnits)

    def test_units_g(self):
        """g maps to AccelUnits.G."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["g"] == constants.AccelUnits.G

    def test_units_ms2(self):
        """m/s**2 maps to AccelUnits.METERS_PER_SECOND_SQUARED."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["m/s**2"] == constants.AccelUnits.METERS_PER_SECOND_SQUARED

    def test_units_mV_per_N(self):
        """mV/N maps to ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert (
            UNITS["mV/N"]
            == constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON
        )

    def test_units_N(self):
        """N maps to ForceUnits.NEWTONS."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["N"] == constants.ForceUnits.NEWTONS

    def test_units_V(self):
        """V maps to VoltageUnits.VOLTS."""
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS

        assert UNITS["V"] == constants.VoltageUnits.VOLTS

    def test_units_invalid_key_raises_keyerror(self):
        """Accessing an invalid key raises KeyError."""
        from nidaqwrapper.utils import UNITS

        with pytest.raises(KeyError):
            _ = UNITS["invalid_unit"]


class TestListDevices:
    """Tests for list_devices() function."""

    def test_list_devices_returns_list(self, mock_system):
        """list_devices() returns a list."""
        system = mock_system()
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert isinstance(result, list)

    def test_list_devices_two_devices(self, mock_system):
        """list_devices() returns correct dicts for two connected devices."""
        system = mock_system(
            devices=[("cDAQ1Mod1", "NI 9234"), ("cDAQ1Mod2", "NI 9263")]
        )
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert len(result) == 2
            assert result[0] == {"name": "cDAQ1Mod1", "product_type": "NI 9234"}
            assert result[1] == {"name": "cDAQ1Mod2", "product_type": "NI 9263"}

    def test_list_devices_no_devices(self, mock_system):
        """list_devices() returns empty list when no devices are connected."""
        system = mock_system(devices=[])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert result == []

    def test_list_devices_dict_keys(self, mock_system):
        """Each device dict has exactly 'name' and 'product_type' keys."""
        system = mock_system(devices=[("Dev1", "NI 9234")])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert set(result[0].keys()) == {"name", "product_type"}

    def test_list_devices_dict_values_are_strings(self, mock_system):
        """Device dict values are strings."""
        system = mock_system(devices=[("Dev1", "NI 9234")])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_devices

            result = list_devices()
            assert isinstance(result[0]["name"], str)
            assert isinstance(result[0]["product_type"], str)

    def test_list_devices_raises_without_nidaqmx(self):
        """list_devices() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.utils import list_devices

            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                list_devices()


class TestListTasks:
    """Tests for list_tasks() function."""

    def test_list_tasks_returns_list(self, mock_system):
        """list_tasks() returns a list."""
        system = mock_system(task_names=["MyTask"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_tasks

            result = list_tasks()
            assert isinstance(result, list)

    def test_list_tasks_two_tasks(self, mock_system):
        """list_tasks() returns the correct names for two saved tasks."""
        system = mock_system(task_names=["MyInputTask", "MyOutputTask"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_tasks

            result = list_tasks()
            assert result == ["MyInputTask", "MyOutputTask"]

    def test_list_tasks_no_tasks(self, mock_system):
        """list_tasks() returns an empty list when no tasks are saved."""
        system = mock_system(task_names=[])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_tasks

            result = list_tasks()
            assert result == []

    def test_list_tasks_return_type_is_list_of_strings(self, mock_system):
        """Every element returned by list_tasks() is a str."""
        system = mock_system(task_names=["TaskA", "TaskB"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import list_tasks

            result = list_tasks()
            assert all(isinstance(name, str) for name in result)

    def test_list_tasks_raises_without_nidaqmx(self):
        """list_tasks() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.utils import list_tasks

            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                list_tasks()


class TestGetTaskByName:
    """Tests for get_task_by_name() function."""

    def test_get_task_by_name_success(self, mock_system):
        """get_task_by_name() returns the loaded task object on success."""
        system = mock_system(task_names=["MyTask"])
        loaded_task = system.tasks.__iter__().__next__().load.return_value
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            result = get_task_by_name("MyTask")
            assert result is loaded_task

    def test_get_task_by_name_not_found_raises_keyerror(self, mock_system):
        """get_task_by_name() raises KeyError when no task matches the name."""
        system = mock_system(task_names=["Other"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(KeyError) as exc_info:
                get_task_by_name("NonExistent")

            error_message = str(exc_info.value)
            assert "NonExistent" in error_message
            assert "Other" in error_message

    def test_get_task_by_name_error_200089_returns_none(self, mock_system):
        """get_task_by_name() returns None when error code -200089 is raised.

        Error -200089 means the task is already loaded elsewhere; the function
        swallows the exception and returns None.
        """
        from nidaqmx.errors import DaqError

        system = mock_system(task_names=["MyTask"])
        # Retrieve the pre-built mock task from the tasks collection
        mock_task_obj = list(system.tasks)[0]
        mock_task_obj.load.side_effect = DaqError("Task already loaded", -200089)

        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            result = get_task_by_name("MyTask")
            assert result is None

    def test_get_task_by_name_error_201003_raises_connection_error(
        self, mock_system
    ):
        """get_task_by_name() raises ConnectionError when error code -201003 is raised.

        Error -201003 means the device is disconnected or in use by another
        application.
        """
        from nidaqmx.errors import DaqError

        system = mock_system(task_names=["MyTask"])
        mock_task_obj = list(system.tasks)[0]
        mock_task_obj.load.side_effect = DaqError("Device disconnected", -201003)

        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(ConnectionError) as exc_info:
                get_task_by_name("MyTask")

            error_message = str(exc_info.value)
            assert "disconnected" in error_message or "in use" in error_message

    def test_get_task_by_name_other_error_propagates(self, mock_system):
        """get_task_by_name() re-raises DaqError for unrecognised error codes."""
        from nidaqmx.errors import DaqError

        system = mock_system(task_names=["MyTask"])
        mock_task_obj = list(system.tasks)[0]
        original_error = DaqError("Unknown error", -99999)
        mock_task_obj.load.side_effect = original_error

        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(DaqError) as exc_info:
                get_task_by_name("MyTask")

            assert exc_info.value is original_error

    def test_get_task_by_name_raises_without_nidaqmx(self):
        """get_task_by_name() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                get_task_by_name("AnyTask")

    def test_get_task_by_name_empty_name_raises_keyerror(self, mock_system):
        """get_task_by_name('') raises KeyError — empty string is not a valid name."""
        system = mock_system(task_names=["RealTask"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_task_by_name

            with pytest.raises(KeyError):
                get_task_by_name("")


class TestGetConnectedDevices:
    """Tests for get_connected_devices() function."""

    def test_get_connected_devices_two_devices(self, mock_system):
        """get_connected_devices() returns the name set for two connected devices."""
        system = mock_system(
            devices=[("cDAQ1Mod1", "NI 9234"), ("cDAQ1Mod2", "NI 9263")]
        )
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_connected_devices

            result = get_connected_devices()
            assert result == {"cDAQ1Mod1", "cDAQ1Mod2"}

    def test_get_connected_devices_no_devices(self, mock_system):
        """get_connected_devices() returns an empty set when no devices are connected."""
        system = mock_system(devices=[])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_connected_devices

            result = get_connected_devices()
            assert result == set()

    def test_get_connected_devices_return_type_is_set(self, mock_system):
        """get_connected_devices() always returns a set, never a list or tuple."""
        system = mock_system(devices=[("Dev1", "NI 9234")])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_connected_devices

            result = get_connected_devices()
            assert isinstance(result, set)

    def test_get_connected_devices_values_are_strings(self, mock_system):
        """Every element in the returned set is a str."""
        system = mock_system(
            devices=[("cDAQ1Mod1", "NI 9234"), ("cDAQ1Mod2", "NI 9263")]
        )
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import get_connected_devices

            result = get_connected_devices()
            assert all(isinstance(name, str) for name in result)

    def test_get_connected_devices_raises_without_nidaqmx(self):
        """get_connected_devices() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.utils import get_connected_devices

            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                get_connected_devices()


# ---------------------------------------------------------------------------
# Helper for system_info mock devices
# ---------------------------------------------------------------------------

def _make_phys_chan(name: str) -> MagicMock:
    """Create a mock PhysicalChannel with a .name attribute."""
    chan = MagicMock()
    chan.name = name
    return chan


def _make_rich_device(
    name: str,
    product_type: str,
    serial_num: int,
    ai_chans: list[str] | None = None,
    ao_chans: list[str] | None = None,
) -> MagicMock:
    """Create a mock device with serial number and physical channels."""
    dev = MagicMock()
    dev.name = name
    dev.product_type = product_type
    dev.dev_serial_num = serial_num
    dev.ai_physical_chans = [_make_phys_chan(c) for c in (ai_chans or [])]
    dev.ao_physical_chans = [_make_phys_chan(c) for c in (ao_chans or [])]
    return dev


def _make_rich_system(devices=None, task_names=None):
    """Create a mock system with rich device info for system_info() tests."""
    system = MagicMock()
    system.devices = devices or []

    tasks_collection = MagicMock()
    tasks_collection.task_names = task_names or []
    system.tasks = tasks_collection

    return system


class TestSystemInfo:
    """Tests for system_info() function."""

    def test_returns_dict(self):
        """system_info() returns a dict."""
        system = _make_rich_system()
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            result = system_info()
            assert isinstance(result, dict)

    def test_has_devices_and_tasks_keys(self):
        """Returned dict has exactly 'devices' and 'tasks' keys."""
        system = _make_rich_system()
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            result = system_info()
            assert set(result.keys()) == {"devices", "tasks"}

    def test_device_dict_keys(self):
        """Each device dict has expected keys."""
        dev = _make_rich_device(
            "Dev1", "PCIe-6320", 0x01B4C8A0,
            ai_chans=["Dev1/ai0"], ao_chans=["Dev1/ao0"],
        )
        system = _make_rich_system(devices=[dev])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            result = system_info()
            device = result["devices"][0]
            assert set(device.keys()) == {
                "name", "product_type", "serial_number",
                "ai_channels", "ao_channels",
            }

    def test_device_name_and_product_type(self):
        """Device name and product_type are correctly populated."""
        dev = _make_rich_device("cDAQ1Mod1", "NI 9234", 0xABCD)
        system = _make_rich_system(devices=[dev])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            device = system_info()["devices"][0]
            assert device["name"] == "cDAQ1Mod1"
            assert device["product_type"] == "NI 9234"

    def test_serial_number_is_hex_string(self):
        """Serial number is returned as a hex string."""
        dev = _make_rich_device("Dev1", "PCIe-6320", 0x01B4C8A0)
        system = _make_rich_system(devices=[dev])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            serial = system_info()["devices"][0]["serial_number"]
            assert serial == "0x1b4c8a0"
            assert isinstance(serial, str)

    def test_ai_channels_stripped_to_local_name(self):
        """AI channel names are stripped of device prefix."""
        dev = _make_rich_device(
            "Dev1", "PCIe-6320", 0x1234,
            ai_chans=["Dev1/ai0", "Dev1/ai1", "Dev1/ai2"],
        )
        system = _make_rich_system(devices=[dev])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            channels = system_info()["devices"][0]["ai_channels"]
            assert channels == ["ai0", "ai1", "ai2"]

    def test_ao_channels_stripped_to_local_name(self):
        """AO channel names are stripped of device prefix."""
        dev = _make_rich_device(
            "Dev1", "PCIe-6320", 0x1234,
            ao_chans=["Dev1/ao0", "Dev1/ao1"],
        )
        system = _make_rich_system(devices=[dev])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            channels = system_info()["devices"][0]["ao_channels"]
            assert channels == ["ao0", "ao1"]

    def test_device_with_no_channels(self):
        """Device with no AI or AO channels returns empty lists."""
        dev = _make_rich_device("Dev1", "NI 9401", 0x5678)
        system = _make_rich_system(devices=[dev])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            device = system_info()["devices"][0]
            assert device["ai_channels"] == []
            assert device["ao_channels"] == []

    def test_multiple_devices(self):
        """system_info() returns info for multiple devices."""
        devices = [
            _make_rich_device(
                "cDAQ1Mod1", "NI 9234", 0xAAAA,
                ai_chans=["cDAQ1Mod1/ai0", "cDAQ1Mod1/ai1"],
            ),
            _make_rich_device(
                "cDAQ1Mod2", "NI 9263", 0xBBBB,
                ao_chans=["cDAQ1Mod2/ao0"],
            ),
        ]
        system = _make_rich_system(devices=devices)
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            result = system_info()
            assert len(result["devices"]) == 2
            assert result["devices"][0]["name"] == "cDAQ1Mod1"
            assert result["devices"][1]["name"] == "cDAQ1Mod2"

    def test_tasks_populated(self):
        """Tasks list contains saved NI MAX task names."""
        system = _make_rich_system(task_names=["MyInput", "MyOutput"])
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            assert system_info()["tasks"] == ["MyInput", "MyOutput"]

    def test_empty_system(self):
        """system_info() returns empty lists when no devices or tasks exist."""
        system = _make_rich_system()
        with patch("nidaqmx.system.System.local", return_value=system):
            from nidaqwrapper.utils import system_info

            result = system_info()
            assert result["devices"] == []
            assert result["tasks"] == []

    def test_raises_without_nidaqmx(self):
        """system_info() raises RuntimeError when nidaqmx is unavailable."""
        with patch("nidaqwrapper.utils._NIDAQMX_AVAILABLE", False):
            from nidaqwrapper.utils import system_info

            with pytest.raises(RuntimeError, match="NI-DAQmx drivers"):
                system_info()


class TestUNITSReverse:
    """Tests for the UNITS_REVERSE dictionary."""

    def test_units_reverse_is_dict(self):
        """UNITS_REVERSE is a dict."""
        from nidaqwrapper.utils import UNITS_REVERSE

        assert isinstance(UNITS_REVERSE, dict)

    def test_units_reverse_maps_constants_to_strings(self):
        """For each (k, v) in UNITS, UNITS_REVERSE[v] is a string key in UNITS."""
        from nidaqwrapper.utils import UNITS, UNITS_REVERSE

        for key, val in UNITS.items():
            # The reverse mapping must return some string that is a key in UNITS
            reverse_key = UNITS_REVERSE.get(val)
            assert isinstance(reverse_key, str), (
                f"UNITS_REVERSE[{val!r}] expected str, got {type(reverse_key)}"
            )
            assert reverse_key in UNITS, (
                f"UNITS_REVERSE[{val!r}] = {reverse_key!r} is not a valid UNITS key"
            )

    def test_units_reverse_round_trips(self):
        """UNITS[UNITS_REVERSE[v]] == v for all unique values in UNITS."""
        from nidaqwrapper.utils import UNITS, UNITS_REVERSE

        # Find values that appear only once in UNITS (unambiguous round-trip)
        from collections import Counter
        value_counts = Counter(UNITS.values())
        unique_values = {v for v, count in value_counts.items() if count == 1}

        for key, val in UNITS.items():
            if val in unique_values:
                assert UNITS[UNITS_REVERSE[val]] == val, (
                    f"Round-trip failed: UNITS[UNITS_REVERSE[{val!r}]] != {val!r}"
                )

    def test_units_reverse_collision_mV_g(self):
        """UNITS_REVERSE[MILLIVOLTS_PER_G] returns a valid string key from UNITS.

        Both 'mV/g' and 'mV/m/s**2' map to MILLIVOLTS_PER_G in UNITS.
        The reverse must return one of the two valid keys.
        """
        from nidaqmx import constants

        from nidaqwrapper.utils import UNITS, UNITS_REVERSE

        millivolts_per_g = constants.AccelSensitivityUnits.MILLIVOLTS_PER_G
        result = UNITS_REVERSE.get(millivolts_per_g)
        assert result in ("mV/g", "mV/m/s**2"), (
            f"Expected 'mV/g' or 'mV/m/s**2', got {result!r}"
        )
        # The returned key must exist in UNITS
        assert result in UNITS

    def test_units_reverse_get_returns_fallback(self):
        """UNITS_REVERSE.get(unknown_constant, 'fallback') returns 'fallback'."""
        from nidaqwrapper.utils import UNITS_REVERSE

        sentinel = object()  # an unknown key not in UNITS_REVERSE
        result = UNITS_REVERSE.get(sentinel, "fallback")
        assert result == "fallback"


class TestResolveSystem:
    """``_resolve_system()`` picks the local or remote NI-DAQmx system."""

    def test_none_returns_local_system(self):
        """No grpc_options means the locally-installed driver."""
        from nidaqwrapper.utils import _resolve_system

        with patch("nidaqwrapper.utils.nidaqmx.system.System.local") as local:
            assert _resolve_system(None) is local.return_value
        local.assert_called_once_with()

    def test_none_is_the_default(self):
        """_resolve_system() with no argument is the local path."""
        from nidaqwrapper.utils import _resolve_system

        with patch("nidaqwrapper.utils.nidaqmx.system.System.local") as local:
            assert _resolve_system() is local.return_value

    def test_options_return_remote_system(self):
        """grpc_options routes through System.remote()."""
        from nidaqwrapper.utils import _resolve_system

        opts = MagicMock(name="grpc_options")
        with patch("nidaqwrapper.utils.nidaqmx.system.System.remote") as remote:
            assert _resolve_system(opts) is remote.return_value
        remote.assert_called_once_with(opts)

    def test_options_do_not_touch_the_local_driver(self):
        """The local driver is never consulted on the remote path."""
        from nidaqwrapper.utils import _resolve_system

        opts = MagicMock(name="grpc_options")
        with (
            patch("nidaqwrapper.utils.nidaqmx.system.System.remote"),
            patch("nidaqwrapper.utils.nidaqmx.system.System.local") as local,
        ):
            _resolve_system(opts)
        local.assert_not_called()


class TestUtilsGrpcOptions:
    """The public query helpers accept and forward ``grpc_options``."""

    @pytest.mark.parametrize(
        "func_name",
        [
            "system_info",
            "list_tasks",
            "get_connected_devices",
            "list_devices",
        ],
    )
    def test_helper_queries_the_remote_system(self, func_name, mock_system):
        """Each helper resolves its system through the given options."""
        import nidaqwrapper.utils as utils

        opts = MagicMock(name="grpc_options")
        system = mock_system(task_names=["SavedTask"])
        with patch.object(utils, "_resolve_system", return_value=system) as res:
            getattr(utils, func_name)(opts)
        res.assert_called_once_with(opts)

    @pytest.mark.parametrize(
        "func_name",
        [
            "system_info",
            "list_tasks",
            "get_connected_devices",
            "list_devices",
        ],
    )
    def test_helper_defaults_to_the_local_system(self, func_name, mock_system):
        """Called with no options, each helper stays on the local driver."""
        import nidaqwrapper.utils as utils

        system = mock_system(task_names=["SavedTask"])
        with patch.object(utils, "_resolve_system", return_value=system) as res:
            getattr(utils, func_name)()
        res.assert_called_once_with(None)

    def test_get_task_by_name_forwards_options(self, mock_system):
        """get_task_by_name() resolves its system through the options too."""
        import nidaqwrapper.utils as utils

        opts = MagicMock(name="grpc_options")
        system = mock_system(task_names=["SavedTask"])
        with patch.object(utils, "_resolve_system", return_value=system) as res:
            utils.get_task_by_name("SavedTask", opts)
        res.assert_called_once_with(opts)

    def test_list_devices_returns_remote_devices(self, mock_system):
        """The remote system's devices are what actually come back."""
        import nidaqwrapper.utils as utils

        opts = MagicMock(name="grpc_options")
        system = mock_system(devices=[("cDAQ9Mod1", "NI 9232")])
        with patch.object(utils, "_resolve_system", return_value=system):
            assert utils.list_devices(opts) == [
                {"name": "cDAQ9Mod1", "product_type": "NI 9232"}
            ]
