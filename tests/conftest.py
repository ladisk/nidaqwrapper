"""Shared test fixtures with mocked nidaqmx objects.

Mock Strategy
-------------
All unit tests run WITHOUT nidaqmx installed. Instead of globally patching
sys.modules, each test uses targeted fixtures that provide mock objects
mimicking the nidaqmx API. Tests use ``unittest.mock.patch`` for per-test
isolation.

Fixtures
--------
mock_constants
    Mock nidaqmx.constants with all enum values used by the package.
mock_device
    Factory for mock NI device objects with name and product_type.
mock_system
    Factory for mock nidaqmx.system.System.local() with devices and tasks.
mock_task
    Mock nidaqmx.Task with channels, timing, read/write, lifecycle methods.
mock_daq_error
    Factory for DaqError-like exceptions with configurable error_code.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# mock_constants — nidaqmx.constants enum values
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_constants():
    """Provide a mock object mimicking ``nidaqmx.constants``.

    Contains all enum values used across the nidaqwrapper package so that
    tests can reference the same "constants" without importing nidaqmx.
    """
    constants = MagicMock()

    # AccelSensitivityUnits
    constants.AccelSensitivityUnits.MILLIVOLTS_PER_G = "MILLIVOLTS_PER_G"
    constants.AccelSensitivityUnits.M_VOLTS_PER_METERS_PER_SECOND_SQUARED = (
        "M_VOLTS_PER_METERS_PER_SECOND_SQUARED"
    )

    # AccelUnits
    constants.AccelUnits.G = "ACCEL_G"
    constants.AccelUnits.METERS_PER_SECOND_SQUARED = "METERS_PER_SECOND_SQUARED"

    # ForceIEPESensorSensitivityUnits
    constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON = (
        "MILLIVOLTS_PER_NEWTON"
    )

    # ForceUnits
    constants.ForceUnits.NEWTONS = "NEWTONS"

    # VoltageUnits
    constants.VoltageUnits.VOLTS = "VOLTS"
    constants.VoltageUnits.FROM_CUSTOM_SCALE = "FROM_CUSTOM_SCALE"

    # AcquisitionType
    constants.AcquisitionType.CONTINUOUS = "CONTINUOUS"
    constants.AcquisitionType.FINITE = "FINITE"

    # RegenerationMode
    constants.RegenerationMode.ALLOW_REGENERATION = "ALLOW_REGENERATION"
    constants.RegenerationMode.DO_NOT_ALLOW_REGENERATION = "DO_NOT_ALLOW_REGENERATION"

    # READ_ALL_AVAILABLE
    constants.READ_ALL_AVAILABLE = -1

    return constants


# ---------------------------------------------------------------------------
# mock_device — factory for mock NI devices
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_device():
    """Factory fixture that creates a mock NI device.

    Parameters
    ----------
    name : str
        Device name (e.g., ``"cDAQ1Mod1"``).
    product_type : str
        Product type string (e.g., ``"NI 9234"``).

    Returns
    -------
    MagicMock
        A mock device with ``.name`` and ``.product_type`` attributes.
    """
    def _make_device(name: str = "cDAQ1Mod1", product_type: str = "NI 9234"):
        device = MagicMock()
        device.name = name
        device.product_type = product_type
        return device

    return _make_device


# ---------------------------------------------------------------------------
# mock_system — mocked nidaqmx.system.System.local()
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_system(mock_device):
    """Factory fixture for a mocked ``nidaqmx.system.System.local()``.

    Parameters
    ----------
    devices : list[tuple[str, str]] | None
        List of ``(name, product_type)`` tuples. Defaults to two devices.
    task_names : list[str] | None
        List of saved task name strings. Defaults to empty.

    Returns
    -------
    MagicMock
        A mock system object with ``.devices`` and ``.tasks`` attributes.
    """
    def _make_system(
        devices: list[tuple[str, str]] | None = None,
        task_names: list[str] | None = None,
    ):
        if devices is None:
            devices = [("cDAQ1Mod1", "NI 9234"), ("cDAQ1Mod2", "NI 9263")]
        if task_names is None:
            task_names = []

        system = MagicMock()
        system.devices = [mock_device(name=n, product_type=pt) for n, pt in devices]

        # Tasks setup: system.tasks is iterable of task objects with ._name
        # system.tasks.task_names returns list of name strings
        tasks_collection = MagicMock()
        mock_tasks = []
        for tn in task_names:
            t = MagicMock()
            t._name = tn
            t.load.return_value = MagicMock(name=f"loaded_{tn}")
            mock_tasks.append(t)

        tasks_collection.__iter__ = MagicMock(side_effect=lambda: iter(mock_tasks))
        tasks_collection.task_names = task_names
        system.tasks = tasks_collection

        return system

    return _make_system


# ---------------------------------------------------------------------------
# mock_task — mocked nidaqmx.Task
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_task():
    """Create a mock ``nidaqmx.Task`` with channel collections and methods.

    The mock provides:
    - ``.ai_channels.add_ai_accel_chan()``, ``.add_ai_force_iepe_chan()``,
      ``.add_ai_voltage_chan()``
    - ``.ao_channels.add_ao_voltage_chan()``
    - ``.di_channels.add_di_chan()``
    - ``.do_channels.add_do_chan()``
    - ``.timing.cfg_samp_clk_timing()``
    - ``.read()``, ``.write()``
    - ``.start()``, ``.stop()``, ``.close()``
    - ``.name``, ``.channels``, ``.devices``
    """
    task = MagicMock()
    task.name = "MockTask"

    # Channel collections
    task.ai_channels = MagicMock()
    task.ao_channels = MagicMock()
    task.di_channels = MagicMock()
    task.do_channels = MagicMock()

    # Timing
    task.timing = MagicMock()

    # Channels / devices collections
    task.channels = MagicMock()
    task.devices = MagicMock()

    return task


# ---------------------------------------------------------------------------
# mock_daq_error — factory for DaqError-like exceptions
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_daq_error():
    """Factory fixture for creating ``DaqError``-like exceptions.

    Parameters
    ----------
    error_code : int
        The NI-DAQmx error code (e.g., ``-200089``).
    message : str
        Optional error message text.

    Returns
    -------
    Exception
        An exception instance with an ``error_code`` attribute.
    """
    def _make_error(error_code: int, message: str = "DAQ error"):
        error = Exception(message)
        error.error_code = error_code
        return error

    return _make_error


# ===========================================================================
# Simulated Device Fixtures — Real nidaqmx with simulated hardware
# ===========================================================================

SIMULATED_DEVICE_NAME = "SimDev1"
SIMULATED_TASK_NAME = "SimTask1"

SKIP_MSG = (
    f"Simulated device '{SIMULATED_DEVICE_NAME}' not found. "
    "Run 'sudo scripts/setup_simulated_devices.sh' to create it. "
    "See TESTING.md for details."
)


@pytest.fixture(scope="session")
def simulated_device_name():
    """Provide the simulated device name if it exists, otherwise skip tests.

    Returns
    -------
    str
        The simulated device name (e.g., "SimDev1").

    Raises
    ------
    pytest.skip
        If the simulated device does not exist.
    """
    try:
        import nidaqmx.system

        devices = [d.name for d in nidaqmx.system.System.local().devices]
        if SIMULATED_DEVICE_NAME not in devices:
            pytest.skip(SKIP_MSG)
        return SIMULATED_DEVICE_NAME
    except ImportError:
        pytest.skip("nidaqmx not installed")


@pytest.fixture(scope="session")
def sim_device(simulated_device_name):
    """Provide the simulated Device object.

    Returns
    -------
    nidaqmx.system.Device
        The simulated device object.
    """
    import nidaqmx.system

    system = nidaqmx.system.System.local()
    for device in system.devices:
        if device.name == simulated_device_name:
            return device
    pytest.skip(SKIP_MSG)


@pytest.fixture(scope="session")
def sim_task_name(simulated_device_name):
    """Provide the simulated task name if it exists, otherwise skip tests.

    Returns
    -------
    str
        The simulated task name (e.g., "SimTask1").

    Raises
    ------
    pytest.skip
        If the simulated task does not exist.
    """
    try:
        import nidaqmx.system

        tasks = [t._name for t in nidaqmx.system.System.local().tasks]
        if SIMULATED_TASK_NAME not in tasks:
            pytest.skip(
                f"Simulated task '{SIMULATED_TASK_NAME}' not found. "
                "Run 'sudo scripts/setup_simulated_devices.sh' to create it."
            )
        return SIMULATED_TASK_NAME
    except ImportError:
        pytest.skip("nidaqmx not installed")


@pytest.fixture
def sim_ai_task(sim_device_name):
    """Create an AITask with 2 AI channels on SimDev1 at 10kHz.

    Yields
    ------
    AITask
        Configured AITask instance with 2 voltage channels.
    """
    from nidaqwrapper import AITask

    task = AITask("test_sim_ai", sample_rate=10000)
    try:
        # Add 2 AI voltage channels (ai0, ai1)
        task.add_channel(
            "ai0", device=sim_device_name, channel_ind=0, units="V"
        )
        task.add_channel(
            "ai1", device=sim_device_name, channel_ind=1, units="V"
        )
        yield task
    finally:
        try:
            task.clear_task()
        except Exception:
            pass


@pytest.fixture
def sim_ao_task(sim_device_name):
    """Create an AOTask with 1 AO channel on SimDev1 at 10kHz.

    Yields
    ------
    AOTask
        Configured AOTask instance with 1 voltage channel.
    """
    from nidaqwrapper import AOTask

    task = AOTask("test_sim_ao", sample_rate=10000)
    try:
        # Add 1 AO voltage channel (ao0)
        task.add_channel(
            "ao0",
            device=sim_device_name,
            channel_ind=0,
            min_val=-10.0,
            max_val=10.0,
        )
        yield task
    finally:
        try:
            task.clear_task()
        except Exception:
            pass


@pytest.fixture
def sim_di_task(simulated_device_name):
    """Create a DITask with 4 DI lines on SimDev1 in on-demand mode.

    Yields
    ------
    DITask
        Configured DITask instance with 4 digital input lines.
    """
    from nidaqwrapper import DITask

    task = DITask("test_sim_di")
    try:
        # Add 4 DI lines (port0/line0:3)
        task.add_channel("di_ch", lines=f"{simulated_device_name}/port0/line0:3")
        yield task
    finally:
        try:
            task.clear_task()
        except Exception:
            pass


@pytest.fixture
def sim_do_task(simulated_device_name):
    """Create a DOTask with 4 DO lines on SimDev1 in on-demand mode.

    Yields
    ------
    DOTask
        Configured DOTask instance with 4 digital output lines.
    """
    from nidaqwrapper import DOTask

    task = DOTask("test_sim_do")
    try:
        # Add 4 DO lines (port1/line0:3)
        task.add_channel("do_ch", lines=f"{simulated_device_name}/port1/line0:3")
        yield task
    finally:
        try:
            task.clear_task()
        except Exception:
            pass


@pytest.fixture
def sim_device_name(simulated_device_name):
    """Return the device name string for SimDev1.

    Returns
    -------
    str
        Device name string (e.g. "SimDev1") for use in add_channel(device=...).
    """
    return simulated_device_name


SIMULATED_DEVICE2_NAME = "SimDev2"


@pytest.fixture(scope="session")
def sim_device2_name():
    """Provide the second simulated device name (cross-device tests).

    Returns
    -------
    str
        The second simulated device name (e.g. "SimDev2").

    Raises
    ------
    pytest.skip
        If the second simulated device does not exist.
    """
    try:
        import nidaqmx.system

        devices = [d.name for d in nidaqmx.system.System.local().devices]
        if SIMULATED_DEVICE2_NAME not in devices:
            pytest.skip(
                f"Simulated device '{SIMULATED_DEVICE2_NAME}' not found. "
                "Run 'sudo scripts/setup_simulated_devices.sh' to create it. "
                "See TESTING.md for details."
            )
        return SIMULATED_DEVICE2_NAME
    except ImportError:
        pytest.skip("nidaqmx not installed")


# ===========================================================================
# Hardware Role Resolution — real cDAQ rig (IEPE / SAR / AO), by product_type
# ===========================================================================


@pytest.fixture(scope="session")
def hw_roles() -> dict[str, str]:
    """Map module roles to live device names on a real cDAQ rig.

    Resolves by ``product_type`` substring so the hardware suite auto-targets
    the modules regardless of chassis enumeration (e.g. a USB cDAQ chassis
    re-enumerating cDAQ5 -> cDAQ6).  Tests use ``hw_roles.get("sar")`` etc. and
    skip cleanly when a role is absent.

    Returns
    -------
    dict[str, str]
        Keys among ``{"iepe", "sar", "ao"}`` mapped to device names:
        ``"iepe"`` -> NI 9234, ``"sar"`` -> NI 9215, ``"ao"`` -> NI 9260.
        Empty when no driver/devices are available.
    """
    roles: dict[str, str] = {}
    try:
        from nidaqwrapper import list_devices

        for dev in list_devices():
            product_type = dev["product_type"]
            if "9234" in product_type:
                roles["iepe"] = dev["name"]
            elif "9215" in product_type:
                roles["sar"] = dev["name"]
            elif "9260" in product_type:
                roles["ao"] = dev["name"]
    except Exception:
        pass
    return roles
