"""Hardware integration tests for nidaqwrapper.

All tests in this module require real NI-DAQmx hardware and are marked with
``@pytest.mark.hardware``.  Run them with::

    uv run pytest tests/test_hardware.py -v -m hardware

Or exclude them from a normal run with::

    uv run pytest -m "not hardware"

Hardware configuration (resolved by product_type — see ``_resolve_roles``)
-------------------------------------------------------------------------
- NI 9234 (IEPE / delta-sigma AI)  -> role ``iepe``  (e.g. cDAQ6Mod1)
- NI 9215 (SAR AI)                 -> role ``sar``   (e.g. cDAQ6Mod2)
- NI 9260 (delta-sigma AO)         -> role ``ao``    (e.g. cDAQ6Mod3)

Loopback wiring: 9260 ao0 -> 9215 ai0 ; 9260 ao1 -> 9234 ai0.

Note: the USB cDAQ chassis re-enumerates across reboots (cDAQ1 -> cDAQ2 ->
… -> cDAQ6).  Modules are therefore resolved by ``product_type`` substring,
not by device name, so the suite auto-targets them regardless of enumeration.
No digital module is present on this rig (``DI_LINES``/``DO_LINES`` = None).

NI MAX tasks
------------
- ``IM3`` : Pre-existing saved task (do NOT delete)

Notes
-----
Each test uses a unique task name to prevent NI MAX collisions.  All tests
clean up after themselves via try/finally or context managers so that a test
failure never leaves stale hardware tasks behind.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.hardware

# ---------------------------------------------------------------------------
# Hardware role resolution — by product_type, enumeration-independent.
# Resolves to None (tests skip) when the driver/devices are unavailable, so the
# module imports cleanly on CI under `-m "not hardware"`.  A session-scoped
# `hw_roles` fixture in conftest.py mirrors this for new role-based tests.
# ---------------------------------------------------------------------------


def _resolve_roles() -> dict[str, str]:
    """Map roles ``{iepe, sar, ao}`` -> device name by product_type substring."""
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
    except Exception:  # no driver / no devices — tests skip via skip_if_no_device
        pass
    return roles


_ROLES = _resolve_roles()

# Analog input: NI 9215 (SAR)
AI_DEVICE_NAME = _ROLES.get("sar")
AI_DEVICE = AI_DEVICE_NAME
AI_SAMPLE_RATE = 25600  # exact rate supported by NI 9215
AI_VOLTAGE_MIN = -10.0
AI_VOLTAGE_MAX = 10.0

# Analog output: NI 9260 (delta-sigma)
AO_DEVICE_NAME = _ROLES.get("ao")
AO_DEVICE = AO_DEVICE_NAME
AO_SAMPLE_RATE = 25600  # exact rate supported by NI 9260
AO_VOLTAGE_RANGE = 4.242  # NI 9260 max output ±4.242641V (use slightly under)

# IEPE / delta-sigma AI: NI 9234 (issue #5 lives here)
IEPE_DEVICE_NAME = _ROLES.get("iepe")
IEPE_DEVICE = IEPE_DEVICE_NAME

# No digital module on this rig — digital tests skip.
DI_LINES = None
DO_LINES = None

# Second AI device for multi-task tests: the NI 9234 (delta-sigma).
AI2_DEVICE_NAME = _ROLES.get("iepe")
AI2_DEVICE = AI2_DEVICE_NAME

# NI MAX task — set to None if no saved tasks exist (IM3 present on this rig).
NI_MAX_TASK_NAME = "IM3"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def skip_if_no_device():
    """Skip all tests in this module if no NI devices are connected."""
    from nidaqwrapper import list_devices

    devices = list_devices()
    if not devices:
        pytest.skip("No NI-DAQmx devices connected")


# ===========================================================================
# Task Group 3: Device Discovery Smoke Tests
# ===========================================================================


class TestDeviceDiscovery:
    """Verify list_devices() and list_tasks() against real hardware."""

    def test_list_devices_returns_real_devices(self) -> None:
        """list_devices() returns a non-empty list with correct structure.

        Each entry must have 'name' and 'product_type' keys.  Prints
        discovered devices for documentation.
        """
        from nidaqwrapper import list_devices

        devices = list_devices()

        assert isinstance(devices, list)
        assert len(devices) > 0, "Expected at least one NI device connected"

        for dev in devices:
            assert "name" in dev, f"Device entry missing 'name': {dev}"
            assert "product_type" in dev, f"Device entry missing 'product_type': {dev}"

        # Log discovered devices for documentation
        print("\n--- Discovered NI Devices ---")
        for dev in devices:
            print(f"  {dev['name']}: {dev['product_type']}")
        print("---")

        # Verify expected devices are present
        device_names = [d["name"] for d in devices]
        assert AI_DEVICE_NAME in device_names, (
            f"Expected {AI_DEVICE_NAME} in device list, got: {device_names}"
        )

    def test_list_tasks_returns_list(self) -> None:
        """list_tasks() returns a list (may contain saved NI MAX tasks).

        Prints discovered tasks for documentation.
        """
        from nidaqwrapper import list_tasks

        tasks = list_tasks()

        assert isinstance(tasks, list)

        # Log discovered tasks
        print(f"\n--- Discovered NI MAX Tasks: {tasks} ---")

        if tasks:
            for t in tasks:
                assert isinstance(t, str), f"Expected string task name, got: {type(t)}"
        if tasks and NI_MAX_TASK_NAME is not None:
            assert NI_MAX_TASK_NAME in tasks, (
                f"Expected '{NI_MAX_TASK_NAME}' in task list, got: {tasks}"
            )


# ===========================================================================
# Task Group 4: AITask Voltage Channel Acquisition Tests
# ===========================================================================


class TestAITaskHardware:
    """Validate AITask voltage acquisition against real NI 9215."""

    def test_nitask_voltage_channel_acquisition(self) -> None:
        """Create a voltage channel, start, acquire, verify data shape.

        Uses the priming read pattern: first read(-1) may return 0 samples,
        so we discard it, sleep, and assert on the second read.
        """
        from nidaqwrapper import AITask

        task = AITask("hw_acq_voltage", sample_rate=AI_SAMPLE_RATE)
        try:
            task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")
            task.configure()
            task.start()

            # Priming read — discard
            time.sleep(0.1)
            task.acquire()

            # Real acquisition
            time.sleep(0.2)
            data = task.acquire()

            assert isinstance(data, np.ndarray), f"Expected ndarray, got {type(data)}"
            assert data.ndim == 2, f"Expected 2-D array, got shape {data.shape}"
            assert data.shape[1] == 1, f"Expected 1 channel column, got shape {data.shape}"
            assert data.shape[0] > 0, "Expected at least one sample"
        finally:
            task.clear_task()

    def test_nitask_sample_rate_accuracy(self) -> None:
        """Acquired sample count matches expected rate within 20% tolerance.

        Acquires for 0.5s at 25600 Hz. Expected ~12800 samples.
        """
        from nidaqwrapper import AITask

        task = AITask("hw_rate_check", sample_rate=AI_SAMPLE_RATE)
        try:
            task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")
            task.configure()
            task.start()

            # Priming read
            time.sleep(0.1)
            task.acquire()

            # Timed acquisition
            time.sleep(0.5)
            data = task.acquire()

            expected_samples = int(AI_SAMPLE_RATE * 0.5)
            actual_samples = data.shape[0]
            tolerance = 0.20
            assert abs(actual_samples - expected_samples) / expected_samples < tolerance, (
                f"Sample count {actual_samples} deviates >20% from expected "
                f"{expected_samples} (rate={AI_SAMPLE_RATE} Hz, duration=0.5s)"
            )
        finally:
            task.clear_task()

    def test_aitask_context_manager(self) -> None:
        """AITask context manager cleans up properly."""
        from nidaqwrapper import AITask

        task_name = "hw_ctx_nitask"
        with AITask(task_name, sample_rate=AI_SAMPLE_RATE) as task:
            task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")
            task.configure()
            task.start()

            time.sleep(0.1)
            task.acquire()
            time.sleep(0.1)
            data = task.acquire()
            assert data.shape[0] > 0

        # After exit, task handle is released
        assert task.task is None

        # Can re-create with same name (proves cleanup)
        with AITask(task_name, sample_rate=AI_SAMPLE_RATE) as task2:
            task2.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")
            task2.configure()


# ===========================================================================
# Task Group 5: AOTask AO Generation Tests
# ===========================================================================


class TestAOTaskHardware:
    """Validate AOTask analog output against real NI 9260."""

    def test_nitaskoutput_ao_generation(self) -> None:
        """Create AO channel, start, generate sine wave, clear task."""
        if AO_DEVICE_NAME is None:
            pytest.skip("No AO device available")

        from nidaqwrapper import AOTask

        task = AOTask("hw_ao_gen", sample_rate=AO_SAMPLE_RATE)
        try:
            task.add_channel(
                "ao0", device=AO_DEVICE, channel_ind=0,
                min_val=-AO_VOLTAGE_RANGE, max_val=AO_VOLTAGE_RANGE,
            )
            task.configure()

            # Generate a short sine wave (1 second, within NI 9260 range)
            t = np.linspace(0, 1, AO_SAMPLE_RATE, endpoint=False)
            signal = (2.0 * np.sin(2 * np.pi * 10 * t)).reshape(-1, 1)
            task.generate(signal)

            # Let it run briefly
            time.sleep(0.2)
        finally:
            task.clear_task()

    def test_nitaskoutput_context_manager(self) -> None:
        """AOTask context manager cleans up properly."""
        if AO_DEVICE_NAME is None:
            pytest.skip("No AO device available")

        from nidaqwrapper import AOTask

        task_name = "hw_ctx_ao"
        with AOTask(task_name, sample_rate=AO_SAMPLE_RATE) as task:
            task.add_channel(
                "ao0", device=AO_DEVICE, channel_ind=0,
                min_val=-AO_VOLTAGE_RANGE, max_val=AO_VOLTAGE_RANGE,
            )
            task.configure()

            t = np.linspace(0, 1, AO_SAMPLE_RATE, endpoint=False)
            signal = (1.0 * np.sin(2 * np.pi * 10 * t)).reshape(-1, 1)
            task.generate(signal)
            time.sleep(0.1)

        assert task.task is None


# ===========================================================================
# Task Group 6: DAQHandler NI MAX Task Test
# ===========================================================================


class TestWrapperNIMaxTask:
    """Validate DAQHandler with a pre-existing NI MAX task."""

    def test_wrapper_ni_max_task(self) -> None:
        """Configure, connect, introspect, and disconnect with NI MAX task.

        Uses the pre-existing 'IM3' task saved in NI MAX.
        """
        if NI_MAX_TASK_NAME is None:
            pytest.skip("No NI MAX task available")

        from nidaqwrapper import DAQHandler

        wrapper = DAQHandler()
        try:
            wrapper.configure(task_in=NI_MAX_TASK_NAME)
            result = wrapper.connect()
            assert result is True, "connect() should return True for NI MAX task"

            # Introspect
            ch_names = wrapper.get_channel_names()
            assert len(ch_names) > 0, "Expected at least one channel from NI MAX task"

            sample_rate = wrapper.get_sample_rate()
            assert sample_rate > 0, f"Expected positive sample rate, got {sample_rate}"
        finally:
            wrapper.disconnect()


# ===========================================================================
# Task Group 7: DAQHandler Programmatic Task Test
# ===========================================================================


class TestWrapperProgrammatic:
    """Validate DAQHandler full lifecycle with programmatic AITask."""

    def test_wrapper_programmatic_full_lifecycle(self) -> None:
        """Configure, connect, set trigger, acquire, disconnect.

        Uses a very low trigger level so noise triggers it quickly.
        """
        from nidaqwrapper import DAQHandler, AITask

        n_samples = 5000

        task = AITask("hw_wrap_prog", sample_rate=AI_SAMPLE_RATE)
        task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")

        wrapper = DAQHandler()
        try:
            wrapper.configure(task_in=task)
            result = wrapper.connect()
            assert result is True, "connect() should return True"

            assert wrapper.get_channel_names() == ["ch0"]
            assert wrapper.get_sample_rate() == AI_SAMPLE_RATE

            # Set trigger with very low level so noise triggers immediately
            wrapper.set_trigger(
                n_samples=n_samples,
                trigger_channel=0,
                trigger_level=0.001,
                trigger_type="abs",
                presamples=100,
            )

            data = wrapper.acquire()

            assert isinstance(data, np.ndarray)
            assert data.shape == (n_samples, 1), (
                f"Expected shape ({n_samples}, 1), got {data.shape}"
            )
        finally:
            wrapper.disconnect()

    def test_wrapper_acquire(self) -> None:
        """read_all_available() returns (n_samples, n_channels) data."""
        from nidaqwrapper import DAQHandler, AITask

        task = AITask("hw_wrap_raa", sample_rate=AI_SAMPLE_RATE)
        task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")

        wrapper = DAQHandler()
        try:
            wrapper.configure(task_in=task)
            wrapper.connect()

            # Start the task manually
            wrapper._task_in.start()

            # Priming read
            time.sleep(0.1)
            wrapper.read_all_available()

            # Real read
            time.sleep(0.2)
            data = wrapper.read_all_available()

            assert isinstance(data, np.ndarray)
            assert data.ndim == 2
            assert data.shape[1] == 1, f"Expected 1 channel, got shape {data.shape}"
            assert data.shape[0] > 0, "Expected at least one sample"

            # Verify voltage range
            assert np.all(data >= AI_VOLTAGE_MIN)
            assert np.all(data <= AI_VOLTAGE_MAX)
        finally:
            wrapper.disconnect()


# ===========================================================================
# Task Group 8: Single-Sample Read/Write Tests
# ===========================================================================


class TestSingleSample:
    """Validate single-sample read() and write() on real hardware."""

    def test_single_sample_read(self) -> None:
        """read() returns (n_channels,) array with reasonable voltages."""
        from nidaqwrapper import DAQHandler, AITask

        task = AITask("hw_ss_read", sample_rate=AI_SAMPLE_RATE)
        task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")

        wrapper = DAQHandler()
        try:
            wrapper.configure(task_in=task)
            wrapper.connect()

            # Start the task for single-sample reads
            wrapper._task_in.start()

            data = wrapper.read()

            assert isinstance(data, np.ndarray)
            assert data.shape == (1,), f"Expected shape (1,), got {data.shape}"
            assert AI_VOLTAGE_MIN <= data[0] <= AI_VOLTAGE_MAX, (
                f"Voltage {data[0]:.4f} outside expected range"
            )
        finally:
            wrapper.disconnect()

    def test_single_sample_write(self) -> None:
        """write() sets output voltage on NI 9260 without error."""
        if AO_DEVICE_NAME is None:
            pytest.skip("No AO device available")

        from nidaqwrapper import DAQHandler, AOTask

        task_out = AOTask("hw_ss_write", sample_rate=AO_SAMPLE_RATE)
        task_out.add_channel(
            "ao0", device=AO_DEVICE, channel_ind=0,
            min_val=-AO_VOLTAGE_RANGE, max_val=AO_VOLTAGE_RANGE,
        )

        wrapper = DAQHandler()
        try:
            wrapper.configure(task_out=task_out)
            wrapper.connect()

            # Write various values — should not raise
            wrapper.write(0.0)
            wrapper.write(1.0)
            wrapper.write(0.0)  # Reset to zero
        finally:
            wrapper.disconnect()


# ===========================================================================
# Task Group 9: Context Manager Cleanup Tests
# ===========================================================================


class TestWrapperContextManager:
    """Validate DAQHandler context manager releases hardware resources."""

    def test_context_manager_normal_exit(self) -> None:
        """Resources released on normal with-block exit."""
        from nidaqwrapper import DAQHandler, AITask

        task_name = "hw_ctx_norm"

        with DAQHandler() as wrapper:
            task = AITask(task_name, sample_rate=AI_SAMPLE_RATE)
            task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")
            wrapper.configure(task_in=task)
            wrapper.connect()

        # After context exit, should be disconnected
        # Verify resources released: can create a new task with the same name
        new_task = AITask(task_name, sample_rate=AI_SAMPLE_RATE)
        try:
            new_task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")
            new_task.configure()
        finally:
            new_task.clear_task()

    def test_context_manager_exception_cleanup(self) -> None:
        """Resources released even when exception occurs in with-block."""
        from nidaqwrapper import DAQHandler, AITask

        task_name = "hw_ctx_exc"

        with pytest.raises(RuntimeError, match="deliberate"):
            with DAQHandler() as wrapper:
                task = AITask(task_name, sample_rate=AI_SAMPLE_RATE)
                task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")
                wrapper.configure(task_in=task)
                wrapper.connect()
                raise RuntimeError("deliberate test exception")

        # After exception, resources should still be released
        new_task = AITask(task_name, sample_rate=AI_SAMPLE_RATE)
        try:
            new_task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")
            new_task.configure()
        finally:
            new_task.clear_task()


# ===========================================================================
# Task Group 10: Digital I/O Hardware Tests
# ===========================================================================


class TestDigitalIO:
    """Validate standalone DITask/DOTask on PCIe-6320.

    Note: wrapper digital integration (read_digital/write_digital) is
    skipped — depends on Phase 6b (digital-wrapper-integration).
    """

    def test_digital_input_read(self) -> None:
        """DITask reads a boolean value from Dev1 DI line."""
        if DI_LINES is None:
            pytest.skip("No digital input hardware available")

        from nidaqwrapper import DITask

        di = DITask("hw_di_read")
        try:
            di.add_channel("di_ch", lines=DI_LINES)
            di.configure()
            di.start()
            data = di.read()

            assert isinstance(data, np.ndarray)
            assert data.size >= 1, "Expected at least one value from DI read"
            # Each value should be boolean-like (0 or 1)
            for val in data.flat:
                assert val in (True, False, 0, 1), f"Unexpected DI value: {val}"
        finally:
            di.clear_task()

    def test_digital_output_write(self) -> None:
        """DOTask writes True/False to Dev1 DO line without error."""
        if DO_LINES is None:
            pytest.skip("No digital output hardware available")

        from nidaqwrapper import DOTask

        do = DOTask("hw_do_write")
        try:
            do.add_channel("do_ch", lines=DO_LINES)
            do.configure()
            do.start()

            do.write(True)
            do.write(False)
        finally:
            do.clear_task()

    def test_digital_context_manager(self) -> None:
        """Digital I/O context managers clean up properly."""
        if DI_LINES is None:
            pytest.skip("No digital input hardware available")

        from nidaqwrapper import DITask

        task_name = "hw_di_ctx"
        with DITask(task_name) as di:
            di.add_channel("di_ch", lines=DI_LINES)
            di.configure()
            di.start()
            data = di.read()
            assert data.size >= 1

        assert di.task is None

    def test_wrapper_digital_integration(self) -> None:
        """DAQHandler.read_digital() and write_digital() work with real hardware.

        Configures a wrapper with digital input (Dev1 port0/line0) and digital
        output (Dev1 port1/line0), connects, writes True/False, reads, and
        disconnects.
        """
        if DI_LINES is None or DO_LINES is None:
            pytest.skip("Both DI and DO hardware required for wrapper digital test")

        from nidaqwrapper import DITask, DOTask, DAQHandler

        di = DITask("hw_wrap_di")
        di.add_channel("di_ch", lines=DI_LINES)

        do = DOTask("hw_wrap_do")
        do.add_channel("do_ch", lines=DO_LINES)

        wrapper = DAQHandler()
        wrapper.configure(task_digital_in=di, task_digital_out=do)

        try:
            result = wrapper.connect()
            assert result is True

            # Write True, then read digital input
            wrapper.write_digital(True)
            data = wrapper.read_digital()
            assert isinstance(data, np.ndarray)
            assert data.size >= 1

            # Write False
            wrapper.write_digital(False)
        finally:
            wrapper.disconnect()


# ===========================================================================
# Task Group 11: MultiHandler Multi-Task Test
# ===========================================================================


class TestMultiHandlerHardware:
    """Validate MultiHandler on real hardware.

    Tests both single-task software trigger and (if 2+ modules are available)
    multi-task hardware trigger modes.
    """

    def test_multihandler_single_task_software_trigger(self) -> None:
        """MultiHandler with a single input task and software trigger."""
        from nidaqwrapper import MultiHandler, AITask

        n_samples = 5000

        task = AITask("hw_adv_st", sample_rate=AI_SAMPLE_RATE)
        try:
            task.add_channel("ch0", device=AI_DEVICE, channel_ind=0, units="V")
            task.configure()

            adv = MultiHandler()
            try:
                result = adv.configure(input_tasks=[task.task])
                assert result is True, "configure() should return True"

                result = adv.connect()
                assert result is True, "connect() should return True"

                # Set trigger with very low level
                adv.set_trigger(
                    n_samples=n_samples,
                    trigger_channel=0,
                    trigger_level=0.001,
                    trigger_type="abs",
                )

                data = adv.acquire()

                # Software trigger returns dict with channel names + 'time'
                assert isinstance(data, dict)
                assert "time" in data
                non_time_keys = [k for k in data if k != "time"]
                assert len(non_time_keys) >= 1

                # Verify data shape
                for key in non_time_keys:
                    assert isinstance(data[key], np.ndarray)
                    assert len(data[key]) == n_samples, (
                        f"Expected {n_samples} samples for '{key}', got {len(data[key])}"
                    )
            finally:
                adv.disconnect()
        finally:
            task.clear_task()


# ===========================================================================
# fix-gh-issues-5-8: IEPE consecutive-add regression (GitHub issue #5)
# ===========================================================================


def _find_iepe_module() -> str | None:
    """Return the name of the first connected IEPE-capable module, if any.

    Issue #5 reproduces only on delta-sigma modules that reject on-demand
    timing (e.g. NI 9234, cDAQ-9132-hosted modules); E/M/X-series devices
    accept on-demand timing and never raised -201087.
    """
    from nidaqwrapper import list_devices

    for dev in list_devices():
        product_type = dev["product_type"]
        if "9234" in product_type or "9132" in product_type:
            return dev["name"]
    return None


class TestIEPEConsecutiveAdd:
    """Issue #5: consecutive accel adds on a delta-sigma module.

    Before fix-gh-issues-5-8, the second add_channel() raised DaqError
    -201087: the duplicate-channel pre-check iterated live channel objects,
    triggering implicit task verification on a task without configured
    timing — which IEPE modules categorically reject.  add_channel() now
    configures timing after every add, so the iteration is safe.
    """

    def test_two_accel_channels_consecutive_add(self) -> None:
        """Two consecutive accel adds succeed; both channels are present.

        SAFETY: uses ``channel_ind`` 1 and 2, NOT 0.  On the loopback rig the
        9234's ai0 is wired to a live 9260 AO output, and IEPE excitation
        (2 mA constant current) MUST NEVER be driven into an active output.
        ai1/ai2 are unwired, so enabling IEPE on them is safe.
        """
        device = _find_iepe_module()
        if device is None:
            pytest.skip(
                "No IEPE-capable module (NI 9234 / cDAQ-9132) connected"
            )

        from nidaqwrapper import AITask

        task = AITask("test_issue5_iepe_adds", sample_rate=25600)
        try:
            task.add_channel(
                "acc0", device=device, channel_ind=1,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            # This second add raised DaqError -201087 before the fix
            task.add_channel(
                "acc1", device=device, channel_ind=2,
                sensitivity=100.0, sensitivity_units="mV/g", units="g",
            )
            assert task.channel_list == ["acc0", "acc1"]
        finally:
            task.clear_task()


# ===========================================================================
# cdaq6-hardware-validation: behaviors validated on the real cDAQ6 rig
# ===========================================================================


class TestAOToAILoopback:
    """AO->AI loopback signal-path validation (RMS / FFT).

    Generates a known sine on the NI 9260 and captures it on the wired NI 9215
    AI channel, proving the end-to-end signal path.  IEPE is OFF (voltage
    channel).  Discard-first-read + settle per the pacing rules.  Skips unless
    both the ``ao`` and ``sar`` roles are present and physically wired
    (9260 ao0 -> 9215 ai0).
    """

    def test_loopback_sine_rms_and_frequency(self, hw_roles) -> None:
        """A +-2 V / 50 Hz sine reads back at ~1.414 Vrms, dominant bin 50 Hz."""
        ao_dev = hw_roles.get("ao")
        sar_dev = hw_roles.get("sar")
        if ao_dev is None or sar_dev is None:
            pytest.skip("Loopback requires NI 9260 (ao) + NI 9215 (sar)")

        from nidaqwrapper import AITask, AOTask

        rate = AO_SAMPLE_RATE  # 25600, valid on both modules
        amp = 2.0              # 2 V peak (1.41 Vrms) — well under 9260 ±4.242 V
        freq = 50              # 50 integer cycles in a 1 s buffer
        n_buf = rate
        n_acq = 4096

        t = np.arange(n_buf) / rate
        sine = (amp * np.sin(2 * np.pi * freq * t)).reshape(-1, 1)

        ao = AOTask("hw_lb_ao", sample_rate=rate, samples_per_channel=n_buf)
        ai = AITask("hw_lb_ai", sample_rate=rate)
        try:
            ao.add_channel(
                "out", device=ao_dev, channel_ind=0,
                min_val=-AO_VOLTAGE_RANGE, max_val=AO_VOLTAGE_RANGE,
            )
            ao.configure()
            ao.generate(sine)        # auto-starts, regenerates continuously
            time.sleep(0.2)          # delta-sigma group delay + settle

            ai.add_channel("in", device=sar_dev, channel_ind=0, units="V")
            ai.configure()
            ai.start()
            time.sleep(0.1)
            ai.acquire(n_acq)        # discard first read
            time.sleep(0.2)
            data = ai.acquire(n_acq)

            assert data.shape == (n_acq, 1)
            x = data.reshape(-1)
            ac_rms = float(np.std(x))            # AC RMS (demeaned)
            expected_rms = amp / np.sqrt(2)      # 1.414 V
            assert abs(ac_rms - expected_rms) / expected_rms <= 0.10, (
                f"AC-RMS {ac_rms:.4f} V vs expected {expected_rms:.4f} V"
            )

            # Dominant FFT bin (hann-windowed, DC removed)
            win = np.hanning(n_acq)
            spec = np.abs(np.fft.rfft((x - x.mean()) * win))
            spec[0] = 0.0
            bin_hz = rate / n_acq
            peak_hz = int(np.argmax(spec)) * bin_hz
            assert abs(peak_hz - freq) <= 2 * bin_hz, (
                f"Dominant bin {peak_hz:.1f} Hz vs expected {freq} Hz"
            )
        finally:
            try:
                ao.stop()
            except Exception:
                pass
            ao.clear_task()
            ai.clear_task()


class TestAITaskFiniteMode:
    """AITask finite mode returns exactly N samples."""

    def test_finite_returns_exact_sample_count(self) -> None:
        """configure(sample_mode='finite', samples_per_channel=N) -> (N, 1)."""
        if AI_DEVICE is None:
            pytest.skip("No SAR AI device available")

        from nidaqwrapper import AITask

        n = 2048
        task = AITask("hw_finite_ai", sample_rate=AI_SAMPLE_RATE)
        try:
            task.add_channel("v0", device=AI_DEVICE, channel_ind=0, units="V")
            task.configure(sample_mode="finite", samples_per_channel=n)
            task.start()
            data = task.acquire(None)  # finite blocks and returns exactly N
            assert data.shape == (n, 1), data.shape
        finally:
            task.clear_task()


class TestDuplicateAOChannel:
    """Issue #6: duplicate physical AO channel raises ValueError on hardware."""

    def test_duplicate_physical_ao_raises_value_error(self) -> None:
        """Adding the same physical AO channel twice -> ValueError (-200371)."""
        if AO_DEVICE is None:
            pytest.skip("No AO device available")

        from nidaqwrapper import AOTask

        task = AOTask("hw_dup_ao", sample_rate=AO_SAMPLE_RATE)
        try:
            task.add_channel(
                "out0", device=AO_DEVICE, channel_ind=0,
                min_val=-AO_VOLTAGE_RANGE, max_val=AO_VOLTAGE_RANGE,
            )
            with pytest.raises(ValueError):
                task.add_channel(
                    "out0b", device=AO_DEVICE, channel_ind=0,
                    min_val=-AO_VOLTAGE_RANGE, max_val=AO_VOLTAGE_RANGE,
                )
        finally:
            task.clear_task()
