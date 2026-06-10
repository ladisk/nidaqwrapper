"""Simulated device tests for DAQHandler full lifecycle.

All tests run against SimDev1 (PCIe-6361 simulated device) and SimTask1
(persisted AI task) and are marked with ``@pytest.mark.simulated``.

Run with::

    uv run pytest tests/test_sim_handler.py -v -m simulated

Requirements
------------
- SimDev1 must exist (created by scripts/setup_simulated_devices.sh)
- SimTask1 must exist (4 AI channels on SimDev1/ai0:3, 10kHz, continuous, RSE)
"""

from __future__ import annotations

import time
from concurrent.futures import Future

import numpy as np
import pytest

pytestmark = pytest.mark.simulated


# ===========================================================================
# Group 7: DAQHandler Full Lifecycle Simulated Tests
# ===========================================================================


class TestHandlerNIMaxTask:
    """Validate DAQHandler lifecycle with NI MAX task (SimTask1)."""

    def test_ni_max_task_lifecycle(self, sim_task_name: str) -> None:
        """Configure DAQHandler with SimTask1, connect, introspect, disconnect.

        Verifies:
        - connect() returns True
        - get_channel_names() returns 4 channels
        - get_sample_rate() returns 10000
        - read_all_available() returns shape (n, 4)
        - disconnect() succeeds
        """
        from nidaqwrapper import DAQHandler

        handler = DAQHandler()
        try:
            handler.configure(task_in=sim_task_name)

            result = handler.connect()
            assert result is True, f"connect() failed for {sim_task_name}"

            # Introspect
            ch_names = handler.get_channel_names()
            assert len(ch_names) == 4, f"Expected 4 channels, got {len(ch_names)}"

            sample_rate = handler.get_sample_rate()
            assert sample_rate == pytest.approx(10000.0, rel=0.01), f"Expected ~10000 Hz, got {sample_rate}"

            # Let buffer fill, then read
            time.sleep(0.15)
            data = handler.read_all_available()

            assert isinstance(data, np.ndarray)
            assert data.ndim == 2, f"Expected 2D array, got shape {data.shape}"
            assert data.shape[1] == 4, f"Expected 4 channels, got {data.shape[1]}"
            assert data.shape[0] > 0, "Expected at least one sample"

        finally:
            handler.disconnect()


class TestHandlerProgrammaticAITask:
    """Validate DAQHandler with programmatic AITask."""

    def test_programmatic_aitask_lifecycle(
        self, sim_device_name: str
    ) -> None:
        """Configure DAQHandler with AITask, connect, read, disconnect.

        Creates AITask with 2 channels, configures handler, connects,
        reads data, and disconnects.
        """
        from nidaqwrapper import AITask, DAQHandler

        task = AITask("test_handler_ai", sample_rate=10000)
        task.add_channel("ai0", device=sim_device_name, channel_ind=0, units="V")
        task.add_channel("ai1", device=sim_device_name, channel_ind=1, units="V")

        handler = DAQHandler()
        try:
            handler.configure(task_in=task)

            result = handler.connect()
            assert result is True, "connect() should return True"

            # Verify metadata
            assert handler.get_channel_names() == ["ai0", "ai1"]
            assert handler.get_sample_rate() == 10000.0

            # First read auto-starts the task, may return 0 (driver init artifact)
            handler.read_all_available()
            time.sleep(0.15)

            data = handler.read_all_available()

            assert isinstance(data, np.ndarray)
            assert data.shape[1] == 2, f"Expected 2 channels, got {data.shape[1]}"
            assert data.shape[0] > 0, "Expected at least one sample"

        finally:
            handler.disconnect()


class TestHandlerRawTaskInjection:
    """Validate raw nidaqmx.Task injection via from_task()."""

    def test_raw_task_injection(self, simulated_device_name: str) -> None:
        """Pass raw nidaqmx.Task to configure(), verify it works.

        Creates raw nidaqmx.Task with AI channels, passes to configure(),
        connects, reads, disconnects.
        """
        import nidaqmx
        from nidaqwrapper import DAQHandler

        # Create raw nidaqmx task
        raw_task = nidaqmx.Task("test_raw_task")
        try:
            # Add 2 AI voltage channels
            raw_task.ai_channels.add_ai_voltage_chan(
                f"{simulated_device_name}/ai0:1"
            )
            raw_task.timing.cfg_samp_clk_timing(
                rate=10000,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
            )
            # Start the raw task before passing to handler (externally-owned)
            raw_task.start()

            handler = DAQHandler()
            try:
                # Pass raw task — should auto-wrap via AITask.from_task()
                handler.configure(task_in=raw_task)

                result = handler.connect()
                assert result is True, "connect() should return True for raw task"

                # Let buffer fill
                time.sleep(0.15)

                data = handler.read_all_available()

                assert isinstance(data, np.ndarray)
                assert data.ndim == 2
                # raw_task has 2 channels (ai0:1)
                assert data.shape[1] == 2, f"Expected 2 channels, got {data.shape[1]}"
                assert data.shape[0] > 0, "Expected at least one sample"

            finally:
                handler.disconnect()

        finally:
            # Raw task cleanup — handler doesn't own it
            try:
                raw_task.close()
            except Exception:
                pass


class TestHandlerTriggeredAcquisition:
    """Validate set_trigger() and acquire() with pyTrigger."""

    def test_triggered_acquisition(self, sim_device_name: str) -> None:
        """Configure trigger, acquire n_samples, verify shape.

        On simulated devices, triggers fire quickly because simulated noise
        crosses any threshold rapidly. Uses very low trigger level.
        """
        pytest.importorskip("pyTrigger", reason="pyTrigger not installed")

        from nidaqwrapper import AITask, DAQHandler

        n_samples = 1000

        task = AITask("test_trigger", sample_rate=10000)
        task.add_channel("ai0", device=sim_device_name, channel_ind=0, units="V")

        handler = DAQHandler()
        try:
            handler.configure(task_in=task)
            handler.connect()

            # Set trigger with very low level (simulated noise triggers quickly)
            handler.set_trigger(
                n_samples=n_samples,
                trigger_channel=0,
                trigger_level=0.0,  # Very low to trigger immediately
                trigger_type="abs",
            )

            data = handler.acquire()

            assert isinstance(data, np.ndarray)
            assert data.shape == (n_samples, 1), (
                f"Expected ({n_samples}, 1), got {data.shape}"
            )

        finally:
            handler.disconnect()

    def test_triggered_acquisition_return_dict(
        self, sim_device_name: str
    ) -> None:
        """acquire(return_dict=True) returns dict with channel names and time."""
        pytest.importorskip("pyTrigger", reason="pyTrigger not installed")

        from nidaqwrapper import AITask, DAQHandler

        n_samples = 500

        task = AITask("test_trigger_dict", sample_rate=10000)
        task.add_channel("ai0", device=sim_device_name, channel_ind=0, units="V")

        handler = DAQHandler()
        try:
            handler.configure(task_in=task)
            handler.connect()

            handler.set_trigger(
                n_samples=n_samples,
                trigger_channel=0,
                trigger_level=0.0,
                trigger_type="abs",
            )

            data = handler.acquire(return_dict=True)

            assert isinstance(data, dict)
            assert "time" in data
            assert "ai0" in data
            assert isinstance(data["ai0"], np.ndarray)
            assert len(data["ai0"]) == n_samples
            assert isinstance(data["time"], np.ndarray)
            assert len(data["time"]) == n_samples

        finally:
            handler.disconnect()


class TestHandlerSingleSample:
    """Validate single-sample read()."""

    def test_single_sample_read(self, sim_ai_task) -> None:
        """read() returns shape (n_channels,) array.

        Uses sim_ai_task fixture (2 channels).
        """
        from nidaqwrapper import DAQHandler

        handler = DAQHandler()
        try:
            handler.configure(task_in=sim_ai_task)
            handler.connect()

            data = handler.read()

            assert isinstance(data, np.ndarray)
            assert data.ndim == 1, f"Expected 1D array, got shape {data.shape}"
            assert data.shape[0] == 2, f"Expected 2 values, got {data.shape[0]}"

        finally:
            handler.disconnect()


class TestHandlerAOIntegration:
    """Validate AO integration with generate() and stop_generation()."""

    def test_ao_integration(self, sim_ai_task, sim_ao_task) -> None:
        """Configure with AI and AO, generate sine wave, stop generation.

        Uses both sim_ai_task and sim_ao_task fixtures.
        """
        from nidaqwrapper import DAQHandler

        handler = DAQHandler()
        try:
            handler.configure(task_in=sim_ai_task, task_out=sim_ao_task)

            result = handler.connect()
            assert result is True, "connect() should return True"

            # Generate a 1-second sine wave (10000 samples at 10kHz)
            t = np.linspace(0, 1, 10000, endpoint=False)
            signal = 2.0 * np.sin(2 * np.pi * 10 * t)

            handler.generate(signal)

            # Let it run briefly
            time.sleep(0.2)

            handler.stop_generation()

        finally:
            handler.disconnect()

    def test_single_sample_write(self, sim_ao_task) -> None:
        """write() sets output voltage on single channel."""
        from nidaqwrapper import DAQHandler

        handler = DAQHandler()
        try:
            handler.configure(task_out=sim_ao_task)
            handler.connect()

            # Single-sample write — validates the write mechanism works
            handler.write(1.5)

        finally:
            handler.disconnect()


class TestHandlerDigitalIntegration:
    """Validate digital I/O integration with read_digital() and write_digital()."""

    def test_digital_integration(self, sim_di_task, sim_do_task) -> None:
        """Configure with digital tasks, read and write.

        Uses sim_di_task (4 DI lines) and sim_do_task (4 DO lines) fixtures.
        """
        from nidaqwrapper import DAQHandler

        handler = DAQHandler()
        try:
            handler.configure(task_digital_in=sim_di_task, task_digital_out=sim_do_task)

            result = handler.connect()
            assert result is True, "connect() should return True"

            # Read digital input
            data = handler.read_digital()
            assert isinstance(data, np.ndarray)
            assert data.size == 4, f"Expected 4 values, got {data.size}"

            # Write digital output
            handler.write_digital([True, False, True, False])

            # No assertion needed — if no exception, operations succeeded

        finally:
            handler.disconnect()

    def test_digital_read_without_task_raises(self) -> None:
        """read_digital() raises RuntimeError if no digital input configured."""
        from nidaqwrapper import DAQHandler

        handler = DAQHandler()

        with pytest.raises(RuntimeError, match="No digital input task configured"):
            handler.read_digital()

    def test_digital_write_without_task_raises(self) -> None:
        """write_digital() raises RuntimeError if no digital output configured."""
        from nidaqwrapper import DAQHandler

        handler = DAQHandler()

        with pytest.raises(RuntimeError, match="No digital output task configured"):
            handler.write_digital(True)


class TestHandlerContextManager:
    """Validate context manager cleanup."""

    def test_context_manager_normal_exit(self, sim_device_name: str) -> None:
        """Resources released on normal with-block exit."""
        from nidaqwrapper import AITask, DAQHandler

        task_name = "test_ctx_normal"

        with DAQHandler() as handler:
            task = AITask(task_name, sample_rate=10000)
            task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )

            handler.configure(task_in=task)
            handler.connect()

            # First read auto-starts the task, may return 0 (driver init artifact)
            handler.read_all_available()
            time.sleep(0.1)

            data = handler.read_all_available()
            assert data.shape[0] > 0

        # After exit, handler should be disconnected
        # Verify by creating a new task with same name (proves cleanup)
        new_task = AITask(task_name, sample_rate=10000)
        try:
            new_task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            new_task.configure()
        finally:
            new_task.clear_task()

    def test_context_manager_exception_cleanup(
        self, sim_device_name: str
    ) -> None:
        """Resources released even when exception occurs in with-block."""
        from nidaqwrapper import AITask, DAQHandler

        task_name = "test_ctx_exception"

        with pytest.raises(RuntimeError, match="deliberate"):
            with DAQHandler() as handler:
                task = AITask(task_name, sample_rate=10000)
                task.add_channel(
                    "ai0", device=sim_device_name, channel_ind=0, units="V"
                )

                handler.configure(task_in=task)
                handler.connect()

                raise RuntimeError("deliberate test exception")

        # After exception, resources should still be released
        new_task = AITask(task_name, sample_rate=10000)
        try:
            new_task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            new_task.configure()
        finally:
            new_task.clear_task()


class TestHandlerBlockingRead:
    """Validate blocking read_all_available(n_samples)."""

    def test_blocking_read_exact_samples(self, sim_ai_task) -> None:
        """read_all_available(n_samples) blocks until n_samples available.

        Uses sim_ai_task fixture (2 AI channels at 10kHz).
        """
        from nidaqwrapper import DAQHandler

        handler = DAQHandler()
        try:
            handler.configure(task_in=sim_ai_task)
            handler.connect()

            # Blocking read for exactly 500 samples per channel
            data = handler.read_all_available(n_samples=500)

            assert isinstance(data, np.ndarray)
            assert data.shape == (500, 2), f"Expected (500, 2), got {data.shape}"

        finally:
            handler.disconnect()


# ===========================================================================
# Task Group 2: DAQHandler.acquire(blocking=False) simulated tests
# ===========================================================================


class TestHandlerNonBlockingAcquire:
    """Validate the non-blocking Future path of DAQHandler.acquire()."""

    def test_acquire_returns_future(self, sim_device_name: str) -> None:
        """acquire(blocking=False) returns a Future whose result is correct.

        Verifies:
        - Return value is a concurrent.futures.Future instance
        - future.result(timeout=15) returns a numpy array
        - Array shape is (500, 1) — (n_samples, n_channels)
        """
        pytest.importorskip("pyTrigger", reason="pyTrigger not installed")

        from nidaqwrapper import AITask, DAQHandler

        n_samples = 500
        task = AITask("test_future_result", sample_rate=10000)
        task.add_channel("ai0", device=sim_device_name, channel_ind=0, units="V")

        handler = DAQHandler()
        try:
            handler.configure(task_in=task)
            handler.connect()

            # abs trigger at level 0.0 fires immediately on simulated near-zero noise
            handler.set_trigger(
                n_samples=n_samples,
                trigger_channel=0,
                trigger_level=0.0,
                trigger_type="abs",
            )

            future = handler.acquire(blocking=False)

            # Must return a Future (not block until data is ready)
            assert isinstance(future, Future), (
                f"Expected concurrent.futures.Future, got {type(future)}"
            )

            # Block here to get the result; generous timeout for simulated device
            data = future.result(timeout=15)

            assert isinstance(data, np.ndarray)
            assert data.shape == (n_samples, 1), (
                f"Expected ({n_samples}, 1), got {data.shape}"
            )

        finally:
            handler.disconnect()

    def test_acquire_nonblocking_does_not_block(
        self, sim_device_name: str
    ) -> None:
        """acquire(blocking=False) returns in well under 1 second.

        Verifies the non-blocking contract: the call returns before
        acquisition is complete, not after.
        """
        pytest.importorskip("pyTrigger", reason="pyTrigger not installed")

        from nidaqwrapper import AITask, DAQHandler

        n_samples = 200
        task = AITask("test_nonblocking_timing", sample_rate=10000)
        task.add_channel("ai0", device=sim_device_name, channel_ind=0, units="V")

        handler = DAQHandler()
        try:
            handler.configure(task_in=task)
            handler.connect()

            handler.set_trigger(
                n_samples=n_samples,
                trigger_channel=0,
                trigger_level=0.0,
                trigger_type="abs",
            )

            t_start = time.monotonic()
            future = handler.acquire(blocking=False)
            t_elapsed = time.monotonic() - t_start

            # The call must return in well under 1 second
            assert t_elapsed < 1.0, (
                f"acquire(blocking=False) took {t_elapsed:.3f}s — should return immediately"
            )

            # Must return a Future, not data
            assert isinstance(future, Future), (
                f"Expected concurrent.futures.Future, got {type(future)}"
            )

            # Wait for the background thread to finish before disconnect
            future.result(timeout=15)

        finally:
            handler.disconnect()


# ===========================================================================
# Task Group 3 (sync-validation-simulated-coverage): acquisition abort
# ===========================================================================


class TestHandlerStopAcquisition:
    """Validate stop_acquisition() abort and is_running() on SimDev1."""

    def test_stop_acquisition_aborts_unreachable_trigger(
        self, sim_device_name: str
    ) -> None:
        """Abort an acquisition whose trigger level (99 V) can never fire.

        Verifies:
        - is_running() turns True once the acquisition loop starts
        - stop_acquisition() makes the Future resolve within a bounded
          timeout (non-daemon worker thread — a hang here hangs pytest)
        - the aborted acquisition returns the ring-buffer contents
        - the input task is stopped afterwards (is_running() False)
        """
        pytest.importorskip("pyTrigger", reason="pyTrigger not installed")

        from nidaqwrapper import AITask, DAQHandler

        n_samples = 10000
        task = AITask("test_handler_abort", sample_rate=10000)
        task.add_channel("ai0", device=sim_device_name, channel_ind=0, units="V")

        handler = DAQHandler()
        try:
            handler.configure(task_in=task)
            handler.connect()

            # 99 V is unreachable on a +-10 V simulated signal — the
            # trigger never fires and the loop polls until aborted.
            handler.set_trigger(
                n_samples=n_samples,
                trigger_channel=0,
                trigger_level=99.0,
                trigger_type="abs",
            )

            future = handler.acquire(blocking=False)
            assert isinstance(future, Future)

            # Wait (bounded) until the acquisition loop is actually running
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if handler.is_running():
                    break
                time.sleep(0.05)
            else:
                pytest.fail("acquisition never started within 5 s")

            handler.stop_acquisition()

            data = future.result(timeout=10)

            assert isinstance(data, np.ndarray), (
                "aborted acquisition must still return ring-buffer contents"
            )
            assert data.shape == (n_samples, 1), (
                f"Expected ring buffer ({n_samples}, 1), got {data.shape}"
            )

            # The aborted acquisition stops the input task
            assert handler.is_running() is False, (
                "input task must be stopped after the aborted acquisition"
            )
        finally:
            handler.disconnect()
