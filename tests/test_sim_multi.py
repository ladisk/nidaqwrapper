"""Simulated device tests for MultiHandler.

Tests multi-task synchronization, trigger configuration, and the advanced
acquisition features (hardware-trigger hooks, non-blocking acquire,
acquisition abort, health introspection) against SimDev1/SimDev2.
All tests use the @pytest.mark.simulated marker.

Simulated devices auto-fire start triggers immediately — triggered tests
assert that the synchronized burst completes with triggers configured, not
that the tasks wait for a real edge.
"""

from __future__ import annotations

import time
from concurrent.futures import Future

import numpy as np
import pytest

from nidaqwrapper import AITask, DITask, MultiHandler


@pytest.mark.simulated
class TestMultiHandlerBasics:
    """Basic MultiHandler configuration and software trigger tests."""

    def test_single_task_software_trigger(self, sim_device_name):
        """Test software trigger acquisition with a single AI task.

        Creates one AITask with 2 channels, starts it, configures MultiHandler,
        verifies trigger_type is 'software', sets a low-level trigger, and
        acquires data.
        """
        # Create AITask with 2 channels
        ai_task = AITask("test_multi_single", sample_rate=10000)
        handler = None
        try:
            ai_task.add_channel(
                "ai0",
                device=sim_device_name,
                channel_ind=0,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task.add_channel(
                "ai1",
                device=sim_device_name,
                channel_ind=1,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task.configure()

            # Configure MultiHandler with the underlying nidaqmx task
            handler = MultiHandler()
            result = handler.configure(input_tasks=[ai_task.task])

            assert result is True, "configure() should return True"
            assert handler.trigger_type == "software", (
                "trigger_type should be 'software' when no hardware trigger is configured"
            )

            # Set trigger with low level (0.1V)
            handler.set_trigger(
                n_samples=100,
                trigger_channel=0,
                trigger_level=0.1,
            )

            # Acquire data
            data = handler.acquire()

            # Verify data structure
            assert isinstance(data, dict), "acquire() should return dict in software mode"
            assert "time" in data, "Result should contain 'time' key"

            # Data should have channels from the task
            channel_names = ai_task.channel_list
            for ch_name in channel_names:
                assert ch_name in data, f"Channel {ch_name} should be in result"
                assert isinstance(data[ch_name], np.ndarray), (
                    f"Channel {ch_name} data should be numpy array"
                )
        finally:
            ai_task.clear_task()
            if handler is not None:
                handler.disconnect()

    def test_multi_task_validation(self, sim_device_name):
        """Test validation when configuring multiple AI tasks.

        Creates two AITasks on different channel sets, both at 10kHz, starts
        both, configures MultiHandler with both, verifies validation runs
        (sample rate check, timing check).
        """
        # Create first AITask with channels ai0:1
        ai_task1 = AITask("test_multi_task1", sample_rate=10000)
        ai_task2 = AITask("test_multi_task2", sample_rate=10000)
        handler = None

        try:
            # Configure first task: ai0, ai1
            ai_task1.add_channel(
                "ai0",
                device=sim_device_name,
                channel_ind=0,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task1.add_channel(
                "ai1",
                device=sim_device_name,
                channel_ind=1,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task1.configure()

            # Configure second task: ai2, ai3
            ai_task2.add_channel(
                "ai2",
                device=sim_device_name,
                channel_ind=2,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task2.add_channel(
                "ai3",
                device=sim_device_name,
                channel_ind=3,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task2.configure()

            # Configure MultiHandler with both tasks
            handler = MultiHandler()
            result = handler.configure(
                input_tasks=[ai_task1.task, ai_task2.task]
            )

            # Validation should pass — both tasks have same sample rate and timing
            assert result is True, (
                "configure() should return True when tasks have matching sample rates"
            )

            # Both tasks should be stored
            assert len(handler.input_tasks) == 2, (
                "MultiHandler should store both input tasks"
            )

            # Verify sample rate was cached from first task
            assert handler.input_sample_rate == 10000, (
                "input_sample_rate should match the configured rate"
            )
        finally:
            ai_task1.clear_task()
            ai_task2.clear_task()
            if handler is not None:
                handler.disconnect()

    def test_trigger_type_detection_no_hardware(self, sim_device_name):
        """Test that trigger_type is set to 'software' when no hardware triggers exist.

        Creates a task without hardware triggers, configures MultiHandler,
        verifies trigger_type == 'software'.
        """
        ai_task = AITask("test_trigger_detect", sample_rate=10000)
        handler = None
        try:
            ai_task.add_channel(
                "ai0",
                device=sim_device_name,
                channel_ind=0,
                units="V",
                min_val=-10.0,
                max_val=10.0,
            )
            ai_task.configure()

            # Configure MultiHandler
            handler = MultiHandler()
            result = handler.configure(input_tasks=[ai_task.task])

            assert result is True, "configure() should succeed"
            assert handler.trigger_type == "software", (
                "trigger_type should default to 'software' when no hardware "
                "trigger is configured (bug fix #2)"
            )
        finally:
            ai_task.clear_task()
            if handler is not None:
                handler.disconnect()


@pytest.mark.simulated
class TestMultiHandlerSampleRateMismatch:
    """Test validation failure when tasks have different sample rates."""

    def test_sample_rate_mismatch_rejected(self, sim_device_name):
        """Test that configure() returns False when tasks have different sample rates.

        Creates two AITasks with different sample rates, verifies configure() rejects them.
        """
        ai_task1 = AITask("test_mismatch1", sample_rate=10000)
        ai_task2 = AITask("test_mismatch2", sample_rate=20000)
        handler = None

        try:
            # Configure both tasks
            ai_task1.add_channel(
                "ai0",
                device=sim_device_name,
                channel_ind=0,
                units="V",
            )
            ai_task1.configure()

            ai_task2.add_channel(
                "ai1",
                device=sim_device_name,
                channel_ind=1,
                units="V",
            )
            ai_task2.configure()

            # Configure should fail due to sample rate mismatch
            handler = MultiHandler()
            result = handler.configure(
                input_tasks=[ai_task1.task, ai_task2.task]
            )

            assert result is False, (
                "configure() should return False when tasks have different sample rates"
            )
        finally:
            ai_task1.clear_task()
            ai_task2.clear_task()
            if handler is not None:
                handler.disconnect()


@pytest.mark.simulated
class TestMultiHandlerSynchronizedBurst:
    """OpenEOL synchronized-burst pattern end to end on simulated devices."""

    def test_openeol_ai_di_burst_with_hooks(self, sim_device_name):
        """Full OpenEOL pattern: AI master + DI slave, shared PFI trigger.

        Finite AI (onboard clock) and DI (slaved to /SimDev1/ai/SampleClock)
        share a digital edge trigger on /SimDev1/PFI0. configure() must now
        return True with the clock-source UserWarning (relaxed FR-5.5).
        acquire(custom_mode='x') must call the support hook with 'x' before
        arming and the start hook after arming, and return the nested
        {task: {channel: data}} result.
        """
        n_samples = 1000
        rate = 10000
        events: list = []

        ai = AITask("test_multi_burst_ai", sample_rate=rate)
        di = DITask("test_multi_burst_di", sample_rate=rate)
        handler = None
        try:
            ai.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            ai.add_channel(
                "ai1", device=sim_device_name, channel_ind=1, units="V"
            )
            ai.configure(sample_mode="finite", samples_per_channel=n_samples)
            ai.set_start_trigger(f"/{sim_device_name}/PFI0", edge="rising")

            di.add_channel(
                "di_ch", lines=f"{sim_device_name}/port0/line0:3"
            )
            di.configure(
                sample_mode="finite",
                samples_per_channel=n_samples,
                clock_source=f"/{sim_device_name}/ai/SampleClock",
            )
            di.set_start_trigger(f"/{sim_device_name}/PFI0", edge="rising")

            handler = MultiHandler()
            # Each hook also captures ai.task.is_task_done() to prove the
            # order relative to ARMING (not just relative hook order):
            # before arming the task was never started -> is_task_done()
            # is True; after arming the 100 ms finite acquisition is in
            # flight -> False. Hook-order-only assertions would also pass
            # if both hooks ran before (or after) arming.
            handler.set_hardware_trigger_functions(
                start_function=lambda: events.append(
                    ("start", ai.task.is_task_done())
                ),
                support_function=lambda mode: events.append(
                    ("support", mode, ai.task.is_task_done())
                ),
            )

            # DI slave first in the list so it arms before the AI master
            # starts the shared sample clock.
            with pytest.warns(UserWarning, match="sample-clock sources"):
                result = handler.configure(input_tasks=[di.task, ai.task])

            assert result is True, (
                "configure() must accept the master/slave pattern after the "
                "FR-5.5 relaxation"
            )
            assert handler.trigger_type == "hardware"

            data = handler.acquire(custom_mode="x")

            # Hook order: support('x') before arming (task not yet
            # started, is_task_done() True), start after arming (finite
            # acquisition in flight, is_task_done() False)
            assert events == [("support", "x", True), ("start", False)], (
                f"Expected [('support', 'x', True), ('start', False)], "
                f"got {events}"
            )

            # Nested {task: {channel: data}} result with both tasks complete
            assert isinstance(data, dict)
            assert set(data.keys()) == {ai.task.name, di.task.name}
            assert set(data[ai.task.name].keys()) == {"ai0", "ai1"}
            for channel_data in data[ai.task.name].values():
                assert isinstance(channel_data, np.ndarray)
                assert channel_data.shape == (n_samples,)
            assert len(data[di.task.name]) == 4, "expected 4 DI lines"
            for channel_data in data[di.task.name].values():
                assert isinstance(channel_data, np.ndarray)
                assert channel_data.shape == (n_samples,)
        finally:
            ai.clear_task()
            di.clear_task()
            if handler is not None:
                handler.disconnect()

    def test_cross_device_burst(self, sim_device_name, sim_device2_name):
        """Cross-device burst: AI on SimDev1 + AI on SimDev2, shared trigger.

        Both finite tasks trigger on /SimDev1/PFI0. The clock-source
        readbacks differ (/SimDev1/... vs /SimDev2/...), so configure()
        warns and passes — this is the regression the FR-5.5 relaxation
        fixes and it is asserted unconditionally.

        The burst itself requires the driver to route /SimDev1/PFI0 to
        SimDev2, which needs a registered RTSI cable — unavailable for
        simulated PCIe devices. When the driver rejects the route
        (DaqError -89125) the acquire portion is skipped with the reason;
        full cross-device acquisition remains a hardware test.
        """
        from nidaqmx.errors import DaqError

        n_samples = 1000
        rate = 10000

        ai1 = AITask("test_multi_xdev_ai1", sample_rate=rate)
        ai2 = AITask("test_multi_xdev_ai2", sample_rate=rate)
        handler = None
        try:
            ai1.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            ai1.configure(sample_mode="finite", samples_per_channel=n_samples)
            ai1.set_start_trigger(f"/{sim_device_name}/PFI0", edge="rising")

            ai2.add_channel(
                "ai0", device=sim_device2_name, channel_ind=0, units="V"
            )
            ai2.configure(sample_mode="finite", samples_per_channel=n_samples)
            ai2.set_start_trigger(f"/{sim_device_name}/PFI0", edge="rising")

            handler = MultiHandler()
            with pytest.warns(UserWarning, match="sample-clock sources"):
                result = handler.configure(input_tasks=[ai1.task, ai2.task])

            assert result is True, (
                "configure() must accept cross-device tasks after the "
                "FR-5.5 relaxation"
            )
            assert handler.trigger_type == "hardware"

            try:
                data = handler.acquire()
            except DaqError as exc:
                if exc.error_code == -89125:
                    pytest.skip(
                        "cross-device trigger routing requires a registered "
                        "RTSI cable — not available between simulated PCIe "
                        f"devices: {exc}"
                    )
                raise

            assert set(data.keys()) == {ai1.task.name, ai2.task.name}
            for task_data in data.values():
                for channel_data in task_data.values():
                    assert channel_data.shape == (n_samples,)
        finally:
            ai1.clear_task()
            ai2.clear_task()
            if handler is not None:
                handler.disconnect()


@pytest.mark.simulated
class TestMultiHandlerNonBlockingAcquire:
    """acquire(blocking=False) returns a Future on simulated devices."""

    def test_acquire_nonblocking_returns_future(self, sim_device_name):
        """acquire(blocking=False) returns a Future resolving to the data."""
        pytest.importorskip("pyTrigger", reason="pyTrigger not installed")

        n_samples = 500
        ai = AITask("test_multi_nonblocking", sample_rate=10000)
        handler = None
        try:
            ai.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            ai.configure()

            handler = MultiHandler()
            assert handler.configure(input_tasks=[ai.task]) is True
            # abs trigger at 0.0 fires immediately on simulated noise
            handler.set_trigger(
                n_samples=n_samples, trigger_channel=0, trigger_level=0.0
            )

            future = handler.acquire(blocking=False)

            assert isinstance(future, Future), (
                f"Expected concurrent.futures.Future, got {type(future)}"
            )

            data = future.result(timeout=15)

            assert isinstance(data, dict)
            assert "time" in data
            assert "ai0" in data
            assert len(data["ai0"]) == n_samples
        finally:
            ai.clear_task()
            if handler is not None:
                handler.disconnect()


@pytest.mark.simulated
class TestMultiHandlerStopAcquisition:
    """stop_acquisition() aborts a never-firing software trigger."""

    def test_stop_acquisition_aborts_unreachable_trigger(
        self, sim_device_name
    ):
        """Abort an acquisition whose trigger level (99 V) can never fire.

        The Future must resolve with the ring-buffer contents within a
        bounded timeout and the input task must be stopped afterwards.
        """
        pytest.importorskip("pyTrigger", reason="pyTrigger not installed")

        n_samples = 10000
        ai = AITask("test_multi_abort", sample_rate=10000)
        handler = None
        try:
            ai.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            ai.configure()

            handler = MultiHandler()
            assert handler.configure(input_tasks=[ai.task]) is True
            # 99 V is unreachable on a +-10 V simulated signal — the
            # trigger never fires and the loop polls until aborted.
            handler.set_trigger(
                n_samples=n_samples, trigger_channel=0, trigger_level=99.0
            )

            future = handler.acquire(blocking=False)

            # Wait (bounded) until the acquisition loop is actually running
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if handler.is_running():
                    break
                time.sleep(0.05)
            else:
                pytest.fail("acquisition never started within 5 s")

            handler.stop_acquisition()

            # Bounded wait — the worker thread is non-daemon; a hang here
            # would hang pytest (Batch 9 finding), hence the timeout.
            data = future.result(timeout=10)

            assert isinstance(data, dict), (
                "aborted acquisition must still return ring-buffer contents"
            )
            assert "ai0" in data

            # The aborted acquisition stops the input task
            assert handler.is_running() is False, (
                "input task must be stopped after the aborted acquisition"
            )
        finally:
            ai.clear_task()
            if handler is not None:
                handler.disconnect()


@pytest.mark.simulated
class TestMultiHandlerIntrospection:
    """check_state(), is_running(), get_device_info() on simulated devices."""

    def test_health_introspection(self, sim_device_name):
        """check_state() == 'connected', is_running() reflects task state,
        get_device_info() lists channels and rates."""
        rate = 10000
        ai = AITask("test_multi_introspect", sample_rate=rate)
        handler = None
        try:
            ai.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            ai.add_channel(
                "ai1", device=sim_device_name, channel_ind=1, units="V"
            )
            ai.configure()

            handler = MultiHandler()
            assert handler.configure(input_tasks=[ai.task]) is True
            assert handler.connect() is True
            assert handler.check_state() == "connected"

            # Not started yet — nothing running
            assert handler.is_running() is False

            ai.task.start()
            assert handler.is_running() is True

            ai.task.stop()
            assert handler.is_running() is False

            info = handler.get_device_info()
            assert set(info.keys()) == {"input"}
            assert set(info["input"].keys()) == {ai.task.name}
            task_info = info["input"][ai.task.name]
            assert task_info["channel_names"] == ["ai0", "ai1"]
            assert task_info["sample_rate"] == pytest.approx(rate, rel=0.01)
        finally:
            ai.clear_task()
            if handler is not None:
                handler.disconnect()
