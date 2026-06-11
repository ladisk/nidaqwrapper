"""Simulated-device regression tests for GitHub issues #5-#8.

All tests in this module run against NI-DAQmx simulated devices (not mocks).
They are marked with ``@pytest.mark.simulated`` and use the simulated device
fixtures from conftest.py.

Run with::

    uv run pytest tests/test_sim_issues.py -v -m simulated

Covered scenarios (OpenSpec change fix-gh-issues-5-8):
- Issue #5: consecutive add_channel() calls must not trip implicit task
  verification (regression guard for the add -> cfg timing -> TASK_VERIFY
  sequence; the -201087 IEPE case itself needs real cDAQ hardware, see
  test_hardware.py::TestIEPEConsecutiveAdd).
- Issue #6: invalid channel parameters (out-of-range AO limits, -200077)
  raise inside the offending add_channel() call.
- Issue #7/#8: save() -> from_name() -> acquire() NI MAX round-trip and
  stop() after start().

Device Configuration
---------------------
- SimDev1 : PCIe-6361 (simulated) — 16 AI, 2 AO, 24 DI lines, 24 DO lines
"""

from __future__ import annotations

import time

import pytest
from nidaqmx.errors import DaqError

pytestmark = pytest.mark.simulated


class TestEagerValidationSimulated:
    """Issue #6: validation errors surface inside add_channel()."""

    def test_insane_ao_range_raises_inside_add_channel(self, sim_device_name):
        """An out-of-range AO min/max raises DaqError inside add_channel().

        Before fix-gh-issues-5-8 the driver deferred this validation to the
        next verification-forcing operation (e.g. configure() or a rate
        read), pointing the traceback at the wrong line.
        """
        from nidaqwrapper import AOTask

        task = AOTask("test_issue6_ao_range", sample_rate=10000)
        try:
            with pytest.raises(DaqError) as exc_info:
                task.add_channel(
                    "ao0",
                    device=sim_device_name,
                    channel_ind=0,
                    min_val=-1e5,
                    max_val=1e5,
                )
            assert exc_info.value.error_code == -200077, (
                f"Expected -200077 (value not supported), "
                f"got {exc_info.value.error_code}"
            )
        finally:
            task.clear_task()

    def test_ai_two_channel_consecutive_add(self, sim_device_name):
        """Issue #5 regression guard: consecutive adds work with the new
        add -> cfg_samp_clk_timing -> TASK_VERIFY sequence, and the task
        remains fully usable afterwards."""
        from nidaqwrapper import AITask

        task = AITask("test_issue5_two_adds", sample_rate=10000)
        try:
            task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            task.add_channel(
                "ai1", device=sim_device_name, channel_ind=1, units="V"
            )
            assert task.channel_list == ["ai0", "ai1"]

            task.configure()
            task.start()
            data = task.acquire(n_samples=50)
            assert data.shape == (50, 2)
        finally:
            task.clear_task()


class TestSaveRoundTripSimulated:
    """Issues #7/#8: save() -> from_name() -> acquire() NI MAX round-trip."""

    TASK_NAME = "nidaqw_issue8_roundtrip"

    @staticmethod
    def _delete_persisted_task(name: str) -> None:
        """Delete a persisted NI MAX task, ignoring 'does not exist' errors."""
        from nidaqmx.system.storage.persisted_task import PersistedTask

        try:
            PersistedTask(name).delete()
        except DaqError:
            pass

    def test_save_from_name_acquire_round_trip(self, sim_device_name):
        """The reporter's issue #8 workflow, via the supported API:
        AITask.save() persists, AITask.from_name() reloads an owned wrapper,
        acquire() returns data."""
        from nidaqwrapper import AITask

        # Defensive cleanup of leftovers from earlier aborted runs
        self._delete_persisted_task(self.TASK_NAME)

        loaded = None
        try:
            task = AITask(self.TASK_NAME, sample_rate=10000)
            task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            task.configure()
            task.save(clear_task=True)  # AITask's historical default, explicit
            assert task.task is None  # save(clear_task=True) released the task

            loaded = AITask.from_name(self.TASK_NAME)
            assert loaded._owns_task is True
            assert loaded.number_of_ch == 1

            loaded.start()
            data = loaded.acquire(n_samples=500)
            assert data.shape == (500, 1)
        finally:
            if loaded is not None:
                try:
                    loaded.clear_task()
                except Exception:
                    pass
            self._delete_persisted_task(self.TASK_NAME)


class TestStopSimulated:
    """Issue #8 gap: task classes provide stop(), symmetric to start()."""

    def test_stop_after_start(self, sim_device_name):
        """stop() halts a running task; the task remains restartable."""
        from nidaqwrapper import AITask

        task = AITask("test_stop_sim", sample_rate=10000)
        try:
            task.add_channel(
                "ai0", device=sim_device_name, channel_ind=0, units="V"
            )
            task.configure()
            task.start()
            time.sleep(0.05)
            task.stop()
            assert task.task.is_task_done() is True

            # The stopped task is restartable — stop() did not release it
            task.start()
            data = task.acquire(n_samples=10)
            assert data.shape == (10, 1)
        finally:
            task.clear_task()
