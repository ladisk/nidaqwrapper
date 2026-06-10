"""Simulated tests for synchronized timing configuration (task level).

Covers the task-level scenarios of OpenSpec change
sync-validation-simulated-coverage: finite acquisition mode, digital edge
start triggers, analog edge start triggers via APFI0, and master/slave
sample-clock routing on SimDev1 (simulated PCIe-6361).

Run with::

    uv run pytest tests/test_sim_sync.py -v -m simulated

Simulated-device constraints (probed 2026-06-11 on simulated PCIe-6361)
-----------------------------------------------------------------------
- Simulated devices auto-fire start triggers immediately. Triggered tests
  therefore assert that the acquisition *completes with the trigger
  configured* — real trigger-wait semantics remain hardware-test only.
- Analog edge triggering is accepted only via the ``APFI0`` terminal;
  ``/SimDev1/ai0`` is rejected by the driver ("invalid analog trigger
  source").
- A master task must use the onboard clock — configuring its clock_source
  to its own exported terminal raises a "same terminal" DaqError. Only
  slave tasks reference ``/SimDev1/ai/SampleClock``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.simulated


# ===========================================================================
# Finite acquisition mode — blocking acquire returns the full sample count
# ===========================================================================


class TestFiniteAcquisition:
    """Finite-mode configure() + blocking acquire on AITask and DITask."""

    def test_finite_ai_acquire(self, simulated_device_name: str) -> None:
        """Finite AITask acquires exactly N samples with shape (N, 1)."""
        from nidaqwrapper import AITask

        n_samples = 1000
        ai = AITask("test_sync_finite_ai", sample_rate=10000)
        try:
            ai.add_channel(
                "ai0", device=simulated_device_name, channel_ind=0, units="V"
            )
            ai.configure(sample_mode="finite", samples_per_channel=n_samples)
            ai.start()

            data = ai.acquire(n_samples)

            assert isinstance(data, np.ndarray)
            assert data.shape == (n_samples, 1), (
                f"Expected ({n_samples}, 1), got {data.shape}"
            )
        finally:
            ai.clear_task()

    def test_finite_di_acquire(self, simulated_device_name: str) -> None:
        """Finite DITask acquires exactly N samples with shape (N, 4)."""
        from nidaqwrapper import DITask

        n_samples = 500
        di = DITask("test_sync_finite_di", sample_rate=10000)
        try:
            di.add_channel(
                "di_ch", lines=f"{simulated_device_name}/port0/line0:3"
            )
            di.configure(sample_mode="finite", samples_per_channel=n_samples)
            di.start()

            data = di.acquire(n_samples)

            assert isinstance(data, np.ndarray)
            assert data.shape == (n_samples, 4), (
                f"Expected ({n_samples}, 4), got {data.shape}"
            )
        finally:
            di.clear_task()


# ===========================================================================
# Digital edge start trigger — /SimDev1/PFI0 (auto-fired by simulation)
# ===========================================================================


class TestDigitalEdgeTrigger:
    """set_start_trigger() on AI and DI finite tasks completes acquisition."""

    def test_ai_digital_trigger_completes(
        self, simulated_device_name: str
    ) -> None:
        """Finite AI with digital edge trigger on PFI0 acquires fully."""
        from nidaqwrapper import AITask

        n_samples = 1000
        ai = AITask("test_sync_trig_ai", sample_rate=10000)
        try:
            ai.add_channel(
                "ai0", device=simulated_device_name, channel_ind=0, units="V"
            )
            ai.configure(sample_mode="finite", samples_per_channel=n_samples)
            ai.set_start_trigger(
                f"/{simulated_device_name}/PFI0", edge="rising"
            )
            ai.start()

            # Simulated devices fire the trigger immediately — assert the
            # triggered acquisition completes, not that the task waits.
            data = ai.acquire(n_samples)

            assert data.shape == (n_samples, 1), (
                f"Expected ({n_samples}, 1), got {data.shape}"
            )
        finally:
            ai.clear_task()

    def test_di_digital_trigger_completes(
        self, simulated_device_name: str
    ) -> None:
        """Finite DI with digital edge trigger on PFI0 acquires fully."""
        from nidaqwrapper import DITask

        n_samples = 500
        di = DITask("test_sync_trig_di", sample_rate=10000)
        try:
            di.add_channel(
                "di_ch", lines=f"{simulated_device_name}/port0/line0:3"
            )
            di.configure(sample_mode="finite", samples_per_channel=n_samples)
            di.set_start_trigger(
                f"/{simulated_device_name}/PFI0", edge="falling"
            )
            di.start()

            data = di.acquire(n_samples)

            assert data.shape == (n_samples, 4), (
                f"Expected ({n_samples}, 4), got {data.shape}"
            )
        finally:
            di.clear_task()


# ===========================================================================
# Analog edge start trigger — APFI0 only (driver rejects /SimDev1/ai0)
# ===========================================================================


class TestAnalogEdgeTrigger:
    """set_analog_start_trigger() via APFI0 completes acquisition."""

    def test_ai_analog_trigger_apfi0_completes(
        self, simulated_device_name: str
    ) -> None:
        """Finite AI with analog edge trigger on APFI0 acquires fully.

        Exercises the slope and level kwargs explicitly. APFI0 is the only
        analog trigger source the driver accepts on the simulated PCIe-6361
        (an AI terminal like '/SimDev1/ai0' raises DaqError).
        """
        from nidaqwrapper import AITask

        n_samples = 1000
        ai = AITask("test_sync_anlg_trig_ai", sample_rate=10000)
        try:
            ai.add_channel(
                "ai0", device=simulated_device_name, channel_ind=0, units="V"
            )
            ai.configure(sample_mode="finite", samples_per_channel=n_samples)
            ai.set_analog_start_trigger("APFI0", slope="falling", level=0.5)
            ai.start()

            data = ai.acquire(n_samples)

            assert data.shape == (n_samples, 1), (
                f"Expected ({n_samples}, 1), got {data.shape}"
            )
        finally:
            ai.clear_task()


# ===========================================================================
# Master/slave sample-clock routing — AI onboard master, DI slave
# ===========================================================================


class TestMasterSlaveClockRouting:
    """AI master (onboard clock) + DI slave on /SimDev1/ai/SampleClock."""

    def test_ai_master_di_slave_both_acquire_fully(
        self, simulated_device_name: str
    ) -> None:
        """Both finite tasks acquire their full sample counts.

        The master uses the onboard clock (setting its clock_source to its
        own exported terminal raises a 'same terminal' DaqError); the slave
        references the exported '/SimDev1/ai/SampleClock'. The slave starts
        first so it is armed before the master's clock begins.
        """
        from nidaqwrapper import AITask, DITask

        n_samples = 1000
        rate = 10000
        ai = AITask("test_sync_master_ai", sample_rate=rate)
        di = DITask("test_sync_slave_di", sample_rate=rate)
        try:
            ai.add_channel(
                "ai0", device=simulated_device_name, channel_ind=0, units="V"
            )
            ai.configure(sample_mode="finite", samples_per_channel=n_samples)

            di.add_channel(
                "di_ch", lines=f"{simulated_device_name}/port0/line0:3"
            )
            di.configure(
                sample_mode="finite",
                samples_per_channel=n_samples,
                clock_source=f"/{simulated_device_name}/ai/SampleClock",
            )

            # Slave arms first, master starts the shared clock
            di.start()
            ai.start()

            ai_data = ai.acquire(n_samples)
            di_data = di.acquire(n_samples)

            assert ai_data.shape == (n_samples, 1), (
                f"Expected ({n_samples}, 1), got {ai_data.shape}"
            )
            assert di_data.shape == (n_samples, 4), (
                f"Expected ({n_samples}, 4), got {di_data.shape}"
            )

            # Document WHY MultiHandler._validate_timing was relaxed: the
            # readback strings of master and slave differ even though the
            # slave is driven by the master's clock.
            assert (
                ai.task.timing.samp_clk_src != di.task.timing.samp_clk_src
            ), "master/slave clock-source readbacks unexpectedly identical"
        finally:
            ai.clear_task()
            di.clear_task()


# ===========================================================================
# AO slaved to the AI sample clock — probe (skip if the driver rejects it)
# ===========================================================================


class TestAOSlaveClockProbe:
    """AOTask slaved to /SimDev1/ai/SampleClock while an AI task runs."""

    def test_ao_slaved_to_ai_sample_clock(
        self, simulated_device_name: str
    ) -> None:
        """AO generation on the exported AI sample clock, or skip.

        Probes whether the simulated PCIe-6361 accepts an AO task slaved to
        '/SimDev1/ai/SampleClock' while an AI task is running. The route may
        be rejected on simulated devices — in that case the test skips with
        the DaqError reason (design decision 4 of
        sync-validation-simulated-coverage).
        """
        from nidaqmx.errors import DaqError

        from nidaqwrapper import AITask, AOTask

        rate = 10000
        ai = AITask("test_sync_ao_probe_ai", sample_rate=rate)
        ao = AOTask(
            "test_sync_ao_probe_ao",
            sample_rate=rate,
            samples_per_channel=1000,
        )
        try:
            # Continuous AI master keeps the exported clock running
            ai.add_channel(
                "ai0", device=simulated_device_name, channel_ind=0, units="V"
            )
            ai.configure()
            ai.start()

            ao.add_channel(
                "ao0",
                device=simulated_device_name,
                channel_ind=0,
                min_val=-10.0,
                max_val=10.0,
            )

            # Only the route itself may be rejected — keep the DaqError
            # net narrow (configure + the auto-starting generate, where
            # route reservation can also surface). stop() stays outside:
            # a failure there would be a genuine bug, not a rejected route.
            try:
                ao.configure(
                    clock_source=f"/{simulated_device_name}/ai/SampleClock"
                )
                t = np.arange(1000) / rate
                signal = 2.0 * np.sin(2 * np.pi * 100 * t)
                ao.generate(signal)  # auto-starts on the slaved clock
            except DaqError as exc:
                pytest.skip(
                    "AO slaving to "
                    f"/{simulated_device_name}/ai/SampleClock rejected on "
                    f"simulated PCIe-6361: {exc}"
                )

            time.sleep(0.2)
            ao.task.stop()
        finally:
            ao.clear_task()
            ai.clear_task()
