# Testing Guide

## Overview

nidaqwrapper uses a three-tier test strategy to balance speed, coverage, and hardware requirements:

| Tier | Tests | Requirements | Purpose |
|------|-------|-------------|---------|
| **Mocked** | 916 | None | Fast unit tests with mocked nidaqmx for CI/CD |
| **Simulated** | 81 + 1 xfail (+1 env-dependent skip) | NI-DAQmx driver + simulated devices | Real driver API validation without physical hardware |
| **Hardware** | 36 | Physical NI-DAQmx devices | Real-world timing, triggers, and signal validation |

The simulated tier bridges the gap between mocked tests and hardware tests. All 4 bugs found during hardware testing were invisible to mocked tests because `MagicMock` auto-generates any attribute on access, masking real API mismatches. Simulated devices use the real NI-DAQmx driver with simulated hardware, catching API contract violations while remaining fast and deterministic.

## Quick Start

```bash
# Mocked tests (default, no driver required, fastest)
uv run pytest

# Simulated device tests (requires NI-DAQmx driver + setup)
uv run pytest -m simulated -v

# Hardware tests (requires physical NI-DAQmx devices)
uv run pytest -m hardware -v

# Mocked + simulated (recommended for development)
uv run pytest -m "not hardware" -v

# Everything (override default excludes)
uv run pytest -o addopts="" -v
```

The default `uv run pytest` excludes both simulated and hardware tests to ensure fast, dependency-free CI runs.

## Test Tiers

### Mocked Tests (916 tests)

**What they test:**
- Public API contracts (function signatures, return types)
- Error handling and validation logic
- State transitions (connected/disconnected, started/stopped)
- Configuration save/load (TOML)
- Device enumeration logic

**How they work:**
- Use `unittest.mock.MagicMock` to replace `nidaqmx.Task` and related objects
- Tests in `tests/test_*.py` (not matching `test_sim_*` or `test_hardware*`)
- Fixtures in `tests/conftest.py` provide mocked tasks and devices

**Limitations:**
- `MagicMock` auto-generates any attribute, masking typos and API mismatches
- Cannot validate real driver behavior (timing, buffer management, error codes)
- Cannot test real data flow between channels and buffers

### Simulated Device Tests

**What they test:**
- Real nidaqmx API calls with simulated hardware
- Data type contracts (numpy arrays with correct shape and dtype)
- Driver error codes and exceptions
- Channel configuration validation (voltage ranges, terminal configs)
- Task lifecycle (create, configure, start, read/write, stop, clear)
- TOML config with real device names
- Raw task pass-through pattern (`from_task()`)
- Synchronized acquisition: finite mode, digital/analog edge start
  triggers, master/slave sample-clock routing, the MultiHandler
  synchronized burst with trigger hooks, and acquisition abort
  (`tests/test_sim_sync.py`, `tests/test_sim_multi.py`)

**What they catch that mocks miss:**
The following bugs were found only during hardware testing because mocks could not detect them:

1. **Transposed data in AOTask.generate()** — mocked tests passed, hardware test caught reversed dimensions
2. **First read(-1) returns 0 samples** — driver initialization artifact not modeled in mocks
3. **Fortran-order arrays rejected by driver** — `np.ascontiguousarray()` required for writes
4. **Missing error handling for invalid device indices** — mocks returned success, real driver raised exception

Simulated tests detect these issues without requiring physical hardware.

**Requirements:**
- NI-DAQmx driver installed (version 25.8 or later recommended)
- Simulated device `SimDev1` (PCIe-6361) configured via setup script
- Persisted NI MAX task `SimTask1` for handler tests

**Limitations:**
- AI channels return synthetic sine + noise, not real sensor data
- Counter inputs always return 0 (documented NI-DAQmx limitation)
- Triggers fire immediately (no real timing constraints) — triggered
  tests assert completion with the trigger configured, not waiting
- Analog edge triggers are accepted only via the `APFI0` terminal
- Cross-device trigger routing (e.g. `/SimDev1/PFI0` to SimDev2) needs a
  registered RTSI cable and is rejected for simulated PCIe devices
  (DaqError -89125) — full cross-device bursts are hardware-test only
- cDAQ chassis/modules cannot be simulated on Linux

### Hardware Tests (`tests/test_hardware.py`, `tests/test_hardware_task_input.py`)

The hardware tier spans two files: `tests/test_hardware.py` (the consolidated,
product-type-resolved suite — IEPE/SAR/AO rig) and the older
`tests/test_hardware_task_input.py` (task-input lifecycle on real hardware).
Both are `@pytest.mark.hardware` and run together under `-m hardware`.

**What they test:**
- Real-world timing accuracy and buffer fill rates
- IEPE/delta-sigma consecutive `add_channel()` (GitHub issue #5 — DaqError -201087)
- AO→AI loopback signal-path validation (RMS / FFT)
- Eager validation (duplicate physical AO → ValueError; over-range → -200077)
- MultiHandler software-trigger acquisition

**What hardware is needed:**
- A cDAQ chassis (e.g. NI cDAQ-9174, USB) with these C-Series modules, resolved by
  product type so the suite is enumeration-independent (cDAQ5 → cDAQ6 etc.):
  - NI 9234 (IEPE / delta-sigma AI) — the issue-#5 gate
  - NI 9215 (SAR AI)
  - NI 9260 (delta-sigma AO)
- Loopback wiring for the signal-path test: 9260 ao0 → 9215 ai0, 9260 ao1 → 9234 ai0.
- No digital module is required; digital I/O tests skip when absent.

**Running hardware tests:**
```bash
# All hardware tests across both files (serial, stop on first failure — recommended)
uv run pytest -m hardware -x -p no:randomly -v
```

## Setting Up Simulated Devices

### Linux (Ubuntu/Debian)

**Prerequisites:**
- NI-DAQmx driver installed (version 25.8 or later recommended)
- `nidaqmxconfig` command-line tool available in PATH
- `uv` installed for Python operations

**Setup Steps:**

1. From the repository root, run the setup script with sudo:
   ```bash
   sudo scripts/setup_simulated_devices.sh
   ```

2. Verify the simulated device was created:
   ```bash
   uv run python -c "import nidaqmx.system; [print(d.name, d.product_type, d.is_simulated) for d in nidaqmx.system.System.local().devices]"
   ```

   You should see output like:
   ```
   SimDev1 PCIe-6361 True
   ```

3. Run the simulated tests:
   ```bash
   uv run pytest -m simulated -v
   ```

**What the script does:**

- Imports `config/simulated_devices.ini` using `nidaqmxconfig --import` to create SimDev1
- SimDev1 is a PCIe-6361 with:
  - 16 AI channels (analog input)
  - 2 AO channels (analog output)
  - 24 DI lines (digital input)
  - 24 DO lines (digital output)
  - 4 CI channels (counter input)
  - 5 CO channels (counter output)
- Creates persisted NI MAX task `SimTask1` with 4 AI voltage channels (10 kHz, continuous mode)

**To remove the simulated device:**

Export your current configuration, manually remove the `[DAQmxDevice SimDev1]` section, and re-import:

```bash
sudo nidaqmxconfig --export current_config.ini
# Edit current_config.ini to remove SimDev1 section
sudo nidaqmxconfig --import current_config.ini
```

### Windows

**Note: The Python setup script for Windows has NOT been tested by the development team.**

**Option A: NI MAX (Recommended)**

1. Open NI MAX (Measurement & Automation Explorer)
2. In the tree view, right-click on "Devices and Interfaces"
3. Select "Create New..." → "Simulated NI-DAQmx Device or Modular Instrument"
4. From the device list, select "PCIe-6361"
5. Click OK
6. Right-click on the newly created device and select "Rename"
7. Rename it to "SimDev1"
8. Run the Python script to create the persisted task:
   ```cmd
   python scripts/setup_simulated_devices.py
   ```

**Option B: Python Script (Untested)**

```cmd
python scripts/setup_simulated_devices.py
```

Follow the printed instructions. The script will attempt to create SimDev1 and SimTask1 automatically.

### Checking Status

To check if your simulated devices and tasks are configured correctly:

```bash
python scripts/setup_simulated_devices.py --check
```

This will list all NI-DAQmx devices and persisted tasks on your system, highlighting which are simulated.

## Troubleshooting

### "Simulated device 'SimDev1' not found"

The simulated device hasn't been set up yet. See "Setting Up Simulated Devices" above.

On Linux, run:
```bash
sudo scripts/setup_simulated_devices.sh
```

On Windows, create the device in NI MAX (see Windows setup instructions above).

### Permission denied on Linux

The `nidaqmxconfig` command requires sudo privileges to modify the NI-DAQmx system configuration.

Ensure you run the setup script with sudo:
```bash
sudo scripts/setup_simulated_devices.sh
```

Do NOT run pytest with sudo. Only the one-time setup requires elevated privileges.

### nidaqmxconfig not found

The NI-DAQmx driver is not installed or not in your PATH.

Download and install the NI-DAQmx driver from [ni.com](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html).

Version 25.8 or later is recommended. After installation, `nidaqmxconfig` should be available:
```bash
which nidaqmxconfig  # Linux
where nidaqmxconfig  # Windows
```

### Tests skip silently

If you run `uv run pytest -m simulated -v` and all tests are skipped, the simulated device is not configured.

Check the skip messages in pytest output. They will indicate if SimDev1 or SimTask1 is missing.

Run the setup script to create the missing resources (see "Setting Up Simulated Devices" above).

### cDAQ modules cannot be simulated

This is a limitation of NI-DAQmx on Linux. Only PCIe, PCI, and USB multifunction devices (like the PCIe-6361) can be simulated.

cDAQ chassis and modules (like NI 9234, NI 9215, NI 9260) require real hardware. Tests that specifically require cDAQ devices are marked with `@pytest.mark.hardware` and are excluded from simulated test runs.

### Windows script fails or behaves unexpectedly

The `scripts/setup_simulated_devices.py` script has not been tested on Windows. If you encounter issues:

1. Use NI MAX to create the simulated device manually (Option A above)
2. Report the issue with detailed error messages so we can improve the script

### Driver version differences

These tests and scripts were developed and validated using NI-DAQmx version 25.8 on Ubuntu 24.04 LTS.

Older or newer driver versions may behave differently. If you encounter issues, check:
```bash
python -c "import nidaqmx; print(nidaqmx.__version__)"
```

Known compatible versions:
- NI-DAQmx 25.8 (Linux, Windows)

If you find issues with other versions, please report them.

## Writing New Tests

### Mocked Tests

Location: `tests/test_<module>.py`

Use fixtures from `tests/conftest.py`:
- `mock_task` — mocked `nidaqmx.Task`
- `mock_device` — mocked `nidaqmx.system.device.Device`
- `mock_system` — mocked `nidaqmx.system.System`

Example:
```python
def test_ai_task_configuration(mock_task):
    task = AITask('test_task', sample_rate=25600)
    task.add_channel('ch0', device='Dev1', channel_ind=0, units='V')
    task.configure()
    assert task.task_name == 'test_task'
    assert task.sample_rate == 25600
```

### Simulated Tests

Location: `tests/test_sim_<topic>.py`

Use the `@pytest.mark.simulated` decorator and fixtures from `tests/conftest.py`:
- `simulated_device_name` — returns `"SimDev1"` or skips test if not available
- `sim_device2_name` — returns `"SimDev2"` (cross-device tests) or skips
- `sim_device` — returns the `nidaqmx.system.device.Device` object
- `simulated_task_name` — returns `"SimTask1"` or skips if not available

Example:
```python
import numpy as np
import pytest
from nidaqwrapper import AITask

@pytest.mark.simulated
def test_ai_task_read(simulated_device_name):
    task = AITask('sim_test', sample_rate=10000)
    task.add_channel('ch0', device=simulated_device_name, channel_ind=0, units='V')
    task.configure()
    task.start()
    data = task.acquire(n_samples=100)
    assert data.shape == (100, 1)  # 100 samples, 1 channel
    assert data.dtype == np.float64
    task.clear_task()
```

### Hardware Tests

Location: `tests/test_hardware.py`

Use the `@pytest.mark.hardware` decorator. Clearly document required hardware in the test docstring.

Example:
```python
import pytest
from nidaqwrapper import AITask

@pytest.mark.hardware
def test_accelerometer_acquisition():
    """Test accelerometer acquisition.

    Requires:
    - cDAQ chassis with NI 9234 module in slot 1
    - Accelerometer connected to channel 0
    """
    task = AITask('accel_test', sample_rate=25600)
    task.add_channel('accel', device='cDAQ1Mod1', channel_ind=0,
                     sensitivity=100, sensitivity_units='mV/g', units='g')
    task.configure()
    task.start()
    data = task.acquire(n_samples=1000)
    assert data.shape == (1000, 1)
    task.clear_task()
```

## Continuous Integration

The default pytest configuration (`pyproject.toml`) excludes both simulated and hardware tests:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not hardware and not simulated'"
```

This ensures that `uv run pytest` runs only mocked tests, which require no NI-DAQmx driver or hardware.

To run simulated tests in CI, you would need:
1. Self-hosted runner with NI-DAQmx driver installed
2. Simulated device configured during runner setup
3. CI configuration to run: `uv run pytest -m "not hardware" -v`

CI/CD pipeline integration is deferred to a future release.
