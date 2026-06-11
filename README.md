# nidaqwrapper

Unified NI-DAQmx Python Wrapper.

A Python package that provides a clean, high-level interface to NI-DAQmx hardware. It consolidates analog input, analog output, and digital I/O into a single package with two layers: task classes for channel configuration and orchestrator classes for acquisition lifecycle management.

The architecture uses direct delegation -- `nidaqmx.Task` is the single source of truth. No intermediate state is maintained; every channel addition and timing configuration delegates immediately to the underlying driver.

## Installation

```bash
pip install nidaqwrapper
```

**Requires NI-DAQmx drivers installed on the system.** The `nidaqmx` Python package communicates with the NI-DAQmx C driver, which must be installed separately from [ni.com](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html).

For development:

```bash
git clone https://github.com/tibor-barsi/nidaqwrapper.git
cd nidaqwrapper
pip install -e ".[dev]"
```

## Quick Start

Accelerometer acquisition with software triggering:

```python
from nidaqwrapper import AITask, DAQHandler

# Define an analog input task with two accelerometer channels
task = AITask('vibration_test', sample_rate=25600)
task.add_channel('accel_x', device="Dev1", channel_ind=0,
                 sensitivity=100, sensitivity_units='mV/g', units='g')
task.add_channel('accel_y', device="Dev1", channel_ind=1,
                 sensitivity=100, sensitivity_units='mV/g', units='g')

# Use DAQHandler for triggered acquisition
wrapper = DAQHandler()
wrapper.configure(task_in=task)
wrapper.connect()
wrapper.set_trigger(n_samples=25600, trigger_channel=0,
                    trigger_level=0.5, trigger_type='up', presamples=2560)
data = wrapper.acquire()  # shape: (25600, 2)
wrapper.disconnect()
```

## Features

- **Analog input** (`AITask`) -- voltage, accelerometer (IEPE), force (IEPE), and custom linear scales
- **Analog output** (`AOTask`) -- voltage generation with continuous buffer regeneration
- **Digital I/O** (`DITask` / `DOTask`) -- on-demand single-sample and clocked continuous modes
- **Single-task handler** (`DAQHandler`) -- configure, connect, acquire/generate, disconnect lifecycle with software triggering via pyTrigger
- **Multi-task synchronization** (`MultiHandler`) -- hardware-triggered finite acquisition and validated multi-task pipelines, with trigger-firing hooks and non-blocking acquisition
- **Synchronized timing** -- finite/continuous acquisition modes, shared sample clocks (`clock_source`), and digital/analog edge start triggers (`set_start_trigger()`, `set_analog_start_trigger()`) on all task classes
- **Acquisition control** -- `stop_acquisition()` cooperative abort, `is_running()`, `check_state()` auto-reconnection on both handlers
- **Eager channel validation** -- `add_channel()` validates the configuration with the driver immediately (`TASK_VERIFY`), so invalid parameters raise on the offending line and IEPE/delta-sigma modules (NI 9234 etc.) work correctly (v0.2.0 behavior change: errors that previously surfaced at a later operation now surface inside `add_channel()`)
- **NI MAX persistence** -- `save()` on all task classes persists the task to NI MAX; `from_name()` loads it back as a wrapper
- **Task lifecycle** -- `configure()` / `start()` / `stop()` / `clear_task()` on all task classes
- **TOML configuration** -- `save_config()` / `from_config()` for portable, human-readable task definitions with device aliases
- **Device discovery** -- `list_devices()`, `list_tasks()`, `get_connected_devices()` for hardware enumeration
- **Raw task injection** -- `from_task()` on all task classes wraps pre-configured `nidaqmx.Task` objects
- **Factory classmethods** -- `from_name()` creates tasks from a saved NI MAX task name, `from_config()` from TOML
- **System introspection** -- `system_info()` returns structured device/driver/task inventory
- **Context manager support** -- automatic resource cleanup with `with` statements
- **Thread safety** -- `DAQHandler` and `MultiHandler` use per-instance `RLock` for concurrent access

## Usage Examples

### Load an NI MAX task by name

```python
from nidaqwrapper import DAQHandler

wrapper = DAQHandler()
wrapper.configure(task_in='MyInputTask', task_out='MyOutputTask')
wrapper.connect()
data = wrapper.acquire()
wrapper.disconnect()
```

### NI MAX round-trip (save a task, reload it by name)

Define a task programmatically once, persist it to NI MAX with `save()`,
and reload it anywhere with `from_name()` -- no reconfiguration needed:

```python
from nidaqwrapper import AITask

# Define and persist
task = AITask('vibration', sample_rate=25600)
task.add_channel('acc0', device='cDAQ1Mod1', channel_ind=0,
                 sensitivity=100, sensitivity_units='mV/g', units='g')
task.configure()
task.save()  # persists to NI MAX; AITask's default clear_task=True releases the handle

# Later (or in another script): reload and acquire
task = AITask.from_name('vibration')
task.start()
data = task.acquire(1000)   # (1000, n_channels)
task.stop()
task.clear_task()
```

`AOTask`, `DITask`, and `DOTask` provide the same `save()` / `from_name()`
pair (their `save()` keeps the task open by default, `clear_task=False`).
Note: `get_task_by_name()` returns a *raw* `nidaqmx.task.Task`; use
`from_name()` when you want a wrapper with `acquire()` / `generate()`.

### Digital output

```python
from nidaqwrapper import DOTask

with DOTask('relay_control') as do:
    do.add_channel('relays', lines='Dev1/port0/line0:3')
    do.configure()
    do.write([True, False, True, False])
```

### Digital input

```python
from nidaqwrapper import DITask

with DITask('switches') as di:
    di.add_channel('sw', lines='Dev1/port0/line0:3')
    di.configure()
    state = di.read()  # array of bool values, one per line
```

### Analog output (continuous waveform)

```python
import numpy as np
from nidaqwrapper import AOTask

task = AOTask('sig_gen', sample_rate=10000)
task.add_channel('ao_0', device="Dev1", channel_ind=0)
task.configure()
task.start()

t = np.linspace(0, 1, 10000)
signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine
task.generate(signal)
# ...
task.clear_task()
```

### TOML configuration (portable across machines)

```python
from nidaqwrapper import AITask

# Save task configuration
task = AITask('vibration', sample_rate=25600)
task.add_channel('ch0', device="Dev1", channel_ind=0,
                 sensitivity=100, sensitivity_units='mV/g', units='g')
task.save_config('vibration.toml')
task.clear_task()

# Recreate the same task on another machine
task = AITask.from_config('vibration.toml')
task.configure()
task.start()
```

The generated TOML file uses device aliases, so only the `[devices]` section needs editing when moving between machines:

```toml
[task]
name = "vibration"
sample_rate = 25600
type = "input"

[devices]
dev0 = "cDAQ1Mod1"  # NI 9234

[[channels]]
name = "ch0"
device = "dev0"
channel = 0
sensitivity = 100
sensitivity_units = "mV/g"
units = "g"
```

### Multi-task synchronized acquisition

```python
from nidaqwrapper import MultiHandler

adv = MultiHandler()
adv.configure(input_tasks=[task1, task2])
adv.connect()
adv.set_trigger(n_samples=25600, trigger_channel=0, trigger_level=0.5)
data = adv.acquire()
adv.disconnect()
```

### Hardware-triggered synchronized burst (no raw nidaqmx needed)

Build finite tasks sharing one sample clock and one start trigger entirely with
nidaqwrapper, then acquire them as a synchronized burst:

```python
from nidaqwrapper import AITask, DITask, MultiHandler

# Master analog task: finite acquisition, onboard clock
ai = AITask('vibration', sample_rate=25600)
ai.add_channel('acc0', device='cDAQ1Mod1', channel_ind=0,
               sensitivity=100.0, sensitivity_units='mV/g', units='g')
ai.configure(sample_mode='finite', samples_per_channel=25600,
             clock_source='/cDAQ1Mod1/ai/SampleClock')
ai.set_start_trigger('/cDAQ1/PFI0', edge='rising')

# Slave digital task: same clock, same trigger
di = DITask('flags', sample_rate=25600)
di.add_channel('gate', lines='cDAQ1Mod2/port0/line0')
di.configure(sample_mode='finite', samples_per_channel=25600,
             clock_source='/cDAQ1Mod1/ai/SampleClock')
di.set_start_trigger('/cDAQ1/PFI0', edge='rising')

adv = MultiHandler()
adv.configure(input_tasks=[ai.task, di.task])
adv.connect()

# Optional: hooks that select a bench procedure and fire the physical trigger
adv.set_hardware_trigger_functions(start_function=fire_pfi_line,
                                   support_function=select_procedure)

future = adv.acquire(custom_mode='run_up', blocking=False)  # arms tasks, fires trigger
data = future.result()   # {task_name: {channel_name: samples}}
adv.disconnect()
```

`stop_acquisition()` aborts a software-triggered acquisition whose trigger never
fires; `is_running()` and `check_state()` support watchdog-style health polling.

> Note: tasks reporting different clock-source readbacks (a master on the onboard
> clock reads back its timebase terminal, a slave the exported sample-clock terminal)
> produce a `UserWarning` from `MultiHandler.configure()` instead of failing
> validation — verify the clock wiring matches the warning. Mismatched sample rates
> or samples-per-channel still fail validation.

### Context manager (automatic cleanup)

```python
from nidaqwrapper import DAQHandler

with DAQHandler(task_in='MyTask') as wrapper:
    wrapper.connect()
    wrapper.set_trigger(n_samples=1000, trigger_channel=0, trigger_level=0.1)
    data = wrapper.acquire()
# Resources automatically cleaned up
```

### Wrap a raw nidaqmx.Task

```python
import nidaqmx
from nidaqwrapper import AITask

raw_task = nidaqmx.Task('external')
raw_task.ai_channels.add_ai_voltage_chan('Dev1/ai0')
raw_task.timing.cfg_samp_clk_timing(rate=25600)

wrapped = AITask.from_task(raw_task)
# Use wrapped task with DAQHandler or read directly
data = wrapped.acquire()  # shape: (n_channels, n_samples)
raw_task.close()  # Caller retains ownership
```

## API Reference

### Task Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `BaseTask` | `base_task` | Shared lifecycle, properties, `start()`, `stop()`, `save()`, `from_task()`, `from_name()` |
| `AITask` | `ai_task` | Analog input -- channels, timing, acquisition |
| `AOTask` | `ao_task` | Analog output -- channels, timing, generation |
| `DITask` | `digital` | Digital input -- on-demand and clocked reads |
| `DOTask` | `digital` | Digital output -- on-demand and clocked writes |

### Orchestrators

| Class | Module | Purpose |
|-------|--------|---------|
| `DAQHandler` | `handler` | Single-task handler with software triggering, auto-reconnection, acquisition abort |
| `MultiHandler` | `multi_handler` | Multi-task orchestrator with hardware trigger validation, trigger hooks, non-blocking acquire, health checks |

### Utility Functions

| Function | Purpose |
|----------|---------|
| `list_devices()` | List connected NI-DAQmx devices with product types |
| `list_tasks()` | List tasks saved in NI MAX |
| `get_connected_devices()` | Get set of connected device name strings |
| `get_task_by_name(name)` | Load a pre-configured task from NI MAX |
| `system_info()` | Structured dict of driver version, devices, and persisted tasks |
| `UNITS` | Dict mapping unit strings (`'g'`, `'mV/g'`, `'V'`, etc.) to nidaqmx constants |

## Data Format

The public API uses `(n_samples, n_channels)` for all multi-channel data. Internal transposition to nidaqmx's `(n_channels, n_samples)` layout is handled automatically.

- `DAQHandler.acquire()` returns `(n_samples, n_channels)` or a dict
- `DAQHandler.read_all_available()` returns `(n_samples, n_channels)`
- `DAQHandler.read()` returns `(n_channels,)` -- single sample
- `AITask.acquire()` returns `(n_samples, n_channels)`
- `AOTask.generate(signal)` accepts `(n_samples, n_channels)` or `(n_samples,)`

## Requirements

- Python >= 3.9
- NI-DAQmx drivers (system-level installation)
- numpy >= 1.20
- nidaqmx >= 0.8.0
- pyTrigger >= 0.3.0
- tomli >= 1.0 (Python < 3.11 only; Python 3.11+ uses built-in `tomllib`)

## Testing

nidaqwrapper uses a three-tier test strategy:

| Tier | Command | Requirements |
|------|---------|-------------|
| Mocked | `uv run pytest` | None (default) |
| Simulated | `uv run pytest -m simulated -v` | NI-DAQmx driver + simulated device |
| Hardware | `uv run pytest -m hardware -v` | Physical NI hardware |

The mocked tier (663 tests) runs by default and requires no NI-DAQmx driver. The simulated tier uses the real driver with simulated devices to catch API contract violations. The hardware tier validates real-world timing and physical signals.

See [TESTING.md](TESTING.md) for detailed setup instructions, troubleshooting, and how to configure simulated devices.

## License

MIT License -- Copyright (c) 2026 Tibor Barsi and contributors
