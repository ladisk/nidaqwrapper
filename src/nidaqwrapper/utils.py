"""Utility functions and constants for nidaqwrapper.

Provides the UNITS constant dictionary mapping physical unit strings to
nidaqmx constant enum values, plus device and task discovery helpers.
All hardware-dependent functions require nidaqmx to be installed.

Notes
-----
nidaqmx 1.4.1 does not have ``AccelSensitivityUnits.M_VOLTS_PER_METERS_PER_SECOND_SQUARED``.
The ``mV/m/s**2`` key currently maps to ``MILLIVOLTS_PER_G`` as a temporary
compatibility measure.  When NI ships the dedicated m/s² sensitivity constant,
update the mapping here.  See .claude-notes/agent-notes/units-nidaqmx-compat.md.
"""

from __future__ import annotations

import warnings
from typing import Any

try:
    import nidaqmx
    from nidaqmx import constants
    from nidaqmx.errors import DaqError

    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False
    warnings.warn(
        "nidaqmx not available. Hardware functions will raise RuntimeError.",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# UNITS — physical unit string → nidaqmx constant mapping
# ---------------------------------------------------------------------------

if _NIDAQMX_AVAILABLE:
    # TODO: Replace 'mV/m/s**2' mapping with
    #       AccelSensitivityUnits.M_VOLTS_PER_METERS_PER_SECOND_SQUARED once
    #       NI ships that constant.  nidaqmx 1.4.1 only has MILLIVOLTS_PER_G
    #       and VOLTS_PER_G in AccelSensitivityUnits.
    UNITS: dict[str, Any] = {
        "mV/g": constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
        "mV/m/s**2": constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
        "g": constants.AccelUnits.G,
        "m/s**2": constants.AccelUnits.METERS_PER_SECOND_SQUARED,
        "mV/N": constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,
        "N": constants.ForceUnits.NEWTONS,
        "V": constants.VoltageUnits.VOLTS,
    }
else:
    UNITS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# UNITS_REVERSE — nidaqmx constant → unit string (inverse of UNITS)
# ---------------------------------------------------------------------------

if _NIDAQMX_AVAILABLE:
    # Built by dict comprehension — keeps the last key for any constant that
    # appears multiple times.  The only collision is MILLIVOLTS_PER_G, which
    # is shared by both 'mV/g' and 'mV/m/s**2' (nidaqmx 1.4.1 compat note in
    # module docstring).  The comprehension retains 'mV/m/s**2' (the later key).
    # from_config() accepts both strings, so round-trip integrity is preserved.
    UNITS_REVERSE: dict[Any, str] = {v: k for k, v in UNITS.items()}
else:
    UNITS_REVERSE: dict[Any, str] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_nidaqmx() -> None:
    """Raise RuntimeError when nidaqmx is not available.

    Raises
    ------
    RuntimeError
        If the nidaqmx package is not installed or NI-DAQmx drivers are absent.
    """
    if not _NIDAQMX_AVAILABLE:
        raise RuntimeError(
            "NI-DAQmx drivers are required for this operation. "
            "Install the package with: pip install nidaqmx"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def system_info() -> dict[str, Any]:
    """Return NI system discovery information.

    Aggregates connected device details and saved NI MAX task names
    into a single dict for convenient discovery in notebooks and scripts.

    Returns
    -------
    dict[str, Any]
        A dict with two keys:

        - ``"devices"`` — list of dicts, one per connected device:
            - ``"name"`` — device identifier (e.g., ``"Dev1"``)
            - ``"product_type"`` — product model (e.g., ``"PCIe-6320"``)
            - ``"serial_number"`` — hex serial string (e.g., ``"0x1b4c8a0"``)
            - ``"ai_channels"`` — analog input channel names (e.g., ``["ai0", "ai1"]``)
            - ``"ao_channels"`` — analog output channel names (e.g., ``["ao0"]``)
        - ``"tasks"`` — list of NI MAX saved task name strings

    Raises
    ------
    RuntimeError
        If nidaqmx is not installed or NI-DAQmx drivers are unavailable.

    Examples
    --------
    >>> info = system_info()
    >>> info["devices"][0]["name"]
    'Dev1'
    >>> info["tasks"]
    ['MyInputTask', 'MyOutputTask']
    """
    _require_nidaqmx()
    system = nidaqmx.system.System.local()

    devices = []
    for dev in system.devices:
        devices.append({
            "name": dev.name,
            "product_type": dev.product_type,
            "serial_number": hex(dev.dev_serial_num),
            "ai_channels": [
                ch.name.split("/")[-1] for ch in dev.ai_physical_chans
            ],
            "ao_channels": [
                ch.name.split("/")[-1] for ch in dev.ao_physical_chans
            ],
        })

    return {"devices": devices, "tasks": list(system.tasks.task_names)}


def list_tasks() -> list[str]:
    """List all tasks saved in NI MAX.

    Returns
    -------
    list[str]
        Task name strings from NI MAX. Empty list if no tasks are saved.

    Raises
    ------
    RuntimeError
        If nidaqmx is not installed or NI-DAQmx drivers are unavailable.

    Examples
    --------
    >>> list_tasks()
    ['MyInputTask', 'MyOutputTask']

    >>> list_tasks()  # no tasks saved in NI MAX
    []
    """
    _require_nidaqmx()
    system = nidaqmx.system.System.local()
    return list(system.tasks.task_names)


def get_task_by_name(name: str) -> nidaqmx.task.Task | None:
    """Load a pre-configured NI-DAQmx task from NI MAX by name.

    Iterates over tasks saved in NI MAX, matches by name, and calls
    ``.load()`` to return a ready-to-use ``nidaqmx.Task`` object.

    This is a low-level helper: it returns a **raw** ``nidaqmx.task.Task``
    by design, without ``acquire()``, ``generate()``, or ``configure()``.
    To load a saved task as a wrapper object, use the ``from_name()``
    classmethod of the matching task class instead::

        task = AITask.from_name("MyInputTask")   # or AOTask / DITask / DOTask
        task.start()
        data = task.acquire(1000)

    Parameters
    ----------
    name : str
        The exact name of the task as saved in NI MAX.

    Returns
    -------
    nidaqmx.Task or None
        The loaded task object, or ``None`` if the task is already loaded
        by another process (error code -200089). The caller is responsible
        for deciding how to proceed.

    See Also
    --------
    nidaqwrapper.AITask.from_name : Load a saved AI task as an AITask.
    nidaqwrapper.AOTask.from_name : Load a saved AO task as an AOTask.
    nidaqwrapper.DITask.from_name : Load a saved DI task as a DITask.
    nidaqwrapper.DOTask.from_name : Load a saved DO task as a DOTask.

    Raises
    ------
    KeyError
        If no task with ``name`` exists in NI MAX.  The message includes
        the requested name and a list of all available task names.
    ConnectionError
        If the device associated with the task is inaccessible (error code
        -201003) — the device may be disconnected or held by another
        application.
    DaqError
        Any other NI-DAQmx error is re-raised unchanged so the caller can
        handle hardware-specific conditions.
    RuntimeError
        If nidaqmx is not installed or NI-DAQmx drivers are unavailable.

    Examples
    --------
    >>> task = get_task_by_name("MyInputTask")
    >>> task.start()

    Notes
    -----
    An empty string is treated as a missing task and raises ``KeyError``,
    because NI MAX does not permit blank task names.
    """
    _require_nidaqmx()
    system = nidaqmx.system.System.local()

    # Collect available names up front so the KeyError message is informative.
    # The iterator is recreated for each call to system.tasks because the mock
    # (and the real nidaqmx API) supports multiple passes over the collection.
    available_names = [t._name for t in system.tasks]

    for task in system.tasks:
        if task._name != name:
            continue

        try:
            return task.load()
        except DaqError as exc:
            if exc.error_code == -200089:
                return None
            if exc.error_code == -201003:
                raise ConnectionError(
                    f"Task '{name}' cannot be accessed. The device may be "
                    "disconnected or in use by another application. "
                    "Check the hardware connection."
                ) from exc
            raise

    raise KeyError(
        f"No task named '{name}' was found in NI MAX. "
        f"Available tasks: {available_names}"
    )


def get_connected_devices() -> set[str]:
    """Return the set of currently connected NI-DAQmx device names.

    Queries the local NI-DAQmx system and returns only the device name
    strings, without product type metadata.  Useful when the caller needs
    a fast membership check (``"cDAQ1Mod1" in get_connected_devices()``).

    Returns
    -------
    set[str]
        Device name strings (e.g. ``{"cDAQ1Mod1", "cDAQ1Mod2"}``).
        Empty set if no devices are connected.

    Raises
    ------
    RuntimeError
        If nidaqmx is not installed or NI-DAQmx drivers are unavailable.

    Examples
    --------
    >>> get_connected_devices()
    {'cDAQ1Mod1', 'cDAQ1Mod2'}

    >>> get_connected_devices()  # no hardware connected
    set()

    See Also
    --------
    list_devices : Returns full device info dicts including product type.
    """
    _require_nidaqmx()
    system = nidaqmx.system.System.local()
    return {dev.name for dev in system.devices}


def list_devices() -> list[dict[str, str]]:
    """List all NI-DAQmx compatible devices connected to the system.

    Returns
    -------
    list[dict[str, str]]
        A list of dicts, one per device.  Each dict has exactly two keys:

        - ``"name"`` — the device identifier string (e.g. ``"cDAQ1Mod1"``)
        - ``"product_type"`` — the product model string (e.g. ``"NI 9234"``)

        Returns an empty list when no devices are present.

    Raises
    ------
    RuntimeError
        If nidaqmx is not installed or NI-DAQmx drivers are unavailable.

    Examples
    --------
    >>> list_devices()
    [{'name': 'cDAQ1Mod1', 'product_type': 'NI 9234'}]

    >>> list_devices()  # no hardware connected
    []
    """
    _require_nidaqmx()
    system = nidaqmx.system.System.local()
    return [
        {"name": dev.name, "product_type": dev.product_type}
        for dev in system.devices
    ]
