"""Compile aligned noise schedules without dropping final measurement operations.

This composes a schedule on one fixed ideal circuit, not separately generated DEMs.
Seam-crossing fault propagation is delegated to the full Stim circuit afterwards.
"""
from __future__ import annotations

import numpy as np


def tick_schedule(ticks, changes, *, initial=0):
    if not isinstance(ticks, (int, np.integer)) or ticks < 1:
        raise ValueError("positive integer tick count required")
    if not isinstance(initial, (int, np.integer)) or initial < 0:
        raise ValueError("nonnegative integer initial mode required")
    result = np.full(ticks + 1, initial, dtype=np.int32)
    previous = 0
    for boundary, mode in changes:
        if (not isinstance(boundary, (int, np.integer)) or not previous < boundary <= ticks
                or not isinstance(mode, (int, np.integer)) or mode < 0):
            raise ValueError("strictly increasing tick boundaries and integer modes required")
        result[boundary:] = mode
        previous = boundary
    return result


def schedule_from_detector_modes(ticks, modes):
    """Map equal detector-time bins to tick slots using splice-compatible cuts."""
    modes = np.asarray(modes)
    if (not isinstance(ticks, (int, np.integer)) or ticks < 1 or modes.ndim != 1
            or not len(modes) or modes.dtype.kind not in "iu" or np.any(modes < 0)
            or ticks < len(modes)):
        raise ValueError("integer detector modes require at least one tick per bin")
    slots = np.arange(ticks + 1)
    boundaries = np.asarray([int(ticks * k / len(modes)) for k in range(1, len(modes))])
    bins = np.searchsorted(boundaries, slots, side="right")
    return modes[bins].astype(np.int32)


def compile_noise_schedule(circuits, schedule):
    if not circuits:
        raise ValueError("at least one aligned circuit is required")
    operations = [list(c.flattened()) for c in circuits]
    ideal = circuits[0].without_noise().flattened()
    if any(len(ops) != len(operations[0]) or c.without_noise().flattened() != ideal
           for c, ops in zip(circuits, operations)):
        raise ValueError("ideal circuit topologies differ")
    slots = np.cumsum([0] + [int(op.name == "TICK") for op in operations[0][:-1]])
    ticks = sum(op.name == "TICK" for op in operations[0])
    schedule = np.asarray(schedule)
    if (schedule.shape != (ticks + 1,) or schedule.dtype.kind not in "iu"
            or np.any(schedule < 0) or np.any(schedule >= len(circuits))):
        raise ValueError("one valid integer mode is required per tick slot, including final readout")
    output = type(circuits[0])()
    for index, column in enumerate(zip(*operations)):
        first = column[0]
        if any(op.name != first.name or op.targets_copy() != first.targets_copy() for op in column[1:]):
            raise ValueError("unaligned operation identities")
        output.append(column[int(schedule[slots[index]])])
    return output
