"""Causal nuisance-mode filtering from surface-code detector-count emissions."""

from __future__ import annotations

import numpy as np


def fit_count_emission(
    nominal_counts: np.ndarray, burst_counts: np.ndarray, detector_count: int,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Return a smoothed P(count | mode) table with shape (2, detectors + 1)."""
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    table = np.empty((2, detector_count + 1), dtype=float)
    for mode, counts in enumerate((nominal_counts, burst_counts)):
        if np.any((counts < 0) | (counts > detector_count)):
            raise ValueError("detector count is out of range")
        mass = np.bincount(counts.astype(int), minlength=detector_count + 1).astype(float)
        mass += pseudocount
        table[mode] = mass / mass.sum()
    return table


def count_mode_posterior(
    counts: np.ndarray, emission: np.ndarray, transition: np.ndarray,
    stationary: np.ndarray, *, temporal: bool,
) -> np.ndarray:
    """Return causal P(mode=burst) for independent streams of detector counts."""
    if counts.ndim != 2:
        raise ValueError("counts must have shape (streams, time)")
    if emission.shape[0] != 2 or transition.shape != (2, 2):
        raise ValueError("binary emission and transition tables are required")
    posterior = np.empty(counts.shape, dtype=float)
    previous = np.broadcast_to(stationary, (counts.shape[0], 2)).copy()
    for time in range(counts.shape[1]):
        prior = previous @ transition if temporal else stationary
        current = prior * emission[:, counts[:, time]].T
        current /= current.sum(axis=1, keepdims=True)
        posterior[:, time] = current[:, 1]
        previous = current
    return posterior
