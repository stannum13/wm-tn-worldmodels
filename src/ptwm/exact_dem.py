"""Offline exact distribution and two-mode Bayes teacher for very small DEMs."""

from __future__ import annotations

import numpy as np

from ptwm.native_surface import fwht


def fault_terms(dem):
    """Each DEM instruction is one Bernoulli variable, including all ^ components."""
    terms = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        if not 0 <= probability < .5:
            raise ValueError("teacher currently requires independent fault probabilities < 1/2")
        mask = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                mask ^= 1 << target.val
            elif target.is_logical_observable_id():
                mask ^= 1 << (dem.num_detectors + target.val)
        terms.append((mask, probability))
    return terms


def exact_distribution(dem, *, max_bits=25):
    bits = dem.num_detectors + dem.num_observables
    if bits > max_bits:
        raise ValueError(f"{bits} bits exceed the offline enumeration budget of {max_bits}")
    terms = fault_terms(dem)
    values = np.zeros(1 << bits, dtype=np.float64)
    for mask, probability in terms:
        values[mask] += .5 * np.log1p(-2 * probability)
    total = float(values.sum())
    fwht(values)
    values *= -1
    values += total
    positive_log = float(values.max())
    if positive_log > 1e-10:
        raise ArithmeticError("material positive log characteristic")
    np.minimum(values, 0, out=values)
    np.exp(values, out=values)
    values[0] = 1.0
    fwht(values)
    values /= len(values)
    diagnostics = {"bits": bits, "table_bytes": values.nbytes, "terms": len(terms),
                   "raw_mass": float(values.sum()), "minimum_mass": float(values.min()),
                   "negative_mass": float(-values[values < 0].sum()),
                   "maximum_positive_log_characteristic": positive_log}
    if diagnostics["negative_mass"] > 1e-9 or abs(diagnostics["raw_mass"] - 1) > 1e-10:
        raise ArithmeticError(f"exact distribution failed numerical budget: {diagnostics}")
    np.maximum(values, 0, out=values)
    values /= values.sum()
    return values, diagnostics


def syndrome_indices(detectors):
    if detectors.shape[1] > 62:
        raise ValueError("syndrome index exceeds int64 capacity")
    return np.asarray(detectors, dtype=np.int64) @ (1 << np.arange(detectors.shape[1], dtype=np.int64))


def bayes_modes(tables, syndromes, *, temporal, switch_probability=.02):
    """Predict from pre-syndrome belief, then update it once using syndrome evidence.

    tables has dimensions (mode, logical, syndrome); syndromes is (stream, time).
    """
    if tables.shape[:2] != (2, 2) or syndromes.ndim != 2:
        raise ValueError("two modes, one logical bit and 2D streams are required")
    if np.any(syndromes < 0) or np.any(syndromes >= tables.shape[2]):
        raise ValueError("syndrome outside table")
    if not 0 <= switch_probability <= 1:
        raise ValueError("invalid switching probability")
    prediction = np.empty(syndromes.shape, dtype=bool)
    posterior = np.empty(syndromes.shape, dtype=float)
    previous = np.full(len(syndromes), .5)
    for step in range(syndromes.shape[1]):
        prior = (switch_probability + (1 - 2 * switch_probability) * previous
                 if temporal else np.full(len(syndromes), .5))
        current = tables[:, :, syndromes[:, step]]
        joint = (1 - prior) * current[0] + prior * current[1]
        prediction[:, step] = joint[1] > joint[0]
        mass = current.sum(axis=1)
        denominator = (1 - prior) * mass[0] + prior * mass[1]
        if np.any(denominator <= 0):
            raise ArithmeticError("observed syndrome has zero probability in both teacher modes")
        previous = prior * mass[1] / denominator
        posterior[:, step] = previous
    return prediction, posterior
