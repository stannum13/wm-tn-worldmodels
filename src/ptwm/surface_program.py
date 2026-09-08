"""Compile local affine evidence into fixed detector/pair operations.

Only the mode probability persists between completed shots. Decoder graph choice
can therefore precede matching; matching energies are not required by this lane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class LocalFeatures:
    detector_times: np.ndarray
    pairs: np.ndarray
    temporal: np.ndarray
    spatial: np.ndarray

    @classmethod
    def from_circuit(cls, circuit):
        coordinates = circuit.get_detector_coordinates()
        xyz = np.asarray([coordinates[i][:3] for i in range(len(coordinates))])
        pairs, temporal, spatial = [], [], []
        for first in range(len(xyz)):
            for second in range(first + 1, len(xyz)):
                delta = np.abs(xyz[first] - xyz[second])
                is_time = bool(np.all(delta[:2] == 0) and delta[2] == 1)
                is_space = bool(delta[2] == 0 and np.sum(delta[:2] ** 2) <= 4.01)
                if is_time or is_space:
                    pairs.append((first, second))
                    temporal.append(is_time)
                    spatial.append(is_space)
        _, times = np.unique(xyz[:, 2], return_inverse=True)
        return cls(times, np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
                   np.asarray(temporal), np.asarray(spatial))

    def transform(self, detectors):
        products = detectors[:, self.pairs[:, 0]] & detectors[:, self.pairs[:, 1]]
        aggregate = np.column_stack((
            detectors.sum(axis=1), products[:, self.temporal].sum(axis=1),
            products[:, self.spatial].sum(axis=1),
            *[detectors[:, self.detector_times == t].sum(axis=1)
              for t in range(int(self.detector_times.max()) + 1)],
        ))
        return np.column_stack((detectors, products, aggregate)).astype(np.float32)

    def compile(self, model):
        """Algebraically fold normalization and redundant count features."""
        raw = np.asarray(model["weights"]) / np.asarray(model["scale"])
        bias = float(model["bias"] - np.asarray(model["mean"]) @ raw)
        nd, npairs = len(self.detector_times), len(self.pairs)
        aggregate = raw[nd + npairs:]
        expected = 3 + int(self.detector_times.max()) + 1
        if len(aggregate) != expected:
            raise ValueError("evidence model and feature schema disagree")
        detector_weights = raw[:nd] + aggregate[0] + aggregate[3:][self.detector_times]
        pair_weights = (raw[nd:nd + npairs] + self.temporal * aggregate[1]
                        + self.spatial * aggregate[2])
        return EvidenceProgram(detector_weights, self.pairs.copy(), pair_weights, bias)


@dataclass
class EvidenceProgram:
    detector_weights: np.ndarray
    pairs: np.ndarray
    pair_weights: np.ndarray
    bias: float

    def score(self, detectors):
        products = detectors[..., self.pairs[:, 0]] & detectors[..., self.pairs[:, 1]]
        return (detectors @ self.detector_weights + products @ self.pair_weights
                + self.bias)

    @property
    def constant_bytes(self):
        return int(self.detector_weights.nbytes + self.pairs.nbytes
                   + self.pair_weights.nbytes + 8)


def update_probability(log_likelihood_ratio, previous, switch_probability=0.02):
    prior = switch_probability + (1.0 - 2.0 * switch_probability) * previous
    return expit(log_likelihood_ratio + np.log(prior) - np.log1p(-prior))


def select_before_decode(detectors, chosen, decoders, *, correlated):
    """Exactly one Matching invocation per row (two internal passes if correlated)."""
    chosen = np.asarray(chosen, dtype=bool).ravel()
    if len(chosen) != len(detectors):
        raise ValueError("one graph choice is required per shot")
    predictions = np.empty(len(chosen), dtype=bool)
    for mode, decoder in enumerate(decoders):
        rows = np.flatnonzero(chosen == mode)
        if len(rows):
            predictions[rows] = decoder.decode_batch(
                detectors[rows], enable_correlations=correlated
            )[:, 0]
    return predictions
