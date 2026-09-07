import numpy as np
import pytest

stim = pytest.importorskip("stim")

from ptwm.rigetti import (
    detector_measurement_indices,
    detector_worldline_parities,
    local_detector_pairs,
    local_pair_features,
    soft_detector_probabilities,
)


def test_soft_detector_probability_matches_hard_parity_limits():
    circuit = stim.Circuit("""
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """).flattened()
    hard = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=bool)
    detectors, _ = circuit.compile_m2d_converter().convert(
        measurements=hard, separate_observables=True
    )
    indices = detector_measurement_indices(circuit)
    soft = soft_detector_probabilities(hard.astype(float), indices, hard, detectors)
    assert np.array_equal(soft >= 0.5, detectors)
    assert np.array_equal(indices[0], np.asarray([1, 0]))


def test_graph_features_use_only_local_geometry_and_past_worldline_state():
    circuit = stim.Circuit("""
        M 0 1
        DETECTOR(0, 0, 1) rec[-1]
        DETECTOR(2, 0, 1) rec[-2]
        M 0 1
        DETECTOR(0, 0, 2) rec[-1]
        DETECTOR(2, 0, 2) rec[-2]
    """).flattened()
    detectors = np.asarray([[0, 1, 1, 1], [1, 0, 1, 0]], dtype=float)
    pairs = local_detector_pairs(circuit)
    assert len(pairs) == 6  # two same-round plus all four adjacent-round pairs
    assert local_pair_features(detectors, pairs).shape == (2, 6)
    parity = detector_worldline_parities(detectors, circuit)
    assert np.array_equal(parity, np.asarray([[0, 1, 1, 0], [1, 0, 0, 0]]))
