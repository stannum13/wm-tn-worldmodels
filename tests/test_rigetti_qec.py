import numpy as np
import pytest

stim = pytest.importorskip("stim")

from ptwm.rigetti import detector_measurement_indices, soft_detector_probabilities


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
