import numpy as np
import pytest

stim = pytest.importorskip("stim")

from ptwm.rigetti import (
    detector_measurement_indices,
    detector_round_symbols,
    detector_worldline_parities,
    fit_markov_syndrome_decoder,
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


def test_markov_decoder_uses_sequence_dynamics_not_only_symbol_counts():
    # Both classes have equal zero/one marginals. Class 0 persists; class 1 alternates.
    persistent = np.tile([0, 0, 1, 1], (100, 2))
    alternating = np.tile([0, 1, 0, 1], (100, 2))
    symbols = np.r_[persistent, alternating]
    labels = np.r_[np.zeros(100), np.ones(100)]
    iid = fit_markov_syndrome_decoder(symbols, labels, order=0)
    markov = fit_markov_syndrome_decoder(symbols, labels, order=1)
    iid_error = np.mean((iid.predict(symbols) >= 0.5) != labels)
    markov_error = np.mean((markov.predict(symbols) >= 0.5) != labels)
    assert iid_error >= 0.49
    assert markov_error == 0.0
