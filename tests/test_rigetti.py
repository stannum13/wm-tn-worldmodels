import numpy as np
import pytest

from ptwm.rigetti import (
    block_error_differences,
    block_score_differences,
    chronological_block_error_differences,
    chronological_preparation_split,
    spitz_pair_probability,
    spitz_boundary_probability,
    fit_iq_heads,
    fit_spitz_pairwise_matching,
    measurement_error_signatures,
    build_soft_reweighted_matching,
    prepare_soft_reweighting,
    typed_circuit_noise_model,
    calibrated_uncertainty_route,
    probability_metrics,
)


def test_spitz_pair_probability_recovers_independent_edge_rate():
    rng = np.random.default_rng(17)
    shots = 1_000_000
    shared = rng.random(shots) < 0.08
    first = shared ^ (rng.random(shots) < 0.03)
    second = shared ^ (rng.random(shots) < 0.05)
    estimate = spitz_pair_probability(first, second)
    assert abs(estimate - 0.08) < 0.002


def test_spitz_boundary_probability_recovers_single_defect_rate():
    rng = np.random.default_rng(23)
    shots = 1_000_000
    shared = rng.random(shots) < 0.08
    boundary = rng.random(shots) < 0.03
    first = shared ^ boundary
    estimate = spitz_boundary_probability(first, [0.08])
    assert abs(estimate - 0.03) < 0.002


def test_pairwise_matching_fit_preserves_topology_and_fault_ids():
    pymatching = pytest.importorskip("pymatching")

    rng = np.random.default_rng(29)
    shots = 1_000_000
    shared = rng.random(shots) < 0.08
    detectors = np.c_[
        shared ^ (rng.random(shots) < 0.03),
        shared ^ (rng.random(shots) < 0.05),
    ]
    template = pymatching.Matching()
    template.add_edge(0, 1, fault_ids={0}, error_probability=0.1)
    template.add_boundary_edge(0, error_probability=0.1)
    template.add_boundary_edge(1, error_probability=0.1)
    fitted, diagnostics = fit_spitz_pairwise_matching(
        template, detectors, floor_probability=1e-4
    )
    assert fitted.get_edge_data(0, 1)["fault_ids"] == {0}
    assert abs(fitted.get_edge_data(0, 1)["error_probability"] - 0.08) < 0.002
    assert diagnostics["pair_edges"] == 1
    assert diagnostics["boundary_edges"] == 2


def test_measurement_error_signature_tracks_detector_and_observable():
    stim = pytest.importorskip("stim")

    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    signatures = measurement_error_signatures(circuit)
    assert signatures == [((0,), frozenset({0}))]


def test_soft_reweighting_replaces_average_measurement_contribution():
    pymatching = pytest.importorskip("pymatching")

    base = pymatching.Matching()
    total = 0.1 + 0.05 - 2 * 0.1 * 0.05
    base.add_edge(0, 1, fault_ids={0}, error_probability=total)
    plan, diagnostics = prepare_soft_reweighting(
        base, [((0, 1), frozenset({0}))], np.asarray([0.05])
    )
    updated = build_soft_reweighted_matching(plan, np.asarray([0.2]))
    expected = 0.1 + 0.2 - 2 * 0.1 * 0.2
    assert abs(updated.get_edge_data(0, 1)["error_probability"] - expected) < 1e-9
    assert diagnostics["matched_measurements"] == 1


def test_measurement_noise_is_a_record_flip_not_persistent_state_flip():
    pymatching = pytest.importorskip("pymatching")
    stim = pytest.importorskip("stim")

    circuit = stim.Circuit("R 0\nM 0\nM 0\nDETECTOR rec[-1] rec[-2]")
    model = typed_circuit_noise_model(
        circuit, measurement_probability=0.1,
        one_qubit_probability=0.0, two_qubit_probability=0.0,
    )
    matching = pymatching.Matching.from_detector_error_model(model)
    assert abs(matching.get_boundary_edge_data(0)["error_probability"] - 0.18) < 1e-9


def test_uncertainty_route_threshold_is_fixed_on_calibration_scores():
    train = np.arange(100, dtype=float)
    test = np.asarray([0.0, 49.0, 89.5, 99.0, 120.0])
    routed, threshold = calibrated_uncertainty_route(train, test, budget=0.1)
    assert threshold == np.quantile(train, 0.9)
    assert np.array_equal(routed, [False, False, True, True, True])


def test_chronological_split_preserves_both_classes_without_overlap():
    labels = np.r_[np.zeros(10), np.ones(10)]
    train, test = chronological_preparation_split(labels, train_fraction=0.6)
    assert set(train).isdisjoint(test)
    assert np.array_equal(labels[train], np.r_[np.zeros(6), np.ones(6)])
    assert np.array_equal(labels[test], np.r_[np.zeros(4), np.ones(4)])


def test_iq_heads_fit_separable_calibration_data():
    rng = np.random.default_rng(1)
    labels = np.r_[np.zeros(200), np.ones(200)]
    features = np.c_[2 * labels - 1 + rng.normal(0, 0.1, 400), rng.normal(0, 0.1, 400)]
    heads = fit_iq_heads(features, labels)
    for head in heads.values():
        assert probability_metrics(head.predict(features), labels)["classification_error"] < 0.01


def test_block_difference_sign_means_first_has_more_errors():
    labels = np.r_[np.zeros(4), np.ones(4)]
    first = 1 - labels
    second = labels
    differences = block_error_differences(first, second, labels, block_size=2)
    assert np.array_equal(differences, np.ones(4))
    score_differences = block_score_differences(
        first, second, labels, block_size=2, score="brier_loss"
    )
    assert np.allclose(score_differences, np.ones(4), atol=3e-6)
    chronological = chronological_block_error_differences(
        first, second, labels, block_size=2
    )
    assert np.array_equal(chronological, np.ones(4))
