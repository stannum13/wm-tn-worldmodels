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
    soft_reweight_array,
    soft_reweight_matrix,
    prepare_soft_reweighting,
    typed_circuit_noise_model,
    calibrated_uncertainty_route,
    probability_metrics,
    pack_affine_iq_heads,
    pack_soft_reweighting,
    temporal_vertex_separation,
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


def test_soft_reweight_array_matches_rebuilt_graph_weights():
    pymatching = pytest.importorskip("pymatching")

    base = pymatching.Matching()
    base.add_edge(0, 1, fault_ids={0}, error_probability=0.14)
    base.add_boundary_edge(1, fault_ids={0}, error_probability=0.08)
    signatures = [((0, 1), frozenset({0})), ((1,), frozenset({0}))]
    plan, _ = prepare_soft_reweighting(base, signatures, np.asarray([0.04, 0.03]))
    shot = np.asarray([0.21, 0.12])
    rebuilt = build_soft_reweighted_matching(plan, shot)
    updates = soft_reweight_array(plan, shot)
    observed = {(int(a), None if b == -1 else int(b)): w for a, b, w in updates}
    assert observed[(0, 1)] == pytest.approx(rebuilt.get_edge_data(0, 1)["weight"])
    assert observed[(1, None)] == pytest.approx(
        rebuilt.get_boundary_edge_data(1)["weight"]
    )
    batch = soft_reweight_matrix(plan, np.stack([shot, shot * 0.5]))
    assert batch.shape == (2, 2, 3)
    assert np.array_equal(batch[0], updates)
    packed = pack_soft_reweighting(plan)
    assert np.allclose(packed.build(np.stack([shot, shot * 0.5])), batch)
    assert np.allclose(packed.build(shot), updates)


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


def test_packed_affine_iq_heads_match_individual_heads():
    rng = np.random.default_rng(43)
    measurement_qubits = np.asarray([7, 8, 7, 8])
    features = rng.normal(size=(300, 2))
    labels = (features[:, 0] - 0.3 * features[:, 1] > 0).astype(float)
    head = fit_iq_heads(features, labels)["linear_logistic_iq"]
    heads = {7: head, 8: head}
    soft = (
        rng.normal(size=(20, 4)) + 1j * rng.normal(size=(20, 4))
    ).astype(np.complex64)
    packed = pack_affine_iq_heads(measurement_qubits, heads)
    expected = np.empty((20, 4), dtype=np.float32)
    for qubit in (7, 8):
        columns = np.flatnonzero(measurement_qubits == qubit)
        values = soft[:, columns].reshape(-1)
        expected[:, columns] = head.predict(
            np.c_[values.real, values.imag]
        ).reshape(20, -1)
    assert np.allclose(packed.predict(soft), expected, atol=2e-7)


def test_temporal_vertex_separation_for_path_graph():
    pymatching = pytest.importorskip("pymatching")
    matching = pymatching.Matching()
    matching.add_edge(0, 1)
    matching.add_edge(1, 2)
    matching.add_edge(2, 3)
    audit = temporal_vertex_separation(
        matching, {node: [0.0, 0.0, float(node)] for node in range(4)}
    )
    assert audit["maximum_active_separator"] == 1
    assert audit["path_decomposition_bag_upper_bound"] == 2
    assert audit["one_logical_parity_state_upper_bound"] == 4


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
