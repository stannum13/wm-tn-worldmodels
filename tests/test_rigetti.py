import numpy as np

from ptwm.rigetti import (
    block_error_differences,
    block_score_differences,
    chronological_block_error_differences,
    chronological_preparation_split,
    spitz_pair_probability,
    spitz_boundary_probability,
    fit_iq_heads,
    fit_spitz_pairwise_matching,
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
    import pymatching

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
