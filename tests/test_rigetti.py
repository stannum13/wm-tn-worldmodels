import numpy as np

from ptwm.rigetti import (
    block_error_differences,
    block_score_differences,
    chronological_preparation_split,
    fit_iq_heads,
    probability_metrics,
)


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
