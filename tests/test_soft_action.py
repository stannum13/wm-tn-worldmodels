import numpy as np
import pytest

from ptwm.exact_dem import bayes_modes
from ptwm.soft_action import fit_soft_action, spline_features, spline_knots, teacher_logical_probability
from ptwm.surface_mode import affine_log_likelihood_ratio


def test_teacher_probability_agrees_with_exact_bayes_action():
    tables = np.asarray([[[.50, .15], [.05, .30]], [[.10, .25], [.40, .25]]])
    syndromes = np.asarray([[0, 1, 0, 1], [1, 1, 1, 0]])
    probability = teacher_logical_probability(tables, syndromes)
    action, _ = bayes_modes(tables, syndromes, temporal=True)
    assert np.array_equal(probability > .5, action)
    assert np.all((probability >= 0) & (probability <= 1))


def test_soft_labels_recover_logistic_response_without_bernoulli_label_noise():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(512, 2))
    truth = features @ np.asarray([1.2, -.7]) + .1
    probability = 1 / (1 + np.exp(-truth))
    model = fit_soft_action(features, probability, l2=1e-7)
    np.testing.assert_allclose(affine_log_likelihood_ratio(features, model), truth, atol=2e-3)


def test_knots_are_training_only_and_invalid_soft_labels_rejected():
    features = np.arange(400).reshape(100, 4)
    knots = spline_knots(features)
    copied = knots.copy()
    assert spline_features(features, knots).shape == (100, 24)
    spline_features(features + 1000, knots)
    assert np.array_equal(knots, copied)
    with pytest.raises(ValueError):
        fit_soft_action(features, np.full(100, 1.1))
