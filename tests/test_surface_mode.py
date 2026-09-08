import numpy as np
import pytest

from ptwm.surface_mode import (
    affine_log_likelihood_ratio,
    count_mode_posterior,
    fit_affine_log_likelihood_ratio,
    fit_count_emission,
    log_likelihood_mode_posterior,
)


def test_count_emission_is_normalized_and_smoothed():
    emission = fit_count_emission(
        np.asarray([0, 0, 1]), np.asarray([2, 3, 3]), detector_count=3,
    )
    assert np.all(emission > 0)
    assert np.allclose(emission.sum(axis=1), 1.0)


def test_temporal_count_filter_accumulates_weak_evidence():
    emission = np.asarray([[0.6, 0.4], [0.4, 0.6]])
    transition = np.asarray([[0.99, 0.01], [0.01, 0.99]])
    counts = np.ones((1, 5), dtype=int)
    posterior = count_mode_posterior(
        counts, emission, transition, np.asarray([0.5, 0.5]), temporal=True,
    )
    assert posterior[0, -1] > posterior[0, 0]


def test_count_emission_rejects_out_of_range_counts():
    with pytest.raises(ValueError):
        fit_count_emission(np.asarray([4]), np.asarray([0]), detector_count=3)


def test_affine_likelihood_ratio_separates_balanced_classes():
    first = np.asarray([[-2.0], [-1.0], [-0.5]])
    second = np.asarray([[0.5], [1.0], [2.0]])
    model = fit_affine_log_likelihood_ratio(first, second)
    assert np.all(affine_log_likelihood_ratio(first, model) < 0)
    assert np.all(affine_log_likelihood_ratio(second, model) > 0)


def test_log_likelihood_filter_accumulates_persistent_evidence():
    evidence = np.full((1, 5), 0.2)
    posterior = log_likelihood_mode_posterior(
        evidence,
        np.asarray([[0.99, 0.01], [0.01, 0.99]]),
        np.asarray([0.5, 0.5]),
        temporal=True,
    )
    assert posterior[0, -1] > posterior[0, 0]


def test_memoryless_log_likelihood_filter_handles_multiple_streams():
    evidence = np.zeros((3, 4))
    posterior = log_likelihood_mode_posterior(
        evidence,
        np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        np.asarray([0.5, 0.5]),
        temporal=False,
    )
    assert posterior.shape == evidence.shape
    assert np.allclose(posterior, 0.5)
