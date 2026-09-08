import numpy as np
import pytest

from ptwm.surface_mode import count_mode_posterior, fit_count_emission


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
