import numpy as np
import pytest

stim = pytest.importorskip("stim")

from scripts.run_surface_residual_router import (  # noqa: E402
    action_features,
    stream_data,
    threshold_prediction,
)
from scripts.run_surface_regime_switch import REGIMES, circuit  # noqa: E402


def test_stream_data_is_aligned_and_reproducible():
    circuits = [circuit(3, *parameters) for parameters in REGIMES]
    first = stream_data(circuits, seed=5, streams=2, horizon=8)
    second = stream_data(circuits, seed=5, streams=2, horizon=8)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert first[0].shape[0] == first[1].size == first[2].size


def test_threshold_prediction_selects_endpoints():
    probability = np.asarray([[0.2, 0.8]])
    endpoints = [np.asarray([0, 0]), np.asarray([1, 1])]
    assert np.array_equal(
        threshold_prediction(probability, endpoints, 0.5), [[0, 1]]
    )


def test_action_features_append_finite_log_odds():
    result = action_features(np.ones((2, 3)), np.asarray([[0.0, 1.0]]))
    assert result.shape == (2, 4)
    assert np.all(np.isfinite(result))
