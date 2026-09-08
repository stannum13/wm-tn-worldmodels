import numpy as np
import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")

from scripts.run_surface_crossfit_compiler import energy_action_features  # noqa: E402


def test_energy_action_features_append_graph_native_weights():
    features = np.ones((2, 3))
    probability = np.asarray([[0.25, 0.75]])
    weights = [np.asarray([1.0, 2.0]), np.asarray([3.0, 1.0])]
    result = energy_action_features(features, probability, weights)
    assert result.shape == (2, 7)
    assert np.array_equal(result[:, -1], weights[1] - weights[0])
    assert np.all(np.isfinite(result))
