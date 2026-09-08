import numpy as np
import pytest

stim = pytest.importorskip("stim")

from scripts.run_surface_regime_switch import (  # noqa: E402
    TRANSITION,
    circuit,
    morphology_features,
    regime_modes,
)


def test_regime_modes_are_reproducible_and_binary():
    first = regime_modes(3, 4, 20)
    second = regime_modes(3, 4, 20)
    assert np.array_equal(first, second)
    assert set(np.unique(first)) <= {0, 1}
    assert np.allclose(TRANSITION.sum(axis=1), 1.0)


def test_morphology_features_align_with_shots():
    value = circuit(3, 0.001, 0.02)
    detectors = value.compile_detector_sampler(seed=4).sample(shots=7)
    features = morphology_features(detectors, value)
    assert features.shape[0] == 7
    assert features.shape[1] >= 4
    assert np.array_equal(features[:, 0], detectors.sum(axis=1))
