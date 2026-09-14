import json

import numpy as np
import pytest

from ptwm.directional_features import (
    parity_features, connected_motifs, select_columns, feature_dictionary,
)


def test_connected_motifs_exclude_pair_redundancy_and_are_deterministic():
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    actual = connected_motifs(4, pairs)
    assert actual == [(0, 1, 2), (1, 2, 3), (0, 1, 2, 3)]
    assert connected_motifs(4, pairs[::-1]) == actual
    assert all(len(m) >= 3 for m in actual)
    assert json.loads(json.dumps(actual)) == [list(m) for m in actual]


def test_parities_and_empty_dictionary():
    detectors = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
    assert parity_features(detectors, [(0, 1, 2)]).ravel().tolist() == [0, 1, 1]
    assert parity_features(detectors, []).shape == (3, 0)


def test_selector_ignores_constant_features_and_is_training_only():
    x = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=float)
    target = x[:, 0]
    chosen = select_columns(x, target, 2)
    assert chosen.tolist() == [0, 2]
    frozen = chosen.copy()
    _ = np.full((6, 3), 100.)[:, chosen]
    np.testing.assert_array_equal(chosen, frozen)
    with pytest.raises(ValueError):
        select_columns(x, target, 0)


def test_feature_grammar_has_declared_interactions_not_hidden_labels():
    base = np.array([[0., 1.], [1., 0.]])
    motif = np.array([[1.], [0.]])
    p = np.array([.25, .75])
    actual = feature_dictionary(base, motif, p, "both")
    np.testing.assert_array_equal(actual[:, :3], np.column_stack((base, motif)))
    np.testing.assert_array_equal(actual[:, 3:], np.column_stack((base, motif)) * (2 * p[:, None] - 1))
    with pytest.raises(ValueError):
        feature_dictionary(base, motif, p, "unknown")
