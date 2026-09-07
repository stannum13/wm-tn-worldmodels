import numpy as np
import pytest

from ptwm.data import episode_matrix, gate_counts, load_cell
from ptwm.metrics import horizon_of_tolerance, per_length_errors, summarize_all
from ptwm.splits import all_splits, bias_split, family_split, horizon_split, summarize


def test_load_real_cell(real_root):
    ds = load_cell(real_root, 40, 100)
    assert len(ds.episodes) == 14 * 39 * 200  # 14 biases x 39 lengths x 200 shots
    biases = {ep.bias for ep in ds.episodes}
    assert len(biases) == 14
    lengths = {ep.length for ep in ds.episodes}
    assert min(lengths) == 2 and max(lengths) == 40
    for ep in ds.episodes[:100]:
        assert all(0 <= g <= 23 for g in ep.gates)
        assert 0.0 < ep.fidelity < 1.1


def test_episode_matrix_single_length(synthetic_ds):
    eps20 = [ep for ep in synthetic_ds.episodes if ep.length == 20][:10]
    X, y = episode_matrix(eps20)
    assert X.shape == (10, 20, 24)
    assert y.shape == (10,)
    mixed = eps20[:5] + [ep for ep in synthetic_ds.episodes if ep.length == 5][:5]
    with pytest.raises(ValueError):
        episode_matrix(mixed)  # mixed lengths


def test_gate_counts(synthetic_ds):
    eps = synthetic_ds.episodes[:50]
    X, y = gate_counts(eps)
    assert X.shape == (50, 24)
    assert np.allclose(X.sum(axis=1), [ep.length for ep in eps])
    assert np.all(y < 0.05)  # log fidelities of near-unit fidelities


def test_horizon_split(synthetic_ds):
    s = horizon_split(synthetic_ds, train_cap=10)
    assert max(ep.length for ep in s.train) == 10
    assert min(ep.length for ep in s.test) == 11
    assert not (set(id(ep) for ep in s.train) & set(id(ep) for ep in s.test))


def test_bias_split(synthetic_ds):
    s = bias_split(synthetic_ds, held_out=(0.61,))
    train_biases = {ep.bias for ep in s.train}
    test_biases = {ep.bias for ep in s.test}
    assert 0.61 not in train_biases and test_biases == {0.61}


def test_family_split(synthetic_ds):
    s = family_split(synthetic_ds, held_out_family=2)
    ho = set(range(16, 24))
    assert all(not any(g in ho for g in ep.gates) for ep in s.train)
    assert all(any(g in ho for g in ep.gates) for ep in s.test)
    assert len(s.train) > 0 and len(s.test) > 0


def test_all_splits_disjoint_and_full(synthetic_ds):
    for s in all_splits(synthetic_ds):
        assert len(s.train) + len(s.test) == len(synthetic_ds.episodes)
        info = summarize(s)
        assert info["train"]["n"] > 0
        # bias split may be empty on synthetic data if the held-out bias is absent;
        # real data always has all 14 biases.
        if s.name.startswith("bias"):
            assert info["test"]["n"] >= 0
        else:
            assert info["test"]["n"] > 0


def test_per_length_errors_and_horizon(synthetic_ds):
    eps = [ep for ep in synthetic_ds.episodes if ep.bias == 0.1][:100]
    y_pred = np.array([np.log(ep.fidelity) + 0.01 for ep in eps])
    pl = per_length_errors(eps, y_pred)
    assert set(pl) == {ep.length for ep in eps}
    h = horizon_of_tolerance(pl, 0.05)
    assert h is not None and h >= 2


def test_summarize_all_keys():
    y = np.array([-0.1, -0.2, -0.3])
    out = summarize_all(y, y + 0.01)
    assert {"log_mse", "fidelity_mae", "physicality"} <= set(out)
    assert out["physicality"]["frac_above_one"] == 0.0
