"""Tests for the iterative rollout RL module (synthetic data, few episodes)."""

import numpy as np

from ptwm.data import EpisodeDataset
from ptwm.rl import LinearPolicy, OnlineTransitionModel, RolloutEnv, run_rl


def test_linear_policy_contract():
    rng = np.random.default_rng(0)
    pol = LinearPolicy(rng)
    s = pol.state(0.3, 100, 5, 10, last_gate=7)
    assert s.shape == (4 + 24,)
    g, p = pol.act(s)
    assert 0 <= g < 24
    assert abs(p.sum() - 1.0) < 1e-9
    h = pol.entropy(s)
    assert 0 <= h <= np.log(24) + 1e-9


def test_policy_update_changes_logits():
    rng = np.random.default_rng(0)
    pol = LinearPolicy(rng, lr=0.5)
    s = pol.state(0.3, 100, 0, 4, None)
    before = pol.probs(s).copy()
    states, actions = [s, s], [3, 5]
    pol.update(states, actions, advantage=1.0)
    after = pol.probs(s)
    assert not np.allclose(before, after)
    # chosen gates should gain probability under positive advantage
    assert after[3] > before[3] and after[5] > before[5]


def test_online_transition_model_learns():
    rng = np.random.default_rng(0)
    model = OnlineTransitionModel(lr=0.05)
    errs = []
    for i in range(200):
        counts = np.zeros(24)
        counts[rng.integers(24)] += 1
        y = -0.02 * counts.sum() - 0.01 * np.argmax(counts) / 24
        errs.append(model.update(counts, 0.3, 100, y))
    assert errs[-1] < errs[0]


def test_run_rl_end_to_end(synthetic_ds, tmp_path, monkeypatch):
    import ptwm.data as pdata

    ds = synthetic_ds
    shrunk = EpisodeDataset(length_cap=ds.length_cap, idle=ds.idle)
    shrunk.episodes = ds.episodes[:400]
    monkeypatch.setattr(pdata, "load_cell", lambda root, cap, idle: shrunk)
    res = run_rl(
        root="unused", length_cap=ds.length_cap, idle=ds.idle, out_dir=str(tmp_path),
        n_episodes=50, seq_cap=6, seed=0,
    )
    assert set(res["learning_curve"]) == {"episode", "reward", "pred_err", "entropy", "phys_violation", "grad_norm"}
    assert len(res["learning_curve"]["episode"]) == 50
    assert "online_transition_mae" in res["eval"]
    assert "batch_markov_mae" in res["eval"]
    assert (tmp_path / f"rl_len{ds.length_cap}_idle{ds.idle}.json").exists()
