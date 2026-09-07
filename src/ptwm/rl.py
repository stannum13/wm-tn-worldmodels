"""Iterative rollout-based RL on real RB sequences.

Design (per the plan's "iterative, not whole-batch" requirement):

- An agent assembles gate sequences one gate at a time. Its policy is a linear
  softmax over a state feature vector (context + last gate), so each episode is a
  rollout of sequential decisions under the current model.
- A linear transition model predicts the sequence's log-fidelity from gate counts;
  it is updated by SGD after EVERY episode (one episode = one update), never by
  whole-batch gradient passes.
- Reward = prediction quality of the assembled sequence (negative absolute error in
  log-fidelity) minus a physicality penalty when the predicted fidelity leaves the
  observable range. The policy is updated by REINFORCE with a running baseline.
- Learning curves track per-episode reward, prediction error, policy entropy, and
  physicality violations.

Baselines for comparison: (a) the same linear transition model trained by batch
least squares (the Markov channel), (b) a random policy with the same online
transition model (isolates the policy's contribution), (c) the batch model's error
on the agent's chosen sequences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import Episode, N_GATES
from .models import MarkovChannel, context_vec
from .splits import horizon_split

BIAS_MAX = 0.64
IDLE_MAX = 180.0


class RolloutEnv:
    """Episode environment over real RB data.

    Each episode samples a real (bias, idle, length, measured fidelity) record and
    lets the agent assemble a gate sequence of that length. The reward uses the REAL
    measured fidelity of that record vs. the transition model's prediction for the
    assembled sequence.
    """

    def __init__(self, episodes: list[Episode], rng: np.random.Generator):
        self.episodes = episodes
        self.rng = rng

    def sample_context(self) -> tuple[int, float, int, float]:
        """Returns (episode_idx, bias, idle, true_fidelity)."""
        i = self.rng.integers(len(self.episodes))
        ep = self.episodes[i]
        return i, ep.bias, ep.idle, ep.fidelity


class LinearPolicy:
    """Linear softmax policy over gates. State: [1, bias_n, idle_n, progress, last_gate_onehot]."""

    def __init__(self, rng: np.random.Generator, lr: float = 0.05):
        self.rng = rng
        self.lr = lr
        self.W = np.zeros((N_GATES, 4 + N_GATES), dtype=np.float64)  # logits per gate

    def state(self, bias: float, idle: int, t: int, L: int, last_gate: int | None) -> np.ndarray:
        s = np.zeros(4 + N_GATES)
        s[0] = 1.0
        s[1] = bias / BIAS_MAX
        s[2] = idle / IDLE_MAX
        s[3] = t / max(L, 1)
        if last_gate is not None:
            s[4 + last_gate] = 1.0
        return s

    def probs(self, state: np.ndarray) -> np.ndarray:
        logits = self.W @ state
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def act(self, state: np.ndarray) -> tuple[int, np.ndarray]:
        p = self.probs(state)
        g = int(self.rng.choice(N_GATES, p=p))
        return g, p

    def update(self, states: list[np.ndarray], actions: list[int], advantage: float) -> float:
        """REINFORCE step for one episode. Returns gradient norm."""
        grad = np.zeros_like(self.W)
        for s, a in zip(states, actions):
            p = self.probs(s)
            grad[a] += (1.0 - p[a]) * s
            for g in range(N_GATES):
                if g != a:
                    grad[g] -= p[g] * s
        self.W += self.lr * advantage * grad
        return float(np.linalg.norm(grad))

    def entropy(self, state: np.ndarray) -> float:
        p = self.probs(state)
        return float(-np.sum(p * np.log(np.clip(p, 1e-12, None))))


class OnlineTransitionModel:
    """Linear log-fidelity model updated by SGD once per episode.

    Stability guards for online (non-batch) training: count features are normalized
    by sequence length (comparable step size across lengths), the error is clipped,
    and any non-finite prediction resets the model to its prior.
    """

    ERR_CLIP = 3.0

    def __init__(self, lr: float = 0.02):
        self.lr = lr
        self.w = np.zeros(N_GATES)
        self.c = np.zeros(2)
        self.b = 0.0

    def features(self, counts: np.ndarray, bias: float, idle: int) -> np.ndarray:
        L = max(counts.sum(), 1.0)
        return np.concatenate([counts / L, [bias / BIAS_MAX, idle / IDLE_MAX]])

    def predict(self, counts: np.ndarray, bias: float, idle: int) -> float:
        L = max(counts.sum(), 1.0)
        return float(
            self.w @ (counts / L)
            + self.c[0] * bias / BIAS_MAX
            + self.c[1] * idle / IDLE_MAX
            + self.b
        )

    def update(self, counts: np.ndarray, bias: float, idle: int, y_true: float) -> float:
        y_pred = self.predict(counts, bias, idle)
        if not np.isfinite(y_pred):
            # Diverged: reset to prior (predict 0 = fidelity 1) and continue.
            self.w[:] = 0.0
            self.c[:] = 0.0
            self.b = 0.0
            y_pred = 0.0
        err = float(np.clip(y_pred - y_true, -self.ERR_CLIP, self.ERR_CLIP))
        L = max(counts.sum(), 1.0)
        self.w -= self.lr * err * counts / L
        self.c[0] -= self.lr * err * bias / BIAS_MAX
        self.c[1] -= self.lr * err * idle / IDLE_MAX
        self.b -= self.lr * err
        return abs(err)


def run_rl(root: str, length_cap: int, idle: int, out_dir: str,
           n_episodes: int = 3000, seq_cap: int = 20, seed: int = 0,
           lr_policy: float = 0.05, lr_transition: float = 0.02,
           lambda_phys: float = 2.0, eps_start: float = 0.2, eps_end: float = 0.05) -> dict:
    """Train the agent online; evaluate against batch baselines; save curves.

    Epsilon-greedy exploration: with probability eps (decaying linearly from
    eps_start to eps_end over the run) the agent takes a uniform-random gate
    instead of the policy's sample. This keeps the online transition model's gate
    coverage from collapsing as the policy concentrates.
    """
    from .data import load_cell

    rng = np.random.default_rng(seed)
    ds = load_cell(root, length_cap, idle)
    train_eps = [ep for ep in ds.episodes if ep.length <= seq_cap]
    test_eps = [ep for ep in ds.episodes if ep.length > seq_cap]
    env = RolloutEnv(train_eps, rng)

    policy = LinearPolicy(rng, lr=lr_policy)
    online = OnlineTransitionModel(lr=lr_transition)

    # Batch baseline for comparison (whole-batch least squares).
    batch = MarkovChannel()
    batch.fit(train_eps)

    curve = {
        "episode": [],
        "reward": [],
        "pred_err": [],
        "entropy": [],
        "phys_violation": [],
        "grad_norm": [],
    }
    baseline_reward = 0.0  # running baseline for REINFORCE

    for ep_i in range(n_episodes):
        idx, bias, ep_idle, f_true = env.sample_context()
        true_ep = train_eps[idx]
        L = true_ep.length
        y_true = float(np.log(max(f_true, 1e-12)))

        # Rollout: assemble L gates under the current policy (epsilon-greedy).
        states, actions = [], []
        counts = np.zeros(N_GATES)
        last_gate = None
        ent = 0.0
        eps = eps_start + (eps_end - eps_start) * ep_i / max(n_episodes - 1, 1)
        for t in range(L):
            s = policy.state(bias, ep_idle, t, L, last_gate)
            if rng.random() < eps:
                g = int(rng.integers(N_GATES))
            else:
                g, _ = policy.act(s)
            states.append(s)
            actions.append(g)
            counts[g] += 1
            last_gate = g
            ent += policy.entropy(s)
        ent /= L

        # Transition model prediction + physicality residual.
        y_pred = online.predict(counts, bias, ep_idle)
        f_pred = float(np.exp(y_pred))
        phys_viol = max(0.0, f_pred - 1.05) + max(0.0, -0.05 - f_pred)

        # Reward: prediction quality minus physicality penalty.
        pred_err = abs(y_pred - y_true)
        reward = -pred_err - lambda_phys * phys_viol

        # Online updates: transition model EVERY episode; policy via REINFORCE.
        online.update(counts, bias, ep_idle, y_true)
        baseline_reward = 0.95 * baseline_reward + 0.05 * reward
        advantage = float(np.clip(reward - baseline_reward, -5.0, 5.0))
        grad_norm = policy.update(states, actions, advantage)

        curve["episode"].append(ep_i)
        curve["reward"].append(reward)
        curve["pred_err"].append(pred_err)
        curve["entropy"].append(ent)
        curve["phys_violation"].append(phys_viol)
        curve["grad_norm"].append(grad_norm)

    # ---- Evaluation ----
    # 1. Online transition model vs batch model on held-out long sequences.
    test_counts = np.zeros((len(test_eps), N_GATES))
    y_test = np.empty(len(test_eps))
    for i, ep in enumerate(test_eps):
        for g in ep.gates:
            test_counts[i, g] += 1
        y_test[i] = np.log(max(ep.fidelity, 1e-12))
    online_pred = np.array([
        online.predict(test_counts[i], ep.bias, ep.idle)
        for i, ep in enumerate(test_eps)
    ])
    batch_pred = batch.predict_log(test_eps)
    online_mae = float(np.mean(np.abs(online_pred - y_test)))
    batch_mae = float(np.mean(np.abs(batch_pred - y_test)))

    # 2. Frozen policy rollouts: does the agent pick gates the models predict well?
    n_eval = 500
    eval_rng = np.random.default_rng(seed + 1)
    agent_errs, batch_errs_on_agent = [], []
    gate_hist = np.zeros(N_GATES)
    for _ in range(n_eval):
        idx = eval_rng.integers(len(test_eps))
        ep = test_eps[idx]
        L = ep.length
        counts = np.zeros(N_GATES)
        last = None
        for t in range(L):
            s = policy.state(ep.bias, ep.idle, t, L, last)
            g, _ = policy.act(s)
            counts[g] += 1
            last = g
        y_true = np.log(max(ep.fidelity, 1e-12))
        agent_errs.append(abs(online.predict(counts, ep.bias, ep.idle) - y_true))
        batch_pred = float(
            np.dot(batch.coef_, counts) + batch.ctx_coef_ @ context_vec(ep) + batch.intercept_
        )
        batch_errs_on_agent.append(abs(batch_pred - y_true))
        gate_hist += counts
    gate_hist /= gate_hist.sum()

    # Random-policy control: same online model, random actions.
    rng_rand = np.random.default_rng(seed + 2)
    rand_errs = []
    rand_env = RolloutEnv(train_eps, rng_rand)
    online_rand = OnlineTransitionModel(lr=lr_transition)
    for _ in range(n_episodes):
        idx, bias, ep_idle, f_true = rand_env.sample_context()
        L = train_eps[idx].length
        y_true = float(np.log(max(f_true, 1e-12)))
        counts = np.zeros(N_GATES)
        for _t in range(L):
            counts[rng_rand.integers(N_GATES)] += 1
        online_rand.update(counts, bias, ep_idle, y_true)
    rand_pred = np.array([
        online_rand.predict(test_counts[i], ep.bias, ep.idle)
        for i, ep in enumerate(test_eps)
    ])
    rand_mae = float(np.mean(np.abs(rand_pred - y_test)))

    results = {
        "cell": {"length_cap": length_cap, "idle": idle},
        "config": {"n_episodes": n_episodes, "seq_cap": seq_cap, "seed": seed,
                   "lr_policy": lr_policy, "lr_transition": lr_transition,
                   "lambda_phys": lambda_phys, "eps_start": eps_start, "eps_end": eps_end},
        "learning_curve": curve,
        "eval": {
            "online_transition_mae": online_mae,
            "batch_markov_mae": batch_mae,
            "random_policy_online_mae": rand_mae,
            "agent_rollout_pred_err_mean": float(np.mean(agent_errs)),
            "batch_err_on_agent_seqs_mean": float(np.mean(batch_errs_on_agent)),
            "agent_gate_distribution": gate_hist.tolist(),
            "uniform_gate_distribution": (1 / N_GATES),
        },
        "summary": {
            "reward_first100": float(np.mean(curve["reward"][:100])),
            "reward_last100": float(np.mean(curve["reward"][-100:])),
            "pred_err_first100": float(np.mean(curve["pred_err"][:100])),
            "pred_err_last100": float(np.mean(curve["pred_err"][-100:])),
            "entropy_first100": float(np.mean(curve["entropy"][:100])),
            "entropy_last100": float(np.mean(curve["entropy"][-100:])),
            "phys_violation_rate": float(np.mean(np.array(curve["phys_violation"]) > 0)),
        },
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rl_len{length_cap}_idle{idle}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path}", flush=True)

    # Learning-curve figure.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("pred_err", "per-episode |pred - true| (log-fid)"),
        ("reward", "per-episode reward"),
        ("entropy", "policy entropy (nats/gate)"),
        ("phys_violation", "physicality violation"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        ax.plot(curve["episode"], curve[key], lw=0.4)
        # running mean overlay
        w = 100
        if len(curve[key]) > w:
            rm = np.convolve(curve[key], np.ones(w) / w, mode="valid")
            ax.plot(curve["episode"][w - 1:], rm, lw=2, color="crimson")
        ax.set_title(title)
        ax.set_xlabel("episode")
    fig.suptitle(f"Iterative rollout RL — len{length_cap}/idle{idle}, {n_episodes} episodes")
    fig.tight_layout()
    figpath = out / f"rl_curves_len{length_cap}_idle{idle}.png"
    fig.savefig(figpath, dpi=150)
    print(f"wrote {figpath}", flush=True)
    return results


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/pt_recovery/experiment_data")
    ap.add_argument("--length-cap", type=int, default=40)
    ap.add_argument("--idle", type=int, default=100)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-episodes", type=int, default=3000)
    ap.add_argument("--seq-cap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = run_rl(args.root, args.length_cap, args.idle, args.out,
                 n_episodes=args.n_episodes, seq_cap=args.seq_cap, seed=args.seed)
    print("summary:", json.dumps(res["summary"], indent=2))
    print("eval:", json.dumps(res["eval"], indent=2))


if __name__ == "__main__":
    main()
