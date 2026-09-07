"""Minimal directional tests for memory and physical-state constraints.

The generators are deliberately small and transparent. They are controls for the
real-data study, not substitutes for experimental evidence.
"""

from __future__ import annotations

import numpy as np


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def generate_memory_process(
    *, seed: int, n_sequences: int, horizon: int, noise: float = 0.015, memory_gain: float = 0.35
) -> dict[str, np.ndarray]:
    """Scalar controlled process with one genuine hidden-memory term."""
    rng = np.random.default_rng(seed)
    controls = rng.choice([-1.0, 1.0], size=(n_sequences, horizon))
    states = np.zeros((n_sequences, horizon + 1))
    for t in range(horizon):
        previous = controls[:, t - 1] if t else 0.0
        states[:, t + 1] = (
            0.72 * states[:, t] + 0.42 * controls[:, t] + memory_gain * previous
            + rng.normal(0.0, noise, n_sequences)
        )
    return {"observations": states, "controls": controls}


def generate_qubit_process(
    *, seed: int, n_sequences: int, horizon: int, control_scale: float = 1.0
) -> dict[str, np.ndarray]:
    """Damped qubit Bloch-vector dynamics with a scalar drive."""
    rng = np.random.default_rng(seed)
    controls = rng.uniform(-control_scale, control_scale, size=(n_sequences, horizon))
    states = np.zeros((n_sequences, horizon + 1, 3))
    states[:, 0] = rng.uniform(-0.25, 0.25, size=(n_sequences, 3))
    A = np.diag([0.94, 0.90, 0.86])
    drive = np.array([0.08, -0.05, 0.03])
    for t in range(horizon):
        states[:, t + 1] = states[:, t] @ A.T + controls[:, t, None] * drive
        states[:, t + 1] += rng.normal(0.0, 0.002, size=(n_sequences, 3))
        norms = np.linalg.norm(states[:, t + 1], axis=1)
        over = norms > 0.98
        states[over, t + 1] *= (0.98 / norms[over])[:, None]
    return {"observations": states, "controls": controls}


def _features(observations: np.ndarray, controls: np.ndarray, memory: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for seq, us in zip(observations, controls):
        for t in range(len(us)):
            history = [us[t - k] if t >= k else 0.0 for k in range(memory + 1)]
            x.append(np.r_[1.0, seq[t], history])
            y.append(seq[t + 1])
    return np.asarray(x), np.asarray(y)


def fit_linear_predictor(data: dict[str, np.ndarray], *, memory: int = 0) -> np.ndarray:
    x, y = _features(data["observations"], data["controls"], memory)
    return _ridge(x, y)


def rollout_linear(
    weights: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    horizon: int,
    memory: int | None = None,
    return_states: bool = False,
) -> float | dict[str, np.ndarray]:
    """Recursive rollout; horizon is measured from each sequence's initial state."""
    obs, controls = data["observations"], data["controls"]
    state_dim = 1 if obs.ndim == 2 else obs.shape[-1]
    memory = weights.shape[0] - state_dim - 2 if memory is None else memory
    predictions, targets = [], []
    for seq, us in zip(obs, controls):
        state = np.array(seq[0], copy=True)
        for t in range(min(horizon, len(us))):
            history = [us[t - k] if t >= k else 0.0 for k in range(memory + 1)]
            x = np.r_[1.0, state, history]
            state = np.asarray(x @ weights)
            predictions.append(state.copy())
            targets.append(seq[t + 1])
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if return_states:
        return {"predictions": predictions, "targets": targets, "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2)))}
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def trace_distance_bloch(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(0.5 * np.linalg.norm(predictions - targets, axis=-1)))


def rollout_bloch_constrained(weights: np.ndarray, data: dict[str, np.ndarray], *, horizon: int) -> dict[str, float]:
    raw = rollout_linear(weights, data, horizon=horizon, memory=0, return_states=True)
    predictions = raw["predictions"].copy()
    norms = np.linalg.norm(predictions, axis=1)
    mask = norms > 1.0
    predictions[mask] /= norms[mask, None]
    targets = raw["targets"]
    return {
        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "trace_distance": trace_distance_bloch(predictions, targets),
        "min_eigenvalue": float(np.min((1.0 - np.linalg.norm(predictions, axis=1)) / 2.0)),
        "invalid_fraction": float(np.mean(np.linalg.norm(predictions, axis=1) > 1.0 + 1e-10)),
    }
