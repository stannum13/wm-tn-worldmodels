"""Small baselines for the public correlated-noise RB benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def load_rb_json(path: str | Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("RB file must contain a list")
    rows = []
    for row in payload:
        if not isinstance(row, dict) or "cl_ops" not in row or "p0" not in row:
            raise ValueError("each RB row needs cl_ops and p0")
        rows.append({"cl_ops": [int(x) for x in row["cl_ops"]], "p0": float(row["p0"])})
    return rows


def split_by_length(rows: Iterable[Mapping[str, object]], *, max_train_length: int) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    train, test = [], []
    for row in rows:
        target = train if len(row["cl_ops"]) <= max_train_length else test
        target.append(row)
    if not train or not test:
        raise ValueError("length split produced an empty partition")
    return train, test


def make_features(rows: Iterable[Mapping[str, object]], *, n_actions: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Length + action counts + ordered adjacent-pair counts.

    The pair counts are the smallest order-sensitive extension over the paper's
    length-only RB summary. They do not use p0, so there is no target leakage.
    """
    features, targets = [], []
    for row in rows:
        actions = [int(a) for a in row["cl_ops"]]
        if any(a < 0 or a >= n_actions for a in actions):
            raise ValueError("action id outside configured alphabet")
        counts = np.bincount(actions, minlength=n_actions).astype(float)
        pairs = np.zeros((n_actions, n_actions), dtype=float)
        for left, right in zip(actions, actions[1:]):
            pairs[left, right] += 1.0
        features.append(np.r_[len(actions), counts, pairs.ravel()])
        targets.append(float(row["p0"]))
    return np.asarray(features), np.asarray(targets)


def fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> np.ndarray:
    x1 = np.c_[np.ones(len(x)), x]
    return np.linalg.solve(x1.T @ x1 + alpha * np.eye(x1.shape[1]), x1.T @ y)


def predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(x)), x] @ weights


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predictions) - targets) ** 2)))


def fit_rb_decay(lengths: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fit the standard RB form A * alpha**length + B."""
    from scipy.optimize import least_squares

    lengths = np.asarray(lengths, dtype=float)
    targets = np.asarray(targets, dtype=float)

    def residual(params):
        amplitude, alpha, floor = params
        return amplitude * alpha**lengths + floor - targets

    fit = least_squares(
        residual,
        x0=np.array([0.5, 0.99, 0.5]),
        bounds=(np.array([-1.0, 0.0, 0.0]), np.array([1.5, 1.05, 1.1])),
    )
    return fit.x


def predict_rb_decay(params: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    amplitude, alpha, floor = params
    return amplitude * alpha ** np.asarray(lengths, dtype=float) + floor
