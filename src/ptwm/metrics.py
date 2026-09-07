"""Metrics for Experiment A.

Primary metric: mean squared error in log-fidelity space (log MSE), because sequence
fidelities decay multiplicatively and the log scale is where model errors are comparable
across lengths. Secondary: absolute fidelity error, per-length error curves (forecast
horizon analysis), and causality residuals for latent models.
"""

from __future__ import annotations

import numpy as np

from .data import Episode


def log_mse(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    return float(np.mean((y_true_log - y_pred_log) ** 2))


def fidelity_mae(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    return float(np.mean(np.abs(np.exp(y_true_log) - np.exp(y_pred_log))))


def per_length_errors(
    eps: list[Episode], y_pred_log: np.ndarray
) -> dict[int, dict[str, float]]:
    """Group absolute fidelity errors by sequence length (forecast-horizon curve)."""
    y_true = np.array([ep.fidelity for ep in eps])
    y_pred = np.exp(y_pred_log)
    abs_err = np.abs(y_true - y_pred)
    out: dict[int, dict[str, float]] = {}
    for L in sorted({ep.length for ep in eps}):
        mask = np.array([ep.length == L for ep in eps])
        out[int(L)] = {
            "mae": float(abs_err[mask].mean()),
            "bias": float((y_pred[mask] - y_true[mask]).mean()),
            "n": int(mask.sum()),
        }
    return out


def horizon_of_tolerance(
    per_length: dict[int, dict[str, float]], tol: float
) -> int | None:
    """Longest length whose mean absolute error stays under tol (forecast horizon)."""
    ok = [L for L, m in sorted(per_length.items()) if m["mae"] <= tol]
    return ok[-1] if ok else None


def physicality_residuals(y_pred_log: np.ndarray) -> dict[str, float]:
    """Observable-space physicality: predicted fidelities must lie in [0, 1+eps]."""
    y_pred = np.exp(y_pred_log)
    return {
        "frac_above_one": float(np.mean(y_pred > 1.0 + 1e-6)),
        "frac_below_zero": float(np.mean(y_pred < -1e-6)),
        "max_value": float(y_pred.max()),
        "min_value": float(y_pred.min()),
    }


def summarize_all(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    return {
        "log_mse": log_mse(y_true_log, y_pred_log),
        "fidelity_mae": fidelity_mae(y_true_log, y_pred_log),
        "physicality": physicality_residuals(y_pred_log),
    }
