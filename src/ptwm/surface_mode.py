"""Causal nuisance-mode filtering from surface-code detector-count emissions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_count_emission(
    nominal_counts: np.ndarray, burst_counts: np.ndarray, detector_count: int,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Return a smoothed P(count | mode) table with shape (2, detectors + 1)."""
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    table = np.empty((2, detector_count + 1), dtype=float)
    for mode, counts in enumerate((nominal_counts, burst_counts)):
        if np.any((counts < 0) | (counts > detector_count)):
            raise ValueError("detector count is out of range")
        mass = np.bincount(counts.astype(int), minlength=detector_count + 1).astype(float)
        mass += pseudocount
        table[mode] = mass / mass.sum()
    return table


def count_mode_posterior(
    counts: np.ndarray, emission: np.ndarray, transition: np.ndarray,
    stationary: np.ndarray, *, temporal: bool,
) -> np.ndarray:
    """Return causal P(mode=burst) for independent streams of detector counts."""
    if counts.ndim != 2:
        raise ValueError("counts must have shape (streams, time)")
    if emission.shape[0] != 2 or transition.shape != (2, 2):
        raise ValueError("binary emission and transition tables are required")
    posterior = np.empty(counts.shape, dtype=float)
    previous = np.broadcast_to(stationary, (counts.shape[0], 2)).copy()
    for time in range(counts.shape[1]):
        prior = previous @ transition if temporal else stationary
        current = prior * emission[:, counts[:, time]].T
        current /= current.sum(axis=1, keepdims=True)
        posterior[:, time] = current[:, 1]
        previous = current
    return posterior


def fit_affine_log_likelihood_ratio(
    first: np.ndarray,
    second: np.ndarray,
    *,
    l2: float = 1e-3,
) -> dict[str, np.ndarray | float]:
    """Fit a standardized affine regime log-likelihood ratio on balanced data."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("feature matrices must be two-dimensional and aligned")
    stacked = np.vstack((first, second))
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    features = np.vstack(((first - mean) / scale, (second - mean) / scale))
    labels = np.concatenate((-np.ones(len(first)), np.ones(len(second))))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights, bias = parameters[:-1], parameters[-1]
        margin = labels * (features @ weights + bias)
        value = np.logaddexp(0.0, -margin).mean() + 0.5 * l2 * (weights @ weights)
        factor = -labels * expit(-margin) / len(labels)
        gradient = np.concatenate((features.T @ factor + l2 * weights, [factor.sum()]))
        return float(value), gradient

    result = minimize(
        objective,
        np.zeros(features.shape[1] + 1),
        jac=True,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"affine likelihood-ratio fit failed: {result.message}")
    return {
        "mean": mean,
        "scale": scale,
        "weights": result.x[:-1],
        "bias": float(result.x[-1]),
    }


def affine_log_likelihood_ratio(
    features: np.ndarray, model: dict[str, np.ndarray | float]
) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    return standardized @ model["weights"] + model["bias"]


def log_likelihood_mode_posterior(
    log_likelihood_ratio: np.ndarray,
    transition: np.ndarray,
    stationary: np.ndarray,
    *,
    temporal: bool,
) -> np.ndarray:
    """Accumulate per-record log P(x|mode=1)/P(x|mode=0) causally."""
    if log_likelihood_ratio.ndim != 2:
        raise ValueError("log-likelihood ratio must have shape (streams, time)")
    posterior = np.empty_like(log_likelihood_ratio, dtype=float)
    previous = np.broadcast_to(stationary, (log_likelihood_ratio.shape[0], 2)).copy()
    for time in range(log_likelihood_ratio.shape[1]):
        prior = (
            previous @ transition
            if temporal
            else np.broadcast_to(stationary, previous.shape)
        )
        log_odds = (
            np.log(np.clip(prior[:, 1], 1e-300, None))
            - np.log(np.clip(prior[:, 0], 1e-300, None))
            + log_likelihood_ratio[:, time]
        )
        probability = expit(log_odds)
        posterior[:, time] = probability
        previous = np.column_stack((1.0 - probability, probability))
    return posterior
