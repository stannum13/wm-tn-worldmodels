"""Fit small residual actions to logical outcomes or conditional teacher risks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_soft_action(features, probability, *, l2=.001):
    features = np.asarray(features, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if features.ndim != 2 or probability.shape != (len(features),) or not len(features):
        raise ValueError("aligned nonempty features and soft targets required")
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("soft labels must be probabilities")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    x = (features - mean) / scale

    def objective(parameters):
        w, b = parameters[:-1], parameters[-1]
        score = x @ w + b
        value = np.mean(np.logaddexp(0, score) - probability * score) + .5 * l2 * (w @ w)
        residual = (expit(score) - probability) / len(probability)
        gradient = np.r_[x.T @ residual + l2 * w, residual.sum()]
        return value, gradient

    fit = minimize(objective, np.zeros(x.shape[1] + 1), method="L-BFGS-B", jac=True)
    if not fit.success:
        raise RuntimeError(f"soft action fit failed: {fit.message}")
    return {"mean": mean, "scale": scale, "weights": fit.x[:-1], "bias": float(fit.x[-1])}


def spline_knots(features):
    """Five training-only knots on the four continuous graph/action inputs."""
    return np.quantile(features[:, -4:], [.1, .25, .5, .75, .9], axis=0).T


def spline_features(features, knots):
    """An additive linear-spline head: original features plus 20 hinge terms."""
    continuous = features[:, -4:]
    hinges = np.maximum(continuous[:, :, None] - knots[None, :, :], 0)
    return np.column_stack((features, hinges.reshape(len(features), -1)))


def teacher_logical_probability(tables, syndromes, *, switch_probability=.02):
    """P(logical=1 | current and past syndrome), under the known two-mode DEM."""
    output = np.empty(syndromes.shape, dtype=float)
    previous = np.full(len(syndromes), .5)
    for step in range(syndromes.shape[1]):
        prior = switch_probability + (1 - 2 * switch_probability) * previous
        current = tables[:, :, syndromes[:, step]]
        joint = (1 - prior) * current[0] + prior * current[1]
        total = joint.sum(axis=0)
        if np.any(total <= 0):
            raise ArithmeticError("zero probability syndrome")
        output[:, step] = joint[1] / total
        previous = prior * current[1].sum(axis=0) / total
    return output
