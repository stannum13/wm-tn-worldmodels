"""Small, auditable utilities for the public Rigetti Ankaa-2 feedback data."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .streaming import _head_design


@dataclass(frozen=True)
class LogisticIQHead:
    weights: np.ndarray
    feature_min: np.ndarray
    feature_max: np.ndarray
    knots: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        design = _head_design(features, self.feature_min, self.feature_max, self.knots)
        return expit(design @ self.weights)


def load_measurement_fidelity(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load labelled complex I/Q calibration shots and hardware hard decisions."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by real-data runner
        raise RuntimeError("Install the 'real' optional dependencies to read Rigetti HDF5") from exc
    soft_parts, hard_parts, label_parts = [], [], []
    with h5py.File(path, "r") as handle:
        root = handle["reference_data/measurement_fidelity"]
        for prepared, label in (("zeros", 0.0), ("ones", 1.0)):
            result = root[f"{prepared}/result"]
            soft = np.asarray(result["soft_measurements/50"]).reshape(-1)
            hard = np.asarray(result["hard_measurements/50"]).reshape(-1)
            soft_parts.append(np.c_[soft.real, soft.imag])
            hard_parts.append(hard.astype(float))
            label_parts.append(np.full(len(soft), label))
    return np.vstack(soft_parts), np.concatenate(hard_parts), np.concatenate(label_parts)


def chronological_preparation_split(labels: np.ndarray, train_fraction: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Take the early portion of each prepared-state trace for train, later for test."""
    labels = np.asarray(labels)
    train, test = [], []
    for label in np.unique(labels):
        locations = np.flatnonzero(labels == label)
        boundary = int(len(locations) * train_fraction)
        train.extend(locations[:boundary])
        test.extend(locations[boundary:])
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def probability_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(labels, dtype=float)
    prediction = probability >= 0.5
    edges = np.linspace(0.0, 1.0, 11)
    calibration_error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (probability >= left) & (probability < right if right < 1 else probability <= right)
        if np.any(selected):
            calibration_error += np.mean(selected) * abs(np.mean(probability[selected]) - np.mean(labels[selected]))
    return {
        "classification_error": float(np.mean(prediction != labels)),
        "brier_loss": float(np.mean((probability - labels) ** 2)),
        "negative_log_likelihood": float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))),
        "expected_calibration_error_10bin": float(calibration_error),
    }


def block_error_differences(
    first: np.ndarray, second: np.ndarray, labels: np.ndarray, *, block_size: int
) -> np.ndarray:
    """Paired error-rate differences in contiguous blocks within each preparation."""
    differences = []
    for label in np.unique(labels):
        locations = np.flatnonzero(labels == label)
        for start in range(0, len(locations) - block_size + 1, block_size):
            block = locations[start : start + block_size]
            first_error = (np.asarray(first)[block] >= 0.5) != labels[block]
            second_error = (np.asarray(second)[block] >= 0.5) != labels[block]
            differences.append(np.mean(first_error) - np.mean(second_error))
    return np.asarray(differences)


def block_score_differences(
    first: np.ndarray, second: np.ndarray, labels: np.ndarray, *, block_size: int,
    score: str,
) -> np.ndarray:
    """First-minus-second paired score differences in contiguous preparation blocks."""
    first = np.clip(np.asarray(first), 1e-6, 1.0 - 1e-6)
    second = np.clip(np.asarray(second), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(labels)
    if score == "brier_loss":
        first_values, second_values = (first - labels) ** 2, (second - labels) ** 2
    elif score == "negative_log_likelihood":
        first_values = -(labels * np.log(first) + (1 - labels) * np.log(1 - first))
        second_values = -(labels * np.log(second) + (1 - labels) * np.log(1 - second))
    else:
        raise ValueError(f"unsupported score: {score}")
    differences = []
    for label in np.unique(labels):
        locations = np.flatnonzero(labels == label)
        for start in range(0, len(locations) - block_size + 1, block_size):
            block = locations[start : start + block_size]
            differences.append(np.mean(first_values[block] - second_values[block]))
    return np.asarray(differences)


def timed_head_prediction(head: LogisticIQHead, features: np.ndarray, repeats: int = 20) -> tuple[np.ndarray, dict[str, float]]:
    """Measure vectorized throughput and Python batch-one call latency."""
    prediction = head.predict(features)
    started = perf_counter_ns()
    for _ in range(repeats):
        head.predict(features)
    batch_ns = (perf_counter_ns() - started) / (repeats * len(features))
    calls = []
    for row in features[: min(2000, len(features))]:
        started = perf_counter_ns()
        head.predict(row[None, :])
        calls.append(perf_counter_ns() - started)
    return prediction, {
        "vectorized_ns_per_sample": float(batch_ns),
        "python_batch_one_p50_ns": float(np.quantile(calls, 0.50)),
        "python_batch_one_p99_ns": float(np.quantile(calls, 0.99)),
    }


def _fit_logistic_head(
    features: np.ndarray, labels: np.ndarray, *, knots: int, ridge: float = 1e-4
) -> LogisticIQHead:
    features, labels = np.asarray(features), np.asarray(labels)
    low, high = np.quantile(features, [0.01, 0.99], axis=0)
    design = _head_design(features, low, high, knots)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        scores = design @ weights
        probability = expit(scores)
        penalty = ridge * np.sum(weights[1:] ** 2) / 2.0
        loss = np.mean(np.logaddexp(0.0, scores) - labels * scores) + penalty
        gradient = design.T @ (probability - labels) / len(labels)
        gradient[1:] += ridge * weights[1:]
        return float(loss), gradient

    result = minimize(
        objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"logistic head fit failed: {result.message}")
    return LogisticIQHead(result.x, low, high, knots)


def fit_iq_heads(features: np.ndarray, labels: np.ndarray) -> dict[str, LogisticIQHead]:
    """Matched logistic comparison: affine versus a tiny additive spline/KAN head."""
    return {
        "linear_logistic_iq": _fit_logistic_head(features, labels, knots=1),
        "spline_kan_logistic_iq": _fit_logistic_head(features, labels, knots=6),
    }
