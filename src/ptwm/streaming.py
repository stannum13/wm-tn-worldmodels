"""Causal streaming benchmark for delayed control under hidden switching noise.

The benchmark represents a detuning condition observed through repeated noisy probes.
It deliberately separates physical switching from detector artifacts and scores the
control selected now against the hidden condition when that control takes effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class StreamParameters:
    transition: np.ndarray
    means: np.ndarray
    stds: np.ndarray


@dataclass(frozen=True)
class CausalHead:
    """Small ridge-fitted causal head with optional univariate spline edges."""

    weights: np.ndarray
    feature_min: np.ndarray
    feature_max: np.ndarray
    knots: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.clip(_head_design(features, self.feature_min, self.feature_max, self.knots) @ self.weights, 0.0, 1.0)


def _head_design(x: np.ndarray, low: np.ndarray, high: np.ndarray, knots: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scaled = np.clip((x - low) / np.maximum(high - low, 1e-9), 0.0, 1.0)
    if knots <= 1:
        return np.c_[np.ones(len(x)), scaled]
    centers = np.linspace(0.0, 1.0, knots)
    width = 1.0 / (knots - 1)
    basis = np.maximum(1.0 - np.abs(scaled[..., None] - centers) / width, 0.0)
    return np.c_[np.ones(len(x)), basis.reshape(len(x), -1)]


def fit_causal_head(features: np.ndarray, targets: np.ndarray, *, knots: int = 6, ridge: float = 1e-3) -> CausalHead:
    """Fit a linear head (knots=1) or additive piecewise-linear KAN-like head."""
    x = np.asarray(features, dtype=float)
    low, high = np.quantile(x, [0.01, 0.99], axis=0)
    design = _head_design(x, low, high, knots)
    weights = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ np.asarray(targets))
    return CausalHead(weights, low, high, knots)


def causal_context_features(observations: np.ndarray, belief: np.ndarray, *, window: int, stride: int, delay: int, states: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Build bounded-influence past-only summaries at slow-head update ticks."""
    rows, targets, locations = [], [], []
    for s in range(observations.shape[0]):
        for tick in range(window - 1, observations.shape[1] - delay, stride):
            chunk = observations[s, tick - window + 1:tick + 1]
            finite = chunk[np.isfinite(chunk)]
            median = np.median(finite) if len(finite) else 0.0
            mad = np.median(np.abs(finite - median)) if len(finite) else 0.0
            last = np.clip(finite[-1], median - 4 * max(mad, 1e-3), median + 4 * max(mad, 1e-3)) if len(finite) else 0.0
            rows.append([belief[s, tick], median, mad, last, 1.0 - len(finite) / window])
            locations.append((s, tick))
            if states is not None:
                targets.append(states[s, tick + delay])
    return np.asarray(rows), (np.asarray(targets) if states is not None else None), np.asarray(locations)


def slow_head_denoiser(base: np.ndarray, head_values: np.ndarray, locations: np.ndarray, *, stride: int, mix: float = 0.5, smoothing: float = 0.25) -> np.ndarray:
    """Asynchronously hold a slow context estimate for cheap hot-path fusion."""
    if stride < 1:
        raise ValueError("stride must be positive")
    out = np.array(base, copy=True)
    by_stream: dict[int, list[tuple[int, float]]] = {}
    for (stream, tick), value in zip(locations, head_values):
        by_stream.setdefault(int(stream), []).append((int(tick), float(value)))
    for stream, updates in by_stream.items():
        held, cursor, active = base[stream, 0], 0, False
        for tick in range(base.shape[1]):
            if cursor < len(updates) and tick == updates[cursor][0]:
                candidate = updates[cursor][1]
                held = candidate if not active else held + smoothing * (candidate - held)
                active = True
                cursor += 1
            if active:
                out[stream, tick] = (1.0 - mix) * base[stream, tick] + mix * held
    return out


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Return the stationary distribution of a two-state transition matrix."""
    transition = np.asarray(transition, dtype=float)
    a = transition.T - np.eye(2)
    a[-1] = 1.0
    b = np.array([0.0, 1.0])
    return np.linalg.solve(a, b)


def simulate_switching_streams(
    *,
    seed: int,
    n_streams: int,
    length: int,
    transition: np.ndarray | None = None,
    means: tuple[float, float] = (-0.8, 0.8),
    observation_std: float = 1.0,
    artifact_probability: float = 0.005,
    artifact_scale: float = 7.0,
    dropout_probability: float = 0.002,
) -> dict[str, np.ndarray]:
    """Simulate latent detuning states and a timestamped noisy probe stream."""
    rng = np.random.default_rng(seed)
    transition = np.asarray(
        transition if transition is not None else [[0.995, 0.005], [0.04, 0.96]],
        dtype=float,
    )
    if transition.shape != (2, 2) or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("transition must be a 2x2 row-stochastic matrix")
    initial = stationary_distribution(transition)
    states = np.empty((n_streams, length), dtype=np.int8)
    states[:, 0] = rng.choice(2, size=n_streams, p=initial)
    for t in range(1, length):
        switch_probability = transition[states[:, t - 1], 1]
        states[:, t] = (rng.random(n_streams) < switch_probability).astype(np.int8)
    observations = np.asarray(means)[states] + rng.normal(
        0.0, observation_std, size=states.shape
    )
    artifacts = rng.random(states.shape) < artifact_probability
    observations[artifacts] += rng.normal(0.0, artifact_scale, artifacts.sum())
    dropouts = rng.random(states.shape) < dropout_probability
    observations[dropouts] = np.nan
    return {
        "states": states,
        "observations": observations,
        "artifacts": artifacts,
        "dropouts": dropouts,
        "timestamps": np.broadcast_to(np.arange(length), states.shape).copy(),
        "transition": transition,
    }


def _emission_log_prob(observations: np.ndarray, params: StreamParameters) -> np.ndarray:
    y = np.asarray(observations, dtype=float)
    safe = np.where(np.isnan(y), 0.0, y)
    out = np.empty(y.shape + (2,), dtype=float)
    for state in range(2):
        z = (safe - params.means[state]) / params.stds[state]
        out[..., state] = -0.5 * z * z - np.log(params.stds[state])
    out[np.isnan(y)] = 0.0
    return out


def fit_gaussian_hmm(
    observations: np.ndarray,
    *,
    iterations: int = 30,
    min_std: float = 0.2,
    clip_quantile: float = 0.995,
) -> StreamParameters:
    """Fit a two-state Gaussian HMM with Baum-Welch on independent streams.

    Extreme detector artifacts are winsorized using only the fitting set. State labels
    are ordered by emission mean, making state 1 the positive-detuning condition.
    """
    y = np.asarray(observations, dtype=float)
    if y.ndim != 2 or y.shape[1] < 3:
        raise ValueError("observations must have shape (streams, time>=3)")
    finite = y[np.isfinite(y)]
    if finite.size < 10:
        raise ValueError("too few finite observations")
    bound = float(np.quantile(np.abs(finite), clip_quantile))
    clean = np.clip(y, -bound, bound)
    means = np.quantile(finite, [0.25, 0.75])
    stds = np.full(2, max(float(np.std(finite)), min_std))
    transition = np.array([[0.98, 0.02], [0.05, 0.95]], dtype=float)
    initial = stationary_distribution(transition)

    for _ in range(iterations):
        params = StreamParameters(transition, means, stds)
        log_emit = _emission_log_prob(clean, params)
        n_streams, n = clean.shape
        log_transition = np.log(transition + 1e-15)
        alpha = np.empty((n_streams, n, 2))
        alpha[:, 0] = np.log(initial + 1e-15) + log_emit[:, 0]
        for t in range(1, n):
            alpha[:, t] = log_emit[:, t] + logsumexp(
                alpha[:, t - 1, :, None] + log_transition[None, :, :], axis=1
            )
        beta = np.zeros((n_streams, n, 2))
        for t in range(n - 2, -1, -1):
            beta[:, t] = logsumexp(
                log_transition[None, :, :]
                + log_emit[:, t + 1, None, :]
                + beta[:, t + 1, None, :],
                axis=2,
            )
        log_gamma = alpha + beta
        log_gamma -= logsumexp(log_gamma, axis=2, keepdims=True)
        gamma = np.exp(log_gamma)
        initial_sum = gamma[:, 0].sum(axis=0)
        log_xi = (
            alpha[:, :-1, :, None]
            + log_transition[None, None, :, :]
            + log_emit[:, 1:, None, :]
            + beta[:, 1:, None, :]
        )
        log_xi -= logsumexp(log_xi, axis=(2, 3), keepdims=True)
        xi_sum = np.exp(log_xi).sum(axis=(0, 1))
        valid = np.isfinite(clean)
        weights = gamma * valid[..., None]
        safe_clean = np.where(valid, clean, 0.0)
        gamma_sum = weights.sum(axis=(0, 1))
        gamma_y = (weights * safe_clean[..., None]).sum(axis=(0, 1))
        gamma_y2 = (weights * safe_clean[..., None] ** 2).sum(axis=(0, 1))
        transition = (xi_sum + 1e-3) / (xi_sum.sum(axis=1, keepdims=True) + 2e-3)
        initial = (initial_sum + 1e-3) / (initial_sum.sum() + 2e-3)
        means = gamma_y / np.maximum(gamma_sum, 1e-12)
        variances = gamma_y2 / np.maximum(gamma_sum, 1e-12) - means**2
        stds = np.sqrt(np.maximum(variances, min_std**2))
        if means[0] > means[1]:
            means = means[::-1]
            stds = stds[::-1]
            initial = initial[::-1]
            transition = transition[::-1, ::-1]
    return StreamParameters(transition=transition, means=means, stds=stds)


def causal_hmm_filter(observations: np.ndarray, params: StreamParameters) -> np.ndarray:
    """Return P(state=1 | observations through t) for every stream and time."""
    y = np.asarray(observations, dtype=float)
    log_emits = _emission_log_prob(y, params)
    log_emits -= np.max(log_emits, axis=2, keepdims=True)
    emits = np.exp(log_emits)
    belief = np.broadcast_to(stationary_distribution(params.transition), (len(y), 2)).copy()
    out = np.empty_like(y)
    for t in range(y.shape[1]):
        if t:
            belief = belief @ params.transition
        belief *= emits[:, t]
        belief /= belief.sum(axis=1, keepdims=True)
        out[:, t] = belief[:, 1]
    return out


def ewma_estimate(observations: np.ndarray, *, alpha: float, center: float, scale: float) -> np.ndarray:
    """Causal robust EWMA mapped to a soft action in [0, 1]."""
    y = np.asarray(observations, dtype=float)
    out = np.empty_like(y)
    value = np.full(y.shape[0], center)
    for t in range(y.shape[1]):
        sample = np.where(np.isfinite(y[:, t]), np.clip(y[:, t], center - 4 * scale, center + 4 * scale), value)
        value = alpha * sample + (1.0 - alpha) * value
        out[:, t] = np.clip(0.5 + (value - center) / (2 * scale), 0.0, 1.0)
    return out


def forecast_belief(probability: np.ndarray, transition: np.ndarray, delay: int) -> np.ndarray:
    """Propagate a causal state belief to the time a selected action takes effect."""
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    power = np.linalg.matrix_power(transition, delay)
    return (1.0 - probability) * power[0, 1] + probability * power[1, 1]


def score_delayed_prediction(
    action: np.ndarray,
    states: np.ndarray,
    *,
    delay: int,
    artifacts: np.ndarray | None = None,
) -> dict[str, float]:
    """Score a soft causal forecast against the latent state at t + delay."""
    if delay >= states.shape[1]:
        raise ValueError("delay must be shorter than the stream")
    usable = states.shape[1] - delay
    chosen = np.asarray(action)[:, :usable]
    target = np.asarray(states)[:, delay:].astype(float)
    error = (chosen - target) ** 2
    binary = chosen >= 0.5
    metrics = {
        "brier_loss": float(error.mean()),
        "classification_error": float(np.mean(binary != target.astype(bool))),
        "false_action_rate": float(binary[target == 0].mean()),
        "missed_fault_rate": float((~binary[target == 1]).mean()),
    }
    if artifacts is not None:
        artifact_now = np.asarray(artifacts)[:, :usable]
        metrics["action_on_artifact_rate"] = float(binary[artifact_now].mean()) if artifact_now.any() else 0.0
    return metrics


def benchmark_estimators(
    data: dict[str, np.ndarray], params: StreamParameters, delays: list[int]
) -> dict[int, dict[str, dict[str, float]]]:
    """Compare robust instantaneous, causal filtering, and delayed forecasting policies."""
    observations = data["observations"]
    started = perf_counter_ns()
    instant = np.clip(0.5 + (np.nan_to_num(observations, nan=0.0) - params.means.mean()) / (2 * np.diff(params.means)[0]), 0.0, 1.0)
    instant_ns = perf_counter_ns() - started
    started = perf_counter_ns()
    ewma = ewma_estimate(observations, alpha=0.2, center=params.means.mean(), scale=np.diff(params.means)[0])
    ewma_ns = perf_counter_ns() - started
    started = perf_counter_ns()
    belief = causal_hmm_filter(observations, params)
    filter_ns = perf_counter_ns() - started
    samples = observations.size
    reports: dict[int, dict[str, dict[str, float]]] = {}
    for delay in delays:
        policies = {
            "instantaneous": (instant, instant_ns),
            "ewma": (ewma, ewma_ns),
            "hmm_filter_current": (belief, filter_ns),
            "hmm_delay_forecast": (forecast_belief(belief, params.transition, delay), filter_ns),
        }
        reports[delay] = {}
        for name, (action, elapsed) in policies.items():
            metrics = score_delayed_prediction(
                action, data["states"], delay=delay, artifacts=data.get("artifacts")
            )
            metrics["processing_ns_per_sample"] = float(elapsed / samples)
            reports[delay][name] = metrics
    return reports
