"""Nonlinear continuous-drift stream and causal particle-filter baselines."""

from __future__ import annotations

from time import perf_counter_ns

import numpy as np


def transition_mean(x: np.ndarray, dt: float = 0.08) -> np.ndarray:
    """Stable double-well drift with attractors near +/-1."""
    return x + dt * (x - x**3)


def observation_mean(x: np.ndarray) -> np.ndarray:
    return x + 0.25 * x**3


def simulate_nonlinear_streams(
    *, seed: int, n_streams: int, length: int, process_std: float = 0.18,
    observation_std: float = 0.45, artifact_probability: float = 0.01,
    artifact_scale: float = 6.0, dropout_probability: float = 0.002,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty((n_streams, length))
    states[:, 0] = rng.choice([-1.0, 1.0], n_streams) + rng.normal(0, 0.2, n_streams)
    for t in range(1, length):
        states[:, t] = transition_mean(states[:, t - 1]) + rng.normal(0, process_std, n_streams)
        states[:, t] = np.clip(states[:, t], -2.0, 2.0)
    observations = observation_mean(states) + rng.normal(0, observation_std, states.shape)
    artifacts = rng.random(states.shape) < artifact_probability
    observations[artifacts] += rng.normal(0, artifact_scale, artifacts.sum())
    dropouts = rng.random(states.shape) < dropout_probability
    observations[dropouts] = np.nan
    return {"states": states, "observations": observations, "artifacts": artifacts,
            "dropouts": dropouts, "timestamps": np.broadcast_to(np.arange(length), states.shape).copy()}


def _forecast(values: np.ndarray, delay: int) -> np.ndarray:
    values = np.array(values, copy=True)
    for _ in range(delay):
        values = transition_mean(values)
    return values


def robust_ekf(observations: np.ndarray, *, delay: int, process_std: float = 0.18,
               observation_std: float = 0.45, innovation_clip: float = 3.0) -> tuple[np.ndarray, dict[str, float]]:
    """Known-model EKF with bounded standardized innovation."""
    y = np.asarray(observations)
    out = np.empty_like(y)
    started = perf_counter_ns()
    for s in range(len(y)):
        mean, var = 0.0, 1.0
        for t, value in enumerate(y[s]):
            jac_f = 1.0 + 0.08 * (1.0 - 3.0 * mean**2)
            mean = float(transition_mean(np.array(mean)))
            var = jac_f**2 * var + process_std**2
            if np.isfinite(value):
                jac_h = 1.0 + 0.75 * mean**2
                innovation_var = jac_h**2 * var + observation_std**2
                innovation = np.clip(value - observation_mean(np.array(mean)),
                                     -innovation_clip * np.sqrt(innovation_var),
                                     innovation_clip * np.sqrt(innovation_var))
                gain = var * jac_h / innovation_var
                mean += gain * innovation
                var = max((1.0 - gain * jac_h) * var, 1e-8)
            out[s, t] = _forecast(np.array(mean), delay)
    elapsed = perf_counter_ns() - started
    return out, {"ns_per_sample": elapsed / y.size}


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions)


def particle_filter(
    observations: np.ndarray, *, seed: int, particles: int, delay: int,
    process_std: float = 0.18, observation_std: float = 0.45,
    contamination: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Bootstrap filter; contamination adds a broad likelihood floor for artifacts."""
    y = np.asarray(observations)
    out = np.empty_like(y)
    ess_values, resamples = [], 0
    started = perf_counter_ns()
    for s in range(len(y)):
        # Per-stream RNG prevents another stream's future resampling decisions from
        # changing this stream's past particle path in batched replay.
        rng = np.random.default_rng(np.random.SeedSequence([seed, s]))
        cloud = rng.normal(0.0, 1.0, particles)
        weights = np.full(particles, 1.0 / particles)
        for t, value in enumerate(y[s]):
            cloud = transition_mean(cloud) + rng.normal(0, process_std, particles)
            cloud = np.clip(cloud, -2.0, 2.0)
            if np.isfinite(value):
                residual = (value - observation_mean(cloud)) / observation_std
                likelihood = np.exp(-0.5 * residual**2) + contamination
                weights *= likelihood
                total = weights.sum()
                weights = weights / total if total > 1e-300 else np.full(particles, 1.0 / particles)
            ess = 1.0 / np.sum(weights**2)
            ess_values.append(ess)
            if ess < particles / 2:
                cloud = cloud[_systematic_resample(weights, rng)]
                weights.fill(1.0 / particles)
                resamples += 1
            out[s, t] = np.sum(weights * _forecast(cloud, delay))
    elapsed = perf_counter_ns() - started
    return out, {"ns_per_sample": elapsed / y.size,
                 "mean_ess_fraction": float(np.mean(ess_values) / particles),
                 "resamples_per_sample": resamples / y.size}


def regression_metrics(prediction: np.ndarray, states: np.ndarray, *, delay: int) -> dict[str, float]:
    usable = states.shape[1] - delay
    error = prediction[:, :usable] - states[:, delay:]
    return {"mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error))),
            "p95_absolute_error": float(np.quantile(np.abs(error), 0.95))}
