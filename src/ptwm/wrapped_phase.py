"""Causal filters for a nonlinear, multimodal wrapped-phase positive control."""

from __future__ import annotations

from time import perf_counter_ns

import numpy as np


TAU = 2.0 * np.pi


def wrap_phase(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value) + np.pi) % TAU - np.pi


def simulate_wrapped_phase(
    *, seed: int, n_streams: int, length: int, drift: float = 0.06,
    process_std: float = 0.08, observation_std: float = 0.18,
    quadrature_interval: int = 8, artifact_probability: float = 0.02,
    artifact_std: float = 3.0, dropout_probability: float = 0.002,
) -> dict[str, np.ndarray]:
    """Simulate noisy cosine readout with sparse orthogonal phase probes."""
    rng = np.random.default_rng(seed)
    states = np.empty((n_streams, length))
    states[:, 0] = rng.uniform(-np.pi, np.pi, n_streams)
    for t in range(1, length):
        states[:, t] = wrap_phase(
            states[:, t - 1] + drift + rng.normal(0.0, process_std, n_streams)
        )
    one_probe = np.zeros(length)
    one_probe[quadrature_interval - 1 :: quadrature_interval] = np.pi / 2.0
    probes = np.broadcast_to(one_probe, states.shape).copy()
    observations = np.cos(states - probes) + rng.normal(0.0, observation_std, states.shape)
    artifacts = rng.random(states.shape) < artifact_probability
    observations[artifacts] += rng.normal(0.0, artifact_std, artifacts.sum())
    dropouts = rng.random(states.shape) < dropout_probability
    observations[dropouts] = np.nan
    return {
        "states": states,
        "observations": observations,
        "probe_phases": probes,
        "artifacts": artifacts,
        "dropouts": dropouts,
    }


def _normal_density(residual: np.ndarray, std: float) -> np.ndarray:
    return np.exp(-0.5 * (residual / std) ** 2) / (np.sqrt(2.0 * np.pi) * std)


def _mixture_likelihood(
    residual: np.ndarray, observation_std: float, contamination: float, artifact_std: float
) -> np.ndarray:
    clean = _normal_density(residual, observation_std)
    if contamination <= 0.0:
        return clean
    broad_std = np.sqrt(observation_std**2 + artifact_std**2)
    return (1.0 - contamination) * clean + contamination * _normal_density(residual, broad_std)


def _circular_mean(points: np.ndarray, weights: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.arctan2(
        np.sum(weights * np.sin(points), axis=axis),
        np.sum(weights * np.cos(points), axis=axis),
    )


def _periodic_transition_fft(grid: np.ndarray, drift: float, process_std: float) -> np.ndarray:
    """FFT of the first column of the circulant wrapped-Gaussian transition."""
    delta = wrap_phase(grid - grid[0] - drift)
    kernel = np.exp(-0.5 * (delta / process_std) ** 2)
    kernel /= kernel.sum()
    return np.fft.fft(kernel)


def grid_filter(
    observations: np.ndarray, probe_phases: np.ndarray, *, delay: int, bins: int = 256,
    drift: float = 0.06, process_std: float = 0.08, observation_std: float = 0.18,
    contamination: float = 0.02, artifact_std: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Discrete Bayes reference; accurate here because the latent state is one-dimensional."""
    y, probes = np.asarray(observations), np.asarray(probe_phases)
    grid = np.linspace(-np.pi, np.pi, bins, endpoint=False)
    transition_fft = _periodic_transition_fft(grid, drift, process_std)
    out = np.empty_like(y)
    mass = []
    started = perf_counter_ns()
    for s in range(len(y)):
        probability = np.full(bins, 1.0 / bins)
        for t, value in enumerate(y[s]):
            if t:
                probability = np.fft.ifft(transition_fft * np.fft.fft(probability)).real
                probability = np.maximum(probability, 0.0)
                probability /= probability.sum()
            if np.isfinite(value):
                residual = value - np.cos(grid - probes[s, t])
                probability *= _mixture_likelihood(
                    residual, observation_std, contamination, artifact_std
                )
                total = probability.sum()
                probability = probability / total if total > 0 else np.full(bins, 1.0 / bins)
            mass.append(probability.sum())
            out[s, t] = wrap_phase(_circular_mean(grid, probability) + delay * drift)
    elapsed = perf_counter_ns() - started
    return out, {
        "ns_per_sample": elapsed / y.size,
        "mean_probability_mass": float(np.mean(mass)),
        "bins": bins,
    }


def phase_ekf(
    observations: np.ndarray, probe_phases: np.ndarray, *, delay: int,
    drift: float = 0.06, process_std: float = 0.08, observation_std: float = 0.18,
    innovation_clip: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Wrapped scalar EKF; a deliberately Gaussian comparator to the multimodal filters."""
    y, probes = np.asarray(observations), np.asarray(probe_phases)
    out = np.empty_like(y)
    started = perf_counter_ns()
    for s in range(len(y)):
        mean, variance = 0.0, np.pi**2 / 3.0
        for t, value in enumerate(y[s]):
            if t:
                mean = float(wrap_phase(mean + drift))
                variance += process_std**2
            if np.isfinite(value):
                jacobian = -np.sin(mean - probes[s, t])
                innovation_variance = jacobian**2 * variance + observation_std**2
                innovation = np.clip(
                    value - np.cos(mean - probes[s, t]),
                    -innovation_clip * np.sqrt(innovation_variance),
                    innovation_clip * np.sqrt(innovation_variance),
                )
                gain = variance * jacobian / innovation_variance
                mean = float(wrap_phase(mean + gain * innovation))
                variance = max((1.0 - gain * jacobian) * variance, 1e-8)
            out[s, t] = wrap_phase(mean + delay * drift)
    elapsed = perf_counter_ns() - started
    return out, {"ns_per_sample": elapsed / y.size}


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions)


def phase_particle_filter(
    observations: np.ndarray, probe_phases: np.ndarray, *, seed: int, particles: int,
    delay: int, drift: float = 0.06, process_std: float = 0.08,
    observation_std: float = 0.18, contamination: float = 0.02,
    artifact_std: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Bootstrap phase filter with normalized contaminated observation likelihood."""
    y, probes = np.asarray(observations), np.asarray(probe_phases)
    out = np.empty_like(y)
    ess_values, resamples = [], 0
    started = perf_counter_ns()
    for s in range(len(y)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, s]))
        cloud = rng.uniform(-np.pi, np.pi, particles)
        weights = np.full(particles, 1.0 / particles)
        for t, value in enumerate(y[s]):
            if t:
                cloud = wrap_phase(cloud + drift + rng.normal(0.0, process_std, particles))
            if np.isfinite(value):
                residual = value - np.cos(cloud - probes[s, t])
                weights *= _mixture_likelihood(
                    residual, observation_std, contamination, artifact_std
                )
                total = weights.sum()
                weights = weights / total if total > 1e-300 else np.full(particles, 1.0 / particles)
            ess = 1.0 / np.sum(weights**2)
            ess_values.append(ess)
            if ess < particles / 2.0:
                cloud = cloud[_systematic_resample(weights, rng)]
                weights.fill(1.0 / particles)
                resamples += 1
            out[s, t] = wrap_phase(_circular_mean(cloud, weights) + delay * drift)
    elapsed = perf_counter_ns() - started
    return out, {
        "ns_per_sample": elapsed / y.size,
        "mean_ess_fraction": float(np.mean(ess_values) / particles),
        "resamples_per_sample": resamples / y.size,
    }


def circular_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """Bounded chord loss: zero when phases match and two at antipodes."""
    return float(np.mean(1.0 - np.cos(np.asarray(prediction) - np.asarray(target))))


def phase_metrics(prediction: np.ndarray, states: np.ndarray, *, delay: int) -> dict[str, float]:
    usable = states.shape[1] - delay
    difference = wrap_phase(prediction[:, :usable] - states[:, delay:])
    return {
        "circular_loss": float(np.mean(1.0 - np.cos(difference))),
        "mean_absolute_phase_error": float(np.mean(np.abs(difference))),
        "p95_absolute_phase_error": float(np.quantile(np.abs(difference), 0.95)),
    }
