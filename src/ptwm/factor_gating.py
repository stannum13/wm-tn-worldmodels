"""Persistent missing-factor mode and local hypothesis-lift controls."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .parity_factor import BASE_INCIDENCE, CORRELATED_INCIDENCE


BINARY_TRANSITION = np.asarray([[0.995, 0.005], [0.10, 0.90]])
BINARY_MEAN = np.asarray([0.0, 2.0])


def binary_stationary(transition: np.ndarray = BINARY_TRANSITION) -> np.ndarray:
    values, vectors = np.linalg.eig(transition.T)
    result = np.abs(np.real(vectors[:, np.argmin(np.abs(values - 1.0))]))
    return result / result.sum()


def generate_persistent_factor(
    *, seed: int, episodes: int, horizon: int, sigma: float,
    base_probability: float = 0.03, off_probability: float = 1e-4,
    on_probability: float = 0.08,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    modes = np.empty((episodes, horizon), dtype=np.uint8)
    modes[:, 0] = rng.choice(2, size=episodes, p=binary_stationary())
    for time in range(1, horizon):
        uniforms = rng.random(episodes)
        for mode in range(2):
            mask = modes[:, time - 1] == mode
            modes[mask, time] = np.searchsorted(
                np.cumsum(BINARY_TRANSITION[mode]), uniforms[mask], side="right"
            )
    observations = BINARY_MEAN[modes] + sigma * rng.standard_normal(modes.shape)
    base = rng.random(modes.shape + (2,)) < base_probability
    factor_probability = np.where(modes, on_probability, off_probability)
    factor = rng.random(modes.shape) < factor_probability
    syndromes = (base.astype(np.uint8) @ BASE_INCIDENCE) & 1
    syndromes ^= factor[..., None].astype(np.uint8) * CORRELATED_INCIDENCE
    return {
        "observations": observations, "syndromes": syndromes,
        "labels": factor.astype(np.uint8), "modes": modes,
    }


def binary_belief(
    observations: np.ndarray, *, sigma: float, temporal: bool,
    past_syndromes: np.ndarray | None = None, base_probability: float = 0.03,
    off_probability: float = 1e-4, on_probability: float = 0.08,
) -> np.ndarray:
    """Return causal pre-decision P(mode=on) using true model parameters.

    When syndromes are supplied, syndrome ``t`` updates the carried belief only after
    the pre-decision belief at ``t`` is stored. Thus no future or current decision
    outcome leaks backward, while past syndromes inform later mode estimates.
    """
    prior = binary_stationary()
    previous = np.broadcast_to(prior, observations.shape[:1] + (2,)).copy()
    output = np.empty(observations.shape + (2,))
    emission = -0.5 * ((observations[..., None] - BINARY_MEAN) / sigma) ** 2
    log_transition = np.log(BINARY_TRANSITION)
    syndrome_likelihood = None
    syndrome_keys = None
    if past_syndromes is not None:
        if past_syndromes.shape != observations.shape + (4,):
            raise ValueError("past_syndromes must align with observations and have four detectors")
        syndrome_keys = past_syndromes @ (1 << np.arange(4))
        syndrome_likelihood = np.column_stack((
            joint_syndrome_logical(
                base_probability=base_probability, factor_probability=off_probability
            ).sum(axis=1),
            joint_syndrome_logical(
                base_probability=base_probability, factor_probability=on_probability
            ).sum(axis=1),
        ))
    for time in range(observations.shape[1]):
        if temporal:
            log_prior = logsumexp(
                np.log(np.clip(previous, 1e-300, None))[..., :, None]
                + log_transition, axis=-2,
            )
        else:
            log_prior = np.log(prior)
        posterior = log_prior + emission[:, time]
        posterior -= logsumexp(posterior, axis=-1, keepdims=True)
        current = np.exp(posterior)
        output[:, time] = current
        if syndrome_likelihood is None:
            previous = current
        else:
            previous = current * syndrome_likelihood[syndrome_keys[:, time]]
            previous /= previous.sum(axis=-1, keepdims=True)
    return output


def joint_syndrome_logical(
    *, base_probability: float, factor_probability: float
) -> np.ndarray:
    """Exact P(syndrome, logical) for two pair faults plus one logical factor."""
    incidence = np.vstack((BASE_INCIDENCE, CORRELATED_INCIDENCE))
    probabilities = np.asarray([base_probability, base_probability, factor_probability])
    configurations = (
        (np.arange(8)[:, None] >> np.arange(3)) & 1
    ).astype(np.uint8)
    syndrome = (configurations @ incidence) & 1
    keys = syndrome @ (1 << np.arange(4))
    mass = np.prod(
        np.where(configurations.astype(bool), probabilities, 1.0 - probabilities), axis=1
    )
    joint = np.zeros((16, 2))
    for row, key in enumerate(keys):
        joint[key, configurations[row, 2]] += mass[row]
    return joint


def predict_mixture(
    syndromes: np.ndarray, probability_on: np.ndarray, *, base_probability: float,
    off_probability: float, on_probability: float,
) -> np.ndarray:
    """Exact local K=2 hypothesis lift marginalized at the logical endpoint."""
    off = joint_syndrome_logical(
        base_probability=base_probability, factor_probability=off_probability
    )
    on = joint_syndrome_logical(
        base_probability=base_probability, factor_probability=on_probability
    )
    keys = syndromes @ (1 << np.arange(syndromes.shape[-1]))
    mixture = (
        (1.0 - probability_on)[..., None] * off[keys]
        + probability_on[..., None] * on[keys]
    )
    return np.argmax(mixture, axis=-1).astype(np.uint8)


def predict_selected_mode(
    syndromes: np.ndarray, probability_on: np.ndarray, *, base_probability: float,
    off_probability: float, on_probability: float, threshold: float = 0.5,
) -> np.ndarray:
    selected = probability_on >= threshold
    return predict_mixture(
        syndromes, selected.astype(float), base_probability=base_probability,
        off_probability=off_probability, on_probability=on_probability,
    )


def compiled_hysteretic_ema(
    observations: np.ndarray, *, alpha: float, low: float, high: float
) -> np.ndarray:
    """Compile observations to a two-state EMA+hysteresis mode selector."""
    if not low < high:
        raise ValueError("low threshold must be below high threshold")
    state = np.zeros(observations.shape[0])
    active = np.zeros(observations.shape[0], dtype=bool)
    output = np.empty(observations.shape, dtype=float)
    for time in range(observations.shape[1]):
        state = alpha * state + (1.0 - alpha) * observations[:, time]
        active = np.where(active, state >= low, state > high)
        output[:, time] = active
    return output
