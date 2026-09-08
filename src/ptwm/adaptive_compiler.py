"""Tiny switching-noise benchmark for compiling causal decoder adaptation.

The trusted decision layer is exact for a classical repetition-code likelihood
objective. Learned or filtered components only supply bounded bit-error probabilities.
This deliberately small benchmark identifies mechanisms; it is not a hardware model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata


TRANSITION = np.asarray([
    [0.990, 0.010, 0.000, 0.000],
    [0.300, 0.000, 0.700, 0.000],
    [0.000, 0.000, 0.600, 0.400],
    [0.100, 0.000, 0.000, 0.900],
])
MODE_MEAN = np.asarray([0.0, 0.8, 2.5, 1.1])
ERROR_PROBABILITY = np.asarray([0.030, 0.080, 0.300, 0.120])


@dataclass(frozen=True)
class SwitchingData:
    observations: np.ndarray  # episode, time, site
    syndromes: np.ndarray  # episode, time, check
    labels: np.ndarray  # episode, time
    modes: np.ndarray  # episode, time, site


def mode_error_probabilities(distance: int) -> np.ndarray:
    """Return site-by-mode rates with mode-specific spatial signatures."""
    position = np.linspace(-1.0, 1.0, distance)
    factors = np.column_stack((
        np.ones(distance),
        1.0 - 0.55 * position,
        0.65 + 1.10 * (np.arange(distance) % 2),
        1.0 + 0.55 * position,
    ))
    return np.clip(factors * ERROR_PROBABILITY, 1e-4, 0.45)


def stationary_distribution(transition: np.ndarray = TRANSITION) -> np.ndarray:
    """Return the normalized left eigenvector of a stochastic transition matrix."""
    values, vectors = np.linalg.eig(transition.T)
    stationary = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    stationary = np.abs(stationary)
    return stationary / stationary.sum()


def generate_switching_repetition(
    *, seed: int, episodes: int, horizon: int, distance: int = 5,
    emission_sigma: float = 1.25, scope: str = "local",
) -> SwitchingData:
    """Generate paired repetition-code records with local or shared nuisance modes."""
    if scope not in {"local", "shared"}:
        raise ValueError("scope must be 'local' or 'shared'")
    rng = np.random.default_rng(seed)
    stationary = stationary_distribution()
    chains = distance if scope == "local" else 1
    latent = np.empty((episodes, horizon, chains), dtype=np.int8)
    latent[:, 0] = rng.choice(4, size=(episodes, chains), p=stationary)
    for time in range(1, horizon):
        uniforms = rng.random((episodes, chains))
        for mode in range(4):
            selected = latent[:, time - 1] == mode
            latent[:, time][selected] = np.searchsorted(
                np.cumsum(TRANSITION[mode]), uniforms[selected], side="right"
            )
    modes = latent if scope == "local" else np.broadcast_to(
        latent, (episodes, horizon, distance)
    ).copy()
    observations = MODE_MEAN[modes] + emission_sigma * rng.standard_normal(modes.shape)
    probability_table = mode_error_probabilities(distance)
    probabilities = probability_table[np.arange(distance)[None, None, :], modes]
    errors = rng.random(modes.shape) < probabilities
    syndromes = np.logical_xor(errors[..., :-1], errors[..., 1:]).astype(np.uint8)
    # The first error bit identifies which of the two syndrome-consistent homology
    # classes occurred.
    labels = errors[..., 0].astype(np.uint8)
    return SwitchingData(observations, syndromes, labels, modes)


def decode_repetition(syndromes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Exact MAP choice between the two repetition-code-consistent error chains."""
    first = np.zeros(syndromes.shape[:-1] + (syndromes.shape[-1] + 1,), dtype=np.uint8)
    first[..., 1:] = np.bitwise_xor.accumulate(syndromes, axis=-1)
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    loss_zero = -np.sum(
        first * np.log(probabilities) + (1 - first) * np.log1p(-probabilities), axis=-1
    )
    complement = 1 - first
    loss_one = -np.sum(
        complement * np.log(probabilities)
        + (1 - complement) * np.log1p(-probabilities), axis=-1
    )
    return (loss_one < loss_zero).astype(np.uint8)


def _emission_log_likelihood(observations: np.ndarray, sigma: float) -> np.ndarray:
    return -0.5 * ((observations[..., None] - MODE_MEAN) / sigma) ** 2


def causal_mode_filter(
    observations: np.ndarray, *, emission_sigma: float, shared: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact causal four-mode HMM; optionally infer one mode shared by all sites."""
    episodes, horizon, sites = observations.shape
    stationary = stationary_distribution()
    beliefs = np.empty((episodes, horizon, 1 if shared else sites, 4))
    previous = np.broadcast_to(stationary, (episodes, 1 if shared else sites, 4)).copy()
    log_transition = np.log(np.clip(TRANSITION, 1e-300, None))
    local_ll = _emission_log_likelihood(observations, emission_sigma)
    for time in range(horizon):
        predicted = logsumexp(
            np.log(np.clip(previous, 1e-300, None))[..., :, None]
            + log_transition, axis=-2,
        )
        evidence = local_ll[:, time].sum(axis=1, keepdims=True) if shared else local_ll[:, time]
        posterior = predicted + evidence
        posterior -= logsumexp(posterior, axis=-1, keepdims=True)
        previous = np.exp(posterior)
        beliefs[:, time] = previous
    site_beliefs = np.broadcast_to(beliefs, (episodes, horizon, sites, 4)) if shared else beliefs
    probabilities = np.sum(
        site_beliefs * mode_error_probabilities(sites)[None, None, :, :], axis=-1
    )
    return probabilities, site_beliefs


def memoryless_mode_filter(
    observations: np.ndarray, *, emission_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    log_belief = np.log(stationary_distribution()) + _emission_log_likelihood(
        observations, emission_sigma
    )
    log_belief -= logsumexp(log_belief, axis=-1, keepdims=True)
    belief = np.exp(log_belief)
    sites = observations.shape[-1]
    probability = np.sum(
        belief * mode_error_probabilities(sites)[None, None, :, :], axis=-1
    )
    return probability, belief


def compiled_ema_probabilities(observations: np.ndarray, alpha: float) -> np.ndarray:
    """One-state symbolic student: leaky state plus site-specific piecewise LUT."""
    state = np.zeros_like(observations)
    previous = np.zeros(observations.shape[::2])
    # observations.shape[::2] is (episodes, sites) for a 3-D tensor.
    for time in range(observations.shape[1]):
        previous = alpha * previous + (1.0 - alpha) * observations[:, time]
        state[:, time] = previous
    order = np.argsort(MODE_MEAN)
    table = mode_error_probabilities(observations.shape[-1])
    output = np.empty_like(state)
    for site in range(observations.shape[-1]):
        output[..., site] = np.interp(state[..., site], MODE_MEAN[order], table[site, order])
    return output


def select_ema_alpha(
    data: SwitchingData, alphas: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.97, 0.99)
) -> tuple[float, dict[float, float]]:
    scores = {}
    for alpha in alphas:
        prediction = decode_repetition(
            data.syndromes, compiled_ema_probabilities(data.observations, alpha)
        )
        scores[alpha] = float(np.mean(prediction != data.labels))
    return min(scores, key=scores.get), scores


def binary_auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    targets = targets.astype(bool).ravel()
    scores = scores.ravel()
    positives, negatives = targets.sum(), (~targets).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[targets].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def route_predictions(
    cheap: np.ndarray, slow: np.ndarray, score: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    selected = score >= threshold
    return np.where(selected, slow, cheap), selected


def entropy(belief: np.ndarray) -> np.ndarray:
    return -np.sum(belief * np.log(np.clip(belief, 1e-300, None)), axis=(-1, -2))
