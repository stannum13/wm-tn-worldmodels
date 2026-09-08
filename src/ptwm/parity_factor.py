"""Exact tiny parity-factor controls for graph-overlay experiments."""

from __future__ import annotations

import numpy as np


BASE_INCIDENCE = np.asarray([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
], dtype=np.uint8)
BASE_LOGICAL = np.asarray([0, 0], dtype=np.uint8)
CORRELATED_INCIDENCE = np.asarray([1, 1, 1, 1], dtype=np.uint8)
WRONG_INCIDENCE = np.asarray([1, 1, 0, 0], dtype=np.uint8)


def compile_parity_factor_decoder(
    incidence: np.ndarray, logicals: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Compile exact minimum-weight-error predictions for every detector syndrome."""
    incidence = np.asarray(incidence, dtype=np.uint8)
    logicals = np.asarray(logicals, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=float)
    if incidence.ndim != 2 or len(incidence) != len(logicals) or len(logicals) != len(probabilities):
        raise ValueError("fault incidence, logicals, and probabilities must align")
    if np.any((probabilities <= 0) | (probabilities >= 0.5)):
        raise ValueError("fault probabilities must lie in (0, 0.5)")
    faults = len(probabilities)
    configurations = (
        (np.arange(1 << faults)[:, None] >> np.arange(faults)) & 1
    ).astype(np.uint8)
    syndromes = (configurations @ incidence) & 1
    outcomes = (configurations @ logicals) & 1
    weights = np.log((1.0 - probabilities) / probabilities)
    costs = configurations @ weights
    powers = 1 << np.arange(incidence.shape[1])
    keys = syndromes @ powers
    table = np.full(1 << incidence.shape[1], 255, dtype=np.uint8)
    for key in np.unique(keys):
        candidates = np.flatnonzero(keys == key)
        table[key] = outcomes[candidates[np.argmin(costs[candidates])]]
    return table


def decode_parity_factor(table: np.ndarray, syndromes: np.ndarray) -> np.ndarray:
    powers = 1 << np.arange(syndromes.shape[-1])
    prediction = table[np.asarray(syndromes, dtype=np.uint8) @ powers]
    if np.any(prediction == 255):
        raise ValueError("decoder model cannot explain a supplied syndrome")
    return prediction


def generate_factor_records(
    *, seed: int, episodes: int, horizon: int, base_probability: float = 0.03,
    factor_probability: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample two independent pair faults and one four-detector logical factor."""
    rng = np.random.default_rng(seed)
    shape = (episodes, horizon)
    base = rng.random(shape + (2,)) < base_probability
    factor = rng.random(shape) < factor_probability
    syndromes = (base.astype(np.uint8) @ BASE_INCIDENCE) & 1
    syndromes ^= factor[..., None].astype(np.uint8) * CORRELATED_INCIDENCE
    labels = factor.astype(np.uint8)
    return syndromes, labels


def decoder_tables(
    *, base_probability: float = 0.03, factor_probability: float = 0.04
) -> dict[str, np.ndarray]:
    base = compile_parity_factor_decoder(
        BASE_INCIDENCE, BASE_LOGICAL, np.full(2, base_probability)
    )
    correct = compile_parity_factor_decoder(
        np.vstack((BASE_INCIDENCE, CORRELATED_INCIDENCE)),
        np.r_[BASE_LOGICAL, 1],
        np.r_[np.full(2, base_probability), factor_probability],
    )
    wrong = compile_parity_factor_decoder(
        np.vstack((BASE_INCIDENCE, WRONG_INCIDENCE)),
        np.r_[BASE_LOGICAL, 0],
        np.r_[np.full(2, base_probability), factor_probability],
    )
    return {"base": base, "insert_correct_factor": correct, "insert_wrong_factor": wrong}
