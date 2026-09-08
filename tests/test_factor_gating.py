import numpy as np
import pytest

from ptwm.factor_gating import (
    BINARY_MEAN,
    BINARY_TRANSITION,
    binary_stationary,
    binary_belief,
    compiled_hysteretic_ema,
    generate_persistent_factor,
    joint_syndrome_logical,
    predict_mixture,
)
from scripts.run_factor_parameter_sweep import paired_episode_interval
from scripts.run_factor_deployment_shift import estimate_rates


def test_paired_episode_interval_collapses_for_identical_predictions() -> None:
    labels = np.array([[False, True], [True, False]])
    prediction = np.array([[False, False], [True, True]])
    assert paired_episode_interval(prediction, prediction, labels) == [0.0, 0.0]


def test_rate_estimator_recovers_grid_cell_from_heldout_stream() -> None:
    data = generate_persistent_factor(
        seed=77, episodes=128, horizon=256, sigma=0.5,
        base_probability=0.06, on_probability=0.16,
    )
    assert estimate_rates(data, 0.5) == (0.06, 0.16)


def test_factor_belief_is_causal():
    data = generate_persistent_factor(seed=1, episodes=4, horizon=40, sigma=1.0)
    changed = data["observations"].copy()
    changed[:, 21:] += 100
    original = binary_belief(
        data["observations"], sigma=1.0, temporal=True,
        past_syndromes=data["syndromes"],
    )
    altered = binary_belief(
        changed, sigma=1.0, temporal=True, past_syndromes=data["syndromes"],
    )
    assert np.array_equal(original[:, :21], altered[:, :21])


def test_hypothesis_lift_uses_syndrome_evidence():
    syndrome = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    off = predict_mixture(
        syndrome, np.asarray([0.0]), base_probability=0.03,
        off_probability=1e-4, on_probability=0.08,
    )
    on = predict_mixture(
        syndrome, np.asarray([1.0]), base_probability=0.03,
        off_probability=1e-4, on_probability=0.08,
    )
    assert off[0] == 0 and on[0] == 1


def test_hysteresis_retains_mode_between_thresholds():
    observations = np.asarray([[0.0, 3.0, 1.0, 0.0]])
    selected = compiled_hysteretic_ema(observations, alpha=0.0, low=0.5, high=2.0)
    assert np.array_equal(selected, [[0.0, 1.0, 1.0, 0.0]])


def test_filter_matches_bruteforce_mode_sequence_posterior():
    observations = np.asarray([[0.2, 1.4, -0.1]])
    syndromes = np.asarray([[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]]], dtype=np.uint8)
    sigma = 0.9
    observed = binary_belief(
        observations, sigma=sigma, temporal=True, past_syndromes=syndromes
    )[0]
    joints = [
        joint_syndrome_logical(base_probability=0.03, factor_probability=p).sum(axis=1)
        for p in (1e-4, 0.08)
    ]
    keys = syndromes[0] @ (1 << np.arange(4))
    for time in range(3):
        mass = np.zeros(2)
        for encoded in range(1 << (time + 1)):
            modes = (encoded >> np.arange(time + 1)) & 1
            probability = binary_stationary()[modes[0]]
            for step in range(time + 1):
                probability *= np.exp(-0.5 * ((observations[0, step] - BINARY_MEAN[modes[step]]) / sigma) ** 2)
                if step < time:
                    probability *= joints[modes[step]][keys[step]]
                    probability *= BINARY_TRANSITION[modes[step], modes[step + 1]]
            mass[modes[-1]] += probability
        mass /= mass.sum()
        assert np.allclose(observed[time], mass)


def test_joint_factor_models_are_normalized():
    for probability in (1e-4, 0.08):
        joint = joint_syndrome_logical(
            base_probability=0.03, factor_probability=probability
        )
        assert np.sum(joint) == pytest.approx(1.0)
