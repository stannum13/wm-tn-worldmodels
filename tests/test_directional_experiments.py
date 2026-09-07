import numpy as np

from ptwm.directional import (
    fit_linear_predictor,
    generate_memory_process,
    generate_qubit_process,
    rollout_linear,
    rollout_bloch_constrained,
    trace_distance_bloch,
)


def test_memory_feature_wins_on_held_out_control_sequence():
    train = generate_memory_process(seed=0, n_sequences=64, horizon=24)
    test = generate_memory_process(seed=1, n_sequences=32, horizon=60)
    markov = fit_linear_predictor(train, memory=0)
    memory = fit_linear_predictor(train, memory=1)
    markov_err = rollout_linear(markov, test, horizon=40)
    memory_err = rollout_linear(memory, test, horizon=40)
    assert memory_err < 0.75 * markov_err


def test_memory_feature_does_not_create_a_gain_without_memory():
    train = generate_memory_process(seed=2, n_sequences=64, horizon=24, memory_gain=0.0)
    test = generate_memory_process(seed=3, n_sequences=32, horizon=60, memory_gain=0.0)
    markov = fit_linear_predictor(train, memory=0)
    memory = fit_linear_predictor(train, memory=1)
    markov_err = rollout_linear(markov, test, horizon=40, memory=0)
    memory_err = rollout_linear(memory, test, horizon=40, memory=1)
    assert memory_err < 1.25 * markov_err


def test_constrained_qubit_rollout_stays_physical_and_is_directionally_accurate():
    train = generate_qubit_process(seed=0, n_sequences=80, horizon=16)
    test = generate_qubit_process(seed=1, n_sequences=40, horizon=40, control_scale=4.0)
    unconstrained = fit_linear_predictor(train, memory=0)
    constrained = rollout_bloch_constrained(unconstrained, test, horizon=30)
    assert constrained["min_eigenvalue"] >= -1e-10
    unconstrained_rollout = rollout_linear(unconstrained, test, horizon=30, return_states=True)
    constrained_err = constrained["trace_distance"]
    unconstrained_err = trace_distance_bloch(
        unconstrained_rollout["predictions"], unconstrained_rollout["targets"]
    )
    assert constrained_err < 1.5 * unconstrained_err
