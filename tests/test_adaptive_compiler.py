import numpy as np

from ptwm.adaptive_compiler import (
    ERROR_PROBABILITY,
    binary_auroc,
    causal_mode_filter,
    compiled_ema_probabilities,
    decode_repetition,
    generate_switching_repetition,
    mode_error_probabilities,
    stationary_distribution,
)


def test_repetition_decoder_recovers_low_weight_consistent_chain():
    errors = np.asarray([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=np.uint8)
    syndrome = np.logical_xor(errors[:, :-1], errors[:, 1:]).astype(np.uint8)
    probabilities = np.full(errors.shape, 0.05)
    assert np.array_equal(decode_repetition(syndrome, probabilities), [0, 1])


def test_switching_generator_is_reproducible_and_filter_is_causal_shape():
    first = generate_switching_repetition(seed=3, episodes=4, horizon=20)
    second = generate_switching_repetition(seed=3, episodes=4, horizon=20)
    assert np.array_equal(first.observations, second.observations)
    probabilities, beliefs = causal_mode_filter(first.observations, emission_sigma=1.25)
    assert probabilities.shape == first.observations.shape
    assert beliefs.shape == first.observations.shape + (4,)
    assert np.allclose(beliefs.sum(axis=-1), 1.0)
    table = mode_error_probabilities(first.observations.shape[-1])
    assert np.all((probabilities >= table.min()) & (probabilities <= table.max()))


def test_stationary_distribution_and_compiled_ema_contract():
    stationary = stationary_distribution()
    assert np.allclose(stationary @ __import__("ptwm.adaptive_compiler", fromlist=["TRANSITION"]).TRANSITION, stationary)
    data = generate_switching_repetition(seed=4, episodes=3, horizon=12)
    probabilities = compiled_ema_probabilities(data.observations, 0.9)
    assert probabilities.shape == data.observations.shape
    assert np.all(np.isfinite(probabilities))


def test_binary_auroc_known_ordering():
    assert binary_auroc(np.asarray([0.1, 0.2, 0.8, 0.9]), np.asarray([0, 0, 1, 1])) == 1.0


def test_causal_filter_is_invariant_to_future_observations():
    data = generate_switching_repetition(seed=8, episodes=3, horizon=30)
    altered = data.observations.copy()
    altered[:, 16:] += 100.0
    _, original_belief = causal_mode_filter(data.observations, emission_sigma=1.25)
    _, altered_belief = causal_mode_filter(altered, emission_sigma=1.25)
    assert np.array_equal(original_belief[:, :16], altered_belief[:, :16])
