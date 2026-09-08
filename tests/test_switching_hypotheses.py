import itertools

import numpy as np
import pytest
from scipy.special import expit

from ptwm.surface_mode import log_likelihood_mode_posterior
from ptwm.switching_hypotheses import switching_hypothesis_filter


def test_single_hypothesis_reduces_to_original_filter():
    llr = np.random.default_rng(39).normal(size=(3, 80))
    actual, rate = switching_hypothesis_filter(llr, rates=(.02,))
    expected = log_likelihood_mode_posterior(llr, np.asarray([[.98, .02], [.02, .98]]),
                                            np.asarray([.5, .5]), temporal=True)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(rate, .02)


def test_joint_filter_matches_exhaustive_state_paths():
    rates, eta = np.asarray([.02, .5]), .05
    evidence = np.asarray([[.4, -1.2, .8]])
    actual, _ = switching_hypothesis_filter(evidence, rates=rates, model_switch=eta)
    emission = expit(evidence[0])
    for length in range(1, 4):
        mass = np.zeros(2)
        for path in itertools.product(range(4), repeat=length):
            weight = .25
            for step, state in enumerate(path):
                rate_index, mode = divmod(state, 2)
                if step:
                    prev_rate, prev_mode = divmod(path[step - 1], 2)
                    weight *= ((1 - eta) * (rate_index == prev_rate) + eta / 2)
                    weight *= rates[rate_index] if mode != prev_mode else 1 - rates[rate_index]
                weight *= emission[step] if mode else 1 - emission[step]
            mass[path[-1] % 2] += weight
        assert actual[0, length - 1] == pytest.approx(mass[1] / mass.sum())


def test_rate_filter_is_causal_bounded_and_responds_to_fast_switching():
    llr = np.tile([10., -10.], (2, 200))
    posterior, rate = switching_hypothesis_filter(llr)
    assert np.all((posterior >= 0) & (posterior <= 1))
    assert np.all((rate >= .005) & (rate <= .5))
    assert np.all(rate[:, -1] > .45)
    changed = llr.copy()
    changed[:, 30:] = 0
    other, _ = switching_hypothesis_filter(changed)
    assert np.array_equal(posterior[:, :30], other[:, :30])
