import itertools

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from ptwm.exact_dem import bayes_modes, exact_distribution, fault_terms
from ptwm.native_surface import NativeProgram, fwht
from ptwm.surface_program import LocalFeatures, update_probability
from scripts.run_surface_regime_switch import REGIMES, circuit


def enumerate_faults(dem):
    terms = fault_terms(dem)
    out = np.zeros(1 << (dem.num_detectors + dem.num_observables))
    for flags in itertools.product((0, 1), repeat=len(terms)):
        mask, probability = 0, 1.
        for flag, (a, p) in zip(flags, terms):
            mask ^= a if flag else 0
            probability *= p if flag else 1 - p
        out[mask] += probability
    return out


def test_fwht_inverse_and_input_validation():
    x = np.random.default_rng(8).normal(size=1024)
    np.testing.assert_allclose(fwht(fwht(x.copy())) / len(x), x, atol=1e-14)
    with pytest.raises(ValueError):
        fwht(np.ones(6))


@pytest.mark.parametrize("text", [
    "error(0.2) D0 L0",
    "error(0.1) D0\nerror(0.2) D0\nerror(0.03) D1 L0",
    "error(0.1) D0 D1 ^ D1 D2 L0\nerror(0.2) D1\nerror(0.01) D0 L0",
])
def test_exact_distribution_matches_exhaustive_fault_enumeration(text):
    dem = stim.DetectorErrorModel(text)
    distribution, diagnostics = exact_distribution(dem)
    np.testing.assert_allclose(distribution, enumerate_faults(dem), atol=3e-16)
    assert diagnostics["negative_mass"] < 1e-14


def test_separators_are_correlated_not_independent_faults():
    separated = stim.DetectorErrorModel("error(0.1) D0 D1 ^ D1 D2 L0")
    whole = stim.DetectorErrorModel("error(0.1) D0 D2 L0")
    assert fault_terms(separated) == fault_terms(whole)
    np.testing.assert_allclose(exact_distribution(separated)[0], exact_distribution(whole)[0])


def test_dem_teacher_rejects_unhandled_probabilities_and_oversize():
    with pytest.raises(ValueError, match="probabilities"):
        exact_distribution(stim.DetectorErrorModel("error(0.5) D0"))
    with pytest.raises(ValueError, match="budget"):
        exact_distribution(stim.DetectorErrorModel("error(0.1) D28"))


def test_causal_teacher_matches_hidden_sequence_enumeration_without_double_counting():
    tables = np.asarray([[[.50, .15], [.05, .30]], [[.10, .25], [.40, .25]]])
    syndromes = np.asarray([[0, 1, 1, 0]])
    prediction, posterior = bayes_modes(tables, syndromes, temporal=True, switch_probability=.1)
    for stop in range(1, 5):
        joint = np.zeros(2)
        mode_mass = np.zeros(2)
        for path in itertools.product((0, 1), repeat=stop):
            p = .5
            for time in range(stop - 1):
                p *= tables[path[time], :, syndromes[0, time]].sum()
                p *= .9 if path[time] == path[time + 1] else .1
            current = p * tables[path[-1], :, syndromes[0, stop - 1]]
            joint += current
            mode_mass[path[-1]] += current.sum()
        assert prediction[0, stop - 1] == (joint[1] > joint[0])
        assert posterior[0, stop - 1] == pytest.approx(mode_mass[1] / mode_mass.sum())


def test_native_feature_and_state_program_matches_reference():
    c = circuit(3, *REGIMES[0])
    d = np.ascontiguousarray(c.compile_detector_sampler(seed=903).sample(1024), dtype=np.uint8)
    mapper = LocalFeatures.from_circuit(c)
    width = mapper.transform(d[:1]).shape[1]
    rng = np.random.default_rng(913)
    model = {"mean": rng.normal(size=width), "scale": rng.uniform(.5, 1, width),
             "weights": rng.normal(size=width), "bias": .2}
    compiled = mapper.compile(model)
    native = NativeProgram(compiled, threshold=.7)
    reference_p = .5
    for row in d:
        llr = compiled.score(row)
        assert native.score_address(row.ctypes.data) == pytest.approx(llr, abs=1e-12)
        reference_p = update_probability(llr, reference_p)
        assert native.choose_address(row.ctypes.data) == (reference_p >= .7)
        assert native.state.probability == pytest.approx(reference_p, abs=1e-12)
