import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from ptwm.surface_mode import (
    affine_log_likelihood_ratio, fit_affine_log_likelihood_ratio,
    log_likelihood_mode_posterior,
)
from ptwm.surface_program import LocalFeatures, select_before_decode, update_probability
from scripts.run_surface_frontier_challenge import (
    CONDITIONS, Seeds, circuit, make_decoder, mixed_circuit, pooled_dem, seed_value, stream,
)
from scripts.run_surface_regime_switch import (
    REGIMES, STATIONARY, TRANSITION, detailed_morphology_features,
)


def test_all_planned_seeds_unique_and_role_reuse_rejected():
    values = []
    for replicate in range(10):
        for distance in (3, 5):
            seeds = Seeds(2026090801, replicate, distance)
            for mode in range(2):
                values.append(seeds.draw("calibration", "stim", mode))
            for role, conditions in (("action", ["nominal"]), ("selection", ["nominal"]),
                                      ("test", CONDITIONS)):
                for condition in conditions:
                    values.append(seeds.draw(role, condition, "modes"))
                    for mode in range(2):
                        values.append(seeds.draw(role, condition, "stim", mode))
            with pytest.raises(ValueError, match="reused"):
                seeds.draw("calibration", "stim", 0)
    assert len(values) == len(set(values)) == 760
    assert seed_value(10, 2, 3, "test") == seed_value(10, 2, 3, "test")


@pytest.mark.parametrize("distance", [3, 5])
def test_compiled_evidence_matches_full_feature_program(distance):
    c = circuit(distance, *REGIMES[0])
    d = c.compile_detector_sampler(seed=902).sample(256)
    local = LocalFeatures.from_circuit(c)
    features = local.transform(d)
    assert np.array_equal(features, detailed_morphology_features(d, c))
    rng = np.random.default_rng(904)
    model = {"mean": rng.normal(size=features.shape[1]), "scale": rng.uniform(.5, 2, features.shape[1]),
             "weights": rng.normal(size=features.shape[1]), "bias": 0.17}
    program = local.compile(model)
    np.testing.assert_allclose(program.score(d), affine_log_likelihood_ratio(features, model), atol=1e-12)


def test_scalar_posterior_causal_and_equivalent_to_vector_reference():
    evidence = np.random.default_rng(905).normal(size=(2, 100))
    expected = log_likelihood_mode_posterior(evidence, TRANSITION, STATIONARY, temporal=True)
    actual = np.empty_like(expected)
    previous = np.full(2, .5)
    for step in range(100):
        previous = update_probability(evidence[:, step], previous)
        actual[:, step] = previous
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    changed = evidence.copy()
    changed[:, 50:] += 20
    other = log_likelihood_mode_posterior(changed, TRANSITION, STATIONARY, temporal=True)
    assert np.array_equal(other[:, :50], expected[:, :50])


def test_pooled_dem_preserves_endpoint_and_mechanism_probabilities():
    a, b = [circuit(3, *rates).detector_error_model(decompose_errors=True) for rates in REGIMES]
    assert pooled_dem(a, b, 0) == a.flattened()
    assert pooled_dem(a, b, 1) == b.flattened()
    for x, y, mixed in zip(a.flattened(), b.flattened(), pooled_dem(a, b, .25)):
        assert mixed.targets_copy() == x.targets_copy() == y.targets_copy()
        if x.type == "error":
            assert mixed.args_copy()[0] == pytest.approx(.75 * x.args_copy()[0] + .25 * y.args_copy()[0])
    with pytest.raises(ValueError):
        pooled_dem(a, b, 1.1)


def test_midpoint_changes_only_noise_and_generates_valid_detectors():
    a, b = [circuit(3, *rates) for rates in REGIMES]
    mixed = mixed_circuit(a, b)
    assert mixed.without_noise() == a.flattened().without_noise()
    assert mixed != a.flattened() and mixed != b.flattened()
    assert mixed.get_detector_coordinates() == a.get_detector_coordinates()
    assert mixed.detector_error_model(decompose_errors=True).num_observables == 1


@pytest.mark.parametrize("correlated", [False, True])
def test_selective_decoder_and_energy_consistency(correlated):
    cs = [circuit(3, *rates) for rates in REGIMES]
    decoders = [make_decoder(str(i), c.detector_error_model(decompose_errors=True), correlated, {})
                for i, c in enumerate(cs)]
    d = cs[1].compile_detector_sampler(seed=907).sample(2048)
    chosen = np.random.default_rng(908).integers(2, size=len(d))
    pair = [decoder.decode(d, weights=True) for decoder in decoders]
    for decoder, (prediction, weight) in zip(decoders, pair):
        assert np.array_equal(prediction, decoder.decode(d))
        assert np.all(np.isfinite(weight))
    expected = np.where(chosen, pair[1][0], pair[0][0])
    actual = select_before_decode(d, chosen, [x.matcher for x in decoders], correlated=correlated)
    assert np.array_equal(expected, actual)


def test_correlated_path_is_exercised_and_changes_decisions():
    c = circuit(3, .01, .001)
    dem = c.detector_error_model(decompose_errors=True)
    d = c.compile_detector_sampler(seed=908).sample(8192)
    plain = make_decoder("plain", dem, False, {}).decode(d)
    correlated = make_decoder("correlated", dem, True, {}).decode(d)
    assert np.any(plain != correlated)


def test_generator_stationarity_and_frozen_modes():
    _, _, iid = stream(3, seeds=Seeds(909, 0, 3), role="test", condition="iid", streams=64, horizon=512)
    assert abs(iid.mean() - .5) < .02
    assert abs((iid[:, 1:] != iid[:, :-1]).mean() - .5) < .02
    _, _, fixed = stream(3, seeds=Seeds(909, 0, 3), role="test", condition="stationary_a", streams=2, horizon=10)
    assert not np.any(fixed)
