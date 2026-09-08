import numpy as np
import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")

from ptwm.native_time_templates import NativeTimeProgram
from ptwm.surface_program import LocalFeatures
from ptwm.time_templates import choose_action, difficulty_bins, mode_logits, mode_rate_filter
from scripts.run_surface_time_templates import circuits


@pytest.mark.parametrize("rule", ["mode", "global", "binned"])
def test_native_logits_probabilities_and_actions_match_reference(rule):
    c = circuits()[0]
    feature_map = LocalFeatures.from_circuit(c)
    d = c.compile_detector_sampler(seed=681).sample(256)
    features = feature_map.transform(d)
    rng = np.random.default_rng(682)
    model = {"mean": rng.normal(size=features.shape[1]), "scale": rng.uniform(.5, 2, features.shape[1]),
             "weights": rng.normal(scale=.1, size=(features.shape[1], 4)), "bias": rng.normal(size=4)}
    risk = {"global": rng.uniform(0, .1, (4, 4)), "binned": rng.uniform(0, .1, (4, 4, 4))}
    expected_logits = mode_logits(features, model)
    probability, _ = mode_rate_filter(expected_logits.reshape(2, 128, 4))
    expected = choose_action(probability.reshape(-1, 4), risk, difficulty_bins(d), rule)
    native = NativeTimeProgram(feature_map, model, risk, rule=rule)
    for i, row in enumerate(d):
        if i % 128 == 0:
            native.reset()
        assert native.choose_address(row.ctypes.data) == expected[i]
        np.testing.assert_allclose(list(native.state.logits), expected_logits[i], atol=1e-12)
        np.testing.assert_allclose(list(native.state.posterior), probability.reshape(-1, 4)[i], atol=1e-12)
    assert native.persistent_bytes == 160
    assert native.constant_bytes > 0


def test_native_rejects_unknown_rule_and_invalid_shapes():
    c = circuits()[0]
    features = LocalFeatures.from_circuit(c)
    n = features.transform(c.compile_detector_sampler(seed=683).sample(1)).shape[1]
    model = {"mean": np.zeros(n), "scale": np.ones(n), "weights": np.zeros((n, 4)), "bias": np.zeros(4)}
    tables = {"global": np.zeros((4, 4)), "binned": np.zeros((4, 4, 4))}
    with pytest.raises(ValueError, match="rule"):
        NativeTimeProgram(features, model, tables, rule="neural")
    model["weights"] = np.zeros((n, 3))
    with pytest.raises(ValueError, match="four"):
        NativeTimeProgram(features, model, tables, rule="mode")


def test_timing_order_is_balanced_in_every_fit():
    from scripts.benchmark_surface_time_templates import balanced_order

    names = ["static", "native_pipeline", "native_frontend"]
    for replicate in range(10):
        orders = [balanced_order(names, replicate, repeat) for repeat in range(3)]
        for column in zip(*orders):
            assert set(column) == set(names)
        for row in orders:
            assert set(row) == set(names)
