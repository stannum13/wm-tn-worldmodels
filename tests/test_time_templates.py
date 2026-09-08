import itertools

import numpy as np
import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")

from ptwm.time_templates import (
    fit_mode_logits, mode_logits, mode_rate_filter, splice_noise, choose_action,
)
from scripts.run_surface_frontier_challenge import circuit, mixed_circuit, REGIMES


def test_splice_preserves_ideal_circuit_and_old_midpoint():
    a, b = [circuit(3, *r) for r in REGIMES]
    assert splice_noise(a, b, .5) == mixed_circuit(a, b)
    for fraction in (1 / 3, .5, 2 / 3):
        c = splice_noise(a, b, fraction)
        assert c.without_noise() == a.flattened().without_noise()
        assert c.get_detector_coordinates() == a.get_detector_coordinates()
        assert c.detector_error_model(decompose_errors=True).num_observables == 1
    with pytest.raises(ValueError):
        splice_noise(a, b, 0)


def test_multiclass_fit_uses_balanced_mode_labels():
    rng = np.random.default_rng(511)
    groups = [rng.normal(size=(100, 4)) + np.eye(4)[m] * 4 for m in range(4)]
    model = fit_mode_logits(groups)
    logits = mode_logits(np.vstack(groups), model)
    assert np.mean(logits.argmax(axis=1) == np.repeat(np.arange(4), 100)) > .95
    with pytest.raises(ValueError):
        fit_mode_logits([groups[0], groups[1][:90]])


def test_four_mode_filter_matches_enumerated_joint_kernel():
    rates = np.array([.02, .75])
    refresh = .1
    logits = np.random.default_rng(512).normal(size=(1, 4, 4))
    actual, _ = mode_rate_filter(logits, rates=rates, refresh=refresh)
    joint = np.ones((2, 4)) / 8
    expected = []
    for x in logits[0]:
        prior = np.zeros_like(joint)
        for old_r, old_m, new_r, new_m in itertools.product(range(2), range(4), range(2), range(4)):
            rp = refresh / 2 + (1 - refresh) * (old_r == new_r)
            mp = (1 - rates[new_r]) if old_m == new_m else rates[new_r] / 3
            prior[new_r, new_m] += joint[old_r, old_m] * rp * mp
        joint = prior * np.exp(x - x.max())
        joint /= joint.sum()
        expected.append(joint.sum(axis=0))
    np.testing.assert_allclose(actual[0], expected, atol=1e-14)


def test_independent_limit_and_no_future_access():
    logits = np.random.default_rng(513).normal(size=(2, 20, 4))
    from scipy.special import softmax

    independent, _ = mode_rate_filter(logits, rates=(.75,))
    np.testing.assert_allclose(independent, softmax(logits, axis=2), atol=1e-14)
    baseline, _ = mode_rate_filter(logits)
    changed = logits.copy()
    changed[:, 10:, 0] += 20
    other, _ = mode_rate_filter(changed)
    np.testing.assert_array_equal(baseline[:, :10], other[:, :10])
    assert np.allclose(baseline.sum(axis=2), 1)


def test_restricted_actions_and_bin_conditioning():
    p = np.array([[0., 1., 0., 0.], [.8, .1, .05, .05]])
    risk = np.ones((4, 4)) - np.eye(4)
    tables = {"global": risk, "binned": np.repeat(risk[:, :, None], 4, axis=2)}
    for rule in ("mode", "global", "binned"):
        assert choose_action(p, tables, np.array([0, 1]), rule, (0, 1, 2, 3)).tolist() == [1, 0]
        assert set(choose_action(p, tables, np.array([0, 1]), rule, (0, 3))) <= {0, 3}
    tables["binned"][1, :, 0] = [0, 1, 1, 1]
    assert choose_action(p, tables, np.array([0, 1]), "binned", (0, 1, 2, 3)).tolist() == [0, 0]


def test_nominal_template_dem_support_is_identical():
    from scripts.run_surface_time_templates import circuits, validate_template_support

    assert len(validate_template_support(circuits())) == 64


def test_multiclass_data_hash_distinguishes_all_mode_values():
    from scripts.run_surface_time_templates import digest_data

    assert len({digest_data(np.array([m], dtype=np.uint8)) for m in range(4)}) == 4


def test_counterfactual_choice_is_exact_and_does_not_take_logical_labels():
    from scripts.run_surface_time_templates import circuits, policy_choices
    from scripts.run_surface_frontier_challenge import make_decoder
    from ptwm.time_templates import fit_risk_tables, difficulty_bins

    cs = circuits()
    graphs = [make_decoder(str(i), c.detector_error_model(decompose_errors=True), True, {}) for i, c in enumerate(cs)]
    d = cs[1].compile_detector_sampler(seed=514).sample(512)
    rng = np.random.default_rng(515)
    probabilities = {"bank": rng.dirichlet(np.ones(4), size=len(d))}
    tables = fit_risk_tables([rng.random((64, 4)) < .1 for _ in range(4)],
                            [rng.integers(4, size=64) for _ in range(4)])
    decoded = np.column_stack([g.decode(d) for g in graphs])
    for choice in policy_choices(probabilities, tables, difficulty_bins(d)).values():
        expected = decoded[np.arange(len(d)), choice]
        selective = np.empty(len(d), dtype=bool)
        for m, graph in enumerate(graphs):
            where = np.flatnonzero(choice == m)
            if len(where):
                selective[where] = graph.decode(d[where])
        np.testing.assert_array_equal(selective, expected)
