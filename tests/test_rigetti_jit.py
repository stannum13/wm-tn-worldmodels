import numpy as np
import pytest

numba = pytest.importorskip("numba")
pymatching = pytest.importorskip("pymatching")

from ptwm.rigetti import compile_frontier_decoder, PackedAffineIQHeads, PackedSoftReweighting
from ptwm.rigetti_jit import (
    decode_frontier_batch,
    decode_frontier_one,
    fused_affine_quantized_weights_one,
    fused_affine_reweights_one,
    pack_frontier_schedule,
)


def test_compiled_frontier_matches_python_and_pymatching():
    matching = pymatching.Matching()
    matching.add_edge(0, 1, fault_ids={0}, weight=1.13)
    matching.add_edge(1, 2, weight=2.07)
    matching.add_boundary_edge(0, weight=3.19)
    matching.add_boundary_edge(2, fault_ids={0}, weight=0.71)
    plan = compile_frontier_decoder(
        matching, {node: [0.0, 0.0, float(node)] for node in range(3)}
    )
    schedule = pack_frontier_schedule(plan)
    weights = np.asarray([attributes["weight"] for _, _, attributes in matching.edges()])
    syndromes = np.asarray([
        [(value >> bit) & 1 for bit in range(3)] for value in range(8)
    ], dtype=np.uint8)
    compiled, margins = decode_frontier_batch(
        syndromes, np.broadcast_to(weights, (len(syndromes), len(weights))), *schedule
    )
    for row, syndrome in enumerate(syndromes):
        expected = int(matching.decode(syndrome)[0])
        python, margin = plan.decode(syndrome, weights)
        one, compiled_margin = decode_frontier_one(syndrome, weights, *schedule)
        assert compiled[row] == one == python == expected
        assert margins[row] == pytest.approx(compiled_margin)
        assert compiled_margin == pytest.approx(margin)


def test_fused_affine_reweights_match_composed_packed_path():
    heads = PackedAffineIQHeads(
        weights=np.asarray([[0.2, -0.4, 0.7], [-0.3, 0.8, 0.1]]),
        feature_min=np.asarray([[-1.0, -2.0], [0.0, -1.0]]),
        feature_range=np.asarray([[2.0, 4.0], [3.0, 2.0]]),
    )
    reweighting = PackedSoftReweighting(
        endpoints=np.asarray([[0.0, 1.0], [1.0, -1.0]]),
        residual_factor=np.asarray([0.8, 0.6]),
        measurement_indices=np.asarray([[0, 1], [1, 0]]),
        measurement_mask=np.asarray([[True, True], [True, False]]),
        floor_probability=1e-5,
        ceiling_probability=0.49,
    )
    soft = np.asarray([0.4 - 0.7j, 1.2 + 0.3j])
    hard = np.asarray([False, True])
    probability = heads.predict(soft)
    expected = reweighting.build(np.where(hard, 1.0 - probability, probability))
    observed = fused_affine_reweights_one(
        soft, hard, heads.weights, heads.feature_min, heads.feature_range,
        reweighting.endpoints, reweighting.residual_factor,
        reweighting.measurement_indices, reweighting.measurement_mask,
        reweighting.floor_probability, reweighting.ceiling_probability,
    )
    assert np.allclose(observed, expected, atol=2e-7)

    floating_weights = fused_affine_quantized_weights_one(
        soft, hard, heads.weights, heads.feature_min, heads.feature_range,
        reweighting.residual_factor, reweighting.measurement_indices,
        reweighting.measurement_mask, reweighting.floor_probability,
        reweighting.ceiling_probability, 0,
    )
    assert np.allclose(floating_weights, observed[:, 2], atol=2e-7)


def test_probability_quantization_is_bounded_and_converges():
    heads = PackedAffineIQHeads(
        weights=np.asarray([[0.2, -0.4, 0.7], [-0.3, 0.8, 0.1]]),
        feature_min=np.asarray([[-1.0, -2.0], [0.0, -1.0]]),
        feature_range=np.asarray([[2.0, 4.0], [3.0, 2.0]]),
    )
    reweighting = PackedSoftReweighting(
        endpoints=np.asarray([[0.0, 1.0], [1.0, -1.0]]),
        residual_factor=np.asarray([0.8, 0.6]),
        measurement_indices=np.asarray([[0, 1], [1, 0]]),
        measurement_mask=np.asarray([[True, True], [True, False]]),
        floor_probability=1e-5,
        ceiling_probability=0.49,
    )
    args = (
        np.asarray([0.4 - 0.7j, 1.2 + 0.3j]), np.asarray([False, True]),
        heads.weights, heads.feature_min, heads.feature_range,
        reweighting.residual_factor, reweighting.measurement_indices,
        reweighting.measurement_mask, reweighting.floor_probability,
        reweighting.ceiling_probability,
    )
    floating = fused_affine_quantized_weights_one(*args, 0)
    coarse = fused_affine_quantized_weights_one(*args, 2)
    fine = fused_affine_quantized_weights_one(*args, 8)
    assert np.all(np.isfinite(coarse))
    assert np.all(coarse >= 0)
    assert np.max(np.abs(fine - floating)) < np.max(np.abs(coarse - floating))


def test_probability_quantization_rejects_unsupported_bit_widths():
    args = (
        np.asarray([0.0 + 0.0j]), np.asarray([False]),
        np.zeros((1, 3)), np.zeros((1, 2)), np.ones((1, 2)),
        np.asarray([1.0]), np.asarray([[0]]), np.asarray([[True]]),
        1e-5, 0.49,
    )
    with pytest.raises(ValueError, match="probability_bits"):
        fused_affine_quantized_weights_one(*args, -1)
    with pytest.raises(ValueError, match="probability_bits"):
        fused_affine_quantized_weights_one(*args, 17)


def test_probability_quantization_contract_on_known_grid_and_complement():
    # Scores logit(1/6), logit(1/2), and logit(5/6) give error probabilities
    # 1/6, 1/2, and (after hard-bit complementation) 1/6. For two bits the
    # [0, 0.5] grid is {0, 1/6, 1/3, 1/2}; the ceiling clips 1/2 to 0.49.
    scores = np.log(np.asarray([1 / 5, 1.0, 5.0]))
    args = (
        np.zeros(3, dtype=np.complex128), np.asarray([False, False, True]),
        np.column_stack((scores, np.zeros((3, 2)))),
        np.zeros((3, 2)), np.ones((3, 2)), np.ones(3),
        np.arange(3)[:, None], np.ones((3, 1), dtype=np.bool_), 1e-5, 0.49,
    )
    weights = fused_affine_quantized_weights_one(*args, 2)
    expected_probability = np.asarray([1 / 6, 0.49, 1 / 6])
    expected = np.log((1.0 - expected_probability) / expected_probability)
    assert np.allclose(weights, expected, atol=2e-7)
