"""Optional Numba kernels for the exact temporal-frontier decoder experiment."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def fused_affine_reweights_one(
    soft_measurements: np.ndarray,
    hard_measurements: np.ndarray,
    head_weights: np.ndarray,
    feature_min: np.ndarray,
    feature_range: np.ndarray,
    endpoints: np.ndarray,
    residual_factor: np.ndarray,
    measurement_indices: np.ndarray,
    measurement_mask: np.ndarray,
    floor_probability: float,
    ceiling_probability: float,
) -> np.ndarray:
    """Fuse affine I/Q inference, surprisal conversion, and dynamic edge LLRs."""
    measurement_count = len(soft_measurements)
    error_probability = np.empty(measurement_count, dtype=np.float64)
    for measurement in range(measurement_count):
        real = (soft_measurements[measurement].real - feature_min[measurement, 0])
        real /= feature_range[measurement, 0]
        imag = (soft_measurements[measurement].imag - feature_min[measurement, 1])
        imag /= feature_range[measurement, 1]
        real = min(1.0, max(0.0, real))
        imag = min(1.0, max(0.0, imag))
        score = (
            head_weights[measurement, 0]
            + real * head_weights[measurement, 1]
            + imag * head_weights[measurement, 2]
        )
        # Match PackedAffineIQHeads.predict's explicit float32 probability contract.
        probability = np.float32(1.0 / (1.0 + np.exp(-score)))
        error_probability[measurement] = (
            1.0 - probability if hard_measurements[measurement] else probability
        )
    output = np.empty((len(endpoints), 3), dtype=np.float64)
    for edge in range(len(endpoints)):
        alpha = residual_factor[edge]
        for position in range(measurement_indices.shape[1]):
            if measurement_mask[edge, position]:
                probability = error_probability[measurement_indices[edge, position]]
                probability = min(ceiling_probability, max(floor_probability, probability))
                alpha *= 1.0 - 2.0 * probability
        probability = (1.0 - alpha) / 2.0
        probability = min(ceiling_probability, max(floor_probability, probability))
        output[edge, 0] = endpoints[edge, 0]
        output[edge, 1] = endpoints[edge, 1]
        output[edge, 2] = np.log((1.0 - probability) / probability)
    return output


@njit(cache=True)
def fused_affine_quantized_weights_one(
    soft_measurements: np.ndarray,
    hard_measurements: np.ndarray,
    head_weights: np.ndarray,
    feature_min: np.ndarray,
    feature_range: np.ndarray,
    residual_factor: np.ndarray,
    measurement_indices: np.ndarray,
    measurement_mask: np.ndarray,
    floor_probability: float,
    ceiling_probability: float,
    probability_bits: int,
) -> np.ndarray:
    """Build edge LLRs after optional uniform quantization of soft probabilities.

    ``probability_bits == 0`` is the floating reference. Positive values use
    ``2**bits`` uniformly spaced reconstruction levels on [0, 0.5], followed by
    the same probability clipping as the floating path.
    """
    if probability_bits < 0 or probability_bits > 16:
        raise ValueError("probability_bits must be in [0, 16]")
    measurement_count = len(soft_measurements)
    error_probability = np.empty(measurement_count, dtype=np.float64)
    levels = (1 << probability_bits) - 1 if probability_bits > 0 else 0
    for measurement in range(measurement_count):
        real = (soft_measurements[measurement].real - feature_min[measurement, 0])
        real /= feature_range[measurement, 0]
        imag = (soft_measurements[measurement].imag - feature_min[measurement, 1])
        imag /= feature_range[measurement, 1]
        real = min(1.0, max(0.0, real))
        imag = min(1.0, max(0.0, imag))
        score = (
            head_weights[measurement, 0]
            + real * head_weights[measurement, 1]
            + imag * head_weights[measurement, 2]
        )
        probability = np.float32(1.0 / (1.0 + np.exp(-score)))
        probability = 1.0 - probability if hard_measurements[measurement] else probability
        probability = min(ceiling_probability, max(floor_probability, probability))
        if probability_bits > 0:
            probability = np.round((probability / 0.5) * levels) * 0.5 / levels
            probability = min(ceiling_probability, max(floor_probability, probability))
        error_probability[measurement] = probability
    output = np.empty(len(residual_factor), dtype=np.float64)
    for edge in range(len(residual_factor)):
        alpha = residual_factor[edge]
        for position in range(measurement_indices.shape[1]):
            if measurement_mask[edge, position]:
                alpha *= 1.0 - 2.0 * error_probability[measurement_indices[edge, position]]
        probability = (1.0 - alpha) / 2.0
        probability = min(ceiling_probability, max(floor_probability, probability))
        output[edge] = np.log((1.0 - probability) / probability)
    return output


def pack_frontier_schedule(plan: object) -> tuple[np.ndarray, ...]:
    """Convert a FrontierDecoderPlan into fixed-width integer arrays."""
    steps = plan.steps
    maximum_bag = max(len(step.active_with_vertex) for step in steps)
    edge_starts = np.zeros(len(steps) + 1, dtype=np.int64)
    edge_indices = []
    parity_toggles = []
    logical_toggles = []
    retained_count = np.zeros(len(steps), dtype=np.int64)
    retained_positions = np.zeros((len(steps), maximum_bag), dtype=np.int64)
    forgotten_count = np.zeros(len(steps), dtype=np.int64)
    forgotten_positions = np.zeros((len(steps), maximum_bag), dtype=np.int64)
    forgotten_vertices = np.zeros((len(steps), maximum_bag), dtype=np.int64)
    bag_width = np.zeros(len(steps), dtype=np.int64)
    for index, step in enumerate(steps):
        bag_width[index] = len(step.active_with_vertex)
        edge_starts[index] = len(edge_indices)
        edge_indices.extend(step.edge_indices)
        parity_toggles.extend(step.parity_toggles)
        logical_toggles.extend(step.logical_toggles)
        retained_count[index] = len(step.retained_positions)
        retained_positions[index, :retained_count[index]] = step.retained_positions
        forgotten_count[index] = len(step.forgotten_positions)
        forgotten_positions[index, :forgotten_count[index]] = step.forgotten_positions
        forgotten_vertices[index, :forgotten_count[index]] = step.forgotten_vertices
    edge_starts[-1] = len(edge_indices)
    return (
        bag_width,
        edge_starts,
        np.asarray(edge_indices, dtype=np.int64),
        np.asarray(parity_toggles, dtype=np.int64),
        np.asarray(logical_toggles, dtype=np.int64),
        retained_count,
        retained_positions,
        forgotten_count,
        forgotten_positions,
        forgotten_vertices,
    )


@njit(cache=True)
def decode_frontier_one(
    syndrome: np.ndarray,
    weights: np.ndarray,
    bag_width: np.ndarray,
    edge_starts: np.ndarray,
    edge_indices: np.ndarray,
    parity_toggles: np.ndarray,
    logical_toggles: np.ndarray,
    retained_count: np.ndarray,
    retained_positions: np.ndarray,
    forgotten_count: np.ndarray,
    forgotten_positions: np.ndarray,
    forgotten_vertices: np.ndarray,
) -> tuple[int, float]:
    max_width = 0
    for value in bag_width:
        max_width = max(max_width, value)
    max_states = 1 << (max_width + 1)
    costs = np.full(max_states, np.inf)
    work = np.full(max_states, np.inf)
    scratch = np.full(max_states, np.inf)
    costs[0] = 0.0
    old_width = 0
    for step in range(len(bag_width)):
        new_width = bag_width[step]
        nnew = 1 << (new_width + 1)
        for state in range(nnew):
            work[state] = np.inf
        for parity in range(1 << old_width):
            work[parity] = costs[parity]
            work[parity | (1 << new_width)] = costs[parity | (1 << old_width)]
        for location in range(edge_starts[step], edge_starts[step + 1]):
            toggle = parity_toggles[location] | (
                logical_toggles[location] << new_width
            )
            weight = weights[edge_indices[location]]
            for state in range(nnew):
                keep = work[state]
                select = work[state ^ toggle] + weight
                scratch[state] = keep if keep <= select else select
            temporary = work
            work = scratch
            scratch = temporary

        next_width = retained_count[step]
        nnext = 1 << (next_width + 1)
        for state in range(nnext):
            costs[state] = np.inf
        for state in range(nnew):
            cost = work[state]
            if not np.isfinite(cost):
                continue
            parity = state & ((1 << new_width) - 1)
            valid = True
            for item in range(forgotten_count[step]):
                position = forgotten_positions[step, item]
                vertex = forgotten_vertices[step, item]
                if ((parity >> position) & 1) != syndrome[vertex]:
                    valid = False
                    break
            if not valid:
                continue
            retained = 0
            for target in range(next_width):
                source = retained_positions[step, target]
                retained |= ((parity >> source) & 1) << target
            logical = (state >> new_width) & 1
            target_state = retained | (logical << next_width)
            if cost < costs[target_state]:
                costs[target_state] = cost
        old_width = next_width
    margin = abs(costs[1] - costs[0])
    return (1 if costs[1] < costs[0] else 0), margin


@njit(cache=True)
def decode_frontier_batch(
    syndromes: np.ndarray, weights: np.ndarray,
    bag_width: np.ndarray, edge_starts: np.ndarray,
    edge_indices: np.ndarray, parity_toggles: np.ndarray,
    logical_toggles: np.ndarray, retained_count: np.ndarray,
    retained_positions: np.ndarray, forgotten_count: np.ndarray,
    forgotten_positions: np.ndarray, forgotten_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.empty(len(syndromes), dtype=np.uint8)
    margins = np.empty(len(syndromes), dtype=np.float64)
    for row in range(len(syndromes)):
        prediction, margin = decode_frontier_one(
            syndromes[row], weights[row], bag_width, edge_starts,
            edge_indices, parity_toggles, logical_toggles, retained_count,
            retained_positions, forgotten_count, forgotten_positions,
            forgotten_vertices,
        )
        predictions[row] = prediction
        margins[row] = margin
    return predictions, margins
