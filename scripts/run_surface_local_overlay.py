#!/usr/bin/env python3
"""Route a spatially local surface-code reset/readout graph overlay."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymatching
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.factor_gating import BINARY_TRANSITION, binary_stationary  # noqa: E402
from ptwm.surface_mode import count_mode_posterior, fit_count_emission  # noqa: E402
from scripts.run_factor_parameter_sweep import paired_episode_interval  # noqa: E402
from scripts.run_surface_mode_routing import modes, paired_metrics, sample  # noqa: E402


NOMINAL_P, BURST_P, CLIFFORD_P = 0.001, 0.01, 0.001


def base_circuit(distance: int) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=CLIFFORD_P,
        before_measure_flip_probability=NOMINAL_P,
        after_reset_flip_probability=NOMINAL_P,
    ).flattened()


def ancilla_coordinates(circuit: stim.Circuit) -> dict[int, tuple[float, ...]]:
    coordinates: dict[int, tuple[float, ...]] = {}
    ancillas: set[int] = set()
    for instruction in circuit:
        if instruction.name == "QUBIT_COORDS":
            target = instruction.targets_copy()[0].value
            coordinates[target] = tuple(instruction.gate_args_copy())
        elif instruction.name == "MR":
            ancillas.update(target.value for target in instruction.targets_copy())
    return {qubit: coordinates[qubit] for qubit in ancillas}


def burst_overlay(circuit: stim.Circuit, selected_ancillas: set[int]) -> stim.Circuit:
    """Compose extra X noise so selected nominal channels total ``BURST_P``."""
    extra_probability = (BURST_P - NOMINAL_P) / (1.0 - 2.0 * NOMINAL_P)
    output = stim.Circuit()
    for instruction in circuit:
        output.append(instruction)
        if instruction.name != "X_ERROR":
            continue
        selected = [
            target.value
            for target in instruction.targets_copy()
            if target.value in selected_ancillas
        ]
        if selected:
            output.append("X_ERROR", selected, extra_probability)
    return output


def routed_prediction(
    counts: np.ndarray,
    emission: np.ndarray,
    nominal_prediction: np.ndarray,
    burst_prediction: np.ndarray,
    *,
    temporal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    probability = count_mode_posterior(
        counts,
        emission,
        BINARY_TRANSITION,
        binary_stationary(),
        temporal=temporal,
    )
    selected = probability >= 0.5
    return np.where(selected, burst_prediction, nominal_prediction), selected


def matching(circuit: stim.Circuit) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )


def run_distance(
    distance: int,
    *,
    streams: int,
    horizon: int,
    calibration_shots: int,
    seed: int,
) -> dict:
    nominal = base_circuit(distance)
    coordinates = ancilla_coordinates(nominal)
    local_ancillas = {q for q, xy in coordinates.items() if xy[0] <= distance}
    circuits = [
        nominal,
        burst_overlay(nominal, local_ancillas),
        burst_overlay(nominal, set(coordinates)),
    ]
    matchings = [matching(item) for item in circuits]
    calibration = [
        sample(item, calibration_shots, seed + 10 + mode)
        for mode, item in enumerate(circuits[:2])
    ]

    stationary = binary_stationary()
    fixed_scores = []
    for decoder in matchings:
        errors = [
            np.mean(decoder.decode_batch(detectors)[:, 0] != labels)
            for detectors, labels in calibration
        ]
        fixed_scores.append(float(stationary @ np.asarray(errors)))
    benchmark_index = int(np.argmin(fixed_scores))

    detector_coordinates = nominal.get_detector_coordinates()
    local_detector_mask = np.asarray(
        [
            detector_coordinates[index][0] <= distance
            for index in range(nominal.num_detectors)
        ],
        dtype=bool,
    )
    local_emission = fit_count_emission(
        calibration[0][0][:, local_detector_mask].sum(axis=1),
        calibration[1][0][:, local_detector_mask].sum(axis=1),
        int(local_detector_mask.sum()),
    )
    global_emission = fit_count_emission(
        calibration[0][0].sum(axis=1),
        calibration[1][0].sum(axis=1),
        nominal.num_detectors,
    )

    hidden = modes(seed, streams, horizon)
    flat_mode = hidden.ravel()
    total = flat_mode.size
    detectors = np.empty((total, nominal.num_detectors), dtype=bool)
    labels = np.empty(total, dtype=bool)
    for mode, physical_circuit in enumerate(circuits[:2]):
        positions = np.flatnonzero(flat_mode == mode)
        mode_detectors, mode_labels = sample(
            physical_circuit, len(positions), seed + 20 + mode
        )
        detectors[positions] = mode_detectors
        labels[positions] = mode_labels

    decoded = [
        decoder.decode_batch(detectors)[:, 0]
        .reshape(streams, horizon)
        .astype(bool)
        for decoder in matchings
    ]
    truth = labels.reshape(streams, horizon)
    mode_mask = hidden.astype(bool)
    local_counts = detectors[:, local_detector_mask].sum(axis=1).reshape(
        streams, horizon
    )
    global_counts = detectors.sum(axis=1).reshape(streams, horizon)
    local_memoryless, memoryless_selected = routed_prediction(
        local_counts,
        local_emission,
        decoded[0],
        decoded[1],
        temporal=False,
    )
    local_temporal, local_selected = routed_prediction(
        local_counts,
        local_emission,
        decoded[0],
        decoded[1],
        temporal=True,
    )
    global_temporal, global_selected = routed_prediction(
        global_counts,
        global_emission,
        decoded[0],
        decoded[1],
        temporal=True,
    )
    predictions = {
        "static_nominal": decoded[0],
        "static_local_burst": decoded[1],
        "static_global_burst": decoded[2],
        "mode_local_graph": np.where(mode_mask, decoded[1], decoded[0]),
        "mode_wrong_global_graph": np.where(mode_mask, decoded[2], decoded[0]),
        "local_memoryless_router": local_memoryless,
        "local_temporal_router": local_temporal,
        "global_temporal_router": global_temporal,
    }
    benchmark = decoded[benchmark_index]
    methods = {
        name: paired_metrics(value, decoded[0], truth)
        for name, value in predictions.items()
    }
    methods["mode_local_graph"][
        "paired_stream_95pct_interval_vs_wrong_global"
    ] = paired_episode_interval(
        predictions["mode_local_graph"],
        predictions["mode_wrong_global_graph"],
        truth,
    )
    methods["local_temporal_router"][
        "paired_stream_95pct_interval_vs_memoryless"
    ] = paired_episode_interval(local_temporal, local_memoryless, truth)
    methods["local_temporal_router"][
        "paired_stream_95pct_interval_vs_global_observation"
    ] = paired_episode_interval(local_temporal, global_temporal, truth)
    methods["local_temporal_router"][
        "paired_stream_95pct_interval_vs_selected_fixed_benchmark"
    ] = paired_episode_interval(local_temporal, benchmark, truth)

    null_detectors, null_labels = sample(nominal, total, seed + 30)
    null_static = (
        matchings[0]
        .decode_batch(null_detectors)[:, 0]
        .reshape(streams, horizon)
        .astype(bool)
    )
    null_burst = (
        matchings[1]
        .decode_batch(null_detectors)[:, 0]
        .reshape(streams, horizon)
        .astype(bool)
    )
    null_counts = null_detectors[:, local_detector_mask].sum(axis=1).reshape(
        streams, horizon
    )
    null_router, null_selected = routed_prediction(
        null_counts,
        local_emission,
        null_static,
        null_burst,
        temporal=True,
    )
    null_truth = null_labels.reshape(streams, horizon)
    null = paired_metrics(null_router, null_static, null_truth)
    null["static_logical_error"] = float(np.mean(null_static != null_truth))
    null["overlay_selection_fraction"] = float(null_selected.mean())

    return {
        "distance": distance,
        "rounds": distance,
        "records": total,
        "local_ancillas": len(local_ancillas),
        "total_ancillas": len(coordinates),
        "local_detectors": int(local_detector_mask.sum()),
        "total_detectors": nominal.num_detectors,
        "burst_mode_fraction": float(hidden.mean()),
        "fixed_graph_calibration_weighted_ler": {
            name: score
            for name, score in zip(
                ("nominal", "local_burst", "global_burst"), fixed_scores
            )
        },
        "selected_fixed_benchmark": (
            "nominal", "local_burst", "global_burst"
        )[benchmark_index],
        "selected_fixed_benchmark": (
            "nominal",
            "local_burst",
            "global_burst",
        )[benchmark_index],
        "fixed_calibration_logical_error": {
            name: score
            for name, score in zip(
                ("nominal", "local_burst", "global_burst"), fixed_scores
            )
        },
        "router_diagnostics": {
            "memoryless_selection_fraction": float(memoryless_selected.mean()),
            "local_temporal_selection_fraction": float(local_selected.mean()),
            "global_temporal_selection_fraction": float(global_selected.mean()),
            "local_temporal_mode_error": float(np.mean(local_selected != mode_mask)),
            "global_temporal_mode_error": float(
                np.mean(global_selected != mode_mask)
            ),
        },
        "methods": methods,
        "nominal_only_null": null,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--streams", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--calibration-shots", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol": "docs/surface-local-overlay-protocol.md",
        "versions": {
            "stim": stim.__version__,
            "pymatching": pymatching.__version__,
        },
        "noise": {
            "nominal_reset_measurement_flip": NOMINAL_P,
            "local_burst_reset_measurement_flip": BURST_P,
            "after_clifford_depolarization": CLIFFORD_P,
        },
        "streams": args.streams,
        "horizon": args.horizon,
        "calibration_shots_per_mode": args.calibration_shots,
        "seed": args.seed,
        "distances": [
            run_distance(
                distance,
                streams=args.streams,
                horizon=args.horizon,
                calibration_shots=args.calibration_shots,
                seed=args.seed + 100 * index,
            )
            for index, distance in enumerate(args.distances)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
