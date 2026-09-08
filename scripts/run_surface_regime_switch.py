#!/usr/bin/env python3
"""Route incompatible surface-code noise regimes with a tiny causal state."""

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
from ptwm.surface_mode import (  # noqa: E402
    affine_log_likelihood_ratio,
    fit_affine_log_likelihood_ratio,
    log_likelihood_mode_posterior,
)
from scripts.run_factor_parameter_sweep import paired_episode_interval  # noqa: E402
from scripts.run_surface_mode_routing import paired_metrics, sample  # noqa: E402


TRANSITION = np.asarray([[0.98, 0.02], [0.02, 0.98]])
STATIONARY = np.asarray([0.5, 0.5])
CLIFFORD_GRID = (0.001, 0.003, 0.005, 0.007, 0.01)
MEASUREMENT_GRID = (0.001, 0.005, 0.01, 0.015, 0.02)
REGIMES = ((0.001, 0.02), (0.01, 0.001))


def circuit(distance: int, clifford: float, measurement: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=clifford,
        before_measure_flip_probability=measurement,
        after_reset_flip_probability=measurement,
    )


def matching(circuit_value: stim.Circuit) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(
        circuit_value.detector_error_model(decompose_errors=True)
    )


def regime_modes(seed: int, streams: int, horizon: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((streams, horizon), dtype=np.uint8)
    result[:, 0] = rng.integers(2, size=streams)
    for time in range(1, horizon):
        prior = result[:, time - 1]
        result[:, time] = rng.random(streams) >= TRANSITION[prior, 0]
    return result


def morphology_features(
    detectors: np.ndarray, circuit_value: stim.Circuit
) -> np.ndarray:
    """Cheap count, spacetime-coincidence, and per-round detector features."""
    coordinates = circuit_value.get_detector_coordinates()
    xyz = np.asarray([coordinates[index][:3] for index in range(len(coordinates))])
    temporal_pairs = []
    spatial_pairs = []
    for first in range(len(xyz)):
        for second in range(first + 1, len(xyz)):
            delta = np.abs(xyz[first] - xyz[second])
            if np.all(delta[:2] == 0) and delta[2] == 1:
                temporal_pairs.append((first, second))
            if delta[2] == 0 and np.sum(delta[:2] ** 2) <= 4.01:
                spatial_pairs.append((first, second))

    def coincidence(pairs: list[tuple[int, int]]) -> np.ndarray:
        if not pairs:
            return np.zeros(len(detectors))
        output = np.zeros(len(detectors), dtype=np.int32)
        for start in range(0, len(pairs), 32):
            first, second = np.asarray(pairs[start : start + 32]).T
            output += np.sum(
                detectors[:, first] & detectors[:, second], axis=1
            ).astype(np.int32)
        return output

    times = np.unique(xyz[:, 2])
    per_time = np.column_stack(
        [detectors[:, xyz[:, 2] == time].sum(axis=1) for time in times]
    )
    return np.column_stack(
        (
            detectors.sum(axis=1),
            coincidence(temporal_pairs),
            coincidence(spatial_pairs),
            per_time,
        )
    ).astype(float)


def detailed_morphology_features(
    detectors: np.ndarray, circuit_value: stim.Circuit
) -> np.ndarray:
    """Expose local detector and neighboring-pair terms to a bounded affine head."""
    coordinates = circuit_value.get_detector_coordinates()
    xyz = np.asarray([coordinates[index][:3] for index in range(len(coordinates))])
    pairs = []
    for first in range(len(xyz)):
        for second in range(first + 1, len(xyz)):
            delta = np.abs(xyz[first] - xyz[second])
            temporal = np.all(delta[:2] == 0) and delta[2] == 1
            spatial = delta[2] == 0 and np.sum(delta[:2] ** 2) <= 4.01
            if temporal or spatial:
                pairs.append((first, second))
    local_products = np.empty((len(detectors), len(pairs)), dtype=np.float32)
    for start in range(0, len(pairs), 32):
        first, second = np.asarray(pairs[start : start + 32]).T
        local_products[:, start : start + len(first)] = (
            detectors[:, first] & detectors[:, second]
        )
    aggregate = morphology_features(detectors, circuit_value).astype(np.float32)
    return np.column_stack(
        (detectors.astype(np.float32), local_products, aggregate)
    )


def select_fixed_benchmark(
    distance: int,
    calibration: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pymatching.Matching, dict]:
    candidates = []
    for clifford in CLIFFORD_GRID:
        for measurement in MEASUREMENT_GRID:
            decoder = matching(circuit(distance, clifford, measurement))
            regime_ler = [
                float(np.mean(decoder.decode_batch(detectors)[:, 0] != labels))
                for detectors, labels in calibration
            ]
            candidates.append(
                {
                    "clifford": clifford,
                    "measurement": measurement,
                    "regime_ler": regime_ler,
                    "mean_ler": float(np.mean(regime_ler)),
                    "decoder": decoder,
                }
            )
    selected = min(candidates, key=lambda item: item["mean_ler"])
    metadata = {
        key: value for key, value in selected.items() if key != "decoder"
    }
    metadata["candidate_count"] = len(candidates)
    return selected["decoder"], metadata


def route_predictions(
    features: np.ndarray,
    model: dict[str, np.ndarray | float],
    endpoints: list[np.ndarray],
    *,
    streams: int,
    horizon: int,
    temporal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    evidence = affine_log_likelihood_ratio(features, model).reshape(streams, horizon)
    probability = log_likelihood_mode_posterior(
        evidence, TRANSITION, STATIONARY, temporal=temporal
    )
    selected = probability >= 0.5
    prediction = np.where(
        selected,
        endpoints[1].reshape(streams, horizon),
        endpoints[0].reshape(streams, horizon),
    )
    return prediction, selected


def run_distance(
    distance: int,
    *,
    streams: int,
    horizon: int,
    calibration_shots: int,
    seed: int,
) -> dict:
    endpoint_circuits = [circuit(distance, *parameters) for parameters in REGIMES]
    endpoint_matchings = [matching(value) for value in endpoint_circuits]
    calibration = [
        sample(value, calibration_shots, seed + 10 + mode)
        for mode, value in enumerate(endpoint_circuits)
    ]
    calibration_features = [
        morphology_features(detectors, endpoint_circuits[0])
        for detectors, _ in calibration
    ]
    model = fit_affine_log_likelihood_ratio(*calibration_features)
    benchmark_matching, benchmark_metadata = select_fixed_benchmark(
        distance, calibration
    )

    hidden = regime_modes(seed, streams, horizon)
    flat_mode = hidden.ravel()
    total = flat_mode.size
    detectors = np.empty((total, endpoint_circuits[0].num_detectors), dtype=bool)
    labels = np.empty(total, dtype=bool)
    for mode, physical_circuit in enumerate(endpoint_circuits):
        positions = np.flatnonzero(flat_mode == mode)
        mode_detectors, mode_labels = sample(
            physical_circuit, len(positions), seed + 20 + mode
        )
        detectors[positions] = mode_detectors
        labels[positions] = mode_labels

    features = morphology_features(detectors, endpoint_circuits[0])
    endpoints = [
        decoder.decode_batch(detectors)[:, 0].astype(bool)
        for decoder in endpoint_matchings
    ]
    benchmark = (
        benchmark_matching.decode_batch(detectors)[:, 0]
        .reshape(streams, horizon)
        .astype(bool)
    )
    memoryless, memoryless_selected = route_predictions(
        features,
        model,
        endpoints,
        streams=streams,
        horizon=horizon,
        temporal=False,
    )
    causal, causal_selected = route_predictions(
        features,
        model,
        endpoints,
        streams=streams,
        horizon=horizon,
        temporal=True,
    )
    truth = labels.reshape(streams, horizon)
    mode_informed = np.where(
        hidden.astype(bool),
        endpoints[1].reshape(streams, horizon),
        endpoints[0].reshape(streams, horizon),
    )
    predictions = {
        "selected_static_benchmark": benchmark,
        "memoryless_affine_router": memoryless,
        "causal_affine_router": causal,
        "mode_informed_endpoints": mode_informed,
    }
    methods = {
        name: paired_metrics(value, benchmark, truth)
        for name, value in predictions.items()
    }
    methods["causal_affine_router"][
        "paired_stream_95pct_interval_vs_memoryless"
    ] = paired_episode_interval(causal, memoryless, truth)
    oracle_gain = (
        methods["selected_static_benchmark"]["logical_error"]
        - methods["mode_informed_endpoints"]["logical_error"]
    )
    causal_gain = (
        methods["selected_static_benchmark"]["logical_error"]
        - methods["causal_affine_router"]["logical_error"]
    )
    methods["causal_affine_router"]["mode_informed_gain_recovery"] = (
        float(causal_gain / oracle_gain) if oracle_gain > 0 else None
    )

    stationary_controls = []
    for mode, physical_circuit in enumerate(endpoint_circuits):
        control_detectors, control_labels = sample(
            physical_circuit, total, seed + 30 + mode
        )
        control_endpoints = [
            decoder.decode_batch(control_detectors)[:, 0].astype(bool)
            for decoder in endpoint_matchings
        ]
        control_features = morphology_features(
            control_detectors, endpoint_circuits[0]
        )
        control_router, control_selected = route_predictions(
            control_features,
            model,
            control_endpoints,
            streams=streams,
            horizon=horizon,
            temporal=True,
        )
        reference = control_endpoints[mode].reshape(streams, horizon)
        control_truth = control_labels.reshape(streams, horizon)
        stationary_controls.append(
            {
                "physical_regime": mode,
                "endpoint_graph_ler": float(np.mean(reference != control_truth)),
                "router_ler": float(np.mean(control_router != control_truth)),
                "paired_stream_95pct_interval_vs_endpoint_graph":
                    paired_episode_interval(
                        control_router, reference, control_truth
                    ),
                "wrong_graph_selection_fraction": float(
                    np.mean(control_selected != bool(mode))
                ),
            }
        )

    return {
        "distance": distance,
        "rounds": distance,
        "detectors": endpoint_circuits[0].num_detectors,
        "records": total,
        "regime_one_fraction": float(hidden.mean()),
        "selected_static_benchmark": benchmark_metadata,
        "affine_head": {
            "feature_count": int(calibration_features[0].shape[1]),
            "weights": np.asarray(model["weights"]).tolist(),
            "bias": float(model["bias"]),
            "mean": np.asarray(model["mean"]).tolist(),
            "scale": np.asarray(model["scale"]).tolist(),
        },
        "router_diagnostics": {
            "memoryless_mode_error": float(
                np.mean(memoryless_selected != hidden)
            ),
            "causal_mode_error": float(np.mean(causal_selected != hidden)),
        },
        "methods": methods,
        "stationary_controls": stationary_controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--streams", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--calibration-shots", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol": "docs/surface-regime-switch-protocol.md",
        "versions": {
            "stim": stim.__version__,
            "pymatching": pymatching.__version__,
        },
        "transition": TRANSITION.tolist(),
        "regimes": [
            {
                "after_clifford_depolarization": values[0],
                "reset_measurement_flip": values[1],
            }
            for values in REGIMES
        ],
        "static_search": {
            "after_clifford_depolarization": CLIFFORD_GRID,
            "reset_measurement_flip": MEASUREMENT_GRID,
        },
        "streams": args.streams,
        "horizon": args.horizon,
        "calibration_shots_per_regime": args.calibration_shots,
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
