#!/usr/bin/env python3
"""Cross-fit and select a bounded graph router on disjoint streams."""

from __future__ import annotations

import argparse
import gc
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
)
from scripts.run_factor_parameter_sweep import paired_episode_interval  # noqa: E402
from scripts.run_surface_mode_routing import paired_metrics, sample  # noqa: E402
from scripts.run_surface_regime_switch import (  # noqa: E402
    REGIMES,
    circuit,
    detailed_morphology_features,
    matching,
    morphology_features,
    select_fixed_benchmark,
)
from scripts.run_surface_residual_router import (  # noqa: E402
    THRESHOLDS,
    posterior,
    stream_data,
    threshold_prediction,
)


def decode_with_weights(
    decoders: list[pymatching.Matching], detectors: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    predictions = []
    weights = []
    for decoder in decoders:
        prediction, weight = decoder.decode_batch(detectors, return_weights=True)
        predictions.append(prediction[:, 0].astype(bool))
        weights.append(np.asarray(weight, dtype=np.float32))
    return predictions, weights


def energy_action_features(
    detailed: np.ndarray,
    probability: np.ndarray,
    weights: list[np.ndarray],
) -> np.ndarray:
    clipped = np.clip(probability.ravel(), 1e-6, 1.0 - 1e-6)
    log_odds = np.log(clipped) - np.log1p(-clipped)
    return np.column_stack(
        (
            detailed,
            log_odds.astype(np.float32),
            weights[0],
            weights[1],
            weights[1] - weights[0],
        )
    )


def candidate_predictions(
    detailed: np.ndarray,
    aggregate: np.ndarray,
    evidence_model: dict[str, np.ndarray | float],
    aggregate_model: dict[str, np.ndarray | float],
    action_model: dict[str, np.ndarray | float],
    endpoints: list[np.ndarray],
    weights: list[np.ndarray],
    *,
    streams: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    detailed_probability = posterior(
        detailed,
        evidence_model,
        streams=streams,
        horizon=horizon,
        temporal=True,
    )
    aggregate_probability = posterior(
        aggregate,
        aggregate_model,
        streams=streams,
        horizon=horizon,
        temporal=True,
    )
    candidates = {
        "aggregate_0.5": threshold_prediction(
            aggregate_probability, endpoints, 0.5
        )
    }
    for threshold in THRESHOLDS:
        candidates[f"detailed_{threshold:.1f}"] = threshold_prediction(
            detailed_probability, endpoints, threshold
        )
    action_score = affine_log_likelihood_ratio(
        energy_action_features(detailed, detailed_probability, weights),
        action_model,
    )
    shape = detailed_probability.shape
    candidates["energy_residual"] = np.where(
        (action_score >= 0).reshape(shape),
        endpoints[1].reshape(shape),
        endpoints[0].reshape(shape),
    )
    return candidates


def run_distance(
    distance: int,
    *,
    streams: int,
    horizon: int,
    calibration_shots: int,
    action_streams: int,
    selection_streams: int,
    seed: int,
) -> dict:
    circuits = [circuit(distance, *parameters) for parameters in REGIMES]
    endpoint_matchings = [matching(value) for value in circuits]
    calibration = [
        sample(value, calibration_shots, seed + 10 + mode)
        for mode, value in enumerate(circuits)
    ]
    detailed_calibration = [
        detailed_morphology_features(detectors, circuits[0])
        for detectors, _ in calibration
    ]
    evidence_model = fit_affine_log_likelihood_ratio(*detailed_calibration)
    aggregate_calibration = [
        morphology_features(detectors, circuits[0])
        for detectors, _ in calibration
    ]
    aggregate_model = fit_affine_log_likelihood_ratio(*aggregate_calibration)
    benchmark_matching, benchmark_metadata = select_fixed_benchmark(
        distance, calibration
    )
    detailed_feature_count = detailed_calibration[0].shape[1]
    del calibration, detailed_calibration, aggregate_calibration
    gc.collect()

    action_detectors, action_labels, _ = stream_data(
        circuits,
        seed=seed + 1000,
        streams=action_streams,
        horizon=horizon,
    )
    action_detailed = detailed_morphology_features(action_detectors, circuits[0])
    action_probability = posterior(
        action_detailed,
        evidence_model,
        streams=action_streams,
        horizon=horizon,
        temporal=True,
    )
    action_endpoints, action_weights = decode_with_weights(
        endpoint_matchings, action_detectors
    )
    disagreement = action_endpoints[0] != action_endpoints[1]
    action_matrix = energy_action_features(
        action_detailed, action_probability, action_weights
    )[disagreement]
    second_correct = (
        action_endpoints[1][disagreement] == action_labels[disagreement]
    )
    action_model = fit_affine_log_likelihood_ratio(
        action_matrix[~second_correct], action_matrix[second_correct]
    )
    action_fit_metadata = {
        "records": int(action_labels.size),
        "endpoint_disagreements": int(disagreement.sum()),
        "second_endpoint_correct_fraction": float(second_correct.mean()),
    }
    del action_detectors, action_detailed, action_matrix
    gc.collect()

    selection_detectors, selection_labels, _ = stream_data(
        circuits,
        seed=seed + 2000,
        streams=selection_streams,
        horizon=horizon,
    )
    selection_detailed = detailed_morphology_features(
        selection_detectors, circuits[0]
    )
    selection_aggregate = morphology_features(selection_detectors, circuits[0])
    selection_endpoints, selection_weights = decode_with_weights(
        endpoint_matchings, selection_detectors
    )
    selection_candidates = candidate_predictions(
        selection_detailed,
        selection_aggregate,
        evidence_model,
        aggregate_model,
        action_model,
        selection_endpoints,
        selection_weights,
        streams=selection_streams,
        horizon=horizon,
    )
    selection_scores = {
        name: float(np.mean(prediction.ravel() != selection_labels))
        for name, prediction in selection_candidates.items()
    }
    selected_policy = min(selection_scores, key=selection_scores.get)
    selection_metadata = {
        "records": int(selection_labels.size),
        "scores": selection_scores,
        "selected_policy": selected_policy,
    }
    del selection_detectors, selection_detailed, selection_aggregate
    del selection_candidates, selection_endpoints, selection_weights
    gc.collect()

    detectors, labels, hidden = stream_data(
        circuits, seed=seed, streams=streams, horizon=horizon
    )
    detailed = detailed_morphology_features(detectors, circuits[0])
    aggregate = morphology_features(detectors, circuits[0])
    endpoints, weights = decode_with_weights(endpoint_matchings, detectors)
    candidates = candidate_predictions(
        detailed,
        aggregate,
        evidence_model,
        aggregate_model,
        action_model,
        endpoints,
        weights,
        streams=streams,
        horizon=horizon,
    )
    compiled = candidates[selected_policy]
    prior_aggregate = candidates["aggregate_0.5"]
    benchmark = (
        benchmark_matching.decode_batch(detectors)[:, 0]
        .reshape(streams, horizon)
        .astype(bool)
    )
    truth = labels.reshape(streams, horizon)
    mode_informed = np.where(
        hidden.astype(bool),
        endpoints[1].reshape(streams, horizon),
        endpoints[0].reshape(streams, horizon),
    )
    predictions = {
        "selected_static_benchmark": benchmark,
        "prior_aggregate_causal_router": prior_aggregate,
        "compiled_policy": compiled,
        "mode_informed_endpoints": mode_informed,
    }
    methods = {
        name: paired_metrics(value, benchmark, truth)
        for name, value in predictions.items()
    }
    methods["compiled_policy"][
        "paired_stream_95pct_interval_vs_prior_aggregate"
    ] = paired_episode_interval(compiled, prior_aggregate, truth)
    oracle_gain = (
        methods["selected_static_benchmark"]["logical_error"]
        - methods["mode_informed_endpoints"]["logical_error"]
    )
    compiled_gain = (
        methods["selected_static_benchmark"]["logical_error"]
        - methods["compiled_policy"]["logical_error"]
    )
    methods["compiled_policy"]["mode_informed_gain_recovery"] = (
        float(compiled_gain / oracle_gain) if oracle_gain > 0 else None
    )

    stationary_controls = []
    for mode, physical_circuit in enumerate(circuits):
        control_detectors, control_labels = sample(
            physical_circuit, streams * horizon, seed + 30 + mode
        )
        control_detailed = detailed_morphology_features(
            control_detectors, circuits[0]
        )
        control_aggregate = morphology_features(control_detectors, circuits[0])
        control_endpoints, control_weights = decode_with_weights(
            endpoint_matchings, control_detectors
        )
        control_candidates = candidate_predictions(
            control_detailed,
            control_aggregate,
            evidence_model,
            aggregate_model,
            action_model,
            control_endpoints,
            control_weights,
            streams=streams,
            horizon=horizon,
        )
        control_compiled = control_candidates[selected_policy]
        reference = control_endpoints[mode].reshape(streams, horizon)
        control_truth = control_labels.reshape(streams, horizon)
        stationary_controls.append(
            {
                "physical_regime": mode,
                "endpoint_graph_ler": float(np.mean(reference != control_truth)),
                "compiled_ler": float(np.mean(control_compiled != control_truth)),
                "paired_stream_95pct_interval_vs_endpoint_graph":
                    paired_episode_interval(
                        control_compiled, reference, control_truth
                    ),
            }
        )
        del control_detectors, control_detailed, control_aggregate
        del control_candidates, control_endpoints, control_weights
        gc.collect()

    return {
        "distance": distance,
        "records": int(labels.size),
        "detailed_feature_count": int(detailed_feature_count),
        "selected_static_benchmark": benchmark_metadata,
        "action_fit": action_fit_metadata,
        "policy_selection": selection_metadata,
        "compiled_models": {
            "evidence_weights": np.asarray(evidence_model["weights"]).tolist(),
            "evidence_bias": float(evidence_model["bias"]),
            "action_weights": np.asarray(action_model["weights"]).tolist(),
            "action_bias": float(action_model["bias"]),
        },
        "methods": methods,
        "stationary_controls": stationary_controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--streams", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--calibration-shots", type=int, default=65536)
    parser.add_argument("--action-streams", type=int, default=256)
    parser.add_argument("--selection-streams", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261231)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol": "docs/surface-crossfit-compiler-protocol.md",
        "versions": {
            "stim": stim.__version__,
            "pymatching": pymatching.__version__,
        },
        "seed": args.seed,
        "streams": args.streams,
        "horizon": args.horizon,
        "calibration_shots_per_regime": args.calibration_shots,
        "action_fit_streams": args.action_streams,
        "policy_selection_streams": args.selection_streams,
        "distances": [
            run_distance(
                distance,
                streams=args.streams,
                horizon=args.horizon,
                calibration_shots=args.calibration_shots,
                action_streams=args.action_streams,
                selection_streams=args.selection_streams,
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
