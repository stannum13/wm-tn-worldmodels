#!/usr/bin/env python3
"""Compile a detailed posterior or residual router around two trusted MWPM graphs."""

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
    log_likelihood_mode_posterior,
)
from scripts.run_factor_parameter_sweep import paired_episode_interval  # noqa: E402
from scripts.run_surface_mode_routing import paired_metrics, sample  # noqa: E402
from scripts.run_surface_regime_switch import (  # noqa: E402
    REGIMES,
    STATIONARY,
    TRANSITION,
    circuit,
    detailed_morphology_features,
    matching,
    morphology_features,
    regime_modes,
    select_fixed_benchmark,
)


THRESHOLDS = tuple(np.arange(0.1, 1.0, 0.1))


def stream_data(
    circuits: list[stim.Circuit],
    *,
    seed: int,
    streams: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = regime_modes(seed, streams, horizon)
    flat_mode = hidden.ravel()
    detectors = np.empty(
        (flat_mode.size, circuits[0].num_detectors), dtype=bool
    )
    labels = np.empty(flat_mode.size, dtype=bool)
    for mode, physical_circuit in enumerate(circuits):
        positions = np.flatnonzero(flat_mode == mode)
        mode_detectors, mode_labels = sample(
            physical_circuit, len(positions), seed + 10 + mode
        )
        detectors[positions] = mode_detectors
        labels[positions] = mode_labels
    return detectors, labels, hidden


def posterior(
    features: np.ndarray,
    evidence_model: dict[str, np.ndarray | float],
    *,
    streams: int,
    horizon: int,
    temporal: bool,
) -> np.ndarray:
    evidence = affine_log_likelihood_ratio(features, evidence_model).reshape(
        streams, horizon
    )
    return log_likelihood_mode_posterior(
        evidence, TRANSITION, STATIONARY, temporal=temporal
    )


def threshold_prediction(
    probability: np.ndarray,
    endpoints: list[np.ndarray],
    threshold: float,
) -> np.ndarray:
    shape = probability.shape
    return np.where(
        probability >= threshold,
        endpoints[1].reshape(shape),
        endpoints[0].reshape(shape),
    )


def action_features(features: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability.ravel(), 1e-6, 1.0 - 1e-6)
    log_odds = np.log(clipped) - np.log1p(-clipped)
    return np.column_stack((features, log_odds.astype(np.float32)))


def residual_prediction(
    features: np.ndarray,
    probability: np.ndarray,
    endpoints: list[np.ndarray],
    action_model: dict[str, np.ndarray | float],
) -> np.ndarray:
    choose_second = affine_log_likelihood_ratio(
        action_features(features, probability), action_model
    ) >= 0
    shape = probability.shape
    return np.where(
        choose_second.reshape(shape),
        endpoints[1].reshape(shape),
        endpoints[0].reshape(shape),
    )


def run_distance(
    distance: int,
    *,
    streams: int,
    horizon: int,
    calibration_shots: int,
    validation_streams: int,
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
    feature_count = detailed_calibration[0].shape[1]
    del detailed_calibration, aggregate_calibration, calibration
    gc.collect()

    validation_detectors, validation_labels, _ = stream_data(
        circuits,
        seed=seed + 1000,
        streams=validation_streams,
        horizon=horizon,
    )
    validation_features = detailed_morphology_features(
        validation_detectors, circuits[0]
    )
    validation_probability = posterior(
        validation_features,
        evidence_model,
        streams=validation_streams,
        horizon=horizon,
        temporal=True,
    )
    validation_endpoints = [
        decoder.decode_batch(validation_detectors)[:, 0].astype(bool)
        for decoder in endpoint_matchings
    ]
    threshold_scores = {}
    for threshold in THRESHOLDS:
        prediction = threshold_prediction(
            validation_probability, validation_endpoints, threshold
        )
        threshold_scores[str(round(threshold, 1))] = float(
            np.mean(prediction.ravel() != validation_labels)
        )
    best_threshold = min(
        THRESHOLDS, key=lambda value: threshold_scores[str(round(value, 1))]
    )

    disagreement = validation_endpoints[0] != validation_endpoints[1]
    validation_action_features = action_features(
        validation_features, validation_probability
    )[disagreement]
    second_correct = (
        validation_endpoints[1][disagreement]
        == validation_labels[disagreement]
    )
    action_model = fit_affine_log_likelihood_ratio(
        validation_action_features[~second_correct],
        validation_action_features[second_correct],
    )
    residual_validation = residual_prediction(
        validation_features,
        validation_probability,
        validation_endpoints,
        action_model,
    )
    residual_score = float(
        np.mean(residual_validation.ravel() != validation_labels)
    )
    threshold_score = threshold_scores[str(round(best_threshold, 1))]
    selected_policy = "residual" if residual_score < threshold_score else "threshold"
    validation_metadata = {
        "records": int(validation_labels.size),
        "endpoint_disagreements": int(disagreement.sum()),
        "threshold_scores": threshold_scores,
        "best_threshold": float(best_threshold),
        "best_threshold_ler": threshold_score,
        "residual_ler": residual_score,
        "selected_policy": selected_policy,
    }
    del validation_detectors, validation_features, validation_action_features
    gc.collect()

    detectors, labels, hidden = stream_data(
        circuits, seed=seed, streams=streams, horizon=horizon
    )
    features = detailed_morphology_features(detectors, circuits[0])
    detailed_temporal_probability = posterior(
        features,
        evidence_model,
        streams=streams,
        horizon=horizon,
        temporal=True,
    )
    detailed_memoryless_probability = posterior(
        features,
        evidence_model,
        streams=streams,
        horizon=horizon,
        temporal=False,
    )
    endpoints = [
        decoder.decode_batch(detectors)[:, 0].astype(bool)
        for decoder in endpoint_matchings
    ]
    benchmark = (
        benchmark_matching.decode_batch(detectors)[:, 0]
        .reshape(streams, horizon)
        .astype(bool)
    )
    aggregate_features = morphology_features(detectors, circuits[0])
    aggregate_probability = posterior(
        aggregate_features,
        aggregate_model,
        streams=streams,
        horizon=horizon,
        temporal=True,
    )
    aggregate_router = threshold_prediction(
        aggregate_probability, endpoints, 0.5
    )
    detailed_memoryless = threshold_prediction(
        detailed_memoryless_probability, endpoints, best_threshold
    )
    detailed_causal = threshold_prediction(
        detailed_temporal_probability, endpoints, best_threshold
    )
    residual = residual_prediction(
        features, detailed_temporal_probability, endpoints, action_model
    )
    compiled = residual if selected_policy == "residual" else detailed_causal
    truth = labels.reshape(streams, horizon)
    mode_informed = np.where(
        hidden.astype(bool),
        endpoints[1].reshape(streams, horizon),
        endpoints[0].reshape(streams, horizon),
    )
    predictions = {
        "selected_static_benchmark": benchmark,
        "prior_aggregate_causal_router": aggregate_router,
        "detailed_memoryless_router": detailed_memoryless,
        "detailed_causal_router": detailed_causal,
        "residual_action_router": residual,
        "compiled_policy": compiled,
        "mode_informed_endpoints": mode_informed,
    }
    methods = {
        name: paired_metrics(value, benchmark, truth)
        for name, value in predictions.items()
    }
    methods["compiled_policy"][
        "paired_stream_95pct_interval_vs_prior_aggregate"
    ] = paired_episode_interval(compiled, aggregate_router, truth)
    methods["detailed_causal_router"][
        "paired_stream_95pct_interval_vs_detailed_memoryless"
    ] = paired_episode_interval(detailed_causal, detailed_memoryless, truth)
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
        control_features = detailed_morphology_features(
            control_detectors, circuits[0]
        )
        control_probability = posterior(
            control_features,
            evidence_model,
            streams=streams,
            horizon=horizon,
            temporal=True,
        )
        control_endpoints = [
            decoder.decode_batch(control_detectors)[:, 0].astype(bool)
            for decoder in endpoint_matchings
        ]
        if selected_policy == "residual":
            control_compiled = residual_prediction(
                control_features,
                control_probability,
                control_endpoints,
                action_model,
            )
        else:
            control_compiled = threshold_prediction(
                control_probability, control_endpoints, best_threshold
            )
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
        del control_detectors, control_features
        gc.collect()

    return {
        "distance": distance,
        "records": int(labels.size),
        "detailed_feature_count": int(feature_count),
        "selected_static_benchmark": benchmark_metadata,
        "validation_compiler": validation_metadata,
        "compiled_models": {
            "evidence_weights": np.asarray(evidence_model["weights"]).tolist(),
            "evidence_bias": float(evidence_model["bias"]),
            "action_weights": np.asarray(action_model["weights"]).tolist(),
            "action_bias": float(action_model["bias"]),
        },
        "test_mode_error": {
            "detailed_memoryless": float(
                np.mean((detailed_memoryless_probability >= 0.5) != hidden)
            ),
            "detailed_causal": float(
                np.mean((detailed_temporal_probability >= 0.5) != hidden)
            ),
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
    parser.add_argument("--validation-streams", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol": "docs/surface-residual-router-protocol.md",
        "versions": {
            "stim": stim.__version__,
            "pymatching": pymatching.__version__,
        },
        "streams": args.streams,
        "horizon": args.horizon,
        "calibration_shots_per_regime": args.calibration_shots,
        "validation_streams": args.validation_streams,
        "seed": args.seed,
        "distances": [
            run_distance(
                distance,
                streams=args.streams,
                horizon=args.horizon,
                calibration_shots=args.calibration_shots,
                validation_streams=args.validation_streams,
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
