#!/usr/bin/env python3
"""Persistent measurement-burst graph-template routing on rotated surface codes."""

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


NOMINAL_P, BURST_P, CLIFFORD_P = 0.001, 0.01, 0.001


def circuit(distance: int, measurement_probability: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=distance,
        after_clifford_depolarization=CLIFFORD_P,
        before_measure_flip_probability=measurement_probability,
        after_reset_flip_probability=measurement_probability,
    )


def sample(c: stim.Circuit, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    detectors, observables = c.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True,
    )
    return detectors, observables[:, 0]


def modes(seed: int, streams: int, horizon: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((streams, horizon), dtype=np.uint8)
    result[:, 0] = rng.choice(2, size=streams, p=binary_stationary())
    for time in range(1, horizon):
        uniforms = rng.random(streams)
        prior = result[:, time - 1]
        result[:, time] = uniforms >= BINARY_TRANSITION[prior, 0]
    return result


def paired_metrics(
    prediction: np.ndarray, reference: np.ndarray, labels: np.ndarray,
) -> dict:
    return {
        "logical_error": float(np.mean(prediction != labels)),
        "paired_stream_95pct_interval_vs_static": paired_episode_interval(
            prediction, reference, labels,
        ),
        "disagreements_vs_static": int(np.sum(prediction != reference)),
    }


def run_distance(
    distance: int, *, streams: int, horizon: int, calibration_shots: int, seed: int,
) -> dict:
    circuits = [circuit(distance, probability) for probability in (NOMINAL_P, BURST_P)]
    matchings = [
        pymatching.Matching.from_detector_error_model(
            item.detector_error_model(decompose_errors=True),
        )
        for item in circuits
    ]
    calibration = [
        sample(item, calibration_shots, seed + 10 + mode)
        for mode, item in enumerate(circuits)
    ]
    detector_count = circuits[0].num_detectors
    emission = fit_count_emission(
        calibration[0][0].sum(axis=1), calibration[1][0].sum(axis=1), detector_count,
    )
    hidden = modes(seed, streams, horizon)
    flat_mode = hidden.ravel()
    total = flat_mode.size
    detectors = np.empty((total, detector_count), dtype=bool)
    labels = np.empty(total, dtype=bool)
    for mode, item in enumerate(circuits):
        positions = np.flatnonzero(flat_mode == mode)
        mode_detectors, mode_labels = sample(item, len(positions), seed + 20 + mode)
        detectors[positions], labels[positions] = mode_detectors, mode_labels
    decoded = [matching.decode_batch(detectors)[:, 0].astype(bool) for matching in matchings]
    counts = detectors.sum(axis=1).reshape(streams, horizon)
    memoryless_probability = count_mode_posterior(
        counts, emission, BINARY_TRANSITION, binary_stationary(), temporal=False,
    )
    temporal_probability = count_mode_posterior(
        counts, emission, BINARY_TRANSITION, binary_stationary(), temporal=True,
    )
    static = decoded[0].reshape(streams, horizon)
    burst = decoded[1].reshape(streams, horizon)
    truth = labels.reshape(streams, horizon)
    prediction = {
        "static_nominal": static,
        "always_burst": burst,
        "mode_oracle": np.where(hidden.astype(bool), burst, static),
        "memoryless_router": np.where(memoryless_probability >= 0.5, burst, static),
        "temporal_router": np.where(temporal_probability >= 0.5, burst, static),
    }
    methods = {
        name: paired_metrics(value, static, truth) for name, value in prediction.items()
    }
    methods["temporal_router"]["paired_stream_95pct_interval_vs_memoryless"] = paired_episode_interval(
        prediction["temporal_router"], prediction["memoryless_router"], truth,
    )
    oracle_gain = methods["static_nominal"]["logical_error"] - methods["mode_oracle"]["logical_error"]
    for name in ("memoryless_router", "temporal_router"):
        gain = methods["static_nominal"]["logical_error"] - methods[name]["logical_error"]
        methods[name]["oracle_gain_recovery"] = float(gain / oracle_gain) if oracle_gain > 0 else None

    null_detectors, null_labels = sample(circuits[0], total, seed + 30)
    null_static = matchings[0].decode_batch(null_detectors)[:, 0].reshape(streams, horizon).astype(bool)
    null_burst = matchings[1].decode_batch(null_detectors)[:, 0].reshape(streams, horizon).astype(bool)
    null_counts = null_detectors.sum(axis=1).reshape(streams, horizon)
    null_probability = count_mode_posterior(
        null_counts, emission, BINARY_TRANSITION, binary_stationary(), temporal=True,
    )
    null_router = np.where(null_probability >= 0.5, null_burst, null_static)
    null_truth = null_labels.reshape(streams, horizon)
    null = paired_metrics(null_router, null_static, null_truth)
    null["static_logical_error"] = float(np.mean(null_static != null_truth))
    return {
        "distance": distance, "rounds": distance, "detectors": detector_count,
        "records": total, "burst_mode_fraction": float(hidden.mean()),
        "calibration_mean_detector_density": [
            float(value[0].mean()) for value in calibration
        ],
        "methods": methods, "nominal_only_null": null,
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
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "protocol": "docs/surface-mode-routing-protocol.md",
        "versions": {"stim": stim.__version__, "pymatching": pymatching.__version__},
        "noise": {
            "nominal_measure_reset_flip": NOMINAL_P,
            "burst_measure_reset_flip": BURST_P,
            "after_clifford_depolarization": CLIFFORD_P,
        },
        "streams": args.streams, "horizon": args.horizon,
        "calibration_shots_per_mode": args.calibration_shots, "seed": args.seed,
        "distances": [
            run_distance(
                distance, streams=args.streams, horizon=args.horizon,
                calibration_shots=args.calibration_shots,
                seed=args.seed + index * 100,
            )
            for index, distance in enumerate(args.distances)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
