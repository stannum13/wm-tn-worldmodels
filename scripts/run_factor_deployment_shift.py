#!/usr/bin/env python3
"""Paired deployment-shift test of frozen and cheaply recalibrated graph controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.factor_gating import (  # noqa: E402
    binary_belief,
    binary_stationary,
    compiled_hysteretic_ema,
    generate_persistent_factor,
    joint_syndrome_logical,
    predict_mixture,
    predict_selected_mode,
)
from scripts.run_factor_parameter_sweep import paired_episode_interval  # noqa: E402


NOMINAL_BASE, OFF_P, NOMINAL_ON = 0.03, 1e-4, 0.08
BASE_GRID = (0.01, 0.03, 0.06)
ON_GRID = (0.02, 0.04, 0.08, 0.16)
FROZEN = {
    0.5: {"hmm_threshold": 0.01, "fsm": (0.0, 0.25, 1.0)},
    1.25: {"hmm_threshold": 0.02, "fsm": (0.5, 0.25, 1.0)},
}


def estimate_rates(data: dict[str, np.ndarray], sigma: float) -> tuple[float, float]:
    """Grid MLE using syndromes and observation-only soft mode assignments."""
    probability_on = binary_belief(
        data["observations"], sigma=sigma, temporal=True,
    )[..., 1].ravel()
    keys = (data["syndromes"] @ (1 << np.arange(4))).ravel()
    best = None
    for base in BASE_GRID:
        off = joint_syndrome_logical(
            base_probability=base, factor_probability=OFF_P,
        ).sum(axis=1)
        for on in ON_GRID:
            active = joint_syndrome_logical(
                base_probability=base, factor_probability=on,
            ).sum(axis=1)
            likelihood = (1.0 - probability_on) * off[keys] + probability_on * active[keys]
            score = float(np.log(np.clip(likelihood, 1e-300, None)).sum())
            if best is None or score > best[0]:
                best = (score, base, on)
    assert best is not None
    return float(best[1]), float(best[2])


def predictions(
    data: dict[str, np.ndarray], *, sigma: float,
    assumed_base: float, assumed_on: float,
) -> dict[str, np.ndarray]:
    config = FROZEN[sigma]
    belief = binary_belief(
        data["observations"], sigma=sigma, temporal=True,
        past_syndromes=data["syndromes"], base_probability=assumed_base,
        off_probability=OFF_P, on_probability=assumed_on,
    )[..., 1]
    stationary = np.full(data["labels"].shape, binary_stationary()[1])
    fsm = compiled_hysteretic_ema(
        data["observations"], alpha=config["fsm"][0],
        low=config["fsm"][1], high=config["fsm"][2],
    )
    return {
        "static": predict_mixture(
            data["syndromes"], stationary, base_probability=assumed_base,
            off_probability=OFF_P, on_probability=assumed_on,
        ),
        "activate": predict_selected_mode(
            data["syndromes"], belief, threshold=config["hmm_threshold"],
            base_probability=assumed_base, off_probability=OFF_P,
            on_probability=assumed_on,
        ),
        "fork_k2": predict_mixture(
            data["syndromes"], belief, base_probability=assumed_base,
            off_probability=OFF_P, on_probability=assumed_on,
        ),
        "fsm": predict_selected_mode(
            data["syndromes"], fsm, base_probability=assumed_base,
            off_probability=OFF_P, on_probability=assumed_on,
        ),
    }


def evaluate_cell(
    *, seed: int, calibration_seed: int, episodes: int,
    calibration_episodes: int, horizon: int, sigma: float,
    true_base: float, true_on: float,
) -> dict:
    test = generate_persistent_factor(
        seed=seed, episodes=episodes, horizon=horizon, sigma=sigma,
        base_probability=true_base, off_probability=OFF_P,
        on_probability=true_on,
    )
    calibration = generate_persistent_factor(
        seed=calibration_seed, episodes=calibration_episodes, horizon=horizon,
        sigma=sigma, base_probability=true_base, off_probability=OFF_P,
        on_probability=true_on,
    )
    estimated_base, estimated_on = estimate_rates(calibration, sigma)
    arms = {
        "frozen": predictions(
            test, sigma=sigma, assumed_base=NOMINAL_BASE, assumed_on=NOMINAL_ON,
        ),
        "estimated": predictions(
            test, sigma=sigma, assumed_base=estimated_base, assumed_on=estimated_on,
        ),
        "calibrated_oracle": predictions(
            test, sigma=sigma, assumed_base=true_base, assumed_on=true_on,
        ),
    }
    labels = test["labels"]
    reference = arms["frozen"]["static"]
    output = {}
    for arm_name, methods in arms.items():
        output[arm_name] = {
            name: {
                "logical_error": float(np.mean(value != labels)),
                "paired_episode_95pct_interval_vs_frozen_static":
                    paired_episode_interval(value, reference, labels),
                "disagreements_vs_frozen_static": int(np.sum(value != reference)),
            }
            for name, value in methods.items()
        }
    for name in ("activate", "fork_k2", "fsm"):
        frozen = arms["frozen"][name]
        oracle = arms["calibrated_oracle"][name]
        estimated = arms["estimated"][name]
        output["calibrated_oracle"][name]["paired_episode_95pct_interval_vs_frozen_same_operation"] = paired_episode_interval(oracle, frozen, labels)
        output["estimated"][name]["paired_episode_95pct_interval_vs_frozen_same_operation"] = paired_episode_interval(estimated, frozen, labels)
        opportunity = float(np.mean(frozen != labels) - np.mean(oracle != labels))
        output["estimated"][name]["oracle_recalibration_gain_recovery"] = (
            float((np.mean(frozen != labels) - np.mean(estimated != labels)) / opportunity)
            if opportunity > 0 else None
        )
    return {
        "sigma": sigma, "true_base_probability": true_base,
        "true_on_probability": true_on,
        "estimated_base_probability": estimated_base,
        "estimated_on_probability": estimated_on,
        "arms": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--calibration-episodes", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--calibration-seed", type=int, default=20261908)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "protocol": "docs/factor-deployment-shift-protocol.md",
        "episodes_per_cell": args.episodes,
        "calibration_episodes_per_cell": args.calibration_episodes,
        "horizon": args.horizon,
        "test_seed": args.seed,
        "calibration_seed": args.calibration_seed,
        "common_random_numbers_across_cells": True,
        "nominal": {"base_probability": NOMINAL_BASE, "on_probability": NOMINAL_ON},
        "cells": [
            evaluate_cell(
                seed=args.seed, calibration_seed=args.calibration_seed,
                episodes=args.episodes, calibration_episodes=args.calibration_episodes,
                horizon=args.horizon, sigma=sigma, true_base=base, true_on=on,
            )
            for sigma in FROZEN for base in BASE_GRID for on in ON_GRID
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
