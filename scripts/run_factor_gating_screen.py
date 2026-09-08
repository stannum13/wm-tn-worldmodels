#!/usr/bin/env python3
"""Compare ACTIVATE_MODE, compiled hysteresis, and local K=2 factor lifting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.factor_gating import (  # noqa: E402
    binary_belief,
    binary_stationary,
    compiled_hysteretic_ema,
    generate_persistent_factor,
    predict_mixture,
    predict_selected_mode,
)


BASE_P, OFF_P, ON_P = 0.03, 1e-4, 0.08


def interval(candidate: np.ndarray, reference: np.ndarray, labels: np.ndarray) -> list[float]:
    delta = np.mean(
        (candidate != labels).astype(float) - (reference != labels).astype(float), axis=1
    )
    half = stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta)
    return [float(delta.mean() - half), float(delta.mean() + half)]


def predict_all(
    data: dict[str, np.ndarray], sigma: float,
    thresholds: dict[str, float], fsm: tuple[float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    syndrome = data["syndromes"]
    memoryless = binary_belief(data["observations"], sigma=sigma, temporal=False)[..., 1]
    hmm = binary_belief(
        data["observations"], sigma=sigma, temporal=True,
        past_syndromes=data["syndromes"],
    )[..., 1]
    stationary = np.full(data["labels"].shape, binary_stationary()[1])
    prediction = {
        "static_mixture": predict_mixture(syndrome, stationary, base_probability=BASE_P, off_probability=OFF_P, on_probability=ON_P),
        "always_off": predict_selected_mode(syndrome, np.zeros_like(hmm), base_probability=BASE_P, off_probability=OFF_P, on_probability=ON_P),
        "always_on": predict_selected_mode(syndrome, np.ones_like(hmm), base_probability=BASE_P, off_probability=OFF_P, on_probability=ON_P),
        "oracle_mode": predict_selected_mode(syndrome, data["modes"], base_probability=BASE_P, off_probability=OFF_P, on_probability=ON_P),
        "memoryless_activate": predict_selected_mode(syndrome, memoryless, threshold=thresholds["memoryless"], base_probability=BASE_P, off_probability=OFF_P, on_probability=ON_P),
        "hmm_activate": predict_selected_mode(syndrome, hmm, threshold=thresholds["hmm"], base_probability=BASE_P, off_probability=OFF_P, on_probability=ON_P),
        "memoryless_fork_k2": predict_mixture(syndrome, memoryless, base_probability=BASE_P, off_probability=OFF_P, on_probability=ON_P),
        "hmm_fork_k2": predict_mixture(syndrome, hmm, base_probability=BASE_P, off_probability=OFF_P, on_probability=ON_P),
    }
    if fsm is not None:
        selected = compiled_hysteretic_ema(
            data["observations"], alpha=fsm[0], low=fsm[1], high=fsm[2]
        )
        prediction["compiled_fsm"] = predict_selected_mode(
            syndrome, selected, base_probability=BASE_P,
            off_probability=OFF_P, on_probability=ON_P,
        )
    return prediction


def select_fsm(data: dict[str, np.ndarray], sigma: float) -> tuple[tuple[float, float, float], dict[str, float]]:
    scores = {}
    for alpha in (0.0, 0.5, 0.8, 0.9, 0.97):
        for low in (0.25, 0.5, 0.75, 1.0):
            for high in (1.0, 1.25, 1.5, 1.75, 2.0):
                if low >= high:
                    continue
                mode = compiled_hysteretic_ema(
                    data["observations"], alpha=alpha, low=low, high=high
                )
                pred = predict_selected_mode(
                    data["syndromes"], mode, base_probability=BASE_P,
                    off_probability=OFF_P, on_probability=ON_P,
                )
                scores[f"{alpha},{low},{high}"] = float(np.mean(pred != data["labels"]))
    best = min(scores, key=scores.get)
    return tuple(map(float, best.split(","))), scores


def select_activation_threshold(
    data: dict[str, np.ndarray], sigma: float, temporal: bool
) -> tuple[float, dict[float, float]]:
    probability = binary_belief(
        data["observations"], sigma=sigma, temporal=temporal,
        past_syndromes=data["syndromes"] if temporal else None,
    )[..., 1]
    scores = {}
    for threshold in (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90):
        prediction = predict_selected_mode(
            data["syndromes"], probability, threshold=threshold,
            base_probability=BASE_P, off_probability=OFF_P, on_probability=ON_P,
        )
        scores[threshold] = float(np.mean(prediction != data["labels"]))
    return min(scores, key=scores.get), scores


def run_arm(seed: int, episodes: int, horizon: int, sigma: float) -> dict:
    validation = generate_persistent_factor(
        seed=seed, episodes=max(64, episodes // 4), horizon=horizon, sigma=sigma
    )
    test = generate_persistent_factor(
        seed=seed + 1, episodes=episodes, horizon=horizon, sigma=sigma
    )
    fsm, scores = select_fsm(validation, sigma)
    memoryless_threshold, memoryless_scores = select_activation_threshold(
        validation, sigma, temporal=False
    )
    hmm_threshold, hmm_scores = select_activation_threshold(
        validation, sigma, temporal=True
    )
    thresholds = {"memoryless": memoryless_threshold, "hmm": hmm_threshold}
    predictions = predict_all(test, sigma, thresholds, fsm)
    reference = predictions["static_mixture"]
    methods = {}
    for name, prediction in predictions.items():
        methods[name] = {
            "logical_error": float(np.mean(prediction != test["labels"])),
            "paired_episode_95pct_interval_vs_static_mixture": interval(
                prediction, reference, test["labels"]
            ),
            "disagreements_vs_static_mixture": int(np.sum(prediction != reference)),
        }
        methods[name]["paired_episode_95pct_interval_vs_memoryless_activate"] = interval(
            prediction, predictions["memoryless_activate"], test["labels"]
        )
    methods["hmm_fork_k2"]["paired_episode_95pct_interval_vs_hmm_activate"] = interval(
        predictions["hmm_fork_k2"], predictions["hmm_activate"], test["labels"]
    )
    oracle_gain = methods["static_mixture"]["logical_error"] - methods["oracle_mode"]["logical_error"]
    for name in ("memoryless_activate", "hmm_activate", "memoryless_fork_k2", "hmm_fork_k2", "compiled_fsm"):
        gain = methods["static_mixture"]["logical_error"] - methods[name]["logical_error"]
        methods[name]["oracle_gain_recovery"] = float(gain / oracle_gain) if oracle_gain > 0 else None
    methods["hmm_fork_k2"]["paired_episode_95pct_interval_vs_memoryless_fork_k2"] = interval(
        predictions["hmm_fork_k2"], predictions["memoryless_fork_k2"], test["labels"]
    )
    methods["compiled_fsm"]["paired_episode_95pct_interval_vs_hmm_activate"] = interval(
        predictions["compiled_fsm"], predictions["hmm_activate"], test["labels"]
    )
    hmm_activate_gain = methods["static_mixture"]["logical_error"] - methods["hmm_activate"]["logical_error"]
    hmm_fork_gain = methods["static_mixture"]["logical_error"] - methods["hmm_fork_k2"]["logical_error"]
    fsm_gain = methods["static_mixture"]["logical_error"] - methods["compiled_fsm"]["logical_error"]
    methods["compiled_fsm"]["hmm_activate_gain_recovery"] = (
        float(fsm_gain / hmm_activate_gain) if hmm_activate_gain > 0 else None
    )
    methods["compiled_fsm"]["hmm_fork_gain_recovery"] = (
        float(fsm_gain / hmm_fork_gain) if hmm_fork_gain > 0 else None
    )

    # Freeze the active-arm selections and apply them where the high-rate factor mode
    # is physically absent. This is the stationary false-activation control.
    null = generate_persistent_factor(
        seed=seed + 2, episodes=episodes, horizon=horizon, sigma=sigma,
        off_probability=OFF_P, on_probability=OFF_P,
    )
    null_predictions = predict_all(null, sigma, thresholds, fsm)
    null_reference = null_predictions["always_off"]
    null_methods = {
        name: {
            "logical_error": float(np.mean(prediction != null["labels"])),
            "paired_episode_95pct_interval_vs_always_off": interval(
                prediction, null_reference, null["labels"]
            ),
        }
        for name, prediction in null_predictions.items()
    }
    return {
        "sigma": sigma, "records": episodes * horizon,
        "selected_fsm": {"alpha": fsm[0], "low": fsm[1], "high": fsm[2]},
        "selected_activation_thresholds": thresholds,
        "validation_candidates": len(scores), "activation_threshold_candidates": 9,
        "validation_best_ler": scores[",".join(map(str, fsm))],
        "validation_activation_ler": {
            "memoryless": {str(k): v for k, v in memoryless_scores.items()},
            "hmm": {str(k): v for k, v in hmm_scores.items()},
        },
        "methods": methods,
        "high_rate_factor_absent_null": null_methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 1.25])
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "design": {
            "status": "synthetic persistent-factor mode control",
            "operations": ["ACTIVATE_MODE", "FORK"],
            "filter": "privileged known-parameter binary HMM",
            "compiler": "grid search EMA+hysteresis on independent validation episodes",
            "null": "factor rate fixed at background 1e-4; latent mode is irrelevant",
        },
        "arms": [
            run_arm(args.seed + index * 10, args.episodes, args.horizon, sigma)
            for index, sigma in enumerate(args.sigmas)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
