#!/usr/bin/env python3
"""Frozen-program robustness sweep for the persistent-factor control."""

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


OFF_P = 1e-4
FROZEN = {
    0.5: {"hmm_threshold": 0.01, "fsm": (0.0, 0.25, 1.0)},
    1.25: {"hmm_threshold": 0.02, "fsm": (0.5, 0.25, 1.0)},
}


def paired_episode_interval(
    candidate: np.ndarray, reference: np.ndarray, labels: np.ndarray,
) -> list[float]:
    """95% t interval for the candidate-minus-reference episode error rate."""
    difference = np.mean(
        (candidate != labels).astype(float) - (reference != labels).astype(float),
        axis=1,
    )
    mean = float(np.mean(difference))
    if len(difference) < 2:
        return [mean, mean]
    half_width = float(
        stats.t.ppf(0.975, len(difference) - 1)
        * stats.sem(difference)
    )
    return [mean - half_width, mean + half_width]


def evaluate(
    *, seed: int, episodes: int, horizon: int, sigma: float,
    base_probability: float, on_probability: float,
) -> dict:
    data = generate_persistent_factor(
        seed=seed, episodes=episodes, horizon=horizon, sigma=sigma,
        base_probability=base_probability, off_probability=OFF_P,
        on_probability=on_probability,
    )
    config = FROZEN[sigma]
    hmm = binary_belief(
        data["observations"], sigma=sigma, temporal=True,
        past_syndromes=data["syndromes"], base_probability=base_probability,
        off_probability=OFF_P, on_probability=on_probability,
    )[..., 1]
    stationary = np.full(data["labels"].shape, binary_stationary()[1])
    fsm_mode = compiled_hysteretic_ema(
        data["observations"], alpha=config["fsm"][0],
        low=config["fsm"][1], high=config["fsm"][2],
    )
    predictions = {
        "static_mixture": predict_mixture(
            data["syndromes"], stationary, base_probability=base_probability,
            off_probability=OFF_P, on_probability=on_probability,
        ),
        "mode_information_oracle": predict_selected_mode(
            data["syndromes"], data["modes"], base_probability=base_probability,
            off_probability=OFF_P, on_probability=on_probability,
        ),
        "hmm_activate": predict_selected_mode(
            data["syndromes"], hmm, threshold=config["hmm_threshold"],
            base_probability=base_probability, off_probability=OFF_P,
            on_probability=on_probability,
        ),
        "hmm_fork_k2": predict_mixture(
            data["syndromes"], hmm, base_probability=base_probability,
            off_probability=OFF_P, on_probability=on_probability,
        ),
        "compiled_fsm": predict_selected_mode(
            data["syndromes"], fsm_mode, base_probability=base_probability,
            off_probability=OFF_P, on_probability=on_probability,
        ),
    }
    errors = {name: float(np.mean(value != data["labels"])) for name, value in predictions.items()}
    opportunity = errors["static_mixture"] - errors["mode_information_oracle"]
    recovery = {
        name: float((errors["static_mixture"] - value) / opportunity)
        if opportunity > 0 else None
        for name, value in errors.items()
    }
    intervals = {
        name: paired_episode_interval(
            value, predictions["static_mixture"], data["labels"],
        )
        for name, value in predictions.items()
    }
    return {
        "sigma": sigma, "base_probability": base_probability,
        "on_probability": on_probability, "logical_error": errors,
        "mode_oracle_opportunity": opportunity, "mode_oracle_gain_recovery": recovery,
        "paired_episode_95pct_interval_vs_static_mixture": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--base-probabilities", nargs="+", type=float, default=[0.01, 0.03, 0.06])
    parser.add_argument("--on-probabilities", nargs="+", type=float, default=[0.02, 0.04, 0.08, 0.16])
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    for sigma in FROZEN:
        for base in args.base_probabilities:
            for on in args.on_probabilities:
                cells.append(evaluate(
                    seed=args.seed + len(cells), episodes=args.episodes,
                    horizon=args.horizon, sigma=sigma,
                    base_probability=base, on_probability=on,
                ))
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "design": {
            "status": "exploratory frozen decision-rule sweep with oracle-known per-cell noise parameters",
            "frozen_programs": {str(k): v for k, v in FROZEN.items()},
            "selection": "no per-cell threshold or FSM retuning",
            "calibration": "HMM and every decoder likelihood use the true per-cell base and factor probabilities",
        },
        "episodes_per_cell": args.episodes,
        "horizon": args.horizon,
        "records_per_cell": args.episodes * args.horizon,
        "seed": args.seed,
        "off_probability": OFF_P,
        "base_probabilities": args.base_probabilities,
        "on_probabilities": args.on_probabilities,
        "sigmas": list(FROZEN),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
