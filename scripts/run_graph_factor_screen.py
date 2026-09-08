#!/usr/bin/env python3
"""Test INSERT_FACTOR against a missing correlated-fault positive control."""

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
from ptwm.parity_factor import (  # noqa: E402
    decode_parity_factor,
    decoder_tables,
    generate_factor_records,
)


def paired_interval(candidate: np.ndarray, reference: np.ndarray, labels: np.ndarray) -> list[float]:
    delta = np.mean(
        (candidate != labels).astype(float) - (reference != labels).astype(float), axis=1
    )
    half = stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta)
    return [float(np.mean(delta) - half), float(np.mean(delta) + half)]


def run_arm(
    *, seed: int, episodes: int, horizon: int, base_probability: float,
    true_factor_probability: float, decoder_factor_probability: float,
) -> dict:
    syndromes, labels = generate_factor_records(
        seed=seed, episodes=episodes, horizon=horizon,
        base_probability=base_probability, factor_probability=true_factor_probability,
    )
    tables = decoder_tables(
        base_probability=base_probability, factor_probability=decoder_factor_probability
    )
    predictions = {name: decode_parity_factor(table, syndromes) for name, table in tables.items()}
    base = predictions["base"]
    methods = {}
    for name, prediction in predictions.items():
        methods[name] = {
            "logical_error": float(np.mean(prediction != labels)),
            "prediction_disagreements_vs_base": int(np.sum(prediction != base)),
            "paired_episode_95pct_interval_vs_base": paired_interval(
                prediction, base, labels
            ),
        }
    opportunity = methods["base"]["logical_error"] - methods["insert_correct_factor"]["logical_error"]
    wrong_gain = methods["base"]["logical_error"] - methods["insert_wrong_factor"]["logical_error"]
    return {
        "true_factor_probability": true_factor_probability,
        "decoder_factor_probability": decoder_factor_probability,
        "records": episodes * horizon,
        "methods": methods,
        "correct_factor_gain": opportunity,
        "wrong_factor_gain": wrong_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--base-probability", type=float, default=0.03)
    parser.add_argument("--factor-probability", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "design": {
            "status": "synthetic INSERT_FACTOR positive and null control",
            "trusted_layer": "exact minimum-weight error lookup over a tiny parity-factor model",
            "base_faults": "two disjoint pair faults with zero logical effect",
            "correlated_fault": "one four-detector factor with logical effect one",
        },
        "arms": {
            "factor_present": run_arm(
                seed=args.seed, episodes=args.episodes, horizon=args.horizon,
                base_probability=args.base_probability,
                true_factor_probability=args.factor_probability,
                decoder_factor_probability=args.factor_probability,
            ),
            "factor_absent_null": run_arm(
                seed=args.seed + 1, episodes=args.episodes, horizon=args.horizon,
                base_probability=args.base_probability, true_factor_probability=0.0,
                decoder_factor_probability=args.factor_probability,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
