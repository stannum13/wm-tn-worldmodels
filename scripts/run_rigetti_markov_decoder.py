#!/usr/bin/env python3
"""Causal class-conditional syndrome world-model benchmark on Ankaa-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    block_error_differences,
    block_score_differences,
    detector_round_symbols,
    fit_markov_syndrome_decoder,
    load_qec_record,
    probability_metrics,
    timed_markov_prediction,
)


def _interval(values: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def run(path: Path, circuit_group: str, block_size: int) -> dict:
    record = load_qec_record(str(path), circuit_group)
    symbols = detector_round_symbols(record["detectors"], record["circuit"])
    labels = record["observables"][:, 0].astype(float)
    boundary = int(0.6 * len(labels))
    train, test = np.arange(boundary), np.arange(boundary, len(labels))
    rows, predictions = [], {}
    for order in (0, 1, 2):
        decoder = fit_markov_syndrome_decoder(symbols[train], labels[train], order=order)
        prediction, timing = timed_markov_prediction(decoder, symbols[test])
        name = f"markov_order_{order}"
        predictions[name] = prediction
        free_parameters = 1 + sum(
            2 * decoder.alphabet**context_order * (decoder.alphabet - 1)
            for context_order in range(order + 1)
        )
        rows.append({
            "model": name,
            "parameters": free_parameters,
            **probability_metrics(prediction, labels[test]),
            **timing,
        })
    comparisons = []
    for first, second in (("markov_order_0", "markov_order_1"), ("markov_order_1", "markov_order_2")):
        error = block_error_differences(predictions[first], predictions[second], labels[test], block_size=block_size)
        brier = block_score_differences(predictions[first], predictions[second], labels[test], block_size=block_size, score="brier_loss")
        comparisons.append({
            "first": first,
            "second": second,
            "mean_error_reduction": float(np.mean(error)),
            "error_reduction_paired_95pct_t_interval": _interval(error),
            "mean_brier_reduction": float(np.mean(brier)),
            "brier_reduction_paired_95pct_t_interval": _interval(brier),
            "blocks": len(error),
        })
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "data": {
            "source": "https://zenodo.org/records/15364358",
            "file": path.name,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "circuit_group": circuit_group,
            "shots": len(labels),
            "train_shots": len(train),
            "chronological_test_shots": len(test),
            "independent_sessions": 1,
        },
        "design": {
            "rounds": symbols.shape[1],
            "detectors_per_round": 4,
            "alphabet": 16,
            "released_mwpm_error_at_27_rounds": 0.38819,
            "primary_go_condition": "close >=20% of 42.395%-to-38.819% error gap with positive paired block interval",
            "split": "first 60% chronological shots train; final 40% locked test",
            "mechanism": "causal class-conditional syndrome likelihood updated by one table lookup per round",
        },
        "models": rows,
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/rigetti_markov_decoder.json"))
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.block_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"models": payload["models"], "comparisons": payload["comparisons"]}, indent=2))
