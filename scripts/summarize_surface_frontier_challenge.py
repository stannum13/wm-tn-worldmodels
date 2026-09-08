#!/usr/bin/env python3
"""Aggregate full-refit uncertainty; do not treat nested test streams as new fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, sem, t


def interval(values, confidence):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return None
    half = float(t.ppf((1 + confidence) / 2, len(values) - 1) * sem(values))
    return [mean - half, mean + half]


def comparison(rows, condition, candidate, reference):
    differences, candidate_ler, reference_ler = [], [], []
    for row in rows:
        methods = row["conditions"][condition]["methods"]
        candidate_ler.append(methods[candidate]["ler"])
        reference_ler.append(methods[reference]["ler"])
        differences.append(methods[candidate]["ler"] - methods[reference]["ler"])
    wins = int(np.sum(np.asarray(differences) < 0))
    return {"candidate": candidate, "reference": reference, "replicates": len(rows),
            "candidate_ler": float(np.mean(candidate_ler)), "reference_ler": float(np.mean(reference_ler)),
            "mean_difference": float(np.mean(differences)), "replicate_differences": differences,
            "improving_replicates": wins,
            "one_sided_sign_p": float(binomtest(wins, len(rows), .5, alternative="greater").pvalue),
            "replicate_95pct_interval": interval(differences, .95),
            "replicate_97_5pct_interval": interval(differences, .975),
            "replicate_98_75pct_interval": interval(differences, .9875),
            "replicate_99_5pct_interval": interval(differences, .995)}


def summarize(rows):
    identities = [(r["replicate"], r["distance"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate replicate/distance artifact")
    actual_seeds = [seed for row in rows for seed in row["seeds"].values()]
    if len(actual_seeds) != len(set(actual_seeds)):
        raise ValueError("sampler seed collision across campaign")
    hashes = [digest for row in rows for digest in row["data_hashes"].values()]
    if len(hashes) != len(set(hashes)):
        raise ValueError("identical datasets across distinct data roles")
    comparisons = [("compiled", "static"), ("temporal", "memoryless"),
                   ("temporal", "matched_memoryless"), ("correlated_compiled", "correlated_static"),
                   ("one_decode", "static"), ("correlated_temporal", "correlated_static"),
                   ("correlated_temporal", "correlated_memoryless"),
                   ("compiled", "old_grid"), ("compiled", "mode_informed_endpoint"),
                   ("one_decode", "mode_informed_endpoint")]
    distances = {}
    for distance in sorted({r["distance"] for r in rows}):
        subset = sorted((r for r in rows if r["distance"] == distance), key=lambda r: r["replicate"])
        common = set.intersection(*(set(r["conditions"]) for r in subset))
        distances[str(distance)] = {
            "refits": len(subset),
            "selected_policies": {str(r["replicate"]): r["selected"] for r in subset},
            "conditions": {condition: {f"{a}_vs_{b}": comparison(subset, condition, a, b)
                                       for a, b in comparisons} for condition in sorted(common)},
        }
    complete = (set(identities) == {(r, d) for r in range(10) for d in (3, 5)}
                and all(len(r["conditions"]) == 10 and r["config"]["streams"] == 256
                        and r["config"]["horizon"] == 512 and r["config"]["calibration_shots"] == 65536
                        and r["config"]["action_streams"] == 256
                        and r["config"]["selection_streams"] == 256
                        and r["root_seed"] == 2026090801 for r in rows))
    gates = {"complete_protocol": complete}
    if complete:
        nominal = [distances[str(d)]["conditions"]["nominal"] for d in (3, 5)]
        gates["stronger_benchmark"] = all(
            value["compiled_vs_static"]["replicate_97_5pct_interval"][1] < -0.0002
            and value["compiled_vs_static"]["improving_replicates"] == 10 for value in nominal)
        gates["temporal_evidence"] = all(
            value[key]["replicate_97_5pct_interval"][1] < 0 for value in nominal
            for key in ("temporal_vs_memoryless", "temporal_vs_matched_memoryless"))
        gates["correlated_benchmark"] = all(
            value["correlated_compiled_vs_correlated_static"]["replicate_97_5pct_interval"][1] < 0
            for value in nominal)
        gates["one_decode_accuracy"] = all(
            value["one_decode_vs_static"]["replicate_97_5pct_interval"][1] < -.0002
            and value["one_decode_vs_static"]["improving_replicates"] == 10 for value in nominal)
        transfer = [distances[str(d)]["conditions"][c]["compiled_vs_static"] for d in (3, 5)
                    for c in ("scale_075", "scale_125", "less_separated", "slow", "fast")]
        transfer_wins = sum(x["replicate_99_5pct_interval"][1] < 0 for x in transfer)
        gates["transfer"] = bool(all(x["replicate_99_5pct_interval"][1] < .0005 for x in transfer)
                                  and transfer_wins >= 6)
        gates["transfer_cells_with_resolved_gain"] = int(transfer_wins)
        gates["stationary"] = all(
            distances[str(d)]["conditions"][c]["compiled_vs_mode_informed_endpoint"][
                "replicate_98_75pct_interval"][1] < .0002
            for d in (3, 5) for c in ("stationary_a", "stationary_b"))
        gates["iid_temporal_safety"] = all(
            distances[str(d)]["conditions"]["iid"]["temporal_vs_matched_memoryless"][
                "replicate_97_5pct_interval"][1] < .0002 for d in (3, 5))
        gates["selective_decisions_exact"] = all(
            c["selective_decode_disagreements"] == 0 for r in rows for c in r["conditions"].values())
        gates["latency"] = "measured separately; cannot infer from batch throughput"
    return {"schema_version": 1, "artifact_count": len(rows), "actual_unique_seeds": len(actual_seeds),
            "actual_unique_data_hashes": len(hashes), "code_commits": sorted({r["code_commit"] for r in rows}),
            "test_records": sum(c["records"] for r in rows for c in r["conditions"].values()),
            "gates": gates, "distances": distances}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("r*_d*.json"))
    if not files:
        raise ValueError("no replicate artifacts found")
    result = summarize([json.loads(path.read_text()) for path in files])
    result["input_artifacts"] = [str(path) for path in files]
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["gates"], indent=2))
    for distance, record in result["distances"].items():
        for condition, pairs in record["conditions"].items():
            value = pairs["compiled_vs_static"]
            print(f"d{distance} {condition}: {value['mean_difference'] * 100:+.5f} pp; "
                  f"CI {value['replicate_97_5pct_interval']}")


if __name__ == "__main__":
    main()
