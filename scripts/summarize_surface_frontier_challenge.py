#!/usr/bin/env python3
"""Aggregate full-refit uncertainty; do not treat nested test streams as new fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, sem, t

from scripts.run_surface_frontier_challenge import CAMPAIGN, CONDITIONS, seed_value


def validate_provenance(rows):
    expected_versions = {"numpy": "2.4.1", "pymatching": "2.4.0", "stim": "1.16.0"}
    expected_config = {"streams": 256, "horizon": 512, "calibration_shots": 65536,
                       "action_streams": 256, "selection_streams": 256, "optimizer_evaluations": 32}
    source_names = {"scripts/run_surface_frontier_challenge.py", "scripts/run_surface_crossfit_compiler.py",
                    "scripts/run_surface_regime_switch.py", "src/ptwm/surface_mode.py", "src/ptwm/surface_program.py"}
    expected_commit = subprocess.check_output(["git", "rev-parse", "be70e3e"], text=True).strip()
    expected_sources = {name: hashlib.sha256(subprocess.check_output(
        ["git", "show", f"{expected_commit}:{name}"])).hexdigest() for name in source_names}
    dataset_keys = {"calibration_0", "calibration_1", "action", "selection"} | {f"test_{c}" for c in CONDITIONS}
    for row in rows:
        if row["campaign"] != CAMPAIGN or row["code_commit"] != expected_commit:
            raise ValueError("artifact is not from the frozen campaign commit")
        if row["versions"] != expected_versions or row["source_sha256"] != expected_sources:
            raise ValueError("source or dependency provenance differs from frozen campaign")
        if any(row["config"][k] != v for k, v in expected_config.items()):
            raise ValueError("artifact does not use the full frozen configuration")
        if set(row["conditions"]) != set(CONDITIONS) or set(row["data_hashes"]) != dataset_keys:
            raise ValueError("condition or data-role manifest is incomplete")
        keys = [("calibration", "stim", mode) for mode in range(2)]
        for role, conditions in (("action", ("nominal",)), ("selection", ("nominal",)), ("test", CONDITIONS)):
            for condition in conditions:
                keys += [(role, condition, "modes"), (role, condition, "stim", 0), (role, condition, "stim", 1)]
        expected_seeds = {json.dumps(key, separators=(",", ":")): seed_value(
            2026090801, row["replicate"], row["distance"], *key) for key in keys}
        if row["seeds"] != expected_seeds or row["root_seed"] != 2026090801:
            raise ValueError("actual sampler seeds differ from declared derivation")
        for condition in row["conditions"].values():
            if condition["records"] != 256 * 512:
                raise ValueError("incorrect evaluation record count")
            for metrics in condition["methods"].values():
                counts = metrics["errors_per_stream"]
                if len(counts) != 256 or any(not 0 <= n <= 512 for n in counts):
                    raise ValueError("invalid stream error counts")
                if sum(counts) / (256 * 512) != metrics["ler"]:
                    raise ValueError("stored LER disagrees with paired error counts")
    return {"commit": expected_commit, "sources": expected_sources, "versions": expected_versions}


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
    provenance = validate_provenance(rows)
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
            c["selective_decode_disagreements"] == 0 and c["compiled_choice_disagreements"] == 0
            for r in rows for c in r["conditions"].values())
        gates["latency"] = "measured separately; cannot infer from batch throughput"
    return {"schema_version": 1, "provenance_verified": provenance,
            "artifact_count": len(rows), "actual_unique_seeds": len(actual_seeds),
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
    result["input_artifact_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    result["summarizer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["summarizer_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["gates"], indent=2))
    for distance, record in result["distances"].items():
        for condition, pairs in record["conditions"].items():
            value = pairs["compiled_vs_static"]
            print(f"d{distance} {condition}: {value['mean_difference'] * 100:+.5f} pp; "
                  f"CI {value['replicate_97_5pct_interval']}")


if __name__ == "__main__":
    main()
