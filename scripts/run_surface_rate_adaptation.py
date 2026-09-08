#!/usr/bin/env python3
"""Eight-state nuisance filter, with nominal-only selection and fresh shift tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.surface_mode import affine_log_likelihood_ratio, log_likelihood_mode_posterior
from ptwm.surface_program import LocalFeatures
from ptwm.switching_hypotheses import switching_hypothesis_filter
from scripts.benchmark_surface_frontier import restore_model, restore_static
from scripts.run_surface_frontier_challenge import (
    REGIMES, STATIONARY, TRANSITION, Seeds, circuit, data_hash, make_decoder, stream,
)
from scripts.summarize_surface_frontier_challenge import interval

CONDITIONS = ("nominal", "slow", "fast", "iid", "within_shot", "stationary_a", "stationary_b")


def probabilities(llr):
    adaptive, effective_rate = switching_hypothesis_filter(llr)
    return {"adaptive": adaptive,
            "fixed": log_likelihood_mode_posterior(llr, TRANSITION, STATIONARY, temporal=True),
            "memoryless": log_likelihood_mode_posterior(llr, TRANSITION, STATIONARY, temporal=False)}, effective_rate


def run(artifact, *, root_seed, streams, horizon):
    seeds = Seeds(root_seed, artifact["replicate"], 5)
    mapper = LocalFeatures.from_circuit(circuit(5, *REGIMES[0]))
    evidence = restore_model(artifact["models"]["evidence"])
    static = restore_static(artifact, artifact["selected"]["static"])
    endpoints = [make_decoder("endpoint", circuit(5, *r).detector_error_model(decompose_errors=True), True, {})
                 for r in REGIMES]
    thresholds, scores, hashes, results = {}, {}, {}, {}
    for role, conditions in (("rate_selection", ("nominal",)), ("rate_test", CONDITIONS)):
        for condition in conditions:
            d, y, modes = stream(5, seeds=seeds, role=role, condition=condition, streams=streams, horizon=horizon)
            hashes[f"{role}_{condition}"] = data_hash(d, y, modes)
            features = mapper.transform(d)
            llr = affine_log_likelihood_ratio(features, evidence).reshape(streams, horizon)
            values, rate = probabilities(llr)
            decoded = [decoder.decode(d) for decoder in endpoints]
            if role == "rate_selection":
                for name, p in values.items():
                    scores[name] = {str(i / 10): float(np.mean(np.where(p.ravel() >= i / 10, decoded[1], decoded[0]) != y))
                                    for i in range(1, 10)}
                    thresholds[name] = float(min(scores[name], key=lambda x: (scores[name][x], float(x))))
                continue
            predictions = {name: np.where(p.ravel() >= thresholds[name], decoded[1], decoded[0])
                           for name, p in values.items()}
            predictions["static"] = static.decode(d)
            errors = {name: (p != y).reshape(streams, horizon).sum(axis=1) for name, p in predictions.items()}
            results[condition] = {"methods": {name: {"ler": float(counts.mean() / horizon), "errors_per_stream": counts.tolist()}
                                               for name, counts in errors.items()},
                                  "posterior_mean_rate": float(rate.mean()),
                                  "posterior_last_half_mean_rate": float(rate[:, horizon // 2:].mean()),
                                  "actual_switch_probability": float(np.mean(modes[:, 1:] != modes[:, :-1]))}
            print(f"r{artifact['replicate']} {condition}: " + ", ".join(f"{k}={v['ler']:.6f}"
                  for k, v in results[condition]["methods"].items()), flush=True)
    return {"replicate": artifact["replicate"], "thresholds": thresholds, "selection_scores": scores,
            "seeds": seeds.manifest, "data_hashes": hashes, "conditions": results}


def summarize(rows):
    output = {}
    for condition in CONDITIONS:
        comparisons = {}
        for reference in ("fixed", "memoryless", "static"):
            candidates = [r["conditions"][condition]["methods"]["adaptive"]["ler"] for r in rows]
            references = [r["conditions"][condition]["methods"][reference]["ler"] for r in rows]
            deltas = np.asarray(candidates) - references
            comparisons[f"adaptive_vs_{reference}"] = {"adaptive_ler": float(np.mean(candidates)),
                "reference_ler": float(np.mean(references)), "mean_difference": float(deltas.mean()),
                "replicate_differences": deltas.tolist(), "interval_95pct": interval(deltas, .95),
                "interval_97_5pct": interval(deltas, .975)}
        output[condition] = comparisons
    gates = {"complete": len(rows) == 10}
    if len(rows) == 10:
        gates["nominal_retention"] = output["nominal"]["adaptive_vs_fixed"]["interval_97_5pct"][1] <= .0002
        gates["nominal_static_win"] = output["nominal"]["adaptive_vs_static"]["interval_97_5pct"][1] < 0
        gates["iid_fixed_win"] = output["iid"]["adaptive_vs_fixed"]["interval_97_5pct"][1] < 0
        gates["iid_memoryless_safety"] = output["iid"]["adaptive_vs_memoryless"]["interval_97_5pct"][1] <= .0002
    return {"conditions": output, "gates": gates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026090804)
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=512)
    args = parser.parse_args()
    output = {"protocol": "docs/surface-rate-adaptation-protocol.md", "root_seed": args.seed,
              "streams": args.streams, "horizon": args.horizon,
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "records": []}
    for path in sorted(args.input_dir.glob("r*_d5.json")):
        result = run(json.loads(path.read_text()), root_seed=args.seed, streams=args.streams, horizon=args.horizon)
        result["source_artifact"] = str(path)
        result["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        output["records"].append(result)
        output["summary"] = summarize(output["records"])
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"]["gates"], indent=2))


if __name__ == "__main__":
    main()
