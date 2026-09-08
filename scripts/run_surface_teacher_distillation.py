#!/usr/bin/env python3
"""Compare equal-capacity residuals fitted to outcomes and exact conditional risks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.exact_dem import exact_distribution, syndrome_indices
from ptwm.soft_action import fit_soft_action, spline_features, spline_knots, teacher_logical_probability
from ptwm.surface_mode import affine_log_likelihood_ratio
from ptwm.surface_program import LocalFeatures
from scripts.benchmark_surface_frontier import restore_model, restore_static
from scripts.run_surface_crossfit_compiler import energy_action_features
from scripts.run_surface_frontier_challenge import (
    L2_GRID, REGIMES, Seeds, circuit, data_hash, decode_endpoints, json_model, make_decoder, probability, stream,
)
from scripts.summarize_surface_frontier_challenge import interval


def data(artifact, tables, seeds, role, *, streams, horizon):
    d, y, modes = stream(3, seeds=seeds, role=role, condition="nominal", streams=streams, horizon=horizon)
    feature_map = LocalFeatures.from_circuit(circuit(3, *REGIMES[0]))
    evidence = restore_model(artifact["models"]["evidence"])
    features = feature_map.transform(d)
    cheap_probability = probability(features, evidence, streams, horizon, True)
    endpoints = [make_decoder("endpoint", circuit(3, *r).detector_error_model(decompose_errors=True), True, {})
                 for r in REGIMES]
    predictions, weights = decode_endpoints(endpoints, d)
    features_action = energy_action_features(features, cheap_probability, weights)
    oracle_probability = teacher_logical_probability(tables, syndrome_indices(d).reshape(streams, horizon))
    return d, y, modes, features_action, predictions, oracle_probability.ravel()


def run(artifact, tables, args):
    seeds = Seeds(args.seed, artifact["replicate"], 3)
    d, y, modes, features, endpoints, logical_probability = data(
        artifact, tables, seeds, "distill_action", streams=args.streams, horizon=args.horizon)
    hashes = {"action": data_hash(d, y, modes)}
    disagree = endpoints[0] != endpoints[1]
    matrix = features[disagree]
    targets = {"outcome": (endpoints[1][disagree] == y[disagree]).astype(float),
               "teacher": np.where(endpoints[1][disagree], logical_probability[disagree], 1 - logical_probability[disagree])}
    knots = spline_knots(matrix)
    models = {}
    for family, target in targets.items():
        for grammar in ("affine", "spline"):
            x = matrix if grammar == "affine" else spline_features(matrix, knots)
            for l2 in L2_GRID:
                models[f"{family}:{grammar}:{l2}"] = fit_soft_action(x, target, l2=l2)
    training = {"records": int(y.size), "disagreements": int(disagree.sum()),
                "affine_inputs": features.shape[1], "spline_inputs": features.shape[1] + 20}
    selection_scores, selected, output = {}, {}, {}
    for role in ("distill_selection", "distill_test"):
        d, y, modes, features, endpoints, logical_probability = data(
            artifact, tables, seeds, role, streams=args.streams, horizon=args.horizon)
        hashes[role] = data_hash(d, y, modes)
        matrices = {"affine": features, "spline": spline_features(features, knots)}
        predictions = {}
        for name, model in models.items():
            if role == "distill_test" and name not in selected.values():
                continue
            score = affine_log_likelihood_ratio(matrices[name.split(":")[1]], model)
            predictions[name] = np.where(score >= 0, endpoints[1], endpoints[0])
        if role == "distill_selection":
            selection_scores = {name: float(np.mean(p != y)) for name, p in predictions.items()}
            for family in targets:
                selected[family] = min((n for n in selection_scores if n.startswith(family + ":")),
                                       key=lambda n: (selection_scores[n], ":spline:" in n, n))
            continue
        final = {family: predictions[name] for family, name in selected.items()}
        final["static"] = restore_static(artifact, artifact["selected"]["static"]).decode(d)
        final["exact_teacher"] = logical_probability > .5
        final["restricted_teacher"] = np.where(endpoints[0] != endpoints[1], final["exact_teacher"], endpoints[0])
        for name, prediction in final.items():
            errors = (prediction != y).reshape(args.streams, args.horizon).sum(axis=1)
            output[name] = {"ler": float(errors.mean() / args.horizon), "errors_per_stream": errors.tolist()}
    print(f"r{artifact['replicate']} {selected}: " + ", ".join(f"{n}={v['ler']:.6f}" for n, v in output.items()), flush=True)
    return {"replicate": artifact["replicate"], "selected": selected, "selection_scores": selection_scores,
            "training": training, "seeds": seeds.manifest, "data_hashes": hashes, "methods": output,
            "spline_knots": knots.tolist(), "models": {k: json_model(v) for k, v in models.items()}}


def summarize(rows):
    names = ("teacher", "outcome", "static", "restricted_teacher", "exact_teacher")
    means = {name: float(np.mean([r["methods"][name]["ler"] for r in rows])) for name in names}
    differences = np.asarray([r["methods"]["teacher"]["ler"] - r["methods"]["outcome"]["ler"] for r in rows])
    denominator = means["outcome"] - means["restricted_teacher"]
    recovery = (means["outcome"] - means["teacher"]) / denominator if denominator > 0 else None
    bootstrap = []
    rng = np.random.default_rng(80205)
    for _ in range(10000):
        chosen = rng.integers(len(rows), size=len(rows))
        risk = {n: np.mean([rows[i]["methods"][n]["ler"] for i in chosen]) for n in names[:2] + ("restricted_teacher",)}
        gap = risk["outcome"] - risk["restricted_teacher"]
        if gap > 0:
            bootstrap.append((risk["outcome"] - risk["teacher"]) / gap)
    full = {r["replicate"] for r in rows} == set(range(10)) and len(rows) == 10
    ci = interval(differences, .95)
    return {"mean_ler": means, "teacher_minus_outcome": float(differences.mean()),
            "replicate_differences": differences.tolist(), "replicate_95pct_interval": ci,
            "improving_replicates": int(np.sum(differences < 0)), "restricted_teacher_gain_recovery": recovery,
            "recovery_bootstrap_95pct_interval": np.quantile(bootstrap, [.025, .975]).tolist() if bootstrap else None,
            "bootstrap_nonpositive_denominators": 10000 - len(bootstrap),
            "gates": {"complete": full, "teacher_training_advantage": bool(full and ci[1] < -.0001
                                                                       and np.sum(differences < 0) >= 8),
                      "gain_recovery_80pct": bool(full and recovery is not None and recovery >= .8)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026090805)
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=512)
    args = parser.parse_args()
    output = {"protocol": "docs/surface-teacher-distillation-protocol.md", "root_seed": args.seed,
              "streams": args.streams, "horizon": args.horizon, "code_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(), "records": []}
    tables = []
    for rates in REGIMES:
        c = circuit(3, *rates)
        distribution, diagnostic = exact_distribution(c.detector_error_model())
        tables.append(distribution.reshape(2, 1 << c.num_detectors))
    tables = np.stack(tables)
    for path in sorted(args.input_dir.glob("r*_d3.json")):
        result = run(json.loads(path.read_text()), tables, args)
        result["source_artifact"] = str(path)
        result["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        output["records"].append(result)
        output["summary"] = summarize(output["records"])
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
