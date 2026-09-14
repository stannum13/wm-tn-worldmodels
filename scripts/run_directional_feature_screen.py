#!/usr/bin/env python3
"""Development-only feature sufficiency screen with frozen endpoint actions."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.directional_features import connected_motifs, parity_features, feature_dictionary, select_columns
from ptwm.exact_dem import exact_distribution, syndrome_indices
from ptwm.soft_action import fit_soft_action, teacher_logical_probability
from ptwm.surface_mode import affine_log_likelihood_ratio
from ptwm.surface_program import LocalFeatures
from scripts.benchmark_surface_frontier import restore_model, restore_static
from scripts.run_surface_crossfit_compiler import energy_action_features
from scripts.run_surface_frontier_challenge import (
    REGIMES, L2_GRID, Seeds, circuit, data_hash, decode_endpoints, json_model,
    make_decoder, probability, stream,
)


FAMILIES = ("base", "parity", "interaction", "both", "uncapped_base")


def make_data(artifact, tables, seeds, role, args, mapper, motifs, endpoints):
    d, y, modes = stream(3, seeds=seeds, role="feature_screen_" + role,
                        condition="nominal", streams=args.streams, horizon=args.horizon)
    raw = mapper.transform(d)
    p = probability(raw, restore_model(artifact["models"]["evidence"]), args.streams, args.horizon, True).ravel()
    predictions, weights = decode_endpoints(endpoints, d)
    disagreement = predictions[0] != predictions[1]
    base = energy_action_features(raw[disagreement], p[disagreement], [w[disagreement] for w in weights])
    motif = parity_features(d[disagreement], motifs)
    q = teacher_logical_probability(tables, syndrome_indices(d).reshape(args.streams, args.horizon)).ravel()
    return {"detectors": d, "labels": y, "hash": data_hash(d, y, modes), "p": p[disagreement],
            "base": base, "motif": motif, "q": q, "predictions": predictions, "disagreement": disagreement}


def matrix(data, family):
    return feature_dictionary(data["base"], data["motif"], data["p"],
                              "base" if family == "uncapped_base" else family)


def run(artifact, tables, args):
    seeds = Seeds(args.seed, artifact["replicate"], 3)
    mapper = LocalFeatures.from_circuit(circuit(3, *REGIMES[0]))
    motifs = connected_motifs(len(mapper.detector_times), mapper.pairs)
    endpoints = [make_decoder("endpoint", circuit(3, *r).detector_error_model(decompose_errors=True), True, {})
                 for r in REGIMES]
    fit = make_data(artifact, tables, seeds, "fit", args, mapper, motifs, endpoints)
    hashes = {"fit": fit["hash"]}
    mask = fit["disagreement"]
    outcome = (fit["predictions"][1][mask] == fit["labels"][mask]).astype(float)
    teacher = np.where(fit["predictions"][1][mask], fit["q"][mask], 1 - fit["q"][mask])
    models, columns, widths = {}, {}, {}
    for family in FAMILIES:
        x = matrix(fit, family)
        columns[family] = (np.arange(x.shape[1]) if family == "uncapped_base"
                           else select_columns(x, outcome, args.feature_budget))
        widths[family] = {"dictionary": x.shape[1], "selected": len(columns[family])}
        for target_name, target in (("outcome", outcome), ("teacher", teacher)):
            for l2 in L2_GRID:
                key = f"{family}:{target_name}:{l2}"
                models[key] = fit_soft_action(x[:, columns[family]], target, l2=l2)
    training_disagreements = int(mask.sum())
    del fit
    selected, selection_scores, output = {}, {}, {}
    for role in ("selection", "development"):
        data = make_data(artifact, tables, seeds, role, args, mapper, motifs, endpoints)
        hashes[role] = data["hash"]
        decoded = {}
        for family in FAMILIES:
            x = matrix(data, family)[:, columns[family]]
            for target in ("outcome", "teacher"):
                label = f"{family}:{target}"
                keys = [f"{label}:{l2}" for l2 in L2_GRID] if role == "selection" else [selected[label]]
                for key in keys:
                    choice = affine_log_likelihood_ratio(x, models[key]) >= 0
                    prediction = data["predictions"][0].copy()
                    where = np.flatnonzero(data["disagreement"])
                    prediction[where] = np.where(choice, data["predictions"][1][where], prediction[where])
                    if role == "selection":
                        selection_scores[key] = float(np.mean(prediction != data["labels"]))
                    else:
                        decoded[label] = prediction
                if role == "selection":
                    selected[label] = min(keys, key=lambda key: (selection_scores[key], key))
        if role == "selection":
            continue
        decoded["static"] = restore_static(artifact, artifact["selected"]["static"]).decode(data["detectors"])
        decoded["exact_teacher"] = data["q"] > .5
        decoded["restricted_teacher"] = np.where(data["disagreement"], decoded["exact_teacher"], data["predictions"][0])
        reference = decoded["uncapped_base:outcome"] != data["labels"]
        for name, prediction in decoded.items():
            loss = prediction != data["labels"]
            expected = np.where(prediction, 1 - data["q"], data["q"])
            output[name] = {
                "ler": float(loss.mean()), "exact_model_expected_risk": float(expected.mean()),
                "errors_per_stream": loss.reshape(args.streams, args.horizon).sum(axis=1).tolist(),
                "expected_risk_per_stream": expected.reshape(args.streams, args.horizon).mean(axis=1).tolist(),
                "rescues_vs_uncapped_outcome": int(np.sum(reference & ~loss)),
                "harms_vs_uncapped_outcome": int(np.sum(~reference & loss)),
                "paired_error_difference_per_stream": (loss.astype(int) - reference).reshape(args.streams, args.horizon).sum(axis=1).tolist(),
            }
    print(f"r{artifact['replicate']}: " + ", ".join(f"{k}={v['ler']:.6f}" for k, v in output.items()), flush=True)
    return {"replicate": artifact["replicate"], "seeds": seeds.manifest, "data_hashes": hashes,
            "motifs": motifs, "training_disagreements": training_disagreements,
            "feature_columns": {k: v.tolist() for k, v in columns.items()}, "widths": widths,
            "selected": selected, "selection_scores": selection_scores, "methods": output,
            "models": {k: json_model(models[k]) for k in selected.values()}}


def summary(rows):
    names = rows[0]["methods"]
    mean = {k: float(np.mean([r["methods"][k]["ler"] for r in rows])) for k in names}
    risk = {k: float(np.mean([r["methods"][k]["exact_model_expected_risk"] for r in rows])) for k in names}
    gap = mean["uncapped_base:outcome"] - mean["restricted_teacher"]
    return {"mean_ler": mean, "mean_exact_model_expected_risk": risk,
            "restricted_teacher_recovery_vs_uncapped_outcome": {
                k: (mean["uncapped_base:outcome"] - v) / gap if gap > 0 else None for k, v in mean.items()},
            "status": "development_only_no_confirmatory_gate", "replicates": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/surface_frontier_challenge"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=2026090911)
    parser.add_argument("--streams", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--feature-budget", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output path; prior screens are immutable")
    if min(args.streams, args.horizon, args.feature_budget) < 1 or len(set(args.replicates)) != len(args.replicates):
        raise ValueError("positive budgets and unique replicates required")
    start = time.perf_counter()
    sources = [__file__, "src/ptwm/directional_features.py", "src/ptwm/exact_dem.py", "src/ptwm/soft_action.py",
               "src/ptwm/surface_program.py", "scripts/run_surface_frontier_challenge.py",
               "scripts/run_surface_crossfit_compiler.py", "scripts/benchmark_surface_frontier.py",
               "docs/directional-feature-screen-protocol.md"]
    output = {"protocol": "docs/directional-feature-screen-protocol.md", "root_seed": args.seed,
              "config": {"streams": args.streams, "horizon": args.horizon, "feature_budget": args.feature_budget},
              "python": platform.python_version(), "numpy": np.__version__,
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}, "records": []}
    tables, diagnostics = [], []
    for rates in REGIMES:
        c = circuit(3, *rates)
        table, diagnostic = exact_distribution(c.detector_error_model())
        tables.append(table.reshape(2, 1 << c.num_detectors))
        diagnostics.append(diagnostic)
    tables = np.stack(tables)
    output["exact_teacher_diagnostics"] = diagnostics
    print(f"Exact tables ready ({time.perf_counter()-start:.1f}s), bytes={tables.nbytes}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for replicate in args.replicates:
        path = args.input_dir / f"r{replicate:02d}_d3.json"
        row = run(json.loads(path.read_text()), tables, args)
        row["source_artifact"] = str(path)
        row["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        output["records"].append(row)
        output["summary"] = summary(output["records"])
        output["elapsed_seconds"] = time.perf_counter() - start
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
