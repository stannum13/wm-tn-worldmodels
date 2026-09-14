#!/usr/bin/env python3
"""Development screen for persistent evidence over compiled schedule hypotheses."""
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
import pymatching
import scipy
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.schedule_inference import filter_schedule_scores, fit_detector_emissions, schedule_scores
from scripts.run_inferred_schedule_screen import (
    PENALTIES, build_candidates, decode, matching, physical_conditions, sample,
)
from scripts.run_surface_frontier_challenge import REGIMES, Seeds, circuit, data_hash

RETENTIONS = (0., .5, .9, .99, 1.)
BURN_INS = (0, 4, 16, 64)


def filtered_choice(scores, schedules, changes, width, penalty, retention):
    subset = np.flatnonzero(changes <= width)
    local, state = filter_schedule_scores(scores[:, :, subset], schedules[subset],
                                          penalty=penalty, retention=retention)
    return subset[local], state


def stream_sample(c, streams, horizon, seed):
    d, y = sample(c, streams * horizon, seed)
    return d, y, y.reshape(streams, horizon)


def run_replicate(args, replicate):
    started = time.perf_counter()
    seeds = Seeds(args.seed, replicate, 5)
    a, b = [circuit(5, *rates) for rates in REGIMES]
    coordinates = a.get_detector_coordinates()
    times = np.asarray([coordinates[i][2] for i in range(a.num_detectors)])
    unique_times = np.unique(times)
    detector_bins = np.searchsorted(unique_times, times).astype(np.int32)
    if len(unique_times) != 6:
        raise ValueError("this development protocol requires six detector-time bins")
    calibration, hashes = [], {}
    for mode, c in enumerate((a, b)):
        d, y = sample(c, args.calibration_shots, seeds.draw("schedule_memory", "calibration", mode))
        calibration.append(d)
        hashes[f"calibration_{mode}"] = data_hash(d, y)
    emissions = fit_detector_emissions(calibration)
    del calibration
    schedules, changes, candidate_circuits, matchers = build_candidates(a, b, detector_bins)
    conditions = physical_conditions(a, b)
    selection_errors = {width: {(retention, penalty): 0 for retention in RETENTIONS for penalty in PENALTIES}
                        for width in (0, 1, 2)}
    static_errors = np.zeros(len(schedules), dtype=np.int64)
    selection_records = 0
    for name, c, _ in conditions:
        d, y, _ = stream_sample(c, args.selection_streams, args.selection_horizon,
                                seeds.draw("schedule_memory", "selection", name))
        hashes[f"selection_{name}"] = data_hash(d, y)
        raw = schedule_scores(d, emissions, schedules, detector_bins).reshape(
            args.selection_streams, args.selection_horizon, len(schedules))
        scores = raw - raw.max(axis=2, keepdims=True)
        decoded = np.column_stack([decode(m, d) for m in matchers])
        static_errors += np.sum(decoded != y[:, None], axis=0)
        for width in (0, 1, 2):
            for retention in RETENTIONS:
                for penalty in PENALTIES:
                    choice, _ = filtered_choice(scores, schedules, changes, width, penalty, retention)
                    prediction = decoded[np.arange(len(y)), choice.ravel()]
                    selection_errors[width][(retention, penalty)] += int(np.sum(prediction != y))
        selection_records += len(y)
    selected = {width: min(selection_errors[width],
                           key=lambda key: (selection_errors[width][key], key[0], key[1])) for width in (0, 1, 2)}
    static_index = int(np.argmin(static_errors))
    results, equivalence = {}, {"width_0": 0, "width_1": 0, "width_2": 0}
    for name, c, true_sequence in conditions:
        d, y, truth = stream_sample(c, args.evaluation_streams, args.evaluation_horizon,
                                    seeds.draw("schedule_memory", "development", name))
        hashes[f"development_{name}"] = data_hash(d, y)
        raw = schedule_scores(d, emissions, schedules, detector_bins).reshape(
            args.evaluation_streams, args.evaluation_horizon, len(schedules))
        scores = raw - raw.max(axis=2, keepdims=True)
        decoded = np.column_stack([decode(m, d) for m in matchers])
        choices = {}
        predictions = {"static": decoded[:, static_index].reshape(truth.shape),
                       "known_schedule": decode(matching(c), d).reshape(truth.shape)}
        exact_rate = {}
        for width in (0, 1, 2):
            retention, penalty = selected[width]
            choice, _ = filtered_choice(scores, schedules, changes, width, penalty, retention)
            choices[width] = choice
            predictions[f"inferred_{width}"] = decoded[np.arange(len(y)), choice.ravel()].reshape(truth.shape)
            routed = np.empty(len(y), dtype=bool)
            for index in np.unique(choice):
                where = np.flatnonzero(choice.ravel() == index)
                routed[where] = decode(matchers[int(index)], d[where])
            mismatch = int(np.sum(routed.reshape(truth.shape) != predictions[f"inferred_{width}"]))
            equivalence[f"width_{width}"] += mismatch
            if mismatch:
                raise AssertionError("predecoded and routed schedule actions disagree")
            exact_rate[str(width)] = (None if true_sequence is None else
                np.mean(np.all(schedules[choice] == np.asarray(true_sequence), axis=2), axis=0).tolist())
        reference = predictions["static"] != truth
        methods = {}
        for method, prediction in predictions.items():
            loss = prediction != truth
            methods[method] = {"ler": float(loss.mean()), "errors": int(loss.sum()),
                "errors_per_stream": loss.sum(axis=1).tolist(),
                "rescues_vs_static": int(np.sum(reference & ~loss)), "harms_vs_static": int(np.sum(~reference & loss)),
                "post_burn_in_ler": {str(burn): float(loss[:, burn:].mean()) for burn in BURN_INS
                                     if burn < args.evaluation_horizon}}
        results[name] = {"records": len(y), "true_detector_bin_sequence": true_sequence,
            "methods": methods, "exact_sequence_rate_over_time": exact_rate,
            "selected_sequence_counts": {str(width): np.bincount(choice.ravel(), minlength=len(schedules)).tolist()
                                         for width, choice in choices.items()}}
        print(f"r{replicate} {name}: " + ", ".join(f"{k}={v['ler']:.6f}" for k, v in methods.items()), flush=True)
    return {"replicate": replicate, "elapsed_seconds": time.perf_counter() - started,
        "seeds": seeds.manifest, "data_hashes": hashes, "detector_times": unique_times.tolist(),
        "emissions": emissions.tolist(), "candidate_schedules": schedules.tolist(), "candidate_changes": changes.tolist(),
        "candidate_circuit_sha256": [hashlib.sha256(str(c).encode()).hexdigest() for c in candidate_circuits],
        "selection_records": selection_records,
        "selection_errors": {str(w): {f"retention={r},penalty={p}": n for (r, p), n in values.items()}
                             for w, values in selection_errors.items()},
        "selected": {str(w): {"retention": values[0], "penalty": values[1]} for w, values in selected.items()},
        "static_selection_errors": static_errors.tolist(), "static_index": static_index,
        "persistent_state_bytes_float64": {str(w): int(np.sum(changes <= w) * 8) for w in (0, 1, 2)},
        "pipeline_equivalence_disagreements": equivalence, "conditions": results}


def summarize(records):
    conditions = records[0]["conditions"]
    methods = next(iter(conditions.values()))["methods"]
    output = {}
    for method in methods:
        errors = sum(r["conditions"][c]["methods"][method]["errors"] for r in records for c in conditions)
        total = sum(r["conditions"][c]["records"] for r in records for c in conditions)
        by_replicate = []
        for r in records:
            e = sum(r["conditions"][c]["methods"][method]["errors"] for c in conditions)
            n = sum(r["conditions"][c]["records"] for c in conditions)
            by_replicate.append(e / n)
        output[method] = {"ler": errors / total, "errors": errors, "replicate_ler": by_replicate}
    return {"status": "development_only_persistent_conditions", "replicates": len(records),
            "aggregate_equal_condition_ler": output,
            "mean_ler_by_condition": {c: {m: float(np.mean([r["conditions"][c]["methods"][m]["ler"]
                                                             for r in records])) for m in methods}
                                      for c in conditions}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=2026090914)
    parser.add_argument("--calibration-shots", type=int, default=65536)
    parser.add_argument("--selection-streams", type=int, default=16)
    parser.add_argument("--selection-horizon", type=int, default=128)
    parser.add_argument("--evaluation-streams", type=int, default=64)
    parser.add_argument("--evaluation-horizon", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose an unused output path")
    if min(args.calibration_shots, args.selection_streams, args.selection_horizon,
           args.evaluation_streams, args.evaluation_horizon) < 1:
        raise ValueError("positive sample budgets required")
    sources = [__file__, "scripts/run_inferred_schedule_screen.py", "src/ptwm/noise_schedule.py",
               "src/ptwm/schedule_inference.py", "src/ptwm/time_templates.py",
               "scripts/run_surface_frontier_challenge.py", "scripts/run_surface_regime_switch.py",
               "docs/schedule-memory-screen-protocol.md"]
    output = {"protocol": "docs/schedule-memory-screen-protocol.md", "root_seed": args.seed,
        "config": {"calibration_shots": args.calibration_shots, "selection_streams": args.selection_streams,
                   "selection_horizon": args.selection_horizon, "evaluation_streams": args.evaluation_streams,
                   "evaluation_horizon": args.evaluation_horizon, "retentions": RETENTIONS,
                   "penalties": PENALTIES, "burn_ins": BURN_INS},
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "stim": stim.__version__, "pymatching": pymatching.__version__},
        "source_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for replicate in args.replicates:
        output["records"].append(run_replicate(args, replicate))
        output["summary"] = summarize(output["records"])
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
