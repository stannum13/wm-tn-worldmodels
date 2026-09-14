#!/usr/bin/env python3
"""Development screen for bounded within-record schedule inference."""
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
from ptwm.noise_schedule import compile_noise_schedule, schedule_from_detector_modes
from ptwm.schedule_inference import (
    enumerate_binary_schedules, fit_detector_emissions, schedule_scores, select_schedule,
)
from ptwm.time_templates import splice_noise
from scripts.run_surface_frontier_challenge import REGIMES, Seeds, circuit, data_hash

PENALTIES = (0., .5, 1., 2., 4., 8.)


def physical_conditions(a, b):
    output = [("stationary_a", a, [0] * 6), ("stationary_b", b, [1] * 6)]
    for first, second, label in ((a, b, "ab"), (b, a, "ba")):
        for fraction in (.25, 1 / 3, .5, 2 / 3, .75):
            sequence = None
            boundary = fraction * 6
            if abs(boundary - round(boundary)) < 1e-12:
                k = round(boundary)
                sequence = ([0] * k + [1] * (6 - k) if label == "ab"
                            else [1] * k + [0] * (6 - k))
            output.append((f"{label}:{fraction:.6f}", splice_noise(first, second, fraction), sequence))
    output.append(("aba:thirds", splice_noise(splice_noise(a, b, 1 / 3), a, 2 / 3), [0, 0, 1, 1, 0, 0]))
    output.append(("bab:thirds", splice_noise(splice_noise(b, a, 1 / 3), b, 2 / 3), [1, 1, 0, 0, 1, 1]))
    return output


def matching(c):
    return pymatching.Matching.from_detector_error_model(
        c.detector_error_model(decompose_errors=True), enable_correlations=True)


def decode(matcher, detectors):
    return matcher.decode_batch(detectors, enable_correlations=True)[:, 0].astype(bool)


def build_candidates(a, b, detector_bins):
    schedules = enumerate_binary_schedules(len(np.unique(detector_bins)), 2)
    ticks = sum(op.name == "TICK" for op in a.flattened())
    circuits, matchers = [], []
    for row in schedules:
        c = compile_noise_schedule([a, b], schedule_from_detector_modes(ticks, row))
        if c.without_noise() != a.flattened().without_noise() or c.get_detector_coordinates() != a.get_detector_coordinates():
            raise AssertionError("candidate changed the circuit interface")
        circuits.append(c)
        matchers.append(matching(c))
    changes = np.sum(schedules[:, 1:] != schedules[:, :-1], axis=1)
    return schedules, changes, circuits, matchers


def choose(scores, schedules, changes, max_changes, penalty):
    subset = np.flatnonzero(changes <= max_changes)
    local = select_schedule(scores[:, subset], schedules[subset], penalty=penalty)
    return subset[local]


def sample(c, shots, seed):
    d, y = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return d, y[:, 0]


def blocks(loss, size=256):
    loss = np.asarray(loss, dtype=bool)
    return [int(chunk.sum()) for chunk in np.array_split(loss, max(1, len(loss) // size))]


def run_replicate(args, replicate):
    started = time.perf_counter()
    seeds = Seeds(args.seed, replicate, 5)
    a, b = [circuit(5, *rates) for rates in REGIMES]
    coordinates = a.get_detector_coordinates()
    times = np.asarray([coordinates[i][2] for i in range(a.num_detectors)])
    unique_times = np.unique(times)
    detector_bins = np.searchsorted(unique_times, times).astype(np.int32)
    if len(unique_times) != 6:
        raise ValueError("this development protocol requires six d5 detector-time bins")
    calibration, hashes = [], {}
    for mode, c in enumerate((a, b)):
        d, y = sample(c, args.calibration_shots, seeds.draw("inferred_schedule", "calibration", mode))
        calibration.append(d)
        hashes[f"calibration_{mode}"] = data_hash(d, y)
    emissions = fit_detector_emissions(calibration)
    del calibration
    schedules, changes, candidate_circuits, matchers = build_candidates(a, b, detector_bins)
    conditions = physical_conditions(a, b)
    selection_loss = {width: {penalty: 0 for penalty in PENALTIES} for width in (0, 1, 2)}
    static_loss = np.zeros(len(schedules), dtype=np.int64)
    selection_records = 0
    for name, c, _ in conditions:
        d, y = sample(c, args.selection_shots, seeds.draw("inferred_schedule", "selection", name))
        hashes[f"selection_{name}"] = data_hash(d, y)
        score = schedule_scores(d, emissions, schedules, detector_bins)
        decoded = np.column_stack([decode(m, d) for m in matchers])
        static_loss += np.sum(decoded != y[:, None], axis=0)
        for width in (0, 1, 2):
            for penalty in PENALTIES:
                selected = choose(score, schedules, changes, width, penalty)
                selection_loss[width][penalty] += int(np.sum(decoded[np.arange(len(y)), selected] != y))
        selection_records += len(y)
    selected_penalty = {width: min(PENALTIES, key=lambda p: (selection_loss[width][p], p)) for width in (0, 1, 2)}
    static_index = int(np.argmin(static_loss))
    results, equivalence = {}, {"width_0": 0, "width_1": 0, "width_2": 0}
    for name, c, true_sequence in conditions:
        d, y = sample(c, args.evaluation_shots, seeds.draw("inferred_schedule", "development", name))
        hashes[f"development_{name}"] = data_hash(d, y)
        score = schedule_scores(d, emissions, schedules, detector_bins)
        decoded = np.column_stack([decode(m, d) for m in matchers])
        selected_indices = {width: choose(score, schedules, changes, width, selected_penalty[width])
                            for width in (0, 1, 2)}
        actual = decode(matching(c), d)
        predictions = {"static": decoded[:, static_index], "known_schedule": actual}
        for width, indices in selected_indices.items():
            predictions[f"inferred_{width}"] = decoded[np.arange(len(y)), indices]
            routed = np.empty(len(y), dtype=bool)
            for index in np.unique(indices):
                where = np.flatnonzero(indices == index)
                routed[where] = decode(matchers[int(index)], d[where])
            mismatch = int(np.sum(routed != predictions[f"inferred_{width}"]))
            equivalence[f"width_{width}"] += mismatch
            if mismatch:
                raise AssertionError("predecoded and routed actions disagree")
        best = np.any(decoded == y[:, None], axis=1)
        predictions["best_candidate_outcome"] = np.where(best, y, ~y)
        reference = predictions["static"] != y
        methods = {}
        for method, prediction in predictions.items():
            loss = prediction != y
            methods[method] = {"ler": float(loss.mean()), "errors": int(loss.sum()),
                "error_blocks": blocks(loss), "rescues_vs_static": int(np.sum(reference & ~loss)),
                "harms_vs_static": int(np.sum(~reference & loss))}
        sequence_counts = {}
        exact_sequence_rate = {}
        for width, indices in selected_indices.items():
            values, counts = np.unique(indices, return_counts=True)
            sequence_counts[str(width)] = {str(int(i)): int(n) for i, n in zip(values, counts)}
            exact_sequence_rate[str(width)] = (None if true_sequence is None else
                float(np.mean(np.all(schedules[indices] == np.asarray(true_sequence), axis=1))))
        results[name] = {"records": len(y), "true_detector_bin_sequence": true_sequence, "methods": methods,
            "selected_sequence_counts": sequence_counts, "exact_sequence_rate": exact_sequence_rate}
        print(f"r{replicate} {name}: " + ", ".join(f"{k}={v['ler']:.6f}" for k, v in methods.items()
              if k != "best_candidate_outcome"), flush=True)
    return {"replicate": replicate, "elapsed_seconds": time.perf_counter() - started,
        "seeds": seeds.manifest, "data_hashes": hashes, "detector_times": unique_times.tolist(),
        "emissions": emissions.tolist(), "candidate_schedules": schedules.tolist(),
        "candidate_changes": changes.tolist(), "candidate_circuit_sha256":
            [hashlib.sha256(str(c).encode()).hexdigest() for c in candidate_circuits],
        "selection_records": selection_records,
        "selection_errors": {str(w): {str(p): n for p, n in v.items()} for w, v in selection_loss.items()},
        "selected_penalty": {str(k): v for k, v in selected_penalty.items()},
        "static_selection_errors": static_loss.tolist(), "static_index": static_index,
        "pipeline_equivalence_disagreements": equivalence, "conditions": results}


def summarize(records):
    conditions = records[0]["conditions"]
    methods = next(iter(conditions.values()))["methods"]
    means, aggregate = {}, {}
    for condition in conditions:
        means[condition] = {method: float(np.mean([r["conditions"][condition]["methods"][method]["ler"]
                                                   for r in records])) for method in methods}
    for method in methods:
        if method == "best_candidate_outcome":
            continue
        errors = sum(r["conditions"][c]["methods"][method]["errors"] for r in records for c in conditions)
        total = sum(r["conditions"][c]["records"] for r in records for c in conditions)
        aggregate[method] = {"errors": errors, "ler": errors / total}
    return {"status": "development_only_no_confirmatory_gate", "replicates": len(records),
            "mean_ler_by_condition": means, "aggregate_equal_condition_ler": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=2026090913)
    parser.add_argument("--calibration-shots", type=int, default=65536)
    parser.add_argument("--selection-shots", type=int, default=4096)
    parser.add_argument("--evaluation-shots", type=int, default=16384)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose an unused output path")
    if min(args.calibration_shots, args.selection_shots, args.evaluation_shots) < 1:
        raise ValueError("positive sample counts required")
    sources = [__file__, "src/ptwm/noise_schedule.py", "src/ptwm/schedule_inference.py",
               "src/ptwm/time_templates.py", "scripts/run_surface_frontier_challenge.py",
               "scripts/run_surface_regime_switch.py", "docs/directional-feature-screen-protocol.md"]
    output = {"protocol": "docs/directional-feature-screen-protocol.md", "root_seed": args.seed,
        "config": {"calibration_shots": args.calibration_shots, "selection_shots": args.selection_shots,
                   "evaluation_shots": args.evaluation_shots, "penalties": PENALTIES},
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
