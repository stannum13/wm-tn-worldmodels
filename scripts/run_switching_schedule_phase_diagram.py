#!/usr/bin/env python3
"""Development phase diagram for switching compiled schedule hypotheses."""
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
from ptwm.schedule_inference import (
    filter_switching_schedule_scores, fit_detector_emissions, schedule_scores,
)
from scripts.run_inferred_schedule_screen import (
    PENALTIES, build_candidates, decode, matching, physical_conditions, sample,
)
from scripts.run_surface_frontier_challenge import REGIMES, Seeds, circuit, data_hash

RETENTIONS = (.5, .9, .99, 1.)
RESET_MARGINS = (2., 5., 10., 20., np.inf)
SELECTION_DWELLS = (16, 64, 256)
EVALUATION_DWELLS = (4, 8, 16, 32, 64, 128, 256, 512)
MISMATCH_SCALES = (.75, 1., 1.25)
SCHEDULE_NAMES = ("stationary_a", "stationary_b", "ab:0.500000",
                  "ba:0.500000", "aba:thirds", "bab:thirds")


def scaled_regimes(scale):
    return tuple(tuple(float(scale) * x for x in rates) for rates in REGIMES)


def selected_conditions(a, b):
    conditions = {name: (c, sequence) for name, c, sequence in physical_conditions(a, b)}
    return {name: conditions[name] for name in SCHEDULE_NAMES}


def true_candidate_indices(conditions, schedules):
    output = {}
    for name, (_, sequence) in conditions.items():
        where = np.flatnonzero(np.all(schedules == np.asarray(sequence), axis=1))
        if len(where) != 1:
            raise AssertionError(f"schedule {name} is not uniquely represented")
        output[name] = int(where[0])
    return output


def switching_sample(conditions, streams, horizon, dwell, seed):
    """Sample a balanced-start Markov path without per-block sampler overhead."""
    rng = np.random.default_rng(seed)
    names = tuple(conditions)
    segments = (horizon + dwell - 1) // dwell
    starts = np.tile(rng.permutation(len(names)), (streams + len(names) - 1) // len(names))[:streams]
    path = np.empty((streams, horizon), dtype=np.int16)
    segment_path = np.empty((streams, segments), dtype=np.int16)
    for stream in range(streams):
        current = int(starts[stream])
        for segment in range(segments):
            if segment:
                draw = int(rng.integers(len(names) - 1))
                current = draw + (draw >= current)
            segment_path[stream, segment] = current
            begin, end = segment * dwell, min((segment + 1) * dwell, horizon)
            path[stream, begin:end] = current
    sample_seeds = rng.integers(0, (1 << 63) - 1, size=len(names), dtype=np.int64)
    if len(set(map(int, sample_seeds))) != len(names):
        raise AssertionError("derived sampler seed collision")
    first = conditions[names[0]][0]
    detectors = np.empty((streams * horizon, first.num_detectors), dtype=bool)
    observables = np.empty(streams * horizon, dtype=bool)
    flat_path = path.ravel()
    for index, name in enumerate(names):
        positions = np.flatnonzero(flat_path == index)
        d, y = sample(conditions[name][0], len(positions), int(sample_seeds[index]))
        detectors[positions], observables[positions] = d, y
    return detectors, observables, path, segment_path, sample_seeds.tolist()


def controller(scores, schedules, changes, width, config):
    subset = np.flatnonzero(changes <= width)
    local, state, resets = filter_switching_schedule_scores(
        scores[:, :, subset], schedules[subset], penalty=config[2],
        retention=config[0], reset_margin=config[1])
    return subset[local], state, resets


def recovery(static, adaptive, oracle):
    denominator = static - oracle
    return None if denominator <= 0 else float((static - adaptive) / denominator)


def transition_metrics(choice, resets, target, loss, reachable_loss, dwell):
    streams, horizon = choice.shape
    action_delays, reset_delays = [], []
    action_hits = reset_hits = false_resets = 0
    transient_loss = transient_reference = transient_n = 0
    steady_loss = steady_reference = steady_n = 0
    for stream in range(streams):
        for boundary in range(dwell, horizon, dwell):
            end = min(boundary + dwell, horizon)
            wanted = target[stream, boundary]
            delay = dwell
            for step in range(boundary, max(boundary, end - 1)):
                if choice[stream, step] == wanted and choice[stream, step + 1] == wanted:
                    delay = step - boundary
                    action_hits += 1
                    break
            action_delays.append(delay)
            hits = np.flatnonzero(resets[stream, boundary:end])
            if len(hits):
                reset_hits += 1
                reset_delays.append(int(hits[0]))
                false_resets += len(hits) - 1
            else:
                reset_delays.append(dwell)
            cut = min(boundary + min(16, dwell), end)
            transient_loss += int(loss[stream, boundary:cut].sum())
            transient_reference += int(reachable_loss[stream, boundary:cut].sum())
            transient_n += cut - boundary
            if cut < end:
                steady_loss += int(loss[stream, cut:end].sum())
                steady_reference += int(reachable_loss[stream, cut:end].sum())
                steady_n += end - cut
        false_resets += int(resets[stream, :min(dwell, horizon)].sum())
    switches = len(action_delays)
    non_boundary = streams * horizon - switches
    return {
        "switches": switches,
        "action_settle_recall": action_hits / switches if switches else None,
        "action_settle_delay_mean_censored": float(np.mean(action_delays)) if switches else None,
        "action_settle_delay_median_censored": float(np.median(action_delays)) if switches else None,
        "reset_recall": reset_hits / switches if switches else None,
        "reset_delay_mean_censored": float(np.mean(reset_delays)) if switches else None,
        "false_or_retriggered_resets_per_1000_nonboundary": 1000 * false_resets / non_boundary,
        "transient": {"records": transient_n, "errors": transient_loss,
                      "reachable_errors": transient_reference,
                      "excess_errors": transient_loss - transient_reference},
        "steady": {"records": steady_n, "errors": steady_loss,
                   "reachable_errors": steady_reference,
                   "excess_errors": steady_loss - steady_reference},
    }


def evaluate_cell(*, conditions, nominal_matchers, schedules, changes, emissions,
                  true_index, static_index, selected, streams, horizon, dwell, seed):
    d, y, path, segment_path, sample_seeds = switching_sample(
        conditions, streams, horizon, dwell, seed)
    names = tuple(conditions)
    target = np.vectorize(lambda i: true_index[names[int(i)]], otypes=[np.int32])(path)
    detector_times = np.asarray([conditions[names[0]][0].get_detector_coordinates()[i][2]
                                 for i in range(d.shape[1])])
    detector_bins = np.searchsorted(np.unique(detector_times), detector_times).astype(np.int32)
    raw = schedule_scores(d, emissions, schedules, detector_bins).reshape(streams, horizon, len(schedules))
    scores = raw - raw.max(axis=2, keepdims=True)
    decoded = np.column_stack([decode(m, d) for m in nominal_matchers])
    static_prediction = decoded[:, static_index].reshape(streams, horizon)
    reachable_prediction = decoded[np.arange(len(y)), target.ravel()].reshape(streams, horizon)
    physical_prediction = np.empty(len(y), dtype=bool)
    for index, name in enumerate(names):
        where = np.flatnonzero(path.ravel() == index)
        physical_prediction[where] = decode(matching(conditions[name][0]), d[where])
    physical_prediction = physical_prediction.reshape(streams, horizon)
    truth = y.reshape(streams, horizon)
    predictions = {"static": static_prediction, "reachable_schedule": reachable_prediction,
                   "physical_oracle": physical_prediction}
    choices, resets, equivalence = {}, {}, {}
    for width in (0, 1, 2):
        choice, _, reset = controller(scores, schedules, changes, width, selected[width])
        prediction = decoded[np.arange(len(y)), choice.ravel()].reshape(streams, horizon)
        predictions[f"adaptive_{width}"] = prediction
        choices[width], resets[width] = choice, reset
        routed = np.empty(len(y), dtype=bool)
        for index in np.unique(choice):
            where = np.flatnonzero(choice.ravel() == index)
            routed[where] = decode(nominal_matchers[int(index)], d[where])
        equivalence[str(width)] = int(np.sum(routed.reshape(streams, horizon) != prediction))
        if equivalence[str(width)]:
            raise AssertionError("predecoded and routed actions disagree")
    static_loss = predictions["static"] != truth
    reachable_loss = predictions["reachable_schedule"] != truth
    physical_loss = predictions["physical_oracle"] != truth
    methods = {}
    for name, prediction in predictions.items():
        loss = prediction != truth
        methods[name] = {"ler": float(loss.mean()), "errors": int(loss.sum()),
                         "errors_per_stream": loss.sum(axis=1).tolist(),
                         "rescues_vs_static": int(np.sum(static_loss & ~loss)),
                         "harms_vs_static": int(np.sum(~static_loss & loss))}
        if name.startswith("adaptive_"):
            width = int(name[-1])
            methods[name]["exact_action_rate"] = float(np.mean(choices[width] == target))
            methods[name]["physical_oracle_gap_recovery"] = recovery(
                float(static_loss.mean()), float(loss.mean()), float(physical_loss.mean()))
            methods[name]["reachable_action_gap_recovery"] = recovery(
                float(static_loss.mean()), float(loss.mean()), float(reachable_loss.mean()))
            methods[name]["transition"] = transition_metrics(
                choices[width], resets[width], target, loss, reachable_loss, dwell)
    return {"records": len(y), "sample_seeds": sample_seeds,
            "path_hash": hashlib.sha256(path.tobytes()).hexdigest(),
            "segment_paths": segment_path.tolist(), "data_hash": data_hash(d, y),
            "pipeline_equivalence_disagreements": equivalence, "methods": methods}


def select_configs(*, conditions, matchers, schedules, changes, emissions, true_index,
                   streams, horizon, seeds):
    configurations = [(r, margin, p) for r in RETENTIONS
                      for margin in RESET_MARGINS for p in PENALTIES]
    errors = {width: {config: 0 for config in configurations} for width in (0, 1, 2)}
    static_errors = np.zeros(len(schedules), dtype=np.int64)
    hashes = {}
    for dwell in SELECTION_DWELLS:
        seed = seeds.draw("switching_phase", "selection", dwell)
        d, y, _, _, _ = switching_sample(conditions, streams, horizon, dwell, seed)
        hashes[str(dwell)] = data_hash(d, y)
        detector_times = np.asarray([conditions[SCHEDULE_NAMES[0]][0].get_detector_coordinates()[i][2]
                                     for i in range(d.shape[1])])
        detector_bins = np.searchsorted(np.unique(detector_times), detector_times).astype(np.int32)
        raw = schedule_scores(d, emissions, schedules, detector_bins).reshape(streams, horizon, len(schedules))
        scores = raw - raw.max(axis=2, keepdims=True)
        decoded = np.column_stack([decode(m, d) for m in matchers])
        static_errors += np.sum(decoded != y[:, None], axis=0)
        for width in (0, 1, 2):
            for config in configurations:
                choice, _, _ = controller(scores, schedules, changes, width, config)
                errors[width][config] += int(np.sum(decoded[np.arange(len(y)), choice.ravel()] != y))
    selected = {width: min(errors[width], key=lambda c: (errors[width][c], c[0], c[1], c[2]))
                for width in (0, 1, 2)}
    return selected, int(np.argmin(static_errors)), errors, static_errors, hashes


def stationary_check(*, conditions, matchers, schedules, changes, emissions, static_index,
                     selected, streams, horizon, seeds):
    output = {}
    for name in ("stationary_a", "stationary_b"):
        d, y = sample(conditions[name][0], streams * horizon,
                      seeds.draw("switching_phase", "stationary", name))
        detector_times = np.asarray([conditions[name][0].get_detector_coordinates()[i][2]
                                     for i in range(d.shape[1])])
        bins = np.searchsorted(np.unique(detector_times), detector_times).astype(np.int32)
        raw = schedule_scores(d, emissions, schedules, bins).reshape(streams, horizon, len(schedules))
        scores = raw - raw.max(axis=2, keepdims=True)
        decoded = np.column_stack([decode(m, d) for m in matchers])
        static_loss = decoded[:, static_index] != y
        methods = {"static": {"errors": int(static_loss.sum()), "ler": float(static_loss.mean())}}
        for width in (0, 1, 2):
            choice, _, reset = controller(scores, schedules, changes, width, selected[width])
            loss = decoded[np.arange(len(y)), choice.ravel()] != y
            methods[f"adaptive_{width}"] = {
                "errors": int(loss.sum()), "ler": float(loss.mean()),
                "harm_pp_vs_static": float(100 * (loss.mean() - static_loss.mean())),
                "resets": int(reset.sum())}
        output[name] = {"records": len(y), "data_hash": data_hash(d, y), "methods": methods}
    return output


def run_replicate(args, replicate):
    started = time.perf_counter()
    seeds = Seeds(args.seed, replicate, 5)
    nominal_a, nominal_b = [circuit(5, *rates) for rates in REGIMES]
    nominal_conditions = selected_conditions(nominal_a, nominal_b)
    coordinates = nominal_a.get_detector_coordinates()
    times = np.asarray([coordinates[i][2] for i in range(nominal_a.num_detectors)])
    bins = np.searchsorted(np.unique(times), times).astype(np.int32)
    calibration, hashes = [], {}
    for mode, c in enumerate((nominal_a, nominal_b)):
        d, y = sample(c, args.calibration_shots,
                      seeds.draw("switching_phase", "calibration", mode))
        calibration.append(d)
        hashes[f"calibration_{mode}"] = data_hash(d, y)
    emissions = fit_detector_emissions(calibration)
    schedules, changes, candidate_circuits, matchers = build_candidates(nominal_a, nominal_b, bins)
    true_index = true_candidate_indices(nominal_conditions, schedules)
    selected, static_index, selection_errors, static_errors, selection_hashes = select_configs(
        conditions=nominal_conditions, matchers=matchers, schedules=schedules,
        changes=changes, emissions=emissions, true_index=true_index,
        streams=args.selection_streams, horizon=args.selection_horizon, seeds=seeds)
    cells = {}
    for scale in MISMATCH_SCALES:
        a, b = [circuit(5, *rates) for rates in scaled_regimes(scale)]
        conditions = selected_conditions(a, b)
        for dwell in EVALUATION_DWELLS:
            key = f"scale={scale:.2f},dwell={dwell}"
            cells[key] = evaluate_cell(
                conditions=conditions, nominal_matchers=matchers, schedules=schedules,
                changes=changes, emissions=emissions, true_index=true_index,
                static_index=static_index, selected=selected,
                streams=args.evaluation_streams, horizon=args.evaluation_horizon,
                dwell=dwell, seed=seeds.draw("switching_phase", "development", scale, dwell))
            print(f"r{replicate} {key}: " + ", ".join(
                f"{m}={v['ler']:.5f}" for m, v in cells[key]["methods"].items()), flush=True)
    stationary = stationary_check(
        conditions=nominal_conditions, matchers=matchers, schedules=schedules,
        changes=changes, emissions=emissions, static_index=static_index,
        selected=selected, streams=args.evaluation_streams,
        horizon=args.evaluation_horizon, seeds=seeds)
    return {
        "replicate": replicate, "elapsed_seconds": time.perf_counter() - started,
        "seeds": seeds.manifest, "calibration_hashes": hashes,
        "selection_hashes": selection_hashes, "emissions": emissions.tolist(),
        "candidate_schedules": schedules.tolist(), "candidate_changes": changes.tolist(),
        "candidate_circuit_sha256": [hashlib.sha256(str(c).encode()).hexdigest()
                                     for c in candidate_circuits],
        "state_bytes_float64": {str(w): int(np.sum(changes <= w) * 8) for w in (0, 1, 2)},
        "selected": {str(w): {"retention": selected[w][0],
                              "reset_margin": (None if np.isinf(selected[w][1]) else selected[w][1]),
                              "penalty": selected[w][2]} for w in (0, 1, 2)},
        "selection_errors": {str(w): {f"retention={c[0]},reset={c[1]},penalty={c[2]}": n
                                      for c, n in values.items()}
                             for w, values in selection_errors.items()},
        "static_selection_errors": static_errors.tolist(), "static_index": static_index,
        "stationary": stationary, "cells": cells,
    }


def summarize(records):
    keys = records[0]["cells"]
    summary = {}
    for key in keys:
        methods = records[0]["cells"][key]["methods"]
        cell = {}
        for method in methods:
            errors = sum(r["cells"][key]["methods"][method]["errors"] for r in records)
            total = sum(r["cells"][key]["records"] for r in records)
            cell[method] = {"errors": errors, "ler": errors / total,
                            "replicate_ler": [r["cells"][key]["methods"][method]["ler"]
                                              for r in records]}
        static, physical, reachable = (cell[x]["ler"] for x in
                                       ("static", "physical_oracle", "reachable_schedule"))
        for width in (0, 1, 2):
            adaptive = cell[f"adaptive_{width}"]["ler"]
            cell[f"adaptive_{width}"]["physical_oracle_gap_recovery"] = recovery(
                static, adaptive, physical)
            cell[f"adaptive_{width}"]["reachable_action_gap_recovery"] = recovery(
                static, adaptive, reachable)
        summary[key] = cell
    stationary = {}
    for name in ("stationary_a", "stationary_b"):
        stationary[name] = {}
        for method in records[0]["stationary"][name]["methods"]:
            errors = sum(r["stationary"][name]["methods"][method]["errors"] for r in records)
            total = sum(r["stationary"][name]["records"] for r in records)
            stationary[name][method] = {"errors": errors, "ler": errors / total}
    gate = summary["scale=1.00,dwell=64"]["adaptive_2"]
    static_lers = summary["scale=1.00,dwell=64"]["static"]["replicate_ler"]
    adaptive_lers = gate["replicate_ler"]
    maximum_stationary_harm_pp = max(
        100 * (stationary[name]["adaptive_2"]["ler"] - stationary[name]["static"]["ler"])
        for name in stationary)
    route_disagreements = sum(
        sum(sum(cell["pipeline_equivalence_disagreements"].values())
            for cell in r["cells"].values()) for r in records)
    criteria = {
        "nominal_dwell64_physical_oracle_recovery_at_least_0.70":
            gate["physical_oracle_gap_recovery"] is not None
            and gate["physical_oracle_gap_recovery"] >= .70,
        "maximum_stationary_harm_pp_below_0.05": maximum_stationary_harm_pp < .05,
        "positive_improvement_every_replicate":
            all(a < s for s, a in zip(static_lers, adaptive_lers)),
        "zero_action_route_disagreements": route_disagreements == 0,
    }
    return {"status": "development_switching_phase_diagram", "replicates": len(records),
            "cells": summary, "stationary": stationary,
            "gate": {"pass": all(criteria.values()), "criteria": criteria,
                     "maximum_stationary_harm_pp": maximum_stationary_harm_pp,
                     "action_route_disagreements": route_disagreements}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=2026091401)
    parser.add_argument("--calibration-shots", type=int, default=65536)
    parser.add_argument("--selection-streams", type=int, default=8)
    parser.add_argument("--selection-horizon", type=int, default=512)
    parser.add_argument("--evaluation-streams", type=int, default=16)
    parser.add_argument("--evaluation-horizon", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose an unused output path")
    if min(args.calibration_shots, args.selection_streams, args.selection_horizon,
           args.evaluation_streams, args.evaluation_horizon) < 1:
        raise ValueError("positive sample budgets required")
    sources = [__file__, "src/ptwm/schedule_inference.py", "src/ptwm/noise_schedule.py",
               "scripts/run_inferred_schedule_screen.py", "scripts/run_surface_frontier_challenge.py",
               "docs/switching-schedule-phase-diagram-protocol.md"]
    output = {
        "protocol": "docs/switching-schedule-phase-diagram-protocol.md",
        "root_seed": args.seed,
        "config": {"calibration_shots": args.calibration_shots,
                   "selection_streams": args.selection_streams,
                   "selection_horizon": args.selection_horizon,
                   "evaluation_streams": args.evaluation_streams,
                   "evaluation_horizon": args.evaluation_horizon,
                   "selection_dwells": SELECTION_DWELLS,
                   "evaluation_dwells": EVALUATION_DWELLS,
                   "mismatch_scales": MISMATCH_SCALES,
                   "retentions": RETENTIONS,
                   "reset_margins": [None if np.isinf(x) else x for x in RESET_MARGINS],
                   "penalties": PENALTIES, "schedule_names": SCHEDULE_NAMES},
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "stim": stim.__version__,
                     "pymatching": pymatching.__version__},
        "source_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in sources}, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for replicate in args.replicates:
        output["records"].append(run_replicate(args, replicate))
        output["summary"] = summarize(output["records"])
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": output["summary"]["gate"],
                      "nominal_dwell64": output["summary"]["cells"]["scale=1.00,dwell=64"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
