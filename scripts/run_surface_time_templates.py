#!/usr/bin/env python3
"""Prospective AA/AB/BA/BB graph-action benchmark with complete independent refits."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pymatching
import scipy
import stim
from scipy.optimize import minimize
from scipy.special import softmax

from ptwm.surface_program import LocalFeatures
from ptwm.time_templates import (approximate_risks, choose_action, difficulty_bins,
    fit_mode_logits, fit_risk_tables, mode_logits, mode_rate_filter, splice_noise)
from scripts.run_surface_frontier_challenge import REGIMES, Seeds, circuit, make_decoder, seed_value, static_family
from scripts.summarize_surface_frontier_challenge import interval

CONDITIONS = {
    "four_way": ((0, 1, 2, 3), .02, .5),
    "iid": ((0, 1, 2, 3), .75, .5),
    "endpoints": ((0, 3), .02, .5),
    "midpoint": ((1, 2), .02, .5),
    "third": ((1, 2), .02, 1 / 3),
    "two_thirds": ((1, 2), .02, 2 / 3),
    "stationary_a": ((0,), 0., .5),
    "stationary_b": ((3,), 0., .5),
}
RULES = ("mode", "global", "binned")
PAIRINGS = (("four", "static"), ("four", "matched_two"), ("four", "matched_memoryless"),
            ("four", "two"), ("four", "memoryless"),
            ("four", "actual_mode_template"), ("four", "fixed"), ("compiled", "static"))


def circuits(fraction=.5):
    a, b = [circuit(5, *rates) for rates in REGIMES]
    return [a, splice_noise(a, b, fraction), splice_noise(b, a, fraction), b]


def validate_template_support(cs):
    signatures = [[(op.type, str(op.targets_copy()), op.args_copy() if op.type != "error" else None)
                   for op in c.detector_error_model(decompose_errors=True).flattened()] for c in cs]
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError("template DEM support or correlation metadata differ")
    return hashlib.sha256(json.dumps(signatures[0]).encode()).hexdigest()


def digest_data(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        h.update(json.dumps([value.shape, str(value.dtype)]).encode())
        h.update(value.tobytes())  # Multiclass mode values must not be packed as booleans.
    return h.hexdigest()


def generate(seeds, role, condition, streams, horizon):
    active, rate, fraction = CONDITIONS[condition]
    rng = np.random.default_rng(seeds.draw(role, condition, "modes"))
    index = np.empty((streams, horizon), dtype=np.uint8)
    index[:, 0] = rng.integers(len(active), size=streams)
    for step in range(1, horizon):
        jump = rng.integers(1, len(active), size=streams) if len(active) > 1 else 0
        index[:, step] = (index[:, step - 1] + (rng.random(streams) < rate) * jump) % len(active)
    modes = np.asarray(active, dtype=np.uint8)[index]
    cs = circuits(fraction)
    d = np.empty((streams * horizon, cs[0].num_detectors), dtype=bool)
    y = np.empty(streams * horizon, dtype=bool)
    for mode, c in enumerate(cs):
        sampler = c.compile_detector_sampler(seed=seeds.draw(role, condition, "stim", mode))
        where = np.flatnonzero(modes.ravel() == mode)
        detectors, labels = sampler.sample(len(where), separate_observables=True)
        d[where], y[where] = detectors, labels[:, 0]
    return d, y, modes


def temporal_static_fit(calibration, max_evaluations):
    trace = []

    def objective(log_rates):
        rates = np.exp(log_rates)
        c = splice_noise(circuit(5, *rates[:2]), circuit(5, *rates[2:]), .5)
        decoder = make_decoder("fitting", c.detector_error_model(decompose_errors=True), True, {})
        value = float(np.mean([np.mean(decoder.decode(d[:16384]) != y[:16384]) for d, y in calibration]))
        trace.append({"rates": rates.tolist(), "ler": value})
        return value

    starts = []
    for raw in ([*REGIMES[0], *REGIMES[1]], [*REGIMES[1], *REGIMES[0]]):
        initial = np.log(raw)
        simplex = np.vstack((initial, initial + .2 * np.eye(4)))
        result = minimize(objective, initial, method="Nelder-Mead", bounds=[(np.log(.0003), np.log(.03))] * 4,
                          options={"maxfev": max_evaluations // 2, "initial_simplex": simplex})
        starts.append({"initial_rates": raw, "success": bool(result.success), "message": str(result.message)})
    best = min(trace, key=lambda row: row["ler"])
    c = splice_noise(circuit(5, *best["rates"][:2]), circuit(5, *best["rates"][2:]), .5)
    decoder = make_decoder("corr_optimized_time", c.detector_error_model(decompose_errors=True), True, best)
    return decoder, {"starts": starts, "trace": trace}


def posterior_families(features, evidence, streams, horizon):
    logits = mode_logits(features, evidence).reshape(streams, horizon, 4)
    bank, rates = mode_rate_filter(logits)
    fixed, _ = mode_rate_filter(logits, rates=(.02,))
    return {"bank": bank.reshape(-1, 4), "fixed": fixed.reshape(-1, 4),
            "memoryless": softmax(logits, axis=2).reshape(-1, 4)}, rates


def policy_choices(probabilities, tables, bins):
    return {f"{history}:{width}:{rule}": choose_action(p, tables, bins, rule, actions)
            for history, p in probabilities.items() for width, actions in (("four", (0, 1, 2, 3)), ("two", (0, 3)))
            for rule in RULES}


def select_rule(scores, prefix):
    return min((key for key in scores if key.startswith(prefix)),
               key=lambda key: (scores[key], RULES.index(key.split(":")[-1])))


def summarize_risk_calibration(p, tables, bins, choices, errors, rule):
    confidence = np.searchsorted([.5, .8], p.max(axis=1), side="right")
    risk = approximate_risks(p, tables, bins, rule)[np.arange(len(p)), choices]
    return [{"count_bin": int(b), "confidence_bin": int(c), "records": int(mask.sum()),
             "predicted": float(risk[mask].mean()), "observed": float(errors[mask].mean())}
            for b in range(4) for c in range(3) if np.any(mask := ((bins == b) & (confidence == c)))]


def run(args, replicate):
    started = time.perf_counter()
    source_paths = [__file__, "src/ptwm/time_templates.py", "src/ptwm/surface_program.py",
                    "scripts/run_surface_frontier_challenge.py", "scripts/run_surface_regime_switch.py",
                    "scripts/summarize_surface_frontier_challenge.py", "docs/surface-time-template-protocol.md"]
    sources = {str(Path(path).resolve().relative_to(Path.cwd())): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths}
    start_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    seeds = Seeds(args.seed, replicate, 5)
    cs = circuits()
    support_hash = validate_template_support(cs)
    feature_map = LocalFeatures.from_circuit(cs[0])
    graphs = [make_decoder(str(i), c.detector_error_model(decompose_errors=True), True, {}) for i, c in enumerate(cs)]
    hashes, calibration, emission_groups = {}, [], []
    for mode, c in enumerate(cs):
        for role in ("time_evidence", "time_risk"):
            d, y = c.compile_detector_sampler(seed=seeds.draw(role, "stim", mode)).sample(
                args.calibration_shots, separate_observables=True)
            hashes[f"{role}_{mode}"] = digest_data(d, y)
            if role == "time_evidence":
                emission_groups.append(feature_map.transform(d))
            else:
                calibration.append((d, y[:, 0]))
    evidence = fit_mode_logits(emission_groups)
    del emission_groups
    risk_errors = [np.column_stack([g.decode(d) != y for g in graphs]) for d, y in calibration]
    tables = fit_risk_tables(risk_errors, [difficulty_bins(d) for d, _ in calibration])
    risk_diagnostics = []
    for mode, ((d, _), loss) in enumerate(zip(calibration, risk_errors)):
        bins = difficulty_bins(d)
        for b in range(4):
            mask = bins == b
            risk_diagnostics.append({"mode": mode, "count_bin": b, "records": int(mask.sum()),
                "shrinkage_weight": float(100 / (mask.sum() + 100)),
                "observed_action_risks": loss[mask].mean(axis=0).tolist() if np.any(mask) else None,
                "fitted_action_risks": tables["binned"][mode, :, b].tolist()})
    static, optimizer = static_family(5, calibration, max_evaluations=args.optimizer_evaluations)
    for correlated in (False, True):
        for mode in (1, 2):
            name = f"{'corr' if correlated else 'mwpm'}_time_{mode}"
            static[name] = make_decoder(name, cs[mode].detector_error_model(decompose_errors=True), correlated, {})
    extra, optimizer["corr_optimized_time"] = temporal_static_fit(calibration, args.time_optimizer_evaluations)
    static[extra.name] = extra
    del calibration
    print(f"r{replicate}: calibration and {len(static)} static candidates ready ({time.perf_counter()-started:.1f}s)", flush=True)
    selected, selection_scores, results, selection_diagnostics = {}, {}, {}, {}
    for role, conditions in (("time_selection", ("four_way",)), ("time_test", tuple(CONDITIONS))):
        for condition in conditions:
            d, y, modes = generate(seeds, role, condition, args.streams, args.horizon)
            hashes[f"{role}_{condition}"] = digest_data(d, y, modes)
            p, rate = posterior_families(feature_map.transform(d), evidence, args.streams, args.horizon)
            bins = difficulty_bins(d)
            choices = policy_choices(p, tables, bins)
            decoded = np.column_stack([g.decode(d) for g in graphs])
            row_index = np.arange(len(y))
            if role == "time_selection":
                selection_scores = {name: float(np.mean(decoded[row_index, action] != y)) for name, action in choices.items()}
                static_scores = {name: float(np.mean(g.decode(d) != y)) for name, g in static.items()}
                selected = {"four": select_rule(selection_scores, "bank:four:"),
                            "two": select_rule(selection_scores, "bank:two:"),
                            "memoryless": select_rule(selection_scores, "memoryless:four:"),
                            "fixed": select_rule(selection_scores, "fixed:four:"),
                            "static": min(static_scores, key=lambda n: (static_scores[n], n))}
                selected["compiled"] = (selected["four"] if selection_scores[selected["four"]] < static_scores[selected["static"]]
                                        else "static:" + selected["static"])
                chosen = choices[selected["four"]]
                selected_loss = decoded[row_index, chosen] != y
                selection_diagnostics = {"risk_calibration": summarize_risk_calibration(p["bank"], tables, bins, chosen,
                    selected_loss, selected["four"].split(":")[-1]),
                    "regret_vs_unattainable_best_action": float(selected_loss.mean() - np.all(decoded != y[:, None], axis=1).mean())}
                selection_scores.update({"static:" + k: v for k, v in static_scores.items()})
                print(f"r{replicate}: selected {selected}", flush=True)
                continue
            predictions = {name: decoded[row_index, choices[selected[name]]] for name in ("four", "two", "memoryless", "fixed")}
            rule = selected["four"].split(":")[-1]
            predictions["matched_two"] = decoded[row_index, choices[f"bank:two:{rule}"]]
            predictions["matched_memoryless"] = decoded[row_index, choices[f"memoryless:four:{rule}"]]
            predictions["static"] = static[selected["static"]].decode(d)
            predictions["compiled"] = predictions["static"] if selected["compiled"].startswith("static:") else predictions["four"]
            predictions["nominal_mode_template"] = decoded[row_index, modes.ravel()]
            best_mode_actions = tables["global"].argmin(axis=1)[modes.ravel()]
            predictions["calibrated_known_mode_action"] = decoded[row_index, best_mode_actions]
            predictions["best_of_four_outcome"] = np.where(np.any(decoded == y[:, None], axis=1), y, ~y)
            if condition in ("third", "two_thirds"):
                actual_graphs = [make_decoder(str(i), c.detector_error_model(decompose_errors=True), True, {})
                                 for i, c in enumerate(circuits(CONDITIONS[condition][2]))]
                actual = np.empty_like(y)
                for mode in (1, 2):
                    where = np.flatnonzero(modes.ravel() == mode)
                    actual[where] = actual_graphs[mode].decode(d[where])
                predictions["actual_mode_template"] = actual
            else:
                predictions["actual_mode_template"] = predictions["nominal_mode_template"]
            loss = {name: prediction != y for name, prediction in predictions.items()}
            methods = {name: {"ler": float(e.mean()), "errors_per_stream": e.reshape(args.streams, args.horizon).sum(axis=1).tolist()}
                       for name, e in loss.items()}
            chosen = choices[selected["four"]]
            selective = np.empty_like(y)
            for mode, graph in enumerate(graphs):
                where = np.flatnonzero(chosen == mode)
                if len(where):
                    selective[where] = graph.decode(d[where])
            mismatches = int(np.sum(selective != predictions["four"]))
            if mismatches:
                raise AssertionError("select-before-decode changes predictions")
            confusion = np.zeros((4, 4), dtype=int)
            np.add.at(confusion, (modes.ravel(), chosen), 1)
            mode_confusion = np.zeros((4, 4), dtype=int)
            np.add.at(mode_confusion, (modes.ravel(), p["bank"].argmax(axis=1)), 1)
            by_mode = {}
            for mode in np.unique(modes):
                mask = modes.ravel() == mode
                by_mode[str(int(mode))] = {"records": int(mask.sum()),
                    "ler": {name: float(e[mask].mean()) for name, e in loss.items()}}
            results[condition] = {"methods": methods, "per_mode": by_mode,
                "discordant": {f"{a}_vs_{b}": int(np.sum(loss[a] != loss[b])) for a, b in PAIRINGS},
                "mode_action_counts": confusion.tolist(), "true_inferred_mode_counts": mode_confusion.tolist(),
                "mode_log_loss": float(-np.log(np.maximum(p["bank"][row_index, modes.ravel()], 1e-300)).mean()),
                "mean_inferred_rate": float(rate.mean()), "selective_decode_disagreements": mismatches,
                "risk_calibration": summarize_risk_calibration(p["bank"], tables, bins, chosen,
                                                               loss["four"], selected["four"].split(":")[-1])}
            print(f"r{replicate} {condition}: " + ", ".join(f"{name}={methods[name]['ler']:.6f}" for name in ("static", "two", "four", "memoryless")), flush=True)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in sources.items()):
        raise RuntimeError("source changed during experiment")
    return {"protocol": "docs/surface-time-template-protocol.md", "root_seed": args.seed, "replicate": replicate,
        "distance": 5, "config": {k: getattr(args, k) for k in ("streams", "horizon", "calibration_shots", "optimizer_evaluations", "time_optimizer_evaluations")},
        "code_commit": start_commit, "source_sha256": sources, "template_support_sha256": support_hash,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "stim": stim.__version__, "pymatching": pymatching.__version__, "python": platform.python_version()},
        "selected": selected, "selection_scores": selection_scores, "static_candidates": len(static),
        "static_parameters": {name: g.parameters for name, g in static.items()}, "optimizer": optimizer,
        "evidence_model": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in evidence.items()},
        "risk_tables": {k: v.tolist() for k, v in tables.items()}, "seeds": seeds.manifest, "data_hashes": hashes,
        "risk_calibration_diagnostics": risk_diagnostics, "selection_diagnostics": selection_diagnostics,
        "controller_resources": {"persistent_probability_bytes": 20 * 8, "graph_templates": 4,
            "evidence_array_bytes": int(sum(v.nbytes for v in evidence.values() if isinstance(v, np.ndarray))),
            "risk_table_bytes": int(sum(v.nbytes for v in tables.values())),
            "limits": "Float64 array storage only; decoder graphs, object overhead, work buffers and measured latency excluded. "
                      "The benchmark vectorizes complete records; persistent bytes describe the sequential recurrence."},
        "conditions": results, "elapsed_seconds": time.perf_counter() - started}


def summarize(rows):
    identities = [r["replicate"] for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate replicate")
    expected_keys = [(role, "stim", mode) for role in ("time_evidence", "time_risk") for mode in range(4)]
    expected_hash_keys = {f"{role}_{mode}" for role in ("time_evidence", "time_risk") for mode in range(4)}
    for role, conditions in (("time_selection", ("four_way",)), ("time_test", tuple(CONDITIONS))):
        for condition in conditions:
            expected_keys.append((role, condition, "modes"))
            expected_keys.extend((role, condition, "stim", mode) for mode in range(4))
            expected_hash_keys.add(f"{role}_{condition}")
    for row in rows:
        expected_seeds = {json.dumps(key, separators=(",", ":")): seed_value(row["root_seed"], row["replicate"], 5, *key)
                          for key in expected_keys}
        if row["seeds"] != expected_seeds or set(row["data_hashes"]) != expected_hash_keys:
            raise ValueError("incomplete or incorrect data-role/seed manifest")
        if set(row["conditions"]) != set(CONDITIONS):
            raise ValueError("incomplete test conditions")
        streams, horizon = row["config"]["streams"], row["config"]["horizon"]
        for condition, value in row["conditions"].items():
            for method in value["methods"].values():
                counts = method["errors_per_stream"]
                if len(counts) != streams or any(type(n) is not int or not 0 <= n <= horizon for n in counts):
                    raise ValueError("invalid per-stream counts")
                if sum(counts) / (streams * horizon) != method["ler"]:
                    raise ValueError("stored LER differs from paired counts")
            if condition in ("third", "two_thirds") and set(value["per_mode"]) != {"1", "2"}:
                raise ValueError("held-cutpoint test is missing an orientation")
    output = {}
    for condition in CONDITIONS:
        output[condition] = {}
        for a, b in PAIRINGS:
            av = np.array([r["conditions"][condition]["methods"][a]["ler"] for r in rows])
            bv = np.array([r["conditions"][condition]["methods"][b]["ler"] for r in rows])
            delta = av - bv
            output[condition][f"{a}_vs_{b}"] = {"candidate_ler": float(av.mean()), "reference_ler": float(bv.mean()),
                "delta": float(delta.mean()), "relative_gain": float(1 - av.mean() / bv.mean()),
                "replicate_differences": delta.tolist(), "improving_refits": int(np.sum(delta < 0)),
                "ci95": interval(delta, .95), "ci_primary": interval(delta, 1 - .05 / 3),
                "ci_transfer": interval(delta, .9875),
                "discordant": sum(r["conditions"][condition]["discordant"][f"{a}_vs_{b}"] for r in rows)}
    expected = {"streams": 256, "horizon": 512, "calibration_shots": 32768,
                "optimizer_evaluations": 32, "time_optimizer_evaluations": 64}
    complete = set(identities) == set(range(10)) and all(r["config"] == expected and r["root_seed"] == 2026090806 for r in rows)
    gates = {"complete": complete}
    if complete:
        if len({r["code_commit"] for r in rows}) != 1 or any(r["source_sha256"] != rows[0]["source_sha256"] for r in rows):
            raise ValueError("campaign source versions changed across fits")
        for source, digest in rows[0]["source_sha256"].items():
            data = subprocess.check_output(["git", "show", f"{rows[0]['code_commit']}:{source}"])
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("runtime source hash differs from recorded commit")
        if any(r["versions"] != rows[0]["versions"] or r["template_support_sha256"] != rows[0]["template_support_sha256"]
               or r["static_candidates"] != 77 for r in rows):
            raise ValueError("inconsistent versions, graph support or baseline count")
        keys = ("four_vs_static", "four_vs_matched_two", "four_vs_matched_memoryless")
        primary = output["four_way"]
        gates["primary_graph_temporal"] = all(primary[key]["ci_primary"][1] < (-.0002 if key != "four_vs_matched_memoryless" else 0)
                                                and primary[key]["improving_refits"] >= 8 for key in keys)
        gates["family_benchmark"] = all(interval(primary[key]["replicate_differences"], .975)[1] < 0
                                        for key in ("four_vs_two", "four_vs_memoryless"))
        middle = output["midpoint"]["four_vs_matched_two"]
        gates["midpoint_mechanism"] = middle["relative_gain"] >= .15 and middle["ci95"][1] < 0
        gates["safety"] = (all(output[c]["four_vs_actual_mode_template"]["ci_primary"][1] <= .0002 for c in ("stationary_a", "stationary_b"))
                           and output["iid"]["four_vs_matched_memoryless"]["ci_primary"][1] <= .0002)
        gates["held_cutpoint_transfer"] = all(output[c][key]["ci_transfer"][1] < 0
                                               for c in ("third", "two_thirds") for key in ("four_vs_static", "four_vs_matched_two"))
        gates["power_and_equivalence"] = all(primary[key]["discordant"] >= 100 for key in keys) and all(
            c["selective_decode_disagreements"] == 0 for r in rows for c in r["conditions"].values())
    all_seeds = [v for r in rows for v in r["seeds"].values()]
    all_hashes = [v for r in rows for v in r["data_hashes"].values()]
    if len(all_seeds) != len(set(all_seeds)) or len(all_hashes) != len(set(all_hashes)):
        raise ValueError("reused data or sampler seed")
    old_seeds = []
    for path in Path("results/surface_frontier_challenge").glob("r*_d*.json"):
        old_seeds.extend(json.loads(path.read_text())["seeds"].values())
    for name in ("surface_rate_adaptation", "surface_teacher_distillation"):
        path = Path("results") / f"{name}.json"
        if path.exists():
            old_seeds.extend(v for r in json.loads(path.read_text())["records"] for v in r["seeds"].values())
    path = Path("results/surface_exact_teacher.json")
    if path.exists():
        old_seeds.extend(json.loads(path.read_text())["seeds"].values())
    if set(all_seeds) & set(old_seeds):
        raise ValueError("sampler seed overlaps an earlier campaign")
    return {"gates": gates, "conditions": output, "seed_count": len(all_seeds), "dataset_count": len(all_hashes),
            "test_records": sum(r["config"]["streams"] * r["config"]["horizon"] * len(r["conditions"]) for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--seed", type=int, default=2026090806)
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--calibration-shots", type=int, default=32768)
    parser.add_argument("--optimizer-evaluations", type=int, default=32)
    parser.add_argument("--time-optimizer-evaluations", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize:
        for replicate in ([args.replicate] if args.replicate is not None else range(10)):
            target = args.output_dir / f"r{replicate:02d}.json"
            if target.exists():
                raise FileExistsError(f"refusing to overwrite previous result: {target}")
            result = run(args, replicate)
            target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.summarize:
        paths = sorted(args.output_dir.glob("r*.json"))
        result = summarize([json.loads(path.read_text()) for path in paths])
        result["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result["gates"], indent=2))


if __name__ == "__main__":
    main()
