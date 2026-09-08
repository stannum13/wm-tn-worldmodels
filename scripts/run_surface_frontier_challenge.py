#!/usr/bin/env python3
"""Prospective stronger-baseline, full-refit and frozen-transfer challenge."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymatching
import stim
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import sem, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.surface_mode import (affine_log_likelihood_ratio,
                               fit_affine_log_likelihood_ratio,
                               log_likelihood_mode_posterior)
from ptwm.surface_program import LocalFeatures, select_before_decode, update_probability
from scripts.run_surface_crossfit_compiler import energy_action_features
from scripts.run_surface_regime_switch import (
    CLIFFORD_GRID, MEASUREMENT_GRID, REGIMES, STATIONARY, TRANSITION, circuit,
)

CAMPAIGN = "surface-frontier-challenge-v1"
CONDITIONS = {
    "nominal": {"rates": REGIMES, "switch": 0.02},
    "scale_075": {"rates": tuple(tuple(0.75 * x for x in p) for p in REGIMES), "switch": 0.02},
    "scale_125": {"rates": tuple(tuple(1.25 * x for x in p) for p in REGIMES), "switch": 0.02},
    "less_separated": {"rates": ((0.002, 0.015), (0.007, 0.003)), "switch": 0.02},
    "slow": {"rates": REGIMES, "switch": 0.005},
    "fast": {"rates": REGIMES, "switch": 0.1},
    "iid": {"rates": REGIMES, "switch": 0.5},
    "within_shot": {"rates": REGIMES, "switch": 0.02, "midpoint": True},
    "stationary_a": {"rates": REGIMES, "switch": 0.0, "fixed": 0},
    "stationary_b": {"rates": REGIMES, "switch": 0.0, "fixed": 1},
}
L2_GRID = (0.001, 0.01, 0.1)


def seed_value(root_seed, replicate, distance, *key):
    semantic = [CAMPAIGN, int(root_seed), int(replicate), int(distance), *key]
    digest = hashlib.sha256(json.dumps(semantic, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class Seeds:
    def __init__(self, root_seed, replicate, distance):
        self.base = root_seed, replicate, distance
        self.manifest = {}

    def draw(self, *key):
        name = json.dumps(key, separators=(",", ":"))
        value = seed_value(*self.base, *key)
        if name in self.manifest or value in self.manifest.values():
            raise ValueError(f"seed reused: {name}")
        self.manifest[name] = value
        return value


def mixed_circuit(first, second):
    """Switch noise probabilities at a fixed middle tick; keep ideal gates identical."""
    a, b = list(first.flattened()), list(second.flattened())
    if len(a) != len(b):
        raise ValueError("circuit topologies differ")
    midpoint = sum(op.name == "TICK" for op in a) // 2
    ticks, output = 0, stim.Circuit()
    for left, right in zip(a, b):
        if left.name != right.name or left.targets_copy() != right.targets_copy():
            raise ValueError("circuit operations differ")
        output.append(left if ticks < midpoint else right)
        ticks += left.name == "TICK"
    return output


def stream(distance, *, seeds, role, condition, streams, horizon):
    config = CONDITIONS[condition]
    cs = [circuit(distance, *rates) for rates in config["rates"]]
    if config.get("midpoint"):
        cs = [mixed_circuit(cs[0], cs[1]), mixed_circuit(cs[1], cs[0])]
    rng = np.random.default_rng(seeds.draw(role, condition, "modes"))
    modes = np.empty((streams, horizon), dtype=np.uint8)
    modes[:, 0] = (rng.integers(2, size=streams) if "fixed" not in config
                   else config["fixed"])
    for step in range(1, horizon):
        modes[:, step] = modes[:, step - 1] ^ (rng.random(streams) < config["switch"])
    detectors = np.empty((streams * horizon, cs[0].num_detectors), dtype=bool)
    labels = np.empty(streams * horizon, dtype=bool)
    for mode in range(2):
        sampler = cs[mode].compile_detector_sampler(seed=seeds.draw(role, condition, "stim", mode))
        rows = np.flatnonzero(modes.ravel() == mode)
        d, y = sampler.sample(len(rows), separate_observables=True)
        detectors[rows], labels[rows] = d, y[:, 0]
    return detectors, labels, modes


def data_hash(detectors, labels, modes=None):
    h = hashlib.sha256()
    for array in (detectors, labels, modes):
        if array is not None:
            h.update(json.dumps([array.shape, str(array.dtype)]).encode())
            h.update(np.packbits(array).tobytes())
    return h.hexdigest()


def pooled_dem(first, second, fraction):
    """Interpolate aligned primitive probabilities, preserving decomposition metadata.

    The two nominal circuits have exactly the same mechanism ordering/topology.
    This is a fitted independent-mechanism approximation, not a shot-mixture DEM.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("mixture fraction must lie in [0, 1]")
    a, b = list(first.flattened()), list(second.flattened())
    if len(a) != len(b):
        raise ValueError("unaligned detector error models")
    output = stim.DetectorErrorModel()
    for left, right in zip(a, b):
        if left.type != right.type or left.targets_copy() != right.targets_copy():
            raise ValueError("unaligned mechanism targets")
        if left.type == "error":
            p = ((1 - fraction) * left.args_copy()[0] + fraction * right.args_copy()[0])
            output.append("error", p, left.targets_copy())
        else:
            if left != right:
                raise ValueError("detector metadata differ")
            output.append(left)
    return output


@dataclass
class Decoder:
    name: str
    matcher: pymatching.Matching
    correlated: bool
    parameters: dict

    def decode(self, detectors, *, weights=False):
        result = self.matcher.decode_batch(detectors, enable_correlations=self.correlated,
                                           return_weights=weights)
        if weights:
            prediction, energy = result
            return prediction[:, 0].astype(bool), np.asarray(energy, dtype=np.float32)
        return result[:, 0].astype(bool)


def make_decoder(name, dem, correlated, parameters):
    return Decoder(name, pymatching.Matching.from_detector_error_model(
        dem, enable_correlations=correlated), correlated, parameters)


def static_family(distance, calibration, *, max_evaluations=32):
    endpoint_dems = [circuit(distance, *p).detector_error_model(decompose_errors=True) for p in REGIMES]
    family, traces = {}, {}
    tuning = [(d[:16384], y[:16384]) for d, y in calibration]
    for correlated in (False, True):
        backend = "corr" if correlated else "mwpm"
        tuning_scores = {}
        for gate in CLIFFORD_GRID:
            for meas in MEASUREMENT_GRID:
                name = f"{backend}_grid_{gate}_{meas}"
                decoder = make_decoder(name, circuit(distance, gate, meas).detector_error_model(
                    decompose_errors=True), correlated, {"gate": gate, "measurement": meas})
                family[name] = decoder
                tuning_scores[name] = np.mean([np.mean(decoder.decode(d) != y) for d, y in tuning])
        for index in range(1, 10):
            frac = index / 10
            name = f"{backend}_pooled_{frac}"
            family[name] = make_decoder(name, pooled_dem(*endpoint_dems, frac), correlated,
                                        {"pooled_fraction": frac})
        initial = family[min(tuning_scores, key=tuning_scores.get)].parameters
        trace = []

        def objective(log_rates):
            gate, measurement = np.exp(log_rates)
            decoder = make_decoder("fitting", circuit(distance, gate, measurement).detector_error_model(
                decompose_errors=True), correlated, {})
            score = float(np.mean([np.mean(decoder.decode(d) != y) for d, y in tuning]))
            trace.append({"gate": float(gate), "measurement": float(measurement), "ler": score})
            return score

        initial_point = np.log([initial["gate"], initial["measurement"]])
        simplex = np.vstack((initial_point, initial_point + [0.2, 0], initial_point + [0, 0.2]))
        result = minimize(objective, initial_point, method="Nelder-Mead",
                          bounds=[(np.log(0.0003), np.log(0.03))] * 2,
                          options={"maxfev": max_evaluations, "initial_simplex": simplex})
        best = min(trace, key=lambda x: x["ler"])
        name = f"{backend}_optimized_circuit"
        family[name] = make_decoder(name, circuit(distance, best["gate"], best["measurement"])
                                    .detector_error_model(decompose_errors=True), correlated, best)
        traces[name] = {"success": bool(result.success), "message": str(result.message), "trace": trace}
        trace_pool = []

        def pool_objective(fraction):
            decoder = make_decoder("fitting", pooled_dem(*endpoint_dems, fraction), correlated, {})
            score = float(np.mean([np.mean(decoder.decode(d) != y) for d, y in tuning]))
            trace_pool.append({"pooled_fraction": float(fraction), "ler": score})
            return score

        result = minimize_scalar(pool_objective, bounds=(0.01, 0.99), method="bounded",
                                 options={"maxiter": max_evaluations, "xatol": 1e-4})
        best = min(trace_pool, key=lambda x: x["ler"])
        name = f"{backend}_optimized_pooled"
        family[name] = make_decoder(name, pooled_dem(*endpoint_dems, best["pooled_fraction"]), correlated, best)
        traces[name] = {"success": bool(result.success), "message": str(result.message), "trace": trace_pool}
    return family, traces


def probability(features, model, streams, horizon, temporal):
    llr = affine_log_likelihood_ratio(features, model).reshape(streams, horizon)
    return log_likelihood_mode_posterior(llr, TRANSITION, STATIONARY, temporal=temporal)


def decode_endpoints(endpoints, detectors):
    decoded = [decoder.decode(detectors, weights=True) for decoder in endpoints]
    return [x[0] for x in decoded], [x[1] for x in decoded]


def family_predictions(features, probabilities, endpoints, heads, names=None):
    output = {}
    for backend, (predictions, weights) in endpoints.items():
        for history, p in probabilities.items():
            prefix = f"{backend}:{history}"
            for index in range(1, 10):
                name = f"{prefix}:threshold:{index / 10:.1f}"
                if names is not None and name not in names:
                    continue
                output[name] = np.where(p.ravel() >= index / 10, predictions[1], predictions[0])
            energies = [f"{prefix}:energy:{l2}" for l2 in L2_GRID
                        if names is None or f"{prefix}:energy:{l2}" in names]
            if not energies:
                continue
            features_action = energy_action_features(features, p, weights)
            for name in energies:
                score = affine_log_likelihood_ratio(features_action, heads[name])
                output[name] = np.where(score >= 0, predictions[1], predictions[0])
    return output


def policy_key(name, scores):
    invocations = 2 if ":energy:" in name else 1
    state = int(":temporal:" in name)
    # Static candidates use zero controller parameters. Exact LER ties prefer them.
    operations = 0 if name.startswith("static:") else 1
    return scores[name], invocations, state, operations, name


def choose(names, scores):
    return min(names, key=lambda name: policy_key(name, scores))


def json_model(model):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in model.items()}


def conditional_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    width = float(t.ppf((1 + confidence) / 2, len(values) - 1) * sem(values)) if len(values) > 1 else 0.0
    return [mean - width, mean + width]


def evaluate_predictions(predictions, labels, modes, *, streams, horizon):
    truth = labels.reshape(streams, horizon)
    errors = {name: (prediction.reshape(streams, horizon) != truth).sum(axis=1)
              for name, prediction in predictions.items()}
    methods = {name: {"ler": float(counts.sum() / labels.size), "errors_per_stream": counts.tolist()}
               for name, counts in errors.items()}
    for name, counts in errors.items():
        diff = (counts - errors["static"]) / horizon
        methods[name]["delta_vs_static"] = float(diff.mean())
        methods[name]["conditional_95pct_interval_vs_static"] = conditional_interval(diff)
    return {"records": int(labels.size), "mode_1_fraction": float(modes.mean()), "methods": methods}


def run(distance, replicate, args):
    start = time.perf_counter()
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_paths = [Path(__file__), Path(__file__).parent / "run_surface_crossfit_compiler.py",
                    Path(__file__).parent / "run_surface_regime_switch.py",
                    Path(__file__).parents[1] / "src/ptwm/surface_mode.py",
                    Path(__file__).parents[1] / "src/ptwm/surface_program.py"]
    source_hashes = {str(p.relative_to(Path(__file__).resolve().parents[1])):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    seeds = Seeds(args.seed, replicate, distance)
    cs = [circuit(distance, *rates) for rates in REGIMES]
    feature_map = LocalFeatures.from_circuit(cs[0])
    calibration = []
    for mode, c in enumerate(cs):
        d, y = c.compile_detector_sampler(seed=seeds.draw("calibration", "stim", mode)).sample(
            args.calibration_shots, separate_observables=True)
        calibration.append((d, y[:, 0]))
    hashes = {f"calibration_{i}": data_hash(d, y) for i, (d, y) in enumerate(calibration)}
    evidence = fit_affine_log_likelihood_ratio(*[feature_map.transform(d) for d, _ in calibration])
    compiled_evidence = feature_map.compile(evidence)
    static, optimization = static_family(distance, calibration, max_evaluations=args.optimizer_evaluations)
    endpoints = {backend: [make_decoder(f"{backend}_endpoint_{i}", c.detector_error_model(
        decompose_errors=True), correlated, {}) for i, c in enumerate(cs)]
        for backend, correlated in (("mwpm", False), ("corr", True))}
    del calibration
    gc.collect()
    print(f"r{replicate} d{distance}: calibrated and built {len(static)} static candidates", flush=True)

    d, y, hidden = stream(distance, seeds=seeds, role="action", condition="nominal",
                     streams=args.action_streams, horizon=args.horizon)
    hashes["action"] = data_hash(d, y, hidden)
    features = feature_map.transform(d)
    probabilities = {name: probability(features, evidence, args.action_streams, args.horizon, temporal)
                     for name, temporal in (("temporal", True), ("memoryless", False))}
    heads, fits = {}, {}
    for backend, decoders in endpoints.items():
        predictions, weights = decode_endpoints(decoders, d)
        disagreements = predictions[0] != predictions[1]
        second_correct = predictions[1][disagreements] == y[disagreements]
        if not second_correct.any() or second_correct.all():
            raise ValueError("action training requires both endpoint-correct classes")
        fits[backend] = {"disagreements": int(disagreements.sum()),
                         "second_correct": int(second_correct.sum())}
        for history, p in probabilities.items():
            matrix = energy_action_features(features, p, weights)[disagreements]
            for l2 in L2_GRID:
                heads[f"{backend}:{history}:energy:{l2}"] = fit_affine_log_likelihood_ratio(
                    matrix[~second_correct], matrix[second_correct], l2=l2)
    del d, y, features, probabilities, matrix, predictions, weights
    gc.collect()

    d, y, hidden = stream(distance, seeds=seeds, role="selection", condition="nominal",
                     streams=args.selection_streams, horizon=args.horizon)
    hashes["selection"] = data_hash(d, y, hidden)
    features = feature_map.transform(d)
    probabilities = {name: probability(features, evidence, args.selection_streams, args.horizon, temporal)
                     for name, temporal in (("temporal", True), ("memoryless", False))}
    decoded = {backend: decode_endpoints(decoders, d) for backend, decoders in endpoints.items()}
    candidate_predictions = family_predictions(features, probabilities, decoded, heads)
    scores = {name: float(np.mean(prediction != y)) for name, prediction in candidate_predictions.items()}
    for name, decoder in static.items():
        scores[f"static:{name}"] = float(np.mean(decoder.decode(d) != y))
    selected = {
        "compiled": choose(scores, scores),
        "static": choose((n for n in scores if n.startswith("static:")), scores),
        "temporal": choose((n for n in scores if ":temporal:" in n), scores),
        "memoryless": choose((n for n in scores if ":memoryless:" in n), scores),
        "one_decode": choose((n for n in scores if ":temporal:threshold:" in n), scores),
        "old_grid": choose((n for n in scores if n.startswith("static:mwpm_grid_")), scores),
        "correlated_static": choose((n for n in scores if n.startswith("static:corr_")), scores),
        "correlated_temporal": choose((n for n in scores if n.startswith("corr:temporal:")), scores),
        "correlated_memoryless": choose((n for n in scores if n.startswith("corr:memoryless:")), scores),
        "correlated_compiled": choose((n for n in scores if n.startswith(("corr:", "static:corr_"))), scores),
    }
    temporal_name = selected["temporal"]
    if ":energy:" in temporal_name:
        selected["matched_memoryless"] = temporal_name.replace(":temporal:", ":memoryless:")
    else:
        prefix = temporal_name.split(":")[0] + ":memoryless:threshold:"
        selected["matched_memoryless"] = choose((n for n in scores if n.startswith(prefix)), scores)
    print(f"r{replicate} d{distance}: selected {selected}", flush=True)
    del d, y, features, probabilities, candidate_predictions, decoded
    gc.collect()
    conditions = {}
    for condition in args.conditions:
        d, y, modes = stream(distance, seeds=seeds, role="test", condition=condition,
                            streams=args.streams, horizon=args.horizon)
        hashes[f"test_{condition}"] = data_hash(d, y, modes)
        features = feature_map.transform(d)
        probabilities = {name: probability(features, evidence, args.streams, args.horizon, temporal)
                         for name, temporal in (("temporal", True), ("memoryless", False))}
        decoded = {backend: decode_endpoints(decoders, d) for backend, decoders in endpoints.items()}
        family = family_predictions(features, probabilities, decoded, heads, set(selected.values()))
        for name in set(selected.values()):
            if name.startswith("static:"):
                family[name] = static[name.removeprefix("static:")].decode(d)
        predictions = {key: family[name] for key, name in selected.items()}
        static_backend = "corr" if static[selected["static"].removeprefix("static:")].correlated else "mwpm"
        predictions["mode_informed_endpoint"] = np.where(modes.ravel() == 1,
            decoded[static_backend][0][1], decoded[static_backend][0][0])
        conditions[condition] = evaluate_predictions(predictions, y, modes,
                                                     streams=args.streams, horizon=args.horizon)
        conditions[condition]["mode_reference_semantics"] = (
            "orientation-informed nominal endpoint; not a matching shifted model"
            if condition == "within_shot" else
            "mode-informed nominal endpoint; shifted rates are not recalibrated")
        backend, _, _, threshold = selected["one_decode"].split(":")
        chosen = probabilities["temporal"].ravel() >= float(threshold)
        compiled_llr = compiled_evidence.score(d).reshape(args.streams, args.horizon)
        compiled_probability = np.empty_like(compiled_llr)
        previous = np.full(args.streams, 0.5)
        for step in range(args.horizon):
            previous = update_probability(compiled_llr[:, step], previous)
            compiled_probability[:, step] = previous
        compiled_chosen = compiled_probability.ravel() >= float(threshold)
        choices_mismatch = int(np.sum(compiled_chosen != chosen))
        if choices_mismatch:
            raise AssertionError(f"compiled causal choices differ on {choices_mismatch} shots")
        selective = select_before_decode(d, compiled_chosen, [x.matcher for x in endpoints[backend]],
                                          correlated=backend == "corr")
        mismatch = int(np.sum(selective != predictions["one_decode"]))
        if mismatch:
            raise AssertionError(f"selective graph decode differs on {mismatch} shots")
        evidence_difference = float(np.max(np.abs(compiled_llr.ravel() -
                                                  affine_log_likelihood_ratio(features, evidence))))
        conditions[condition]["selective_decode_disagreements"] = mismatch
        conditions[condition]["compiled_choice_disagreements"] = choices_mismatch
        conditions[condition]["compiled_evidence_max_abs_difference"] = evidence_difference
        if condition.startswith("stationary"):
            fixed = CONDITIONS[condition]["fixed"]
            conditions[condition]["one_decode_false_route_fraction"] = float(np.mean(chosen != fixed))
        print(f"r{replicate} d{distance} {condition}: static={conditions[condition]['methods']['static']['ler']:.6f}, "
              f"compiled={conditions[condition]['methods']['compiled']['ler']:.6f}", flush=True)
        del d, y, features, probabilities, decoded, family, predictions, selective
        gc.collect()

    for source in source_paths:
        key = str(source.relative_to(Path(__file__).resolve().parents[1]))
        if source_hashes[key] != hashlib.sha256(source.read_bytes()).hexdigest():
            raise RuntimeError(f"source changed during run: {key}")
    return {"schema_version": 1, "campaign": CAMPAIGN,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": code_commit, "source_sha256": source_hashes,
            "protocol": "docs/surface-frontier-challenge-protocol.md",
            "versions": {"stim": stim.__version__, "pymatching": pymatching.__version__, "numpy": np.__version__},
            "platform": platform.platform(), "root_seed": args.seed, "replicate": replicate, "distance": distance,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "seeds": seeds.manifest, "data_hashes": hashes, "selected": selected,
            "selection_scores": scores, "static_parameters": {k: x.parameters for k, x in static.items()},
            "optimization": optimization, "action_fits": fits,
            "models": {"evidence": json_model(evidence), "actions": {k: json_model(v) for k, v in heads.items()}},
            "program": {"constants_bytes_float64": compiled_evidence.constant_bytes,
                        "mutable_bytes_per_stream_float64": 8, "detector_terms": len(compiled_evidence.detector_weights),
                        "pair_terms": len(compiled_evidence.pair_weights)},
            "conditions": conditions, "elapsed_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--replicates", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=2026090801)
    parser.add_argument("--calibration-shots", type=int, default=65536)
    parser.add_argument("--action-streams", type=int, default=256)
    parser.add_argument("--selection-streams", type=int, default=256)
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--optimizer-evaluations", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for replicate in args.replicates:
        for distance in args.distances:
            target = args.output_dir / f"r{replicate:02d}_d{distance}.json"
            if target.exists():
                raise FileExistsError(f"refusing to overwrite {target}")
            payload = run(distance, replicate, args)
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"saved {target} in {payload['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
