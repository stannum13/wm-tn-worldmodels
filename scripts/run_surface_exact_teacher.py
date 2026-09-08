#!/usr/bin/env python3
"""Use a distance-3 exact DEM teacher to separate state and decoding losses."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.exact_dem import bayes_modes, exact_distribution, fault_terms, syndrome_indices
from ptwm.native_surface import fwht
from scripts.benchmark_surface_frontier import restore_model, restore_static
from scripts.run_surface_frontier_challenge import (
    REGIMES, Seeds, circuit, conditional_interval, data_hash, decode_endpoints,
    family_predictions, make_decoder, probability, stream,
)
from ptwm.surface_program import LocalFeatures


def characteristic(terms, masks):
    values = np.ones(len(masks))
    for mask, p in terms:
        values *= np.where(np.bitwise_count(np.asarray(masks, dtype=np.uint64) & np.uint64(mask)) % 2,
                           1 - 2 * p, 1)
    return values


def validate_table(c, dem, table, seeds, mode, *, shots):
    bits = dem.num_detectors + dem.num_observables
    rng = np.random.default_rng(seeds.draw("teacher_moments", mode))
    masks = np.unique(np.concatenate((1 << np.arange(bits, dtype=np.uint64),
                                     rng.integers(1, 1 << bits, size=64, dtype=np.uint64))))
    transform = fwht(table.copy())
    target = characteristic(fault_terms(dem), masks)
    roundtrip = float(np.max(np.abs(transform[masks] - target)))
    if roundtrip > 1e-9:
        raise ArithmeticError("teacher Fourier moment roundtrip failed")
    del transform
    samples = {}
    physical_d, physical_y = c.compile_detector_sampler(seed=seeds.draw("moment_circuit", mode)).sample(
        shots, separate_observables=True)
    dem_d, dem_y, _ = dem.compile_sampler(seed=seeds.draw("moment_dem", mode)).sample(shots)
    for name, d, y in (("physical_circuit", physical_d, physical_y), ("dem", dem_d, dem_y)):
        labels = np.column_stack((d, y))
        x = syndrome_indices(labels).astype(np.uint64)
        empirical = np.asarray([1 - 2 * np.mean(np.bitwise_count(x & mask) % 2) for mask in masks])
        stderr = np.sqrt(np.maximum(1 - target**2, 1 / shots) / shots)
        z = (empirical - target) / stderr
        samples[name] = {"max_absolute_z": float(np.max(np.abs(z))),
                         "moments": empirical.tolist(), "z": z.tolist()}
        if np.max(np.abs(z)) > 6:
            raise ArithmeticError(f"{name} disagrees with DEM teacher moments")
    return {"fourier_roundtrip_max_abs_error": roundtrip, "masks": masks.tolist(),
            "predicted_moments": target.tolist(), "validation_samples_per_source": shots, "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--moment-shots", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=2026090803)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    artifact = json.loads(args.reference.read_text())
    if artifact["distance"] != 3:
        raise ValueError("this offline teacher is restricted to distance 3")
    seeds = Seeds(args.seed, 0, 3)
    cs = [circuit(3, *rates) for rates in REGIMES]
    distributions, diagnostics = [], []
    for mode, c in enumerate(cs):
        dem = c.detector_error_model()
        decomposed = c.detector_error_model(decompose_errors=True)
        def log_terms(value):
            terms = {}
            for mask, p in fault_terms(value):
                terms[mask] = terms.get(mask, 0.) + np.log1p(-2*p)
            return terms
        raw_terms, separated_terms = log_terms(dem), log_terms(decomposed)
        if set(raw_terms) != set(separated_terms):
            raise AssertionError("decomposition changed full fault-mask support")
        decomposition_error = max(abs(raw_terms[k] - separated_terms[k]) for k in raw_terms)
        if decomposition_error > 1e-12:
            raise AssertionError("decomposition changed full correlated fault probabilities")
        table, diagnostic = exact_distribution(dem)
        diagnostic["decomposition_log_characteristic_max_error"] = decomposition_error
        diagnostic["validation"] = validate_table(c, dem, table, seeds, mode, shots=args.moment_shots)
        distributions.append(table.reshape(2, 1 << c.num_detectors))
        diagnostics.append(diagnostic)
        print(f"mode {mode}: exact DEM table validated, {time.perf_counter()-started:.1f}s", flush=True)
    tables = np.stack(distributions)
    del distributions
    exact_memoryless_risk = .5 * np.minimum(tables[0, 0] + tables[1, 0], tables[0, 1] + tables[1, 1]).sum()
    exact_mode_known_risk = .5 * np.minimum(tables[:, 0], tables[:, 1]).sum()
    mapper = LocalFeatures.from_circuit(cs[0])
    evidence = restore_model(artifact["models"]["evidence"])
    heads = {k: restore_model(v) for k, v in artifact["models"]["actions"].items()}
    static = restore_static(artifact, artifact["selected"]["static"])
    endpoints = {backend: [make_decoder("endpoint", c.detector_error_model(decompose_errors=True),
                                       backend == "corr", {}) for c in cs] for backend in ("mwpm", "corr")}
    results = {}
    for condition in ("nominal", "iid"):
        d, y, modes = stream(3, seeds=seeds, role="teacher_test", condition=condition,
                            streams=args.streams, horizon=args.horizon)
        indices = syndrome_indices(d).reshape(args.streams, args.horizon)
        memoryless, _ = bayes_modes(tables, indices, temporal=False)
        temporal, _ = bayes_modes(tables, indices, temporal=True)
        known_mode = tables[modes, 1, indices] > tables[modes, 0, indices]
        features = mapper.transform(d)
        probabilities = {name: probability(features, evidence, args.streams, args.horizon, flag)
                         for name, flag in (("temporal", True), ("memoryless", False))}
        decoded = {k: decode_endpoints(v, d) for k, v in endpoints.items()}
        family = family_predictions(features, probabilities, decoded, heads, {artifact["selected"]["compiled"]})
        compiled_name = artifact["selected"]["compiled"]
        compiled = (static.decode(d) if compiled_name.startswith("static:") else family[compiled_name])
        predictions = {"static_correlated": static.decode(d).reshape(indices.shape),
                       "selected_compiled": compiled.reshape(indices.shape),
                       "exact_memoryless": memoryless, "exact_causal_frozen_transition": temporal,
                       "exact_known_mode": known_mode}
        if condition == "iid":
            predictions["exact_causal_correct_transition"] = bayes_modes(tables, indices, temporal=True,
                                                                          switch_probability=.5)[0]
        truth = y.reshape(indices.shape)
        errors = {name: (prediction != truth).mean(axis=1) for name, prediction in predictions.items()}
        methods = {}
        for name, values in errors.items():
            methods[name] = {"ler": float(values.mean()), "errors_per_stream": (values * args.horizon).astype(int).tolist(),
                             "delta_vs_memoryless": float((values - errors["exact_memoryless"]).mean()),
                             "paired_95pct_interval_vs_memoryless": conditional_interval(values - errors["exact_memoryless"]),
                             "paired_95pct_interval_vs_compiled": conditional_interval(values - errors["selected_compiled"])}
        failures = predictions["static_correlated"] != truth
        endpoint_pair = decoded["corr"][0]
        pair_fixes = np.logical_or(endpoint_pair[0] == y, endpoint_pair[1] == y).reshape(truth.shape)
        teacher_fixes = temporal == truth
        results[condition] = {"records": int(y.size), "data_hash": data_hash(d, y, modes), "methods": methods,
                              "static_failures": int(failures.sum()),
                              "static_failures_fixable_by_endpoint_pair": int((failures & pair_fixes).sum()),
                              "static_failures_fixed_by_exact_causal_teacher": int((failures & teacher_fixes).sum()),
                              "static_failures_teacher_fixes_but_endpoint_pair_cannot": int((failures & teacher_fixes & ~pair_fixes).sum())}
        print(condition, {k: round(v["ler"], 6) for k, v in methods.items()}, flush=True)
    output = {"reference_artifact": str(args.reference), "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "seed": args.seed, "seeds": seeds.manifest, "code_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(), "stim_version": stim.__version__,
              "diagnostics": diagnostics, "exact_stationary_memoryless_bayes_risk": float(exact_memoryless_risk),
              "exact_stationary_known_mode_bayes_risk": float(exact_mode_known_risk), "conditions": results,
              "claim": "Known-parameter exact DEM diagnostic at d3; not a deployable learned decoder or a general SOTA result.",
              "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
