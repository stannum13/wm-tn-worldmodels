#!/usr/bin/env python3
"""Validate schedule compilation and measure known-schedule decoding opportunity."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pymatching

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.noise_schedule import compile_noise_schedule, tick_schedule
from ptwm.time_templates import splice_noise
from scripts.run_surface_frontier_challenge import REGIMES, Seeds, circuit, data_hash


def decode(c, detectors):
    matching = pymatching.Matching.from_detector_error_model(
        c.detector_error_model(decompose_errors=True), enable_correlations=True)
    return matching.decode_batch(detectors, enable_correlations=True)[:, 0].astype(bool)


def cases(a, b):
    ticks = sum(op.name == "TICK" for op in a.flattened())
    for initial, final, name in ((0, 1, "ab"), (1, 0, "ba")):
        templates = [a, b]
        for fraction in (1 / 3, .5, 2 / 3):
            reference = splice_noise(templates[initial], templates[final], fraction)
            schedule = tick_schedule(ticks, [(int(ticks * fraction), final)], initial=initial)
            midpoint = splice_noise(templates[initial], templates[final], .5)
            yield f"{name}:{fraction:.6f}", schedule, reference, midpoint
    schedule = tick_schedule(ticks, [(ticks // 3, 1), (int(ticks * 2 / 3), 0)], initial=0)
    reference = splice_noise(splice_noise(a, b, 1 / 3), a, 2 / 3)
    yield "aba:thirds", schedule, reference, splice_noise(a, b, .5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--shots", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=2026090912)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose an unused output path")
    if args.shots < 1 or len(set(args.distances)) != len(args.distances):
        raise ValueError("positive shot count and distinct distances required")
    sources = [__file__, "src/ptwm/noise_schedule.py", "src/ptwm/time_templates.py",
               "scripts/run_surface_frontier_challenge.py", "scripts/run_surface_regime_switch.py",
               "docs/directional-feature-screen-protocol.md"]
    output = {"protocol": "docs/directional-feature-screen-protocol.md", "root_seed": args.seed,
              "shots_per_condition": args.shots, "status": "known_schedule_only_no_inference_claim",
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
              "records": [], "seed_manifests": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for distance in args.distances:
        a, b = [circuit(distance, *rates) for rates in REGIMES]
        seeds = Seeds(args.seed, 0, distance)
        for name, schedule, reference, midpoint in cases(a, b):
            compiled = compile_noise_schedule([a, b], schedule)
            if compiled != reference or compiled.without_noise() != a.flattened().without_noise():
                raise AssertionError("schedule compiler changed the reference circuit")
            if compiled.get_detector_coordinates() != a.get_detector_coordinates():
                raise AssertionError("detector interface changed")
            d, y = compiled.compile_detector_sampler(seed=seeds.draw("schedule_screen", name)).sample(
                args.shots, separate_observables=True)
            y = y[:, 0]
            predictions = {key: decode(c, d) for key, c in (
                ("known_schedule", compiled), ("midpoint", midpoint), ("stationary_a", a), ("stationary_b", b))}
            actual = predictions["known_schedule"] != y
            methods = {}
            for key, prediction in predictions.items():
                loss = prediction != y
                methods[key] = {"ler": float(loss.mean()), "errors": int(loss.sum()),
                                "rescued_by_known_schedule": int(np.sum(loss & ~actual)),
                                "harmed_by_known_schedule": int(np.sum(~loss & actual))}
            output["records"].append({"distance": distance, "condition": name,
                "schedule": schedule.tolist(), "reference_circuit_equal": True,
                "compiled_circuit_sha256": hashlib.sha256(str(compiled).encode()).hexdigest(),
                "data_hash": data_hash(d, y), "methods": methods})
            print(f"d{distance} {name}: " + ", ".join(f"{k}={v['ler']:.6f}" for k, v in methods.items()), flush=True)
            output["seed_manifests"][str(distance)] = seeds.manifest
            output["evaluated_shots"] = len(output["records"]) * args.shots
            args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
