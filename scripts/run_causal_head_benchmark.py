#!/usr/bin/env python3
"""Benchmark cheap multi-rate causal heads outside the per-sample hot path."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.streaming import (  # noqa: E402
    StreamParameters,
    causal_context_features,
    causal_hmm_filter,
    fit_causal_head,
    fit_gaussian_hmm,
    forecast_belief,
    score_delayed_prediction,
    simulate_switching_streams,
    slow_head_denoiser,
    stationary_distribution,
)


def run(*, seeds: int, delay: int, strides: list[int]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    # Deliberately use the difficult regime found by the confirmation screen.
    process = dict(
        transition=np.array([[0.995, 0.005], [0.02, 0.98]]),
        means=(-0.4, 0.4),
        artifact_probability=0.02,
    )
    for seed in range(seeds):
        train = simulate_switching_streams(
            seed=seed, n_streams=16, length=1200, **process
        )
        params = fit_gaussian_hmm(train["observations"], clip_quantile=0.98)
        train_belief = causal_hmm_filter(
            train["observations"], params, log_likelihood_ratio_clip=3.0
        )
        test = simulate_switching_streams(
            seed=100_000 + seed, n_streams=40, length=1200, **process
        )
        test_belief = causal_hmm_filter(
            test["observations"], params, log_likelihood_ratio_clip=3.0
        )
        base = forecast_belief(test_belief, params.transition, delay)
        base_metrics = score_delayed_prediction(
            base, test["states"], delay=delay, artifacts=test["artifacts"]
        )
        rows.append({"seed": seed, "model": "hmm_forecast", "stride": 1,
                     "head_parameters": 0, "head_ns_per_update": 0.0,
                     "amortized_head_ns_per_sample": 0.0, **base_metrics})
        prior = np.full_like(base, stationary_distribution(params.transition)[1])
        prior_metrics = score_delayed_prediction(
            prior, test["states"], delay=delay, artifacts=test["artifacts"]
        )
        rows.append({"seed": seed, "model": "stationary_prior", "stride": 1,
                     "head_parameters": 0, "head_ns_per_update": 0.0,
                     "amortized_head_ns_per_sample": 0.0, **prior_metrics})

        for stride in strides:
            train_x, train_y, _ = causal_context_features(
                train["observations"], train_belief, window=32, stride=stride,
                delay=delay, states=train["states"]
            )
            test_x, _, locations = causal_context_features(
                test["observations"], test_belief, window=32, stride=stride,
                delay=delay
            )
            for name, knots in (("linear_head", 1), ("spline_kan_head", 6)):
                head = fit_causal_head(train_x, train_y, knots=knots)
                started = perf_counter_ns()
                head_values = head.predict(test_x)
                elapsed = perf_counter_ns() - started
                action = slow_head_denoiser(
                    base, head_values, locations, stride=stride,
                    mix=0.5, smoothing=0.25
                )
                metrics = score_delayed_prediction(
                    action, test["states"], delay=delay,
                    artifacts=test["artifacts"]
                )
                rows.append({
                    "seed": seed,
                    "model": name,
                    "stride": stride,
                    "head_parameters": int(head.weights.size),
                    "head_ns_per_update": float(elapsed / len(test_x)),
                    "amortized_head_ns_per_sample": float(elapsed / test["observations"].size),
                    **metrics,
                })

    aggregate: list[dict[str, object]] = []
    keys = sorted({(row["model"], row["stride"]) for row in rows})
    for model, stride in keys:
        group = [row for row in rows if row["model"] == model and row["stride"] == stride]
        aggregate.append({
            "model": model,
            "stride": stride,
            **{key: float(np.mean([row[key] for row in group])) for key in (
                "head_parameters", "head_ns_per_update", "amortized_head_ns_per_sample",
                "brier_loss", "classification_error", "false_action_rate",
                "missed_fault_rate", "action_on_artifact_rate"
            )},
        })
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "design": {
            "seeds": seeds, "delay": delay, "strides": strides,
            "train_streams_per_seed": 16, "test_streams_per_seed": 40,
            "stream_length": 1200, "context_window": 32,
            "process": {"p01": 0.005, "p10": 0.02, "emission_amplitude": 0.4,
                        "artifact_probability": 0.02},
            "target_access": "simulator state at delayed actuation time; privileged supervised diagnostic",
            "hot_path": "held head value fused by fixed scalar mix and one-pole denoiser",
        },
        "per_seed": rows,
        "aggregate": aggregate,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--delay", type=int, default=25)
    parser.add_argument("--strides", default="1,8,32,128")
    parser.add_argument("--output", type=Path, default=Path("results/causal_head_benchmark.json"))
    args = parser.parse_args()
    payload = run(seeds=args.seeds, delay=args.delay,
                  strides=[int(value) for value in args.strides.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2))
