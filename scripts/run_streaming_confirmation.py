#!/usr/bin/env python3
"""Fresh parameter-family confirmation for delay-aware streaming control."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.streaming import benchmark_estimators, fit_gaussian_hmm, simulate_switching_streams  # noqa: E402


def one_seed(spec: dict, seed: int) -> dict:
    transition = np.array([[0.995, 0.005], [spec["p10"], 1 - spec["p10"]]])
    means = (-spec["amplitude"], spec["amplitude"])
    common = dict(transition=transition, means=means, artifact_probability=spec["artifact_probability"])
    train = simulate_switching_streams(seed=seed, n_streams=10, length=1000, **common)
    fitted = fit_gaussian_hmm(train["observations"])
    test = simulate_switching_streams(seed=100_000 + seed, n_streams=30, length=1000, **common)
    delays = sorted({0, *[max(1, round(frac / spec["p10"])) for frac in (0.1, 0.25, 0.5, 1.0)]})
    return {"seed": seed, "delays": benchmark_estimators(test, fitted, delays)}


def interval(values: np.ndarray) -> list[float]:
    if np.allclose(values, values[0]):
        return [float(values[0]), float(values[0])]
    lo, hi = t.interval(0.95, len(values) - 1, loc=values.mean(), scale=values.std(ddof=1) / np.sqrt(len(values)))
    return [float(lo), float(hi)]


def run(workers: int) -> dict:
    specs = [
        {"id": f"snr{amp}_dwell{round(1/p10)}_art{art}", "amplitude": amp, "p10": p10, "artifact_probability": art}
        for amp in (0.4, 0.8) for p10 in (0.02, 0.06) for art in (0.005, 0.02)
    ]
    tasks = [(spec, instance * 10_000 + seed) for instance, spec in enumerate(specs) for seed in range(15)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda_pair, tasks))
    output = []
    for instance, spec in enumerate(specs):
        group = rows[instance * 15:(instance + 1) * 15]
        delay_keys = sorted(group[0]["delays"])
        summaries = {}
        for delay in delay_keys:
            development = group[:5]
            candidates = ["instantaneous", "ewma", "hmm_filter_current"]
            baseline = min(candidates, key=lambda name: np.mean([r["delays"][delay][name]["control_mse"] for r in development]))
            confirm = group[5:]
            b = np.array([r["delays"][delay][baseline]["control_mse"] for r in confirm])
            f = np.array([r["delays"][delay]["hmm_delay_forecast"]["control_mse"] for r in confirm])
            relative = (b - f) / b
            summaries[str(delay)] = {
                "baseline_frozen_on_development": baseline,
                "baseline_mse": float(b.mean()),
                "forecast_mse": float(f.mean()),
                "relative_reduction_mean": float(relative.mean()),
                "relative_reduction_95_t_interval": interval(relative),
            }
        output.append({"spec": spec, "results": summaries})
    # Use the delay nearest one quarter of each fault dwell time.
    effects = []
    for row in output:
        target = 0.25 / row["spec"]["p10"]
        key = min(row["results"], key=lambda k: abs(int(k) - target))
        effects.append(row["results"][key]["relative_reduction_mean"])
    effects = np.asarray(effects)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": {"instances": len(specs), "development_seeds_per_instance": 5, "confirmation_seeds_per_instance": 10, "workers": workers},
        "instances": output,
        "quarter_dwell_effect": {"mean": float(effects.mean()), "95_t_interval_across_instances": interval(effects), "all_instances_above_20_percent": bool(np.all(effects > 0.2))},
    }


def lambda_pair(pair: tuple[dict, int]) -> dict:
    return one_seed(*pair)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/streaming_confirmation.json"))
    args = parser.parse_args()
    payload = run(args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["quarter_dwell_effect"], indent=2))
