"""Multi-seed robustness for the torch-based models.

Reruns the horizon split with seeds 0..n_seeds-1 for the stochastic models
(process-MPO chi2, GRU) and reports mean +/- std of log-MSE. Deterministic
models (Markov, transfer tensor) are excluded — they have no seed sensitivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_cell
from .models import GRUModel, ProcessMPO
from .splits import horizon_split


def run(root: str, length_cap: int, idle: int, out_dir: str,
        n_seeds: int = 3, epochs: int = 60) -> dict:
    ds = load_cell(root, length_cap, idle)
    split = horizon_split(ds, train_cap=20)
    y_true = np.array([np.log(max(ep.fidelity, 1e-12)) for ep in split.test])

    results = {
        "cell": {"length_cap": length_cap, "idle": idle},
        "split": split.name,
        "n_seeds": n_seeds,
        "models": {},
    }
    for name, factory in [
        ("process_mpo_chi2", lambda seed: ProcessMPO(chi=2, epochs=epochs, seed=seed)),
        ("gru_h2", lambda seed: GRUModel(hidden=2, epochs=epochs, seed=seed)),
    ]:
        scores = []
        for seed in range(n_seeds):
            model = factory(seed)
            model.fit(split.train)
            y_pred = model.predict_log(split.test)
            mse = float(np.mean((y_pred - y_true) ** 2))
            scores.append(mse)
            print(f"[{length_cap}/{idle}] {name} seed={seed}: log_mse={mse:.5f}", flush=True)
        results["models"][name] = {
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"seeds_len{length_cap}_idle{idle}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/pt_recovery/experiment_data")
    ap.add_argument("--length-cap", type=int, default=40)
    ap.add_argument("--idle", type=int, default=100)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    run(args.root, args.length_cap, args.idle, args.out,
        n_seeds=args.n_seeds, epochs=args.epochs)


if __name__ == "__main__":
    main()
