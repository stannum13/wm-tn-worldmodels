"""Experiment A runner: fit all baselines on all splits of one data cell,
evaluate, and write a metrics table.

Usage:
    python -m ptwm.run_experiments --root data/external/pt_recovery/experiment_data \
        --length-cap 40 --idle 100 --out results/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import EpisodeDataset, load_cell
from .metrics import horizon_of_tolerance, per_length_errors, summarize_all
from .models import GRUModel, MarkovChannel, ProcessMPO, TransferTensor, TransformerModel
from .splits import all_splits, summarize


def build_models(match_params: bool = True):
    """Model ladder at roughly matched parameter counts (~200 params).

    Markov: 27 params. TransferTensor: 6. ProcessMPO chi=2: 24*3+2*2+2+1 = 79.
    GRU hidden=2: 3*(2*26+2)+3*2+2+3 = 173. Transformer d=16: ~4k (small variant).
    Epoch budgets sized for a first full pass on CPU (~minutes per model).
    """
    models = [
        MarkovChannel(),
        TransferTensor(),
        ProcessMPO(chi=2, epochs=60),
        ProcessMPO(chi=4, epochs=60),
        GRUModel(hidden=2, epochs=60),
        GRUModel(hidden=8, epochs=60),
        TransformerModel(d_model=16, nhead=2, layers=1, epochs=40),
    ]
    return models


def run_cell(root: str, length_cap: int, idle: int, out_dir: str,
             quick: bool = False) -> dict:
    ds = load_cell(root, length_cap, idle)
    splits = all_splits(ds)
    models = build_models()
    if quick:
        for m in models:
            if hasattr(m, "epochs"):
                m.epochs = 5

    results = {"cell": {"length_cap": length_cap, "idle": idle}, "splits": []}
    for split in splits:
        t0 = time.time()
        split_entry = {"name": split.name, "description": split.description,
                       "sizes": summarize(split), "models": []}
        y_test_log = np.array([np.log(max(ep.fidelity, 1e-12)) for ep in split.test])
        print(f"[{length_cap}/{idle}] split {split.name}: "
              f"train={len(split.train)} test={len(split.test)}", flush=True)
        for model in models:
            t1 = time.time()
            model.fit(split.train)
            fit_s = time.time() - t1
            y_pred_log = model.predict_log(split.test)
            metrics = summarize_all(y_test_log, y_pred_log)
            per_len = per_length_errors(split.test, y_pred_log)
            metrics["horizon_at_0.05"] = horizon_of_tolerance(per_len, 0.05)
            metrics["horizon_at_0.02"] = horizon_of_tolerance(per_len, 0.02)
            metrics["check"] = model.check()
            metrics["budget"] = model.budget()
            metrics["fit_seconds"] = fit_s
            split_entry["models"].append(metrics)
            print(f"  {model.name}: log_mse={metrics['log_mse']:.5f} "
                  f"mae={metrics['fidelity_mae']:.5f} fit={fit_s:.1f}s", flush=True)
        split_entry["wall_seconds"] = time.time() - t0
        results["splits"].append(split_entry)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"exp_a_len{length_cap}_idle{idle}.json"
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
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run_cell(args.root, args.length_cap, args.idle, args.out, quick=args.quick)


if __name__ == "__main__":
    main()
