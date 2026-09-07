"""Directional residual analysis for Experiment A.

Decomposes each model's test residuals along the experiment's control directions:
- forecast horizon (sequence length),
- bias setting (gate-voltage control),
- idle duration (between-gate idle),
- gate family (which Clifford block dominates the sequence),
- context cell (bias x length) — how much residual structure is systematic
  (explainable by control context) vs. episode-level noise.

Outputs results/residuals_len{cap}_idle{idle}.json and a summary figure.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import Episode, load_cell
from .models import GRUModel, MarkovChannel, ProcessMPO, TransferTensor, TransformerModel
from .splits import all_splits


def gate_family_max(ep: Episode) -> int:
    return max((g // 8 for g in ep.gates), default=0)


def directional_summary(eps: list[Episode], resid: np.ndarray) -> dict:
    lens = np.array([ep.length for ep in eps])
    biases = np.array([ep.bias for ep in eps])
    fams = np.array([gate_family_max(ep) for ep in eps])

    def block(keys: np.ndarray) -> dict:
        out = {}
        for k in sorted(set(keys)):
            m = keys == k
            out[str(k)] = {
                "mean": float(np.mean(resid[m])),
                "rmse": float(np.sqrt(np.mean(resid[m] ** 2))),
                "n": int(m.sum()),
            }
        return out

    # Context-cell decomposition: variance of residuals explained by (bias, length).
    cell = defaultdict(list)
    for r, b, L in zip(resid, biases, lens):
        cell[(b, L)].append(r)
    cell_means = np.array([np.mean(v) for v in cell.values()])
    cell_counts = np.array([len(v) for v in cell.values()])
    var_resid = float(np.var(resid))
    explained = float(np.average(cell_means**2, weights=cell_counts)) if var_resid > 0 else 0.0

    return {
        "overall": {"mean": float(np.mean(resid)), "rmse": float(np.sqrt(np.mean(resid**2))), "var": var_resid},
        "by_length": block(lens),
        "by_bias": block(biases),
        "by_max_gate_family": block(fams),
        "context_cell_explained_frac": explained / var_resid if var_resid > 0 else 0.0,
    }


def run(root: str, length_cap: int, idle: int, out_dir: str):
    ds = load_cell(root, length_cap, idle)
    splits = all_splits(ds)
    models = [
        MarkovChannel(),
        TransferTensor(),
        ProcessMPO(chi=2, epochs=60),
        GRUModel(hidden=2, epochs=60),
    ]
    results = {"cell": {"length_cap": length_cap, "idle": idle}, "splits": {}}
    for split in splits:
        y_true = np.array([np.log(max(ep.fidelity, 1e-12)) for ep in split.test])
        entry = {}
        for model in models:
            model.fit(split.train)
            resid = model.predict_log(split.test) - y_true
            entry[model.name] = directional_summary(split.test, resid)
            print(f"[{split.name}] {model.name}: rmse={entry[model.name]['overall']['rmse']:.4f} "
                  f"ctx_explained={entry[model.name]['context_cell_explained_frac']:.3f}", flush=True)
        results["splits"][split.name] = entry

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"residuals_len{length_cap}_idle{idle}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path}", flush=True)

    # Figure: per-length residual curves per model on the horizon split.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = results["splits"].get("horizon_le20", {})
    if hs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, entry in hs.items():
            lens = sorted(int(k) for k in entry["by_length"])
            means = [entry["by_length"][str(L)]["mean"] for L in lens]
            ax.plot(lens, means, label=name, marker="o", markersize=3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("sequence length (forecast horizon)")
        ax.set_ylabel("mean residual (log-fidelity, pred - true)")
        ax.set_title(f"Directional residuals by horizon — len{length_cap}/idle{idle}")
        ax.legend()
        fig.tight_layout()
        figpath = out / f"residuals_by_length_len{length_cap}_idle{idle}.png"
        fig.savefig(figpath, dpi=150)
        print(f"wrote {figpath}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/pt_recovery/experiment_data")
    ap.add_argument("--length-cap", type=int, default=40)
    ap.add_argument("--idle", type=int, default=100)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    run(args.root, args.length_cap, args.idle, args.out)


if __name__ == "__main__":
    main()
