"""Aggregate Experiment A results across data cells into one table.

Reads results/exp_a_len{cap}_idle{idle}.json for all available cells and writes
results/exp_a_summary.csv plus a markdown table to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)

    rows = []
    for path in sorted(root.glob("exp_a_len*_idle*.json")):
        r = json.load(open(path))
        cap, idle = r["cell"]["length_cap"], r["cell"]["idle"]
        for split in r["splits"]:
            for m in split["models"]:
                rows.append({
                    "cell": f"len{cap}/idle{idle}",
                    "split": split["name"],
                    "model": m["budget"]["name"],
                    "params": m["budget"]["params"],
                    "log_mse": round(m["log_mse"], 5),
                    "fid_mae": round(m["fidelity_mae"], 5),
                    "horizon_0.05": m.get("horizon_at_0.05"),
                    "horizon_0.02": m.get("horizon_at_0.02"),
                    "fit_s": round(m["fit_seconds"], 1),
                })

    out = root / "exp_a_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    # Markdown: best model per (cell, split).
    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["cell"], row["split"])
        if key not in best or row["log_mse"] < best[key]["log_mse"]:
            best[key] = row
    print("\nBest model per cell/split (by log-MSE):")
    print("| cell | split | model | params | log_mse | fid_mae |")
    print("|---|---|---|---|---|---|")
    for (cell, split), row in sorted(best.items()):
        print(f"| {cell} | {split} | {row['model']} | {row['params']} | {row['log_mse']} | {row['fid_mae']} |")


if __name__ == "__main__":
    main()
