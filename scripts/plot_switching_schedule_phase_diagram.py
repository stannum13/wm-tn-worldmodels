#!/usr/bin/env python3
"""Plot the switching schedule recovery boundary from a completed artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.artifact.read_text())
    summary = data["summary"]["cells"]
    dwells = list(data["config"]["evaluation_dwells"])
    scales = list(data["config"]["mismatch_scales"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for scale in scales:
        values = [100 * summary[f"scale={scale:.2f},dwell={d}"]["adaptive_2"]
                  ["physical_oracle_gap_recovery"] for d in dwells]
        axes[0].plot(dwells, values, marker="o", label=f"{scale:.2f}x rates")
    axes[0].axhline(70, color="black", linestyle="--", linewidth=1, label="70% GO")
    axes[0].axvline(64, color="0.6", linestyle=":", linewidth=1)
    axes[0].set(xscale="log", xlabel="Schedule dwell (records)",
                ylabel="Physical-oracle gap recovery (%)", title="Mismatch boundary")
    axes[0].set_xticks(dwells, labels=[str(x) for x in dwells])
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=.2)

    for width, label in ((0, "16 B / zero change"), (1, "96 B / one change"),
                         (2, "256 B / two changes")):
        values = [100 * summary[f"scale=1.00,dwell={d}"][f"adaptive_{width}"]
                  ["physical_oracle_gap_recovery"] for d in dwells]
        axes[1].plot(dwells, values, marker="o", label=label)
    axes[1].axhline(70, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(64, color="0.6", linestyle=":", linewidth=1)
    axes[1].set(xscale="log", xlabel="Schedule dwell (records)",
                ylabel="Physical-oracle gap recovery (%)", title="State-budget boundary")
    axes[1].set_xticks(dwells, labels=[str(x) for x in dwells])
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=.2)

    fig.suptitle("Bounded graph-hypothesis controller under between-record switching")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
