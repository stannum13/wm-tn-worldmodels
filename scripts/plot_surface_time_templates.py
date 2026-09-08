#!/usr/bin/env python3
"""Plot frozen graph-action comparisons and actual versus nominal cut references."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    summary = json.loads(Path("results/surface_time_templates/summary.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted(Path("results/surface_time_templates").glob("r*.json"))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    conditions = ["four_way", "endpoints", "midpoint", "iid", "third", "two_thirds"]
    labels = ["Four modes", "AA/BB only", "Halfway switch", "Independent modes", "Unseen 1/3 switch", "Unseen 2/3 switch"]
    for offset, key, label, color in [(-.12, "four_vs_static", "Selected static", "#0072B2"),
                                      (.12, "four_vs_matched_two", "Matched AA/BB actions", "#D55E00")]:
        values = [summary["conditions"][c][key] for c in conditions]
        means = np.array([v["delta"] * 100 for v in values])
        ci = np.array([v["ci_primary"] for v in values]) * 100
        axes[0].errorbar(means, np.arange(len(values)) + offset,
                        xerr=[means - ci[:, 0], ci[:, 1] - means], fmt="o", color=color,
                        label=label, ms=5, capsize=3)
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="grey", lw=1)
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    axes[0].set_xlabel("Four-action causal − reference LER (percentage points)")
    axes[0].set_title("A  More graph actions help; timing transfer still fails", loc="left", fontsize=11)
    cuts = ["midpoint", "third", "two_thirds"]
    for method, label, color in [("four", "Causal selector", "#0072B2"),
                                 ("nominal_mode_template", "Known direction, halfway graph", "#CC79A7"),
                                 ("actual_mode_template", "Known direction + actual-cut graph", "#009E73")]:
        values = [np.mean([r["conditions"][c]["methods"][method]["ler"] for r in rows]) * 100 for c in cuts]
        axes[1].plot(range(3), values, "o-", color=color, label=label)
    axes[1].set_xticks(range(3), ["Trained 1/2", "Unseen 1/3", "Unseen 2/3"])
    axes[1].set_ylabel("Logical error (%)")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].set_ylim(1.25, 1.64)
    axes[1].set_title("B  Correct timing matters beyond correct direction", loc="left", fontsize=11)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[0] else "y", alpha=.15)
    fig.suptitle("Time-resolved graph adaptation: 10 fresh d5 refits, 10,485,760 held-out shots", fontsize=13)
    fig.supxlabel("A: 98⅓% paired complete-refit intervals (transfer decisions use 98.75%).  B: privileged references, not Bayes oracles.\n"
                  "Synthetic completed-record decisions. Controller state: 160 bytes; graph and service costs are separate.", fontsize=9)
    target = Path("results/figures")
    target.mkdir(exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = target / f"surface_time_templates.{extension}"
        fig.savefig(path, dpi=200)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)


if __name__ == "__main__":
    main()
