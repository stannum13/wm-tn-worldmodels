#!/usr/bin/env python3
"""Standalone paper/blog figures from the archived, uncertainty-aware artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    main_result = json.loads((args.results / "surface_frontier_challenge_summary.json").read_text())
    rates = json.loads((args.results / "surface_rate_adaptation.json").read_text())
    distill = json.loads((args.results / "surface_teacher_distillation.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained",
                              gridspec_kw={"width_ratios": [1.3, 1., 1.1]})
    conditions = ["nominal", "scale_075", "scale_125", "less_separated", "slow", "fast", "iid", "within_shot"]
    labels = ["Nominal", "Noise × 0.75", "Noise × 1.25", "Less separated", "Slow changes", "Fast changes",
              "Independent modes", "Change within shot"]
    for distance, offset, color in (("3", -.12, "#0072B2"), ("5", .12, "#D55E00")):
        pairs = [main_result["distances"][distance]["conditions"][c]["compiled_vs_static"] for c in conditions]
        means = np.asarray([p["mean_difference"] for p in pairs]) * 100
        ci = np.asarray([p["replicate_97_5pct_interval"] for p in pairs]) * 100
        axes[0].errorbar(means, np.arange(len(conditions)) + offset,
                         xerr=np.vstack((means-ci[:, 0], ci[:, 1]-means)), fmt="o", ms=4,
                         capsize=2, color=color, label=f"Distance {distance}")
    axes[0].axvline(0, color="0.35", lw=.8)
    axes[0].set_yticks(range(len(conditions)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Compiled − selected static LER (percentage points)")
    axes[0].set_title("A  The gain has a clear failure boundary", loc="left", fontsize=11)
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(axis="x", alpha=.15)

    method_names = ["static", "memoryless", "fixed", "adaptive"]
    display_names = ["Static", "No history", "Fixed rate", "Rate bank"]
    for condition, color, marker in (("nominal", "#0072B2", "o"), ("iid", "#D55E00", "s")):
        mean = [np.mean([r["conditions"][condition]["methods"][name]["ler"] for r in rates["records"]]) * 100
                for name in method_names]
        axes[1].plot(range(4), mean, marker=marker, color=color, label="Persistent" if condition == "nominal" else "Independent")
    axes[1].set_xticks(range(4), display_names, rotation=25, ha="right")
    axes[1].set_ylabel("Distance-5 logical error (%)")
    axes[1].set_title("B  Infer when history is useful", loc="left", fontsize=11)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=.15)

    names = ["static", "outcome", "teacher", "restricted_teacher", "exact_teacher"]
    display = ["Static matching", "Outcome-trained head", "Teacher-trained head", "Restricted exact teacher", "Full exact teacher"]
    mean = [distill["summary"]["mean_ler"][name] * 100 for name in names]
    colors = ["#777777", "#0072B2", "#009E73", "#CC79A7", "#D55E00"]
    for row, (value, color) in enumerate(zip(mean, colors)):
        axes[2].plot(value, row, "o", color=color, ms=7)
        axes[2].annotate(f"{value:.3f}%", (value, row), xytext=(6, -3), textcoords="offset points", fontsize=9)
    axes[2].set_yticks(range(5), display)
    axes[2].invert_yaxis()
    axes[2].set_xlim(1.72, 2.12)
    axes[2].set_xlabel("Distance-3 logical error (%)")
    axes[2].set_title("C  The small grammar leaves most of the gap", loc="left", fontsize=11)
    axes[2].grid(axis="x", alpha=.15)
    fig.suptitle("Small adaptation laws around trusted QEC decoders: four completed campaigns", fontsize=14)
    fig.supxlabel("A: ten complete refits; 97.5% intervals (shift cells descriptive here).  B/C: fresh follow-ups with reused evidence models.\n"
                  "Exact teachers know the DEM and are offline, exponential-size diagnostics. All results are synthetic completed-shot experiments.",
                  fontsize=9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        target = args.output_dir / f"surface_frontier_campaigns.{extension}"
        fig.savefig(target, dpi=200)
        if extension == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
        print(target)


if __name__ == "__main__":
    main()
