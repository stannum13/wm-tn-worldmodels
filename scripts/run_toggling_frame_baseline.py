#!/usr/bin/env python3
"""Convex Clifford toggling-frame residual baseline for correlated RB noise."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.clifford import infer_clifford_rotations, toggling_frame_features  # noqa: E402
from ptwm.real_benchmark import fit_rb_decay, predict_rb_decay  # noqa: E402

BIAS_NAMES = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.52", "0.54", "0.56", "0.58", "0.6", "0.61", "0.62", "0.63", "0.64"]


def git_json(repo, path):
    return json.loads(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=repo))


def recover_rotations(repo):
    experiments = []
    for bias in BIAS_NAMES:
        split = git_json(repo, f"experiment_data/RB_data_20230104/len40/idle100/rb_data_{bias}/data.json")
        experiments.append([{"cl_ops": row[0], "p0": row[1]} for row in split["train_data"] + split["test_data"]])
    return infer_clifford_rotations(experiments)


def unpack(rows):
    sequences = [row[0] if isinstance(row, list) else row["cl_ops"] for row in rows]
    targets = np.asarray([row[1] if isinstance(row, list) else row["p0"] for row in rows])
    lengths = np.asarray([len(sequence) for sequence in sequences])
    return sequences, lengths, targets


def score(prediction, target):
    error = prediction - target
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))}


def run(repo, condition, biases, max_lags, alphas):
    rotations = recover_rotations(repo)
    output = []
    for bias in biases:
        base = f"experiment_data/RB_data_20230104/len40/{condition}/rb_data_{bias}"
        split = git_json(repo, base + "/data.json")
        raw60 = git_json(repo, f"experiment_data/RB_data_20230104/len60/{condition}/rb_data_{bias}/standard_rb_1q_full_data.json")
        train_s, train_l, train_y = unpack(split["train_data"])
        val_s, val_l, val_y = unpack(split["test_data"])
        test_s, test_l, test_y = unpack([row for row in raw60 if len(row["cl_ops"]) > 40])
        decay = fit_rb_decay(train_l, train_y)
        rb_train, rb_val, rb_test = (predict_rb_decay(decay, lengths) for lengths in (train_l, val_l, test_l))

        candidates = []
        cached = {}
        for max_lag in max_lags:
            cached[max_lag] = tuple(
                np.asarray([toggling_frame_features(sequence, rotations, max_lag) for sequence in sequences])
                for sequences in (train_s, val_s, test_s)
            )
            train_x, val_x, _ = cached[max_lag]
            for alpha in alphas:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-6))
                model.fit(train_x, train_y - rb_train)
                prediction = np.clip(rb_val + model.predict(val_x), 0, 1)
                candidates.append({"max_lag": max_lag, "alpha": alpha, "validation": score(prediction, val_y)})
        selected = min(candidates, key=lambda candidate: candidate["validation"]["rmse"])
        train_x, val_x, test_x = cached[selected["max_lag"]]
        model = make_pipeline(StandardScaler(), Ridge(alpha=selected["alpha"], solver="lsqr", tol=1e-6))
        model.fit(train_x, train_y - rb_train)
        predictions = {
            "train": np.clip(rb_train + model.predict(train_x), 0, 1),
            "validation": np.clip(rb_val + model.predict(val_x), 0, 1),
            "test": np.clip(rb_test + model.predict(test_x), 0, 1),
        }
        output.append({
            "bias": bias,
            "selected": selected,
            "parameter_count": int(train_x.shape[1] + 1),
            "rb": {name: score(prediction, target) for name, prediction, target in (
                ("train", rb_train, train_y), ("validation", rb_val, val_y), ("test", rb_test, test_y)
            )},
            "toggling_residual": {name: score(predictions[name], target) for name, target in (
                ("train", train_y), ("validation", val_y), ("test", test_y)
            )},
            "all_validation_candidates": candidates,
        })
    return {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "protocol": {
            "condition": condition,
            "training": "released train_data at lengths 2-40",
            "model_selection": "released test_data at lengths 2-40",
            "forecast": "independent len60 acquisition at lengths 41-60",
            "target": "residual from training-fitted RB decay",
            "features": "quadratic cumulative toggling frames plus finite-lag frame correlations",
            "max_lags": max_lags,
            "ridge_alphas": alphas,
            "prediction_constraint": "clip to [0,1]",
            "status": "exploratory; this model was designed after both idle conditions had been examined",
        },
        "biases": output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--condition", choices=("idle100", "idle180"), default="idle100")
    parser.add_argument("--biases", default="0.5")
    parser.add_argument("--max-lags", default="0,1,2,4,8")
    parser.add_argument("--alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.repo, args.condition, args.biases.split(","),
        [int(value) for value in args.max_lags.split(",")],
        [float(value) for value in args.alphas.split(",")],
    )
    result = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)
