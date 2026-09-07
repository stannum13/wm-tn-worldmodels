#!/usr/bin/env python3
"""Exploratory sequence-signal and calibration audit of released OQE forecasts.

No OQE is retrained. Calibration uses released validation predictions at lengths
21--40; evaluation reuses already explored lengths 41--60. This is not an
independent confirmation or an estimate of identifiable device parameters.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.real_benchmark import fit_rb_decay, predict_rb_decay  # noqa: E402


def git_json(repo, path):
    return json.loads(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=repo))


def arrays(payload, minimum_length, decay):
    lengths = sorted(int(k) for k in payload["labels"] if int(k) >= minimum_length)
    y = np.concatenate([payload["labels"][str(n)] for n in lengths])
    q = np.concatenate([payload["predicted"][str(n)] for n in lengths])
    n = np.concatenate([np.full(len(payload["labels"][str(m)]), m) for m in lengths])
    return y, q, n, predict_rb_decay(decay, n)


def rmse(predictions, targets):
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def sequence_audit(y, predictions, lengths, baseline):
    means, centered_y, centered_predictions = (np.empty_like(y) for _ in range(3))
    for length in np.unique(lengths):
        ids = lengths == length
        means[ids] = predictions[ids].mean()
        centered_y[ids] = y[ids] - y[ids].mean()
        centered_predictions[ids] = predictions[ids] - means[ids]
    mean_mse = float(np.mean((means - y) ** 2))
    sequence_variance = float(np.mean(centered_predictions**2))
    correlation = None
    if np.std(centered_y) > 1e-12 and np.std(centered_predictions) > 1e-12:
        correlation = float(np.corrcoef(centered_y, centered_predictions)[0, 1])
    return {
        "rb_rmse": rmse(baseline, y),
        "hybrid_rmse": rmse(predictions, y),
        "prediction_length_mean_only_rmse": float(np.sqrt(mean_mse)),
        # This is sqrt(E[MSE]), not E[RMSE], under within-length permutations.
        "sqrt_expected_within_length_permutation_mse": float(np.sqrt(mean_mse + sequence_variance)),
        "within_length_correlation": correlation,
        "mse_gain_from_mean_correction": float(np.mean((baseline - y) ** 2) - mean_mse),
        "mse_gain_from_sequence_correction": float(mean_mse - np.mean((predictions - y) ** 2)),
    }


def depolarized_readout(parameters, q, lengths):
    retention, report_zero_given_zero, report_zero_given_one = parameters
    # The OQE convention has one initial noise step and one after each gate.
    p = 0.5 + retention ** (lengths + 1) * (q - 0.5)
    return report_zero_given_zero * p + report_zero_given_one * (1 - p)


def calibration_audit(vy, vq, vn, y, q, n):
    readout_fit = least_squares(
        lambda v: v[0] * vq + v[1] * (1 - vq) - vy,
        [0.9, 0.1], bounds=(0, 1),
    )
    fits = [
        least_squares(
            lambda v: depolarized_readout(v, vq, vn) - vy,
            [retention, 0.9, 0.1], bounds=(0, 1),
        )
        for retention in (0.95, 0.99, 1.0)
    ]
    fit = min(fits, key=lambda result: np.mean(result.fun**2))
    readout_prediction = readout_fit.x[0] * q + readout_fit.x[1] * (1 - q)
    return {
        "readout_only_parameters": readout_fit.x.tolist(),
        "readout_only_validation_rmse": float(np.sqrt(np.mean(readout_fit.fun**2))),
        "readout_only_test_rmse": rmse(readout_prediction, y),
        "depolarization_readout_parameters": fit.x.tolist(),
        "depolarization_readout_validation_rmse": float(np.sqrt(np.mean(fit.fun**2))),
        "depolarization_readout_test_rmse": rmse(depolarized_readout(fit.x, q, n), y),
        "optimizer_success": bool(fit.success and readout_fit.success),
    }


def run(repo, biases):
    rows = []
    for bias in biases:
        base = f"experiment_data/RB_data_20230104/len40/idle100/rb_data_{bias}"
        train = git_json(repo, base + "/data.json")["train_data"]
        decay = fit_rb_decay(np.array([len(row[0]) for row in train]), np.array([row[1] for row in train]))
        for dimension in (1, 6):
            validation = git_json(repo, base + f"/unitaryfidelities_test_40_full_D{dimension}.json")
            forecast = git_json(repo, base + f"/unitaryfidelities_test_60_full_D{dimension}.json")
            vy, vq, vn, vb = arrays(validation, 21, decay)
            y, q, n, b = arrays(forecast, 41, decay)
            correction = vq - vb
            denominator = float(correction @ correction)
            raw_weight = float(np.clip(correction @ (vy - vb) / denominator, 0, 1)) if denominator else 0.0
            weight = raw_weight if raw_weight >= 0.5 else 0.0
            row = {
                "bias": bias, "dimension": dimension,
                "validation_weight": raw_weight, "gated_weight": weight,
                "sequence_audit": sequence_audit(y, b + weight * (q - b), n, b),
            }
            if dimension == 1:
                row["calibration_audit"] = calibration_audit(vy, vq, vn, y, q, n)
            rows.append(row)
    return {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "protocol": {
            "status": "exploratory_diagnostic_on_previously_examined_test_data",
            "condition": "idle100",
            "calibration_lengths": [21, 40], "forecast_lengths": [41, 60],
            "existing_hybrid_gate_threshold": 0.5,
            "calibration_parameters": ["per_step_retention", "report_zero_given_zero", "report_zero_given_one"],
            "limitations": [
                "Released OQE parameters are not retrained; only output calibration is fitted.",
                "Release prediction/target row alignment is assumed, not independently verified here.",
                "Calibration parameters need not equal physical device parameters.",
                "Per-length mean prediction is an evaluation decomposition, not a deployed predictor.",
                "Permutation quantity is the analytic square root of expected MSE, not expected RMSE.",
                "No independent test, memory witness, or active-mixing transfer claim is made.",
            ],
        },
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--biases", default="0.1,0.4,0.5,0.52,0.54,0.6")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.repo, args.biases.split(","))
    output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)
