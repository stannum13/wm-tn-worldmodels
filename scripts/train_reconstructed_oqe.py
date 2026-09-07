#!/usr/bin/env python3
"""Train a compact OQE using Clifford metadata reconstructed from RB inverses."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.clifford import infer_clifford_rotations  # noqa: E402
from ptwm.oqe import OpenQuantumEvolution, rotations_to_su2  # noqa: E402
from ptwm.real_benchmark import fit_rb_decay, predict_rb_decay  # noqa: E402


def git_json(repo, path):
    return json.loads(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=repo))


def make_tensors(rows):
    sequences = [row[0] if isinstance(row, list) else row["cl_ops"] for row in rows]
    values = [row[1] if isinstance(row, list) else row["p0"] for row in rows]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    actions = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        actions[i, : len(sequence)] = torch.tensor(sequence)
    return actions, lengths, torch.tensor(values, dtype=torch.float32)


def score(model, dataset):
    actions, lengths, targets = dataset
    with torch.no_grad():
        predictions = model(actions, lengths)
    return {
        "rmse": float(torch.sqrt(torch.mean((predictions - targets) ** 2))),
        "mae": float(torch.mean(torch.abs(predictions - targets))),
        "predictions": predictions.numpy(),
    }


def train_once(gates, train, validation, test, base_validation, base_test, memory_dimension, seed, epochs, gate_threshold=0.5, initial_scale=0.05):
    model = OpenQuantumEvolution(gates, memory_dimension, seed=seed, initial_scale=initial_scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_rmse, stale = None, float("inf"), 0
    for epoch in range(epochs):
        order = torch.randperm(len(train[2]), generator=generator)
        for ids in order.split(256):
            optimizer.zero_grad()
            predictions = model(train[0][ids], train[1][ids])
            loss = torch.mean((predictions - train[2][ids]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch % 5 == 4:
            validation_rmse = score(model, validation)["rmse"]
            if validation_rmse < best_rmse - 1e-5:
                best_rmse = validation_rmse
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    validation_score = score(model, validation)
    test_score = score(model, test)
    correction = validation_score["predictions"] - base_validation
    denominator = float(correction @ correction)
    raw_weight = float(np.clip(correction @ (validation[2].numpy() - base_validation) / denominator, 0.0, 1.0)) if denominator else 0.0
    weight = raw_weight if raw_weight >= gate_threshold else 0.0
    hybrid = base_test + weight * (test_score["predictions"] - base_test)
    return {
        "seed": seed,
        "epochs_completed": epoch + 1,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "validation_rmse": validation_score["rmse"],
        "validation_mae": validation_score["mae"],
        "test_rmse": test_score["rmse"],
        "test_mae": test_score["mae"],
        "validation_hybrid_weight": raw_weight,
        "gated_hybrid_weight": weight,
        "hybrid_test_rmse": float(np.sqrt(np.mean((hybrid - test[2].numpy()) ** 2))),
        "hybrid_test_mae": float(np.mean(np.abs(hybrid - test[2].numpy()))),
    }


def run(repo, bias, memory_dimensions, seeds, epochs, condition="idle100", initial_scale=0.05):
    bias_names = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.52", "0.54", "0.56", "0.58", "0.6", "0.61", "0.62", "0.63", "0.64"]
    experiments = []
    for name in bias_names:
        split = git_json(repo, f"experiment_data/RB_data_20230104/len40/idle100/rb_data_{name}/data.json")
        experiments.append([{"cl_ops": row[0], "p0": row[1]} for row in split["train_data"] + split["test_data"]])
    rotations = infer_clifford_rotations(experiments)
    gates = rotations_to_su2(rotations)

    base = f"experiment_data/RB_data_20230104/len40/{condition}/rb_data_{bias}"
    split = git_json(repo, base + "/data.json")
    raw60 = git_json(repo, f"experiment_data/RB_data_20230104/len60/{condition}/rb_data_{bias}/standard_rb_1q_full_data.json")
    test_rows = [row for row in raw60 if len(row["cl_ops"]) > 40]
    train, validation, test = make_tensors(split["train_data"]), make_tensors(split["test_data"]), make_tensors(test_rows)

    train_lengths = train[1].numpy()
    train_targets = train[2].numpy()
    decay = fit_rb_decay(train_lengths, train_targets)
    rb_validation = predict_rb_decay(decay, validation[1].numpy())
    rb_test = predict_rb_decay(decay, test[1].numpy())
    output = {
        "bias": float(bias),
        "condition": condition,
        "initial_scale": initial_scale,
        "train_rows": len(train[2]),
        "validation_rows": len(validation[2]),
        "test_rows": len(test[2]),
        "rb_test_rmse": float(np.sqrt(np.mean((rb_test - test[2].numpy()) ** 2))),
        "runs": [],
    }
    for dimension in memory_dimensions:
        for seed in seeds:
            result = train_once(gates, train, validation, test, rb_validation, rb_test, dimension, seed, epochs, initial_scale=initial_scale)
            output["runs"].append({"memory_dimension": dimension, **result})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--bias", default="0.5")
    parser.add_argument("--memory-dimensions", default="1,2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--condition", choices=("idle100", "idle180"), default="idle100")
    parser.add_argument("--initial-scale", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run(args.repo, args.bias, [int(x) for x in args.memory_dimensions.split(",")], [int(x) for x in args.seeds.split(",")], args.epochs, args.condition, args.initial_scale)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
