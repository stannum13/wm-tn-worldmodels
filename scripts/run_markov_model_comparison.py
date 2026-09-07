#!/usr/bin/env python3
"""Compare dissipative coherent and general memoryless quantum models on RB data."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.clifford import infer_clifford_rotations  # noqa: E402
from ptwm.oqe import DampedCoherentQubit, DissipativeMemoryQubit, MarkovianQubitChannel, QuasiStaticGaussianQubit, rotations_to_su2  # noqa: E402
from ptwm.real_benchmark import fit_rb_decay, predict_rb_decay  # noqa: E402


BIAS_NAMES = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.52", "0.54", "0.56", "0.58", "0.6", "0.61", "0.62", "0.63", "0.64"]


def git_json(repo, path):
    return json.loads(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=repo))


def make_tensors(rows):
    sequences = [row[0] if isinstance(row, list) else row["cl_ops"] for row in rows]
    targets = [row[1] if isinstance(row, list) else row["p0"] for row in rows]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    actions = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        actions[index, : len(sequence)] = torch.tensor(sequence)
    return actions, lengths, torch.tensor(targets, dtype=torch.float32)


def predictions(model, data, batch_size=512):
    values = []
    with torch.no_grad():
        for ids in torch.arange(len(data[2])).split(batch_size):
            values.append(model(data[0][ids], data[1][ids]))
    return torch.cat(values)


def metrics(values, targets):
    error = values - targets
    return {
        "rmse": float(torch.sqrt(torch.mean(error**2))),
        "mae": float(torch.mean(torch.abs(error))),
    }


def centered_error(prediction, target, lengths):
    """MSE after centering predictions and targets within each represented length."""
    pieces = []
    for length in torch.unique(lengths):
        ids = lengths == length
        if torch.count_nonzero(ids) > 1:
            pieces.append((prediction[ids] - prediction[ids].mean()) - (target[ids] - target[ids].mean()))
    return torch.mean(torch.cat(pieces) ** 2) if pieces else prediction.new_tensor(0.0)


def train_once(model, train, validation, test, *, seed, epochs, learning_rate, centered_weight=0.0, batch_size=256):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_validation, stale = None, float("inf"), 0
    started = time.perf_counter()
    for epoch in range(epochs):
        order = torch.randperm(len(train[2]), generator=generator)
        model.train()
        for ids in order.split(batch_size):
            optimizer.zero_grad()
            prediction = model(train[0][ids], train[1][ids])
            loss = torch.mean((prediction - train[2][ids]) ** 2)
            if centered_weight:
                loss = loss + centered_weight * centered_error(prediction, train[2][ids], train[1][ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch % 5 == 4:
            model.eval()
            validation_rmse = metrics(predictions(model, validation), validation[2])["rmse"]
            if validation_rmse < best_validation - 1e-5:
                best_validation = validation_rmse
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    result = {
        "seed": seed,
        "epochs_completed": epoch + 1,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "wall_seconds": time.perf_counter() - started,
        "train": metrics(predictions(model, train), train[2]),
        "validation": metrics(predictions(model, validation), validation[2]),
        "test": metrics(predictions(model, test), test[2]),
    }
    if isinstance(model, (DampedCoherentQubit, DissipativeMemoryQubit, QuasiStaticGaussianQubit)):
        if isinstance(model, DissipativeMemoryQubit):
            result["physical_parameters"] = {"retention": float(torch.sigmoid(model.retention_logit).detach())}
            result["memory_mode"] = model.memory_mode
        else:
            physical = model.physical_parameters()
            result["physical_parameters"] = {name: float(value.detach()) for name, value in physical.items()}
    else:
        with torch.no_grad():
            kraus = model.kraus()
            vectorized = kraus.reshape(len(kraus), -1)
            choi = torch.einsum("ki,kj->ij", vectorized, vectorized.conj())
            result["channel_choi_eigenvalues"] = torch.linalg.eigvalsh(choi).real.numpy().tolist()
            result["channel_trace_preservation_residual"] = float(
                torch.linalg.norm(torch.einsum("kji,kjl->il", kraus.conj(), kraus) - torch.eye(2))
            )
    return result


def recovered_gates(repo):
    experiments = []
    for bias in BIAS_NAMES:
        path = f"experiment_data/RB_data_20230104/len40/idle100/rb_data_{bias}/data.json"
        split = git_json(repo, path)
        experiments.append([{"cl_ops": row[0], "p0": row[1]} for row in split["train_data"] + split["test_data"]])
    return rotations_to_su2(infer_clifford_rotations(experiments))


def run(repo, condition, biases, models, seeds, epochs, centered_weight=0.0):
    gates = recovered_gates(repo)
    rows = []
    for bias in biases:
        base = f"experiment_data/RB_data_20230104/len40/{condition}/rb_data_{bias}"
        split = git_json(repo, base + "/data.json")
        raw60 = git_json(repo, f"experiment_data/RB_data_20230104/len60/{condition}/rb_data_{bias}/standard_rb_1q_full_data.json")
        test_rows = [row for row in raw60 if len(row["cl_ops"]) > 40]
        train = make_tensors(split["train_data"])
        validation = make_tensors(split["test_data"])
        test = make_tensors(test_rows)
        decay = fit_rb_decay(train[1].numpy(), train[2].numpy())
        rb = {
            name: metrics(torch.tensor(predict_rb_decay(decay, data[1].numpy()), dtype=torch.float32), data[2])
            for name, data in (("train", train), ("validation", validation), ("test", test))
        }
        row = {"bias": bias, "rb": rb, "runs": []}
        for name in models:
            for seed in seeds:
                if name == "damped_d1":
                    model, learning_rate = DampedCoherentQubit(gates, seed), 0.02
                elif name == "markov_cptp":
                    model, learning_rate = MarkovianQubitChannel(gates, seed), 0.005
                elif name.startswith("memory_"):
                    model, learning_rate = DissipativeMemoryQubit(gates, name.removeprefix("memory_"), seed), 0.01
                elif name == "quasistatic_gaussian":
                    model, learning_rate = QuasiStaticGaussianQubit(gates, seed=seed), 0.01
                else:
                    raise ValueError(f"unknown model {name}")
                row["runs"].append({
                    "model": name,
                    **train_once(model, train, validation, test, seed=seed, epochs=epochs, learning_rate=learning_rate, centered_weight=centered_weight),
                })
        rows.append(row)
    return {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "protocol": {
            "condition": condition,
            "training": "released train_data: random 60% at every length 2-40",
            "early_stopping": "released test_data: disjoint random 40% at every length 2-40",
            "forecast": "independent len60 acquisition, lengths 41-60",
            "loss": "mean squared error on analog calibrated p0 plus the stated within-length centered term",
            "models": models,
            "seeds": seeds,
            "epochs": epochs,
            "centered_within_length_loss_weight": centered_weight,
            "model_selection_warning": "idle100 is development; idle180 is confirmation only after freezing this script and settings",
        },
        "biases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--condition", choices=("idle100", "idle180"), default="idle100")
    parser.add_argument("--biases", default="0.4,0.5,0.52,0.54")
    parser.add_argument("--models", default="damped_d1,markov_cptp")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--centered-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run(
        args.repo, args.condition, args.biases.split(","), args.models.split(","),
        [int(value) for value in args.seeds.split(",")], args.epochs, args.centered_weight,
    )
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)
