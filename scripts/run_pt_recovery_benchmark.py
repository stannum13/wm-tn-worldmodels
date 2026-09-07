#!/usr/bin/env python3
"""Benchmark simple predictors on the public correlated-noise RB data.

The primary split trains on sequence lengths <= 20 and tests on lengths 21--40,
so no random time-point split can leak a trajectory. This is deliberately a baseline
study; it does not claim to reproduce the paper's OQE reconstruction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.real_benchmark import (  # noqa: E402
    fit_ridge,
    load_rb_json,
    make_features,
    predict,
    fit_rb_decay,
    predict_rb_decay,
    rmse,
    split_by_length,
)


def _fit_gru(train_rows, test_rows, *, epochs: int, seed: int = 0):
    """Tiny sequence model; kept here so the core package stays NumPy-only."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(24, 8)
            self.gru = nn.GRU(8, 16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, actions, lengths):
            outputs, _ = self.gru(self.embedding(actions))
            index = torch.arange(actions.shape[0])
            return self.head(outputs[index, lengths - 1]).squeeze(-1)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    max_train = max(len(r["cl_ops"]) for r in train_rows)
    sequences = torch.zeros((len(train_rows), max_train), dtype=torch.long)
    lengths = torch.tensor([len(r["cl_ops"]) for r in train_rows], dtype=torch.long)
    for i, row in enumerate(train_rows):
        sequences[i, : lengths[i]] = torch.tensor(row["cl_ops"], dtype=torch.long)
    targets = torch.tensor([r["p0"] for r in train_rows], dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(sequences, lengths) - targets) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        max_test = max(len(r["cl_ops"]) for r in test_rows)
        test_sequences = torch.zeros((len(test_rows), max_test), dtype=torch.long)
        test_lengths = torch.tensor([len(r["cl_ops"]) for r in test_rows], dtype=torch.long)
        for i, row in enumerate(test_rows):
            test_sequences[i, : test_lengths[i]] = torch.tensor(row["cl_ops"], dtype=torch.long)
        predictions = model(test_sequences, test_lengths).numpy()
    return predictions, sum(p.numel() for p in model.parameters())


def _rmse_interval(errors: np.ndarray, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = [float(np.sqrt(np.mean(rng.choice(errors, size=len(errors), replace=True) ** 2))) for _ in range(1000)]
    return {"mean": float(np.sqrt(np.mean(errors**2))), "p05": float(np.quantile(draws, 0.05)), "p95": float(np.quantile(draws, 0.95))}


def run(path: str | Path, *, max_train_length: int = 20, gru_epochs: int = 120, include_gru: bool = True) -> dict[str, object]:
    rows = load_rb_json(path)
    train, test = split_by_length(rows, max_train_length=max_train_length)
    x_train, y_train = make_features(train)
    x_test, y_test = make_features(test)

    # Feature blocks: [length | 24 one-hot counts | 576 ordered pairs].
    blocks = {
        "length_only": slice(0, 1),
        "unordered_actions": slice(0, 25),
        "ordered_pairs": slice(0, 601),
    }
    result = {
        "dataset": str(path),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_lengths": [min(len(r["cl_ops"]) for r in train), max(len(r["cl_ops"]) for r in train)],
        "test_lengths": [min(len(r["cl_ops"]) for r in test), max(len(r["cl_ops"]) for r in test)],
        "models": {},
    }
    train_lengths = np.asarray([len(r["cl_ops"]) for r in train])
    test_lengths = np.asarray([len(r["cl_ops"]) for r in test])
    rb_params = fit_rb_decay(train_lengths, y_train)
    rb_predictions = predict_rb_decay(rb_params, test_lengths)
    result["models"]["rb_exponential"] = {
        "parameter_count": 3,
        "parameters": {"amplitude": float(rb_params[0]), "alpha": float(rb_params[1]), "floor": float(rb_params[2])},
        "rmse": rmse(rb_predictions, y_test),
        "mae": float(np.mean(np.abs(rb_predictions - y_test))),
        "bootstrap_test_rmse": _rmse_interval(rb_predictions - y_test),
        "by_length": {
            str(length): rmse(rb_predictions[test_lengths == length], y_test[test_lengths == length])
            for length in sorted(set(test_lengths))
        },
    }
    for name, block in blocks.items():
        weights = fit_ridge(x_train[:, block], y_train, alpha=1.0)
        predictions = predict(weights, x_test[:, block])
        model = {
            "parameter_count": int(weights.size),
            "rmse": rmse(predictions, y_test),
            "mae": float(np.mean(np.abs(predictions - y_test))),
            "bootstrap_test_rmse": _rmse_interval(predictions - y_test),
            "by_length": {},
        }
        for length in sorted({len(r["cl_ops"]) for r in test}):
            mask = np.asarray([len(r["cl_ops"]) == length for r in test])
            model["by_length"][str(length)] = rmse(predictions[mask], y_test[mask])
        result["models"][name] = model
    if include_gru:
        gru_predictions, parameter_count = _fit_gru(train, test, epochs=gru_epochs)
        result["models"]["gru"] = {
        "parameter_count": parameter_count,
        "epochs": gru_epochs,
        "rmse": rmse(gru_predictions, y_test),
        "mae": float(np.mean(np.abs(gru_predictions - y_test))),
        "bootstrap_test_rmse": _rmse_interval(gru_predictions - y_test),
        "by_length": {
            str(length): rmse(gru_predictions[np.asarray([len(r["cl_ops"]) == length for r in test])], y_test[np.asarray([len(r["cl_ops"]) == length for r in test])])
            for length in sorted({len(r["cl_ops"]) for r in test})
        },
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="standard_rb_1q_full_data.json")
    parser.add_argument("--max-train-length", type=int, default=20)
    parser.add_argument("--gru-epochs", type=int, default=120)
    parser.add_argument("--skip-gru", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(args.data, max_train_length=args.max_train_length, gru_epochs=args.gru_epochs, include_gru=not args.skip_gru), indent=2, sort_keys=True)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
