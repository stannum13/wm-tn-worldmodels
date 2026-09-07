#!/usr/bin/env python3
"""Probe compact action-conditioned memory architectures on RB residuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.real_benchmark import fit_rb_decay, load_rb_json, predict_rb_decay  # noqa: E402


def tensors(rows, max_length=None):
    lengths = torch.tensor([len(r["cl_ops"]) for r in rows], dtype=torch.long)
    width = max_length or int(lengths.max())
    actions = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        actions[i, : lengths[i]] = torch.tensor(row["cl_ops"], dtype=torch.long)
    targets = torch.tensor([r["p0"] for r in rows], dtype=torch.float32)
    return actions, lengths, targets


class OscillatorMemory(nn.Module):
    def __init__(self, modes: int):
        super().__init__()
        self.raw_radius = nn.Parameter(torch.full((modes,), 2.0))
        self.frequency = nn.Parameter(torch.linspace(0.05, 1.5, modes))
        self.drive = nn.Parameter(torch.randn(24, modes, 2) * 0.03)
        self.readout = nn.Linear(2 * modes, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, actions, lengths):
        batch, width = actions.shape
        modes = self.raw_radius.numel()
        state = torch.zeros((batch, modes, 2), device=actions.device)
        radius = 0.999 * torch.sigmoid(self.raw_radius)
        c, s = torch.cos(self.frequency), torch.sin(self.frequency)
        for t in range(width):
            x, y = state[..., 0], state[..., 1]
            rotated = torch.stack((radius * (c * x - s * y), radius * (s * x + c * y)), dim=-1)
            candidate = rotated + self.drive[actions[:, t]]
            state = torch.where((t < lengths)[:, None, None], candidate, state)
        return self.readout(state.flatten(1)).squeeze(-1)


class ControlledMPS(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        eye = torch.eye(rank).repeat(24, 1, 1)
        self.transition = nn.Parameter(eye + 0.02 * torch.randn_like(eye))
        self.drive = nn.Parameter(torch.randn(24, rank) * 0.02)
        self.initial = nn.Parameter(torch.zeros(rank))
        self.readout = nn.Linear(rank, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, actions, lengths):
        state = self.initial.expand(actions.shape[0], -1)
        for t in range(actions.shape[1]):
            matrix = self.transition[actions[:, t]]
            candidate = torch.tanh(torch.bmm(matrix, state.unsqueeze(-1)).squeeze(-1) + self.drive[actions[:, t]])
            state = torch.where((t < lengths)[:, None], candidate, state)
        return self.readout(state).squeeze(-1)


def fit_model(model, train, base_train, *, epochs: int, seed: int):
    torch.manual_seed(seed)
    actions, lengths, targets = train
    residual = targets - base_train
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(targets), generator=generator)
        for ids in order.split(256):
            optimizer.zero_grad()
            loss = torch.mean((model(actions[ids], lengths[ids]) - residual[ids]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate(model, dataset, base):
    actions, lengths, targets = dataset
    with torch.no_grad():
        predictions = torch.clamp(base + model(actions, lengths), 0.0, 1.0)
    errors = predictions - targets
    return {
        "rmse": float(torch.sqrt(torch.mean(errors**2))),
        "mae": float(torch.mean(torch.abs(errors))),
        "residual_correlation": float(np.corrcoef((targets - base).numpy(), (predictions - base).numpy())[0, 1]),
    }


def run(path: Path, *, ranks: list[int], seeds: list[int], epochs: int):
    torch.set_num_threads(1)
    rows = load_rb_json(path)
    train_rows = [r for r in rows if len(r["cl_ops"]) <= 20]
    test_rows = [r for r in rows if len(r["cl_ops"]) > 20]
    decay = fit_rb_decay(
        np.array([len(r["cl_ops"]) for r in train_rows]),
        np.array([r["p0"] for r in train_rows]),
    )
    train = tensors(train_rows)
    test = tensors(test_rows)
    base_train = torch.tensor(predict_rb_decay(decay, train[1].numpy()), dtype=torch.float32)
    base_test = torch.tensor(predict_rb_decay(decay, test[1].numpy()), dtype=torch.float32)
    output = {"data": str(path), "epochs": epochs, "baseline": evaluate(nn.Identity(), test, base_test) if False else {
        "rmse": float(torch.sqrt(torch.mean((base_test - test[2]) ** 2))),
        "mae": float(torch.mean(torch.abs(base_test - test[2]))),
    }, "runs": []}
    for family in ("oscillator", "controlled_mps"):
        for rank in ranks:
            for seed in seeds:
                torch.manual_seed(seed)
                model = OscillatorMemory(rank) if family == "oscillator" else ControlledMPS(rank)
                fit_model(model, train, base_train, epochs=epochs, seed=seed)
                metrics = evaluate(model, test, base_test)
                output["runs"].append({
                    "family": family,
                    "rank": rank,
                    "seed": seed,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    **metrics,
                })
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--ranks", default="2,4,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.data, ranks=[int(x) for x in args.ranks.split(",")], seeds=[int(x) for x in args.seeds.split(",")], epochs=args.epochs)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
