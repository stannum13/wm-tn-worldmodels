"""Shared types for Experiment A.

The dataset axis convention follows the pt_recovery repo:
- bias (gamma) in {0.1..0.64}: 14 settings, actual Vbias = 0.4 * gamma
- idle in {100, 180} ns
- sequence length cap in {40, 60}

An Episode is one randomized-benchmarking sequence: a list of Clifford gate
indices plus the measured sequence fidelity p0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BIASES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.61, 0.62, 0.63, 0.64)
IDLES = (100, 180)
LENGTH_CAPS = (40, 60)
N_GATES = 24  # single-qubit Clifford set size; gate indices are 0..23


@dataclass(frozen=True)
class Episode:
    """One RB sequence: gates (ints, 0..23) and measured sequence fidelity."""

    gates: tuple[int, ...]
    fidelity: float
    bias: float
    idle: int
    length_cap: int

    @property
    def length(self) -> int:
        return len(self.gates)


@dataclass
class EpisodeDataset:
    """All episodes for one (length_cap, idle) cell, grouped by bias."""

    length_cap: int
    idle: int
    episodes: list[Episode] = field(default_factory=list)

    def by_bias(self) -> dict[float, list[Episode]]:
        out: dict[float, list[Episode]] = {}
        for ep in self.episodes:
            out.setdefault(ep.bias, []).append(ep)
        return out


def load_full_json(path: str, bias: float, idle: int, length_cap: int) -> list[Episode]:
    """Load a standard_rb_1q_full_data.json file into Episodes.

    Each entry is {"cl_ops": [gate indices], "p0": measured fidelity}.
    """
    import json

    with open(path) as f:
        data = json.load(f)
    episodes = []
    for entry in data:
        gates = tuple(int(g) for g in entry["cl_ops"])
        episodes.append(
            Episode(
                gates=gates,
                fidelity=float(entry["p0"]),
                bias=bias,
                idle=idle,
                length_cap=length_cap,
            )
        )
    return episodes


def load_cell(root: str, length_cap: int, idle: int) -> EpisodeDataset:
    """Load all 14 bias folders for one (length_cap, idle) cell."""
    from pathlib import Path

    ds = EpisodeDataset(length_cap=length_cap, idle=idle)
    base = Path(root) / f"RB_data_20230104" / f"len{length_cap}" / f"idle{idle}"
    if not base.exists():
        raise FileNotFoundError(f"missing data cell: {base}")
    for gamma in BIASES:
        folder = base / f"rb_data_{gamma}"
        path = folder / "standard_rb_1q_full_data.json"
        if not path.exists():
            raise FileNotFoundError(f"missing full data json: {path}")
        ds.episodes.extend(load_full_json(str(path), gamma, idle, length_cap))
    return ds


def load_paper_results(path: str) -> dict:
    """Load a paper result json (RB_data_len40_idle*_unitary_results.json)."""
    import json

    with open(path) as f:
        return json.load(f)


def episode_matrix(eps: list[Episode], n_gates: int = N_GATES) -> tuple[np.ndarray, np.ndarray]:
    """Stack episodes into (X, y) arrays.

    X: one-hot gate counts per position, shape (n, L, n_gates) — ragged lengths are
    NOT padded here; callers should group by length or use the flattened
    count representation below.
    y: log-fidelity targets, shape (n,).
    """
    lengths = {ep.length for ep in eps}
    if len(lengths) != 1:
        raise ValueError(f"episode_matrix expects a single length, got {sorted(lengths)}")
    L = lengths.pop()
    X = np.zeros((len(eps), L, n_gates), dtype=np.float32)
    y = np.empty(len(eps), dtype=np.float64)
    for i, ep in enumerate(eps):
        for t, g in enumerate(ep.gates):
            X[i, t, g] = 1.0
        y[i] = np.log(max(ep.fidelity, 1e-12))
    return X, y


def gate_counts(eps: list[Episode], n_gates: int = N_GATES) -> tuple[np.ndarray, np.ndarray]:
    """Order-invariant gate count features, shape (n, n_gates), plus log-fidelity y."""
    X = np.zeros((len(eps), n_gates), dtype=np.float32)
    y = np.empty(len(eps), dtype=np.float64)
    for i, ep in enumerate(eps):
        for g in ep.gates:
            X[i, g] += 1.0
        y[i] = np.log(max(ep.fidelity, 1e-12))
    return X, y
