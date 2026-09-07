"""Split discipline for Experiment A.

The plan forbids random time-point splits (they leak the same physical trajectory).
Splits here are always structural:

- horizon split: train on sequences with length <= train_cap, test on longer sequences
  (extrapolation in sequence length);
- bias split: train on a subset of bias settings, test on held-out biases
  (extrapolation in the control/bias direction);
- family split: hold out a gate family (a subset of Clifford indices) — sequences are
  reweighted so that held-out-family sequences form the test set (extrapolation in the
  control-family direction).

All splits are deterministic given their parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Episode, EpisodeDataset

# Gate families over the 24 single-qubit Cliffords, grouped by the Pauli they map to
# (index mod 24 is arbitrary but deterministic; we group by index blocks so families
# are fixed across runs). Family 0: indices 0..7, family 1: 8..15, family 2: 16..23.
GATE_FAMILIES = {0: range(0, 8), 1: range(8, 16), 2: range(16, 24)}


def gate_family(gate: int) -> int:
    for fam, rng in GATE_FAMILIES.items():
        if gate in rng:
            return fam
    raise ValueError(f"gate {gate} outside 0..23")


@dataclass(frozen=True)
class Split:
    name: str
    train: list[Episode]
    test: list[Episode]
    description: str


def horizon_split(ds: EpisodeDataset, train_cap: int = 20) -> Split:
    """Train on length <= train_cap, test on length > train_cap."""
    train = [ep for ep in ds.episodes if ep.length <= train_cap]
    test = [ep for ep in ds.episodes if ep.length > train_cap]
    return Split(
        name=f"horizon_le{train_cap}",
        train=train,
        test=test,
        description=f"train length <= {train_cap}, test length > {train_cap} (cap {ds.length_cap})",
    )


def bias_split(ds: EpisodeDataset, held_out: tuple[float, ...] = (0.61, 0.63)) -> Split:
    """Train on all biases except held_out; test only on held_out biases."""
    ho = set(held_out)
    train = [ep for ep in ds.episodes if ep.bias not in ho]
    test = [ep for ep in ds.episodes if ep.bias in ho]
    return Split(
        name=f"bias_ho{'_'.join(str(b) for b in held_out)}",
        train=train,
        test=test,
        description=f"held-out biases {held_out}",
    )


def family_split(ds: EpisodeDataset, held_out_family: int = 2) -> Split:
    """Hold out one Clifford family.

    Train: sequences containing no gate from the held-out family.
    Test: sequences containing at least one gate from the held-out family.
    This is a control-family extrapolation split.
    """
    ho = set(GATE_FAMILIES[held_out_family])
    train, test = [], []
    for ep in ds.episodes:
        if any(g in ho for g in ep.gates):
            test.append(ep)
        else:
            train.append(ep)
    return Split(
        name=f"family_ho{held_out_family}",
        train=train,
        test=test,
        description=f"train: no gates from family {held_out_family}; test: >=1 gate from it",
    )


def all_splits(ds: EpisodeDataset, train_cap: int | None = None) -> list[Split]:
    if train_cap is None:
        train_cap = min(20, ds.length_cap - 1)
    return [
        horizon_split(ds, train_cap=train_cap),
        bias_split(ds),
        family_split(ds),
    ]


def summarize(split: Split) -> dict:
    def stats(eps: list[Episode]) -> dict:
        if not eps:
            return {"n": 0}
        fids = np.array([ep.fidelity for ep in eps])
        lens = np.array([ep.length for ep in eps])
        return {
            "n": len(eps),
            "len_min": int(lens.min()),
            "len_max": int(lens.max()),
            "fid_mean": float(fids.mean()),
            "fid_std": float(fids.std()),
        }

    return {"name": split.name, "description": split.description, "train": stats(split.train), "test": stats(split.test)}
