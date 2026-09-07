"""Synthetic-episode fixtures for tests (no data/ dependency)."""

import numpy as np
import pytest

from ptwm.data import Episode, EpisodeDataset


def make_episode(gates, fidelity, bias=0.1, idle=100, cap=40):
    return Episode(gates=tuple(gates), fidelity=fidelity, bias=bias, idle=idle, length_cap=cap)


def synthetic_dataset(n_per_len: int = 20, max_len: int = 20, seed: int = 0) -> EpisodeDataset:
    """Synthetic RB-like data: log-fidelity decays with length plus per-gate noise,
    with a bias-dependent decay rate so bias splits are meaningful."""
    rng = np.random.default_rng(seed)
    ds = EpisodeDataset(length_cap=max_len, idle=100)
    for bias in (0.1, 0.3, 0.5, 0.61):
        decay = 0.01 + 0.02 * bias  # higher bias -> faster decay
        for L in range(2, max_len + 1):
            for _ in range(n_per_len):
                gates = rng.integers(0, 24, size=L)
                logf = -decay * L + rng.normal(0, 0.01)
                # gate-quality heterogeneity: gates 16..23 slightly worse
                n_bad = int(np.sum(gates >= 16))
                logf -= 0.002 * n_bad
                ds.episodes.append(make_episode(gates, float(np.exp(logf)), bias=bias))
    return ds


@pytest.fixture(scope="session")
def synthetic_ds():
    return synthetic_dataset()


@pytest.fixture(scope="session")
def real_root(tmp_path_factory):
    """Root of the real pt_recovery data if present, else skip."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "external" / "pt_recovery" / "experiment_data"
    if not root.exists():
        pytest.skip("real data not fetched; run scripts/fetch_data.sh")
    return str(root)
