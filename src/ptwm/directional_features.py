"""Small, deterministic feature dictionaries for an offline directional screen.

No logical labels or teacher state enter runtime feature construction. Column
selection is a separate fitting operation and its indices must be frozen.
"""
from __future__ import annotations

import numpy as np


def connected_motifs(nodes, pairs, *, max_size=4, limit=256):
    if nodes < 1 or max_size < 3 or limit < 1:
        raise ValueError("invalid motif dictionary bounds")
    adjacency = [set() for _ in range(nodes)]
    level = set()
    for u, v in np.asarray(pairs, dtype=int).reshape(-1, 2):
        if not 0 <= u < nodes or not 0 <= v < nodes or u == v:
            raise ValueError("invalid local edge")
        adjacency[u].add(int(v))
        adjacency[v].add(int(u))
        level.add(tuple(sorted((int(u), int(v)))))
    result = []
    for _ in range(3, max_size + 1):
        grown = set()
        for support in level:
            neighbors = set().union(*(adjacency[v] for v in support))
            for v in neighbors.difference(support):
                grown.add(tuple(sorted((*support, v))))
        level = grown
        result.extend(sorted(level))
        if len(result) >= limit:
            break
    return result[:limit]


def parity_features(detectors, motifs):
    d = np.asarray(detectors, dtype=bool)
    if d.ndim != 2:
        raise ValueError("need a detector matrix")
    result = np.empty((len(d), len(motifs)), dtype=np.float32)
    for column, support in enumerate(motifs):
        result[:, column] = np.logical_xor.reduce(d[:, support], axis=1)
    return result


def feature_dictionary(base, motifs, probability, family):
    base, motifs = np.asarray(base), np.asarray(motifs)
    p = np.asarray(probability).ravel()
    if (base.ndim != 2 or motifs.ndim != 2 or len(base) != len(motifs)
            or len(base) != len(p) or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1))):
        raise ValueError("aligned feature matrices and valid beliefs required")
    if family not in ("base", "parity", "interaction", "both"):
        raise ValueError("unknown feature grammar")
    x = np.column_stack((base, motifs)) if family in ("parity", "both") else base
    if family in ("interaction", "both"):
        x = np.column_stack((x, x * (2 * p[:, None] - 1)))
    return np.asarray(x, dtype=np.float32)


def select_columns(features, target, budget):
    x, y = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or budget < 1:
        raise ValueError("nonempty aligned training matrix and positive budget required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("nonfinite feature selection data")
    centered = x - x.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    covariance = np.abs(centered.T @ (y - y.mean()))
    score = np.divide(covariance, norm, out=np.full_like(norm, -np.inf), where=norm > 1e-12)
    # Stable index tie-breaking and exclusion of constants make replay deterministic.
    order = np.lexsort((np.arange(x.shape[1]), -score))
    return order[np.isfinite(score[order])][:budget]
