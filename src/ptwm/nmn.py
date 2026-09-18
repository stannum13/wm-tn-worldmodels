"""Physicality residuals for the NMN-tomo multi-time process matrices.

The published artifacts (Ws/*.mat) contain, per detuning point, the experimentally
reconstructed process matrix (Wexp), its physical projection (Wphys), and its
Markovian fit (Wmark). This module computes the observable-space residuals the
plan's `check` contract calls for at the process-matrix level:

- hermiticity defect: ||W - W^dagger||_2 / ||W||_2
- trace (normalization convention as published)
- minimum eigenvalue of the Hermitian part (CP physicality: >= 0)
- negative-eigenvalue mass of the raw reconstruction (a PSD-violation measure)
- partial-transpose negativity of the normalized physical process matrix (the
  quantum non-Markovianity measure used by the source experiment)
- displacement: trace-norm distance ||Wexp - Wphys||_1 (cost of projecting onto
  the physical set) and ||Wexp - Wmark||_1 (distance from the Markovian fit)

These are reference values grounding the CPTP-residual metric that the
sequence-fidelity cells can only surrogate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _herm_part(W: np.ndarray) -> np.ndarray:
    return (W + W.conj().T) / 2


def process_residuals(W: np.ndarray) -> dict:
    W = np.asarray(W, dtype=complex)
    H = _herm_part(W)
    herm_defect = float(np.linalg.norm(W - W.conj().T, 2) / max(np.linalg.norm(W, 2), 1e-12))
    eigs = np.linalg.eigvalsh(H)
    neg = eigs[eigs < -1e-12]
    return {
        "hermiticity_defect": herm_defect,
        "trace_real": float(np.trace(W).real),
        "trace_imag": float(np.trace(W).imag),
        "min_eig": float(eigs.min()),
        "negative_eigenvalue_mass": float(-neg.sum()) if neg.size else 0.0,
        "n_negative_eigs": int(neg.size),
    }


def ppt_negativity(W: np.ndarray, subsystem_dim: int = 4) -> float:
    """Return bipartite partial-transpose negativity.

    NMN-tomo treats the 16x16 process matrix as a 4x4 bipartite state and
    reports the larger negative-eigenvalue mass obtained by transposing either
    subsystem. ``W`` is expected to have unit trace.
    """
    W = np.asarray(W, dtype=complex)
    expected = subsystem_dim**2
    if W.shape != (expected, expected):
        raise ValueError(f"expected a {expected}x{expected} matrix, got {W.shape}")
    tensor = W.reshape(subsystem_dim, subsystem_dim, subsystem_dim, subsystem_dim)
    partial_a = tensor.transpose(2, 1, 0, 3).reshape(expected, expected)
    partial_b = tensor.transpose(0, 3, 2, 1).reshape(expected, expected)

    def negative_mass(matrix: np.ndarray) -> float:
        eigs = np.linalg.eigvalsh(_herm_part(matrix))
        return float(-eigs[eigs < -1e-12].sum())

    return max(negative_mass(partial_a), negative_mass(partial_b))


def trace_norm_distance(A: np.ndarray, B: np.ndarray) -> float:
    """||A - B||_1 (sum of singular values)."""
    return float(np.linalg.svd(A - B, compute_uv=False).sum())


def analyze(nmn_root: str = "data/external/NMN-tomo") -> dict:
    ws = Path(nmn_root) / "Ws"
    exp = loadmat(ws / "Wexp_all.mat")
    phys = loadmat(ws / "Wphys_all.mat")
    mark = loadmat(ws / "Wmark_all.mat")

    def strip(mat: dict) -> dict:
        return {k: v for k, v in mat.items() if not k.startswith("__")}

    exp, phys, mark = strip(exp), strip(phys), strip(mark)

    def detuning_of(key: str) -> str:
        # keys look like Wexp_ibm_2121 / Wph_ibm_797 / Wm_uq_9797
        tail = key.split("_")[-1]
        return tail

    results = {"points": {}}
    for key, W in sorted(exp.items()):
        tail = detuning_of(key)
        # match phys/mark by trailing detuning tag
        p_key = next((k for k in phys if k.split("_")[-1] == tail), None)
        m_key = next((k for k in mark if k.split("_")[-1] == tail), None)
        entry = {"exp": process_residuals(W)}
        if p_key is not None:
            entry["phys"] = process_residuals(phys[p_key])
            entry["disp_exp_phys"] = trace_norm_distance(W, phys[p_key])
            trace = np.trace(phys[p_key]).real
            entry["quantum_non_markovianity"] = ppt_negativity(phys[p_key] / trace)
        if m_key is not None:
            entry["mark"] = process_residuals(mark[m_key])
            entry["disp_exp_mark"] = trace_norm_distance(W, mark[m_key])
        results["points"][tail] = entry

    # Summary across the 9-point IBM detuning grid.
    tails = sorted(results["points"])
    agg = {}
    for variant in ("exp", "phys", "mark"):
        vals = [results["points"][t].get(variant) for t in tails]
        vals = [v for v in vals if v]
        agg[variant] = {
            "negative_eigenvalue_mass_mean": float(np.mean([v["negative_eigenvalue_mass"] for v in vals])),
            "negative_eigenvalue_mass_max": float(np.max([v["negative_eigenvalue_mass"] for v in vals])),
            "min_eig_mean": float(np.mean([v["min_eig"] for v in vals])),
            "hermiticity_defect_max": float(np.max([v["hermiticity_defect"] for v in vals])),
        }
    qnm = [results["points"][t]["quantum_non_markovianity"] for t in tails if "quantum_non_markovianity" in results["points"][t]]
    agg["quantum_non_markovianity_mean"] = float(np.mean(qnm))
    agg["quantum_non_markovianity_max"] = float(np.max(qnm))
    agg["disp_exp_phys_mean"] = float(np.mean([results["points"][t]["disp_exp_phys"] for t in tails if "disp_exp_phys" in results["points"][t]]))
    agg["disp_exp_mark_mean"] = float(np.mean([results["points"][t]["disp_exp_mark"] for t in tails if "disp_exp_mark" in results["points"][t]]))
    results["summary"] = agg
    return results


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external/NMN-tomo")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    res = analyze(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "nmn_physicality.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {path}")
    s = res["summary"]
    print(f"raw PSD violation: exp mean={s['exp']['negative_eigenvalue_mass_mean']:.4f} "
          f"max={s['exp']['negative_eigenvalue_mass_max']:.4f} | "
          f"phys mean={s['phys']['negative_eigenvalue_mass_mean']:.4f} | "
          f"mark mean={s['mark']['negative_eigenvalue_mass_mean']:.4f}")
    print(f"physical-matrix PPT negativity: mean={s['quantum_non_markovianity_mean']:.4f} "
          f"max={s['quantum_non_markovianity_max']:.4f}")
    print(f"min eig: exp mean={s['exp']['min_eig_mean']:.4f} | phys mean={s['phys']['min_eig_mean']:.4f} | mark mean={s['mark']['min_eig_mean']:.4f}")
    print(f"displacement: exp->phys {s['disp_exp_phys_mean']:.4f} | exp->mark {s['disp_exp_mark_mean']:.4f}")


if __name__ == "__main__":
    main()
