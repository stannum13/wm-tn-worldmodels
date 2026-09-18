"""Tests for the NMN-tomo process-matrix physicality residuals."""

import numpy as np

from ptwm.nmn import ppt_negativity, process_residuals, trace_norm_distance


def _cp_process_matrix() -> np.ndarray:
    """A normalized CP process matrix: W = |psi><psi| style rank-1 PSD (16x16)."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=16) + 1j * rng.normal(size=16)
    W = np.outer(v, v.conj())
    return W / np.trace(W).real


def test_cp_matrix_has_zero_psd_violation():
    W = _cp_process_matrix()
    r = process_residuals(W)
    assert r["negative_eigenvalue_mass"] == 0.0
    assert r["n_negative_eigs"] == 0
    assert r["min_eig"] >= -1e-12
    assert r["hermiticity_defect"] < 1e-12
    assert abs(r["trace_real"] - 1.0) < 1e-12


def test_nonphysical_matrix_has_positive_psd_violation():
    W = _cp_process_matrix()
    # add a negative direction
    d = np.zeros(16)
    d[0] = -0.05
    W2 = W + np.diag(d)
    r = process_residuals(W2)
    assert r["negative_eigenvalue_mass"] > 0.04
    assert r["n_negative_eigs"] >= 1


def test_trace_norm_distance():
    A = _cp_process_matrix()
    B = _cp_process_matrix() * 0.0  # zero matrix
    assert trace_norm_distance(A, B) > 0
    assert trace_norm_distance(A, A) == 0.0


def test_ppt_negativity_distinguishes_entangled_and_product_states():
    product = np.zeros((16, 16), dtype=complex)
    product[0, 0] = 1.0

    bell = np.zeros(16, dtype=complex)
    bell[0] = bell[5] = 1 / np.sqrt(2)  # (|00> + |11>) / sqrt(2)
    entangled = np.outer(bell, bell.conj())

    assert ppt_negativity(product) == 0.0
    assert abs(ppt_negativity(entangled) - 0.5) < 1e-12
