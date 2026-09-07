import numpy as np
import torch

from ptwm.clifford import signed_permutation_rotations
from ptwm.oqe import (
    DampedCoherentQubit,
    DissipativeMemoryQubit,
    MarkovianQubitChannel,
    OpenQuantumEvolution,
    QuasiStaticGaussianQubit,
    rotations_to_su2,
)


def test_su2_lift_is_unitary_and_matches_bloch_rotation():
    rotations = signed_permutation_rotations()
    gates = rotations_to_su2(rotations)
    identity = np.eye(2)
    assert max(np.linalg.norm(g.conj().T @ g - identity) for g in gates) < 1e-12
    pauli = np.array([
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ])
    recovered = np.array([
        [[0.5 * np.trace(pauli[i] @ gate @ pauli[j] @ gate.conj().T).real for j in range(3)] for i in range(3)]
        for gate in gates
    ])
    assert np.max(np.abs(recovered - rotations)) < 1e-12


def test_oqe_predictions_are_valid_probabilities():
    gates = rotations_to_su2(signed_permutation_rotations())
    model = OpenQuantumEvolution(gates, memory_dimension=2, seed=0)
    actions = torch.tensor([[0, 1, 2], [3, 4, 0]])
    lengths = torch.tensor([3, 2])
    probabilities = model(actions, lengths)
    assert sum(parameter.numel() for parameter in model.parameters()) == 16
    assert torch.all(probabilities >= -1e-6)
    assert torch.all(probabilities <= 1.0 + 1e-6)


def test_damped_coherent_qubit_is_valid_and_identity_sequence_decays_exactly():
    gates = rotations_to_su2(signed_permutation_rotations())
    model = DampedCoherentQubit(gates, seed=0)
    with torch.no_grad():
        model.error_vector.zero_()
        model.basis_vector.zero_()
        model.retention_logit.copy_(torch.logit(torch.tensor(0.97)))
        model.report_zero_given_zero_logit.copy_(torch.logit(torch.tensor(0.9)))
        model.report_zero_given_one_logit.copy_(torch.logit(torch.tensor(0.1)))
    actions = torch.zeros((3, 5), dtype=torch.long)
    lengths = torch.tensor([1, 3, 5])
    probabilities = model(actions, lengths)
    expected_state = 0.5 + 0.97 ** (lengths + 1) * 0.5
    expected = 0.9 * expected_state + 0.1 * (1 - expected_state)
    assert torch.allclose(probabilities, expected, atol=1e-6)


def test_general_markov_channel_is_trace_preserving_and_predicts_probabilities():
    gates = rotations_to_su2(signed_permutation_rotations())
    model = MarkovianQubitChannel(gates, seed=0)
    kraus = model.kraus()
    completeness = torch.einsum("kji,kjl->il", kraus.conj(), kraus)
    assert torch.allclose(completeness, torch.eye(2, dtype=torch.complex64), atol=1e-5)
    actions = torch.tensor([[0, 1, 2], [3, 4, 0]])
    lengths = torch.tensor([3, 2])
    probabilities = model(actions, lengths)
    assert torch.all(torch.isfinite(probabilities))
    assert torch.all(probabilities >= -1e-5)
    assert torch.all(probabilities <= 1.0 + 1e-5)


def test_dissipative_memory_modes_preserve_density_matrix_and_probabilities():
    gates = rotations_to_su2(signed_permutation_rotations())
    actions = torch.tensor([[0, 1, 2], [3, 4, 0]])
    lengths = torch.tensor([3, 2])
    for mode in ("retain", "dephase", "reset"):
        model = DissipativeMemoryQubit(gates, memory_mode=mode, seed=0)
        probabilities = model(actions, lengths)
        assert torch.all(torch.isfinite(probabilities))
        assert torch.all(probabilities >= -1e-5)
        assert torch.all(probabilities <= 1.0 + 1e-5)

        random = torch.randn(2, 4, 4, dtype=torch.complex64)
        state = random @ random.conj().transpose(-1, -2)
        state = state / torch.diagonal(state, dim1=-2, dim2=-1).sum(-1)[:, None, None]
        transformed = model._noise_step(state, model.unitary())
        assert torch.allclose(
            torch.diagonal(transformed, dim1=-2, dim2=-1).sum(-1),
            torch.ones(2, dtype=torch.complex64), atol=1e-5,
        )
        assert torch.min(torch.linalg.eigvalsh(transformed).real) >= -1e-5


def test_quasistatic_classical_model_has_normalized_ensemble_and_valid_predictions():
    gates = rotations_to_su2(signed_permutation_rotations())
    model = QuasiStaticGaussianQubit(gates, quadrature_order=5, seed=0)
    assert torch.allclose(model.noise_weights.sum(), torch.tensor(1.0), atol=1e-7)
    actions = torch.tensor([[0, 1, 2], [3, 4, 0]])
    lengths = torch.tensor([3, 2])
    probabilities = model(actions, lengths)
    assert torch.all(torch.isfinite(probabilities))
    assert torch.all(probabilities >= -1e-5)
    assert torch.all(probabilities <= 1.0 + 1e-5)
