"""Compact differentiable open-quantum-evolution primitives."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn


_PAULI = torch.tensor(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
    dtype=torch.complex64,
)


def _su2_from_vector(vector: torch.Tensor) -> torch.Tensor:
    """Exponentiate -i vector.sigma without an unobservable identity term."""
    pauli = _PAULI.to(vector.device)
    return torch.matrix_exp(-1j * torch.einsum("...i,ijk->...jk", vector.to(torch.complex64), pauli))


def rotations_to_su2(rotations: np.ndarray) -> np.ndarray:
    """Lift proper SO(3) rotations to SU(2); signs/global phases are irrelevant."""
    quaternion = Rotation.from_matrix(np.asarray(rotations, dtype=float)).as_quat()
    x, y, z, w = quaternion.T
    output = np.empty((len(rotations), 2, 2), dtype=np.complex128)
    output[:, 0, 0] = w - 1j * z
    output[:, 0, 1] = -y - 1j * x
    output[:, 1, 0] = y - 1j * x
    output[:, 1, 1] = w + 1j * z
    return output


class OpenQuantumEvolution(nn.Module):
    """Repeated unitary evolution of a qubit and a finite memory."""

    def __init__(self, gates: np.ndarray, memory_dimension: int, seed: int = 0, initial_scale: float = 0.05):
        super().__init__()
        torch.manual_seed(seed)
        self.memory_dimension = memory_dimension
        self.register_buffer("gates", torch.tensor(gates, dtype=torch.complex64))
        dimension = 2 * memory_dimension
        self.generator_diagonal = nn.Parameter(torch.randn(dimension) * initial_scale)
        self.generator_upper_real = nn.Parameter(torch.randn(dimension * (dimension - 1) // 2) * initial_scale)
        self.generator_upper_imag = nn.Parameter(torch.randn(dimension * (dimension - 1) // 2) * initial_scale)

    def unitary(self):
        dimension = self.generator_diagonal.numel()
        rows, columns = torch.triu_indices(dimension, dimension, offset=1)
        hermitian = torch.diag(self.generator_diagonal).to(torch.complex64)
        upper = self.generator_upper_real.to(torch.complex64) + 1j * self.generator_upper_imag.to(torch.complex64)
        hermitian[rows, columns] = upper
        hermitian[columns, rows] = upper.conj()
        return torch.matrix_exp(-1j * hermitian)

    def forward(self, actions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, width = actions.shape
        memory = self.memory_dimension
        state = torch.zeros((batch, 2, memory), dtype=torch.complex64, device=actions.device)
        state[:, 0, 0] = 1.0
        evolution = self.unitary()
        state = (state.reshape(batch, -1) @ evolution.T).reshape(batch, 2, memory)
        for t in range(width):
            gate = self.gates[actions[:, t]]
            after_gate = torch.bmm(gate, state)
            candidate = (after_gate.reshape(batch, -1) @ evolution.T).reshape(batch, 2, memory)
            state = torch.where((t < lengths)[:, None, None], candidate, state)
        return torch.sum(torch.abs(state[:, 0, :]) ** 2, dim=1).real


class DampedCoherentQubit(nn.Module):
    """Coherent one-qubit error plus isotropic contraction and bounded readout.

    A shared basis unitary aligns the recovered abstract Clifford representation with
    the prepared/measured laboratory axis. Isotropic contraction commutes with every
    unitary, so pure-state propagation remains exact for this model.
    """

    def __init__(self, gates: np.ndarray, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("gates", torch.tensor(gates, dtype=torch.complex64))
        self.error_vector = nn.Parameter(torch.randn(3) * 0.05)
        self.basis_vector = nn.Parameter(torch.randn(3) * 0.02)
        self.retention_logit = nn.Parameter(torch.logit(torch.tensor(0.995)))
        self.report_zero_given_zero_logit = nn.Parameter(torch.logit(torch.tensor(0.9)))
        self.report_zero_given_one_logit = nn.Parameter(torch.logit(torch.tensor(0.1)))

    def physical_parameters(self):
        return {
            "retention": torch.sigmoid(self.retention_logit),
            "report_zero_given_zero": torch.sigmoid(self.report_zero_given_zero_logit),
            "report_zero_given_one": torch.sigmoid(self.report_zero_given_one_logit),
        }

    def forward(self, actions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, width = actions.shape
        error = _su2_from_vector(self.error_vector)
        basis = _su2_from_vector(self.basis_vector)
        initial = basis[:, 0].expand(batch, -1)
        state = initial @ error.T
        for t in range(width):
            after_gate = torch.bmm(self.gates[actions[:, t]], state[:, :, None]).squeeze(-1)
            candidate = after_gate @ error.T
            state = torch.where((t < lengths)[:, None], candidate, state)
        pure_probability = torch.abs(torch.sum(initial.conj() * state, dim=1)) ** 2
        parameters = self.physical_parameters()
        contracted = 0.5 + parameters["retention"] ** (lengths + 1) * (pure_probability.real - 0.5)
        return (
            parameters["report_zero_given_zero"] * contracted
            + parameters["report_zero_given_one"] * (1 - contracted)
        )


class MarkovianQubitChannel(nn.Module):
    """A general time-homogeneous qubit CPTP channel with four Kraus operators."""

    def __init__(self, gates: np.ndarray, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("gates", torch.tensor(gates, dtype=torch.complex64))
        raw = torch.randn(4, 2, 2, dtype=torch.complex64) * 0.01
        raw[0] += torch.eye(2)
        self.kraus_raw_real = nn.Parameter(raw.real)
        self.kraus_raw_imag = nn.Parameter(raw.imag)
        self.coherent_vector = nn.Parameter(torch.randn(3) * 0.05)
        self.basis_vector = nn.Parameter(torch.randn(3) * 0.02)
        self.report_zero_given_zero_logit = nn.Parameter(torch.logit(torch.tensor(0.9)))
        self.report_zero_given_one_logit = nn.Parameter(torch.logit(torch.tensor(0.1)))

    def kraus(self) -> torch.Tensor:
        raw = self.kraus_raw_real.to(torch.complex64) + 1j * self.kraus_raw_imag.to(torch.complex64)
        gram = torch.einsum("kji,kjl->il", raw.conj(), raw)
        values, vectors = torch.linalg.eigh(gram)
        inverse_sqrt = (vectors * torch.rsqrt(torch.clamp(values, min=1e-7))) @ vectors.conj().T
        return raw @ inverse_sqrt @ _su2_from_vector(self.coherent_vector)

    @staticmethod
    def apply_channel(state: torch.Tensor, kraus: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kij,bjl,kml->bim", kraus, state, kraus.conj())

    def forward(self, actions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, width = actions.shape
        basis = _su2_from_vector(self.basis_vector)
        initial = basis[:, 0]
        state = torch.einsum("i,j->ij", initial, initial.conj()).expand(batch, -1, -1).clone()
        kraus = self.kraus()
        state = self.apply_channel(state, kraus)
        for t in range(width):
            gate = self.gates[actions[:, t]]
            after_gate = gate @ state @ gate.conj().transpose(-1, -2)
            candidate = self.apply_channel(after_gate, kraus)
            state = torch.where((t < lengths)[:, None, None], candidate, state)
        ideal_zero = torch.einsum("i,bij,j->b", initial.conj(), state, initial).real
        a = torch.sigmoid(self.report_zero_given_zero_logit)
        b = torch.sigmoid(self.report_zero_given_one_logit)
        return a * ideal_zero + b * (1 - ideal_zero)


class DissipativeMemoryQubit(nn.Module):
    """Qubit plus D2 memory with local damping and controlled memory retention.

    ``memory_mode`` chooses whether the memory is retained coherently, dephased in a
    fixed (but interaction-adaptable) basis, or reset after every noise step.
    """

    def __init__(self, gates: np.ndarray, memory_mode: str = "retain", seed: int = 0):
        super().__init__()
        if memory_mode not in {"retain", "dephase", "reset"}:
            raise ValueError("memory_mode must be retain, dephase, or reset")
        torch.manual_seed(seed)
        self.memory_mode = memory_mode
        self.register_buffer("gates", torch.tensor(gates, dtype=torch.complex64))
        self.generator_diagonal = nn.Parameter(torch.randn(3) * 0.05)
        self.generator_upper_real = nn.Parameter(torch.randn(6) * 0.05)
        self.generator_upper_imag = nn.Parameter(torch.randn(6) * 0.05)
        self.basis_vector = nn.Parameter(torch.randn(3) * 0.02)
        self.retention_logit = nn.Parameter(torch.logit(torch.tensor(0.995)))
        self.report_zero_given_zero_logit = nn.Parameter(torch.logit(torch.tensor(0.9)))
        self.report_zero_given_one_logit = nn.Parameter(torch.logit(torch.tensor(0.1)))

    def unitary(self):
        diagonal = torch.cat((self.generator_diagonal, -self.generator_diagonal.sum()[None]))
        rows, columns = torch.triu_indices(4, 4, offset=1)
        hermitian = torch.diag(diagonal).to(torch.complex64)
        upper = self.generator_upper_real.to(torch.complex64) + 1j * self.generator_upper_imag.to(torch.complex64)
        hermitian[rows, columns] = upper
        hermitian[columns, rows] = upper.conj()
        return torch.matrix_exp(-1j * hermitian)

    @staticmethod
    def _system_replacement(state: torch.Tensor) -> torch.Tensor:
        shaped = state.reshape(-1, 2, 2, 2, 2)
        memory_state = torch.einsum("bseue->bsu", shaped)
        identity = torch.eye(2, dtype=state.dtype, device=state.device) / 2
        return torch.einsum("ij,bkl->bikjl", identity, memory_state).reshape(-1, 4, 4)

    def _memory_operation(self, state: torch.Tensor) -> torch.Tensor:
        shaped = state.reshape(-1, 2, 2, 2, 2)
        if self.memory_mode == "retain":
            return state
        if self.memory_mode == "dephase":
            mask = torch.eye(2, dtype=state.dtype, device=state.device)
            return (shaped * mask[None, None, :, None, :]).reshape(-1, 4, 4)
        system_state = torch.einsum("bseue->bsu", shaped)
        memory_zero = torch.zeros((2, 2), dtype=state.dtype, device=state.device)
        memory_zero[0, 0] = 1
        return torch.einsum("bij,kl->bikjl", system_state, memory_zero).reshape(-1, 4, 4)

    def _noise_step(self, state: torch.Tensor, unitary: torch.Tensor) -> torch.Tensor:
        state = unitary @ state @ unitary.conj().T
        retention = torch.sigmoid(self.retention_logit)
        state = retention * state + (1 - retention) * self._system_replacement(state)
        return self._memory_operation(state)

    def forward(self, actions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, width = actions.shape
        basis = _su2_from_vector(self.basis_vector)
        system_zero = basis[:, 0]
        joint_zero = torch.kron(system_zero, torch.tensor([1, 0], dtype=torch.complex64, device=actions.device))
        state = torch.einsum("i,j->ij", joint_zero, joint_zero.conj()).expand(batch, -1, -1).clone()
        unitary = self.unitary()
        state = self._noise_step(state, unitary)
        memory_identity = torch.eye(2, dtype=torch.complex64, device=actions.device)
        for t in range(width):
            gate = self.gates[actions[:, t]]
            joint_gate = torch.einsum("bij,kl->bikjl", gate, memory_identity).reshape(batch, 4, 4)
            after_gate = joint_gate @ state @ joint_gate.conj().transpose(-1, -2)
            candidate = self._noise_step(after_gate, unitary)
            state = torch.where((t < lengths)[:, None, None], candidate, state)
        shaped = state.reshape(batch, 2, 2, 2, 2)
        system_state = torch.einsum("bseue->bsu", shaped)
        ideal_zero = torch.einsum("i,bij,j->b", system_zero.conj(), system_state, system_zero).real
        a = torch.sigmoid(self.report_zero_given_zero_logit)
        b = torch.sigmoid(self.report_zero_given_one_logit)
        return a * ideal_zero + b * (1 - ideal_zero)


class QuasiStaticGaussianQubit(nn.Module):
    """Classical sequence-persistent Gaussian error averaged by fixed quadrature."""

    def __init__(self, gates: np.ndarray, quadrature_order: int = 5, seed: int = 0):
        super().__init__()
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        torch.manual_seed(seed)
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
        self.register_buffer("gates", torch.tensor(gates, dtype=torch.complex64))
        self.register_buffer("noise_nodes", torch.tensor(np.sqrt(2) * nodes, dtype=torch.float32))
        self.register_buffer("noise_weights", torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float32))
        self.coherent_vector = nn.Parameter(torch.randn(3) * 0.1)
        self.noise_axis_raw = nn.Parameter(torch.randn(3))
        self.noise_scale_raw = nn.Parameter(torch.tensor(-2.0))
        self.basis_vector = nn.Parameter(torch.randn(3) * 0.02)
        self.retention_logit = nn.Parameter(torch.logit(torch.tensor(0.995)))
        self.report_zero_given_zero_logit = nn.Parameter(torch.logit(torch.tensor(0.9)))
        self.report_zero_given_one_logit = nn.Parameter(torch.logit(torch.tensor(0.1)))

    def physical_parameters(self):
        return {
            "noise_scale": torch.nn.functional.softplus(self.noise_scale_raw),
            "retention": torch.sigmoid(self.retention_logit),
            "report_zero_given_zero": torch.sigmoid(self.report_zero_given_zero_logit),
            "report_zero_given_one": torch.sigmoid(self.report_zero_given_one_logit),
        }

    def forward(self, actions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, width = actions.shape
        axis = self.noise_axis_raw / torch.clamp(torch.linalg.norm(self.noise_axis_raw), min=1e-7)
        scale = torch.nn.functional.softplus(self.noise_scale_raw)
        vectors = self.coherent_vector[None, :] + scale * self.noise_nodes[:, None] * axis[None, :]
        errors = _su2_from_vector(vectors)
        basis = _su2_from_vector(self.basis_vector)
        initial = basis[:, 0]
        state = initial.expand(batch, len(errors), -1)
        state = torch.einsum("kij,bkj->bki", errors, state)
        for t in range(width):
            gate = self.gates[actions[:, t]]
            after_gate = torch.einsum("bij,bkj->bki", gate, state)
            candidate = torch.einsum("kij,bkj->bki", errors, after_gate)
            state = torch.where((t < lengths)[:, None, None], candidate, state)
        amplitudes = torch.einsum("i,bki->bk", initial.conj(), state)
        pure_probability = torch.sum(self.noise_weights[None, :] * torch.abs(amplitudes) ** 2, dim=1)
        parameters = self.physical_parameters()
        contracted = 0.5 + parameters["retention"] ** (lengths + 1) * (pure_probability.real - 0.5)
        return (
            parameters["report_zero_given_zero"] * contracted
            + parameters["report_zero_given_one"] * (1 - contracted)
        )
