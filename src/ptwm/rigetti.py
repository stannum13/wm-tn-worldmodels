"""Small, auditable utilities for the public Rigetti Ankaa-2 feedback data."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .streaming import _head_design


@dataclass(frozen=True)
class LogisticIQHead:
    weights: np.ndarray
    feature_min: np.ndarray
    feature_max: np.ndarray
    knots: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        design = _head_design(features, self.feature_min, self.feature_max, self.knots)
        return expit(design @ self.weights)


@dataclass(frozen=True)
class MarkovSyndromeDecoder:
    """Causal class-conditional lookup model over per-round syndrome symbols."""

    log_prior: np.ndarray
    log_conditionals: tuple[np.ndarray, ...]
    alphabet: int
    order: int

    def predict(self, symbols: np.ndarray) -> np.ndarray:
        symbols = np.asarray(symbols, dtype=int)
        scores = np.broadcast_to(self.log_prior, (len(symbols), 2)).copy()
        for t in range(symbols.shape[1]):
            context_order = min(t, self.order)
            if context_order:
                context = np.zeros(len(symbols), dtype=int)
                for previous in symbols[:, t - context_order : t].T:
                    context = context * self.alphabet + previous
            else:
                context = np.zeros(len(symbols), dtype=int)
            for label in (0, 1):
                scores[:, label] += self.log_conditionals[context_order][
                    label, context, symbols[:, t]
                ]
        return expit(scores[:, 1] - scores[:, 0])


def load_measurement_fidelity(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load labelled complex I/Q calibration shots and hardware hard decisions."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by real-data runner
        raise RuntimeError("Install the 'real' optional dependencies to read Rigetti HDF5") from exc
    soft_parts, hard_parts, label_parts = [], [], []
    with h5py.File(path, "r") as handle:
        root = handle["reference_data/measurement_fidelity"]
        for prepared, label in (("zeros", 0.0), ("ones", 1.0)):
            result = root[f"{prepared}/result"]
            soft = np.asarray(result["soft_measurements/50"]).reshape(-1)
            hard = np.asarray(result["hard_measurements/50"]).reshape(-1)
            soft_parts.append(np.c_[soft.real, soft.imag])
            hard_parts.append(hard.astype(float))
            label_parts.append(np.full(len(soft), label))
    return np.vstack(soft_parts), np.concatenate(hard_parts), np.concatenate(label_parts)


def load_qec_record(path: str, circuit_group: str) -> dict[str, object]:
    """Reconstruct chronological Stim measurement records from per-qubit HDF5 arrays."""
    try:
        import h5py
        import stim
    except ImportError as exc:  # pragma: no cover - exercised by real-data runner
        raise RuntimeError("Install the 'real' optional dependencies for QEC replay") from exc
    with h5py.File(path, "r") as handle:
        group = handle[circuit_group]
        circuit_text = group["stim_circuits/stim_circuit_0"][()].decode()
        circuit = stim.Circuit(circuit_text).flattened()
        mapping = np.asarray(group["qubit_mapping"])
        physical_qubit = {local: int(row[2]) for local, row in enumerate(mapping)}
        hard_group = group["result/hard_measurements"]
        soft_group = group["result/soft_measurements"]
        cursors: dict[int, int] = {}
        hard_columns, soft_columns, order = [], [], []
        for instruction in circuit:
            if instruction.name != "M":
                continue
            for target in instruction.targets_copy():
                qubit = physical_qubit[target.value]
                column = cursors.get(qubit, 0)
                hard_columns.append(np.asarray(hard_group[str(qubit)])[:, column])
                soft_columns.append(np.asarray(soft_group[str(qubit)])[:, column])
                order.append(qubit)
                cursors[qubit] = column + 1
        hard = np.stack(hard_columns, axis=1).astype(bool)
        soft = np.stack(soft_columns, axis=1).astype(np.complex64)
    if hard.shape[1] != circuit.num_measurements:
        raise ValueError(f"reconstructed {hard.shape[1]} measurements; Stim expects {circuit.num_measurements}")
    detectors, observables = circuit.compile_m2d_converter().convert(
        measurements=hard, separate_observables=True
    )
    return {
        "circuit": circuit,
        "hard_measurements": hard,
        "soft_measurements": soft,
        "measurement_qubits": np.asarray(order),
        "detectors": detectors.astype(bool),
        "observables": observables.astype(bool),
    }


def detector_measurement_indices(circuit: object) -> list[np.ndarray]:
    """Resolve Stim rec offsets for each detector in an already-flattened circuit."""
    measurement_cursor = 0
    output = []
    for instruction in circuit:
        if instruction.name == "M":
            measurement_cursor += len(instruction.targets_copy())
        elif instruction.name == "DETECTOR":
            indices = [
                measurement_cursor + target.value
                for target in instruction.targets_copy()
                if target.is_measurement_record_target
            ]
            output.append(np.asarray(indices, dtype=int))
    return output


def soft_detector_probabilities(
    measurement_probability: np.ndarray, detector_indices: list[np.ndarray],
    hard_measurements: np.ndarray, hard_detectors: np.ndarray,
) -> np.ndarray:
    """Propagate independent bit probabilities through detector parities.

    Stim reference-sample offsets are inferred by comparing raw hard parities to its
    converter output. This avoids silently assuming every detector has zero offset.
    """
    output = np.empty((len(measurement_probability), len(detector_indices)))
    for detector, indices in enumerate(detector_indices):
        probability = 0.5 * (1.0 - np.prod(1.0 - 2.0 * measurement_probability[:, indices], axis=1))
        raw = np.logical_xor.reduce(hard_measurements[:, indices], axis=1)
        offset = bool(raw[0] != hard_detectors[0, detector])
        if not np.all((raw != offset) == hard_detectors[:, detector]):
            raise ValueError("detector parity does not match a fixed Stim reference offset")
        output[:, detector] = 1.0 - probability if offset else probability
    return output


def detector_geometry(circuit: object) -> np.ndarray:
    coordinates = circuit.get_detector_coordinates()
    return np.asarray([coordinates[index][:3] for index in range(len(coordinates))], dtype=float)


def local_detector_pairs(
    circuit: object, *, maximum_time_lag: float = 1.0, maximum_spatial_distance: float = 2.01
) -> np.ndarray:
    """Circuit-local detector pairs without target-informed feature selection."""
    geometry = detector_geometry(circuit)
    pairs = []
    for first in range(len(geometry)):
        delta = np.abs(geometry[first + 1 :] - geometry[first])
        selected = np.flatnonzero(
            (delta[:, 2] <= maximum_time_lag)
            & (delta[:, 0] + delta[:, 1] <= maximum_spatial_distance)
        )
        pairs.extend((first, first + 1 + int(offset)) for offset in selected)
    return np.asarray(pairs, dtype=int)


def local_pair_features(detectors: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    detectors = np.asarray(detectors, dtype=float)
    return detectors[:, pairs[:, 0]] * detectors[:, pairs[:, 1]]


def detector_worldline_parities(detectors: np.ndarray, circuit: object) -> np.ndarray:
    """Cumulative parity state along each repeated detector coordinate."""
    detectors = np.asarray(detectors)
    geometry = detector_geometry(circuit)
    output = np.empty_like(detectors, dtype=float)
    for coordinate in np.unique(geometry[:, :2], axis=0):
        columns = np.flatnonzero(np.all(geometry[:, :2] == coordinate, axis=1))
        columns = columns[np.argsort(geometry[columns, 2])]
        if np.all((detectors[:, columns] == 0) | (detectors[:, columns] == 1)):
            output[:, columns] = np.logical_xor.accumulate(detectors[:, columns].astype(bool), axis=1)
        else:
            output[:, columns] = 0.5 * (
                1.0 - np.cumprod(1.0 - 2.0 * detectors[:, columns], axis=1)
            )
    return output


def detector_round_symbols(detectors: np.ndarray, circuit: object) -> np.ndarray:
    """Pack the four spatial detector bits at each time into a categorical symbol."""
    geometry = detector_geometry(circuit)
    times = np.unique(geometry[:, 2])
    groups = [np.flatnonzero(geometry[:, 2] == time) for time in times]
    widths = {len(group) for group in groups}
    if len(widths) != 1:
        raise ValueError(f"detector width changes across rounds: {sorted(widths)}")
    output = np.empty((len(detectors), len(groups)), dtype=int)
    for t, columns in enumerate(groups):
        columns = columns[np.lexsort((geometry[columns, 1], geometry[columns, 0]))]
        output[:, t] = np.asarray(detectors[:, columns], dtype=int) @ (1 << np.arange(len(columns)))
    return output


def fit_markov_syndrome_decoder(
    symbols: np.ndarray, labels: np.ndarray, *, order: int, alpha: float = 1.0
) -> MarkovSyndromeDecoder:
    """Fit an order-k generative decoder with Dirichlet-smoothed lookup tables."""
    symbols, labels = np.asarray(symbols, dtype=int), np.asarray(labels, dtype=int)
    if order < 0 or order >= symbols.shape[1]:
        raise ValueError("order must be non-negative and shorter than the sequence")
    alphabet = int(symbols.max()) + 1
    # The packed alphabet is a power of two even when some symbols are absent in train.
    alphabet = 1 << int(np.ceil(np.log2(max(alphabet, 2))))
    prior = np.bincount(labels, minlength=2).astype(float) + alpha
    prior /= prior.sum()
    tables = []
    for context_order in range(order + 1):
        counts = np.full((2, alphabet**context_order, alphabet), alpha, dtype=float)
        for t in range(context_order, symbols.shape[1]):
            if context_order:
                context = np.zeros(len(symbols), dtype=int)
                for previous in symbols[:, t - context_order : t].T:
                    context = context * alphabet + previous
            else:
                context = np.zeros(len(symbols), dtype=int)
            np.add.at(counts, (labels, context, symbols[:, t]), 1.0)
        counts /= counts.sum(axis=2, keepdims=True)
        tables.append(np.log(counts))
    return MarkovSyndromeDecoder(np.log(prior), tuple(tables), alphabet, order)


def timed_markov_prediction(
    decoder: MarkovSyndromeDecoder, symbols: np.ndarray, repeats: int = 20
) -> tuple[np.ndarray, dict[str, float]]:
    prediction = decoder.predict(symbols)
    started = perf_counter_ns()
    for _ in range(repeats):
        decoder.predict(symbols)
    vectorized = (perf_counter_ns() - started) / (repeats * len(symbols))
    calls = []
    for row in symbols[: min(2000, len(symbols))]:
        started = perf_counter_ns()
        decoder.predict(row[None, :])
        calls.append(perf_counter_ns() - started)
    return prediction, {
        "vectorized_ns_per_shot": float(vectorized),
        "python_batch_one_p50_ns": float(np.quantile(calls, 0.50)),
        "python_batch_one_p99_ns": float(np.quantile(calls, 0.99)),
    }


def uniform_circuit_noise_model(circuit: object, probability: float) -> object:
    """Build a transparent circuit-level noise control when no calibrated DEM ships."""
    try:
        import stim
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'real' optional dependencies for Stim") from exc
    noisy = stim.Circuit()
    single_qubit_gates = {
        "H", "H_XY", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "X", "Y", "Z"
    }
    two_qubit_gates = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
    reset_gates = {"R", "RX", "RY"}
    measure_reset_gates = {"MR", "MRX", "MRY"}
    for instruction in circuit.flattened():
        targets = instruction.targets_copy()
        if instruction.name in {"M", *measure_reset_gates}:
            noisy.append("X_ERROR", targets, probability)
            noisy.append(instruction)
            if instruction.name in measure_reset_gates:
                noisy.append("X_ERROR", targets, probability)
        else:
            noisy.append(instruction)
            if instruction.name in two_qubit_gates:
                noisy.append("DEPOLARIZE2", targets, probability)
            elif instruction.name in single_qubit_gates:
                noisy.append("DEPOLARIZE1", targets, probability)
            elif instruction.name in reset_gates:
                noisy.append("X_ERROR", targets, probability)
    return noisy.detector_error_model(decompose_errors=True)


def matching_predictions(
    circuit: object, detectors: np.ndarray, *, probability: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Decode with PyMatching using the explicit uniform circuit-noise control."""
    try:
        import pymatching
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'real' optional dependencies for PyMatching") from exc
    model = uniform_circuit_noise_model(circuit, probability)
    matching = pymatching.Matching.from_detector_error_model(model)
    started = perf_counter_ns()
    prediction = matching.decode_batch(np.asarray(detectors, dtype=np.uint8))
    batch_ns = (perf_counter_ns() - started) / len(detectors)
    calls = []
    for row in detectors[: min(2000, len(detectors))]:
        started = perf_counter_ns()
        matching.decode(np.asarray(row, dtype=np.uint8))
        calls.append(perf_counter_ns() - started)
    return np.asarray(prediction)[:, 0].astype(float), {
        "vectorized_ns_per_shot": float(batch_ns),
        "python_batch_one_p50_ns": float(np.quantile(calls, 0.50)),
        "python_batch_one_p99_ns": float(np.quantile(calls, 0.99)),
        "detector_error_model_terms": len(model),
    }


def chronological_preparation_split(labels: np.ndarray, train_fraction: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Take the early portion of each prepared-state trace for train, later for test."""
    labels = np.asarray(labels)
    train, test = [], []
    for label in np.unique(labels):
        locations = np.flatnonzero(labels == label)
        boundary = int(len(locations) * train_fraction)
        train.extend(locations[:boundary])
        test.extend(locations[boundary:])
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def probability_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(labels, dtype=float)
    prediction = probability >= 0.5
    edges = np.linspace(0.0, 1.0, 11)
    calibration_error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (probability >= left) & (probability < right if right < 1 else probability <= right)
        if np.any(selected):
            calibration_error += np.mean(selected) * abs(np.mean(probability[selected]) - np.mean(labels[selected]))
    return {
        "classification_error": float(np.mean(prediction != labels)),
        "brier_loss": float(np.mean((probability - labels) ** 2)),
        "negative_log_likelihood": float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))),
        "expected_calibration_error_10bin": float(calibration_error),
    }


def block_error_differences(
    first: np.ndarray, second: np.ndarray, labels: np.ndarray, *, block_size: int
) -> np.ndarray:
    """Paired error-rate differences in contiguous blocks within each preparation."""
    differences = []
    for label in np.unique(labels):
        locations = np.flatnonzero(labels == label)
        for start in range(0, len(locations) - block_size + 1, block_size):
            block = locations[start : start + block_size]
            first_error = (np.asarray(first)[block] >= 0.5) != labels[block]
            second_error = (np.asarray(second)[block] >= 0.5) != labels[block]
            differences.append(np.mean(first_error) - np.mean(second_error))
    return np.asarray(differences)


def block_score_differences(
    first: np.ndarray, second: np.ndarray, labels: np.ndarray, *, block_size: int,
    score: str,
) -> np.ndarray:
    """First-minus-second paired score differences in contiguous preparation blocks."""
    first = np.clip(np.asarray(first), 1e-6, 1.0 - 1e-6)
    second = np.clip(np.asarray(second), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(labels)
    if score == "brier_loss":
        first_values, second_values = (first - labels) ** 2, (second - labels) ** 2
    elif score == "negative_log_likelihood":
        first_values = -(labels * np.log(first) + (1 - labels) * np.log(1 - first))
        second_values = -(labels * np.log(second) + (1 - labels) * np.log(1 - second))
    else:
        raise ValueError(f"unsupported score: {score}")
    differences = []
    for label in np.unique(labels):
        locations = np.flatnonzero(labels == label)
        for start in range(0, len(locations) - block_size + 1, block_size):
            block = locations[start : start + block_size]
            differences.append(np.mean(first_values[block] - second_values[block]))
    return np.asarray(differences)


def timed_head_prediction(head: LogisticIQHead, features: np.ndarray, repeats: int = 20) -> tuple[np.ndarray, dict[str, float]]:
    """Measure vectorized throughput and Python batch-one call latency."""
    prediction = head.predict(features)
    started = perf_counter_ns()
    for _ in range(repeats):
        head.predict(features)
    batch_ns = (perf_counter_ns() - started) / (repeats * len(features))
    calls = []
    for row in features[: min(2000, len(features))]:
        started = perf_counter_ns()
        head.predict(row[None, :])
        calls.append(perf_counter_ns() - started)
    return prediction, {
        "vectorized_ns_per_sample": float(batch_ns),
        "python_batch_one_p50_ns": float(np.quantile(calls, 0.50)),
        "python_batch_one_p99_ns": float(np.quantile(calls, 0.99)),
    }


def _fit_logistic_head(
    features: np.ndarray, labels: np.ndarray, *, knots: int, ridge: float = 1e-4
) -> LogisticIQHead:
    features, labels = np.asarray(features), np.asarray(labels)
    low, high = np.quantile(features, [0.01, 0.99], axis=0)
    design = _head_design(features, low, high, knots)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        scores = design @ weights
        probability = expit(scores)
        penalty = ridge * np.sum(weights[1:] ** 2) / 2.0
        loss = np.mean(np.logaddexp(0.0, scores) - labels * scores) + penalty
        gradient = design.T @ (probability - labels) / len(labels)
        gradient[1:] += ridge * weights[1:]
        return float(loss), gradient

    result = minimize(
        objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"logistic head fit failed: {result.message}")
    return LogisticIQHead(result.x, low, high, knots)


def fit_logistic_head(features: np.ndarray, labels: np.ndarray, *, knots: int = 1) -> LogisticIQHead:
    return _fit_logistic_head(features, labels, knots=knots)


def calibrate_measurement_probabilities(
    soft: np.ndarray, hard: np.ndarray, measurement_qubits: np.ndarray,
    train_shots: np.ndarray, *, knots: int, max_examples_per_qubit: int = 100_000,
) -> tuple[np.ndarray, int]:
    """Fit one I/Q-to-hardware-bit head per qubit and evaluate every measurement."""
    probability = np.empty(hard.shape, dtype=np.float32)
    parameter_count = 0
    for qubit in np.unique(measurement_qubits):
        columns = np.flatnonzero(measurement_qubits == qubit)
        values = soft[np.ix_(train_shots, columns)].reshape(-1)
        targets = hard[np.ix_(train_shots, columns)].reshape(-1).astype(float)
        if len(values) > max_examples_per_qubit:
            selected = np.linspace(0, len(values) - 1, max_examples_per_qubit, dtype=int)
            values, targets = values[selected], targets[selected]
        features = np.c_[values.real, values.imag]
        head = _fit_logistic_head(features, targets, knots=knots)
        all_values = soft[:, columns].reshape(-1)
        probability[:, columns] = head.predict(np.c_[all_values.real, all_values.imag]).reshape(len(soft), -1)
        parameter_count += len(head.weights)
    return probability, parameter_count


def fit_iq_heads(features: np.ndarray, labels: np.ndarray) -> dict[str, LogisticIQHead]:
    """Matched logistic comparison: affine versus a tiny additive spline/KAN head."""
    return {
        "linear_logistic_iq": _fit_logistic_head(features, labels, knots=1),
        "spline_kan_logistic_iq": _fit_logistic_head(features, labels, knots=6),
    }
