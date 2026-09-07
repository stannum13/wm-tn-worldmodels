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
class PackedAffineIQHeads:
    """One vectorized hot-path kernel for per-measurement affine I/Q heads."""

    weights: np.ndarray
    feature_min: np.ndarray
    feature_range: np.ndarray

    def predict(self, soft: np.ndarray) -> np.ndarray:
        soft = np.asarray(soft)
        scaled_real = np.clip(
            (soft.real - self.feature_min[:, 0]) / self.feature_range[:, 0], 0.0, 1.0
        )
        scaled_imag = np.clip(
            (soft.imag - self.feature_min[:, 1]) / self.feature_range[:, 1], 0.0, 1.0
        )
        scores = (
            self.weights[:, 0]
            + scaled_real * self.weights[:, 1]
            + scaled_imag * self.weights[:, 2]
        )
        return expit(scores).astype(np.float32)


@dataclass(frozen=True)
class PackedSoftReweighting:
    """Vectorized fixed-topology map from measurement surrogates to edge LLRs."""

    endpoints: np.ndarray
    residual_factor: np.ndarray
    measurement_indices: np.ndarray
    measurement_mask: np.ndarray
    floor_probability: float
    ceiling_probability: float

    def build(self, shot_error_probability: np.ndarray) -> np.ndarray:
        values = np.asarray(shot_error_probability, dtype=float)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("shot_error_probability must be one- or two-dimensional")
        selected = np.clip(
            values[:, self.measurement_indices],
            self.floor_probability,
            self.ceiling_probability,
        )
        factors = np.where(
            self.measurement_mask[None, :, :], 1.0 - 2.0 * selected, 1.0
        )
        alpha = self.residual_factor[None, :] * np.prod(factors, axis=2)
        probability = np.clip(
            (1.0 - alpha) / 2.0,
            self.floor_probability,
            self.ceiling_probability,
        )
        output = np.empty((len(values), len(self.endpoints), 3), dtype=float)
        output[:, :, :2] = self.endpoints
        output[:, :, 2] = np.log((1.0 - probability) / probability)
        return output[0] if single else output


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
    """Reconstruct Stim measurement order from per-qubit HDF5 arrays."""
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


def measurement_error_signatures(circuit: object) -> list[tuple[tuple[int, ...], frozenset[int]]]:
    """Map each measurement-bit flip to detector endpoints and logical fault IDs."""
    circuit = circuit.flattened()
    detector_indices = detector_measurement_indices(circuit)
    observable_indices: dict[int, set[int]] = {}
    measurement_cursor = 0
    for instruction in circuit:
        if instruction.name == "M":
            measurement_cursor += len(instruction.targets_copy())
        elif instruction.name == "OBSERVABLE_INCLUDE":
            observable = int(instruction.gate_args_copy()[0])
            locations = observable_indices.setdefault(observable, set())
            for target in instruction.targets_copy():
                if target.is_measurement_record_target:
                    index = measurement_cursor + target.value
                    if index in locations:
                        locations.remove(index)
                    else:
                        locations.add(index)
    signatures = []
    for measurement in range(circuit.num_measurements):
        detectors = tuple(
            index for index, indices in enumerate(detector_indices)
            if measurement in indices
        )
        observables = frozenset(
            observable for observable, indices in observable_indices.items()
            if measurement in indices
        )
        signatures.append((detectors, observables))
    return signatures


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


def spitz_pair_probability(first: np.ndarray, second: np.ndarray) -> float:
    """Invert two defect moments into an independent pair-event probability.

    This is Eq. (3) of Spitz et al., Adv. Quantum Technol. 1, 1800012
    (2018). A negative radicand is returned as NaN so callers must declare their
    finite-sample regularization instead of receiving a silent clip.
    """
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("defect arrays must have the same shape")
    covariance = np.mean(first & second) - np.mean(first) * np.mean(second)
    denominator = 1.0 - 2.0 * np.mean(first ^ second)
    if denominator <= 0.0:
        return float("nan")
    radicand = 0.25 - covariance / denominator
    if radicand < 0.0:
        return float("nan")
    return float(0.5 - np.sqrt(radicand))


def spitz_boundary_probability(defect: np.ndarray, incident_pair_probabilities: list[float]) -> float:
    """Infer the single-defect boundary-event probability after pair edges."""
    product = float(np.prod([1.0 - 2.0 * probability for probability in incident_pair_probabilities]))
    if product <= 0.0:
        return float("nan")
    return float(0.5 + (np.mean(np.asarray(defect, dtype=bool)) - 0.5) / product)


def fit_spitz_pairwise_matching(
    template_matching: object, calibration_detectors: np.ndarray, *,
    floor_probability: float = 1e-4, ceiling_probability: float = 0.49,
) -> tuple[object, dict[str, int | float]]:
    """Fit Spitz pairwise probabilities on a fixed, circuit-derived graph.

    The graph topology and logical fault IDs come only from ``template_matching``.
    Finite-sample invalid or sub-floor estimates use the declared global floor.
    """
    try:
        import pymatching
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'real' optional dependencies for PyMatching") from exc
    if not 0.0 < floor_probability < ceiling_probability < 0.5:
        raise ValueError("probability bounds must satisfy 0 < floor < ceiling < 0.5")
    detectors = np.asarray(calibration_detectors, dtype=bool)
    if detectors.ndim != 2 or detectors.shape[1] != template_matching.num_detectors:
        raise ValueError("calibration detector width must match the template graph")

    fitted = pymatching.Matching()
    incident: dict[int, list[float]] = {node: [] for node in range(template_matching.num_detectors)}
    boundary_edges = []
    invalid = clipped_low = clipped_high = 0
    pair_edges = 0
    for first, second, attributes in template_matching.edges():
        if second is None:
            boundary_edges.append((first, attributes))
            continue
        raw = spitz_pair_probability(detectors[:, first], detectors[:, second])
        if not np.isfinite(raw):
            invalid += 1
            raw = floor_probability
        if raw < floor_probability:
            clipped_low += 1
        if raw > ceiling_probability:
            clipped_high += 1
        probability = float(np.clip(raw, floor_probability, ceiling_probability))
        weight = float(np.log((1.0 - probability) / probability))
        fitted.add_edge(
            first, second, fault_ids=attributes["fault_ids"], weight=weight,
            error_probability=probability,
        )
        incident[first].append(probability)
        incident[second].append(probability)
        pair_edges += 1

    for node, attributes in boundary_edges:
        raw = spitz_boundary_probability(detectors[:, node], incident[node])
        if not np.isfinite(raw):
            invalid += 1
            raw = floor_probability
        if raw < floor_probability:
            clipped_low += 1
        if raw > ceiling_probability:
            clipped_high += 1
        probability = float(np.clip(raw, floor_probability, ceiling_probability))
        weight = float(np.log((1.0 - probability) / probability))
        fitted.add_boundary_edge(
            node, fault_ids=attributes["fault_ids"], weight=weight,
            error_probability=probability,
        )

    return fitted, {
        "pair_edges": pair_edges,
        "boundary_edges": len(boundary_edges),
        "invalid_estimates": invalid,
        "clipped_low": clipped_low,
        "clipped_high": clipped_high,
        "calibration_shots": len(detectors),
        "floor_probability": floor_probability,
        "ceiling_probability": ceiling_probability,
    }


def _matching_edge_key(
    first: int, second: int | None, fault_ids: set[int] | frozenset[int]
) -> tuple[int, int | None, frozenset[int]]:
    if second is not None and second < first:
        first, second = second, first
    return first, second, frozenset(fault_ids)


def prepare_soft_reweighting(
    base_matching: object,
    signatures: list[tuple[tuple[int, ...], frozenset[int]]],
    average_error_probability: np.ndarray,
    *, floor_probability: float = 1e-5,
    ceiling_probability: float = 0.49,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Factor average measurement errors out of matching-edge probabilities."""
    average = np.asarray(average_error_probability, dtype=float)
    if len(average) != len(signatures):
        raise ValueError("one average error probability is required per measurement")
    plan = []
    locations: dict[tuple[int, int | None, frozenset[int]], int] = {}
    for first, second, attributes in base_matching.edges():
        probability = float(attributes["error_probability"])
        if not 0.0 <= probability < 0.5:
            probability = 1.0 / (1.0 + np.exp(float(attributes["weight"])))
        key = _matching_edge_key(first, second, attributes["fault_ids"])
        locations[key] = len(plan)
        plan.append({
            "first": first,
            "second": second,
            "fault_ids": frozenset(attributes["fault_ids"]),
            "residual_probability": probability,
            "measurements": [],
        })
    unmatched = unsupported = matched = residual_clipped = 0
    for measurement, (detectors, fault_ids) in enumerate(signatures):
        if not detectors:
            continue
        if len(detectors) > 2:
            unsupported += 1
            continue
        first, second = detectors[0], detectors[1] if len(detectors) == 2 else None
        key = _matching_edge_key(first, second, fault_ids)
        if key not in locations:
            unmatched += 1
            continue
        row = plan[locations[key]]
        probability = float(row["residual_probability"])
        measurement_probability = float(np.clip(average[measurement], floor_probability, ceiling_probability))
        residual = (probability - measurement_probability) / (1.0 - 2.0 * measurement_probability)
        if residual < floor_probability or residual > ceiling_probability:
            residual_clipped += 1
        row["residual_probability"] = float(np.clip(residual, floor_probability, ceiling_probability))
        row["measurements"].append(measurement)
        matched += 1
    return plan, {
        "matched_measurements": matched,
        "unmatched_measurements": unmatched,
        "unsupported_measurements": unsupported,
        "residual_clipped": residual_clipped,
    }


def build_soft_reweighted_matching(
    plan: list[dict[str, object]], shot_error_probability: np.ndarray, *,
    floor_probability: float = 1e-5, ceiling_probability: float = 0.49,
) -> object:
    """Instantiate one graph after replacing average measurement errors by shot values."""
    try:
        import pymatching
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'real' optional dependencies for PyMatching") from exc
    shot_error = np.asarray(shot_error_probability, dtype=float)
    matching = pymatching.Matching()
    for row in plan:
        probability = float(row["residual_probability"])
        for measurement in row["measurements"]:
            value = float(np.clip(shot_error[measurement], floor_probability, ceiling_probability))
            probability = probability + value - 2.0 * probability * value
        probability = float(np.clip(probability, floor_probability, ceiling_probability))
        weight = float(np.log((1.0 - probability) / probability))
        if row["second"] is None:
            matching.add_boundary_edge(
                row["first"], fault_ids=row["fault_ids"], weight=weight,
                error_probability=probability,
            )
        else:
            matching.add_edge(
                row["first"], row["second"], fault_ids=row["fault_ids"], weight=weight,
                error_probability=probability,
            )
    return matching


def soft_reweight_array(
    plan: list[dict[str, object]], shot_error_probability: np.ndarray, *,
    floor_probability: float = 1e-5, ceiling_probability: float = 0.49,
) -> np.ndarray:
    """Return mutable-backend rows ``[node1, node2_or_-1, weight]``.

    Only edges with a shot-dependent measurement contribution are emitted.  The
    fixed graph retains all other weights, endpoints, and logical fault IDs.
    """
    shot_error = np.asarray(shot_error_probability, dtype=float)
    rows = []
    for row in plan:
        if not row["measurements"]:
            continue
        probability = float(row["residual_probability"])
        for measurement in row["measurements"]:
            value = float(np.clip(
                shot_error[measurement], floor_probability, ceiling_probability
            ))
            probability = probability + value - 2.0 * probability * value
        probability = float(np.clip(probability, floor_probability, ceiling_probability))
        rows.append((
            float(row["first"]),
            -1.0 if row["second"] is None else float(row["second"]),
            float(np.log((1.0 - probability) / probability)),
        ))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def soft_reweight_matrix(
    plan: list[dict[str, object]], shot_error_probability: np.ndarray, *,
    floor_probability: float = 1e-5, ceiling_probability: float = 0.49,
) -> np.ndarray:
    """Vectorize mutable edge weights for a batch of shots.

    Returns shape ``(shots, updated_edges, 3)``. Endpoints are fixed across the
    first dimension; only the final weight column varies.
    """
    shot_error = np.asarray(shot_error_probability, dtype=float)
    if shot_error.ndim != 2:
        raise ValueError("shot_error_probability must have shape (shots, measurements)")
    dynamic = [row for row in plan if row["measurements"]]
    output = np.empty((len(shot_error), len(dynamic), 3), dtype=float)
    for edge, row in enumerate(dynamic):
        probability = np.full(len(shot_error), float(row["residual_probability"]))
        for measurement in row["measurements"]:
            value = np.clip(
                shot_error[:, measurement], floor_probability, ceiling_probability
            )
            probability = probability + value - 2.0 * probability * value
        probability = np.clip(probability, floor_probability, ceiling_probability)
        output[:, edge, 0] = float(row["first"])
        output[:, edge, 1] = -1.0 if row["second"] is None else float(row["second"])
        output[:, edge, 2] = np.log((1.0 - probability) / probability)
    return output


def pack_soft_reweighting(
    plan: list[dict[str, object]], *, floor_probability: float = 1e-5,
    ceiling_probability: float = 0.49,
) -> PackedSoftReweighting:
    """Compile dynamic plan rows into dense measurement-to-edge indices."""
    dynamic = [row for row in plan if row["measurements"]]
    width = max((len(row["measurements"]) for row in dynamic), default=0)
    indices = np.zeros((len(dynamic), width), dtype=int)
    mask = np.zeros((len(dynamic), width), dtype=bool)
    endpoints = np.empty((len(dynamic), 2), dtype=float)
    residual_factor = np.empty(len(dynamic), dtype=float)
    for edge, row in enumerate(dynamic):
        measurements = np.asarray(row["measurements"], dtype=int)
        indices[edge, :len(measurements)] = measurements
        mask[edge, :len(measurements)] = True
        endpoints[edge] = (
            float(row["first"]),
            -1.0 if row["second"] is None else float(row["second"]),
        )
        residual_factor[edge] = 1.0 - 2.0 * float(row["residual_probability"])
    return PackedSoftReweighting(
        endpoints, residual_factor, indices, mask,
        floor_probability, ceiling_probability,
    )


def calibrated_uncertainty_route(
    calibration_scores: np.ndarray, evaluation_scores: np.ndarray, *, budget: float
) -> tuple[np.ndarray, float]:
    """Route high-uncertainty evaluations using a calibration-only quantile."""
    if not 0.0 <= budget <= 1.0:
        raise ValueError("route budget must be between zero and one")
    calibration = np.asarray(calibration_scores, dtype=float)
    evaluation = np.asarray(evaluation_scores, dtype=float)
    if budget == 0.0:
        return np.zeros(len(evaluation), dtype=bool), float("inf")
    if budget == 1.0:
        return np.ones(len(evaluation), dtype=bool), float("-inf")
    threshold = float(np.quantile(calibration, 1.0 - budget))
    return evaluation >= threshold, threshold


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
    return typed_circuit_noise_model(
        circuit,
        measurement_probability=probability,
        one_qubit_probability=probability,
        two_qubit_probability=probability,
    )


def typed_circuit_noise_model(
    circuit: object, *, measurement_probability: float,
    one_qubit_probability: float, two_qubit_probability: float,
) -> object:
    """Circuit model whose three rates can be calibrated off the hot path."""
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
            if instruction.gate_args_copy():
                raise ValueError("typed noise model expects noiseless measurement instructions")
            # A readout-classification error flips the classical record without
            # changing the post-measurement qubit state. X_ERROR before M is not
            # equivalent when the qubit is measured again without a reset.
            noisy.append(instruction.name, targets, measurement_probability)
            if instruction.name in measure_reset_gates:
                noisy.append("X_ERROR", targets, measurement_probability)
        else:
            noisy.append(instruction)
            if instruction.name in two_qubit_gates:
                noisy.append("DEPOLARIZE2", targets, two_qubit_probability)
            elif instruction.name in single_qubit_gates:
                noisy.append("DEPOLARIZE1", targets, one_qubit_probability)
            elif instruction.name in reset_gates:
                noisy.append("X_ERROR", targets, measurement_probability)
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


def typed_matching_predictions(
    circuit: object, detectors: np.ndarray, *, measurement_probability: float,
    one_qubit_probability: float, two_qubit_probability: float,
) -> tuple[np.ndarray, dict[str, float]]:
    try:
        import pymatching
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'real' optional dependencies for PyMatching") from exc
    model = typed_circuit_noise_model(
        circuit,
        measurement_probability=measurement_probability,
        one_qubit_probability=one_qubit_probability,
        two_qubit_probability=two_qubit_probability,
    )
    matching = pymatching.Matching.from_detector_error_model(model)
    started = perf_counter_ns()
    prediction = matching.decode_batch(np.asarray(detectors, dtype=np.uint8))
    elapsed = perf_counter_ns() - started
    return np.asarray(prediction)[:, 0].astype(float), {
        "vectorized_ns_per_shot": float(elapsed / len(detectors)),
        "detector_error_model_terms": len(model),
    }


def chronological_preparation_split(labels: np.ndarray, train_fraction: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Split early and later HDF5 row ranges within each prepared-state trace."""
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


def chronological_block_error_differences(
    first: np.ndarray, second: np.ndarray, labels: np.ndarray, *, block_size: int
) -> np.ndarray:
    """Paired error differences in raw acquisition-order blocks."""
    first, second, labels = np.asarray(first), np.asarray(second), np.asarray(labels)
    differences = []
    for start in range(0, len(labels) - block_size + 1, block_size):
        block = slice(start, start + block_size)
        first_error = (first[block] >= 0.5) != labels[block]
        second_error = (second[block] >= 0.5) != labels[block]
        differences.append(np.mean(first_error) - np.mean(second_error))
    return np.asarray(differences)


def chronological_block_score_differences(
    first: np.ndarray, second: np.ndarray, labels: np.ndarray, *, block_size: int,
    score: str,
) -> np.ndarray:
    """Paired proper-score differences in raw acquisition-order blocks."""
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
    return np.asarray([
        np.mean(first_values[start : start + block_size] - second_values[start : start + block_size])
        for start in range(0, len(labels) - block_size + 1, block_size)
    ])


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


def fit_measurement_iq_heads(
    soft: np.ndarray, hard: np.ndarray, measurement_qubits: np.ndarray,
    train_shots: np.ndarray, *, knots: int, max_examples_per_qubit: int = 100_000,
    balanced: bool = False,
) -> dict[int, LogisticIQHead]:
    """Fit one I/Q-to-hardware-bit head per qubit."""
    heads = {}
    for qubit in np.unique(measurement_qubits):
        columns = np.flatnonzero(measurement_qubits == qubit)
        values = soft[np.ix_(train_shots, columns)].reshape(-1)
        targets = hard[np.ix_(train_shots, columns)].reshape(-1).astype(float)
        if balanced:
            class_locations = [np.flatnonzero(targets == label) for label in (0.0, 1.0)]
            per_class = min(*(len(locations) for locations in class_locations), max_examples_per_qubit // 2)
            selected = np.concatenate([
                locations[np.linspace(0, len(locations) - 1, per_class, dtype=int)]
                for locations in class_locations
            ])
            values, targets = values[selected], targets[selected]
        elif len(values) > max_examples_per_qubit:
            selected = np.linspace(0, len(values) - 1, max_examples_per_qubit, dtype=int)
            values, targets = values[selected], targets[selected]
        features = np.c_[values.real, values.imag]
        heads[int(qubit)] = _fit_logistic_head(features, targets, knots=knots)
    return heads


def predict_measurement_probabilities(
    soft: np.ndarray, measurement_qubits: np.ndarray,
    heads: dict[int, LogisticIQHead],
) -> np.ndarray:
    """Apply fitted per-qubit I/Q heads while preserving measurement order."""
    probability = np.empty(soft.shape, dtype=np.float32)
    for qubit, head in heads.items():
        columns = np.flatnonzero(measurement_qubits == qubit)
        all_values = soft[:, columns].reshape(-1)
        probability[:, columns] = head.predict(np.c_[all_values.real, all_values.imag]).reshape(len(soft), -1)
    return probability


def pack_affine_iq_heads(
    measurement_qubits: np.ndarray, heads: dict[int, LogisticIQHead]
) -> PackedAffineIQHeads:
    """Pack knots=1 heads into arrays indexed directly by measurement location."""
    qubits = np.asarray(measurement_qubits)
    rows = [heads[int(qubit)] for qubit in qubits]
    if any(head.knots != 1 or len(head.weights) != 3 for head in rows):
        raise ValueError("only three-parameter affine I/Q heads can be packed")
    low = np.stack([head.feature_min for head in rows])
    high = np.stack([head.feature_max for head in rows])
    return PackedAffineIQHeads(
        weights=np.stack([head.weights for head in rows]),
        feature_min=low,
        feature_range=np.maximum(high - low, 1e-9),
    )


def calibrate_measurement_probabilities(
    soft: np.ndarray, hard: np.ndarray, measurement_qubits: np.ndarray,
    train_shots: np.ndarray, *, knots: int, max_examples_per_qubit: int = 100_000,
    balanced: bool = False,
) -> tuple[np.ndarray, int]:
    """Fit one I/Q-to-hardware-bit head per qubit and evaluate every measurement."""
    heads = fit_measurement_iq_heads(
        soft, hard, measurement_qubits, train_shots, knots=knots,
        max_examples_per_qubit=max_examples_per_qubit, balanced=balanced,
    )
    probability = predict_measurement_probabilities(soft, measurement_qubits, heads)
    return probability, sum(len(head.weights) for head in heads.values())


def fit_iq_heads(features: np.ndarray, labels: np.ndarray) -> dict[str, LogisticIQHead]:
    """Matched logistic comparison: affine versus a tiny additive spline/KAN head."""
    return {
        "linear_logistic_iq": _fit_logistic_head(features, labels, knots=1),
        "spline_kan_logistic_iq": _fit_logistic_head(features, labels, knots=6),
    }
