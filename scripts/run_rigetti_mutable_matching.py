#!/usr/bin/env python3
"""Validate fixed-topology per-shot reweighting against graph reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

MUTABLE_BACKEND_URL = "https://github.com/Allenator/PyMatching"
MUTABLE_BACKEND_COMMIT = "435dc7ec85c10314c09f069a3d924d3a3dee8251"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    build_soft_reweighted_matching,
    fit_spitz_pairwise_matching,
    fit_measurement_iq_heads,
    load_qec_record,
    measurement_error_signatures,
    pack_affine_iq_heads,
    prepare_soft_reweighting,
    predict_measurement_probabilities,
    soft_reweight_matrix,
    typed_circuit_noise_model,
)


def _quantiles_ns(values: np.ndarray) -> dict[str, float]:
    return {
        "p50_ns": float(np.quantile(values, 0.50)),
        "p95_ns": float(np.quantile(values, 0.95)),
        "p99_ns": float(np.quantile(values, 0.99)),
    }


def run(
    path: Path, circuit_group: str, train_fraction: float,
    equivalence_shots: int, latency_shots: int, batch_repeats: int,
) -> dict:
    import pymatching

    signature = inspect.signature(pymatching.Matching.decode)
    if "edge_reweights" not in signature.parameters:
        raise RuntimeError(
            "This experiment requires an explicitly supported edge_reweights API; "
            "upstream PyMatching 2.4's catch-all **kwargs is not sufficient."
        )

    record = load_qec_record(str(path), circuit_group)
    labels = record["observables"][:, 0].astype(np.uint8)
    boundary = int(train_fraction * len(labels))
    train = np.arange(boundary)
    test = np.arange(boundary, len(labels))
    heads = fit_measurement_iq_heads(
        record["soft_measurements"], record["hard_measurements"],
        record["measurement_qubits"], train, knots=1, balanced=True,
    )
    started = perf_counter_ns()
    generic_probabilities = predict_measurement_probabilities(
        record["soft_measurements"], record["measurement_qubits"], heads
    )
    generic_iq_batch_ns = perf_counter_ns() - started
    packed_head = pack_affine_iq_heads(record["measurement_qubits"], heads)
    started = perf_counter_ns()
    probabilities = packed_head.predict(record["soft_measurements"])
    packed_iq_batch_ns = perf_counter_ns() - started
    maximum_iq_probability_difference = float(np.max(np.abs(
        probabilities - generic_probabilities
    )))
    calibration_parameters = sum(len(head.weights) for head in heads.values())
    measurement_error = np.where(
        record["hard_measurements"], 1.0 - probabilities, probabilities
    )
    template = pymatching.Matching.from_detector_error_model(typed_circuit_noise_model(
        record["circuit"], measurement_probability=0.03,
        one_qubit_probability=0.003, two_qubit_probability=0.03,
    ))
    pairwise, graph_diagnostics = fit_spitz_pairwise_matching(
        template, record["detectors"][train], floor_probability=1e-4
    )
    plan, soft_diagnostics = prepare_soft_reweighting(
        pairwise, measurement_error_signatures(record["circuit"]),
        np.mean(measurement_error[train], axis=0),
    )
    # A disconnected, always-zero dummy detector seeds PyMatching's weight
    # normalisation ceiling. It cannot affect a logical prediction, and prevents
    # high-confidence shot weights from triggering graph regeneration in the fork.
    dummy_node = pairwise.num_nodes
    maximum_weight = float(np.log((1.0 - 1e-5) / 1e-5))
    pairwise.add_boundary_edge(
        dummy_node, fault_ids=set(), weight=maximum_weight,
        error_probability=1e-5,
    )
    base_edge_weights = {
        (int(first), None if second is None else int(second)):
            float(attributes["weight"])
        for first, second, attributes in pairwise.edges()
    }
    if len(base_edge_weights) != pairwise.num_edges:
        raise RuntimeError("mutable adapter requires endpoint-unique matching edges")
    dummy_rows = [
        (first, second, attributes)
        for first, second, attributes in pairwise.edges()
        if first == dummy_node or second == dummy_node
    ]
    if len(dummy_rows) != 1 or dummy_rows[0][1] is not None or dummy_rows[0][2]["fault_ids"]:
        raise RuntimeError("normalisation dummy must be one empty-fault boundary edge")
    started = perf_counter_ns()
    update_matrix = soft_reweight_matrix(plan, measurement_error[test])
    update_total_ns = perf_counter_ns() - started
    updates = [update_matrix[position] for position in range(len(test))]

    check_count = min(equivalence_shots, len(test))
    reference = np.empty(check_count, dtype=np.uint8)
    reference_call_ns = np.empty(check_count)
    for position, row in enumerate(test[:check_count]):
        started = perf_counter_ns()
        reference[position] = build_soft_reweighted_matching(
            plan, measurement_error[row]
        ).decode(record["detectors"][row].astype(np.uint8))[0]
        reference_call_ns[position] = perf_counter_ns() - started

    detectors = np.pad(
        record["detectors"][test].astype(np.uint8), ((0, 0), (0, 1))
    )
    if np.any(detectors[:, dummy_node]):
        raise RuntimeError("normalisation dummy syndrome must remain zero")
    base_before = np.asarray(pairwise.decode_batch(detectors[:check_count]))[:, 0]
    batch_call_ns = np.empty(batch_repeats)
    mutable = None
    repeated_disagreements = 0
    for repeat in range(batch_repeats):
        started = perf_counter_ns()
        prediction = np.asarray(pairwise.decode_batch(
            detectors, edge_reweights=updates, reweight_stride=1
        ))[:, 0]
        batch_call_ns[repeat] = perf_counter_ns() - started
        if mutable is None:
            mutable = prediction
        else:
            repeated_disagreements += int(np.sum(mutable != prediction))
    assert mutable is not None

    one_count = min(latency_shots, len(test))
    mutable_call_ns = np.empty(one_count)
    iq_single_call_ns = np.empty(one_count)
    for position in range(one_count):
        row = test[position]
        started = perf_counter_ns()
        packed_head.predict(record["soft_measurements"][row : row + 1])
        iq_single_call_ns[position] = perf_counter_ns() - started
        started = perf_counter_ns()
        pairwise.decode(detectors[position], edge_reweights=updates[position])
        mutable_call_ns[position] = perf_counter_ns() - started
    base_after = np.asarray(pairwise.decode_batch(detectors[:check_count]))[:, 0]

    edge_weights_after = {
        (int(first), None if second is None else int(second)):
            float(attributes["weight"])
        for first, second, attributes in pairwise.edges()
    }
    base_max_weight = max(base_edge_weights.values())
    update_maxima = np.asarray([np.max(array[:, 2]) for array in updates])
    if np.any(update_matrix[:, :, 2] < 0) or np.any(update_maxima > maximum_weight):
        raise RuntimeError("dynamic weights escaped the nonnegative clipped seed range")
    commit = os.environ.get("PTWM_CODE_COMMIT")
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pymatching_module": str(Path(pymatching.__file__).resolve()),
            "pymatching_fork_url": MUTABLE_BACKEND_URL,
            "pymatching_fork_commit": MUTABLE_BACKEND_COMMIT,
            "pymatching_install": (
                f"python -m pip install git+{MUTABLE_BACKEND_URL}.git@"
                f"{MUTABLE_BACKEND_COMMIT}"
            ),
        },
        "data": {
            "source": "https://zenodo.org/records/15364358",
            "file": path.name,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "circuit_group": circuit_group,
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "design": {
            "backend": "experimental fixed-topology edge_reweights API",
            "reference": "fresh PyMatching graph reconstruction per shot",
            "equivalence_gate": "zero logical-prediction disagreements",
            "latency_gate": "batch amortized <100 us/shot and >=20x faster than reconstruction",
        },
        "calibration_parameters": calibration_parameters,
        "graph": graph_diagnostics,
        "soft_reweighting": soft_diagnostics,
        "updated_edges_per_shot": int(updates[0].shape[0]),
        "normalisation_dummy_detector": int(dummy_node),
        "base_max_weight": base_max_weight,
        "shots_exceeding_base_max_weight": int(np.sum(update_maxima > base_max_weight)),
        "weights_not_restored_after_decode": int(sum(
            edge_weights_after[key] != weight
            for key, weight in base_edge_weights.items()
        )),
        "equivalence": {
            "shots": check_count,
            "prediction_disagreements": int(np.sum(reference != mutable[:check_count])),
            "prediction_disagreements_across_batch_repeats": repeated_disagreements,
            "base_prediction_disagreements_after_restore": int(np.sum(
                base_before != base_after
            )),
        },
        "logical_error": float(np.mean(mutable != labels[test])),
        "latency": {
            "generic_iq_inference_batch_total_ns": float(generic_iq_batch_ns),
            "packed_iq_inference_batch_total_ns": float(packed_iq_batch_ns),
            "packed_iq_inference_batch_amortized_ns_per_shot": float(
                packed_iq_batch_ns / len(labels)
            ),
            "packed_iq_inference_single_record": _quantiles_ns(iq_single_call_ns),
            "weight_matrix_total_ns": float(update_total_ns),
            "weight_matrix_amortized_ns_per_shot": float(update_total_ns / len(test)),
            "reference_rebuild_and_decode": _quantiles_ns(reference_call_ns),
            "mutable_single_decode": _quantiles_ns(mutable_call_ns),
            "mutable_batch_repeats": batch_repeats,
            "mutable_batch_total": _quantiles_ns(batch_call_ns),
            "mutable_batch_amortized_ns_per_shot": float(
                np.median(batch_call_ns) / len(test)
            ),
            "batch_speedup_vs_reference_p50": float(
                np.quantile(reference_call_ns, 0.50)
                / (np.median(batch_call_ns) / len(test))
            ),
        },
        "maximum_packed_vs_generic_iq_probability_difference": maximum_iq_probability_difference,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--equivalence-shots", type=int, default=1000)
    parser.add_argument("--latency-shots", type=int, default=2000)
    parser.add_argument("--batch-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        args.data, args.circuit_group, args.train_fraction,
        args.equivalence_shots, args.latency_shots, args.batch_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
