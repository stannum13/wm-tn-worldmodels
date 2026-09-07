#!/usr/bin/env python3
"""Bounded topology-only search for a smaller frontier-decoder schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.rigetti import (  # noqa: E402
    beam_vertex_separation_order,
    graph_width_lower_bounds,
    load_qec_record,
    temporal_vertex_separation,
    typed_circuit_noise_model,
)


def run(path: Path, circuit_group: str, beam_widths: list[int]) -> dict:
    import pymatching

    record = load_qec_record(str(path), circuit_group)
    circuit = record["circuit"]
    matching = pymatching.Matching.from_detector_error_model(typed_circuit_noise_model(
        circuit, measurement_probability=0.03,
        one_qubit_probability=0.003, two_qubit_probability=0.03,
    ))
    coordinates = circuit.get_detector_coordinates()
    natural = temporal_vertex_separation(matching, coordinates)
    searches = []
    for width in beam_widths:
        result = beam_vertex_separation_order(
            matching, natural["order"], beam_width=width
        )
        checked = temporal_vertex_separation(
            matching, coordinates, order=result["order"]
        )
        result["verified_maximum_active_separator"] = checked[
            "maximum_active_separator"
        ]
        result["verified_maximum_bag_width"] = max(
            checked["active_separator_trace"][:-1], default=0
        ) + 1
        searches.append(result)
        if result["verified_maximum_bag_width"] <= 6:
            break
    best = min(searches, key=lambda row: (
        row["verified_maximum_bag_width"], row["beam_width"]
    ))
    commit = os.environ.get("PTWM_CODE_COMMIT")
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "data": {
            "source": "https://zenodo.org/records/15364358",
            "file": path.name,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "circuit_group": circuit_group,
        },
        "graph": {"nodes": matching.num_nodes, "edges": matching.num_edges},
        "treewidth_lower_bounds": graph_width_lower_bounds(matching),
        "method": (
            "deterministic topology-only beam search; no labels, syndromes, or "
            "edge weights; heuristic upper bounds, not optimality certificates"
        ),
        "natural": {
            "maximum_active_separator": natural["maximum_active_separator"],
            "maximum_bag_width": max(natural["active_separator_trace"][:-1]) + 1,
        },
        "searches": searches,
        "best": best,
        "go_condition": {
            "maximum_bag_width": 6,
            "passes": best["verified_maximum_bag_width"] <= 6,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group, args.beam_widths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
