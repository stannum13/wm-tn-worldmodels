#!/usr/bin/env python3
"""Measure temporal separator width before attempting a frontier decoder."""

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
    load_qec_record,
    temporal_vertex_separation,
    typed_circuit_noise_model,
)


def run(path: Path, circuit_group: str) -> dict:
    import pymatching

    record = load_qec_record(str(path), circuit_group)
    circuit = record["circuit"]
    matching = pymatching.Matching.from_detector_error_model(typed_circuit_noise_model(
        circuit, measurement_probability=0.03,
        one_qubit_probability=0.003, two_qubit_probability=0.03,
    ))
    audit = temporal_vertex_separation(
        matching, circuit.get_detector_coordinates()
    )
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
        "natural_order": "detector time, remaining Stim coordinates, node index",
        "interpretation": (
            "constructive vertex-separation upper bound for this order; "
            "not an optimized or globally minimal pathwidth"
        ),
        "frontier": audit,
        "prototype_gate": {
            "maximum_active_separator": 12,
            "passes": audit["maximum_active_separator"] <= 12,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--circuit-group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.data, args.circuit_group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
