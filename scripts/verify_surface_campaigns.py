#!/usr/bin/env python3
"""Recompute archived follow-up summaries and audit disjoint data provenance.

This validates recorded evidence; it does not retroactively certify unrecorded
runtime dependency versions or regenerate tens of millions of simulator shots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.run_surface_frontier_challenge import seed_value
from scripts.run_surface_rate_adaptation import CONDITIONS, summarize as summarize_rate
from scripts.run_surface_teacher_distillation import summarize as summarize_distillation
from scripts.run_surface_time_templates import summarize as summarize_time
from scripts.summarize_surface_frontier_challenge import summarize as summarize_parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_counts(methods, streams=256, horizon=512):
    for method in methods.values():
        counts = method["errors_per_stream"]
        require(len(counts) == streams and all(type(n) is int and 0 <= n <= horizon for n in counts),
                "invalid stream error counts")
        require(sum(counts) / (streams * horizon) == method["ler"], "LER/count mismatch")


def check_seeds(manifest, root, replicate, distance, keys):
    expected = {json.dumps(key, separators=(",", ":")): seed_value(root, replicate, distance, *key)
                for key in keys}
    require(manifest == expected, "seed namespace/derivation mismatch")


def check_source(path, digest):
    require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, "source artifact hash mismatch")


def verify(root=Path("results")):
    parent_paths = sorted((root / "surface_frontier_challenge").glob("r*_d*.json"))
    parent = [json.loads(p.read_text()) for p in parent_paths]
    parent_summary = summarize_parent(parent)
    require(parent_summary["gates"]["complete_protocol"], "incomplete parent campaign")
    archived_summary = json.loads((root / "surface_frontier_challenge_summary.json").read_text())
    require(all(archived_summary[k] == v for k, v in parent_summary.items()), "parent summary mismatch")
    seeds = [s for row in parent for s in row["seeds"].values()]
    hashes = [h for row in parent for h in row["data_hashes"].values()]
    test_records = parent_summary["test_records"]
    frozen_commits = {}
    for filename, root_seed, distance, short_commit, summarize in (
        ("surface_rate_adaptation.json", 2026090804, 5, "fefa64f", summarize_rate),
        ("surface_teacher_distillation.json", 2026090805, 3, "4b2613c", summarize_distillation),
    ):
        artifact = json.loads((root / filename).read_text())
        commit = subprocess.check_output(["git", "rev-parse", short_commit], text=True).strip()
        require(artifact["code_commit"] == commit, "wrong frozen follow-up commit")
        frozen_commits[filename] = commit
        require(artifact["root_seed"] == root_seed and artifact["streams"] == 256
                and artifact["horizon"] == 512, "wrong follow-up configuration")
        rows = artifact["records"]
        require(len(rows) == 10 and {r["replicate"] for r in rows} == set(range(10)),
                "missing or repeated fitted model")
        require(summarize(rows) == artifact["summary"], "follow-up summary does not reproduce")
        for row in rows:
            check_source(row["source_artifact"], row["source_sha256"])
            source = json.loads(Path(row["source_artifact"]).read_text())
            require((source["replicate"], source["distance"]) == (row["replicate"], distance),
                    "wrong source fitted model")
            if distance == 5:
                roles = [("rate_selection", "nominal")] + [("rate_test", c) for c in CONDITIONS]
                require(set(row["conditions"]) == set(CONDITIONS), "missing rate-test condition")
                require(set(row["data_hashes"]) == {f"{role}_{c}" for role, c in roles}, "wrong rate data roles")
                for condition in row["conditions"].values():
                    check_counts(condition["methods"])
                test_records += len(CONDITIONS) * 256 * 512
            else:
                roles = [(role, "nominal") for role in ("distill_action", "distill_selection", "distill_test")]
                require(set(row["data_hashes"]) == {"action", "distill_selection", "distill_test"},
                        "wrong distillation data roles")
                check_counts(row["methods"])
                test_records += 256 * 512
            keys = [key for role, condition in roles for key in (
                (role, condition, "modes"), (role, condition, "stim", 0), (role, condition, "stim", 1))]
            check_seeds(row["seeds"], root_seed, row["replicate"], distance, keys)
            seeds.extend(row["seeds"].values())
            hashes.extend(row["data_hashes"].values())
    exact = json.loads((root / "surface_exact_teacher.json").read_text())
    check_source(exact["reference_artifact"], exact["reference_sha256"])
    keys = [(role, m) for role in ("teacher_moments", "moment_circuit", "moment_dem") for m in range(2)]
    keys += [key for c in ("nominal", "iid") for key in (
        ("teacher_test", c, "modes"), ("teacher_test", c, "stim", 0), ("teacher_test", c, "stim", 1))]
    check_seeds(exact["seeds"], 2026090803, 0, 3, keys)
    require(exact["seed"] == 2026090803 and exact["stim_version"] == "1.16.0", "wrong exact-teacher config")
    require(set(exact["conditions"]) == {"nominal", "iid"}, "missing exact-teacher condition")
    for condition in exact["conditions"].values():
        require(condition["records"] == 512 * 512, "wrong exact-teacher shot count")
        check_counts(condition["methods"], streams=512)
        test_records += condition["records"]
        hashes.append(condition["data_hash"])
    seeds.extend(exact["seeds"].values())
    time_paths = sorted((root / "surface_time_templates").glob("r*.json"))
    require(len(time_paths) == 10, "missing time-template full refits")
    time_rows = [json.loads(p.read_text()) for p in time_paths]
    time_summary = summarize_time(time_rows)
    time_archive = json.loads((root / "surface_time_templates/summary.json").read_text())
    require(all(time_archive[k] == v for k, v in time_summary.items()), "time-template summary mismatch")
    require(time_summary["gates"]["complete"], "incomplete time-template protocol")
    for row in time_rows:
        seeds.extend(row["seeds"].values())
        hashes.extend(row["data_hashes"].values())
    test_records += time_summary["test_records"]
    native = json.loads((root / "surface_time_template_latency.json").read_text())
    require(len(native["records"]) == 10 and {r["replicate"] for r in native["records"]} == set(range(10)),
            "missing native timing fit")
    for path, digest in native["source_sha256"].items():
        source = subprocess.check_output(["git", "show", f"{native['code_commit']}:{path}"])
        require(hashlib.sha256(source).hexdigest() == digest, "uncommitted native timing source")
    choice_records = 0
    for row in native["records"]:
        check_source(row["artifact"], row["artifact_sha256"])
        require(row["records_per_repeat"] == 32768 and row["repeats"] == 3, "wrong timing protocol size")
        require(set(row["native_validation"]) == set(time_rows[0]["conditions"]), "missing native replay condition")
        for c in row["native_validation"].values():
            require(c["records"] == 256 * 512 and c["choice_disagreements"] == 0, "native semantic disagreement")
            choice_records += c["records"]
        require(not any(row["timed_output_disagreements"].values()), "native timed output differs")
        for column in zip(*row["method_orders"]):
            require(set(column) == {"static", "native_pipeline", "native_frontend"}, "unbalanced timing order")
        ratio = row["methods"]["native_pipeline"]["pooled"]["p99_us"] / row["methods"]["static"]["pooled"]["p99_us"]
        require(ratio == row["pipeline_to_static_p99_ratio"], "timing ratio mismatch")
    require(native["gates"] == {"complete": True, "all_choices_exact": True, "timed_outputs_exact": True,
                                "service_cost": all(r["pipeline_to_static_p99_ratio"] <= 1.25 for r in native["records"])},
            "native gate mismatch")
    require(len(seeds) == len(set(seeds)), "cross-campaign seed reuse")
    require(len(hashes) == len(set(hashes)), "cross-campaign dataset reuse")
    return {"verified": True, "distinct_recorded_seeds": len(seeds),
            "distinct_recorded_dataset_hashes": len(hashes), "heldout_test_records": test_records,
            "followup_frozen_commits": frozen_commits,
            "time_template_frozen_commit": time_rows[0]["code_commit"],
            "native_time_template_choice_replays": choice_records,
            "native_time_template_gates": native["gates"],
            "limits": "Recorded counts/manifests/summaries and source-artifact bytes verified; not a simulator rerun. "
                      "Earlier rate/distillation follow-ups record source commit but not every runtime dependency version. "
                      "Latency intentionally replays parent data and is not an independent accuracy test."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.results)
    result["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
