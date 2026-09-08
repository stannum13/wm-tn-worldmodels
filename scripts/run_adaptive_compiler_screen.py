#!/usr/bin/env python3
"""Run the first graph-overlay compiler mechanism screen.

This screen covers observation-conditioned REWEIGHT only. ACTIVATE_MODE and structural INSERT_FACTOR/FORK and
localized-solve arms require separate generators so the benchmark does not bake the
answer into one noise model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptwm.adaptive_compiler import (  # noqa: E402
    binary_auroc,
    causal_mode_filter,
    compiled_ema_probabilities,
    decode_repetition,
    entropy,
    generate_switching_repetition,
    memoryless_mode_filter,
    mode_error_probabilities,
    select_ema_alpha,
    stationary_distribution,
)


def loss_interval(candidate: np.ndarray, reference: np.ndarray, labels: np.ndarray) -> list[float]:
    delta = np.mean(
        (candidate != labels).astype(float) - (reference != labels).astype(float), axis=1
    )
    half = stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta)
    return [float(np.mean(delta) - half), float(np.mean(delta) + half)]


def score(prediction: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(prediction != labels))


def method_row(prediction: np.ndarray, labels: np.ndarray, static: np.ndarray) -> dict:
    return {
        "logical_error": score(prediction, labels),
        "paired_episode_95pct_interval_vs_static": loss_interval(prediction, static, labels),
    }


def routing_rows(
    *, test_score: np.ndarray, cheap: np.ndarray, slow: np.ndarray,
    labels: np.ndarray, fractions: tuple[float, ...],
) -> list[dict]:
    rows = []
    cheap_error = cheap != labels
    available = score(cheap, labels) - score(slow, labels)
    helpful = (cheap != labels) & (slow == labels)
    for fraction in fractions:
        count = max(1, int(round(fraction * test_score.size)))
        # Fixed-budget retrospective ranking. Stable flat-index tie breaking keeps the
        # budget exact but is not an online threshold/deployment result.
        order = np.lexsort((np.arange(test_score.size), -test_score.ravel()))
        selected = np.zeros(test_score.size, dtype=bool)
        selected[order[:count]] = True
        selected = selected.reshape(test_score.shape)
        routed = np.where(selected, slow, cheap)
        gain = score(cheap, labels) - score(routed, labels)
        rows.append({
            "target_fraction": fraction,
            "actual_fraction": float(np.mean(selected)),
            "logical_error": score(routed, labels),
            "net_slow_gain_recovery": float(gain / available) if available > 0 else None,
            "cheap_failure_capture": (
                float(np.sum(selected & cheap_error) / np.sum(cheap_error))
                if np.any(cheap_error) else None
            ),
            "helpful_event_capture": float(np.sum(selected & helpful) / np.sum(helpful)) if np.any(helpful) else None,
        })
    return rows


def run_arm(
    scope: str, sigma: float, episodes: int, horizon: int, distance: int, seed: int
) -> dict:
    validation = generate_switching_repetition(
        seed=seed, episodes=max(32, episodes // 4), horizon=horizon,
        distance=distance, emission_sigma=sigma, scope=scope,
    )
    test = generate_switching_repetition(
        seed=seed + 1, episodes=episodes, horizon=horizon,
        distance=distance, emission_sigma=sigma, scope=scope,
    )
    table = mode_error_probabilities(test.observations.shape[-1])
    stationary_probability = table @ stationary_distribution()
    static_probability = np.broadcast_to(stationary_probability, test.observations.shape)
    oracle_probability = table[np.arange(test.observations.shape[-1])[None, None, :], test.modes]
    memoryless_probability, memoryless_belief = memoryless_mode_filter(
        test.observations, emission_sigma=sigma
    )
    local_probability, local_belief = causal_mode_filter(
        test.observations, emission_sigma=sigma, shared=False
    )
    shared_probability, shared_belief = causal_mode_filter(
        test.observations, emission_sigma=sigma, shared=True
    )
    teacher_probability = local_probability if scope == "local" else shared_probability
    teacher_belief = local_belief if scope == "local" else shared_belief
    alpha, validation_alpha_scores = select_ema_alpha(validation)
    compiled_probability = compiled_ema_probabilities(test.observations, alpha)
    probabilities = {
        "static": static_probability,
        "oracle": oracle_probability,
        "memoryless": memoryless_probability,
        "local_hmm": local_probability,
        "shared_hmm": shared_probability,
        "compiled_ema": compiled_probability,
    }
    predictions = {
        name: decode_repetition(test.syndromes, probability)
        for name, probability in probabilities.items()
    }
    static = predictions["static"]
    methods = {
        name: method_row(prediction, test.labels, static)
        for name, prediction in predictions.items()
    }
    for name, prediction in predictions.items():
        methods[name]["paired_episode_95pct_interval_vs_memoryless"] = loss_interval(
            prediction, predictions["memoryless"], test.labels
        )
    static_error = methods["static"]["logical_error"]
    teacher_error = methods["local_hmm" if scope == "local" else "shared_hmm"]["logical_error"]
    compiled_error = methods["compiled_ema"]["logical_error"]
    oracle_error = methods["oracle"]["logical_error"]
    methods["compiled_ema"]["teacher_gain_recovery"] = (
        float((static_error - compiled_error) / (static_error - teacher_error))
        if static_error > teacher_error else None
    )
    methods["local_hmm"]["oracle_gap_recovery"] = (
        float((static_error - methods["local_hmm"]["logical_error"]) / (static_error - oracle_error))
        if static_error > oracle_error else None
    )
    methods["shared_hmm"]["oracle_gap_recovery"] = (
        float((static_error - methods["shared_hmm"]["logical_error"]) / (static_error - oracle_error))
        if static_error > oracle_error else None
    )

    ambiguity_test = entropy(memoryless_belief)
    cheap = predictions["memoryless"]
    slow = predictions["local_hmm" if scope == "local" else "shared_hmm"]
    helpful = (cheap != test.labels) & (slow == test.labels)
    test_innovation = np.max(np.abs(np.diff(
        test.observations, axis=1, prepend=np.zeros_like(test.observations[:, :1])
    )), axis=-1)
    test_density = np.mean(test.syndromes, axis=-1)
    routing_scores = {
        "mode_entropy": ambiguity_test,
        "largest_innovation": test_innovation,
        "syndrome_density": test_density,
    }
    router = {}
    for name, test_score in routing_scores.items():
        router[name] = {
            "auroc_for_helpful_event": binary_auroc(test_score, helpful),
            "budgets": routing_rows(
                test_score=test_score, cheap=cheap, slow=slow, labels=test.labels,
                fractions=(0.01, 0.05, 0.10, 0.20),
            ),
        }
    rng = np.random.default_rng(seed + 2)
    random_test = rng.random(test.labels.shape)
    router["random"] = {
        "auroc_for_helpful_event": binary_auroc(random_test, helpful),
        "budgets": routing_rows(
            test_score=random_test, cheap=cheap, slow=slow, labels=test.labels,
            fractions=(0.01, 0.05, 0.10, 0.20),
        ),
    }
    router["helpful_event_count"] = int(np.sum(helpful))
    return {
        "scope": scope, "emission_sigma": sigma, "episodes": episodes,
        "distance": distance,
        "horizon": horizon, "selected_ema_alpha": alpha,
        "validation_ema_logical_error": {str(k): v for k, v in validation_alpha_scores.items()},
        "methods": methods, "router": router,
        "teacher_mode_entropy_mean": float(np.mean(entropy(teacher_belief))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.25])
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = []
    for sigma in args.sigmas:
        for scope in ("local", "shared"):
            arms.append(run_arm(
                scope, sigma, args.episodes, args.horizon, args.distance,
                args.seed + len(arms) * 10,
            ))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "design": {
            "status": "directional synthetic mechanism screen",
            "trusted_layer": "exact MAP between two repetition-code-consistent error chains",
            "covered_operations": ["REWEIGHT"],
            "not_covered": ["ACTIVATE_MODE", "INSERT_FACTOR", "REWIRE", "FORK", "GROW_REGION", "LOCAL_SOLVE"],
            "filter_status": "privileged known-parameter Bayes filter; not fitted teacher",
            "selection": "EMA alpha selected on independent validation episodes by downstream LER",
            "routing": "retrospective exact-budget ranking; not an online threshold result",
        },
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
