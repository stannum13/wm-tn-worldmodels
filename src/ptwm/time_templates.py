"""Bounded inference and action tables for fixed time-resolved decoder graphs."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


def splice_noise(first, second, fraction):
    if not 0 < fraction < 1:
        raise ValueError("cutpoint must lie strictly inside the circuit")
    a, b = list(first.flattened()), list(second.flattened())
    if len(a) != len(b) or first.without_noise().flattened() != second.without_noise().flattened():
        raise ValueError("ideal circuit topologies differ")
    cut = int(sum(op.name == "TICK" for op in a) * fraction)
    ticks, output = 0, type(first)()
    for left, right in zip(a, b):
        if left.name != right.name or left.targets_copy() != right.targets_copy():
            raise ValueError("unaligned circuit operations")
        output.append(left if ticks < cut else right)
        ticks += left.name == "TICK"
    return output


def fit_mode_logits(groups, l2=.01):
    groups = [np.asarray(g) for g in groups]
    if len(groups) < 2 or len({g.shape for g in groups}) != 1 or groups[0].ndim != 2:
        raise ValueError("mode calibration must have equal, aligned group shapes")
    if not len(groups[0]) or l2 < 0 or any(not np.all(np.isfinite(g)) for g in groups):
        raise ValueError("invalid mode calibration")
    x = np.vstack(groups).astype(float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    x = (x - mean) / scale
    labels = np.repeat(np.arange(len(groups)), len(groups[0]))
    width, modes, n = x.shape[1], len(groups), len(x)

    def objective(parameters):
        w = parameters[:width * modes].reshape(width, modes)
        bias = parameters[width * modes:]
        logits = x @ w + bias
        value = np.mean(logsumexp(logits, axis=1) - logits[np.arange(n), labels])
        value += .5 * l2 * np.sum(w * w)
        gradient = softmax(logits, axis=1)
        gradient[np.arange(n), labels] -= 1
        gradient /= n
        return float(value), np.r_[(x.T @ gradient + l2 * w).ravel(), gradient.sum(axis=0)]

    result = minimize(objective, np.zeros((width + 1) * modes), jac=True, method="L-BFGS-B",
                      options={"maxiter": 300})
    if not result.success:
        raise RuntimeError(f"mode fit did not converge: {result.message}")
    return {"mean": mean, "scale": scale, "weights": result.x[:width * modes].reshape(width, modes),
            "bias": result.x[width * modes:], "iterations": int(result.nit), "objective": float(result.fun)}


def mode_logits(features, model):
    return ((features - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"]


def mode_rate_filter(logits, *, rates=(.005, .02, .1, .5, .75), refresh=.002):
    logits, rates = np.asarray(logits, dtype=float), np.asarray(rates, dtype=float)
    if logits.ndim != 3 or logits.shape[-1] < 2 or rates.ndim != 1 or not len(rates):
        raise ValueError("need (streams,time,modes) evidence and a nonempty rate bank")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(rates)):
        raise ValueError("nonfinite evidence or rates")
    modes, k = logits.shape[2], len(rates)
    if np.any((rates < 0) | (rates > (modes - 1) / modes)) or not 0 <= refresh <= 1:
        raise ValueError("invalid switching/refresh probability")
    joint = np.full((logits.shape[0], k, modes), 1 / (k * modes))
    posterior, effective = np.empty_like(logits), np.empty(logits.shape[:2])
    off = rates[None, :, None] / (modes - 1)
    for step in range(logits.shape[1]):
        mixed = (1 - refresh) * joint + refresh * joint.sum(axis=1, keepdims=True) / k
        prior = (1 - rates[None, :, None] - off) * mixed + off * mixed.sum(axis=2, keepdims=True)
        x = logits[:, step]
        evidence = np.exp(np.maximum(x - x.max(axis=1, keepdims=True), -700))
        joint = prior * evidence[:, None, :]
        joint /= joint.sum(axis=(1, 2), keepdims=True)
        posterior[:, step] = joint.sum(axis=1)
        effective[:, step] = joint.sum(axis=2) @ rates
    return posterior, effective


def difficulty_bins(detectors):
    return np.searchsorted([1, 4, 8], detectors.sum(axis=1), side="right")


def fit_risk_tables(errors, bins, pseudocount=100.):
    """Rows are independent risk-calibration shots in each true mode."""
    global_risk = np.stack([np.mean(e, axis=0) for e in errors])
    binned = np.empty((len(errors), errors[0].shape[1], 4))
    for mode, (loss, group) in enumerate(zip(errors, bins)):
        for b in range(4):
            chosen = group == b
            binned[mode, :, b] = (loss[chosen].sum(axis=0) + pseudocount * global_risk[mode]) / (chosen.sum() + pseudocount)
    return {"global": global_risk, "binned": binned}


def approximate_risks(posterior, tables, bins, rule):
    if rule == "binned":
        return np.einsum("nm,man->na", posterior, tables["binned"][:, :, bins])
    return posterior @ tables["global"]


def choose_action(posterior, tables, bins, rule, actions=(0, 1, 2, 3)):
    actions = np.asarray(actions, dtype=int)
    if rule == "mode":
        if np.array_equal(actions, [0, 3]):
            return np.where(posterior[:, 3] + .5 * (posterior[:, 1] + posterior[:, 2]) > .5, 3, 0)
        if not np.array_equal(actions, [0, 1, 2, 3]):
            raise ValueError("mode action rule requires the full or endpoint action set")
        return posterior.argmax(axis=1)
    if rule not in ("global", "binned"):
        raise ValueError("unknown action rule")
    risks = approximate_risks(posterior, tables, bins, rule)
    return actions[risks[:, actions].argmin(axis=1)]
