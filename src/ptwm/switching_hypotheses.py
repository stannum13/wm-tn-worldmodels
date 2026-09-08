"""A bounded filter over nuisance switching rates, not Pauli error configurations."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def switching_hypothesis_filter(log_likelihood_ratio, *, rates=(.005, .02, .1, .5),
                                model_switch=.002):
    """Forward-filter the joint (switching-rate hypothesis, physical regime) state.

    At each step the rate hypothesis either persists or refreshes uniformly, then
    the binary regime transitions at the newly selected rate. The same current-shot
    evidence is shared by all hypotheses. Only 2*K probabilities persist per stream.
    """
    llr = np.asarray(log_likelihood_ratio, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if llr.ndim != 2 or rates.ndim != 1 or not len(rates):
        raise ValueError("require (stream,time) evidence and at least one rate")
    if np.any((rates < 0) | (rates > .5)) or not 0 <= model_switch <= 1:
        raise ValueError("invalid transition probability")
    if not np.all(np.isfinite(llr)):
        raise ValueError("evidence must be finite")
    count = len(rates)
    joint = np.full((llr.shape[0], count, 2), 1 / (2 * count))
    posterior = np.empty_like(llr)
    effective_rate = np.empty_like(llr)
    for step in range(llr.shape[1]):
        mixed = (1 - model_switch) * joint + model_switch * joint.sum(axis=1, keepdims=True) / count
        forecast = np.empty_like(joint)
        forecast[:, :, 0] = mixed[:, :, 0] * (1 - rates) + mixed[:, :, 1] * rates
        forecast[:, :, 1] = mixed[:, :, 1] * (1 - rates) + mixed[:, :, 0] * rates
        # Rescaling both emissions by the same per-record factor cancels exactly.
        evidence = np.clip(expit(llr[:, step]), 1e-15, 1 - 1e-15)
        forecast[:, :, 0] *= (1 - evidence[:, None])
        forecast[:, :, 1] *= evidence[:, None]
        joint = forecast / forecast.sum(axis=(1, 2), keepdims=True)
        posterior[:, step] = joint[:, :, 1].sum(axis=1)
        effective_rate[:, step] = joint.sum(axis=2) @ rates
    return posterior, effective_rate
