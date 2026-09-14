"""Bounded generative inference over binary detector-time schedules."""
from __future__ import annotations

import itertools

import numpy as np


def enumerate_binary_schedules(bins, max_changes):
    if (not isinstance(bins, (int, np.integer)) or bins < 1
            or not isinstance(max_changes, (int, np.integer)) or max_changes < 0):
        raise ValueError("positive bins and nonnegative integer change budget required")
    values = [row for row in itertools.product((0, 1), repeat=bins)
              if sum(a != b for a, b in zip(row[:-1], row[1:])) <= max_changes]
    return np.asarray(values, dtype=np.uint8)


def fit_detector_emissions(groups, *, pseudocount=1.):
    groups = [np.asarray(group, dtype=bool) for group in groups]
    if (len(groups) != 2 or pseudocount <= 0 or any(g.ndim != 2 or not len(g) for g in groups)
            or len({g.shape for g in groups}) != 1):
        raise ValueError("two aligned nonempty detector groups and positive smoothing required")
    return np.asarray([(g.sum(axis=0) + pseudocount) / (len(g) + 2 * pseudocount) for g in groups])


def schedule_scores(detectors, emissions, schedules, detector_bins):
    d = np.asarray(detectors, dtype=bool)
    p = np.asarray(emissions, dtype=float)
    schedules = np.asarray(schedules)
    detector_bins = np.asarray(detector_bins)
    if (d.ndim != 2 or p.shape != (2, d.shape[1]) or schedules.ndim != 2
            or detector_bins.shape != (d.shape[1],) or detector_bins.dtype.kind not in "iu"
            or np.any(detector_bins < 0) or np.any(detector_bins >= schedules.shape[1])
            or np.any((p <= 0) | (p >= 1)) or np.any((schedules < 0) | (schedules > 1))):
        raise ValueError("incompatible detector emissions, schedules, or detector bins")
    detector_index = np.arange(d.shape[1])
    candidate_probability = p[schedules[:, detector_bins], detector_index]
    log_p, log_q = np.log(candidate_probability), np.log1p(-candidate_probability)
    return d.astype(float) @ log_p.T + (~d).astype(float) @ log_q.T


def select_schedule(scores, schedules, *, penalty):
    scores, schedules = np.asarray(scores, dtype=float), np.asarray(schedules)
    if (scores.ndim != 2 or schedules.ndim != 2 or scores.shape[1] != len(schedules)
            or penalty < 0 or not np.isfinite(penalty)):
        raise ValueError("aligned candidate scores and nonnegative finite penalty required")
    changes = np.sum(schedules[:, 1:] != schedules[:, :-1], axis=1)
    return np.argmax(scores - penalty * changes, axis=1)


def filter_schedule_scores(scores, schedules, *, penalty, retention):
    scores, schedules = np.asarray(scores, dtype=float), np.asarray(schedules)
    if (scores.ndim != 3 or schedules.ndim != 2 or scores.shape[2] != len(schedules)
            or not 0 <= retention <= 1 or penalty < 0
            or not np.isfinite(retention) or not np.isfinite(penalty)):
        raise ValueError("need stream/time/candidate scores and valid filter parameters")
    changes = np.sum(schedules[:, 1:] != schedules[:, :-1], axis=1)
    state = np.zeros((scores.shape[0], scores.shape[2]), dtype=float)
    choices, history = np.empty(scores.shape[:2], dtype=np.int32), np.empty_like(scores)
    for step in range(scores.shape[1]):
        state = retention * state + scores[:, step]
        history[:, step] = state
        choices[:, step] = np.argmax(state - penalty * changes, axis=1)
    return choices, history


def filter_switching_schedule_scores(scores, schedules, *, penalty, retention,
                                     reset_margin=np.inf):
    """Filter schedule evidence, optionally resetting on decisive local disagreement.

    A reset is raised when the best single-record penalized hypothesis differs
    from the persistent choice and exceeds it by ``reset_margin`` log-score units.
    The current record is then the first record in the new state.  Infinite margin
    is the no-reset exponentially weighted filter.
    """
    scores, schedules = np.asarray(scores, dtype=float), np.asarray(schedules)
    if (scores.ndim != 3 or schedules.ndim != 2 or scores.shape[2] != len(schedules)
            or not 0 <= retention <= 1 or penalty < 0 or reset_margin < 0
            or not np.isfinite(retention) or not np.isfinite(penalty)
            or not (np.isfinite(reset_margin) or np.isposinf(reset_margin))):
        raise ValueError("need stream/time/candidate scores and valid switching parameters")
    changes = np.sum(schedules[:, 1:] != schedules[:, :-1], axis=1)
    cost = penalty * changes
    state = np.zeros((scores.shape[0], scores.shape[2]), dtype=float)
    choices = np.empty(scores.shape[:2], dtype=np.int32)
    resets = np.zeros(scores.shape[:2], dtype=bool)
    history = np.empty_like(scores)
    stream_index = np.arange(scores.shape[0])
    for step in range(scores.shape[1]):
        local_value = scores[:, step] - cost
        local_choice = np.argmax(local_value, axis=1)
        if step:
            persistent_choice = np.argmax(state - cost, axis=1)
            advantage = (local_value[stream_index, local_choice]
                         - local_value[stream_index, persistent_choice])
            reset = (local_choice != persistent_choice) & (advantage >= reset_margin)
            state[reset] = 0
            resets[:, step] = reset
        state = retention * state + scores[:, step]
        history[:, step] = state
        choices[:, step] = np.argmax(state - cost, axis=1)
    return choices, history, resets
