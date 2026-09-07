"""Recover the one-qubit Clifford group metadata omitted from the data release."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import permutations, product

import numpy as np


def _order(element, table):
    value = 0
    for order in range(1, 25):
        value = table[value, element]
        if value == 0:
            return order
    raise ValueError("invalid finite group table")


def _closure(generators, table):
    seen = {0}
    while True:
        expanded = seen | {int(table[a, g]) for a in seen for g in generators}
        if expanded == seen:
            return seen
        seen = expanded


def infer_multiplication_table(rows_by_experiment):
    """Infer all 24x24 products from RB inverse sequences of lengths two and three."""
    inverse_votes = defaultdict(Counter)
    rows = [row for experiment in rows_by_experiment for row in experiment]
    for row in rows:
        sequence = row["cl_ops"] if isinstance(row, dict) else row[0]
        if len(sequence) == 2:
            inverse_votes[sequence[0]][sequence[1]] += 1
    if len(inverse_votes) != 24:
        raise ValueError("all 24 Clifford labels are required")
    inverse = {label: votes.most_common(1)[0][0] for label, votes in inverse_votes.items()}

    known = {}
    for row in rows:
        sequence = row["cl_ops"] if isinstance(row, dict) else row[0]
        if len(sequence) == 3:
            key, value = (sequence[0], sequence[1]), inverse[sequence[2]]
            if key in known and known[key] != value:
                raise ValueError("inconsistent Clifford product constraints")
            known[key] = value
    for a in range(24):
        for b in range(24):
            reverse = (inverse[b], inverse[a])
            if (a, b) not in known and reverse in known:
                known[a, b] = inverse[known[reverse]]

    # Typically one inverse pair remains unseen. Associativity uniquely completes it.
    for a in range(24):
        for b in range(24):
            if (a, b) in known:
                continue
            candidates = []
            for candidate in range(24):
                checks = [
                    known[candidate, c] == known[a, known[b, c]]
                    for c in range(24)
                    if (candidate, c) in known and (b, c) in known and (a, known[b, c]) in known
                ]
                if checks and all(checks):
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise ValueError(f"could not uniquely complete product {(a, b)}")
            known[a, b] = candidates[0]
    table = np.array([[known[a, b] for b in range(24)] for a in range(24)], dtype=int)
    if not all(table[0, a] == a and table[a, 0] == a for a in range(24)):
        raise ValueError("inferred label zero is not the identity")
    return table


def signed_permutation_rotations():
    rotations = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=int)
            for i, j in enumerate(permutation):
                matrix[i, j] = signs[i]
            if round(np.linalg.det(matrix)) == 1:
                rotations.append(matrix)
    identity = next(i for i, matrix in enumerate(rotations) if np.array_equal(matrix, np.eye(3, dtype=int)))
    rotations[0], rotations[identity] = rotations[identity], rotations[0]
    return np.asarray(rotations)


def infer_clifford_rotations(rows_by_experiment):
    """Return a faithful 3D rotation for every released Clifford label.

    The isomorphism is defined up to a global octahedral conjugation, which is a gauge
    freedom for downstream models that use the full rotation matrix.
    """
    label_table = infer_multiplication_table(rows_by_experiment)
    rotations = signed_permutation_rotations()
    lookup = {tuple(matrix.ravel()): i for i, matrix in enumerate(rotations)}
    canonical = np.array([
        [lookup[tuple((rotations[a] @ rotations[b]).ravel())] for b in range(24)]
        for a in range(24)
    ])
    label_generators = next(
        (a, b)
        for a in range(24) for b in range(24)
        if _order(a, label_table) == 4 and _order(b, label_table) == 3
        and len(_closure((a, b), label_table)) == 24
    )
    for image_a in range(24):
        for image_b in range(24):
            if _order(image_a, canonical) != 4 or _order(image_b, canonical) != 3:
                continue
            if len(_closure((image_a, image_b), canonical)) != 24:
                continue
            mapping, queue, valid = {0: 0}, deque([0]), True
            while queue and valid:
                label = queue.popleft()
                image = mapping[label]
                for generator, generator_image in zip(label_generators, (image_a, image_b)):
                    new_label = int(label_table[label, generator])
                    new_image = int(canonical[image, generator_image])
                    if new_label in mapping and mapping[new_label] != new_image:
                        valid = False
                        break
                    if new_label not in mapping:
                        mapping[new_label] = new_image
                        queue.append(new_label)
            if valid and len(mapping) == 24:
                if all(mapping[int(label_table[a, b])] == canonical[mapping[a], mapping[b]] for a in range(24) for b in range(24)):
                    # The released list is applied left-to-right, so physical state
                    # updates compose in the reverse order of the inferred table.
                    return np.asarray([rotations[mapping[label]].T for label in range(24)])
    raise ValueError("released labels are not isomorphic to the one-qubit Clifford group")


def toggling_frame_features(sequence, rotations, max_lag=4):
    """Quadratic error-path features in the cumulative-control toggling frame.

    Rotations act on column Bloch vectors. For listed controls G_1,...,G_t the
    cumulative frame is G_t...G_1, and a laboratory error vector is represented in
    that toggling frame by the transpose of the cumulative rotation.
    """
    frame = np.eye(3)
    path = []
    for action in sequence:
        frame = rotations[action] @ frame
        path.append(frame.T.ravel())
    path = np.asarray(path)
    total = path.sum(axis=0)
    output = [len(sequence), *total, *np.outer(total, total).ravel()]
    for lag in range(1, max_lag + 1):
        correlation = path[lag:].T @ path[:-lag] if len(path) > lag else np.zeros((9, 9))
        output.extend(correlation.ravel())
    return np.asarray(output, dtype=float)
