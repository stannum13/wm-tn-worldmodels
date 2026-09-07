import numpy as np

from ptwm.clifford import (
    infer_clifford_rotations,
    infer_multiplication_table,
    signed_permutation_rotations,
    toggling_frame_features,
)


def test_inverse_sequences_recover_the_clifford_group():
    rotations = signed_permutation_rotations()
    lookup = {tuple(matrix.ravel()): i for i, matrix in enumerate(rotations)}
    table = np.array([
        [lookup[tuple((rotations[a] @ rotations[b]).ravel())] for b in range(24)]
        for a in range(24)
    ])
    inverse = {a: int(np.flatnonzero(table[a] == 0)[0]) for a in range(24)}
    rows = [{"cl_ops": [a, inverse[a]], "p0": 1.0} for a in range(24)]
    rows += [
        {"cl_ops": [a, b, inverse[int(table[a, b])]], "p0": 1.0}
        for a in range(24) for b in range(24)
    ]
    recovered = infer_multiplication_table([rows])
    assert np.array_equal(recovered, table)
    inferred_rotations = infer_clifford_rotations([rows])
    assert len({tuple(matrix.ravel()) for matrix in inferred_rotations}) == 24
    sequence_total = np.eye(3, dtype=int)
    for action in rows[-1]["cl_ops"]:
        sequence_total = inferred_rotations[action] @ sequence_total
    assert np.array_equal(sequence_total, np.eye(3, dtype=int))


def test_toggling_features_have_fixed_size_for_variable_sequences():
    rotations = signed_permutation_rotations()
    short = toggling_frame_features([1, 2], rotations, max_lag=2)
    long = toggling_frame_features([1, 2, 3, 4], rotations, max_lag=2)
    assert short.shape == long.shape == (253,)


def test_toggling_features_follow_column_state_control_chronology():
    rotations = signed_permutation_rotations()
    sequence = [1, 2, 3]
    frame = np.eye(3)
    toggling_frames = []
    state = np.array([0.2, -0.3, 0.7])
    for action in sequence:
        frame = rotations[action] @ frame
        toggling_frames.append(frame.T)
    direct = rotations[3] @ rotations[2] @ rotations[1] @ state
    assert np.allclose(frame @ state, direct)
    features = toggling_frame_features(sequence, rotations, max_lag=0)
    assert np.allclose(features[1:10], np.sum(toggling_frames, axis=0).ravel())
