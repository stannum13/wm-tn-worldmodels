import numpy as np
import pytest

from ptwm.schedule_inference import (
    enumerate_binary_schedules, filter_schedule_scores,
    filter_switching_schedule_scores, fit_detector_emissions, schedule_scores,
    select_schedule,
)


def test_enumeration_has_exact_change_bound_and_stable_order():
    zero = enumerate_binary_schedules(4, 0)
    one = enumerate_binary_schedules(4, 1)
    two = enumerate_binary_schedules(4, 2)
    assert len(zero) == 2
    assert len(one) == 8
    assert len(two) == 14
    assert np.all(np.sum(one[:, 1:] != one[:, :-1], axis=1) <= 1)
    np.testing.assert_array_equal(two[0], [0, 0, 0, 0])
    np.testing.assert_array_equal(two[-1], [1, 1, 1, 1])


def test_emissions_and_schedule_score_identify_clean_change():
    a = np.zeros((20, 4), dtype=bool)
    b = np.ones((20, 4), dtype=bool)
    emission = fit_detector_emissions([a, b], pseudocount=1)
    sequences = enumerate_binary_schedules(2, 1)
    detector_bins = np.array([0, 0, 1, 1])
    records = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=bool)
    scores = schedule_scores(records, emission, sequences, detector_bins)
    chosen = select_schedule(scores, sequences, penalty=0)
    np.testing.assert_array_equal(sequences[chosen], [[0, 1], [1, 0]])
    assert np.all(select_schedule(scores, sequences, penalty=100)[:, None] >= 0)


def test_inference_rejects_invalid_shapes_and_parameters():
    with pytest.raises(ValueError):
        enumerate_binary_schedules(0, 1)
    with pytest.raises(ValueError):
        fit_detector_emissions([np.zeros((2, 3)), np.zeros((3, 3))])
    sequences = enumerate_binary_schedules(2, 1)
    emission = np.full((2, 3), .5)
    with pytest.raises(ValueError):
        schedule_scores(np.zeros((2, 2)), emission, sequences, np.array([0, 1, 1]))
    with pytest.raises(ValueError):
        select_schedule(np.zeros((2, len(sequences))), sequences, penalty=-1)


def test_persistent_filter_accumulates_and_resets_each_stream():
    sequences = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    scores = np.array([[[1., 0.], [0., 2.], [0., 0.]],
                       [[0., 3.], [2., 0.], [0., 0.]]])
    choice, state = filter_schedule_scores(scores, sequences, penalty=0, retention=1)
    np.testing.assert_array_equal(choice, [[0, 1, 1], [1, 1, 1]])
    np.testing.assert_array_equal(state[:, 0], scores[:, 0])
    memoryless, _ = filter_schedule_scores(scores, sequences, penalty=0, retention=0)
    np.testing.assert_array_equal(memoryless[:, :2], [[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        filter_schedule_scores(scores, sequences, penalty=0, retention=1.1)


def test_switching_filter_resets_only_on_decisive_disagreement():
    sequences = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    scores = np.array([[[5., 0.], [5., 0.], [0., 9.], [0., 9.]]])
    no_reset, _, flags = filter_switching_schedule_scores(
        scores, sequences, penalty=0, retention=1, reset_margin=np.inf)
    reset, state, reset_flags = filter_switching_schedule_scores(
        scores, sequences, penalty=0, retention=1, reset_margin=2)
    np.testing.assert_array_equal(no_reset, [[0, 0, 0, 1]])
    np.testing.assert_array_equal(reset, [[0, 0, 1, 1]])
    np.testing.assert_array_equal(flags, False)
    np.testing.assert_array_equal(reset_flags, [[False, False, True, False]])
    np.testing.assert_array_equal(state[0, 2], [0., 9.])
    with pytest.raises(ValueError):
        filter_switching_schedule_scores(scores, sequences, penalty=0,
                                         retention=1, reset_margin=-1)
