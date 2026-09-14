import numpy as np
import pytest

stim = pytest.importorskip("stim")

from ptwm.noise_schedule import compile_noise_schedule, schedule_from_detector_modes, tick_schedule
from ptwm.time_templates import splice_noise
from scripts.run_surface_frontier_challenge import circuit, REGIMES


def test_schedule_compiler_matches_independent_splice_reference():
    a, b = [circuit(3, *r) for r in REGIMES]
    ticks = sum(op.name == "TICK" for op in a.flattened())
    for fraction in (1 / 3, .5, 2 / 3):
        schedule = tick_schedule(ticks, [(int(ticks * fraction), 1)], initial=0)
        actual = compile_noise_schedule([a, b], schedule)
        assert actual == splice_noise(a, b, fraction)
        assert actual.without_noise() == a.flattened().without_noise()
        assert actual.get_detector_coordinates() == a.get_detector_coordinates()
    two = tick_schedule(ticks, [(ticks // 3, 1), (int(ticks * 2 / 3), 0)], initial=0)
    assert compile_noise_schedule([a, b], two) == splice_noise(splice_noise(a, b, 1 / 3), a, 2 / 3)


def test_schedule_carries_final_slot_and_rejects_ambiguous_inputs():
    a = stim.Circuit("R 0\nX_ERROR(0.01) 0\nTICK\nX_ERROR(0.02) 0\nM 0")
    b = stim.Circuit("R 0\nX_ERROR(0.03) 0\nTICK\nX_ERROR(0.04) 0\nM 0")
    result = compile_noise_schedule([a, b], np.array([0, 1]))
    assert result[-2].gate_args_copy() == [.04]
    for invalid in ([0], [0, 2], [0, .5]):
        with pytest.raises(ValueError):
            compile_noise_schedule([a, b], invalid)
    with pytest.raises(ValueError):
        compile_noise_schedule([a, stim.Circuit("R 0\nH 0\nTICK\nM 0")], [0, 1])
    with pytest.raises(ValueError):
        tick_schedule(5, [(2, 1), (2, 0)])


def test_detector_modes_compile_at_fractional_tick_boundaries():
    assert schedule_from_detector_modes(35, [0, 0, 1, 1, 1, 1]).tolist() == [0] * 11 + [1] * 25
    assert schedule_from_detector_modes(35, [1, 1, 0, 0, 1, 1]).tolist() == [1] * 11 + [0] * 12 + [1] * 13
    with pytest.raises(ValueError):
        schedule_from_detector_modes(2, [0, 1, 0, 1])
