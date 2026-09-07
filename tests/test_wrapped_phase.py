import numpy as np

from ptwm.wrapped_phase import (
    circular_loss,
    grid_filter,
    phase_ekf,
    phase_particle_filter,
    simulate_wrapped_phase,
)


def test_wrapped_phase_stream_is_reproducible_and_bounded():
    first = simulate_wrapped_phase(seed=1, n_streams=3, length=80)
    second = simulate_wrapped_phase(seed=1, n_streams=3, length=80)
    assert np.array_equal(first["states"], second["states"])
    assert np.array_equal(first["observations"], second["observations"], equal_nan=True)
    assert np.all(first["states"] >= -np.pi) and np.all(first["states"] < np.pi)
    assert np.array_equal(first["probe_phases"][:, 7::8], np.full((3, 10), np.pi / 2))


def test_phase_filters_are_causal_finite_and_normalized():
    data = simulate_wrapped_phase(seed=2, n_streams=2, length=60)
    args = (data["observations"], data["probe_phases"])
    grid, grid_info = grid_filter(*args, delay=3, bins=128)
    pf, pf_info = phase_particle_filter(*args, seed=3, particles=128, delay=3)
    ekf, _ = phase_ekf(*args, delay=3)
    changed = data["observations"].copy()
    changed[:, 40:] = 10.0
    changed_pf, _ = phase_particle_filter(
        changed, data["probe_phases"], seed=3, particles=128, delay=3
    )
    assert np.allclose(pf[:, :40], changed_pf[:, :40])
    assert np.isfinite(grid).all() and np.isfinite(pf).all() and np.isfinite(ekf).all()
    assert 0.999999 < grid_info["mean_probability_mass"] < 1.000001
    assert 0 < pf_info["mean_ess_fraction"] <= 1


def test_grid_filter_beats_uninformed_phase_prediction():
    data = simulate_wrapped_phase(seed=4, n_streams=10, length=240)
    estimate, _ = grid_filter(
        data["observations"], data["probe_phases"], delay=4, bins=256
    )
    usable = data["states"].shape[1] - 4
    grid_loss = circular_loss(estimate[:, :usable], data["states"][:, 4:])
    zero_loss = circular_loss(np.zeros_like(estimate[:, :usable]), data["states"][:, 4:])
    assert grid_loss < zero_loss
