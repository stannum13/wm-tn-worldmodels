import numpy as np

from ptwm.nonlinear_streaming import particle_filter, regression_metrics, robust_ekf, simulate_nonlinear_streams


def test_nonlinear_stream_is_reproducible():
    a = simulate_nonlinear_streams(seed=1, n_streams=3, length=50)
    b = simulate_nonlinear_streams(seed=1, n_streams=3, length=50)
    assert np.array_equal(a["states"], b["states"])
    assert np.array_equal(a["observations"], b["observations"], equal_nan=True)


def test_particle_filter_is_causal_and_finite():
    data = simulate_nonlinear_streams(seed=2, n_streams=2, length=80)
    first, info = particle_filter(data["observations"], seed=3, particles=32, delay=4, contamination=1e-3)
    changed = data["observations"].copy(); changed[:, 50:] = 100
    second, _ = particle_filter(changed, seed=3, particles=32, delay=4, contamination=1e-3)
    assert np.allclose(first[:, :50], second[:, :50])
    assert np.isfinite(first).all() and 0 < info["mean_ess_fraction"] <= 1


def test_estimators_beat_zero_prediction():
    data = simulate_nonlinear_streams(seed=4, n_streams=8, length=200)
    ekf, _ = robust_ekf(data["observations"], delay=5)
    pf, _ = particle_filter(data["observations"], seed=5, particles=64, delay=5, contamination=1e-3)
    zero = np.zeros_like(ekf)
    assert regression_metrics(ekf, data["states"], delay=5)["mse"] < regression_metrics(zero, data["states"], delay=5)["mse"]
    assert regression_metrics(pf, data["states"], delay=5)["mse"] < regression_metrics(zero, data["states"], delay=5)["mse"]
