import numpy as np

from ptwm.streaming import (
    StreamParameters,
    benchmark_estimators,
    causal_hmm_filter,
    fit_gaussian_hmm,
    forecast_belief,
    simulate_switching_streams,
)


def test_stream_generator_is_reproducible_and_timestamped():
    a = simulate_switching_streams(seed=3, n_streams=4, length=100)
    b = simulate_switching_streams(seed=3, n_streams=4, length=100)
    assert np.array_equal(a["states"], b["states"])
    assert np.array_equal(a["observations"], b["observations"], equal_nan=True)
    assert np.all(np.diff(a["timestamps"], axis=1) == 1)


def test_causal_filter_does_not_use_future_samples():
    data = simulate_switching_streams(seed=4, n_streams=3, length=80)
    params = StreamParameters(data["transition"], np.array([-0.8, 0.8]), np.ones(2))
    original = causal_hmm_filter(data["observations"], params)
    changed = data["observations"].copy()
    changed[:, 50:] = 100.0
    modified = causal_hmm_filter(changed, params)
    assert np.allclose(original[:, :50], modified[:, :50])


def test_hmm_fit_recovers_state_order_and_switching_scale():
    data = simulate_switching_streams(
        seed=5, n_streams=20, length=500, artifact_probability=0.0, dropout_probability=0.0
    )
    fitted = fit_gaussian_hmm(data["observations"], iterations=20)
    assert fitted.means[0] < 0 < fitted.means[1]
    assert abs(fitted.transition[0, 1] - data["transition"][0, 1]) < 0.02
    assert abs(fitted.transition[1, 0] - data["transition"][1, 0]) < 0.04


def test_delay_forecast_improves_delayed_control_for_oracle_model():
    data = simulate_switching_streams(seed=6, n_streams=80, length=1000)
    params = StreamParameters(data["transition"], np.array([-0.8, 0.8]), np.ones(2))
    report = benchmark_estimators(data, params, delays=[25])
    assert report[25]["hmm_delay_forecast"]["brier_loss"] < report[25]["hmm_filter_current"]["brier_loss"]


def test_forecast_at_zero_delay_is_identity():
    q = np.linspace(0, 1, 9)
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert np.allclose(forecast_belief(q, transition, 0), q)
