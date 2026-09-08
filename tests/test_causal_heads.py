import numpy as np

from ptwm.streaming import causal_context_features, fit_causal_head, slow_head_denoiser


def test_causal_features_do_not_change_when_future_changes():
    y = np.arange(40.0)[None, :]
    belief = np.linspace(0, 1, 40)[None, :]
    a, _, loc = causal_context_features(y, belief, window=8, stride=4, delay=3)
    y[:, 30:] = -999
    b, _, _ = causal_context_features(y, belief, window=8, stride=4, delay=3)
    assert np.allclose(a[loc[:, 1] < 30], b[loc[:, 1] < 30])


def test_spline_head_fits_nonlinearity_better_than_linear_head():
    x = np.linspace(-1, 1, 200)[:, None]
    target = x[:, 0] ** 2
    linear = fit_causal_head(x, target, knots=1).predict(x)
    spline = fit_causal_head(x, target, knots=6).predict(x)
    assert np.mean((spline - target) ** 2) < 0.2 * np.mean((linear - target) ** 2)


def test_slow_head_is_held_between_updates_and_bounded():
    base = np.full((1, 12), 0.2)
    locations = np.array([[0, 3], [0, 7]])
    out = slow_head_denoiser(base, np.array([0.8, 0.1]), locations, stride=4)
    assert out.shape == base.shape
    assert np.all((0 <= out) & (out <= 1))
    assert np.allclose(out[:, :3], base[:, :3])
