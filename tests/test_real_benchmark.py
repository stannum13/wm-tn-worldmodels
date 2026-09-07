import numpy as np

from ptwm.real_benchmark import fit_rb_decay, make_features, predict_rb_decay, split_by_length


def test_sequence_features_encode_order_and_length_without_leaking_targets():
    rows = [{"cl_ops": [0, 1, 0], "p0": 0.9}, {"cl_ops": [1, 0], "p0": 0.8}]
    x, y = make_features(rows, n_actions=2)
    assert x.shape == (2, 7)  # length + 2 one-hot counts + 4 ordered bigrams
    assert y.tolist() == [0.9, 0.8]
    assert x[0, 0] == 3
    assert x[0, 4] == 1  # (0, 1)
    assert x[0, 5] == 1  # (1, 0)


def test_length_split_is_extrapolative_and_disjoint():
    rows = [{"cl_ops": [0] * n, "p0": 1.0} for n in (2, 3, 20, 21, 40)]
    train, test = split_by_length(rows, max_train_length=20)
    assert all(len(r["cl_ops"]) <= 20 for r in train)
    assert all(len(r["cl_ops"]) > 20 for r in test)


def test_rb_decay_recovers_a_known_exponential():
    lengths = np.arange(2, 21)
    targets = 0.45 * 0.97**lengths + 0.5
    params = fit_rb_decay(lengths, targets)
    assert np.max(np.abs(predict_rb_decay(params, lengths) - targets)) < 1e-5
