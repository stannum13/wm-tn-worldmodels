import numpy as np
import pytest

from ptwm.models import GRUModel, MarkovChannel, ProcessMPO, TransferTensor, TransformerModel

ALL_MODELS = [
    lambda: MarkovChannel(),
    lambda: TransferTensor(),
    lambda: ProcessMPO(chi=2, epochs=5),
    lambda: GRUModel(hidden=2, epochs=3),
    lambda: TransformerModel(d_model=8, nhead=1, layers=1, epochs=2),
]


@pytest.mark.parametrize("idx", range(len(ALL_MODELS)))
def test_model_fit_predict_contract(synthetic_ds, idx):
    from ptwm.splits import horizon_split

    s = horizon_split(synthetic_ds, train_cap=12)
    model = ALL_MODELS[idx]()
    model.fit(s.train)
    y = model.predict_log(s.test[:50])
    assert y.shape == (50,)
    assert np.all(np.isfinite(y))
    # log-fidelity predictions should be mostly negative (fidelities < 1)
    assert np.mean(y < 0.1) > 0.5


def test_markov_recovers_decay(synthetic_ds):
    """On synthetic data with known per-gate decay, Markov channel should beat
    constant prediction (it uses gate identity)."""
    from ptwm.splits import horizon_split

    s = horizon_split(synthetic_ds, train_cap=12)
    m = MarkovChannel()
    m.fit(s.train)
    y = m.predict_log(s.test)
    y_true = np.array([np.log(ep.fidelity) for ep in s.test])
    const = np.full_like(y_true, np.mean(np.log([ep.fidelity for ep in s.train])))
    assert np.mean((y - y_true) ** 2) < np.mean((const - y_true) ** 2)


def test_process_mpo_causality_constraint(synthetic_ds):
    from ptwm.splits import horizon_split

    s = horizon_split(synthetic_ds, train_cap=12)
    m = ProcessMPO(chi=2, epochs=5)
    m.fit(s.train)
    chk = m.check()
    assert chk["causality_violation"] <= 1e-6
    assert chk["max_spectral_norm"] <= m.rho_max + 1e-6


def test_transfer_tensor_recovers_exponential(synthetic_ds):
    from ptwm.splits import horizon_split

    s = horizon_split(synthetic_ds, train_cap=12)
    m = TransferTensor()
    m.fit(s.train)
    # tau positive and finite; linear coefficient b negative (decay). On synthetic
    # memoryless data tau can be tiny (fast transient) — only sanity-bound it.
    assert 0.0 < m.tau_ < 1e4
    assert m.b_ < 0


def test_gru_learns(synthetic_ds):
    from ptwm.splits import horizon_split

    s = horizon_split(synthetic_ds, train_cap=12)
    m = GRUModel(hidden=2, epochs=50)
    m.fit(s.train)
    # loss must drop below target variance (it learned *something*); 50 epochs
    # converges on this easy synthetic task
    y_true = np.array([np.log(ep.fidelity) for ep in s.train])
    var = float(np.var(y_true))
    assert m._final_train_loss < var
    y = m.predict_log(s.test[:20])
    assert np.all(np.isfinite(y))
