"""Model interface for Experiment A (the plan's common contract, adapted to the
sequence-fidelity observables this dataset provides).

Every model implements:

- observe: ingest the control sequence (gate indices) and context (bias, idle)
- act:    the control direction along which we evaluate (bias/idle/family) — models
          are conditioned on these at fit time
- latent: the compressed predictive state (scalar, vector, or matrix bond)
- predict: whole-sequence observable forecast, causally (no future gates visible)
- check:  physicality/causality residuals of the model's outputs
- budget: parameter count and fit cost

All models are conditioned on the control context (bias, idle) so that held-out-bias
evaluation is meaningful: the context enters as a normalized 2-vector feature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .data import Episode, N_GATES

BIAS_MAX = 0.64
IDLE_MAX = 180.0


def context_vec(ep: Episode) -> np.ndarray:
    """Normalized control context: [bias/0.64, idle/180]."""
    return np.array([ep.bias / BIAS_MAX, ep.idle / IDLE_MAX], dtype=np.float32)


class BaseModel(ABC):
    """Common interface. Subclasses implement fit/predict/check/budget."""

    name: str = "base"

    @abstractmethod
    def fit(self, train_eps: list[Episode]) -> None: ...

    @abstractmethod
    def predict_log(self, eps: list[Episode]) -> np.ndarray: ...

    def check(self) -> dict:
        """Physicality/causality residuals; default: nothing to report."""
        return {}

    def budget(self) -> dict:
        return {"name": self.name, "params": self.n_params()}

    def n_params(self) -> int:
        return 0


class MarkovChannel(BaseModel):
    """Time-homogeneous Markov channel: per-gate mean log-fidelity decrement.

    log f(seq) = sum_t log f_gate(g_t) + w_ctx . ctx + const. No memory: each gate
    contributes independently. Fit by ridge least squares on gate-count features
    plus context features with intercept.
    """

    name = "markov"

    def __init__(self, n_gates: int = N_GATES, ridge: float = 1e-6):
        self.n_gates = n_gates
        self.ridge = ridge
        self.coef_: np.ndarray | None = None
        self.ctx_coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, train_eps: list[Episode]) -> None:
        from .data import gate_counts

        X, y = gate_counts(train_eps, self.n_gates)
        C = np.array([context_vec(ep) for ep in train_eps])
        # Ridge-closed form with intercept.
        A = np.hstack([X, C, np.ones((X.shape[0], 1))])
        reg = np.eye(A.shape[1]) * self.ridge
        reg[-1, -1] = 0.0  # don't penalize intercept
        w = np.linalg.solve(A.T @ A + reg, A.T @ y)
        ng = self.n_gates
        self.coef_ = w[:ng]
        self.ctx_coef_ = w[ng : ng + 2]
        self.intercept_ = float(w[-1])

    def predict_log(self, eps: list[Episode]) -> np.ndarray:
        from .data import gate_counts

        X, _ = gate_counts(eps, self.n_gates)
        C = np.array([context_vec(ep) for ep in eps])
        return X @ self.coef_ + C @ self.ctx_coef_ + self.intercept_

    def n_params(self) -> int:
        return self.n_gates + 2 + 1

    def check(self) -> dict:
        # Physicality: exp(coef) per-gate survival factors should be in (0, 1].
        factors = np.exp(self.coef_)
        return {
            "gate_factors_in_unit_range": float(np.mean((factors > 0) & (factors <= 1.0))),
            "max_gate_factor": float(factors.max()),
        }


class TransferTensor(BaseModel):
    """Scalar transfer-tensor / exponential-kernel autoregression.

    log f(L, ctx) = a + b*L + c*exp(-L/tau) + d . ctx — the c term is the
    non-Markovian memory kernel with time constant tau; ctx carries the control
    context linearly. Fit by nonlinear least squares (6 params).
    """

    name = "transfer_tensor"

    def __init__(self):
        self.a_ = 0.0
        self.b_ = 0.0
        self.c_ = 0.0
        self.tau_ = 1.0
        self.d_ = np.zeros(2)

    def fit(self, train_eps: list[Episode]) -> None:
        from scipy.optimize import least_squares

        L = np.array([ep.length for ep in train_eps], dtype=np.float64)
        y = np.array([np.log(max(ep.fidelity, 1e-12)) for ep in train_eps])
        C = np.array([context_vec(ep) for ep in train_eps], dtype=np.float64)
        # Length x context-binned means stabilize the fit against shot noise.
        lens_unique = np.unique(L)
        ctx_keys = np.unique(np.round(C, 3), axis=0)
        Lu, Cu, mu = [], [], []
        for lu in lens_unique:
            for ck in ctx_keys:
                mask = (L == lu) & (np.abs(C - ck).max(axis=1) < 1e-3)
                if mask.sum() > 0:
                    Lu.append(lu)
                    Cu.append(ck)
                    mu.append(y[mask].mean())
        Lu = np.array(Lu)
        Cu = np.array(Cu)
        mu = np.array(mu)

        def resid(params):
            a, b, c, log_tau, d0, d1 = params
            tau = np.exp(log_tau)
            pred = a + b * Lu + c * np.exp(-Lu / tau) + d0 * Cu[:, 0] + d1 * Cu[:, 1]
            return pred - mu

        best = None
        for tau0 in (2.0, 5.0, 10.0, 20.0):
            x0 = np.array([
                mu.mean(),
                -0.01,
                0.0,
                np.log(tau0),
                0.0,
                0.0,
            ])
            try:
                sol = least_squares(resid, x0, method="lm", max_nfev=4000)
            except Exception:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        if best is None:
            # Degenerate/no-memory data: fall back to pure linear decay (c=0).
            p = np.polyfit(Lu, mu, 1)
            self.a_, self.b_, self.c_, self.tau_ = float(p[1]), float(p[0]), 0.0, 1.0
            self.d_ = np.zeros(2)
            return
        self.a_, self.b_, self.c_ = best.x[0], best.x[1], best.x[2]
        self.tau_ = float(np.exp(best.x[3]))
        self.d_ = best.x[4:6]

    def predict_log(self, eps: list[Episode]) -> np.ndarray:
        L = np.array([ep.length for ep in eps], dtype=np.float64)
        C = np.array([context_vec(ep) for ep in eps], dtype=np.float64)
        return self.a_ + self.b_ * L + self.c_ * np.exp(-L / self.tau_) + C @ self.d_

    def n_params(self) -> int:
        return 6

    def budget(self) -> dict:
        return {**super().budget(), "tau": self.tau_}


class ProcessMPO(BaseModel):
    """Causality-constrained process-MPO with bond dimension chi.

    The latent state z_t in R^chi evolves linearly under each gate's MPO tensor:
        z_0     = W_ctx @ ctx                      (context-initialized bond)
        z_{t+1} = alpha_{g_t} * Ahat_{g_t} z_t     (contractive gate map)
    and the observable head reads out log-fidelity contribution per step:
        log f = sum_t <w, z_t> + b.

    Causality constraint: Ahat_g is the unit-spectral-norm-normalized gate matrix and
    alpha_g = tanh(raw) * rho_max, so every gate map has spectral norm <= rho_max < 1.
    The induced superoperator on the latent cannot grow — the linear-algebra
    surrogate for a CPTP/contractive process tensor in observable space.
    """

    name = "process_mpo"

    def __init__(self, chi: int = 2, rho_max: float = 0.999, n_gates: int = N_GATES,
                 lr: float = 0.05, epochs: int = 300, seed: int = 0):
        self.chi = chi
        self.rho_max = rho_max
        self.n_gates = n_gates
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.A_: np.ndarray | None = None  # (n_gates, chi, chi) final gate maps
        self.W_ctx_: np.ndarray | None = None  # (chi, 2)
        self.w_: np.ndarray | None = None  # (chi,)
        self.b_: float = 0.0

    def fit(self, train_eps: list[Episode]) -> None:
        import torch

        torch.manual_seed(self.seed)
        ng, chi = self.n_gates, self.chi
        # Leaf parameters: per-gate matrix (normalized to unit spectral norm at use
        # time) times a damped contraction coefficient, context read-in, readout, bias.
        raw_A = torch.randn(ng, chi, chi) * 0.1
        raw_A.requires_grad_(True)
        raw_alpha = torch.zeros(ng, requires_grad=True)
        W_ctx = torch.randn(chi, 2) * 0.1
        W_ctx.requires_grad_(True)
        w = torch.randn(chi) * 0.01
        w.requires_grad_(True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([raw_A, raw_alpha, W_ctx, w, b], lr=self.lr)

        # Group episodes by length for batched recurrence.
        by_len: dict[int, list[Episode]] = {}
        for ep in train_eps:
            by_len.setdefault(ep.length, []).append(ep)
        batches = []
        for L, eps in sorted(by_len.items()):
            gates = torch.tensor([ep.gates for ep in eps], dtype=torch.long)  # (B, L)
            ctx = torch.tensor(
                np.array([context_vec(ep) for ep in eps], dtype=np.float32)
            )
            y = torch.tensor([np.log(max(ep.fidelity, 1e-12)) for ep in eps], dtype=torch.float32)
            batches.append((gates, ctx, y))

        for epoch in range(self.epochs):
            total = 0.0
            for gates, ctx, y in batches:
                B, L = gates.shape
                z = ctx @ W_ctx.T  # (B, chi) context-initialized bond
                logf = b.expand(B).clone()
                for t in range(L):
                    gt = gates[:, t]  # (B,)
                    alpha = torch.tanh(raw_alpha[gt]) * self.rho_max  # (B,)
                    Ag = raw_A[gt]  # (B, chi, chi)
                    # Normalize spatial part to unit spectral norm, then damp.
                    Ag = Ag / (torch.linalg.matrix_norm(Ag, ord=2).clamp_min(1e-8)).unsqueeze(-1).unsqueeze(-1)
                    z = torch.einsum("bij,bj->bi", Ag, z) * alpha.unsqueeze(-1)
                    logf = logf + z @ w
                loss = ((logf - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss)
            self._final_train_loss = float(total)

        with torch.no_grad():
            alpha = torch.tanh(raw_alpha) * self.rho_max
            Ag = raw_A.clone()
            Ag = Ag / torch.linalg.matrix_norm(Ag, ord=2, keepdim=True).clamp_min(1e-8)
            self.A_ = (Ag * alpha.unsqueeze(-1).unsqueeze(-1)).numpy()
            self.W_ctx_ = W_ctx.numpy()
            self.w_ = w.numpy()
            self.b_ = float(b[0])

    def predict_log(self, eps: list[Episode]) -> np.ndarray:
        import torch

        with torch.no_grad():
            A = torch.tensor(self.A_, dtype=torch.float32)
            W_ctx = torch.tensor(self.W_ctx_, dtype=torch.float32)
            w = torch.tensor(self.w_, dtype=torch.float32)
            out = np.empty(len(eps), dtype=np.float64)
            # Batch by length for vectorized recurrence.
            by_len: dict[int, list[int]] = {}
            for i, ep in enumerate(eps):
                by_len.setdefault(ep.length, []).append(i)
            for L, idxs in by_len.items():
                gates = torch.tensor([eps[i].gates for i in idxs], dtype=torch.long)  # (B, L)
                ctx = torch.tensor(
                    np.array([context_vec(eps[i]) for i in idxs], dtype=np.float32)
                )
                z = ctx @ W_ctx.T  # (B, chi)
                logf = torch.full((len(idxs),), self.b_)
                for t in range(L):
                    gt = gates[:, t]
                    z = torch.einsum("bij,bj->bi", A[gt], z)
                    logf = logf + z @ w
                out[np.array(idxs)] = logf.numpy()
        return out

    def n_params(self) -> int:
        # alpha (ng) + A (ng*chi*chi - ng*(chi*chi - chi) after normalization)
        # + W_ctx (2*chi) + w (chi) + b (1)
        return self.n_gates * (1 + self.chi) + 2 * self.chi + self.chi + 1

    def check(self) -> dict:
        """Causality residuals: spectral norms of A_g must be <= rho_max."""
        assert self.A_ is not None
        norms = np.linalg.norm(self.A_, ord=2, axis=(1, 2))
        return {
            "max_spectral_norm": float(norms.max()),
            "rho_max": self.rho_max,
            "causality_violation": float(max(0.0, norms.max() - self.rho_max)),
            "final_train_loss": getattr(self, "_final_train_loss", float("nan")),
        }

    def budget(self) -> dict:
        return {**super().budget(), "chi": self.chi, "rho_max": self.rho_max}


class GRUModel(BaseModel):
    """GRU baseline: gate one-hots -> GRU (context-initialized hidden) -> per-step
    readout of log-fidelity. Nonlinear recurrent latent with hidden size h; same
    causal contract as the MPO: context enters only through the initial latent.
    """

    name = "gru"

    def __init__(self, hidden: int = 8, lr: float = 0.01, epochs: int = 150, seed: int = 0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, train_eps: list[Episode]) -> None:
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        h = self.hidden
        self.gru_ = nn.GRU(input_size=N_GATES, hidden_size=h, batch_first=True)
        self.ctx_ = nn.Linear(2, h)  # context -> initial hidden
        self.head_ = nn.Linear(h, 1)
        nn.init.zeros_(self.head_.weight)
        nn.init.zeros_(self.head_.bias)
        opt = torch.optim.Adam(
            list(self.gru_.parameters()) + list(self.ctx_.parameters()) + list(self.head_.parameters()),
            lr=self.lr,
        )

        by_len: dict[int, list[Episode]] = {}
        for ep in train_eps:
            by_len.setdefault(ep.length, []).append(ep)
        batches = []
        for L, eps in sorted(by_len.items()):
            X = np.zeros((len(eps), L, N_GATES), dtype=np.float32)
            for i, ep in enumerate(eps):
                for t, g in enumerate(ep.gates):
                    X[i, t, g] = 1.0
            ctx = torch.tensor(
                np.array([context_vec(ep) for ep in eps], dtype=np.float32)
            )
            y = torch.tensor([np.log(max(ep.fidelity, 1e-12)) for ep in eps], dtype=torch.float32)
            batches.append((torch.tensor(X), ctx, y))

        for epoch in range(self.epochs):
            total = 0.0
            for X, ctx, y in batches:
                h0 = self.ctx_(ctx).unsqueeze(0)  # (1, B, H)
                out, _ = self.gru_(X, h0)  # (B, L, H)
                logf = self.head_(out).squeeze(-1).sum(dim=1)  # (B,)
                loss = ((logf - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss)
            self._final_train_loss = float(total)

    def predict_log(self, eps: list[Episode]) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = np.empty(len(eps), dtype=np.float64)
            by_len: dict[int, list[int]] = {}
            for i, ep in enumerate(eps):
                by_len.setdefault(ep.length, []).append(i)
            for L, idxs in by_len.items():
                X = torch.zeros(len(idxs), L, N_GATES)
                for j, i in enumerate(idxs):
                    for t, g in enumerate(eps[i].gates):
                        X[j, t, g] = 1.0
                ctx = torch.tensor(
                    np.array([context_vec(eps[i]) for i in idxs], dtype=np.float32)
                )
                h0 = self.ctx_(ctx).unsqueeze(0)
                o, _ = self.gru_(X, h0)
                out[np.array(idxs)] = self.head_(o).squeeze(-1).sum(dim=1).numpy()
        return out

    def n_params(self) -> int:
        h = self.hidden
        gru = 3 * (h * (h + N_GATES) + h)
        ctx = 2 * h + h
        head = h + 1
        return gru + ctx + head

    def check(self) -> dict:
        return {"final_train_loss": getattr(self, "_final_train_loss", float("nan"))}


class TransformerModel(BaseModel):
    """Small causal Transformer baseline (context prepended as a token)."""

    name = "transformer"

    def __init__(self, d_model: int = 32, nhead: int = 2, layers: int = 2,
                 lr: float = 0.005, epochs: int = 100, seed: int = 0):
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def _build(self):
        import torch
        from torch import nn

        class TinyTransformer(nn.Module):
            def __init__(self, d_model, nhead, layers):
                super().__init__()
                self.embed = nn.Linear(N_GATES, d_model)
                self.ctx_embed = nn.Linear(2, d_model)
                self.pos = nn.Parameter(torch.zeros(1, 64, d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=64,
                    batch_first=True,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=layers)
                self.head = nn.Linear(d_model, 1)

            def forward(self, x, ctx):
                # x: (B, T, G), ctx: (B, 2). Context token prepended; causal mask
                # keeps every position causal w.r.t. gates at later times.
                B, T, _ = x.shape
                h = self.embed(x) + self.pos[:, :T]
                ctx_tok = self.ctx_embed(ctx).unsqueeze(1)  # (B, 1, D)
                h = torch.cat([ctx_tok, h], dim=1)  # (B, T+1, D)
                Tp = T + 1
                mask = torch.triu(torch.ones(Tp, Tp, dtype=torch.bool), diagonal=1)
                h = self.enc(h, mask=mask)
                return self.head(h[:, 1:]).squeeze(-1)  # (B, T) gate-position readouts

        return TinyTransformer(self.d_model, self.nhead, self.layers)

    def fit(self, train_eps: list[Episode]) -> None:
        import torch

        torch.manual_seed(self.seed)
        self.net_ = self._build()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)

        by_len: dict[int, list[Episode]] = {}
        for ep in train_eps:
            by_len.setdefault(ep.length, []).append(ep)
        batches = []
        for L, eps in sorted(by_len.items()):
            X = np.zeros((len(eps), L, N_GATES), dtype=np.float32)
            for i, ep in enumerate(eps):
                for t, g in enumerate(ep.gates):
                    X[i, t, g] = 1.0
            ctx = torch.tensor(
                np.array([context_vec(ep) for ep in eps], dtype=np.float32)
            )
            y = torch.tensor([np.log(max(ep.fidelity, 1e-12)) for ep in eps], dtype=torch.float32)
            batches.append((torch.tensor(X), ctx, y))

        for epoch in range(self.epochs):
            total = 0.0
            for X, ctx, y in batches:
                logf = self.net_(X, ctx).sum(dim=1)
                loss = ((logf - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss)
            self._final_train_loss = float(total)

    def predict_log(self, eps: list[Episode]) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = np.empty(len(eps), dtype=np.float64)
            by_len: dict[int, list[int]] = {}
            for i, ep in enumerate(eps):
                by_len.setdefault(ep.length, []).append(i)
            for L, idxs in by_len.items():
                X = torch.zeros(len(idxs), L, N_GATES)
                for j, i in enumerate(idxs):
                    for t, g in enumerate(eps[i].gates):
                        X[j, t, g] = 1.0
                ctx = torch.tensor(
                    np.array([context_vec(eps[i]) for i in idxs], dtype=np.float32)
                )
                out[np.array(idxs)] = self.net_(X, ctx).sum(dim=1).numpy()
        return out

    def n_params(self) -> int:
        d = self.d_model
        per_layer = 4 * d * d + 2 * d * 64 + 3 * d  # attn + ffn + norms (approx)
        return 2 * per_layer + N_GATES * d + 2 * d + 64 * d + d + 1
