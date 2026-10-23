"""GlucoTwin Phase 2: models.

  * `PersistenceBaseline` - carry the last glucose value forward, alarm above a
    tuned threshold.
  * `train_lgbm` / `train_lgbm_regressor` - LightGBM for the CGM-only,
    EHR-only and fused comparisons, plus the glucose forecasters.
  * `FusionNet` - the main two-branch model: a dilated causal TCN over the last
    6 hours of 5-channel time series plus an MLP over static EHR, concatenated
    into a joint head that emits one event logit and three forecast values.

The network is implemented in numpy (this machine cannot install PyTorch; see
docs/DECISIONS.md) with manual backprop. Its gradients are checked against
finite differences in tests/test_models.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

SEED = 42

# ---------------------------------------------------------------- baseline


class PersistenceBaseline:
    """Forecast = hold the current glucose; alarm when it is already above 180."""

    name = "persistence"

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "PersistenceBaseline":
        self.g_index = FEATURE_INDEX_OF_CURRENT_GLUCOSE
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Score in [0, 1] from current glucose (squashed for AUROC/AUPRC)."""
        g = X[:, self.g_index]
        return 1.0 / (1.0 + np.exp(-(g - 180.0) / 12.0))


FEATURE_INDEX_OF_CURRENT_GLUCOSE = 0  # "g_now" is always the first feature


# ---------------------------------------------------------------- lightgbm

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.06,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 1,
    "verbosity": -1,
    "seed": SEED,
    "deterministic": True,
}


def train_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 250,
    pos_weight: float = 1.0,
) -> lgb.Booster:
    params = dict(LGBM_PARAMS)
    params["scale_pos_weight"] = float(pos_weight)
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=True)
    return lgb.train(params, dtrain, num_boost_round=n_estimators)


def train_lgbm_regressor(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 150,
) -> lgb.Booster:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.07,
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_threads": 1,
        "verbosity": -1,
        "seed": SEED,
        "deterministic": True,
    }
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=True)
    return lgb.train(params, dtrain, num_boost_round=n_estimators)


class IsotonicCalibrator:
    """Isotonic regression fitted on the validation split."""

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict(p)


# ---------------------------------------------------------------- fusion net

CHANNELS = 5
SEQ_LEN = 36  # 6 hours at 10-minute resolution


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608
    t = np.tanh(c * (x + 0.044715 * x**3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * c * (1 + 3 * 0.044715 * x**2)


def _he(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
    return (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float64)


@dataclass
class FusionNet:
    """Two-branch fusion network (numpy implementation)."""

    static_dim: int = 20
    width: int = 16
    mlp_hidden: int = 32
    seq_len: int = SEQ_LEN
    seed: int = SEED
    params: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)

    # ---- construction -------------------------------------------------
    def init(self) -> "FusionNet":
        rng = np.random.default_rng(self.seed)
        w = self.width
        def conv_w(n_in: int, n_out: int) -> np.ndarray:
            return (rng.standard_normal((3, n_in, n_out)) * np.sqrt(2.0 / (3 * n_in)))

        p: dict[str, np.ndarray] = {
            # sequence branch (kernel size 3, dilated, causal)
            "c0_W": conv_w(CHANNELS, w),
            "c0_b": np.zeros(w),
            "c1_W": conv_w(w, w),
            "c1_b": np.zeros(w),
            "c2_W": conv_w(w, w),
            "c2_b": np.zeros(w),
            "c3_W": conv_w(w, w),
            "c3_b": np.zeros(w),
            "c4_W": conv_w(w, w),
            "c4_b": np.zeros(w),
            # static branch
            "s0_W": _he(rng, self.static_dim, self.mlp_hidden),
            "s0_b": np.zeros(self.mlp_hidden),
            "s1_W": _he(rng, self.mlp_hidden, w),
            "s1_b": np.zeros(w),
            # joint head
            "h0_W": _he(rng, 3 * w, self.mlp_hidden),
            "h0_b": np.zeros(self.mlp_hidden),
            "h1_W": _he(rng, self.mlp_hidden, 4),
            "h1_b": np.zeros(4),
        }
        self.params = p
        return self

    # ---- helpers ------------------------------------------------------
    @staticmethod
    def _shift(x: np.ndarray, off: int) -> np.ndarray:
        """Shift right by `off` steps (causal padding with zeros)."""
        if off == 0:
            return x
        out = np.zeros_like(x)
        out[:, off:, :] = x[:, :-off, :]
        return out

    @staticmethod
    def _shift_back(dx: np.ndarray, off: int) -> np.ndarray:
        """Adjoint of `_shift`."""
        if off == 0:
            return dx
        out = np.zeros_like(dx)
        out[:, :-off, :] = dx[:, off:, :]
        return out

    @staticmethod
    def _layernorm(x: np.ndarray, eps: float = 1e-5):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        out = (x - mu) * inv
        return out, (mu, var, inv)

    @staticmethod
    def _layernorm_backward(dout: np.ndarray, x: np.ndarray, cache) -> np.ndarray:
        mu, var, inv = cache
        n = x.shape[-1]
        xhat = (x - mu) * inv
        dvar = (dout * xhat * inv * inv * -0.5).sum(axis=-1, keepdims=True)
        dmu = (-dout * inv).sum(axis=-1, keepdims=True) + dvar * (-2.0 / n) * (x - mu).sum(
            axis=-1, keepdims=True
        )
        return dout * inv + dvar * 2.0 * (x - mu) / n + dmu / n

    def _conv(self, x, W, b, dilation, cache=None):
        k = W.shape[0]
        T = x.shape[1]
        out = np.tile(b, (x.shape[0], T, 1))
        for j in range(k):
            off = dilation * (k - 1 - j)
            out = out + self._shift(x, off) @ W[j]
        if cache is not None:
            cache["x"] = x
            cache["W"] = W
            cache["dilation"] = dilation
        return out

    def _conv_backward(self, dout, cache):
        x, W, d = cache["x"], cache["W"], cache["dilation"]
        k = W.shape[0]
        dx = np.zeros_like(x)
        dW = np.zeros_like(W)
        for j in range(k):
            off = d * (k - 1 - j)
            xs = self._shift(x, off)
            dW[j] = np.einsum("btc,btd->cd", xs, dout)
            dx += self._shift_back(dout @ W[j].T, off)
        return dx, dW

    # ---- forward ------------------------------------------------------
    def forward(self, seq: np.ndarray, static: np.ndarray, keep_cache: bool = False):
        p = self.params
        caches: dict = {}

        c0 = self._conv(seq, p["c0_W"], p["c0_b"], 1, caches.setdefault("c0", {}) if keep_cache else None)
        a0 = gelu(c0)
        h, lnc = self._layernorm(a0)
        if keep_cache:
            caches["c0_out"] = c0
            caches["ln0"] = (a0, lnc)

        for i, dilation in ((1, 2), (2, 4), (3, 8), (4, 16)):
            res_in = h
            cin = self._conv(
                h, p[f"c{i}_W"], p[f"c{i}_b"], dilation, caches.setdefault(f"c{i}", {}) if keep_cache else None
            )
            a = gelu(cin)
            hn, lnc2 = self._layernorm(a)
            h = res_in + hn
            if keep_cache:
                caches[f"c{i}_out"] = cin
                caches[f"ln{i}"] = (a, lnc2)

        pool = np.concatenate([h.mean(axis=1), h[:, -1, :]], axis=1)

        s = static @ p["s0_W"] + p["s0_b"]
        sa = gelu(s)
        s1 = sa @ p["s1_W"] + p["s1_b"]
        s2 = gelu(s1)

        joint = np.concatenate([pool, s2], axis=1)
        hh = joint @ p["h0_W"] + p["h0_b"]
        ha = gelu(hh)
        out = ha @ p["h1_W"] + p["h1_b"]

        if keep_cache:
            caches.update(
                {
                    "pool": pool,
                    "static": static,
                    "s": s,
                    "sa": sa,
                    "s1": s1,
                    "s2": s2,
                    "joint": joint,
                    "hh": hh,
                    "ha": ha,
                }
            )
        return out, caches

    # ---- backward -----------------------------------------------------
    def backward(self, dout: np.ndarray, caches: dict) -> dict:
        p = self.params
        g: dict[str, np.ndarray] = {}
        g["h1_W"] = caches["ha"].T @ dout
        g["h1_b"] = dout.sum(axis=0)
        dha = dout @ p["h1_W"].T
        dhh = dha * gelu_grad(caches["hh"])
        g["h0_W"] = caches["joint"].T @ dhh
        g["h0_b"] = dhh.sum(axis=0)
        djoint = dhh @ p["h0_W"].T
        n_pool = caches["pool"].shape[1]
        dpool = djoint[:, :n_pool]
        ds2 = djoint[:, n_pool:]

        ds1 = ds2 * gelu_grad(caches["s1"])
        g["s1_W"] = caches["sa"].T @ ds1
        g["s1_b"] = ds1.sum(axis=0)
        dsa = ds1 @ p["s1_W"].T
        ds = dsa * gelu_grad(caches["s"])
        g["s0_W"] = caches["static"].T @ ds
        g["s0_b"] = ds.sum(axis=0)

        w = self.width
        t_steps = caches["ln1"][0].shape[1]
        dh = np.broadcast_to(dpool[:, None, :w] / t_steps, (dpool.shape[0], t_steps, w)).copy()
        dh[:, -1, :] += dpool[:, w:]

        for i in (4, 3, 2, 1):
            a, lnc = caches[f"ln{i}"]
            dres = dh
            dln = self._layernorm_backward(dh, a, lnc)
            dcin = dln * gelu_grad(caches[f"c{i}_out"])
            dx, dW = self._conv_backward(dcin, caches[f"c{i}"])
            g[f"c{i}_W"] = dW
            g[f"c{i}_b"] = dcin.sum(axis=(0, 1))
            dh = dres + dx

        a0, lnc0 = caches["ln0"]
        dln0 = self._layernorm_backward(dh, a0, lnc0)
        da0 = dln0 * gelu_grad(caches["c0_out"])
        _dx0, dW0 = self._conv_backward(da0, caches["c0"])
        g["c0_W"] = dW0
        g["c0_b"] = da0.sum(axis=(0, 1))
        return g

    # ---- persistence --------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, **{f"p_{k}": v for k, v in self.params.items()}, **{f"a_{k}": v for k, v in self.attrs.items()})

    @classmethod
    def load(cls, path: str) -> "FusionNet":
        z = np.load(path, allow_pickle=False)
        net = cls()
        net.params = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
        net.attrs = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
        return net

    def predict_logits(self, seq: np.ndarray, static: np.ndarray, batch: int = 4096) -> np.ndarray:
        """Inference in float32 (about 2x faster than float64 on this CPU)."""
        if getattr(self, "_params32", None) is None or self._params32.keys() != self.params.keys():
            self._params32 = {k: v.astype(np.float32) for k, v in self.params.items()}
        saved = self.params
        self.params = self._params32
        try:
            seq = np.asarray(seq, dtype=np.float32)
            static = np.asarray(static, dtype=np.float32)
            out = []
            for i in range(0, len(seq), batch):
                logits, _ = self.forward(seq[i : i + batch], static[i : i + batch])
                out.append(logits)
            return np.concatenate(out, axis=0).astype(np.float64)
        finally:
            self.params = saved


class Adam:
    def __init__(self, params: dict, lr: float = 1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.t += 1
        for k, grad in grads.items():
            if k not in params:
                continue
            if self.wd:
                grad = grad + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grad
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grad * grad)
            mhat = self.m[k] / (1 - self.b1**self.t)
            vhat = self.v[k] / (1 - self.b2**self.t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + self.eps)
