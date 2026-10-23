"""Phase 2 tests: feature pipeline + fusion network gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import HORIZON, N_FEATURES, _features_for_patient, build_dataset, load_ehr, load_streams, patient_split  # noqa: E402
from models import FusionNet, PersistenceBaseline  # noqa: E402


# --------------------------------------------------------------- features

@pytest.fixture(scope="module")
def small_dataset():
    ehr = load_ehr()
    streams = load_streams()
    idx = np.arange(0, 6)
    ds = build_dataset(ehr, streams, idx, stride=12)
    return ds, ehr, streams


def test_dataset_shapes_and_labels(small_dataset) -> None:
    ds, _, _ = small_dataset
    assert ds["X"].shape[1] == N_FEATURES
    assert set(np.unique(ds["y_event"])) <= {0, 1}
    assert len(ds["X"]) == len(ds["y_event"]) == len(ds["pid"])
    assert 0.05 < ds["y_event"].mean() < 0.95
    assert np.isfinite(ds["X"]).all()
    assert np.isfinite(ds["y30"]).all()


def test_event_label_matches_bruteforce(small_dataset) -> None:
    _, ehr, streams = small_dataset
    g = streams["glucose"][0].astype(np.float32)
    g[g < 0] = np.nan
    # fill gaps the same way the pipeline does
    idx = np.arange(len(g))
    g[np.isnan(g)] = np.interp(idx[np.isnan(g)], idx[~np.isnan(g)], g[~np.isnan(g)])
    out = _features_for_patient(
        streams["glucose"][0],
        streams["hr"][0],
        streams["hrv"][0],
        streams["steps"][0],
        streams["stage"][0],
        ehr.iloc[0],
    )
    above = g > 180
    for t in range(50, 400, 37):
        hit = False
        # a step s qualifies when the 30-minute window ENDING at s is all above 180
        for s in range(t + 1, min(t + HORIZON + 1, len(g) - 1)):
            if above[max(s - 5, 0) : s + 1].all() and s >= 5:
                hit = True
                break
        assert bool(out["y_event"][t] > 0.5) == hit, f"label mismatch at {t}"


def test_no_future_leakage_in_features(small_dataset) -> None:
    """Perturbing the future must not change features at earlier timesteps."""
    _, ehr, streams = small_dataset
    g = streams["glucose"][1].copy()
    out_a = _features_for_patient(g, streams["hr"][1], streams["hrv"][1], streams["steps"][1], streams["stage"][1], ehr.iloc[1])
    g2 = g.copy()
    cutoff = 2000
    g2[cutoff:] = np.where(g2[cutoff:] > 0, np.clip(g2[cutoff:] + 60, 40, 450), g2[cutoff:])
    out_b = _features_for_patient(g2, streams["hr"][1], streams["hrv"][1], streams["steps"][1], streams["stage"][1], ehr.iloc[1])
    assert np.allclose(out_a["X"][: cutoff - 72], out_b["X"][: cutoff - 72], atol=1e-5)


def test_patient_split_is_disjoint() -> None:
    s = patient_split(100)
    assert len(s["train"]) == 60 and len(s["val"]) == 20 and len(s["test"]) == 20
    assert not (set(s["train"]) & set(s["val"]))
    assert not (set(s["train"]) & set(s["test"]))
    assert not (set(s["val"]) & set(s["test"]))


# --------------------------------------------------------------- network

def _tiny_net() -> FusionNet:
    net = FusionNet(static_dim=3, width=4, mlp_hidden=5, seq_len=7, seed=3)
    net.init()
    return net


def test_gradients_match_finite_differences() -> None:
    net = _tiny_net()
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(2, 7, 5))
    static = rng.normal(size=(2, 3))
    weights = rng.normal(size=(2, 4))

    def loss() -> float:
        out, _ = net.forward(seq, static)
        return float((out * weights).sum())

    logits, caches = net.forward(seq, static, keep_cache=True)
    grads = net.backward(weights, caches)

    checks = ["c0_W", "c2_W", "c4_W", "s0_W", "h1_W", "c1_b"]
    for name in checks:
        param = net.params[name]
        flat = param.reshape(-1)
        grad = grads[name].reshape(-1)
        for k in rng.choice(flat.size, size=min(4, flat.size), replace=False):
            eps = 1e-5
            orig = flat[k]
            flat[k] = orig + eps
            lp = loss()
            flat[k] = orig - eps
            lm = loss()
            flat[k] = orig
            numeric = (lp - lm) / (2 * eps)
            assert abs(numeric - grad[k]) <= 1e-5 + 1e-4 * abs(numeric), (
                f"{name}[{k}] analytic {grad[k]:.6f} numeric {numeric:.6f}"
            )


def test_network_output_shapes() -> None:
    net = _tiny_net()
    seq = np.zeros((3, 7, 5), dtype=np.float64)
    static = np.zeros((3, 3), dtype=np.float64)
    out, _ = net.forward(seq, static)
    assert out.shape == (3, 4)


def test_persistence_baseline_scores_rise_with_glucose() -> None:
    model = PersistenceBaseline().fit(np.zeros((3, 1)), np.zeros(3))
    X = np.zeros((3, N_FEATURES))
    X[:, 0] = [100.0, 180.0, 260.0]
    p = model.predict_proba(X)
    assert p[0] < p[1] < p[2]
    assert np.all((p >= 0) & (p <= 1))
