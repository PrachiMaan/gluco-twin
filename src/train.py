"""GlucoTwin Phase 2: training driver.

Stages (each can run on its own to stay inside a laptop-sized time budget):

  python src/train.py --trees      # three LightGBM event models
  python src/train.py --forecast   # three LightGBM glucose forecasters
  python src/train.py --net        # the two-branch fusion network

Everything is seeded. Split membership is written to results/splits.json so
evaluation and the web export use exactly the same patients.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from features import (
    FEATURE_NAMES,
    N_FEATURES,
    build_dataset,
    load_ehr,
    load_streams,
    patient_split,
)
from models import (
    CHANNELS,
    SEQ_LEN,
    Adam,
    FusionNet,
    IsotonicCalibrator,
    train_lgbm,
    train_lgbm_regressor,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
SEED = 42
TRAIN_STRIDE = 6     # 30-minute sampling for training windows
VAL_STRIDE = 3       # 15-minute sampling for validation


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def bce_with_logits(logits: np.ndarray, y: np.ndarray, pos_weight: float):
    """Numerically stable weighted BCE; returns (loss, dlogits)."""
    n = len(y)
    p = _sigmoid(logits)
    eps = 1e-12
    loss = -np.mean(pos_weight * y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    dlogits = (pos_weight * y * (p - 1.0) + (1 - y) * p) / n
    return float(loss), dlogits


def _split_path() -> Path:
    return RESULT_DIR / "splits.json"


def ensure_splits() -> dict[str, list[int]]:
    RESULT_DIR.mkdir(exist_ok=True)
    ehr = load_ehr()
    splits = patient_split(len(ehr))
    payload = {k: [int(x) for x in v] for k, v in splits.items()}
    _split_path().write_text(json.dumps(payload, indent=2))
    return payload


def load_splits() -> dict[str, np.ndarray]:
    path = _split_path()
    if not path.exists():
        ensure_splits()
    payload = json.loads(path.read_text())
    return {k: np.array(v) for k, v in payload.items()}


# ------------------------------------------------------------------ trees


def train_trees() -> None:
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    ehr = load_ehr()
    streams = load_streams()
    splits = load_splits()

    ds_tr = build_dataset(ehr, streams, splits["train"], stride=TRAIN_STRIDE, include="all")
    ds_va = build_dataset(ehr, streams, splits["val"], stride=VAL_STRIDE, include="all")
    print(
        f"[train] train windows {len(ds_tr['X'])} ({ds_tr['y_event'].mean():.1%} positive), "
        f"val windows {len(ds_va['X'])}"
    )

    pos_weight = float((1 - ds_tr["y_event"].mean()) / max(ds_tr["y_event"].mean(), 1e-6))
    summary: dict = {"pos_weight": pos_weight, "models": {}}

    for variant in ("cgm", "ehr", "all"):
        from features import STATIC_FEATURES

        if variant == "all":
            cols = list(range(N_FEATURES))
        elif variant == "cgm":
            cols = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("g_")]
        else:  # ehr: static record fields only, no clock context
            cols = [i for i, n in enumerate(FEATURE_NAMES) if n in set(STATIC_FEATURES)]
        names = [FEATURE_NAMES[i] for i in cols]
        booster = train_lgbm(ds_tr["X"][:, cols], ds_tr["y_event"], names, pos_weight=pos_weight)
        name = {"cgm": "lgbm_cgm", "ehr": "lgbm_ehr", "all": "lgbm_fused"}[variant]
        booster.save_model(str(MODEL_DIR / f"{name}.txt"))

        val_p = booster.predict(ds_va["X"][:, cols])
        cal = IsotonicCalibrator().fit(val_p, ds_va["y_event"])
        with open(MODEL_DIR / f"{name}_cal.pkl", "wb") as fh:
            pickle.dump(cal, fh)

        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(ds_va["y_event"], val_p))
        summary["models"][name] = {"val_auc": auc, "n_features": len(cols)}
        print(f"[train] {name}: val AUROC {auc:.4f} ({len(cols)} features)")

    (MODEL_DIR / "tree_meta.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "variants": {
                    "cgm": [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("g_")],
                    "ehr": [
                        i for i, n in enumerate(FEATURE_NAMES) if n in set(__import__("features").STATIC_FEATURES)
                    ],
                    "all": list(range(N_FEATURES)),
                },
                "train_stride": TRAIN_STRIDE,
                "val_stride": VAL_STRIDE,
            },
            indent=2,
        )
    )
    (RESULT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] trees done in {time.time() - t0:.1f}s")


def train_forecast() -> None:
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    ehr = load_ehr()
    streams = load_streams()
    splits = load_splits()
    ds_tr = build_dataset(ehr, streams, splits["train"], stride=TRAIN_STRIDE, include="all")
    for horizon in (30, 60, 120):
        y = ds_tr[f"y{horizon}"]
        booster = train_lgbm_regressor(ds_tr["X"], y, FEATURE_NAMES)
        booster.save_model(str(MODEL_DIR / f"lgbm_f{horizon}.txt"))
        print(f"[train] forecast +{horizon}min trained ({len(y)} rows)")
    print(f"[train] forecasters done in {time.time() - t0:.1f}s")


# -------------------------------------------------------------------- net


def gather_sequences(streams: dict, pid: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Raw (B, 36, 5) tensor: glucose, hr, hrv, steps, sleep stage (10-min steps)."""
    offsets = np.arange(SEQ_LEN - 1, -1, -1) * 2
    idx = step[:, None] - offsets[None, :]
    idx = np.clip(idx, 0, streams["glucose"].shape[1] - 1)
    rows = pid[:, None]
    g = streams["glucose"][rows, idx].astype(np.float32)
    g[g < 0] = np.nan
    # forward/backward fill within the window
    for i in range(g.shape[0]):
        row = g[i]
        if np.isnan(row).all():
            row[:] = 120.0
            continue
        mask = np.isnan(row)
        if mask.any():
            row[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), row[~mask])
    hr = streams["hr"][rows, idx].astype(np.float32)
    hrv = streams["hrv"][rows, idx].astype(np.float32) / 10.0
    steps = streams["steps"][rows, idx].astype(np.float32)
    stage = streams["stage"][rows, idx].astype(np.float32)
    stage = np.where(stage == 1, 0.25, np.where(stage == 2, 0.5, np.where(stage == 3, 0.8, np.where(stage == 4, 1.0, 0.0))))
    seq = np.stack([g, hr, hrv, steps, stage], axis=-1)
    return seq.astype(np.float32)


def static_matrix(ehr, pid: np.ndarray) -> np.ndarray:
    from features import STATIC_FEATURES

    rows = ehr.iloc[pid]
    vals = {
        "age": rows["age"].to_numpy(float),
        "sex_m": (rows["sex"] == "M").to_numpy(float),
        "bmi": rows["bmi"].to_numpy(float),
        "waist_cm": rows["waist_cm"].to_numpy(float),
        "veg_diet": (rows["diet_type"] == "veg").to_numpy(float),
        "family_history": rows["family_history"].to_numpy(float),
        "years_since_dx": rows["years_since_dx"].to_numpy(float),
        "hba1c": rows["hba1c"].to_numpy(float),
        "fpg": rows["fpg_mgdl"].to_numpy(float),
        "egfr": rows["egfr"].to_numpy(float),
        "ldl": rows["ldl"].to_numpy(float),
        "triglycerides": rows["triglycerides"].to_numpy(float),
        "systolic_bp": rows["systolic_bp"].to_numpy(float),
        "metformin": rows["metformin"].to_numpy(float),
        "sulfonylurea": rows["sulfonylurea"].to_numpy(float),
        "dpp4": rows["dpp4"].to_numpy(float),
        "insulin": rows["insulin"].to_numpy(float),
        "hypertension": rows["hypertension"].to_numpy(float),
        "tcf7l2_dose": rows["tcf7l2"].map({"CC": 0.0, "CT": 1.0, "TT": 2.0}).to_numpy(float),
        "prior_hypo_count": rows["prior_hypo_count"].to_numpy(float),
    }
    return np.stack([vals[n] for n in STATIC_FEATURES], axis=1).astype(np.float32)


def train_net(epochs: int = 2, batch: int = 128, samples_per_epoch: int = 110_000) -> None:
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    ehr = load_ehr()
    streams = load_streams()
    splits = load_splits()
    rng = np.random.default_rng(SEED)

    from features import _features_for_patient

    def label_rows(patient_idx: np.ndarray, stride: int):
        rows = []
        for p in patient_idx:
            out = _features_for_patient(
                streams["glucose"][p], streams["hr"][p], streams["hrv"][p],
                streams["steps"][p], streams["stage"][p], ehr.iloc[int(p)],
            )
            idx = np.arange(0, streams["glucose"].shape[1], stride)
            idx = idx[out["valid"][idx]]
            rows.append(
                (
                    np.full(len(idx), int(p), dtype=np.int32),
                    idx.astype(np.int32),
                    (out["y_event"][idx] > 0.5).astype(np.float32),
                    out["y30"][idx],
                    out["y60"][idx],
                    out["y120"][idx],
                )
            )
        pid = np.concatenate([r[0] for r in rows])
        step = np.concatenate([r[1] for r in rows])
        y = np.concatenate([r[2] for r in rows])
        yf = np.stack([np.concatenate([r[3] for r in rows]),
                       np.concatenate([r[4] for r in rows]),
                       np.concatenate([r[5] for r in rows])], axis=1)
        return pid, step, y, yf

    pid_tr, step_tr, y_tr, yf_tr = label_rows(splits["train"], TRAIN_STRIDE)
    print(f"[net] train windows {len(pid_tr)} ({y_tr.mean():.1%} positive)")

    # channel / target scalers from the training split
    sample = rng.choice(len(pid_tr), size=min(4000, len(pid_tr)), replace=False)
    seq_s = gather_sequences(streams, pid_tr[sample], step_tr[sample])
    chan_mean = seq_s.mean(axis=(0, 1))
    chan_std = seq_s.std(axis=(0, 1)) + 1e-6
    static_all = static_matrix(ehr, pid_tr[sample])
    st_mean = static_all.mean(axis=0)
    st_std = static_all.std(axis=0) + 1e-6
    fc_mean = yf_tr.mean(axis=0)
    fc_std = yf_tr.std(axis=0) + 1e-6

    net = FusionNet(static_dim=static_all.shape[1], width=16, mlp_hidden=32, seq_len=SEQ_LEN, seed=SEED)
    net.init()
    net.attrs = {
        "chan_mean": chan_mean, "chan_std": chan_std,
        "static_mean": st_mean, "static_std": st_std,
        "fc_mean": fc_mean, "fc_std": fc_std,
    }
    opt = Adam(net.params, lr=2e-3)
    pos_weight = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))

    # fixed validation sample for early stopping
    pid_va, step_va, y_va, yf_va = label_rows(splits["val"], VAL_STRIDE)
    va_idx = rng.choice(len(pid_va), size=min(15000, len(pid_va)), replace=False)
    seq_va = gather_sequences(streams, pid_va[va_idx], step_va[va_idx])
    seq_va = (seq_va - chan_mean) / chan_std
    st_va = (static_matrix(ehr, pid_va[va_idx]) - st_mean) / st_std
    y_va_s = y_va[va_idx]

    from sklearn.metrics import roc_auc_score

    best_auc, best_params = -1.0, None
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(pid_tr))
        n_batches = max(len(order) // batch, 1)
        n_batches = min(n_batches, samples_per_epoch // batch)
        running = 0.0
        lr = 2e-3 * (0.5**epoch)
        for b in range(n_batches):
            sel = order[b * batch : (b + 1) * batch]
            seq = gather_sequences(streams, pid_tr[sel], step_tr[sel])
            seq = (seq - chan_mean) / chan_std
            stat = (static_matrix(ehr, pid_tr[sel]) - st_mean) / st_std
            yb = y_tr[sel]
            tgt = (yf_tr[sel] - fc_mean) / fc_std

            logits, caches = net.forward(seq, stat, keep_cache=True)
            loss_bce, dlogit = bce_with_logits(logits[:, 0], yb, pos_weight)
            resid = logits[:, 1:] - tgt
            loss_fc = float(np.mean(resid**2))
            dout = np.zeros_like(logits)
            dout[:, 0] = dlogit
            dout[:, 1:] = 0.3 * 2.0 * resid / resid.size
            grads = net.backward(dout, caches)
            opt.step(net.params, grads, lr=lr)
            running += loss_bce + 0.3 * loss_fc
        val_logits = net.predict_logits(seq_va, st_va)
        auc = float(roc_auc_score(y_va_s, val_logits[:, 0]))
        rmse = float(np.mean((val_logits[:, 1:] * fc_std + fc_mean - (yf_va[va_idx])) ** 2) ** 0.5)
        history.append({"epoch": epoch + 1, "train_loss": running / n_batches, "val_auc": auc, "val_forecast_rmse": rmse})
        print(f"[net] epoch {epoch + 1}: loss {running / n_batches:.4f} val AUROC {auc:.4f} forecast RMSE {rmse:.1f}")
        if auc > best_auc:
            best_auc = auc
            best_params = {k: v.copy() for k, v in net.params.items()}

    if best_params is not None:
        net.params = best_params

    # temperature scaling on the validation logits
    logits = net.predict_logits(seq_va, st_va)[:, 0]
    grid = np.linspace(0.3, 4.0, 60)
    nll = []
    for T in grid:
        p = _sigmoid(logits / T)
        nll.append(-np.mean(y_va_s * np.log(p + 1e-9) + (1 - y_va_s) * np.log(1 - p + 1e-9)))
    temperature = float(grid[int(np.argmin(nll))])
    net.attrs["temperature"] = np.array([temperature])
    net.attrs["static_dim"] = np.array([static_all.shape[1]])
    net.save(str(MODEL_DIR / "fusion_net.npz"))
    (RESULT_DIR / "net_history.json").write_text(json.dumps({"history": history, "temperature": temperature}, indent=2))
    print(f"[net] saved (temperature {temperature:.2f}, best val AUROC {best_auc:.4f}) in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", action="store_true")
    parser.add_argument("--forecast", action="store_true")
    parser.add_argument("--net", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    ensure_splits()
    if args.all or args.trees:
        train_trees()
    if args.all or args.forecast:
        train_forecast()
    if args.all or args.net:
        train_net()


if __name__ == "__main__":
    main()
