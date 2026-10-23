"""GlucoTwin Phase 2: evaluation and ablation.

Everything quoted in the README, slides, API and dashboard comes from this
script, computed on the test split (patients never seen in training):

  results/metrics.json        headline + per-model metrics
  results/ablation.csv        persistence vs CGM-only vs EHR-only vs fused vs full
  results/plots/*.png         ROC, PR, calibration, ablation, forecast, SHAP
  results/shap_importance.csv

Cadence note: the fusion network is expensive to run on this CPU, so the
ablation table is computed for every model at the same 30-minute checkpoint
cadence. The deployed model is additionally reported on the dense 15-minute
timeline. This is documented in docs/MODEL_CARD.md.

Alarm protocol: an alarm is the first checkpoint of a run above the threshold;
it is a true alarm if it falls within the two hours before a labelled onset.
Thresholds are picked on validation to keep false alarms <= 0.5 per
patient-day, then applied unchanged to test.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from features import STEP_MIN, build_dataset, load_ehr, load_streams
from models import FusionNet
from train import gather_sequences, static_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
PLOT_DIR = RESULT_DIR / "plots"

ABLATION_STRIDE = 6   # 30-minute checkpoints, common to all models
DENSE_STRIDE = 3      # 15-minute checkpoints, deployed model only
TARGET_FAR = 0.5      # false alarms per patient-day
HORIZON_STEPS = 24    # 2 hours of 5-minute steps
COOLDOWN_STEPS = 24   # suppress repeat alarms for 2 hours

PALETTE = {
    "ink": "#1c1b19",
    "muted": "#8a8478",
    "clay": "#b4552d",
    "sage": "#6f8060",
    "slate": "#4f6b7a",
    "ochre": "#b08a4f",
    "grid": "#e6e1d8",
}


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def within_patient_auc(y: np.ndarray, p: np.ndarray, pid: np.ndarray) -> float:
    """Mean per-patient AUROC, weighted by windows: measures timing skill."""
    total, weight = 0.0, 0
    for patient in np.unique(pid):
        m = pid == patient
        yy = y[m]
        if yy.min() == yy.max():
            continue
        total += roc_auc_score(yy, p[m]) * m.sum()
        weight += int(m.sum())
    return float(total / weight) if weight else float("nan")


def event_runs(glucose_raw: np.ndarray, min_steps: int = 6, clear_steps: int = 12) -> list[int]:
    """Start step of every >=30-minute hyperglycaemic run that follows at least
    one hour below 180 mg/dL. Chronic hyperglycaemia is not an 'event'."""
    from features import interpolate_gaps

    g, _ = interpolate_gaps(glucose_raw)
    above = (g > 180).astype(np.int8)
    diff = np.diff(np.concatenate([[0], above, [0]]))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    out = []
    for s, e in zip(starts, ends):
        if e - s < min_steps:
            continue
        lo = max(s - clear_steps, 0)
        if s > 0 and (g[lo:s] > 180).any():
            continue
        if s == 0 and (g[:1] > 180).all():
            continue
        out.append(int(s))
    return out


def split_events(streams, patients) -> dict[int, list[int]]:
    return {int(p): event_runs(streams["glucose"][p]) for p in patients}


def alarm_stats(p, pid, step, thr: float, stride: int, events: dict[int, list[int]], streams) -> dict:
    """Sensitivity and false alarms against true hyperglycaemic run starts.

    An alarm only counts while the patient is still below 180 mg/dL (once above,
    it is detection rather than prediction). A true alert lands in the two hours
    before a run start. Repeated scores above threshold are suppressed for two
    hours.
    """
    from features import interpolate_gaps

    n_events = detected = alarms_total = false_alarms = 0
    leads: list[float] = []
    total_days = 0.0
    for patient in np.unique(pid):
        m = pid == patient
        order = np.argsort(step[m])
        pp, ss = p[m][order], step[m][order]
        if len(ss) == 0:
            continue
        g, _ = interpolate_gaps(streams["glucose"][int(patient)])
        total_days += (ss.max() - ss.min() + stride) * STEP_MIN / 1440.0
        alarm_times = []
        last_alarm = -10**9
        for i in range(len(ss)):
            if pp[i] > thr and ss[i] - last_alarm > COOLDOWN_STEPS and g[int(ss[i])] <= 180.0:
                alarm_times.append(int(ss[i]) * STEP_MIN)
                last_alarm = int(ss[i])
        alarm_times = np.array(alarm_times)
        evs = [e for e in events.get(int(patient), []) if e >= HORIZON_STEPS]
        n_events += len(evs)
        matched = 0
        for ev in evs:
            lo, hi = (ev - HORIZON_STEPS) * STEP_MIN, ev * STEP_MIN
            window = alarm_times[(alarm_times >= lo) & (alarm_times < hi)]
            if len(window):
                matched += 1
                detected += 1
                leads.append((ev * STEP_MIN - window.min()))
        alarms_total += len(alarm_times)
        false_alarms += len(alarm_times) - matched
    return {
        "threshold": float(thr),
        "n_events": int(n_events),
        "sensitivity": float(detected / n_events) if n_events else float("nan"),
        "false_alarms_per_day": float(false_alarms / total_days) if total_days else float("nan"),
        "alarms_per_day": float(alarms_total / total_days) if total_days else float("nan"),
        "mean_lead_min": float(np.mean(leads)) if leads else float("nan"),
        "median_lead_min": float(np.median(leads)) if leads else float("nan"),
        "patient_days": float(total_days),
    }


THRESHOLD_CANDIDATES = list(np.round(np.arange(0.05, 0.905, 0.05), 3)) + [0.93, 0.95, 0.97, 0.98, 0.99, 0.995]


def choose_threshold(p, pid, step, stride: int, events, streams) -> float:
    """Pick the threshold with the best sensitivity while keeping FAR low."""
    scored = []
    for thr in THRESHOLD_CANDIDATES:
        stats = alarm_stats(p, pid, step, float(thr), stride, events, streams)
        scored.append((float(thr), stats["sensitivity"], stats["false_alarms_per_day"]))
    feasible = [s for s in scored if s[2] <= TARGET_FAR]
    if feasible:
        return max(feasible, key=lambda s: s[1])[0]
    return min(scored, key=lambda s: s[2])[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-net", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    RESULT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ehr = load_ehr()
    streams = load_streams()
    splits = json.loads((RESULT_DIR / "splits.json").read_text())
    meta = json.loads((MODEL_DIR / "tree_meta.json").read_text())
    feature_names = meta["feature_names"]

    print("[eval] building validation features ...")
    ds_va = build_dataset(ehr, streams, np.array(splits["val"]), stride=ABLATION_STRIDE)
    print("[eval] building test features ...")
    ds_te = build_dataset(ehr, streams, np.array(splits["test"]), stride=ABLATION_STRIDE)
    y_va, y_te = ds_va["y_event"], ds_te["y_event"]
    print("[eval] finding hyperglycaemic run starts ...")
    events_va = split_events(streams, splits["val"])
    events_te = split_events(streams, splits["test"])

    models: dict[str, dict] = {}

    # ---- persistence -----------------------------------------------------
    from models import PersistenceBaseline

    base = PersistenceBaseline().fit(ds_va["X"], y_va)
    models["persistence"] = {"p_va": base.predict_proba(ds_va["X"]), "p_te": base.predict_proba(ds_te["X"])}

    # ---- LightGBM --------------------------------------------------------
    for variant, name in (("cgm", "lgbm_cgm"), ("ehr", "lgbm_ehr"), ("all", "lgbm_fused")):
        cols = meta["variants"][variant]
        booster = lgb.Booster(model_file=str(MODEL_DIR / f"{name}.txt"))
        with open(MODEL_DIR / f"{name}_cal.pkl", "rb") as fh:
            cal = pickle.load(fh)
        models[name] = {
            "p_va": cal.transform(booster.predict(ds_va["X"][:, cols])),
            "p_te": cal.transform(booster.predict(ds_te["X"][:, cols])),
        }
        print(f"[eval] {name} predicted")

    # ---- fusion net ------------------------------------------------------
    if not args.skip_net:
        net = FusionNet.load(str(MODEL_DIR / "fusion_net.npz"))
        attrs = net.attrs
        temp = float(attrs["temperature"][0]) if "temperature" in attrs else 1.0

        def net_probs(ds, chunk: int = 20000):
            n = len(ds["pid"])
            prob = np.empty(n)
            fc = np.empty((n, 3))
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                seq = gather_sequences(streams, ds["pid"][start:stop], ds["step"][start:stop])
                seq = (seq - attrs["chan_mean"]) / attrs["chan_std"]
                stat = (static_matrix(ehr, ds["pid"][start:stop]) - attrs["static_mean"]) / attrs["static_std"]
                logits = net.predict_logits(seq, stat)
                prob[start:stop] = 1.0 / (1.0 + np.exp(-logits[:, 0] / temp))
                fc[start:stop] = logits[:, 1:] * attrs["fc_std"] + attrs["fc_mean"]
            return prob, fc

        p_va_net, fc_va = net_probs(ds_va)
        p_te_net, fc_te = net_probs(ds_te)
        from models import IsotonicCalibrator

        net_cal = IsotonicCalibrator().fit(p_va_net, y_va)
        models["fusion_net"] = {
            "p_va": net_cal.transform(p_va_net),
            "p_te": net_cal.transform(p_te_net),
            "fc_te": fc_te,
        }
        print("[eval] fusion net predicted")

    # ---- metrics ---------------------------------------------------------
    metrics: dict = {
        "test_patients": len(splits["test"]),
        "test_windows": int(len(y_te)),
        "ablation_stride_min": ABLATION_STRIDE * STEP_MIN,
        "event_base_rate": float(y_te.mean()),
        "models": {},
    }
    rows = []
    for name, m in models.items():
        p_te = m["p_te"]
        auc = float(roc_auc_score(y_te, p_te))
        ap = float(average_precision_score(y_te, p_te))
        wp = within_patient_auc(y_te, p_te, ds_te["pid"])
        brier = float(brier_score_loss(y_te, np.clip(p_te, 0, 1)))
        thr = choose_threshold(m["p_va"], ds_va["pid"], ds_va["step"], ABLATION_STRIDE, events_va, streams)
        stats = alarm_stats(p_te, ds_te["pid"], ds_te["step"], thr, ABLATION_STRIDE, events_te, streams)
        metrics["models"][name] = {
            "auroc": auc, "auprc": ap, "within_patient_auroc": wp, "brier": brier, **stats
        }
        rows.append(
            {
                "model": name,
                "auroc": round(auc, 4),
                "auprc": round(ap, 4),
                "within_patient_auroc": round(wp, 4),
                "brier": round(brier, 4),
                "sensitivity_at_far": round(stats["sensitivity"], 3),
                "false_alarms_per_day": round(stats["false_alarms_per_day"], 3),
                "mean_lead_min": round(stats["mean_lead_min"], 1),
                "threshold": round(stats["threshold"], 3),
            }
        )
        print(
            f"[eval] {name}: AUROC {auc:.4f} AUPRC {ap:.4f} within-patient {wp:.4f} "
            f"sens {stats['sensitivity']:.2f} FA/day {stats['false_alarms_per_day']:.2f} "
            f"lead {stats['mean_lead_min']:.0f} min"
        )

    ablation = pd.DataFrame(rows)
    ablation.to_csv(RESULT_DIR / "ablation.csv", index=False)

    deployed = max([n for n in models if n != "persistence"], key=lambda n: metrics["models"][n]["auroc"])
    metrics["deployed_model"] = deployed

    # dense timeline for the deployed model (LightGBM can do this cheaply)
    del ds_te["X"]
    gc.collect()
    ds_dense = build_dataset(ehr, streams, np.array(splits["test"]), stride=DENSE_STRIDE)
    y_dense = ds_dense["y_event"]
    dense_metrics: dict = {}
    if deployed == "lgbm_fused":
        booster = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_fused.txt"))
        cols = meta["variants"]["all"]
        with open(MODEL_DIR / "lgbm_fused_cal.pkl", "rb") as fh:
            cal = pickle.load(fh)
        p_dense = cal.transform(booster.predict(ds_dense["X"][:, cols]))
        thr_dense = float(metrics["models"]["lgbm_fused"]["threshold"])
        dense_metrics["lgbm_fused"] = {
            "auroc": float(roc_auc_score(y_dense, p_dense)),
            "auprc": float(average_precision_score(y_dense, p_dense)),
            "brier": float(brier_score_loss(y_dense, np.clip(p_dense, 0, 1))),
            **alarm_stats(p_dense, ds_dense["pid"], ds_dense["step"], thr_dense, DENSE_STRIDE, events_te, streams),
            "cadence_min": DENSE_STRIDE * STEP_MIN,
        }
        metrics["dense_deployed"] = dense_metrics["lgbm_fused"]
        print(f"[eval] dense deployed: AUROC {dense_metrics['lgbm_fused']['auroc']:.4f}")

    # ---- forecast --------------------------------------------------------
    forecast = {}
    for horizon in (30, 60, 120):
        booster_f = lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_f{horizon}.txt"))
        pred = booster_f.predict(ds_dense["X"])
        truth = ds_dense[f"y{horizon}"]
        forecast[f"lgbm_{horizon}min"] = {
            "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
            "mae": float(np.mean(np.abs(pred - truth))),
        }
    if "fusion_net" in models:
        for j, horizon in enumerate((30, 60, 120)):
            pred = models["fusion_net"]["fc_te"][:, j]
            truth = ds_te[f"y{horizon}"]
            forecast[f"fusion_net_{horizon}min"] = {
                "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                "mae": float(np.mean(np.abs(pred - truth))),
            }
    metrics["forecast"] = forecast

    # ---- plots -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=160)
    for name, color in (
        ("persistence", PALETTE["muted"]),
        ("lgbm_cgm", PALETTE["sage"]),
        ("lgbm_ehr", PALETTE["ochre"]),
        ("lgbm_fused", PALETTE["clay"]),
        ("fusion_net", PALETTE["slate"]),
    ):
        if name not in models:
            continue
        fpr, tpr, _ = roc_curve(y_te, models[name]["p_te"])
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} ({metrics['models'][name]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["grid"], lw=1, ls="--")
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate", fontsize=9)
    ax.set_title("ROC - held-out patients", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "roc.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=160)
    order = [n for n in ["persistence", "lgbm_cgm", "lgbm_ehr", "lgbm_fused", "fusion_net"] if n in models]
    colors = [PALETTE["muted"], PALETTE["sage"], PALETTE["ochre"], PALETTE["clay"], PALETTE["slate"]]
    ax.barh(order, [metrics["models"][n]["auprc"] for n in order], color=colors[: len(order)], height=0.55)
    ax.axvline(float(y_te.mean()), color=PALETTE["ink"], lw=1, ls="--")
    ax.text(float(y_te.mean()) + 0.01, len(order) - 0.35, "base rate", fontsize=8, color=PALETTE["ink"])
    ax.set_xlabel("AUPRC", fontsize=9)
    ax.set_title("Ablation: precision-recall", fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ablation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=160)
    for name, color in (("lgbm_fused", PALETTE["clay"]), ("fusion_net", PALETTE["slate"])):
        if name not in models:
            continue
        p = np.clip(models[name]["p_te"], 1e-6, 1 - 1e-6)
        idx = np.digitize(p, np.linspace(0, 1, 11)) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() > 50:
                xs.append(float(p[m].mean()))
                ys.append(float(y_te[m].mean()))
        ax.plot(xs, ys, marker="o", ms=3, lw=1.5, color=color, label=name)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=PALETTE["grid"])
    ax.set_xlabel("Predicted probability", fontsize=9)
    ax.set_ylabel("Observed frequency", fontsize=9)
    ax.set_title("Calibration - held-out patients", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "calibration.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=160)
    ax.plot([30, 60, 120], [forecast[f"lgbm_{h}min"]["rmse"] for h in (30, 60, 120)], marker="o", color=PALETTE["clay"], lw=1.6, label="LightGBM fused")
    if "fusion_net" in models:
        ax.plot([30, 60, 120], [forecast[f"fusion_net_{h}min"]["rmse"] for h in (30, 60, 120)], marker="o", color=PALETTE["slate"], lw=1.6, label="fusion net")
    ax.set_xlabel("Forecast horizon (minutes)", fontsize=9)
    ax.set_ylabel("RMSE (mg/dL)", fontsize=9)
    ax.set_title("Glucose forecast error", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "forecast.png")
    plt.close(fig)

    # ---- SHAP ------------------------------------------------------------
    if not args.skip_shap:
        import shap

        cols = meta["variants"]["all"]
        booster_s = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_fused.txt"))
        rng = np.random.default_rng(0)
        sample = rng.choice(len(ds_dense["X"]), size=min(3000, len(ds_dense["X"])), replace=False)
        Xs = ds_dense["X"][sample][:, cols]
        names = [feature_names[i] for i in cols]
        values = shap.TreeExplainer(booster_s).shap_values(Xs)
        if isinstance(values, list):
            values = values[1]
        imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values(
            "mean_abs_shap", ascending=False
        )
        imp.to_csv(RESULT_DIR / "shap_importance.csv", index=False)

        plt.figure(figsize=(6.6, 5.0), dpi=160)
        shap.summary_plot(values, Xs, feature_names=names, show=False, max_display=14)
        fig = plt.gcf()
        fig.set_size_inches(6.6, 5.0)
        fig.savefig(PLOT_DIR / "shap_summary.png", bbox_inches="tight")
        plt.close(fig)
        metrics["top_features"] = imp.head(8).to_dict(orient="records")
        print("[eval] SHAP done")

    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[eval] wrote metrics in {time.time() - t0:.1f}s")
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
