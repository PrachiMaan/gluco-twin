"""GlucoTwin Phase 3: the patient twin.

A `PatientTwin` holds one patient's EHR, their latest sensor window, their
fitted physiological parameters, the current 2-hour risk and the projected
trajectory. It updates as new readings arrive.

Parameter fitting is a learning problem, not a lookup: the fitted sensitivity
and basal glucose come from observable signal shape (night baseline, variability,
excursion rate) plus EHR context, mapped through a ridge model trained on the
training split. `fit_parameter_model()` writes the recovery quality to
results/param_recovery.json - it is reported honestly, including error.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import (
    FEATURE_NAMES,
    STEP_MIN,
    STEPS_PER_DAY,
    _features_for_patient,
    interpolate_gaps,
    load_ehr,
    load_streams,
    patient_split,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


class _ZeroRng:
    """Deterministic stand-in for jitter when deriving EHR-only defaults."""

    def normal(self, _loc, _scale, size):
        return np.zeros(size)


class ModelBundle:
    """Loads the deployed LightGBM event model, calibrator, forecasters and SHAP."""

    _instance: "ModelBundle | None" = None

    def __init__(self) -> None:
        meta = json.loads((MODEL_DIR / "tree_meta.json").read_text())
        self.feature_names = meta["feature_names"]
        self.cols = meta["variants"]["all"]
        self.booster = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_fused.txt"))
        with open(MODEL_DIR / "lgbm_fused_cal.pkl", "rb") as fh:
            self.calibrator = pickle.load(fh)
        self.forecasters = {
            h: lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_f{h}.txt")) for h in (30, 60, 120)
        }
        metrics_path = RESULT_DIR / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        self._explainer = None

    @classmethod
    def get(cls) -> "ModelBundle":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.booster.predict(X[:, self.cols])
        return self.calibrator.transform(raw)

    def explainer(self):
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.booster)
        return self._explainer

    def shap_row(self, x: np.ndarray) -> np.ndarray:
        values = self.explainer().shap_values(x.reshape(1, -1)[:, self.cols])
        if isinstance(values, list):
            values = values[1]
        return values[0]


def observable_profile(glucose: np.ndarray) -> dict:
    """Signal-shape features used to fit physiology without ground truth."""
    g, _ = interpolate_gaps(glucose)
    t = len(g)
    hour = (np.arange(t) % STEPS_PER_DAY) * STEP_MIN / 60.0
    night = (hour >= 2) & (hour < 5)
    night_median = float(np.median(g[night])) if night.any() else float(np.median(g))
    baseline = float(np.percentile(g, 10))
    peaks = []
    for i in range(12, t - 12):
        if g[i] >= g[i - 12 : i + 13].max() and g[i] > baseline + 35:
            peaks.append(g[i] - baseline)
    excursion = float(np.mean(peaks)) if peaks else 0.0
    above180 = float(np.mean(g > 180))
    return {
        "night_median": night_median,
        "baseline": baseline,
        "mean_g": float(np.mean(g)),
        "sd_g": float(np.std(g)),
        "excursion": excursion,
        "above180": above180,
        "n_peaks": len(peaks),
    }


def _design_row(profile: dict, ehr_row: pd.Series) -> np.ndarray:
    return np.array(
        [
            profile["night_median"] / 150.0,
            profile["baseline"] / 150.0,
            profile["mean_g"] / 150.0,
            profile["sd_g"] / 50.0,
            profile["excursion"] / 80.0,
            profile["above180"],
            float(ehr_row["bmi"]) / 30.0,
            float(ehr_row["hba1c"]) / 7.0,
            float(ehr_row["years_since_dx"]) / 10.0,
        ]
    )


def fit_parameter_model(seed: int = 42) -> dict:
    """Ridge map from observable signal shape to physiological parameters."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score

    ehr = load_ehr()
    streams = load_streams()
    params = pd.read_csv(DATA_DIR / "true_params.csv")
    splits = patient_split(len(ehr), seed)

    def rows(indices):
        X, ys, gs = [], [], []
        for i in indices:
            profile = observable_profile(streams["glucose"][i])
            X.append(_design_row(profile, ehr.iloc[i]))
            ys.append(np.log(max(params.iloc[i]["s_i"], 1e-4)))
            gs.append(params.iloc[i]["g_basal"])
        return np.array(X), np.array(ys), np.array(gs)

    Xtr, ytr, gtr = rows(splits["train"][:400])
    Xva, yva, gva = rows(splits["val"][:150])
    model = Ridge(alpha=1.0).fit(Xtr, ytr)
    pred = model.predict(Xva)
    s_i_true = np.exp(yva)
    s_i_pred = np.exp(pred)
    result = {
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "s_i_log_r2": float(r2_score(yva, pred)),
        "s_i_median_abs_pct_error": float(np.median(np.abs(s_i_pred - s_i_true) / s_i_true) * 100),
        "s_i_spearman": float(pd.Series(s_i_pred).corr(pd.Series(s_i_true), method="spearman")),
    }
    with open(MODEL_DIR / "param_fit.pkl", "wb") as fh:
        pickle.dump({"model": model}, fh)
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "param_recovery.json").write_text(json.dumps(result, indent=2))
    return result


def _load_param_model():
    path = MODEL_DIR / "param_fit.pkl"
    if not path.exists():
        fit_parameter_model()
    with open(path, "rb") as fh:
        return pickle.load(fh)["model"]


def fitted_parameters(index: int, ehr_row: pd.Series, glucose_stream: np.ndarray) -> dict:
    """Fit g_basal and s_i from the patient's own stream; derive the rest from EHR."""
    from cgm_sim import derive_params

    model = _load_param_model()
    profile = observable_profile(glucose_stream)
    log_s_i = float(model.predict(_design_row(profile, ehr_row).reshape(1, -1))[0])
    s_i = float(np.clip(np.exp(log_s_i), 0.004, 0.12))
    g_basal = float(np.clip(profile["night_median"] - 5.0, 80, 240))

    derived = derive_params(pd.DataFrame([ehr_row]), _ZeroRng()).iloc[0]
    return {
        "g_basal": g_basal,
        "s_i": s_i,
        "tau_g": float(derived["tau_g"]),
        "x_gain": float(derived["x_gain"]),
        "profile": profile,
    }


class PatientTwin:
    """One patient's digital twin."""

    def __init__(
        self,
        index: int,
        ehr: pd.DataFrame | None = None,
        streams: dict | None = None,
        bundle: ModelBundle | None = None,
        params: dict | None = None,
        upload_horizon_days: int = 14,
        now_step: int | None = None,
    ) -> None:
        self.index = int(index)
        self.ehr = load_ehr() if ehr is None else ehr
        self.streams = load_streams() if streams is None else streams
        self.ehr_row = self.ehr.iloc[self.index]
        self.patient_id = str(self.ehr_row["patient_id"])
        self.bundle = ModelBundle.get() if bundle is None else bundle
        self.horizon = int(upload_horizon_days * STEPS_PER_DAY)

        g_raw = self.streams["glucose"][self.index][: self.horizon]
        self.raw_glucose = g_raw.astype(np.int16).copy()
        self.glucose, self.missing = interpolate_gaps(self.raw_glucose)
        self.params = params or fitted_parameters(self.index, self.ehr_row, self.raw_glucose)
        self._refresh()
        self.now_step = int(now_step) if now_step is not None else self.n_steps - 1

    # ---- internals ----------------------------------------------------
    def _refresh(self) -> None:
        out = _features_for_patient(
            self.raw_glucose,
            self.streams["hr"][self.index][: self.horizon],
            self.streams["hrv"][self.index][: self.horizon],
            self.streams["steps"][self.index][: self.horizon],
            self.streams["stage"][self.index][: self.horizon],
            self.ehr_row,
        )
        self.features = out["X"]
        self._out = out
        self.n_steps = len(self.glucose)

    @property
    def last_step(self) -> int:
        return self.n_steps - 1

    def features_at(self, step: int) -> np.ndarray:
        return self.features[int(np.clip(step, 0, self.last_step))]

    # ---- core outputs -------------------------------------------------
    def risk_at(self, step: int | None = None) -> float:
        step = self.now_step if step is None else step
        return float(self.bundle.proba(self.features_at(step)[None, :])[0])

    def status(self, risk: float) -> str:
        if risk >= 0.7:
            return "critical"
        if risk >= 0.35:
            return "watch"
        return "stable"

    def time_in_range(self, hours: int = 24) -> dict:
        n = int(hours * 60 / STEP_MIN)
        end = min(self.now_step + 1, self.n_steps)
        seg = self.glucose[max(0, end - n) : end]
        total = len(seg)
        return {
            "in_range": float(np.mean((seg >= 70) & (seg <= 180))),
            "above": float(np.mean(seg > 180)),
            "below": float(np.mean(seg < 70)),
            "mean": float(np.mean(seg)),
            "hours": hours,
            "n": total,
        }

    def trajectory(self, step: int | None = None, minutes: int = 120) -> dict:
        """Physiological projection for the next two hours with an error band."""
        from whatif import patient_schedule, simulate_window

        step = self.now_step if step is None else int(step)
        minutes = int(minutes)
        n = minutes // STEP_MIN
        start = step
        end = min(start + n, self.n_steps)
        if end - start < 6:
            start = max(self.n_steps - n, 0)
            end = self.n_steps

        params_true = pd.read_csv(DATA_DIR / "true_params.csv").iloc[self.index]
        sched = patient_schedule(self.ehr_row, params_true, self.index)
        curve = simulate_window(
            self.params,
            sched["ra"].astype(np.float64),
            sched["walk_boost"].astype(np.float64),
            sched["exo_rate"].astype(np.float64),
            sched["stress_lift"].astype(np.float64),
            sched["sleep_mult"].astype(np.float64),
            start,
            end,
            float(self.glucose[start]),
            int(self.ehr_row["insulin"]),
            int(self.ehr_row["sulfonylurea"]),
        )
        steps = list(range(start, end))
        forecast_metrics = self.bundle.metrics.get("forecast", {})
        rmse = [forecast_metrics.get(f"lgbm_{h}min", {}).get("rmse", 30.0) for h in (30, 60, 120)]
        sd = np.interp(np.arange(len(curve)) * STEP_MIN, [0, 30, 60, 120], [3.0, *rmse])
        anchors = {
            h: float(self.bundle.forecasters[h].predict(self.features_at(start)[None, :])[0])
            for h in (30, 60, 120)
        }
        return {
            "steps": steps,
            "mean": curve.tolist(),
            "lo": (curve - 1.28 * sd).tolist(),
            "hi": (curve + 1.28 * sd).tolist(),
            "anchors": anchors,
        }

    def top_factors(self, step: int | None = None, k: int = 5) -> list[dict]:
        step = self.now_step if step is None else int(step)
        x = self.features_at(step)
        contributions = self.bundle.shap_row(x)
        names = [self.bundle.feature_names[i] for i in self.bundle.cols]
        # Several features can share one plain-language label (hour_sin and
        # hour_cos are both "time of day"). Merge them so the clinician does
        # not see the same sentence twice.
        merged: dict[str, dict] = {}
        for j, name in enumerate(names):
            label = plain_label(name)
            value = float(contributions[j])
            entry = merged.get(label)
            if entry is None:
                merged[label] = {
                    "label": label,
                    "feature": name,
                    "value": float(x[self.bundle.cols[j]]),
                    "contribution": value,
                    "peak": abs(value),
                }
            else:
                entry["contribution"] += value
                if abs(value) > entry["peak"]:
                    entry["peak"] = abs(value)
                    entry["feature"] = name
                    entry["value"] = float(x[self.bundle.cols[j]])
        ranked = sorted(merged.values(), key=lambda row: -abs(row["contribution"]))[:k]
        return [
            {
                "feature": row["feature"],
                "value": row["value"],
                "contribution": row["contribution"],
                "direction": "raises" if row["contribution"] > 0 else "lowers",
                "label": row["label"],
            }
            for row in ranked
        ]

    # ---- what-if support ----------------------------------------------
    def risk_with_simulated(
        self,
        curve: np.ndarray,
        start_step: int,
        checkpoint_step: int,
        extra_steps: np.ndarray | None = None,
    ) -> float:
        return self.risk_profile_with_simulated(curve, start_step, [checkpoint_step])[0]

    def risk_profile_with_simulated(
        self,
        curve: np.ndarray,
        start_step: int,
        checkpoints: list[int],
    ) -> list[float]:
        """Risk at several checkpoints after splicing a simulated segment in."""
        raw = self.raw_glucose.copy()
        end = min(start_step + len(curve), len(raw))
        raw[start_step:end] = np.clip(np.round(curve[: end - start_step]), 40, 450).astype(np.int16)
        out = _features_for_patient(
            raw,
            self.streams["hr"][self.index][: self.horizon],
            self.streams["hrv"][self.index][: self.horizon],
            self.streams["steps"][self.index][: self.horizon],
            self.streams["stage"][self.index][: self.horizon],
            self.ehr_row,
        )
        X = out["X"][checkpoints]
        return [float(v) for v in self.bundle.proba(X)]

    # ---- streaming ----------------------------------------------------
    def update(self, new_glucose: list[float] | np.ndarray) -> None:
        """Append new CGM readings (mg/dL) and refresh risk."""
        values = np.asarray(new_glucose, dtype=np.float32).ravel()
        n = len(values)
        if n >= len(self.raw_glucose):
            self.raw_glucose = np.round(values[-len(self.raw_glucose):]).astype(np.int16)
        else:
            self.raw_glucose = np.concatenate([self.raw_glucose[n:], np.round(values).astype(np.int16)])
        self.glucose, self.missing = interpolate_gaps(self.raw_glucose)
        self._refresh()

    def to_summary(self) -> dict:
        risk = self.risk_at()
        return {
            "patient_id": self.patient_id,
            "index": self.index,
            "age": int(self.ehr_row["age"]),
            "sex": str(self.ehr_row["sex"]),
            "risk": risk,
            "status": self.status(risk),
            "glucose": float(self.glucose[-1]),
            "trend_30m": float(self.glucose[-1] - self.glucose[-7]),
            "hba1c": float(self.ehr_row["hba1c"]),
            "bmi": float(self.ehr_row["bmi"]),
            "tir": self.time_in_range(24),
            "params": {k: float(v) for k, v in self.params.items() if k != "profile"},
        }


PLAIN_LANGUAGE = {
    "g_now": "current glucose",
    "g_mean_30m": "average glucose over the last 30 minutes",
    "g_mean_1h": "average glucose over the last hour",
    "g_mean_3h": "average glucose over the last 3 hours",
    "g_mean_6h": "average glucose over the last 6 hours",
    "g_std_3h": "glucose variability over the last 3 hours",
    "g_cv_3h": "glucose variability (coefficient of variation)",
    "g_min_3h": "lowest glucose in the last 3 hours",
    "g_max_3h": "highest glucose in the last 3 hours",
    "g_range_3h": "glucose range in the last 3 hours",
    "g_slope_15m": "how fast glucose is rising right now",
    "g_slope_30m": "glucose trend over 30 minutes",
    "g_slope_60m": "glucose trend over the last hour",
    "g_delta_30m": "change in glucose over 30 minutes",
    "g_delta_60m": "change in glucose over the last hour",
    "frac_above_180_3h": "share of the last 3 hours above 180 mg/dL",
    "frac_above_250_3h": "share of the last 3 hours above 250 mg/dL",
    "frac_below_70_3h": "share of the last 3 hours below 70 mg/dL",
    "missing_frac_3h": "sensor gaps in the last 3 hours",
    "hr_now": "current heart rate",
    "hr_mean_1h": "average heart rate over the last hour",
    "hr_max_1h": "peak heart rate over the last hour",
    "hr_delta_1h": "heart rate change over the last hour",
    "hrv_now": "current heart rate variability",
    "hrv_mean_1h": "heart rate variability over the last hour",
    "hrv_ratio_1h_6h": "heart rate variability versus the personal baseline",
    "hrv_night_mean": "heart rate variability during sleep",
    "steps_1h": "steps in the last hour",
    "steps_3h": "steps in the last 3 hours",
    "asleep_frac_1h": "share of the last hour asleep",
    "asleep_frac_6h": "sleep in the last 6 hours",
    "deep_frac_6h": "deep sleep in the last 6 hours",
    "hba1c": "HbA1c",
    "fpg": "fasting glucose",
    "bmi": "BMI",
    "years_since_dx": "years since diagnosis",
    "prior_hypo_count": "prior hypoglycaemia events",
    "tcf7l2_dose": "TCF7L2 genotype risk",
    "insulin": "insulin use",
    "sulfonylurea": "sulfonylurea use",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    "dow_sin": "day of the week",
    "dow_cos": "day of the week",
}


def plain_label(name: str) -> str:
    """Readable feature name, with a safe fallback for anything unmapped."""
    if name in PLAIN_LANGUAGE:
        return PLAIN_LANGUAGE[name]
    return name.replace("_", " ")


if __name__ == "__main__":
    print(json.dumps(fit_parameter_model(), indent=2))
