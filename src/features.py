"""GlucoTwin Phase 2: feature pipeline.

Turns the simulated streams into a supervised dataset:

  * label `y_event` = 1 when glucose exceeds 180 mg/dL for at least 30
    continuous minutes somewhere in the next 2 hours (24 x 5-min steps);
  * forecast targets: glucose at +30, +60 and +120 minutes;
  * features: CGM shape/trend, wearable load and sleep, static EHR, clock.

All rolling statistics look backwards only, so there is no future leakage.
Patients are split by ID in `patient_split` (never by window).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

STEP_MIN = 5
STEPS_PER_DAY = 288
HORIZON = 24          # 2 hours ahead
RUN_STEPS = 6         # 30 minutes above 180 mg/dL
N_CHANNELS = 5

CGM_FEATURES = [
    "g_now",
    "g_mean_30m",
    "g_mean_1h",
    "g_mean_3h",
    "g_mean_6h",
    "g_std_3h",
    "g_cv_3h",
    "g_min_3h",
    "g_max_3h",
    "g_range_3h",
    "g_slope_15m",
    "g_slope_30m",
    "g_slope_60m",
    "g_delta_30m",
    "g_delta_60m",
    "frac_above_180_3h",
    "frac_above_250_3h",
    "frac_below_70_3h",
    "missing_frac_3h",
]

WEARABLE_FEATURES = [
    "hr_now",
    "hr_mean_1h",
    "hr_max_1h",
    "hr_delta_1h",
    "hrv_now",
    "hrv_mean_1h",
    "hrv_ratio_1h_6h",
    "hrv_night_mean",
    "steps_1h",
    "steps_3h",
    "asleep_frac_1h",
    "asleep_frac_6h",
    "deep_frac_6h",
]

CONTEXT_FEATURES = ["hour_sin", "hour_cos", "is_night"]

STATIC_FEATURES = [
    "age",
    "sex_m",
    "bmi",
    "waist_cm",
    "veg_diet",
    "family_history",
    "years_since_dx",
    "hba1c",
    "fpg",
    "egfr",
    "ldl",
    "triglycerides",
    "systolic_bp",
    "metformin",
    "sulfonylurea",
    "dpp4",
    "insulin",
    "hypertension",
    "tcf7l2_dose",
    "prior_hypo_count",
]

FEATURE_NAMES = CGM_FEATURES + WEARABLE_FEATURES + CONTEXT_FEATURES + STATIC_FEATURES
N_FEATURES = len(FEATURE_NAMES)


def load_streams() -> dict[str, np.ndarray]:
    z = np.load(DATA_DIR / "cgm.npz")
    return {k: z[k] for k in z.files}


def load_ehr() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ehr.csv")


def patient_split(n_patients: int, seed: int = 42) -> dict[str, np.ndarray]:
    """60 / 20 / 20 split by patient, so no patient appears in two splits."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_patients)
    n_train = int(0.6 * n_patients)
    n_val = int(0.2 * n_patients)
    return {
        "train": np.sort(order[:n_train]),
        "val": np.sort(order[n_train : n_train + n_val]),
        "test": np.sort(order[n_train + n_val :]),
    }


def interpolate_gaps(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of sensor gaps; returns (filled, missing mask)."""
    missing = x < 0
    filled = x.astype(np.float32).copy()
    if not missing.any():
        return filled, missing
    idx = np.arange(len(x))
    if missing.all():
        filled[:] = 120.0
        return filled, missing
    filled[missing] = np.interp(idx[missing], idx[~missing], filled[~missing])
    return filled, missing


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.repeat(x[:1], k), x[:-k]])


def _rolling_slope(x: np.ndarray, k: int) -> np.ndarray:
    """Least-squares slope over the trailing k samples, per sample."""
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if n <= k:
        return out.astype(np.float32)
    xf = x.astype(np.float64)
    t = np.arange(n, dtype=np.float64)
    cs_x = np.concatenate([[0.0], np.cumsum(xf)])
    cs_tx = np.concatenate([[0.0], np.cumsum(t * xf)])
    sx = cs_x[k:] - cs_x[:-k]
    stx = cs_tx[k:] - cs_tx[:-k]
    start = np.arange(n - k + 1, dtype=np.float64)
    sum_j = k * (k - 1) / 2.0
    sum_j2 = (k - 1) * k * (2 * k - 1) / 6.0
    out[k - 1 :] = (stx - start * sx - sum_j * sx) / (sum_j2 - sum_j**2 / k)
    return out.astype(np.float32)


def _features_for_patient(
    g_raw: np.ndarray,
    hr: np.ndarray,
    hrv: np.ndarray,
    steps: np.ndarray,
    stage: np.ndarray,
    ehr_row: pd.Series,
) -> dict[str, np.ndarray]:
    g, missing = interpolate_gaps(g_raw)
    t = len(g)

    h = hr.astype(np.float32)
    v = hrv.astype(np.float32) / 10.0
    s = steps.astype(np.float32)
    st = stage

    asleep = (st > 0).astype(np.float32)
    deep = (st == 2).astype(np.float32)

    def rmean(x: np.ndarray, k: int) -> np.ndarray:
        return uniform_filter1d(x, size=k, mode="nearest")

    g_mean_3h = rmean(g, 36)
    g_var = np.clip(rmean(g * g, 36) - g_mean_3h**2, 0, None)
    g_min = minimum_filter1d(g, size=36, mode="nearest")
    g_max = maximum_filter1d(g, size=36, mode="nearest")

    hour = (np.arange(t) % STEPS_PER_DAY) / 12.0

    out = {
        "g_now": g,
        "g_mean_30m": rmean(g, 6),
        "g_mean_1h": rmean(g, 12),
        "g_mean_3h": g_mean_3h,
        "g_mean_6h": rmean(g, 72),
        "g_std_3h": np.sqrt(g_var),
        "g_cv_3h": np.sqrt(g_var) / np.clip(g_mean_3h, 1, None) * 100.0,
        "g_min_3h": g_min,
        "g_max_3h": g_max,
        "g_range_3h": g_max - g_min,
        "g_slope_15m": _rolling_slope(g, 3) / STEP_MIN,
        "g_slope_30m": _rolling_slope(g, 6) / STEP_MIN,
        "g_slope_60m": _rolling_slope(g, 12) / STEP_MIN,
        "g_delta_30m": g - _lag(g, 6),
        "g_delta_60m": g - _lag(g, 12),
        "frac_above_180_3h": rmean((g > 180).astype(np.float32), 36),
        "frac_above_250_3h": rmean((g > 250).astype(np.float32), 36),
        "frac_below_70_3h": rmean((g < 70).astype(np.float32), 36),
        "missing_frac_3h": rmean(missing.astype(np.float32), 36),
        "hr_now": h,
        "hr_mean_1h": rmean(h, 12),
        "hr_max_1h": maximum_filter1d(h, size=12, mode="nearest"),
        "hr_delta_1h": h - _lag(h, 12),
        "hrv_now": v,
        "hrv_mean_1h": rmean(v, 12),
        "hrv_ratio_1h_6h": np.clip(rmean(v, 12) / np.clip(rmean(v, 72), 1, None), 0, 3),
        "hrv_night_mean": rmean(v * asleep, 72) / np.clip(rmean(asleep, 72), 0.05, None),
        "steps_1h": rmean(s, 12) * 12.0,
        "steps_3h": rmean(s, 36) * 36.0,
        "asleep_frac_1h": rmean(asleep, 12),
        "asleep_frac_6h": rmean(asleep, 72),
        "deep_frac_6h": rmean(deep, 72),
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "is_night": ((hour >= 22) | (hour < 6)).astype(np.float32),
        "age": ehr_row["age"],
        "sex_m": float(ehr_row["sex"] == "M"),
        "bmi": ehr_row["bmi"],
        "waist_cm": ehr_row["waist_cm"],
        "veg_diet": float(ehr_row["diet_type"] == "veg"),
        "family_history": ehr_row["family_history"],
        "years_since_dx": ehr_row["years_since_dx"],
        "hba1c": ehr_row["hba1c"],
        "fpg": ehr_row["fpg_mgdl"],
        "egfr": ehr_row["egfr"],
        "ldl": ehr_row["ldl"],
        "triglycerides": ehr_row["triglycerides"],
        "systolic_bp": ehr_row["systolic_bp"],
        "metformin": ehr_row["metformin"],
        "sulfonylurea": ehr_row["sulfonylurea"],
        "dpp4": ehr_row["dpp4"],
        "insulin": ehr_row["insulin"],
        "hypertension": ehr_row["hypertension"],
        "tcf7l2_dose": {"CC": 0.0, "CT": 1.0, "TT": 2.0}[ehr_row["tcf7l2"]],
        "prior_hypo_count": ehr_row["prior_hypo_count"],
    }
    feats = np.empty((t, N_FEATURES), dtype=np.float32)
    for j, name in enumerate(FEATURE_NAMES):
        value = out[name]
        feats[:, j] = value if np.ndim(value) else value

    # ---- labels ---------------------------------------------------------
    above = (g > 180).astype(np.float32)
    run = np.zeros(t, dtype=np.float32)
    if t >= RUN_STEPS:
        run[RUN_STEPS - 1 :] = (
            sliding_window_view(above, RUN_STEPS).sum(axis=1) >= RUN_STEPS - 1e-6
        ).astype(np.float32)
    padded = np.concatenate([run, np.zeros(HORIZON, dtype=np.float32)])
    # window at row i is steps i..i+HORIZON; we want i+1..i+HORIZON
    y_event = sliding_window_view(padded, HORIZON + 1)[:, 1:].max(axis=1).astype(np.float32)

    valid = np.ones(t, dtype=bool)
    valid[-HORIZON:] = False
    return {
        "X": feats,
        "y_event": y_event,
        "y30": np.roll(g, -6),
        "y60": np.roll(g, -12),
        "y120": np.roll(g, -24),
        "valid": valid,
        "glucose": g,
    }


def build_dataset(
    ehr: pd.DataFrame,
    streams: dict[str, np.ndarray],
    patient_idx: np.ndarray,
    stride: int,
    include: str = "all",
) -> dict[str, np.ndarray]:
    """Build the supervised dataset for a set of patients.

    include: "all" | "cgm" | "ehr" -- used by the ablation study.
    """
    keep_idx = _columns_for(include)
    g_all = streams["glucose"]
    stride = int(stride)
    t_steps = g_all.shape[1]
    n_windows = t_steps // stride

    n_pat = len(patient_idx)
    X = np.empty((n_pat * n_windows, len(keep_idx)), dtype=np.float32)
    y_event = np.empty(n_pat * n_windows, dtype=np.int8)
    y30 = np.empty(n_pat * n_windows, dtype=np.float32)
    y60 = np.empty(n_pat * n_windows, dtype=np.float32)
    y120 = np.empty(n_pat * n_windows, dtype=np.float32)
    pid = np.empty(n_pat * n_windows, dtype=np.int32)
    step = np.empty(n_pat * n_windows, dtype=np.int32)

    pos = 0
    for p in patient_idx:
        out = _features_for_patient(
            g_all[p], streams["hr"][p], streams["hrv"][p], streams["steps"][p], streams["stage"][p],
            ehr.iloc[int(p)],
        )
        idx = np.arange(0, t_steps, stride)
        idx = idx[out["valid"][idx]]
        n = len(idx)
        X[pos : pos + n] = out["X"][idx][:, keep_idx]
        y_event[pos : pos + n] = (out["y_event"][idx] > 0.5).astype(np.int8)
        y30[pos : pos + n] = out["y30"][idx]
        y60[pos : pos + n] = out["y60"][idx]
        y120[pos : pos + n] = out["y120"][idx]
        pid[pos : pos + n] = int(p)
        step[pos : pos + n] = idx
        pos += n

    return {
        "X": X[:pos],
        "y_event": y_event[:pos],
        "y30": y30[:pos],
        "y60": y60[:pos],
        "y120": y120[:pos],
        "pid": pid[:pos],
        "step": step[:pos],
        "feature_names": np.array([FEATURE_NAMES[i] for i in keep_idx]),
    }


def _columns_for(include: str) -> list[int]:
    if include == "all":
        return list(range(N_FEATURES))
    if include == "cgm":
        names = set(CGM_FEATURES)
    elif include == "ehr":
        names = set(STATIC_FEATURES) | set(CONTEXT_FEATURES)
    else:
        raise ValueError(f"unknown feature set {include}")
    return [i for i, n in enumerate(FEATURE_NAMES) if n in names]


def event_starts(y_event: np.ndarray, pid: np.ndarray, step: np.ndarray, stride: int) -> list[tuple[int, int]]:
    """Return (patient, step) for every labelled event onset."""
    order = np.lexsort((step, pid))
    y = y_event[order]
    p = pid[order]
    s = step[order]
    starts: list[tuple[int, int]] = []
    prev_step = -10**9
    prev_pid = -1
    for i in range(len(y)):
        if y[i] == 1 and (p[i] != prev_pid or s[i] - prev_step > stride * 1.5):
            starts.append((int(p[i]), int(s[i])))
        prev_step = int(s[i])
        prev_pid = int(p[i])
    return starts


def main() -> None:
    ehr = load_ehr()
    streams = load_streams()
    splits = patient_split(len(ehr))
    for name, idx in splits.items():
        ds = build_dataset(ehr, streams, idx, stride=24)
        print(
            f"[features] {name}: {len(idx)} patients, {len(ds['X'])} windows, "
            f"event prevalence {ds['y_event'].mean():.1%}, {ds['X'].shape[1]} features"
        )


if __name__ == "__main__":
    main()
