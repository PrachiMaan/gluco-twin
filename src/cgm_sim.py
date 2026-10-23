"""GlucoTwin Phase 1: CGM + wearable simulator.

Simulates 14 days per patient at 5-minute resolution from that patient's own
EHR record. The physiology is a meal-absorption + insulin-sensitivity model in
the style of the Bergman minimal model: a glucose pool pulled toward a basal
set point (glucose effectiveness), a delayed endogenous insulin-action
compartment, an exogenous insulin-action compartment for insulin users, and
insulin-dependent clearance. Patient parameters are derived from EHR fields
(fasting glucose -> basal set point; BMI / HbA1c / years since diagnosis /
TCF7L2 genotype / medications -> insulin sensitivity, secretion gain, meal
handling), so the EHR and CGM streams are genuinely linked.

Wearables (heart rate, HRV, steps, sleep stages) are generated from the same
behaviour schedules and couple back into glucose: a post-meal walk raises
insulin sensitivity for ~2 hours, poor sleep lowers the next day's
sensitivity, stress lifts glucose.

Outputs (data/):
  cgm.npz           glucose/hr/hrv/steps/stage arrays, shape (n_patients, T)
  true_params.csv   physiological parameters + behaviour flags per patient
  sim_manifest.json generator settings

Run:  python src/cgm_sim.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DAYS = 14
STEP_MIN = 5
STEPS_PER_DAY = 288
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# physiological constants (documented in docs/DATA_CARD.md)
CARB_GAIN = 1.6        # mg/dL of glucose appearance per gram of carbohydrate
U2X = 6.0              # (insulin U/min) -> insulin-action units
TAU_X = 30.0           # endogenous insulin action delay (min)
TAU_E = 45.0           # exogenous insulin action delay (min)
X_GAIN_BASAL = 0.143   # basal signalling level of the normalised drive
DAWN_RATE = 0.08       # mg/dL/min peak of the dawn phenomenon
STRESS_RATE = 0.12     # mg/dL/min at stress level 1.0
BOOST_DECAY = 0.94     # per 5-min step, persistence of the exercise boost

MEAL_KINDS = {  # carbohydrate absorption rate k (1/min)
    "breakfast": 0.024,
    "rice": 0.028,
    "roti": 0.021,
    "snack": 0.031,
    "sweets": 0.035,
    "dinner": 0.023,
}

STAGE_NOT_IN_BED = 0
STAGE_LIGHT = 1
STAGE_DEEP = 2
STAGE_REM = 3
STAGE_AWAKE = 4


def derive_params(ehr: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Derive physiological parameters from a patient's EHR row.

    This function *is* the EHR -> simulator linkage: every parameter is a
    function of record fields plus small individual jitter.
    """
    n = len(ehr)
    j = lambda scale: rng.normal(0, scale, n)  # noqa: E731

    bmi = ehr["bmi"].to_numpy(float)
    ysd = ehr["years_since_dx"].to_numpy(float)
    a1c = ehr["hba1c"].to_numpy(float)
    fpg = ehr["fpg_mgdl"].to_numpy(float)
    tt = (ehr["tcf7l2"] == "TT").to_numpy(float)
    ct = (ehr["tcf7l2"] == "CT").to_numpy(float)

    g_basal = fpg - 4.0 + j(5.0)

    s_i = (
        0.033
        * np.exp(-0.045 * (bmi - 22.0) - 0.030 * ysd - 0.085 * (a1c - 7.0))
        * (1.0 + 0.15 * ehr["metformin"].to_numpy(float) + 0.10 * ehr["dpp4"].to_numpy(float))
        * np.exp(j(0.12))
    )
    s_i = np.clip(s_i, 0.004, 0.12)

    tau_g = np.clip(60.0 + 0.8 * (bmi - 22.0) + 0.6 * ysd + j(5.0), 45.0, 110.0)

    # first-phase secretion is lost with disease duration and the TT genotype
    x_gain = 3.0 - 0.09 * ysd - 1.1 * tt - 0.4 * ct + np.clip(0.6 - 0.06 * a1c, -0.4, 0.6) + j(0.35)
    x_gain = np.clip(x_gain, 0.6, 4.0)
    x_gain = x_gain * (1.0 - 0.40 * ehr["insulin"].to_numpy(float))
    x_gain = x_gain * (1.0 + 0.35 * ehr["sulfonylurea"].to_numpy(float))

    return pd.DataFrame(
        {
            "patient_id": ehr["patient_id"],
            "g_basal": g_basal,
            "s_i": s_i,
            "tau_g": tau_g,
            "x_gain": x_gain,
        }
    )


def _build_patient_schedule(
    ehr_row: pd.Series,
    param_row: pd.Series,
    rng: np.random.Generator,
    days: int,
) -> dict:
    """Build one patient's behaviour schedule: meals, activity, sleep, stress."""
    t_steps = days * STEPS_PER_DAY
    ra = np.zeros(t_steps)            # meal appearance, mg/dL/min
    walk_boost = np.zeros(t_steps)     # 0..0.8 exercise boost on insulin sensitivity
    exo_rate = np.zeros(t_steps)       # insulin units/min
    steps = np.zeros(t_steps)
    stress_lift = np.zeros(t_steps)
    sleep_mult = np.ones(t_steps)

    morning_walk = bool(rng.random() < 0.55)
    post_lunch_walk = bool(rng.random() < 0.35)
    post_dinner_walk = bool(rng.random() < 0.40)
    exercise = bool(rng.random() < 0.18)
    poor_sleeper = bool(
        rng.random()
        < 1.0 / (1.0 + math.exp(-(-1.0 + 0.10 * (ehr_row["bmi"] - 24.0) + 0.03 * (ehr_row["age"] - 55.0))))
    )
    festival_day = int(rng.integers(8, 11)) if rng.random() < 0.5 else -1

    s_i = float(param_row["s_i"])
    dose_adj = float(np.clip(0.02 / s_i, 0.5, 4.0))
    on_insulin = int(ehr_row["insulin"])

    night_windows: list[tuple[float, float, float]] = []
    stress_days = np.zeros(days)

    for d in range(days):
        base = d * STEPS_PER_DAY
        weekend = (d % 7) >= 5

        # ---- meals -------------------------------------------------------
        meals: list[tuple[int, float, str]] = []
        meals.append((int(7 * 60 + rng.normal(0, 25)), float(rng.uniform(45, 70)), "breakfast"))
        lunch_kind = "rice" if rng.random() < 0.6 else "roti"
        meals.append(
            (int(12 * 60 + 50 + rng.normal(0, 30)), float(rng.uniform(70, 110)), lunch_kind)
        )
        if rng.random() < 0.8:
            meals.append((int(16 * 60 + 40 + rng.normal(0, 30)), float(rng.uniform(12, 25)), "snack"))
        dinner_kind = "dinner" if rng.random() < 0.7 else "roti"
        meals.append(
            (int(20 * 60 + 35 + rng.normal(0, 30)), float(rng.uniform(55, 90)), dinner_kind)
        )
        if rng.random() < 0.12:
            meals.append((int(22 * 60 + 40 + rng.normal(0, 15)), float(rng.uniform(15, 30)), "snack"))
        if d == festival_day:
            meals = [(m, c * 1.45, k) for (m, c, k) in meals]
            meals.append((int(18 * 60 + rng.normal(0, 20)), float(rng.uniform(35, 55)), "sweets"))

        for minute, carbs, kind in meals:
            if minute < 0 or minute >= 24 * 60:
                continue
            k = MEAL_KINDS[kind]
            amp = carbs * CARB_GAIN
            start = base + minute // STEP_MIN
            for j_step in range(1, 60):  # 5-hour absorption horizon
                tt = j_step * STEP_MIN
                rate = amp * k * k * tt * math.exp(-k * tt) * STEP_MIN
                idx = start + j_step
                if idx < t_steps:
                    ra[idx] += rate

        # ---- activity ----------------------------------------------------
        def block(start_min: int, dur_min: int, level: float) -> None:
            s = base + int(start_min) // STEP_MIN
            n_steps = max(int(dur_min) // STEP_MIN, 1)
            for j_step in range(n_steps):
                idx = s + j_step
                if 0 <= idx < t_steps:
                    steps[idx] += rng.uniform(level * 0.8, level * 1.2)
                    walk_boost[idx] = 0.8

        if morning_walk:
            block(6 * 60 + 50 + rng.normal(0, 20), 30 + rng.uniform(-8, 10), 480)
        if post_lunch_walk:
            block(13 * 60 + 55 + rng.normal(0, 15), 20 + rng.uniform(-5, 8), 420)
        if post_dinner_walk:
            block(21 * 60 + 20 + rng.normal(0, 20), 28 + rng.uniform(-6, 10), 440)
        if exercise and rng.random() < 0.5:
            block(18 * 60 + rng.uniform(0, 90), 30, 700)

        active_day = bool(rng.random() < (0.30 if weekend else 0.45))
        bg_level = rng.uniform(20, 34) if active_day else rng.uniform(12, 22)
        for m in range(0, 24 * 60, STEP_MIN):
            idx = base + m // STEP_MIN
            if idx >= t_steps:
                break
            hour = m / 60.0
            day_frac = 0.0 if (hour >= 23.5 or hour < 5.5) else (1.0 if 8 <= hour <= 22 else 0.5)
            steps[idx] += rng.uniform(0.5, 1.6) * bg_level * day_frac

        # ---- sleep (night that begins on day d-1 evening) ----------------
        bed = (22 * 60 + 45 + rng.normal(0, 35)) % (24 * 60)
        if weekend:
            bed = min(bed + 45, 24 * 60 + 30)
        wake = 6 * 60 + 15 + rng.normal(0, 30) + (45 if weekend else 0)
        if poor_sleeper:
            wake -= rng.uniform(25, 55)
        quality = float(
            np.clip(rng.uniform(0.5, 0.7) if poor_sleeper else rng.uniform(0.7, 0.95), 0.4, 0.98)
        )
        start_min = (d - 1) * 24 * 60 + bed if d > 0 else bed - 24 * 60
        end_min = d * 24 * 60 + wake
        night_windows.append((start_min, end_min, quality))
        if d > 0:
            sleep_h = max((end_min - start_min) / 60.0, 3.0)
            mult = float(np.clip(1.0 + 0.022 * (sleep_h - 7.0), 0.86, 1.08))
            a = int((d * 24 * 60 + 5 * 60) / STEP_MIN)
            b = min(int(((d + 1) * 24 * 60) / STEP_MIN), t_steps)
            sleep_mult[a:b] = mult

        # ---- stress ------------------------------------------------------
        stress = float(rng.beta(2.0, 5.0))
        if d == festival_day:
            stress = min(stress + 0.2, 1.0)
        stress_days[d] = stress
        a = int((d * 24 * 60) / STEP_MIN)
        b = min(int(((d + 1) * 24 * 60) / STEP_MIN), t_steps)
        hours = (np.arange(a, b) % STEPS_PER_DAY) / (STEPS_PER_DAY / 24.0)
        work = np.where((hours >= 10) & (hours <= 18), 1.3, 1.0)
        stress_lift[a:b] = STRESS_RATE * stress * work

        # ---- insulin dosing (insulin users) ------------------------------
        if on_insulin:
            def dose(minute: int, units: float, spread_min: float = 15.0) -> None:
                s = base + int(minute) // STEP_MIN
                n_steps = max(int(spread_min) // STEP_MIN, 1)
                for j_step in range(n_steps):
                    idx = s + j_step
                    if 0 <= idx < t_steps:
                        exo_rate[idx] += units / (n_steps * STEP_MIN)

            dose(8 * 60, 12.0 * dose_adj, 720)   # morning basal, 12h cover
            dose(22 * 60, 10.0 * dose_adj, 720)   # evening basal
            for minute, carbs, _kind in meals:
                if 20 <= minute < 24 * 60:
                    dose(minute - 20, carbs / 12.0 * dose_adj)

    return {
        "ra": ra.astype(np.float32),
        "walk_boost": walk_boost.astype(np.float32),
        "exo_rate": exo_rate.astype(np.float32),
        "steps": steps.astype(np.float32),
        "stress_lift": stress_lift.astype(np.float32),
        "sleep_mult": sleep_mult.astype(np.float32),
        "night_windows": night_windows,
        "stress_days": stress_days,
        "flags": {
            "morning_walk": int(morning_walk),
            "post_lunch_walk": int(post_lunch_walk),
            "post_dinner_walk": int(post_dinner_walk),
            "exercise_habit": int(exercise),
            "poor_sleeper": int(poor_sleeper),
            "festival_day": festival_day,
            "dose_adj": dose_adj,
        },
    }


def _simulate_chunk(
    ehr_chunk: pd.DataFrame,
    params_chunk: pd.DataFrame,
    rngs: list[np.random.Generator],
    days: int,
) -> dict:
    n = len(ehr_chunk)
    t_steps = days * STEPS_PER_DAY
    dt = STEP_MIN

    schedules = [
        _build_patient_schedule(ehr_chunk.iloc[i], params_chunk.iloc[i], rngs[i], days)
        for i in range(n)
    ]

    ra = np.stack([s["ra"] for s in schedules])
    walk_boost = np.stack([s["walk_boost"] for s in schedules])
    exo_rate = np.stack([s["exo_rate"] for s in schedules])
    steps = np.stack([s["steps"] for s in schedules])
    stress_lift = np.stack([s["stress_lift"] for s in schedules])
    sleep_mult = np.stack([s["sleep_mult"] for s in schedules])

    # exercise keeps working for ~2h after the walk ends
    for t in range(1, t_steps):
        walk_boost[:, t] = np.maximum(walk_boost[:, t], BOOST_DECAY * walk_boost[:, t - 1])

    g_basal = params_chunk["g_basal"].to_numpy(np.float64)[:, None]
    s_i = params_chunk["s_i"].to_numpy(np.float64)[:, None]
    tau_g = params_chunk["tau_g"].to_numpy(np.float64)[:, None]
    x_gain = params_chunk["x_gain"].to_numpy(np.float64)[:, None]
    insulin = ehr_chunk["insulin"].to_numpy(np.float64)[:, None]
    su = ehr_chunk["sulfonylurea"].to_numpy(np.float64)[:, None]

    minute_of_day = (np.arange(t_steps) % STEPS_PER_DAY) * STEP_MIN
    hour = minute_of_day / 60.0
    dawn = (
        DAWN_RATE
        * np.exp(-((hour - 6.5) ** 2) / 6.0)
        * ((hour > 3.5) & (hour < 10.0))
    )  # per-step scalar, indexed with dawn[t]

    dose_adj = np.clip(0.02 / s_i, 0.5, 4.0)
    x0 = x_gain * X_GAIN_BASAL
    # steady-state action of the 12h basal dose (0 for patients not on insulin)
    xe_basal = insulin * (12.0 * dose_adj / 720.0) * U2X * TAU_E

    g = g_basal + np.random.default_rng(0).normal(0, 1.0, (n, 1))
    x = np.full_like(g, 0.0) + x0
    xe = xe_basal + 0.0
    g_hist = np.repeat(g, 4, axis=1)  # ring buffer for the 15-min insulin delay

    glucose = np.empty((n, t_steps), dtype=np.float32)
    ring_ptr = 0
    for t in range(t_steps):
        g_del = g_hist[:, ring_ptr][:, None]  # 15-min delayed glucose, column
        signal = np.clip((g_del - (g_basal - 10.0)) / 70.0, 0.0, 3.0)
        x_target = x_gain * signal
        x = x + dt * (x_target - x) / TAU_X
        xe = xe + dt * (exo_rate[:, t][:, None] * U2X - xe) / TAU_E

        s_i_eff = s_i * sleep_mult[:, t][:, None] * (1.0 + 0.9 * walk_boost[:, t][:, None])
        deviation = np.clip(x + xe - x0 - xe_basal, -0.5, None)
        clear = s_i_eff * deviation * np.clip(g - 55.0, 0.0, None)
        # counter-regulation when glucose drops low (weaker on sulfonylureas)
        rescue = np.where(g < 70.0, 0.04 * (70.0 - g) * (1.0 - 0.40 * su), 0.0)

        g = g + dt * (
            -(g - g_basal) / tau_g
            + dawn[t]
            + ra[:, t][:, None]
            + stress_lift[:, t][:, None]
            + rescue
            - clear
        )
        g = np.clip(g, 52.0, 400.0)
        glucose[:, t] = g[:, 0]

        g_hist[:, ring_ptr] = g[:, 0]
        ring_ptr = (ring_ptr + 1) % 4

    return {
        "glucose": glucose,
        "steps": steps,
        "schedules": schedules,
    }


def _wearables(
    glucose: np.ndarray,
    steps: np.ndarray,
    schedules: list[dict],
    ehr_chunk: pd.DataFrame,
    days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, t_steps = glucose.shape
    rng = np.random.default_rng(12345)
    bmi = ehr_chunk["bmi"].to_numpy(float)
    age = ehr_chunk["age"].to_numpy(float)
    ysd = ehr_chunk["years_since_dx"].to_numpy(float)

    hr_rest = np.clip(64 + 0.28 * (bmi - 24) + 0.18 * (age - 55) + rng.normal(0, 5, n), 52, 92)
    hrv_base = np.clip(38 - 0.55 * (bmi - 24) - 0.35 * ysd + rng.normal(0, 9, n), 9, 85)

    minute_of_day = (np.arange(t_steps) % STEPS_PER_DAY) * STEP_MIN
    hour = minute_of_day / 60.0
    is_night = ((hour >= 23.0) | (hour < 5.5)).astype(np.float32)[None, :]

    hr = (
        hr_rest[:, None]
        + 0.075 * steps
        + 0.05 * np.clip(glucose - 145.0, 0.0, 120.0)
        - 8.0 * is_night
        + rng.normal(0, 2.5, (n, t_steps))
    )

    g_lag = np.concatenate([np.repeat(glucose[:, :1], 6, axis=1), glucose[:, :-6]], axis=1)
    hyper_pen = 0.10 * np.clip((g_lag - 160.0) / 80.0, 0.0, 1.0)
    stress_days = np.stack([s["stress_days"] for s in schedules])
    day_idx = (np.arange(t_steps) // STEPS_PER_DAY)[None, :]
    stress_day = np.take_along_axis(stress_days, np.clip(day_idx, 0, days - 1), axis=1)
    hrv = (
        hrv_base[:, None]
        * (1.0 + 0.35 * is_night)
        * (1.0 - 0.0022 * np.clip(hr - hr_rest[:, None], 0.0, None))
        * (1.0 - hyper_pen)
        * (1.0 - 0.12 * stress_day)
        * np.exp(rng.normal(0, 0.12, (n, t_steps)))
    )

    stage = np.zeros((n, t_steps), dtype=np.uint8)
    sleep_h = np.zeros((n, days), dtype=np.float32)
    for i in range(n):
        for (s, e, q) in schedules[i]["night_windows"]:
            a = max(int(s / STEP_MIN), 0)
            b = min(int(np.ceil(e / STEP_MIN)), t_steps)
            if b - a < 4:
                continue
            length = b - a
            stage[i, a:b] = STAGE_LIGHT
            cyc = 18  # 90-minute cycle at 5-min resolution
            for pos in range(0, length, cyc):
                seg = min(cyc, length - pos)
                third = max(seg // 3, 1)
                stage[i, a + pos : a + pos + third] = STAGE_LIGHT
                stage[i, a + pos + third : a + pos + 2 * third] = STAGE_DEEP
                stage[i, a + pos + 2 * third : a + pos + seg] = STAGE_REM
            awake_n = int(length * (1.0 - q))
            if awake_n > 0:
                r = np.random.default_rng(int(abs(s)) + i)
                for _ in range(max(awake_n // 5, 1)):
                    pos = a + int(r.integers(0, max(length - 1, 1)))
                    stage[i, pos : min(pos + 5, b)] = STAGE_AWAKE
            day = int(e // (24 * 60))
            if 0 <= day < days:
                sleep_h[i, day] = (e - s) / 60.0

    return hr, hrv, stage, sleep_h


def simulate(
    ehr: pd.DataFrame,
    days: int = DAYS,
    seed: int = SEED,
    chunk: int = 250,
) -> dict:
    """Run the full simulator. Returns typed arrays of shape (n, days*288)."""
    n = len(ehr)
    t_steps = days * STEPS_PER_DAY

    master = np.random.default_rng(seed)
    child_seeds = master.integers(1, 2**63, size=n)

    param_rng = np.random.default_rng(seed)
    param_rng.integers(1, 2**63, size=n)  # keep parameter jitter seed-stable
    params = derive_params(ehr, param_rng)

    out = {
        "glucose": np.full((n, t_steps), -1, dtype=np.int16),
        "hr": np.zeros((n, t_steps), dtype=np.uint16),
        "hrv": np.zeros((n, t_steps), dtype=np.uint16),
        "steps": np.zeros((n, t_steps), dtype=np.uint16),
        "stage": np.zeros((n, t_steps), dtype=np.uint8),
    }
    sleep_h_all = np.zeros((n, days), dtype=np.float32)
    flags_rows: list[dict] = [{} for _ in range(n)]

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        ehr_c = ehr.iloc[start:stop].reset_index(drop=True)
        params_c = params.iloc[start:stop].reset_index(drop=True)
        rngs_c = [np.random.default_rng(int(s)) for s in child_seeds[start:stop]]

        sim = _simulate_chunk(ehr_c, params_c, rngs_c, days)
        hr, hrv, stage, sleep_h = _wearables(sim["glucose"], sim["steps"], sim["schedules"], ehr_c, days)
        sleep_h_all[start:stop] = sleep_h

        for i, sched in enumerate(sim["schedules"]):
            flags = dict(sched["flags"])
            flags["mean_sleep_hours"] = round(float(sleep_h[i].mean()), 2)
            flags_rows[start + i] = flags

        # ---- sensor noise, drift and dropouts ----------------------------
        noise_rng = np.random.default_rng(seed + 7 + start)
        obs = sim["glucose"] + noise_rng.normal(0, 4.5, sim["glucose"].shape)
        obs = obs + noise_rng.normal(0, 0.003, (stop - start, 1)) * np.arange(t_steps)
        for i in range(stop - start):
            for _ in range(int(noise_rng.integers(2, 5))):
                gap_len = int(noise_rng.integers(6, 36))          # 30-180 min
                gap_start = int(noise_rng.integers(0, t_steps - gap_len))
                obs[i, gap_start : gap_start + gap_len] = np.nan

        obs = np.clip(obs, 40, 450)
        out["glucose"][start:stop] = np.where(np.isnan(obs), -1, np.round(obs)).astype(np.int16)
        out["hr"][start:stop] = np.clip(np.round(hr), 30, 200).astype(np.uint16)
        out["hrv"][start:stop] = np.clip(np.round(hrv * 10), 0, 1500).astype(np.uint16)
        out["steps"][start:stop] = np.clip(np.round(sim["steps"]), 0, 2000).astype(np.uint16)
        out["stage"][start:stop] = stage

    params_out = params.reset_index(drop=True).copy()
    for col, vals in pd.DataFrame(flags_rows).items():
        params_out[col] = vals

    return {
        **out,
        "sleep_hours": sleep_h_all,
        "params": params_out,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ehr = pd.read_csv(DATA_DIR / "ehr.csv")
    print(f"[cgm_sim] simulating {len(ehr)} patients x {DAYS} days ...")
    result = simulate(ehr)

    np.savez_compressed(
        DATA_DIR / "cgm.npz",
        glucose=result["glucose"],
        hr=result["hr"],
        hrv=result["hrv"],
        steps=result["steps"],
        stage=result["stage"],
        sleep_hours=result["sleep_hours"],
    )
    result["params"].to_csv(DATA_DIR / "true_params.csv", index=False)

    g = result["glucose"].astype(np.float32)
    g[g < 0] = np.nan
    manifest = {
        "seed": SEED,
        "days": DAYS,
        "step_min": STEP_MIN,
        "timesteps_per_patient": int(g.shape[1]),
        "n_patients": int(len(ehr)),
        "carb_gain": CARB_GAIN,
        "missing_fraction": round(float(np.isnan(g).mean()), 4),
        "glucose_mean": round(float(np.nanmean(g)), 1),
        "glucose_p99": round(float(np.nanpercentile(g, 99)), 1),
    }
    (DATA_DIR / "sim_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
