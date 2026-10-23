"""GlucoTwin Phase 3: what-if engine.

Rebuilds a patient's real simulated day, applies one behaviour change, re-runs
the same physiology and re-scores the risk model on the resulting stream.

Scenarios (all use the patient's own fitted parameters):
  walk_after_dinner   30-minute walk after dinner
  swap_rice           evening carbs reduced by 30% (millet-style swap)
  skip_snack          evening chai/snack removed
  sleep_7h            sleep improved to 7 hours (next-day insulin sensitivity)
  metformin_up        modest insulin-sensitivity gain from a dose change

This is a simulation for decision support. It is not medical advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cgm_sim import (
    BOOST_DECAY,
    CARB_GAIN,
    DAWN_RATE,
    DAYS,
    MEAL_KINDS,
    SEED,
    STEP_MIN,
    STEPS_PER_DAY,
    STRESS_RATE,
    TAU_E,
    TAU_X,
    U2X,
    X_GAIN_BASAL,
    _build_patient_schedule,
)

WINDOW_START_MIN = 17 * 60      # 17:00 on the current day
WINDOW_END_MIN = 24 * 60        # midnight
CHECKPOINT_MIN = 22 * 60 + 5    # risk is re-scored here


@dataclass
class Scenario:
    key: str
    label: str
    description: str


SCENARIOS = [
    Scenario("walk_after_dinner", "Walk after dinner", "30-minute walk starting 21:15"),
    Scenario("swap_rice", "Swap rice for millets", "Evening carbohydrate load reduced by 30%"),
    Scenario("skip_snack", "Skip evening snack", "No chai and snack at 16:40"),
    Scenario("sleep_7h", "Sleep 7 hours", "Better sleep improves next-day insulin sensitivity"),
    Scenario("metformin_up", "Metformin dose change", "Modest insulin-sensitivity gain (modelled)"),
]


def child_seed(index: int, n_patients: int = 1500, seed: int = SEED) -> int:
    return int(np.random.default_rng(seed).integers(1, 2**63, size=n_patients)[index])


def patient_schedule(ehr_row: pd.Series, param_row: pd.Series, index: int, days: int = DAYS) -> dict:
    rng = np.random.default_rng(child_seed(index))
    return _build_patient_schedule(ehr_row, param_row, rng, days)


def meals_to_ra(
    meal_log: list[tuple[int, int, float, str]],
    days: int,
    evening_scale: float = 1.0,
    skip_snack: bool = False,
) -> np.ndarray:
    t_steps = days * STEPS_PER_DAY
    ra = np.zeros(t_steps, dtype=np.float64)
    for (d, minute, carbs, kind) in meal_log:
        if kind == "snack" and skip_snack and minute >= 15 * 60:
            continue
        scale = 1.0
        if evening_scale != 1.0 and minute >= 15 * 60:
            scale = evening_scale
        k = MEAL_KINDS[kind]
        amp = carbs * scale * CARB_GAIN
        start = d * STEPS_PER_DAY + minute // STEP_MIN
        for j in range(1, 60):
            tt = j * STEP_MIN
            rate = amp * k * k * tt * math.exp(-k * tt) * STEP_MIN
            idx = start + j
            if idx < t_steps:
                ra[idx] += rate
    return ra


def walks_to_boost(
    walk_log: list[tuple[int, float, float, float]],
    days: int,
    extra_walk_minute: int | None = None,
) -> np.ndarray:
    t_steps = days * STEPS_PER_DAY
    boost = np.zeros(t_steps, dtype=np.float64)
    walks = list(walk_log)
    if extra_walk_minute is not None:
        walks.append((days - 1, float(extra_walk_minute), 30.0, 440.0))
    for (d, start_min, dur_min, _level) in walks:
        start = int(d * STEPS_PER_DAY + start_min // STEP_MIN)
        for j in range(max(int(dur_min) // STEP_MIN, 1)):
            idx = start + j
            if 0 <= idx < t_steps:
                boost[idx] = 0.8
    for t in range(1, t_steps):
        boost[t] = max(boost[t], BOOST_DECAY * boost[t - 1])
    return boost


def simulate_window(
    params: dict,
    day_ra: np.ndarray,
    day_boost: np.ndarray,
    day_exo: np.ndarray,
    day_stress: np.ndarray,
    day_sleep_mult: np.ndarray,
    start_step: int,
    end_step: int,
    g0: float,
    insulin: int,
    sulfonylurea: int,
    s_i_scale: float = 1.0,
    g_basal_shift: float = 0.0,
) -> np.ndarray:
    """Integrate the same glucose model over one window (5-minute steps)."""
    g_basal = float(params["g_basal"]) + g_basal_shift
    s_i = float(params["s_i"]) * s_i_scale
    tau_g = float(params["tau_g"])
    x_gain = float(params["x_gain"])

    dose_adj = float(np.clip(0.02 / max(s_i, 1e-4), 0.5, 4.0))
    x0 = x_gain * X_GAIN_BASAL
    xe_basal = insulin * (12.0 * dose_adj / 720.0) * U2X * TAU_E

    g = np.full(1, g0, dtype=np.float64)
    x = np.full(1, x0)
    xe = np.full(1, xe_basal)
    hist = np.repeat(g, 4)
    ptr = 0
    out = []
    for t in range(start_step, end_step):
        hour = (t % STEPS_PER_DAY) * STEP_MIN / 60.0
        dawn = DAWN_RATE * math.exp(-((hour - 6.5) ** 2) / 6.0) if 3.5 < hour < 10 else 0.0
        g_del = hist[ptr]
        signal = float(np.clip((g_del - (g_basal - 10.0)) / 70.0, 0.0, 4.0))
        x = x + STEP_MIN * (x_gain * signal - x) / TAU_X
        xe = xe + STEP_MIN * (day_exo[t] * U2X - xe) / TAU_E
        s_i_eff = s_i * day_sleep_mult[t] * (1.0 + 0.9 * day_boost[t])
        deviation = max(float(x[0] + xe[0] - x0 - xe_basal), -0.5)
        clear = s_i_eff * deviation * max(float(g[0]) - 55.0, 0.0)
        rescue = 0.03 * (70.0 - g[0]) * (1.0 - 0.40 * sulfonylurea) if g[0] < 70 else 0.0
        g = g + STEP_MIN * (
            -(g - g_basal) / tau_g + dawn + day_ra[t] + day_stress[t] + rescue - clear
        )
        g = np.clip(g, 52.0, 400.0)
        out.append(float(g[0]))
        hist[ptr] = g[0]
        ptr = (ptr + 1) % 4
    return np.asarray(out)


def run_scenarios(
    ehr_row: pd.Series,
    true_param_row: pd.Series,
    fitted_params: dict,
    index: int,
    recorded_glucose: np.ndarray,
) -> dict:
    """Simulate the evening under each scenario and return curves + flags."""
    sched = patient_schedule(ehr_row, true_param_row, index)
    day = DAYS - 1  # the evening of the current day
    start_step = day * STEPS_PER_DAY + WINDOW_START_MIN // STEP_MIN
    end_step = day * STEPS_PER_DAY + WINDOW_END_MIN // STEP_MIN
    checkpoint_step = day * STEPS_PER_DAY + CHECKPOINT_MIN // STEP_MIN

    g0 = float(recorded_glucose[start_step])
    insulin = int(ehr_row["insulin"])
    su = int(ehr_row["sulfonylurea"])
    mean_sleep = float(true_param_row.get("mean_sleep_hours", 6.5) or 6.5)

    def base_inputs(extra_walk=None, evening_scale=1.0, skip_snack=False):
        ra = meals_to_ra(sched["meal_log"], DAYS, evening_scale=evening_scale, skip_snack=skip_snack)
        boost = walks_to_boost(sched["walk_log"], DAYS, extra_walk_minute=extra_walk)
        return ra, boost

    # the walk starts 20 minutes after that evening's dinner
    dinner_minutes = [m for (d, m, _c, k) in sched["meal_log"] if d == day and m >= 18 * 60 and k in ("dinner", "roti")]
    dinner_minute = int(max(dinner_minutes)) if dinner_minutes else 20 * 60 + 35

    variants = {"baseline": dict()}
    variants["walk_after_dinner"] = {"extra_walk": dinner_minute + 20}
    variants["swap_rice"] = {"evening_scale": 0.7}
    variants["skip_snack"] = {"skip_snack": True}
    variants["sleep_7h"] = {"s_i_scale": float(np.clip(1.0 + 0.03 * (7.0 - mean_sleep), 0.94, 1.12))}
    variants["metformin_up"] = {"s_i_scale": 1.10, "g_basal_shift": -6.0}

    out = {}
    for key, opts in variants.items():
        ra, boost = base_inputs(
            extra_walk=opts.get("extra_walk"),
            evening_scale=opts.get("evening_scale", 1.0),
            skip_snack=opts.get("skip_snack", False),
        )
        curve = simulate_window(
            fitted_params,
            ra,
            boost,
            sched["exo_rate"],
            sched["stress_lift"],
            sched["sleep_mult"],
            start_step,
            end_step,
            g0,
            insulin,
            su,
            s_i_scale=opts.get("s_i_scale", 1.0),
            g_basal_shift=opts.get("g_basal_shift", 0.0),
        )
        peak = float(np.max(curve))
        above = int(np.sum(curve > 180) * STEP_MIN)
        out[key] = {
            "curve": curve,
            "peak": peak,
            "minutes_above_180": above,
            "checkpoint_value": float(curve[checkpoint_step - start_step]),
            "event": bool(_event_in_curve(curve)),
        }
    return {
        "start_step": start_step,
        "end_step": end_step,
        "checkpoint_step": checkpoint_step,
        "scenarios": out,
        "window_start_min": WINDOW_START_MIN,
        "minutes": [int((i) * STEP_MIN) for i in range(len(out["baseline"]["curve"]))],
    }


def _event_in_curve(curve: np.ndarray, min_steps: int = 6) -> bool:
    above = curve > 180
    run = 0
    for value in above:
        run = run + 1 if value else 0
        if run >= min_steps:
            return True
    return False


def score_stream(
    twin,
    simulated_curve: np.ndarray,
    start_step: int,
    checkpoint_step: int,
    extra_steps: np.ndarray | None = None,
) -> float:
    """Re-score the risk model on a stream with the simulated segment spliced in."""
    return twin.risk_with_simulated(simulated_curve, start_step, checkpoint_step, extra_steps)
