"""GlucoTwin Phase 4 (support): export artifacts for the web dashboard.

Ranks the whole synthetic cohort by their current 2-hour risk, then writes a
compact bundle for a 60-patient clinic panel:

  public/data/cohort.json          panel list + cohort statistics
  public/data/patients/<id>.json   traces, risk, forecasts, factors, alerts, what-if
  public/data/metrics.json         evaluation numbers (copied from results/)
  public/data/meta.json            build metadata

Everything here is computed by the models trained in Phase 2. The web app only
renders these numbers; it does not invent any.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import event_runs
from features import STEP_MIN, STEPS_PER_DAY, _features_for_patient, interpolate_gaps
from twin import ModelBundle, PatientTwin
from whatif import SCENARIOS, run_scenarios

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "data"
RESULT_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

PANEL_SIZE = 60
N_TOP = 20          # highest-risk patients
N_MID = 15          # moderate-risk patients
N_LOW = 25          # stable patients
THRESHOLD = 0.5
COOLDOWN_STEPS = 24  # 2 hours
NOW_MINUTE = 20 * 60  # the twin's clock stops at 20:00 on the last day, before dinner


NOW_STEP = (14 - 1) * STEPS_PER_DAY + NOW_MINUTE // STEP_MIN


def risk_for_patient(index: int, ehr, streams, bundle) -> float:
    out = _features_for_patient(
        streams["glucose"][index], streams["hr"][index], streams["hrv"][index],
        streams["steps"][index], streams["stage"][index], ehr.iloc[index],
    )
    return float(bundle.proba(out["X"][NOW_STEP][None, :])[0])


def downsample(values: np.ndarray, stride: int) -> list[int]:
    return [int(round(float(v))) for v in values[::stride]]


def build_bundle(idx: int, ehr, streams, true_params, bundle) -> dict:
    now_step = NOW_STEP
    twin = PatientTwin(idx, ehr=ehr, streams=streams, bundle=bundle, now_step=now_step)
    risk_now = twin.risk_at()
    risk_trace = bundle.proba(twin.features[: now_step + 1 : 3])  # 15-minute cadence
    g = twin.glucose
    day_start = (14 - 1) * STEPS_PER_DAY

    # replay checkpoints every 30 minutes across the current day
    checkpoints = []
    for minute in range(0, NOW_MINUTE + 1, 30):
        step = day_start + minute // STEP_MIN
        traj = twin.trajectory(step=step)
        factors = twin.top_factors(step, k=5)
        checkpoints.append(
            {
                "minute": minute,
                "risk": round(float(bundle.proba(twin.features[step][None, :])[0]), 4),
                "glucose": int(round(float(g[step]))),
                "forecast": {
                    "mean": [int(round(v)) for v in traj["mean"]],
                    "lo": [int(round(v)) for v in traj["lo"]],
                    "hi": [int(round(v)) for v in traj["hi"]],
                },
                "factors": [
                    {
                        "label": f["label"],
                        "feature": f["feature"],
                        "value": round(f["value"], 2),
                        "contribution": round(f["contribution"], 4),
                        "direction": f["direction"],
                    }
                    for f in factors
                ],
            }
        )

    # alerts over the last 3 days from the model's own trace
    alert_steps = []
    last_alarm = -10**9
    for step in range(max(0, now_step - 3 * STEPS_PER_DAY), now_step, 3):
        p = float(risk_trace[step // 3]) if step // 3 < len(risk_trace) else 0.0
        if p > THRESHOLD and step - last_alarm > COOLDOWN_STEPS and g[step] <= 180:
            alert_steps.append(step)
            last_alarm = step
    events = event_runs(streams["glucose"][idx])
    alerts = []
    for step in alert_steps:
        event = next((e for e in events if 0 <= e - step <= 24), None)
        alerts.append(
            {
                "minute_of_day": int((step % STEPS_PER_DAY) * STEP_MIN),
                "day": int(step // STEPS_PER_DAY),
                "risk": round(float(risk_trace[step // 3]), 4) if step // 3 < len(risk_trace) else 0.0,
                "lead_min": int((event - step) * STEP_MIN) if event else None,
                "glucose": int(round(float(g[step]))),
                "matched": event is not None,
                "factors": [
                    {"label": f["label"], "contribution": round(f["contribution"], 4), "direction": f["direction"]}
                    for f in twin.top_factors(step, k=3)
                ],
            }
        )

    # what-if on the current evening (17:00 to midnight), scored across the evening
    tp_row = true_params.iloc[idx]
    sim = run_scenarios(twin.ehr_row, tp_row, twin.params, idx, twin.glucose)
    score_steps = [
        sim["start_step"] + (m - sim["window_start_min"]) // STEP_MIN
        for m in range(20 * 60, 24 * 60, 30)
    ]
    base_probs = twin.risk_profile_with_simulated(
        sim["scenarios"]["baseline"]["curve"], sim["start_step"], score_steps
    )
    base_prob = float(np.mean(base_probs))
    whatif = {
        "start_min": sim["window_start_min"],
        "checkpoint_min": 22 * 60 + 5,
        "minutes": [int((i) * STEP_MIN) for i in range(len(sim["scenarios"]["baseline"]["curve"]))],
        "baseline": {
            "curve": [int(round(v)) for v in sim["scenarios"]["baseline"]["curve"]],
            "peak": round(sim["scenarios"]["baseline"]["peak"], 1),
            "minutes_above_180": sim["scenarios"]["baseline"]["minutes_above_180"],
            "prob": round(base_prob, 4),
            "risk_profile": [round(p, 4) for p in base_probs],
        },
        "scenarios": {},
    }
    for sc in SCENARIOS:
        s = sim["scenarios"][sc.key]
        probs = twin.risk_profile_with_simulated(s["curve"], sim["start_step"], score_steps)
        prob = float(np.mean(probs))
        whatif["scenarios"][sc.key] = {
            "label": sc.label,
            "description": sc.description,
            "curve": [int(round(v)) for v in s["curve"]],
            "peak": round(s["peak"], 1),
            "minutes_above_180": s["minutes_above_180"],
            "prob": round(prob, 4),
            "delta_prob": round(prob - base_prob, 4),
            "delta_peak": round(s["peak"] - sim["scenarios"]["baseline"]["peak"], 1),
            "delta_minutes_above_180": s["minutes_above_180"] - sim["scenarios"]["baseline"]["minutes_above_180"],
        }

    # wearable daily summary over the 14 days
    step_arr = streams["steps"][idx]
    hr_arr = streams["hr"][idx]
    hrv_arr = streams["hrv"][idx] / 10.0
    stage_arr = streams["stage"][idx]
    daily = []
    for d in range(14):
        sl = slice(d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY)
        asleep = stage_arr[sl] > 0
        daily.append(
            {
                "day": d,
                "steps": int(step_arr[sl].sum()),
                "hr_mean": round(float(hr_arr[sl].mean()), 1),
                "hrv_mean": round(float(hrv_arr[sl][asleep].mean()) if asleep.any() else float(hrv_arr[sl].mean()), 1),
                "sleep_h": round(float(asleep.sum() * STEP_MIN / 60.0), 2),
                "glucose_mean": round(float(g[sl].mean()), 1),
            }
        )

    replay_day = {
        "day": 14 - 1,
        "step_min": STEP_MIN,
        "glucose": downsample(g[day_start : now_step + 1], 1),
        "hr": [int(v) for v in hr_arr[day_start : now_step + 1]],
        "hrv": [int(v) for v in hrv_arr[day_start : now_step + 1]],
        "steps": [int(v) for v in step_arr[day_start : now_step + 1]],
        "stage": [int(v) for v in stage_arr[day_start : now_step + 1]],
    }

    tir = twin.time_in_range(24)
    return {
        "id": twin.patient_id,
        "index": int(idx),
        "ehr": {
            "age": int(twin.ehr_row["age"]),
            "sex": str(twin.ehr_row["sex"]),
            "bmi": float(twin.ehr_row["bmi"]),
            "waist_cm": float(twin.ehr_row["waist_cm"]),
            "diet_type": str(twin.ehr_row["diet_type"]),
            "family_history": int(twin.ehr_row["family_history"]),
            "years_since_dx": float(twin.ehr_row["years_since_dx"]),
            "hba1c": float(twin.ehr_row["hba1c"]),
            "fpg_mgdl": int(twin.ehr_row["fpg_mgdl"]),
            "egfr": int(twin.ehr_row["egfr"]),
            "ldl": int(twin.ehr_row["ldl"]),
            "triglycerides": int(twin.ehr_row["triglycerides"]),
            "systolic_bp": int(twin.ehr_row["systolic_bp"]),
            "metformin": int(twin.ehr_row["metformin"]),
            "sulfonylurea": int(twin.ehr_row["sulfonylurea"]),
            "dpp4": int(twin.ehr_row["dpp4"]),
            "insulin": int(twin.ehr_row["insulin"]),
            "hypertension": int(twin.ehr_row["hypertension"]),
            "tcf7l2": str(twin.ehr_row["tcf7l2"]),
            "prior_hypo_count": int(twin.ehr_row["prior_hypo_count"]),
        },
        "params": {k: round(float(v), 4) for k, v in twin.params.items() if k != "profile"},
        "profile": {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in twin.params["profile"].items()},
        "summary": {
            "risk": round(risk_now, 4),
            "status": twin.status(risk_now),
            "glucose": int(round(float(g[now_step]))),
            "trend_30m": round(float(g[now_step] - g[now_step - 6]), 1),
            "now_minute": NOW_MINUTE,
            "tir": {k: round(v, 4) if isinstance(v, float) else v for k, v in tir.items()},
            "risk_max_24h": round(float(np.max(risk_trace[-96:])) if len(risk_trace) else 0.0, 4),
            "alerts_72h": len(alerts),
        },
        "series": {
            "glucose_14d": {"step_min": 15, "values": downsample(g, 3)},
            "recent": {"hours": 48, "step_min": 5, "values": downsample(g[-48 * 12 :], 1)},
            "replay_day": replay_day,
            "wearables_daily": daily,
        },
        "risk_trace": {"step_min": 15, "values": [int(round(v * 1000)) for v in risk_trace]},
        "replay_checkpoints": checkpoints,
        "alerts": alerts,
        "whatif": whatif,
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "patients").mkdir(parents=True, exist_ok=True)

    ehr = pd.read_csv(DATA_DIR / "ehr.csv")
    true_params = pd.read_csv(DATA_DIR / "true_params.csv")
    streams = dict(np.load(DATA_DIR / "cgm.npz"))
    bundle = ModelBundle.get()

    print(f"[export] scoring {len(ehr)} patients ...")
    risks = np.array([risk_for_patient(i, ehr, streams, bundle) for i in range(len(ehr))])
    order = np.argsort(-risks)
    rng = np.random.default_rng(7)
    top = list(order[:N_TOP])
    mid_pool = [int(i) for i in order if 0.15 <= risks[i] < 0.9]
    low_pool = [int(i) for i in order if risks[i] < 0.15]
    mid = list(rng.choice(mid_pool, size=min(N_MID, len(mid_pool)), replace=False)) if mid_pool else []
    low = list(rng.choice(low_pool, size=min(N_LOW, len(low_pool)), replace=False))
    panel = sorted(top + mid + low, key=lambda i: -risks[int(i)])

    patients_meta = []
    for idx in panel:
        payload = build_bundle(int(idx), ehr, streams, true_params, bundle)
        (OUT / "patients" / f"{payload['id']}.json").write_text(json.dumps(payload, separators=(",", ":")))
        patients_meta.append(
            {
                "id": payload["id"],
                "age": payload["ehr"]["age"],
                "sex": payload["ehr"]["sex"],
                "hba1c": payload["ehr"]["hba1c"],
                "bmi": payload["ehr"]["bmi"],
                "risk": payload["summary"]["risk"],
                "status": payload["summary"]["status"],
                "glucose": payload["summary"]["glucose"],
                "trend_30m": payload["summary"]["trend_30m"],
                "tir_in_range": payload["summary"]["tir"]["in_range"],
                "alerts_72h": payload["summary"]["alerts_72h"],
            }
        )
        print(f"[export]   {payload['id']} risk {payload['summary']['risk']:.2f}")

    cohort = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "cohort": {
            "n_patients": int(len(ehr)),
            "n_days": 14,
            "step_min": STEP_MIN,
            "panel_size": len(panel),
            "mean_hba1c": round(float(ehr["hba1c"].mean()), 2),
            "mean_age": round(float(ehr["age"].mean()), 1),
            "mean_bmi": round(float(ehr["bmi"].mean()), 2),
            "synthetic": True,
        },
        "patients": patients_meta,
    }
    (OUT / "cohort.json").write_text(json.dumps(cohort, indent=1))

    metrics = json.loads((RESULT_DIR / "metrics.json").read_text())
    ablation = pd.read_csv(RESULT_DIR / "ablation.csv").to_dict(orient="records")
    recovery_path = RESULT_DIR / "param_recovery.json"
    metrics["ablation"] = ablation
    metrics["param_recovery"] = json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[export] done in {time.time() - t0:.1f}s, panel {len(panel)}")


if __name__ == "__main__":
    main()
