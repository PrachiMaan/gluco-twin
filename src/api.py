"""GlucoTwin reference API (FastAPI).

Serves the same twin the dashboard renders, for anyone who wants the model
behind an HTTP boundary instead of a browser:

  GET  /patients                     scored demo panel, highest risk first
  GET  /patients/{id}/twin           current state, projection, top factors
  POST /predict                      stream new CGM readings, get a fresh risk
  POST /whatif                       one counterfactual scenario for one patient

Run: make api  (uvicorn on http://127.0.0.1:8000, interactive docs at /docs)

This is a research prototype on synthetic data. It is not a medical device
and returns no diagnosis or treatment recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from twin import ModelBundle, PatientTwin
from whatif import SCENARIOS, run_scenarios

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULT_DIR = ROOT / "results"
PUBLIC = ROOT / "public" / "data"

STEP_MIN = 5
STEPS_PER_DAY = 24 * 60 // STEP_MIN
NOW_DAY = 13  # last day of the 14-day record (zero-based)
NOW_MINUTE = 20 * 60  # the demo clock stops before dinner, when risk is rising
NOW_STEP = NOW_DAY * STEPS_PER_DAY + NOW_MINUTE // STEP_MIN
SCENARIO_KEYS = {scenario.key: scenario for scenario in SCENARIOS}

app = FastAPI(
    title="GlucoTwin API",
    version="0.1.0",
    description=(
        "Digital twin for type 2 diabetes on synthetic data. Prediction and "
        "counterfactuals only; decision support, not medical advice."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_ehr: pd.DataFrame | None = None
_true_params: pd.DataFrame | None = None
_twins: dict[str, PatientTwin] = {}


def ehr() -> pd.DataFrame:
    global _ehr
    if _ehr is None:
        _ehr = pd.read_csv(DATA / "ehr.csv")
    return _ehr


def true_params() -> pd.DataFrame:
    global _true_params
    if _true_params is None:
        _true_params = pd.read_csv(DATA / "true_params.csv")
    return _true_params


def patient_ids() -> list[str]:
    """Demo panel from the web export, or the first rows of the EHR table."""
    cohort_path = PUBLIC / "cohort.json"
    if cohort_path.exists():
        cohort = json.loads(cohort_path.read_text())
        return [row["id"] for row in cohort["patients"]]
    return ehr()["patient_id"].head(24).tolist()


def twin_for(patient_id: str) -> PatientTwin:
    if patient_id in _twins:
        return _twins[patient_id]
    table = ehr()
    match = table.index[table["patient_id"] == patient_id]
    if len(match) == 0:
        raise HTTPException(status_code=404, detail=f"Unknown patient {patient_id}")
    twin = PatientTwin(int(match[0]), now_step=NOW_STEP)
    _twins[patient_id] = twin
    return twin


def _trajectory(twin: PatientTwin) -> dict:
    traj = twin.trajectory()
    return {
        "minutes": [i * STEP_MIN for i in range(len(traj["mean"]))],
        "mean": [round(float(v), 1) for v in traj["mean"]],
        "lo": [round(float(v), 1) for v in traj["lo"]],
        "hi": [round(float(v), 1) for v in traj["hi"]],
    }


def _summary(twin: PatientTwin) -> dict:
    risk = twin.risk_at()
    glucose = twin.glucose
    return {
        "patient_id": twin.patient_id,
        "risk_2h": round(float(risk), 4),
        "status": twin.status(risk),
        "observed_at": f"20:00 on day {NOW_DAY + 1} of 14 (synthetic)",
        "observed_at_step": int(twin.now_step),
        "glucose_mgdl": int(round(float(glucose[-1]))),
        "trend_30m": int(round(float(glucose[-1] - glucose[-7]))),
        "time_in_range_24h": {k: round(float(v), 4) for k, v in twin.time_in_range(24).items()},
        "ehr": {
            "age": int(twin.ehr_row["age"]),
            "sex": str(twin.ehr_row["sex"]),
            "bmi": round(float(twin.ehr_row["bmi"]), 1),
            "hba1c": round(float(twin.ehr_row["hba1c"]), 1),
            "tcf7l2": str(twin.ehr_row["tcf7l2"]),
            "insulin": bool(twin.ehr_row["insulin"]),
        },
        "fitted": {k: round(float(v), 3) for k, v in twin.params.items() if k != "profile"},
    }


class Reading(BaseModel):
    patient_id: str = Field(..., examples=["GT-0513"])
    glucose: list[float] = Field(
        ...,
        min_length=72,
        description="CGM readings in mg/dL, 5-minute cadence, oldest first (at least 6 hours).",
        examples=[[142.0, 148.0, 155.0]],
    )


class WhatIfRequest(BaseModel):
    patient_id: str = Field(..., examples=["GT-0513"])
    scenario: str = Field(..., examples=["walk_after_dinner"])


@app.get("/health")
def health() -> dict:
    metrics = {}
    path = RESULT_DIR / "metrics.json"
    if path.exists():
        metrics = json.loads(path.read_text())
    return {
        "status": "ok",
        "deployed_model": metrics.get("deployed_model", "lgbm_fused"),
        "panel_size": len(patient_ids()),
        "synthetic_data": True,
    }


@app.get("/patients")
def patients(limit: int = 60) -> dict:
    limit = max(1, min(limit, 200))
    rows = []
    for patient_id in patient_ids()[:limit]:
        rows.append(_summary(twin_for(patient_id)))
    rows.sort(key=lambda row: row["risk_2h"], reverse=True)
    return {"count": len(rows), "patients": rows}


@app.get("/patients/{patient_id}/twin")
def patient_twin(patient_id: str) -> dict:
    twin = twin_for(patient_id)
    return {
        "summary": _summary(twin),
        "trajectory": _trajectory(twin),
        "top_factors": twin.top_factors(k=5),
    }


@app.post("/predict")
def predict(reading: Reading) -> dict:
    """Splice new readings onto the patient's history and re-score."""
    twin = twin_for(reading.patient_id)
    if any(not np.isfinite(value) for value in reading.glucose):
        raise HTTPException(status_code=422, detail="Glucose readings must be finite numbers.")
    if any(value < 20 or value > 600 for value in reading.glucose):
        raise HTTPException(status_code=422, detail="Glucose readings outside 20-600 mg/dL.")
    twin.update(np.asarray(reading.glucose, dtype=np.float32))
    return {
        "summary": _summary(twin),
        "trajectory": _trajectory(twin),
        "top_factors": twin.top_factors(k=5),
    }


@app.post("/whatif")
def whatif(request: WhatIfRequest) -> dict:
    if request.scenario not in SCENARIO_KEYS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown scenario. Choose one of {sorted(SCENARIO_KEYS)}",
        )
    twin = twin_for(request.patient_id)
    index = twin.index
    sim = run_scenarios(twin.ehr_row, true_params().iloc[index], twin.params, index, twin.glucose)

    window_start = sim["start_step"]
    checkpoints = [window_start + (minute - sim["window_start_min"]) // STEP_MIN for minute in range(20 * 60, 24 * 60, 30)]
    baseline_probs = twin.risk_profile_with_simulated(sim["scenarios"]["baseline"]["curve"], window_start, checkpoints)
    scenario_probs = twin.risk_profile_with_simulated(
        sim["scenarios"][request.scenario]["curve"], window_start, checkpoints
    )
    baseline = sim["scenarios"]["baseline"]
    chosen = sim["scenarios"][request.scenario]
    return {
        "patient_id": twin.patient_id,
        "scenario": request.scenario,
        "label": SCENARIO_KEYS[request.scenario].label,
        "description": SCENARIO_KEYS[request.scenario].description,
        "window_start_minute_of_day": sim["window_start_min"],
        "minutes": sim["minutes"],
        "baseline": {
            "curve": [round(float(v), 1) for v in baseline["curve"]],
            "peak": round(float(baseline["peak"]), 1),
            "minutes_above_180": int(baseline["minutes_above_180"]),
            "event_probability": round(float(np.mean(baseline_probs)), 4),
        },
        "scenario_result": {
            "curve": [round(float(v), 1) for v in chosen["curve"]],
            "peak": round(float(chosen["peak"]), 1),
            "minutes_above_180": int(chosen["minutes_above_180"]),
            "event_probability": round(float(np.mean(scenario_probs)), 4),
        },
        "delta": {
            "peak_mgdl": round(float(chosen["peak"] - baseline["peak"]), 1),
            "minutes_above_180": int(chosen["minutes_above_180"] - baseline["minutes_above_180"]),
            "event_probability": round(float(np.mean(scenario_probs) - np.mean(baseline_probs)), 4),
        },
        "disclaimer": "Simulation for decision support with this patient's fitted parameters. Not medical advice.",
    }


@app.get("/model")
def model_card() -> dict:
    """The measured numbers the API's model is running with."""
    path = RESULT_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run `make eval` first.")
    metrics = json.loads(path.read_text())
    return {
        "deployed_model": metrics.get("deployed_model"),
        "test_patients": metrics.get("test_patients"),
        "dense": metrics.get("dense_deployed"),
        "ablation": metrics.get("ablation"),
        "limitations": "Synthetic cohort, retrospective split, not clinically validated.",
    }
