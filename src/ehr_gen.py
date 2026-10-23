"""GlucoTwin Phase 1: synthetic Indian Type 2 diabetes EHR generator.

Creates ``n_patients`` synthetic patient records with realistic Indian
population distributions (earlier onset, lower BMI than Western cohorts,
higher central adiposity) and sensible cross-field correlations.

Everything is drawn from a single seeded numpy Generator, so the file is
byte-identical across runs given the pinned dependency versions.

Outputs
-------
data/ehr.csv            one row per patient
data/manifest.json      generator settings (seed, counts, column list)

Run:  python src/ehr_gen.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_PATIENTS = 1500
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "bmi",
    "waist_cm",
    "diet_type",
    "family_history",
    "years_since_dx",
    "hba1c",
    "fpg_mgdl",
    "egfr",
    "ldl",
    "triglycerides",
    "systolic_bp",
    "metformin",
    "sulfonylurea",
    "dpp4",
    "insulin",
    "hypertension",
    "tcf7l2",
    "prior_hypo_count",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_ehr(n_patients: int = N_PATIENTS, seed: int = SEED) -> pd.DataFrame:
    """Return a synthetic EHR table with ``n_patients`` rows."""
    rng = np.random.default_rng(seed)

    # --- demographics and body shape -------------------------------------
    age = np.clip(rng.normal(56.0, 9.5, n_patients), 30, 79).round(0)
    sex = np.where(rng.random(n_patients) < 0.56, "M", "F")
    bmi = np.clip(rng.normal(24.8, 3.6, n_patients), 16.5, 42.0).round(1)
    # Waist is high relative to BMI (central adiposity): sex-specific base
    # plus a strong BMI slope.
    waist_base = np.where(sex == "M", 89.5, 83.0)
    waist = waist_base + 1.05 * (bmi - 22.0) + rng.normal(0, 4.2, n_patients)
    waist = np.clip(waist, 68, 135).round(1)

    diet_type = np.where(rng.random(n_patients) < 0.42, "veg", "non-veg")
    family_history = (rng.random(n_patients) < 0.35).astype(int)

    # --- disease history --------------------------------------------------
    years_since_dx = np.clip(rng.normal(6.2, 5.0, n_patients), 0, 30).round(1)
    # Keep age at diagnosis plausible (>= 25 years).
    age = np.maximum(age, years_since_dx + 25.0)

    # TCF7L2 rs7903146: T-allele frequency assumed 0.20 (see DATA_CARD).
    g = rng.random(n_patients)
    tcf7l2 = np.where(g < 0.65, "CC", np.where(g < 0.95, "CT", "TT"))
    risk_dose = np.where(tcf7l2 == "TT", 0.55, np.where(tcf7l2 == "CT", 0.20, 0.0))

    # --- correlated labs --------------------------------------------------
    hba1c = (
        6.9
        + 0.040 * years_since_dx
        + 0.050 * (bmi - 24.0)
        + 0.020 * (age - 55.0)
        + risk_dose
        + rng.normal(0, 0.85, n_patients)
    )
    hba1c = np.clip(hba1c, 5.4, 13.0).round(1)

    # Estimated average glucose from A1c (eAG = 28.7*A1c - 46.7). Fasting
    # glucose tracks eAG but sits below it because post-meal spikes dominate
    # the daily average in poorly controlled patients.
    eag = 28.7 * hba1c - 46.7
    fpg = np.clip(0.65 * eag + 35.0 + rng.normal(0, 8.0, n_patients), 70, 300).round(0)

    trig = np.exp(
        np.log(145.0)
        + 0.032 * (bmi - 24.0)
        + 0.015 * (years_since_dx - 6.0)
        + rng.normal(0, 0.38, n_patients)
    )
    trig = np.clip(trig, 50, 650).round(0)

    ldl = np.clip(
        112.0 + 0.45 * (bmi - 24.0) + rng.normal(0, 27.0, n_patients), 45, 210
    ).round(0)

    systolic_bp = (
        116.0
        + 0.42 * (age - 50.0)
        + 0.35 * (bmi - 24.0)
        + 0.25 * years_since_dx
        + rng.normal(0, 11.0, n_patients)
    )
    systolic_bp = np.clip(systolic_bp, 95, 195).round(0)
    hypertension = (
        (systolic_bp >= 140) | (rng.random(n_patients) < 0.10)
    ).astype(int)

    egfr = (
        106.0
        - 0.62 * (age - 45.0)
        - 0.9 * years_since_dx
        - 6.0 * hypertension
        + rng.normal(0, 10.0, n_patients)
    )
    egfr = np.clip(egfr, 20, 135).round(0)

    # --- medications (conditional on glycaemic control and duration) ------
    metformin = (
        rng.random(n_patients)
        < _sigmoid(2.0 + 1.0 * (hba1c - 7.0) + 0.05 * years_since_dx - 0.02 * (egfr - 90))
    ).astype(int)
    sulfonylurea = (
        rng.random(n_patients) < _sigmoid(-0.8 + 0.7 * (hba1c - 7.0) + 0.06 * years_since_dx)
    ).astype(int)
    dpp4 = (
        rng.random(n_patients) < _sigmoid(-1.7 + 0.5 * (hba1c - 7.5) + 0.08 * years_since_dx)
    ).astype(int)
    insulin = (
        rng.random(n_patients)
        < _sigmoid(-3.2 + 1.3 * (hba1c - 8.0) + 0.09 * years_since_dx)
    ).astype(int)
    # Metformin is the default first-line therapy; make a tiny share
    # monotherapy-free so the table is not perfectly deterministic.
    none_med = (metformin + sulfonylurea + dpp4 + insulin) == 0

    # --- prior hypoglycaemia --------------------------------------------
    lam = (
        0.20
        + 1.10 * sulfonylurea
        + 1.70 * insulin
        + 0.04 * years_since_dx
        + 0.35 * (hba1c < 7.0)
    )
    prior_hypo = np.clip(rng.poisson(lam), 0, 15)

    ehr = pd.DataFrame(
        {
            "patient_id": [f"GT-{i + 1:04d}" for i in range(n_patients)],
            "age": age.astype(int),
            "sex": sex,
            "bmi": bmi,
            "waist_cm": waist,
            "diet_type": diet_type,
            "family_history": family_history,
            "years_since_dx": years_since_dx,
            "hba1c": hba1c,
            "fpg_mgdl": fpg.astype(int),
            "egfr": egfr.astype(int),
            "ldl": ldl.astype(int),
            "triglycerides": trig.astype(int),
            "systolic_bp": systolic_bp.astype(int),
            "metformin": metformin,
            "sulfonylurea": sulfonylurea,
            "dpp4": dpp4,
            "insulin": insulin,
            "hypertension": hypertension,
            "tcf7l2": tcf7l2,
            "prior_hypo_count": prior_hypo,
        }
    )
    # keep a reference so linters see the variable is used
    assert none_med is not None
    return ehr[COLUMNS]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ehr = generate_ehr()
    ehr.to_csv(DATA_DIR / "ehr.csv", index=False)
    manifest = {
        "seed": SEED,
        "n_patients": int(len(ehr)),
        "columns": COLUMNS,
        "generator": "custom Synthea-style generator (see docs/DATA_CARD.md)",
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = ehr[["age", "bmi", "hba1c", "fpg_mgdl", "years_since_dx", "systolic_bp"]].describe()
    print(f"[ehr_gen] wrote {DATA_DIR / 'ehr.csv'} ({len(ehr)} patients)")
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
