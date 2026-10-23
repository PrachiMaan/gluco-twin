"""Phase 1 tests: synthetic EHR + CGM simulator.

Run:  python -m pytest tests/test_data.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgm_sim import STEPS_PER_DAY, simulate  # noqa: E402
from ehr_gen import generate_ehr  # noqa: E402

N = 160
DAYS = 14


@pytest.fixture(scope="session")
def ehr() -> pd.DataFrame:
    return generate_ehr(n_patients=N, seed=11)


@pytest.fixture(scope="session")
def sim(ehr: pd.DataFrame) -> dict:
    return simulate(ehr, days=DAYS, seed=42, chunk=160)


def test_ehr_shape_and_ranges(ehr: pd.DataFrame) -> None:
    assert len(ehr) == N
    assert ehr["patient_id"].is_unique
    assert ehr["age"].between(30, 80).all() or ehr["age"].between(30, 85).all()
    assert ehr["bmi"].between(16, 43).all()
    assert ehr["hba1c"].between(5, 13.5).all()
    assert ehr["fpg_mgdl"].between(70, 300).all()
    assert ehr["egfr"].between(20, 135).all()
    assert ehr["systolic_bp"].between(95, 195).all()
    assert set(ehr["sex"]) <= {"M", "F"}
    assert set(ehr["diet_type"]) <= {"veg", "non-veg"}
    assert set(ehr["tcf7l2"]) <= {"CC", "CT", "TT"}
    for med in ["metformin", "sulfonylurea", "dpp4", "insulin", "hypertension"]:
        assert set(ehr[med].unique()) <= {0, 1}
    # clinically plausible prevalence of common comorbidities
    assert 0.5 < ehr["metformin"].mean() < 0.95
    assert 0.15 < ehr["hypertension"].mean() < 0.7


def test_ehr_is_deterministic() -> None:
    a = generate_ehr(n_patients=40, seed=5)
    b = generate_ehr(n_patients=40, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_ehr_correlations_are_sensible(ehr: pd.DataFrame) -> None:
    r_a1c_fpg = np.corrcoef(ehr["hba1c"], ehr["fpg_mgdl"])[0, 1]
    r_bmi_trig = np.corrcoef(ehr["bmi"], ehr["triglycerides"])[0, 1]
    r_bmi_waist = np.corrcoef(ehr["bmi"], ehr["waist_cm"])[0, 1]
    r_age_egfr = np.corrcoef(ehr["age"], ehr["egfr"])[0, 1]
    assert r_a1c_fpg > 0.8
    assert r_bmi_trig > 0.2
    assert r_bmi_waist > 0.6
    assert r_age_egfr < -0.2


def test_simulator_shapes_and_ranges(sim: dict) -> None:
    n, t = N, DAYS * STEPS_PER_DAY
    assert sim["glucose"].shape == (n, t)
    assert sim["hr"].shape == (n, t)
    assert sim["hrv"].shape == (n, t)
    assert sim["steps"].shape == (n, t)
    assert sim["stage"].shape == (n, t)

    g = sim["glucose"].astype(np.float32)
    valid = g > 0
    assert valid.mean() > 0.95  # less than 5% dropout
    assert g[valid].min() >= 40
    assert g[valid].max() <= 450
    assert 100 < g[valid].mean() < 200
    assert sim["hr"].min() >= 30 and sim["hr"].max() <= 200
    assert sim["hrv"].max() <= 1500
    assert sim["steps"].max() <= 2000
    assert set(np.unique(sim["stage"])) <= {0, 1, 2, 3, 4}


def test_simulator_is_deterministic(ehr: pd.DataFrame) -> None:
    a = simulate(ehr.head(20), days=7, seed=42)
    b = simulate(ehr.head(20), days=7, seed=42)
    assert np.array_equal(a["glucose"], b["glucose"])
    assert np.array_equal(a["steps"], b["steps"])


def test_ehr_cgm_linkage(sim: dict, ehr: pd.DataFrame) -> None:
    g = sim["glucose"].astype(np.float32)
    g[g < 0] = np.nan
    hod = (np.arange(g.shape[1]) % STEPS_PER_DAY) / 12.0
    night = (hod >= 2) & (hod < 5)
    mean_g = np.nanmean(g, axis=1)
    fasting = np.nanmean(g[:, night], axis=1)

    r_mean_a1c = np.corrcoef(mean_g, ehr["hba1c"])[0, 1]
    r_mean_fpg = np.corrcoef(mean_g, ehr["fpg_mgdl"])[0, 1]
    r_fast_fpg = np.corrcoef(fasting, ehr["fpg_mgdl"])[0, 1]

    assert r_mean_a1c > 0.45, f"mean glucose vs HbA1c correlation too low: {r_mean_a1c:.3f}"
    assert r_mean_fpg > 0.5
    assert r_fast_fpg > 0.5
    assert np.median(np.abs(fasting - ehr["fpg_mgdl"])) < 15


def test_event_label_prevalence(sim: dict) -> None:
    """The prediction target should be a meaningful minority, not everything."""
    g = sim["glucose"].astype(np.float32)
    g[g < 0] = np.nan
    above = (g > 180).astype(np.int8)
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(above, 6, axis=1)
    run = np.concatenate(
        [np.zeros((g.shape[0], 5), np.int8), (windows.sum(axis=2) == 6).astype(np.int8)], axis=1
    )
    future_any = np.zeros_like(run, dtype=bool)
    slots = np.zeros((g.shape[0], 24), dtype=bool)
    for t in range(g.shape[1] - 1, -1, -1):
        slots[:, t % 24] = run[:, t]
        future_any[:, t] = slots.any(axis=1)
    label = np.zeros_like(future_any)
    label[:, :-1] = future_any[:, 1:]
    prevalence = float(label.mean())
    assert 0.05 < prevalence < 0.65, f"event prevalence {prevalence:.2%} outside sane range"


def test_wearable_coupling(sim: dict) -> None:
    hr = sim["hr"].astype(np.float32)
    steps = sim["steps"].astype(np.float32)
    hrv = sim["hrv"].astype(np.float32) / 10.0
    stage = sim["stage"]

    assert hr[steps >= 350].mean() > hr[steps < 60].mean() + 10
    assert hrv[stage > 0].mean() > hrv[stage == 0].mean()
    daily_steps = steps.sum(axis=1) / DAYS
    assert 2000 < np.median(daily_steps) < 20000


def test_post_dinner_walk_lowers_glucose(sim: dict) -> None:
    """The wearable -> physiology coupling: walking after dinner blunts the peak."""
    g = sim["glucose"].astype(np.float32)
    g[g < 0] = np.nan
    hod = (np.arange(g.shape[1]) % STEPS_PER_DAY) / 12.0
    pre = (hod >= 19) & (hod < 20.4)
    dinner = (hod >= 20.7) & (hod < 23)
    rise = np.nanmax(g[:, dinner], axis=1) - np.nanmedian(g[:, pre], axis=1)
    walks = sim["params"]["post_dinner_walk"].to_numpy()
    walked = rise[walks == 1].mean()
    did_not = rise[walks == 0].mean()
    assert walked < did_not, f"walkers {walked:.1f} vs non-walkers {did_not:.1f}"
