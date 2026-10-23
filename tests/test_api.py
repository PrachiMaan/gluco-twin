"""API contract tests. They build a real twin, so they take a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def patient_id(client: TestClient) -> str:
    response = client.get("/patients", params={"limit": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    return payload["patients"][0]["patient_id"]


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["synthetic_data"] is True
    assert body["deployed_model"] in {"lgbm_fused", "fusion_net"}


def test_panel_sorted_by_risk(client: TestClient) -> None:
    response = client.get("/patients", params={"limit": 6})
    assert response.status_code == 200
    rows = response.json()["patients"]
    assert len(rows) == 6
    risks = [row["risk_2h"] for row in rows]
    assert risks == sorted(risks, reverse=True)
    for row in rows:
        assert 0.0 <= row["risk_2h"] <= 1.0
        assert row["status"] in {"stable", "watch", "critical"}
        assert 40 <= row["glucose_mgdl"] <= 450
        assert row["time_in_range_24h"]["n"] > 0


def test_twin_has_projection_and_reasons(client: TestClient, patient_id: str) -> None:
    response = client.get(f"/patients/{patient_id}/twin")
    assert response.status_code == 200
    body = response.json()
    trajectory = body["trajectory"]
    # 5-minute cadence for the next two hours, starting at the current reading
    assert len(trajectory["mean"]) == len(trajectory["lo"]) == len(trajectory["hi"]) == 24
    assert len(trajectory["minutes"]) == 24
    assert trajectory["minutes"][-1] == 115
    for lo, mean, hi in zip(trajectory["lo"], trajectory["mean"], trajectory["hi"]):
        assert lo <= mean <= hi
    factors = body["top_factors"]
    assert 1 <= len(factors) <= 5
    assert all(factor["label"] for factor in factors)
    assert all(factor["direction"] in {"raises", "lowers"} for factor in factors)


def test_predict_updates_the_twin(client: TestClient, patient_id: str) -> None:
    response = client.post(
        "/predict",
        json={"patient_id": patient_id, "glucose": [176.0] * 72},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["glucose_mgdl"] == 176
    assert 0.0 <= body["summary"]["risk_2h"] <= 1.0
    assert len(body["trajectory"]["mean"]) == 24


def test_predict_rejects_impossible_readings(client: TestClient, patient_id: str) -> None:
    response = client.post(
        "/predict",
        json={"patient_id": patient_id, "glucose": [900.0] * 72},
    )
    assert response.status_code == 422


def test_predict_rejects_short_windows(client: TestClient, patient_id: str) -> None:
    response = client.post(
        "/predict",
        json={"patient_id": patient_id, "glucose": [140.0] * 10},
    )
    assert response.status_code == 422


def test_whatif_returns_a_paired_curve(client: TestClient, patient_id: str) -> None:
    response = client.post(
        "/whatif",
        json={"patient_id": patient_id, "scenario": "walk_after_dinner"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["baseline"]["curve"]) == len(body["scenario_result"]["curve"]) > 0
    assert body["delta"]["peak_mgdl"] == pytest.approx(
        body["scenario_result"]["peak"] - body["baseline"]["peak"], abs=0.15
    )
    assert 0.0 <= body["baseline"]["event_probability"] <= 1.0
    assert 0.0 <= body["scenario_result"]["event_probability"] <= 1.0


def test_whatif_rejects_unknown_scenario(client: TestClient, patient_id: str) -> None:
    response = client.post(
        "/whatif",
        json={"patient_id": patient_id, "scenario": "magic_pill"},
    )
    assert response.status_code == 422


def test_unknown_patient(client: TestClient) -> None:
    assert client.get("/patients/GT-9999/twin").status_code == 404
