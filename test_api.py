"""Tests for the churn API.

The critical one is `test_matches_training_repository`: it pins the API's answer
to the value scikit-learn produces in the training repository. If the spec is
ever replaced with a mismatched export, this fails instead of the service
quietly returning wrong probabilities.

Run with:  pytest -q
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import BATCH_LIMIT, SPEC, app, risk_band, score

client = TestClient(app)

RISKY = {
    "tenure_months": 2, "monthly_charges": 95.0, "total_charges": 190.0,
    "num_support_tickets": 4, "age": 30,
    "contract_type": "Month-to-month", "internet_service": "Fiber optic",
    "payment_method": "Electronic check", "gender": "Female",
    "has_streaming": "No", "paperless_billing": "Yes",
}

LOYAL = {
    "tenure_months": 68, "monthly_charges": 24.0, "total_charges": 1632.0,
    "num_support_tickets": 0, "age": 62,
    "contract_type": "Two year", "internet_service": "DSL",
    "payment_method": "Bank transfer", "gender": "Male",
    "has_streaming": "No", "paperless_billing": "No",
}


# ------------------------------------------------------------------- scoring --
def test_matches_training_repository():
    """scikit-learn returns 0.9583 for this customer. So must we."""
    assert score(RISKY) == pytest.approx(0.9583, abs=5e-5)


def test_probability_is_in_range():
    for customer in (RISKY, LOYAL):
        assert 0.0 <= score(customer) <= 1.0


def test_risky_scores_above_loyal():
    assert score(RISKY) > score(LOYAL)


def test_month_to_month_raises_risk():
    """The dominant coefficient has to actually dominate."""
    two_year = {**RISKY, "contract_type": "Two year"}
    assert score(RISKY) > score(two_year)


def test_more_tickets_raise_risk():
    calm = {**RISKY, "num_support_tickets": 0}
    assert score(RISKY) > score(calm)


def test_longer_tenure_lowers_risk():
    veteran = {**RISKY, "tenure_months": 60}
    assert score(veteran) < score(RISKY)


@pytest.mark.parametrize(
    "probability,expected",
    [(0.0, "Low"), (0.34, "Low"), (0.35, "Medium"),
     (0.59, "Medium"), (0.60, "High"), (1.0, "High")],
)
def test_risk_band_boundaries(probability, expected):
    assert risk_band(probability) == expected


def test_spec_is_internally_consistent():
    """One coefficient per column: numeric fields plus every category."""
    expected = len(SPEC["numeric"]) + sum(
        len(b["categories"]) for b in SPEC["categorical"]
    )
    assert len(SPEC["coef"]) == expected == len(SPEC["feature_order"])


# ----------------------------------------------------------------- endpoints --
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    body = client.get("/model").json()
    assert body["model"] == "LogisticRegression"
    assert "contract_type" in body["categorical_features"]


def test_landing_page_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "Churn" in response.text


def test_predict():
    body = client.post("/predict", json=RISKY).json()
    assert set(body) == {"churn_probability", "churn_prediction", "risk_band"}
    assert body["risk_band"] == "High"
    assert body["churn_prediction"] == 1


def test_predict_loyal_customer_is_low_risk():
    assert client.post("/predict", json=LOYAL).json()["risk_band"] == "Low"


def test_batch():
    response = client.post("/predict/batch", json=[RISKY, LOYAL])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_batch_rejects_empty():
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_batch_rejects_oversized():
    response = client.post("/predict/batch", json=[RISKY] * (BATCH_LIMIT + 1))
    assert response.status_code == 422


def test_rejects_unknown_category():
    bad = {**RISKY, "contract_type": "Three year"}
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_negative_tenure():
    assert client.post("/predict", json={**RISKY, "tenure_months": -1}).status_code == 422


def test_rejects_missing_field():
    incomplete = {k: v for k, v in RISKY.items() if k != "age"}
    assert client.post("/predict", json=incomplete).status_code == 422
