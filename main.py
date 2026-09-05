"""Customer Churn Prediction API.

Serves the logistic regression trained in
https://github.com/fcarruitero24/Customer-Churn-Prediction-End-to-End-ML-Pipeline

The model arrives here as `model_spec.json`: the scaler statistics, the category
order, 21 coefficients and an intercept. Scoring is a dot product and a sigmoid,
written in plain Python — so this service ships without scikit-learn, NumPy or
pandas. That keeps the deployment at a few kilobytes and the cold start close to
instant, which is what matters on serverless.

The exporter that produces the spec verifies it against scikit-learn on 400 real
customers and refuses to emit anything that disagrees by more than 1e-9.

Run locally:
    uvicorn main:app --reload
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

SPEC_PATH = Path(__file__).parent / "model_spec.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

THRESHOLDS = SPEC["risk_thresholds"]
BATCH_LIMIT = 1000

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Scores telecom customers for churn risk. Returns a probability, a "
        "binary prediction at the 0.5 threshold, and the risk band a retention "
        "team acts on."
    ),
    version="1.0.0",
)


# --------------------------------------------------------------------- model --
def score(customer: dict) -> float:
    """Churn probability for one customer, from the spec alone.

    Mirrors `src/export.py:score_one` in the training repository line for line.
    """
    z = SPEC["intercept"]
    k = 0
    for field in SPEC["numeric"]:
        z += SPEC["coef"][k] * (customer[field["name"]] - field["mean"]) / field["scale"]
        k += 1
    for block in SPEC["categorical"]:
        for category in block["categories"]:
            if customer[block["name"]] == category:
                z += SPEC["coef"][k]
            k += 1
    return 1.0 / (1.0 + math.exp(-z))


def risk_band(probability: float) -> str:
    """Map a probability to the band a retention team acts on.

    The cut points are asymmetric on purpose: catching a likely churner early is
    worth more than avoiding a false alarm, so "Medium" starts below 0.5.
    """
    if probability >= THRESHOLDS["high"]:
        return "High"
    if probability >= THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def predict(customer: dict) -> dict:
    probability = score(customer)
    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": int(probability >= 0.5),
        "risk_band": risk_band(probability),
    }


# ------------------------------------------------------------------ schemas --
class Customer(BaseModel):
    """One customer profile. Constraints mirror the training data."""

    tenure_months: int = Field(..., ge=0, le=120, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, le=500, description="Current monthly bill")
    total_charges: float = Field(..., ge=0, description="Lifetime billed amount")
    num_support_tickets: int = Field(..., ge=0, le=50)
    age: int = Field(..., ge=18, le=120)

    contract_type: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    gender: Literal["Female", "Male"]
    has_streaming: Literal["No", "Yes"]
    paperless_billing: Literal["No", "Yes"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 2, "monthly_charges": 95.0, "total_charges": 190.0,
                "num_support_tickets": 4, "age": 30,
                "contract_type": "Month-to-month", "internet_service": "Fiber optic",
                "payment_method": "Electronic check", "gender": "Female",
                "has_streaming": "No", "paperless_billing": "Yes",
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float = Field(..., description="Probability in [0, 1]")
    churn_prediction: int = Field(..., description="1 when probability >= 0.5")
    risk_band: Literal["Low", "Medium", "High"]


# ---------------------------------------------------------------- endpoints --
@app.get("/health")
def health() -> dict:
    """Readiness check."""
    return {"status": "ok", "model": SPEC["model"], "features": len(SPEC["coef"])}


@app.get("/model")
def model_info() -> dict:
    """What the service is serving, and the fields it expects."""
    return {
        "model": SPEC["model"],
        "coefficients": len(SPEC["coef"]),
        "risk_thresholds": THRESHOLDS,
        "numeric_features": [f["name"] for f in SPEC["numeric"]],
        "categorical_features": {
            block["name"]: block["categories"] for block in SPEC["categorical"]
        },
        "source": (
            "https://github.com/fcarruitero24/"
            "Customer-Churn-Prediction-End-to-End-ML-Pipeline"
        ),
    }


@app.post("/predict", response_model=Prediction)
def predict_one(customer: Customer) -> dict:
    """Score a single customer."""
    return predict(customer.model_dump())


@app.post("/predict/batch", response_model=list[Prediction])
def predict_many(customers: list[Customer]) -> list[dict]:
    """Score up to 1 000 customers in one call."""
    if not customers:
        raise HTTPException(status_code=422, detail="The customer list is empty.")
    if len(customers) > BATCH_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=f"Batch limit is {BATCH_LIMIT} customers; received {len(customers)}.",
        )
    return [predict(c.model_dump()) for c in customers]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing() -> str:
    return f"""
<!doctype html>
<html lang="es">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Customer Churn Prediction API</title>
<meta name="description" content="Regresión logística exportada a un JSON de 2.7 KB: predicción de churn sin scikit-learn ni runtime de ML en producción. FastAPI sobre Vercel.">
<link rel="canonical" href="https://churn-prediction-api-ruddy.vercel.app/">

<meta property="og:type" content="website">
<meta property="og:site_name" content="Fabrizio Carruitero">
<meta property="og:locale" content="es_PE">
<meta property="og:title" content="Customer Churn Prediction API">
<meta property="og:description" content="Regresión logística exportada a un JSON de 2.7 KB: predicción de churn sin scikit-learn ni runtime de ML en producción. FastAPI sobre Vercel.">
<meta property="og:image" content="https://churn-prediction-api-ruddy.vercel.app/preview.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Customer Churn Prediction API de Fabrizio Carruitero, construida con FastAPI">
<meta property="og:url" content="https://churn-prediction-api-ruddy.vercel.app/">

<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Customer Churn Prediction API">
<meta name="twitter:description" content="Regresión logística exportada a un JSON de 2.7 KB: predicción de churn sin scikit-learn ni runtime de ML en producción. FastAPI sobre Vercel.">
<meta name="twitter:image" content="https://churn-prediction-api-ruddy.vercel.app/preview.png">
<style>
  :root {{
    color-scheme: light;
    --bg: #f2f3f1; --card: #fdfdfc; --ink: #16181c; --dim: #5f6a68;
    --rule: #dcded9; --accent: #eb6834; --ok: #17a06f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --bg: #10131a; --card: #191d26; --ink: #eef0ee; --dim: #99a0a8;
      --rule: #2b313c; --accent: #d95926; --ok: #199e70;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink); line-height: 1.6;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 780px; margin: 0 auto; padding: 56px 22px 80px; }}
  h1 {{
    font-family: Georgia, serif; font-size: clamp(28px, 5vw, 40px);
    margin: 0 0 10px; letter-spacing: -.02em; line-height: 1.1;
  }}
  h2 {{ font-family: Georgia, serif; font-size: 21px; margin: 30px 0 10px; }}
  .lede {{ color: var(--dim); font-size: 17px; margin: 0 0 26px; max-width: 58ch; }}
  .pill {{
    display: inline-flex; align-items: center; gap: 8px; font-size: 11.5px;
    font-family: ui-monospace, Consolas, monospace; letter-spacing: .07em;
    text-transform: uppercase; color: var(--ok); border: 1px solid currentColor;
    padding: 4px 10px; border-radius: 2px; margin-bottom: 20px;
  }}
  .pill::before {{
    content: ""; width: 7px; height: 7px; background: currentColor; border-radius: 50%;
  }}
  table {{ border-collapse: collapse; width: 100%; margin: 0 0 8px; font-size: 14.5px; }}
  th, td {{ text-align: left; padding: 9px 14px 9px 0; border-bottom: 1px solid var(--rule); }}
  th {{
    font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
    color: var(--dim); font-family: ui-monospace, Consolas, monospace;
  }}
  code {{
    font-family: ui-monospace, Consolas, monospace; font-size: .92em;
    background: var(--card); padding: 2px 6px; border: 1px solid var(--rule);
    border-radius: 2px;
  }}
  pre {{
    background: var(--card); border: 1px solid var(--rule); border-radius: 3px;
    padding: 15px 17px; overflow-x: auto; font-size: 12.5px;
    font-family: ui-monospace, Consolas, monospace; line-height: 1.55;
  }}
  a {{ color: var(--accent); }}
  .cta {{
    display: inline-block; background: var(--accent); color: #fff;
    text-decoration: none; font-weight: 600; padding: 11px 20px;
    border-radius: 3px; margin: 2px 0 6px;
  }}
  footer {{
    margin-top: 34px; padding-top: 18px; border-top: 1px solid var(--rule);
    color: var(--dim); font-size: 13.5px;
  }}
</style>

<div class="wrap">
  <span class="pill">Service online</span>
  <h1>Customer Churn Prediction API</h1>
  <p class="lede">
    Predicts which telecom customers are about to cancel. Send a customer
    profile and get back a churn probability and the risk band a retention team
    acts on.
  </p>

  <a class="cta" href="/docs">Try the API in your browser →</a>

  <h2>Endpoints</h2>
  <table>
    <tr><th>Route</th><th>What it does</th></tr>
    <tr><td><code>GET /health</code></td><td>Readiness check</td></tr>
    <tr><td><code>GET /model</code></td><td>Which model is served and the fields it expects</td></tr>
    <tr><td><code>POST /predict</code></td><td>Score one customer</td></tr>
    <tr><td><code>POST /predict/batch</code></td><td>Score up to 1 000 customers</td></tr>
    <tr><td><code>GET /docs</code></td><td>Interactive documentation (Swagger)</td></tr>
  </table>

  <h2>Example</h2>
  <pre>curl -X POST https://churn-prediction-api-ruddy.vercel.app/predict \\
  -H "Content-Type: application/json" -d '{{
  "tenure_months": 2, "monthly_charges": 95.0, "total_charges": 190.0,
  "num_support_tickets": 4, "age": 30,
  "contract_type": "Month-to-month", "internet_service": "Fiber optic",
  "payment_method": "Electronic check", "gender": "Female",
  "has_streaming": "No", "paperless_billing": "Yes"
}}'

{{"churn_probability": 0.9583, "churn_prediction": 1, "risk_band": "High"}}</pre>

  <footer>
    Serving a <strong>{SPEC["model"]}</strong> with {len(SPEC["coef"])}
    coefficients, without scikit-learn or NumPy on board: the model is exported
    to JSON and evaluated with a weighted sum and a sigmoid.<br>
    Training and evaluation live in
    <a href="https://github.com/fcarruitero24/Customer-Churn-Prediction-End-to-End-ML-Pipeline">the
    project repository</a>.
  </footer>
</div>
"""
