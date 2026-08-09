"""
Integration tests for the FastAPI app (backend/main.py), exercised through
FastAPI's TestClient -- real request/response cycle, no network needed.
"""
import io

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers the app's lifespan
    # (startup/shutdown) -- without it, the models never get loaded and
    # every request would hit a cold, unloaded app.
    with TestClient(app) as c:
        yield c


# --- Meta -----------------------------------------------------------------

def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["freight_model_loaded"] is True
    assert body["invoice_flag_model_loaded"] is True


def test_models_info_endpoint(client):
    res = client.get("/api/models/info")
    assert res.status_code == 200
    body = res.json()
    assert "freight_cost_model" in body
    assert "invoice_flag_model" in body
    assert body["freight_cost_model"]["algorithm"] == "Linear Regression"


def test_docs_available(client):
    assert client.get("/api/docs").status_code == 200
    assert client.get("/api/openapi.json").status_code == 200


def test_frontend_served_at_root(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Invoice Intelligence" in res.text


# --- Freight cost -----------------------------------------------------------

def test_freight_predict_valid(client):
    res = client.post("/api/predict/freight-cost", json={"dollars": 18500})
    assert res.status_code == 200
    body = res.json()
    assert body["dollars"] == 18500.0
    assert body["predicted_freight"] > 0
    assert body["freight_ratio_pct"] > 0


def test_freight_predict_rejects_negative(client):
    res = client.post("/api/predict/freight-cost", json={"dollars": -5})
    assert res.status_code == 422


def test_freight_predict_rejects_zero(client):
    res = client.post("/api/predict/freight-cost", json={"dollars": 0})
    assert res.status_code == 422


def test_freight_predict_missing_field(client):
    res = client.post("/api/predict/freight-cost", json={})
    assert res.status_code == 422


# --- Invoice risk (single) --------------------------------------------------

SAFE_PAYLOAD = {
    "invoice_quantity": 120,
    "invoice_dollars": 5400.0,
    "freight": 210.0,
    "total_item_quantity": 118,
    "total_item_dollars": 5390.0,
}

FLAGGED_PAYLOAD = {
    "invoice_quantity": 48,
    "invoice_dollars": 352.95,
    "freight": 1.73,
    "total_item_quantity": 162,
    "total_item_dollars": 2476.99,
}


def test_invoice_risk_predict_safe_example(client):
    res = client.post("/api/predict/invoice-risk", json=SAFE_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert body["flagged"] is False
    assert body["risk_label"] == "Auto-Approved"
    assert 0.0 <= body["flag_probability"] <= 1.0
    assert 0.0 <= body["confidence_pct"] <= 100.0


def test_invoice_risk_predict_flagged_example(client):
    res = client.post("/api/predict/invoice-risk", json=FLAGGED_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert body["flagged"] is True
    assert body["risk_label"] == "Flagged for Manual Review"


def test_invoice_risk_predict_missing_fields(client):
    res = client.post("/api/predict/invoice-risk", json={"invoice_quantity": 10})
    assert res.status_code == 422


def test_invoice_risk_predict_rejects_negative_freight(client):
    bad = {**SAFE_PAYLOAD, "freight": -1}
    res = client.post("/api/predict/invoice-risk", json=bad)
    assert res.status_code == 422


# --- Invoice risk (batch) ----------------------------------------------------

def _csv_bytes(rows, header=None):
    header = header or ["invoice_quantity", "invoice_dollars", "Freight", "total_item_quantity", "total_item_dollars"]
    lines = [",".join(header)]
    lines += [",".join(str(v) for v in row) for row in rows]
    return ("\n".join(lines)).encode("utf-8")


def test_batch_predict_valid_csv(client):
    csv_bytes = _csv_bytes([
        [120, 5400.0, 210.0, 118, 5390.0],
        [48, 352.95, 1.73, 162, 2476.99],
    ])
    res = client.post(
        "/api/predict/invoice-risk/batch",
        files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["total_rows"] == 2
    assert body["flagged_count"] == 1
    assert body["auto_approved_count"] == 1
    assert len(body["results"]) == 2


def test_batch_predict_rejects_missing_columns(client):
    csv_bytes = _csv_bytes([[1, 2, 3]], header=["a", "b", "c"])
    res = client.post(
        "/api/predict/invoice-risk/batch",
        files={"file": ("bad.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert res.status_code == 400
    assert "missing required column" in res.json()["detail"].lower()


def test_batch_predict_rejects_non_csv_extension(client):
    res = client.post(
        "/api/predict/invoice-risk/batch",
        files={"file": ("sample.txt", io.BytesIO(b"not,a,csv"), "text/plain")},
    )
    assert res.status_code == 400


def test_batch_predict_rejects_empty_file(client):
    res = client.post(
        "/api/predict/invoice-risk/batch",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert res.status_code == 400


def test_batch_predict_accepts_lowercase_freight_column(client):
    # "freight" (lowercase) should be normalized to "Freight" automatically.
    csv_bytes = _csv_bytes(
        [[120, 5400.0, 210.0, 118, 5390.0]],
        header=["invoice_quantity", "invoice_dollars", "freight", "total_item_quantity", "total_item_dollars"],
    )
    res = client.post(
        "/api/predict/invoice-risk/batch",
        files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert res.status_code == 200
