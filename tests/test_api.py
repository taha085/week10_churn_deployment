import json
from pathlib import Path
import sys
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import main

client = TestClient(main.app)
PAYLOAD = json.loads((ROOT / "docs" / "sample_request.json").read_text())

def test_health():
    assert client.get("/health").status_code == 200

def test_prediction_shape_and_threshold():
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["prediction"] in {"Yes", "No"}
    assert body["threshold"] == main.THRESHOLD

def test_successful_prediction_is_logged():
    client.post("/predict", json=PAYLOAD)
    log_path = ROOT / "backend" / "logs" / "predictions.log"
    assert log_path.exists()
    last = json.loads(log_path.read_text().splitlines()[-1])
    assert "timestamp_utc" in last
    assert "input_summary" in last
    assert "churn_probability" in last
    assert "predicted_label" in last

def test_missing_field_is_rejected():
    bad = PAYLOAD.copy()
    bad.pop("tenure")
    assert client.post("/predict", json=bad).status_code == 422

def test_invalid_type_is_rejected():
    bad = PAYLOAD.copy()
    bad["MonthlyCharges"] = "abc"
    assert client.post("/predict", json=bad).status_code == 422

def test_out_of_range_is_rejected():
    bad = PAYLOAD.copy()
    bad["tenure"] = 999
    assert client.post("/predict", json=bad).status_code == 422
