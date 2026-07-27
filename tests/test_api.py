import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api import app  # noqa: E402

client = TestClient(app)
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_lists_three_classes():
    assert len(client.get("/health").json()["classes"]) == 3


def test_predict_returns_expected_schema():
    with open(ASSETS / "sample_authentic.jpg", "rb") as f:
        r = client.post("/predict", files={"file": ("s.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert {"prediction", "confidence", "scores"} <= body.keys()
    assert 0.0 <= body["confidence"] <= 1.0


def test_probabilities_sum_to_one():
    with open(ASSETS / "sample_authentic.jpg", "rb") as f:
        r = client.post("/predict", files={"file": ("s.jpg", f, "image/jpeg")})
    assert sum(r.json()["scores"].values()) == pytest.approx(1.0, abs=1e-3)


def test_rejects_non_image():
    r = client.post(
        "/predict",
        files={"file": ("evil.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert r.status_code == 400


def test_known_sample_classifies_correctly():
    with open(ASSETS / "sample_authentic.jpg", "rb") as f:
        r = client.post("/predict", files={"file": ("s.jpg", f, "image/jpeg")})
    assert r.json()["prediction"] == "Authentic"