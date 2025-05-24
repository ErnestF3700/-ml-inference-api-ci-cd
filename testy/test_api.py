#apiTest

from fastapi.testclient import TestClient
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from main import app

client = TestClient(app)

def test_predict():
    payload = {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert "prediction" in resp.json()
