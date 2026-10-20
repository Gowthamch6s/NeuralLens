from fastapi.testclient import TestClient
from neuralens.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_api():
    response = client.post("/api/tokenize", json={"text": "Hello, AI!"})
    assert response.status_code == 200
    assert [t["text"] for t in response.json()["tokens"]] == ["Hello", ",", "AI", "!"]


def test_api_validates_input():
    assert client.post("/api/tokenize", json={"text": 123}).status_code == 422
