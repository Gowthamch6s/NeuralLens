from fastapi.testclient import TestClient
from neuralens.api import app
from neuralens.sentiment import TRAIN, HOLDOUT, predict_baseline, evaluate_baseline

client = TestClient(app)

def test_training_data_has_all_classes():
    assert {x[1] for x in TRAIN} == {"negative", "neutral", "positive"}

def test_baseline_probabilities():
    d = predict_baseline("I enjoyed this movie")
    assert d["label"] in {"negative", "neutral", "positive"}
    assert abs(sum(d["scores"].values()) - 1) < 0.001

def test_empty_input_rejected():
    import pytest
    with pytest.raises(ValueError):
        predict_baseline("  ")

def test_baseline_api():
    r = client.post("/api/sentiment/baseline", json={"text": "It was wonderful"})
    assert r.status_code == 200
    assert "scores" in r.json()

def test_blank_api():
    assert client.post("/api/sentiment/baseline", json={"text": ""}).status_code == 422

def test_evaluation_api():
    r = client.get("/api/sentiment/evaluation")
    assert r.status_code == 200
    assert r.json()["sample_count"] == len(HOLDOUT)
    assert len(r.json()["confusion_matrix"]) == 3

def test_evaluation_is_explicitly_tiny():
    assert "not a representative" in evaluate_baseline()["warning"]

def test_sentiment_ui():
    html = client.get("/").text
    assert 'id="sentiment-lab"' in html
    assert "/api/sentiment/transformer" in html
