import pytest
from fastapi.testclient import TestClient
from neuralens.api import app
from neuralens.evaluation_lab import summarize


def test_metrics_full_correct():
    result = summarize(["negative", "neutral", "positive"], ["negative", "neutral", "positive"])
    assert result["accuracy"] == result["macro_f1"] == 1.0
    assert result["matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_metrics_misclassification():
    result = summarize(["negative", "neutral", "positive"], ["positive", "neutral", "positive"])
    assert result["accuracy"] == .6667
    assert result["matrix"][0][2] == 1
    assert result["per_class"]["positive"]["precision"] == .5


def test_invalid_labels_and_lengths():
    with pytest.raises(ValueError):
        summarize(["positive"], [])
    with pytest.raises(ValueError):
        summarize(["bogus"], ["positive"])


def test_api_baseline_evaluation():
    result = TestClient(app).get("/api/sentiment/evaluation/compare")
    assert result.status_code == 200
    data = result.json()
    assert data["models"]["baseline"]["count"] == 9
    assert "transformer" not in data["models"]
    assert len(data["rows"]) == 9


def test_api_transformer_stub(monkeypatch):
    import neuralens.evaluation_lab as lab
    monkeypatch.setattr(lab, "predict_transformer", lambda text: {"label": "neutral"})
    data = TestClient(app).get("/api/sentiment/evaluation/compare?include_transformer=true").json()
    assert "transformer" in data["models"]
    assert data["models"]["transformer"]["count"] == 9
