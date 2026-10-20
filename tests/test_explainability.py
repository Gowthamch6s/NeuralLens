from fastapi.testclient import TestClient
from neuralens.api import app
from neuralens.explainability import explain_baseline

client=TestClient(app)

def test_baseline_model_contributions_reconstruct_logits():
    d=explain_baseline("I really enjoyed this purchase")
    assert d["prediction"] in ("positive","negative","neutral")
    for label in d["classes"]:
        reconstructed=d["intercepts"][label]+sum(c["by_class"][label] for c in d["contributions"])
        assert abs(reconstructed-d["logits"][label])<0.0001

def test_empty_input_rejected():
    import pytest
    with pytest.raises(ValueError):explain_baseline("   ")

def test_api_baseline():
    r=client.post('/api/sentiment/explain/baseline',json={"text":"I really enjoyed this purchase"})
    assert r.status_code==200
    assert "contributions" in r.json()

def test_api_invalid():
    assert client.post('/api/sentiment/explain/baseline',json={"text":""}).status_code==422

def test_ui_has_lab():
    from pathlib import Path
    page=(Path(__file__).parent.parent/'static/index.html').read_text(encoding='utf-8')
    assert 'EXPERIMENT 09' in page and 'explain-run-transformer' in page
