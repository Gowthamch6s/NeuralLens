from fastapi.testclient import TestClient
from neuralens.api import app

client = TestClient(app)

def test_input_validation():
    r = client.post('/api/embeddings/compare', json={'sentence_a':'', 'sentence_b':'The bank.', 'target':'bank'})
    assert r.status_code == 422

def test_embedding_route_with_mock(monkeypatch):
    from neuralens import embedding_service
    monkeypatch.setattr(embedding_service, 'compare_embeddings', lambda a,b,t: {'target':t,'contextual_similarity':0.5,'static_similarity':1.0})
    r=client.post('/api/embeddings/compare',json={'sentence_a':'The bank approved my loan.','sentence_b':'river bank','target':'bank'})
    assert r.status_code == 200 and r.json()['static_similarity'] == 1.0
