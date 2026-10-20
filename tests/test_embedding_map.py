import pytest
import numpy as np
from fastapi.testclient import TestClient
from neuralens.api import app
from neuralens.embedding_map import pca_map, DEFAULT_EXAMPLES


def test_pca_shape_and_variance():
    coords, explained = pca_map([[1,0,0],[0,1,0],[0,0,1],[1,1,0]])
    assert len(coords) == 4
    assert all(len(p) == 2 for p in coords)
    assert 0 <= explained <= 1
    assert np.allclose(np.mean(coords, axis=0), [0,0], atol=1e-10)


def test_pca_identical_vectors():
    coords, explained = pca_map([[2,2],[2,2],[2,2]])
    assert explained == 0
    assert np.allclose(coords, 0)


def test_pca_requires_3_vectors():
    with pytest.raises(ValueError):
        pca_map([[0,0],[1,1]])


def test_map_invalid_length_returns_422():
    client = TestClient(app)
    response=client.post('/api/embeddings/map',json={'target':'bank','examples':DEFAULT_EXAMPLES[:2]})
    assert response.status_code == 422


def test_map_endpoint_response_with_mock(monkeypatch):
    import neuralens.embedding_map as m
    monkeypatch.setattr(m, 'build_embedding_map', lambda examples,target: {'target':target,'points':examples,'explained_variance_ratio':0.5})
    response=TestClient(app).post('/api/embeddings/map',json={'target':'bank','examples':DEFAULT_EXAMPLES})
    assert response.status_code == 200
    assert response.json()['target']=='bank'
    assert len(response.json()['points'])==6
