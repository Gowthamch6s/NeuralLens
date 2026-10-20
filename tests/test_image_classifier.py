from io import BytesIO
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from neuralens.api import app
from neuralens.image_classifier import classify, decode_image, ClassificationInputError


def sample():
    b = BytesIO()
    Image.new('RGB', (40, 30), (100, 160, 220)).save(b, 'PNG')
    return b.getvalue()


def test_decoding():
    assert decode_image(sample()).size == (40, 30)


def test_reject_empty():
    with pytest.raises(ClassificationInputError):
        decode_image(b'')


def test_reject_garbage():
    with pytest.raises(ClassificationInputError):
        decode_image(b'not-an-image')


def test_blur_bounds():
    with pytest.raises(ClassificationInputError):
        classify(sample(), blur_radius=9)


def test_content_type_validation():
    r = TestClient(app).post('/api/vision/classify', files={'file': ('image.txt', b'hello', 'text/plain')})
    assert r.status_code == 415


def test_mocked_inference(monkeypatch):
    import torch
    import neuralens.image_classifier as m
    class FakeWeights:
        meta = {'categories': [f'label-{i}' for i in range(1000)]}
        def __str__(self): return 'TEST_WEIGHTS'
        def transforms(self): return lambda image: torch.zeros(3, 224, 224)
    class FakeModel:
        def __call__(self, tensor): return torch.arange(1000, dtype=torch.float).unsqueeze(0)
    monkeypatch.setattr(m, 'get_model', lambda: (FakeModel(), FakeWeights()))
    result = classify(sample(), blur_radius=1)
    assert len(result['predictions']) == 5
    assert result['predictions'][0]['label'] == 'label-999'
    assert result['blur_radius'] == 1
