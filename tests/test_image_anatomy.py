import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from neuralens.api import app
from neuralens.image_anatomy import inspect_image, ImageInputError

def make_image(color=(255, 0, 0), size=(4, 4)):
    out = io.BytesIO()
    Image.new('RGB', size, color).save(out, format='PNG')
    return out.getvalue()

def test_red_pixel_and_channel_means():
    x = inspect_image(make_image(), 64)
    assert x['center_pixel_rgb'] == [255, 0, 0]
    assert x['normalized_center_rgb'] == [1.0, 0.0, 0.0]
    assert x['mean_rgb'] == [255.0, 0.0, 0.0]
    assert x['preview_size'] == [4, 4]
    assert all(x['views'][v].startswith('data:image/png;base64,') for v in ['red', 'blue', 'green', 'grayscale', 'original'])

def test_histogram_totals():
    x = inspect_image(make_image(size=(6, 5)))
    assert all(sum(a) == 30 for a in x['histogram_16_bins'].values())

def test_resizes_without_upscaling():
    x = inspect_image(make_image(size=(400, 200)), 100)
    assert x['preview_size'] == [100, 50]

def test_limits():
    with pytest.raises(ImageInputError): inspect_image(b'')
    with pytest.raises(ImageInputError): inspect_image(b'x'* (5*1024*1024+1))
    with pytest.raises(ImageInputError): inspect_image(make_image(), 5)
    with pytest.raises(ImageInputError): inspect_image(b'not an image')

def test_api_valid():
    response = TestClient(app).post('/api/vision/anatomy', files={'file': ('red.png', make_image(), 'image/png')})
    assert response.status_code == 200
    assert response.json()['center_pixel_rgb'] == [255, 0, 0]

def test_api_invalid_media():
    response = TestClient(app).post('/api/vision/anatomy', files={'file': ('test.txt', b'text', 'text/plain')})
    assert response.status_code == 415

def test_api_bad_bytes():
    response = TestClient(app).post('/api/vision/anatomy', files={'file': ('bad.png', b'wrong', 'image/png')})
    assert response.status_code == 422
