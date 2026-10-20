from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
from neuralens.api import app
from neuralens.advanced_vision import load_image, VisionInputError
import pytest

client = TestClient(app)

def sample():
    b = BytesIO(); Image.new("RGB", (64, 48), (80, 120, 200)).save(b, format="PNG");return b.getvalue()

def test_loader_dimensions():
    assert load_image(sample()).size == (64, 48)

def test_reject_bad_image():
    with pytest.raises(VisionInputError): load_image(b"bad")

def test_reject_empty_image():
    with pytest.raises(VisionInputError): load_image(b"")

def test_reject_oversized():
    with pytest.raises(VisionInputError): load_image(b"a" * (5 * 1024 * 1024 + 1))

def test_detect_route(monkeypatch):
    from neuralens import advanced_vision
    monkeypatch.setattr(advanced_vision, "detect", lambda data, threshold: {"objects": [],"threshold":threshold})
    r=client.post("/api/vision/detect?threshold=0.7", files={"file":("a.png",sample(),"image/png")})
    assert r.status_code == 200 and r.json()["threshold"] == 0.7

def test_ocr_route(monkeypatch):
    from neuralens import advanced_vision
    monkeypatch.setattr(advanced_vision, "read_text", lambda data: {"text":"HELLO"})
    r=client.post("/api/vision/ocr", files={"file":("a.png",sample(),"image/png")})
    assert r.json()["text"] == "HELLO"

def test_vqa_route(monkeypatch):
    from neuralens import advanced_vision
    monkeypatch.setattr(advanced_vision,"answer_image",lambda data,question:{"answer":"red","question":question})
    r=client.post("/api/vision/vqa?question=What%20color%20is%20it%3F",files={"file":("a.png",sample(),"image/png")})
    assert r.status_code == 200 and r.json()["answer"] == "red"

def test_bad_mime():
    r=client.post("/api/vision/ocr", files={"file":("a.txt",b"hello","text/plain")})
    assert r.status_code == 415

def test_bad_question():
    r=client.post("/api/vision/vqa?question=hi", files={"file":("a.png",sample(),"image/png")})
    assert r.status_code == 422
