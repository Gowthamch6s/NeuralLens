"""Run: uvicorn neuralens.api:app --reload"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .tokenizer import analyze

app = FastAPI(title="NeuralLens API", version="0.1.0")
BASE_DIR = Path(__file__).resolve().parent.parent


class TextInput(BaseModel):
    text: str = Field(max_length=10000)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "project": "NeuralLens"}


@app.post("/api/tokenize")
def tokenize_endpoint(payload: TextInput):
    return analyze(payload.text)


@app.post("/api/tokenize/bert")
def bert_tokenize_endpoint(payload: TextInput):
    from fastapi import HTTPException
    from .bert_tokenizer import analyze_bert
    try:
        return analyze_bert(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"BERT tokenizer unavailable: {type(exc).__name__}. Install transformers and check that Hugging Face is reachable on first run.") from exc


class EmbeddingInput(BaseModel):
    sentence_a: str = Field(min_length=1, max_length=500)
    sentence_b: str = Field(min_length=1, max_length=500)
    target: str = Field(min_length=1, max_length=60)


@app.post("/api/embeddings/compare")
def embeddings_compare_endpoint(payload: EmbeddingInput):
    from fastapi import HTTPException
    try:
        from .embedding_service import compare_embeddings
        return compare_embeddings(payload.sentence_a, payload.sentence_b, payload.target)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ImportError, OSError) as exc:
        raise HTTPException(status_code=503, detail="BERT model unavailable. Use Python 3.12 and install torch and transformers; first run requires internet.") from exc


class MapExample(BaseModel):
    sentence: str = Field(min_length=1, max_length=500)
    label: str = Field(default="", max_length=80)


class MapInput(BaseModel):
    target: str = Field(min_length=1, max_length=60)
    examples: list[MapExample] = Field(min_length=3, max_length=12)


@app.post("/api/embeddings/map")
def embedding_map_endpoint(payload: MapInput):
    import logging
    from fastapi import HTTPException
    from .embedding_map import build_embedding_map
    try:
        return build_embedding_map([e.model_dump() for e in payload.examples], payload.target)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception("Embedding map model failed")
        raise HTTPException(status_code=503, detail="Embedding model failed to load or run. Check server logs and Python 3.12 environment.") from exc


class SentimentInput(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.post("/api/sentiment/baseline")
def sentiment_baseline_endpoint(payload: SentimentInput):
    from .sentiment import predict_baseline
    return predict_baseline(payload.text)


@app.post("/api/sentiment/transformer")
def sentiment_transformer_endpoint(payload: SentimentInput):
    import logging
    from fastapi import HTTPException
    from .sentiment import predict_transformer
    try:
        return predict_transformer(payload.text)
    except (ImportError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception("Sentiment transformer failure")
        raise HTTPException(503, detail="Sentiment transformer unavailable. See server logs; check PyTorch, Transformers and first-run model download.") from exc


@app.get("/api/sentiment/evaluation")
def sentiment_evaluation_endpoint():
    from .sentiment import evaluate_baseline
    return evaluate_baseline()

@app.get("/api/sentiment/evaluation/compare")
def sentiment_evaluation_compare(include_transformer: bool = False):
    import logging
    from fastapi import HTTPException
    from .evaluation_lab import evaluate_models
    try:
        return evaluate_models(include_transformer=include_transformer)
    except (ImportError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception("Sentiment evaluation model failed")
        raise HTTPException(status_code=503, detail="Transformer unavailable; run baseline-only or check server logs.") from exc


class ExplainInput(BaseModel):
    text: str = Field(min_length=1, max_length=1000)

@app.post("/api/sentiment/explain/baseline")
def explain_baseline_endpoint(payload: ExplainInput):
    from .explainability import explain_baseline
    return explain_baseline(payload.text)

@app.post("/api/sentiment/explain/transformer")
def explain_transformer_endpoint(payload: ExplainInput):
    import logging
    from fastapi import HTTPException
    from .explainability import explain_transformer
    try:
        return explain_transformer(payload.text)
    except (ImportError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception("RoBERTa explainability failed")
        raise HTTPException(503,detail="RoBERTa unavailable; see server log.") from exc


@app.post("/api/vision/anatomy")
async def vision_anatomy_endpoint(file: __import__("fastapi").UploadFile = __import__("fastapi").File(...), width: int = 96):
    from fastapi import HTTPException
    from .image_anatomy import MAX_BYTES, ImageInputError, inspect_image
    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(415, detail="Upload a PNG, JPEG, or WebP image.")
    data = await file.read(MAX_BYTES + 1)
    try:
        return inspect_image(data, width=width)
    except ImageInputError as exc:
        raise HTTPException(422, detail=str(exc)) from exc

@app.post('/api/vision/classify')
async def vision_classify_endpoint(file: __import__('fastapi').UploadFile = __import__('fastapi').File(...), blur_radius: float = 0):
    import logging
    from fastapi import HTTPException
    from .image_classifier import classify, MAX_BYTES, ClassificationInputError
    if file.content_type not in ('image/png', 'image/jpeg', 'image/webp'):
        raise HTTPException(415, detail='Upload PNG, JPEG, or WebP.')
    data = await file.read(MAX_BYTES + 1)
    try:
        return classify(data, blur_radius=blur_radius)
    except ClassificationInputError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception('Vision model initialization or inference failed')
        raise HTTPException(503, detail='Vision model unavailable. Install matching torch and torchvision; the first model run requires internet. Check server logs.') from exc


# Experiments 12–14: object detection, OCR, and visual question answering.
from fastapi import UploadFile, File, HTTPException
import logging

async def _vision_bytes(file: UploadFile):
    from .advanced_vision import MAX_BYTES
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only PNG, JPEG, and WebP uploads are supported")
    return await file.read(MAX_BYTES + 1)

@app.post("/api/vision/detect")
async def vision_detect(file: UploadFile = File(...), threshold: float = 0.5):
    from .advanced_vision import detect, VisionInputError
    try:
        return detect(await _vision_bytes(file), threshold)
    except VisionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception("Object detection unavailable")
        raise HTTPException(status_code=503, detail="Detection model unavailable. See server logs and first-run weight download.") from exc

@app.post("/api/vision/ocr")
async def vision_ocr(file: UploadFile = File(...)):
    from .advanced_vision import read_text, VisionInputError
    try:
        return read_text(await _vision_bytes(file))
    except VisionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception("OCR unavailable")
        raise HTTPException(status_code=503, detail="OCR engine unavailable. Install Tesseract executable and pytesseract; see README.") from exc

@app.post("/api/vision/vqa")
async def vision_vqa(file: UploadFile = File(...), question: str = ""):
    from .advanced_vision import answer_image, VisionInputError
    try:
        return answer_image(await _vision_bytes(file), question)
    except VisionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).exception("Visual question answering unavailable")
        raise HTTPException(status_code=503, detail="Visual QA model unavailable. Check transformers/PyTorch installation, first-run model download and server logs.") from exc
