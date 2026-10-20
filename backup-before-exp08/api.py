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
