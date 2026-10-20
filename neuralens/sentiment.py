"""Sentiment learning lab: tiny TF-IDF baseline and optional real RoBERTa.

The bundled examples are teaching data, NOT an unbiased benchmark.
"""
from functools import lru_cache

LABELS = ("negative", "neutral", "positive")
# Small, transparent supervised dataset; broaden considerably before publishing metrics.
TRAIN = [
    ("I loved this movie", "positive"), ("What a wonderful experience", "positive"),
    ("The service was excellent", "positive"), ("This is a fantastic product", "positive"),
    ("Everything worked perfectly", "positive"), ("I am very happy with my purchase", "positive"),
    ("The meal tasted delicious", "positive"), ("Absolutely amazing performance", "positive"),
    ("Friendly staff and quick delivery", "positive"), ("The design looks beautiful", "positive"),
    ("I would recommend it", "positive"), ("This exceeded my expectations", "positive"),
    ("I disliked this movie", "negative"), ("What a terrible experience", "negative"),
    ("The service was awful", "negative"), ("This is a disappointing product", "negative"),
    ("Nothing worked correctly", "negative"), ("I am unhappy with this purchase", "negative"),
    ("The meal tasted horrible", "negative"), ("Absolutely poor performance", "negative"),
    ("Rude staff and very slow delivery", "negative"), ("The design looks ugly", "negative"),
    ("I would not recommend it", "negative"), ("This failed my expectations", "negative"),
    ("The package arrived yesterday", "neutral"), ("The meeting begins at noon", "neutral"),
    ("The item is made of plastic", "neutral"), ("The train stops at station five", "neutral"),
    ("The book has two hundred pages", "neutral"), ("She bought a blue notebook", "neutral"),
    ("The temperature is twenty degrees", "neutral"), ("It is located near the office", "neutral"),
    ("The store opens at nine", "neutral"), ("The report was published Monday", "neutral"),
    ("The parcel weighs two kilograms", "neutral"), ("The recipe includes rice and beans", "neutral"),
]
# Separate examples, not used for training. Tiny illustrative check, NOT production evidence.
HOLDOUT = [
    ("The staff were wonderful", "positive"),
    ("I really enjoyed this purchase", "positive"),
    ("The food was delicious and fresh", "positive"),
    ("The staff were rude", "negative"),
    ("I regretted buying this item", "negative"),
    ("The food was awful and cold", "negative"),
    ("The appointment is on Tuesday", "neutral"),
    ("The building has four entrances", "neutral"),
    ("The file contains twelve pages", "neutral"),
]

@lru_cache(maxsize=1)
def _baseline():
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1500, random_state=42),
    )
    pipeline.fit([x for x, _ in TRAIN], [y for _, y in TRAIN])
    return pipeline

def predict_baseline(text):
    if not text.strip():
        raise ValueError("Enter a non-empty sentence.")
    model = _baseline()
    p = model.predict_proba([text])[0]
    scores = {label: round(float(prob), 4) for label, prob in zip(model.classes_, p)}
    return {"label": max(scores, key=scores.get), "scores": scores,
            "method": "TF-IDF (1–2 grams) + logistic regression",
            "note": "Trained on only 36 hand-written teaching examples. Scores are model estimates, not calibrated certainty."}

@lru_cache(maxsize=1)
def _transformer():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    return tokenizer, model

def predict_transformer(text):
    import torch
    if not text.strip():
        raise ValueError("Enter a non-empty sentence.")
    tokenizer, model = _transformer()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        raw = model(**encoded).logits[0]
        probs = torch.softmax(raw, dim=-1).tolist()
    # This exact Cardiff checkpoint uses negative, neutral, positive label order.
    scores = {label: round(float(prob), 4) for label, prob in zip(LABELS, probs)}
    return {"label": max(scores, key=scores.get), "scores": scores,
            "method": "CardiffNLP Twitter RoBERTa sentiment",
            "note": "Pretrained sentiment model. Domain shift, sarcasm and mixed opinions can cause misleading outputs."}

def evaluate_baseline():
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    truths = [y for _, y in HOLDOUT]
    preds = _baseline().predict([x for x, _ in HOLDOUT]).tolist()
    return {"sample_count": len(HOLDOUT), "training_count": len(TRAIN),
            "accuracy": round(float(accuracy_score(truths, preds)), 4),
            "macro_f1": round(float(classification_report(truths, preds, labels=LABELS, output_dict=True, zero_division=0)["macro avg"]["f1-score"]), 4),
            "confusion_matrix": confusion_matrix(truths, preds, labels=LABELS).tolist(),
            "labels": list(LABELS),
            "examples": [{"text": text, "expected": gold, "predicted": pred} for (text, gold), pred in zip(HOLDOUT, preds)],
            "warning": "Only nine hand-written holdout examples. This is an educational smoke test, not a representative benchmark."}
