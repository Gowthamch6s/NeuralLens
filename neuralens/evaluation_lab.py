"""Transparent small-holdout model evaluation for NeuralLens.

This is a teaching smoke test, not a generalization benchmark.
"""
from .sentiment import HOLDOUT, LABELS, predict_baseline, predict_transformer


def summarize(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError("Gold/prediction lengths differ")
    if not gold:
        raise ValueError("At least one sample is required")
    if any(x not in LABELS for x in [*gold, *predicted]):
        raise ValueError("Unknown sentiment label")
    matrix = [[sum(g == truth and p == prediction for g, p in zip(gold, predicted))
               for prediction in LABELS] for truth in LABELS]
    metrics = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(LABELS)) if r != i)
        fn = sum(matrix[i][c] for c in range(len(LABELS)) if c != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": round(precision, 4),
                          "recall": round(recall, 4), "f1": round(f1, 4),
                          "support": sum(matrix[i])}
    n = len(gold)
    return {"accuracy": round(sum(g == p for g, p in zip(gold, predicted)) / n, 4),
            "macro_f1": round(sum(x["f1"] for x in metrics.values()) / len(LABELS), 4),
            "per_class": metrics, "matrix": matrix, "labels": list(LABELS), "count": n}


def evaluate_models(include_transformer=False):
    rows = [{"text": text, "expected": gold,
             "baseline": predict_baseline(text)["label"]} for text, gold in HOLDOUT]
    if include_transformer:
        for row in rows:
            row["transformer"] = predict_transformer(row["text"])["label"]
    gold = [row["expected"] for row in rows]
    models = {"baseline": summarize(gold, [r["baseline"] for r in rows])}
    if include_transformer:
        models["transformer"] = summarize(gold, [r["transformer"] for r in rows])
    return {"models": models, "rows": rows,
            "warning": "Nine manually written holdout items. Labels have not been independently annotated; examples may resemble training templates. Do not interpret these results as real-world performance.",
            "transformer_included": include_transformer}
