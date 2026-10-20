# Experiment 07 — Sentiment Detective

## Upgrade your EXISTING project (no nested folders)

1. Stop Uvicorn (`Ctrl+C`). Back up `neuralens/api.py` and `static/index.html` if you have custom changes.
2. From this ZIP, copy `neuralens/sentiment.py` to your active `neuralens` directory; copy `tests/test_sentiment.py` to `tests`.
3. `neuralens/api.py` and `static/index.html` are full versions based on the Experiment 06 snapshot. Copy them ONLY if you have not modified those files yourself; otherwise merge the new sentiment endpoints and new `<section id="sentiment-lab">` plus its script.
4. Add `scikit-learn>=1.5,<2` to your existing `requirements.txt` (or use the packaged version if unchanged).
5. From `D:\.vscode\nlp+comp vision\neuralens-starter` run:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001 and scroll to Experiment 07. The baseline runs offline; the pretrained transformer downloads a separate sentiment checkpoint on first use (potentially hundreds of MB). No API key is required for this public checkpoint. The app returns a clear unavailable message if it cannot load the transformer.

## Experiments

- Try obvious praise, criticism, neutral facts, sarcasm and mixed sentiment.
- Predict the label before revealing both model outputs.
- Open the evaluation summary to see performance on 9 manually written held-out examples. These are educational smoke tests, not a meaningful generalization benchmark.
- Investigate disagreements and plan a larger independent annotated dataset before asserting real-world accuracy.

## Technical note

This transformer uses `cardiffnlp/twitter-roberta-base-sentiment` with class order negative / neutral / positive. It is a separate sentiment classification model, not the BERT model from previous labs. The first download needs network access. You can keep your Python 3.12 environment and existing BERT cache.
