# NeuralLens Experiment 08 — Model Evaluation & Failure Analysis

## What's included
- Existing Labs 01–07 preserved
- New `neuralens/evaluation_lab.py` for per-class metrics and confusion matrices
- New `GET /api/sentiment/evaluation/compare?include_transformer=false|true`
- Browser lab with two-model evaluation, confusion matrix, per-class precision/recall/F1, and errors/disagreements
- Five new automated tests (33 total in this packaged snapshot)

## Upgrade your existing Windows project (do not nest folders)

1. Stop the currently running Uvicorn server (`Ctrl+C`).
2. Extract ZIP into a **separate temporary directory**, not inside your active project folder.
3. Make a backup copy of your current `neuralens/api.py` and `static/index.html` if modified.
4. Copy `neuralens/evaluation_lab.py` and `tests/test_evaluation_lab.py` into corresponding active project folders.
5. Merge **only** the new `/api/sentiment/evaluation/compare` route from the end of bundled `neuralens/api.py` into your active `api.py`. Do not paste it twice.
6. Merge the `<section id="evaluation-lab">...` and the script immediately after it from bundled `static/index.html` into your existing page, immediately before `</main>`. Do not copy the entire extracted folder over your active project if you customized your webpage.
7. From `D:\.vscode\nlp+comp vision\neuralens-starter`, run:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --host 127.0.0.1 --port 8001
```

8. Open `http://127.0.0.1:8001`, refresh with `Ctrl+Shift+R`, scroll to Experiment 08, and click **Evaluate baseline**. Then try **Compare both models**.

## Scientific limitations
- Both models are evaluated against the same nine manually written holdout items. This makes their output comparable *on that tiny set*, but does not constitute an independent benchmark.
- Transformer evaluation performs nine inferences; the first run may download weights. A failure of this optional model does not prevent baseline evaluation.
- Accuracy, F1, precision and recall are displayed alongside individual errors. High confidence and high accuracy on this small sample do not prove real-world reliability.
- Error/disagreement table includes any sample misclassified by either model or predicted differently between models.
