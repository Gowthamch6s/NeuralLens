# Experiment 05 — Contextual Embedding Map (PCA)

## Important: keep your running project

This ZIP is a **complete reference snapshot** of NeuralLens Labs 01–05, not a command to replace your working folder. Before changing local files, stop your server and make a dated copy of your project.

From `D:\.vscode\nlp+comp vision\neuralens-starter`, copy only these files from the unzipped archive into the identically named paths:

- `neuralens/embedding_map.py` (new)
- `neuralens/api.py` (updated; if you've customized yours, merge the `MapExample`, `MapInput` and `/api/embeddings/map` additions rather than overwriting)
- `static/index.html` (updated; preserve any custom frontend edits)
- `tests/test_embedding_map.py` (new)

You do **not** need to move, recreate, or delete `.venv312`, `embedding_experiment.py`, `embedding_service.py`, or the pre-downloaded BERT checkpoint.

## Commands (PowerShell, in project root)

```powershell
.\.venv312\Scripts\python.exe -m pip install numpy
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --host 127.0.0.1 --port 8001
```

Visit http://127.0.0.1:8001, scroll to Experiment 05 and click **Generate interactive map**. The model is loaded lazily and reused per process, so the first API request can take longer.

## The science

- BERT outputs a 768-D contextual vector per target word occurrence. Multi-piece words are averaged over their overlapping token representations.
- PCA fits a new 2D projection to *these exact samples*. Axes may flip or rotate between runs and must not be interpreted as human semantic dimensions.
- **Explained variance** is the share of variation preserved by the 2D projection, not model accuracy.
- The pair comparison always uses 768-D cosine similarity, not the 2D distance. Two points close together in the projection may have different high-dimensional similarities.
- Preset six examples use one `bank` in each sentence. After testing presets, edit them to create your own examples. Exactly one occurrence of the selected target must appear per sentence.
- This is a visualization learning exercise, not a validated word-sense-disambiguation benchmark.

## Verification

Unit tests mock the backend where appropriate. They cannot prove your local BERT inference works: confirm the browser returns six plotted points and a high-D pair similarity on your machine. The pre-existing Experiment 04 still provides the independent reference: `The bank approved my loan.` vs `The fisherman rested on the river bank.` should be around 0.4533.
