# Experiment 09 — Safe installation (Windows PowerShell)

The package includes a complete snapshot of Labs 01–09. **Do not extract into the active project**. Extract to Downloads first.

1. Stop Uvicorn (`Ctrl+C`). Back up the currently working project or at minimum `neuralens/api.py` and `static/index.html`.
2. From the extracted `neuralens-starter`, copy `neuralens/explainability.py` into the active `neuralens/`, and `tests/test_explainability.py` into active `tests/`.
3. For `neuralens/api.py`, merge the **Experiment 09** block beginning at `class ExplainInput(BaseModel):` from the end of the provided file into the end of the active API file. Do not duplicate it if it is already there.
4. For `static/index.html`, merge the Experiment 09 `<section id="explainability-lab">` and following `<script>` block immediately before `</main>` in your active HTML. Avoid replacing custom edits.
5. Existing dependency requirements cover the new experiment. Use the Python 3.12 environment already installed.
6. Run ` .\.venv312\Scripts\python.exe -m pytest -q tests` in the **active outer** project; then start with ` .\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --host 127.0.0.1 --port 8001`.
7. Open http://127.0.0.1:8001, hard-refresh, scroll to Experiment 09. Start with baseline (offline); RoBERTa can be slower because it runs once per word.

These explanations are diagnostic probes, not proof of a model's reasoning. The prior nine-item holdout should not be claimed as production performance.
