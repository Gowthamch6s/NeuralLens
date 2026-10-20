# Experiment 06 — Challenge Mode

This ZIP is a complete project snapshot. **Do not nest it into your existing project.**

## Safe integration (Windows / PowerShell)
1. Stop the currently running Uvicorn server with Ctrl+C.
2. Extract the ZIP to a **separate temporary folder**.
3. From the extracted `neuralens-starter` folder, copy `static/index.html` into your **existing** project's `static/index.html`. This replaces the Experiment 05 page with an updated page that still includes Experiments 01–05 plus Challenge Mode.
4. Copy `tests/test_challenge_ui.py` to the existing project's `tests/` directory. No dependency or backend changes are needed.
5. Keep your existing `neuralens/`, `.venv312/`, and other tests. **If you edited index.html yourself, merge the new Challenge Mode section instead of overwriting your edits.**
6. In the existing project folder run:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --host 127.0.0.1 --port 8001
```

7. Refresh `http://127.0.0.1:8001` with Ctrl+Shift+R. Scroll to Lab 06 and answer six questions.

**Important:** Challenge Mode uses fixed, previously verified educational observations, rather than calling BERT on every quiz question. Scores are local to the current page and reset after refreshing. This is intentionally separate from live model labs.
