# NeuralLens Experiment 04 — Setup (Windows PowerShell)

This is a complete upgrade containing Labs 01, 02 and 04. Your existing `neuralens/embedding_experiment.py` from Lab 03 is local and is NOT part of this archive. Back it up before replacing folders.

## Recommended safe installation
1. Stop the server with Ctrl+C.
2. Make a copy of your existing `neuralens-starter` folder as a backup.
3. Extract this ZIP to a different directory; it contains one `neuralens-starter` folder.
4. Copy `neuralens/embedding_service.py` and `tests/test_embeddings_api.py` into the corresponding folders in your EXISTING project.
5. In your existing `neuralens/api.py`, append the EmbeddingInput class and `/api/embeddings/compare` handler from this ZIP's `neuralens/api.py` (do not duplicate existing endpoints).
6. For the interface, copy this ZIP's `static/index.html` over your existing static/index.html after backing it up.
7. Continue using the existing `.venv312` environment. Install requirements and PyTorch if not already installed:
   `.\.venv312\Scripts\python.exe -m pip install -r requirements.txt`
   `.\.venv312\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
8. Test: `.\.venv312\Scripts\python.exe -m pytest -q`
9. Run: `.\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --reload`
10. Visit http://127.0.0.1:8000 and scroll to Experiment 04.

The first model request downloads the BERT weights (~440MB) if not already cached. BERT is loaded lazily once per Python process; this demo uses CPU and a limit of 128 tokens per sentence. The first lab remains on the same page. For production, use explicit model cache management and request queueing.

## Learning notes
- Initial word embeddings are looked up from the model's vocabulary embedding layer and don't include position/token type embeddings.
- Final hidden-state embeddings are contextualized by surrounding words.
- For subword targets, this lab averages the corresponding token vectors.
- Comparing raw BERT token vectors by cosine is a teaching demonstration, not a validated semantic similarity metric.
- Input must contain the target word exactly once in each sentence. The script rejects multiple/missing occurrences.
