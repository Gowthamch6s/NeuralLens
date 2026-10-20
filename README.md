# NeuralLens — NLP Lab 01: Tokenization

An intentionally small, testable start to an educational NLP + computer vision platform.
This version implements **one real, transparent baseline tokenizer**, a FastAPI JSON endpoint, and a browser playground. It does not yet use a trained NLP model or computer vision.

## Setup in VS Code

Requirements: Python 3.10 or newer.

### Windows PowerShell

```powershell
cd neuralens-starter
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn neuralens.api:app --reload
```

If the activation script is blocked, use the environment's Python directly: `.venv\Scripts\python.exe -m pip install -r requirements.txt` (remove the unusual separator if copied from a viewer), then `.venv\Scripts\python.exe -m pytest -q` and `.venv\Scripts\python.exe -m uvicorn neuralens.api:app --reload`.

### macOS / Linux

```bash
cd neuralens-starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn neuralens.api:app --reload
```

Visit **http://127.0.0.1:8000** for the visual lab and **http://127.0.0.1:8000/docs** for interactive API documentation.

## What to learn

- **Token**: a unit created when text is split into pieces.
- **Offset**: the start/end character positions of each piece in the original string.
- **Regex tokenizer**: a hand-written set of rules, not an NLP model.
- **Subword tokenizer**: a model-specific algorithm used by BERT and many other transformers; we'll add it in Lab 02.

## Test ideas

| Input | Expected tokens | Lesson |
| --- | --- | --- |
| `I love NLP!` | `I`, `love`, `NLP`, `!` | Baseline |
| `I can't stop.` | `I`, `can't`, `stop`, `.` | Apostrophes |
| `Café 😊` | `Café`, `😊` | Unicode |
| `Wow... really?!` | `Wow`, `.`, `.`, `.`, `really`, `?`, `!` | Punctuation |
| `   ` | none | Empty input |

Try to predict the output before clicking the button. Then find `TOKEN_PATTERN` in `neuralens/tokenizer.py` and change it. Run tests again to see whether your rules still behave as expected.

## HTTP example

```bash
curl -X POST http://127.0.0.1:8000/api/tokenize -H "Content-Type: application/json" -d '{"text":"I love NLP!"}'
```

## Upcoming labs

1. Word tokenizer vs. BERT subword tokenizer and token IDs
2. TF-IDF + cosine similarity
3. Sentiment baseline and transformer comparison
4. Image preprocessing, classification and OCR
5. Visual question answering with image + text

Keep experiment datasets and reported metrics reproducible. Never claim evaluation results before running the tests.

## Experiment 02: side-by-side BERT WordPiece comparison

This version retains Experiment 01's `/api/tokenize` endpoint and adds `/api/tokenize/bert`.
The browser root now opens the Experiment 02 interface.

**Windows PowerShell** (inside the extracted `neuralens-starter` directory):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn neuralens.api:app --reload
```

If `.venv` is not present in the new folder, run `py -m venv .venv` first.
Open `http://127.0.0.1:8000`.

The first BERT comparison downloads the tokenizer files from Hugging Face. An internet connection is necessary on first use, but you do not need to download BERT's full neural-network model weights for tokenization. Subsequent runs can use the cache.

The BERT test is deterministic and mocked for fast offline testing; it verifies the encoding-to-display logic but **does not claim a verified live download or specific vocabulary outputs**. Observe and record real tokens and IDs in the app.
