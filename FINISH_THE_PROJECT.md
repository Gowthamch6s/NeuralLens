# NeuralLens — Labs 01–14: Installation and launch checklist

This ZIP is a **complete standalone project snapshot** built from the Experiment 11 archive plus Labs 12–14. It is not a patch. Do **not** extract its nested `neuralens-starter` into your active project directory.

## Safe installation (Windows PowerShell)

1. Extract the ZIP in Downloads. Keep your existing working project untouched.
2. In your *existing* project, back up `neuralens/api.py`, `static/index.html`, and `requirements.txt`, or better, commit your current work in Git.
3. **Recommended:** run this snapshot as a separate folder first. Open the extracted `neuralens-starter` directly in VS Code (File → Open Folder).
4. Create a **new Python 3.12** virtual environment in that extracted directory:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv312\Scripts\python.exe -m pytest -q tests
.\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --host 127.0.0.1 --port 8002
```

Use http://127.0.0.1:8002 so the old project can continue to run on 8001. Once the new snapshot is verified in your browser, decide whether to switch your Git repository to it or selectively merge changes.

## New labs

- **12 — Object Detection:** `POST /api/vision/detect?threshold=0.5`, multipart `file`; Faster R-CNN MobileNetV3 with COCO categories, boxes and scores. Download weights on first run (internet required).
- **13 — OCR:** `POST /api/vision/ocr`, multipart `file`; Tesseract OCR with word boxes. Install **Tesseract OCR executable** separately, add it to PATH, then install Python `pytesseract`. On Windows you may need to configure the Tesseract executable path if not on PATH.
- **14 — Visual Question Answering:** `POST /api/vision/vqa?question=What%20color%20is%20the%20cup%3F`, multipart `file`; BLIP VQA. First call downloads weights and can consume significant CPU/RAM. Requires Hugging Face Transformers and compatible PyTorch. It can produce incorrect answers.

## Verification before announcing a feature as live

- `python -m pytest -q tests`: the source snapshot passed **60 automated tests** during packaging.
- Verify actual ResNet classification (Lab 11) with two photos and two blur levels.
- Verify Lab 12 with a photo containing more than one COCO object. Record bounding-box quality and missed objects.
- Verify Lab 13 with a legible menu/receipt and compare extracted words to ground truth.
- Verify Lab 14 using an image with a clear, independently checkable answer. Test at least one ambiguous question.
- Retest Labs 01–10 after changing model versions or dependencies. Current tests use mocked inference for some heavy model routes: passing tests **does not** prove models downloaded or produced correct results on your machine.

## Professional polish before a public launch

- Split the monolithic `static/index.html` into reusable frontend components; add dedicated `/nlp` and `/vision` pages.
- Run a small accessibility audit: keyboard navigation, focus indicators, semantic structure, and color contrast.
- Add independent labeled evaluation sets, version-locked model dependencies, a privacy policy for uploaded images, and upload-rate limits.
- Avoid hosting CPU-heavy models on a tiny free web server. Consider deploying a static frontend separately and GPU/CPU inference endpoints on capable infrastructure.
- Never publish private user images, API secrets or model credentials. Uploaded images in the current routes are processed in memory rather than written to disk by these endpoints.

## Honest portfolio metrics

- 14 educational labs in the snapshot.
- 60 tests passed in the packaging environment; the new model-serving tests are mocked, not end-to-end live-inference benchmarks.
- NLP baseline: 8/9 on a tiny handwritten holdout (88.9% accuracy, macro F1 0.886); RoBERTa: 9/9 in the user's verified browser evaluation (100%, macro F1 1.000). These are **educational smoke tests**, not generalization evidence.
- BERT contextual representation: loan-bank vs river-bank cosine similarity 0.4533 in a verified local run; context changes embeddings.
