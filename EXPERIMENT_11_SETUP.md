# Experiment 11 — Image Classification

## Safe upgrade into your existing Windows project

Your active directory is `D:\.vscode\nlp+comp vision\neuralens-starter`.

1. **Extract this ZIP to Downloads**, not into the active project.
2. Stop Uvicorn with Ctrl+C. Back up the active `neuralens/api.py`, `static/index.html`, and `requirements.txt` outside the extracted folder.
3. Copy the NEW files `neuralens/image_classifier.py` and `tests/test_image_classifier.py` into matching active directories.
4. The packaged `neuralens/api.py` and `static/index.html` are complete snapshots of Labs 01–11. If you have not customized the active versions, you can copy the package versions after backing up. If customized, merge only the Experiment 11 endpoint (at bottom of `api.py`) and the `vision-classification-lab` section and following script (near bottom of `index.html`). Do not overwrite unknown edits.
5. Add `torchvision>=0.20,<1` to your active `requirements.txt`. Because you already have CPU PyTorch, install torchvision from the SAME PyTorch CPU index to ensure matching builds:

```powershell
.\.venv312\Scripts\python.exe -m pip install torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pytest -q tests
.\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --host 127.0.0.1 --port 8001
```

6. Open http://127.0.0.1:8001 and navigate to Experiment 11. Upload a cat/dog photograph; compare blur radius 0 and 4.

**First run:** ResNet-18 pretrained weights (~45 MB) are downloaded by torchvision. An internet connection is needed. Tests use a mocked model and do not validate live inference; the live run must be verified locally.

**Scope:** ImageNet-1K model labels are imperfect and softmax scores are uncalibrated. This is image *classification*, not object detection.
