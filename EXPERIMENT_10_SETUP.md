# NeuralLens — Experiment 10: Image Anatomy Lab

## What's new
Deterministic computer-vision preprocessing: RGB isolation, grayscale, resizing, normalization, center-pixel inspection, per-channel mean and 16-bin histograms. A generated color-block sample is included in the browser. **This experiment does not run an image classification model.**

## Windows installation (preserves Experiments 01–09)

1. Stop Uvicorn with `Ctrl+C`. Extract this ZIP **into Downloads, outside your active project**.
2. Back up `neuralens/api.py` and `static/index.html` from your active project.
3. Copy `neuralens/image_anatomy.py` and `tests/test_image_anatomy.py` into the matching directories of your existing project.
4. Your active `neuralens/api.py` and `static/index.html` may contain edits. If they match the Experiment 09 snapshot, copy these two updated files. Otherwise merge only the `POST /api/vision/anatomy` route and the `vision-anatomy-lab` HTML section and its adjacent script. Do not replace the complete project directory and do not copy another `neuralens-starter` folder inside the active one.
5. Add `pillow>=10,<13` and `python-multipart>=0.0.9,<1` to your existing `requirements.txt` (numpy is already used in Experiment 05).
6. In PowerShell at `D:\.vscode\nlp+comp vision\neuralens-starter`, run:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pytest -q tests
.\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001, hard-refresh (`Ctrl+Shift+R`), scroll to Experiment 10 and click **Generate synthetic sample**, then **Inspect pixels**. Try a real PNG or JPEG next. Do not paste the `(.venv) PS ...>` prompt into PowerShell.

## API

`POST /api/vision/anatomy?width=96` multipart form field `file` (PNG/JPEG/WebP, max 5MB, <=12 million pixels). JSON response contains `original_size`, `preview_size`, `center_pixel_rgb`, `normalized_center_rgb`, `mean_rgb`, `histogram_16_bins`, and data-URI previews under `views`.

## Limits

Images are processed locally by FastAPI. EXIF orientation is respected. Transparency is flattened during RGB conversion (not composited against white). Oversized or invalid images receive a validation error. The histogram bars are relative within each channel; compare numerical mean RGB values for cross-channel intensity. No image is saved server-side by this feature.
