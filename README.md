# NeuralLens
### An Interactive NLP, Computer Vision & Multimodal AI Learning Platform

**14 learning labs · 60 passing automated tests · Python · FastAPI · PyTorch · Hugging Face**

NeuralLens is an interactive educational platform designed to make machine learning concepts more transparent.

Instead of treating AI models as black boxes, NeuralLens allows users to experiment with inputs, examine predictions, compare models, visualize representations, and investigate failure cases.

## Application Screenshots

A selection of experiments running locally in NeuralLens.

### Contextual Embeddings
Exploring how BERT represents the same word differently depending on its context.

![BERT Embedding Explorer](docs/screenshots/bert-embeddings.png)

### Sentiment Analysis
Comparing a TF-IDF baseline against pretrained RoBERTa predictions.

![Sentiment Analysis](docs/screenshots/sentiment-analysis.png)

### Model Explainability
Inspecting TF-IDF feature contributions and transformer word-removal sensitivity.

![Model Explainability](docs/screenshots/model-explainability.png)

### Image Anatomy
Visualizing RGB channels, grayscale conversion, and pixel measurements.

![Image Anatomy](docs/screenshots/image-anatomy.png)

### Object Detection
Identifying and locating objects using bounding boxes.

![Object Detection](docs/screenshots/object-detection.png)

### Multimodal Visual Question Answering
Asking natural-language questions about uploaded images.

![Visual Question Answering](docs/screenshots/visual-question-answering.png)

**Repository:** https://github.com/Gowthamch6s/NeuralLens

## Why I built this

I wanted to understand how AI models work beyond simply calling inference APIs.

NeuralLens started as a rule-based NLP tokenization exercise and expanded into a collection of 14 interactive experiments spanning language processing, image understanding, and multimodal AI.

The focus is on experimentation, measurable behavior, and explaining the limitations of model outputs.

## Features

### Natural Language Processing

| Lab | Experiment | Key concepts |
|---|---|---|
| 01 | Rule-Based Tokenization | Regex, tokens, character offsets |
| 02 | BERT WordPiece Tokenization | Subwords, vocabulary IDs, special tokens |
| 03 | Contextual Embeddings | BERT embeddings, cosine similarity |
| 04 | Interactive Embedding Comparison | Context-dependent word representations |
| 05 | Embedding Space Explorer | PCA, dimensionality reduction |
| 06 | NLP Challenge Mode | Interactive learning and knowledge checks |
| 07 | Sentiment Detective | TF-IDF, logistic regression, RoBERTa |
| 08 | Model Evaluation Lab | Accuracy, precision, recall, F1, confusion matrices |
| 09 | Explainability Lab | Linear feature contributions, word-removal perturbations |

### Computer Vision & Multimodal AI

| Lab | Experiment | Key concepts |
|---|---|---|
| 10 | Image Anatomy Lab | RGB channels, grayscale, pixels, resizing |
| 11 | Image Classification | ResNet-18, ImageNet classes, preprocessing |
| 12 | Object Detection | Faster R-CNN, bounding boxes, thresholds |
| 13 | Optical Character Recognition | Tesseract, text extraction |
| 14 | Visual Question Answering | BLIP, image-and-text inference |

## Technology Stack

**Backend:** Python, FastAPI, Pydantic

**Machine Learning:** PyTorch, Torchvision, Hugging Face Transformers, scikit-learn, NumPy

**Computer Vision:** Pillow, pretrained ResNet-18, Faster R-CNN, Tesseract OCR

**Multimodal AI:** BLIP visual question answering

**Frontend:** Browser-based interactive interface using HTML, CSS, and JavaScript

**Testing:** Pytest and FastAPI endpoint tests

## System Architecture

```text
                  User / Web Browser
                          |
                Interactive Lab UI
                          |
                     FastAPI
                          |
         +----------------+----------------+
         |                |                |
      NLP Labs        Vision Labs     Multimodal Lab
         |                |                |
  BERT / RoBERTa     ResNet / R-CNN        BLIP
  TF-IDF / PCA       Pillow / OCR     Image + Question
         |                |                |
         +----------------+----------------+
                          |
                Predictions / Analysis
                          |
              Interactive Visualizations
```

Models are loaded locally or downloaded from their upstream model sources when first required. Individual experiments may have additional dependencies.

## Getting Started

### Requirements

- Windows with Python 3.12
- Internet access for initial model downloads
- Sufficient memory and disk space for pretrained models
- Tesseract installed separately to use OCR

### Installation

Clone the repository:

```powershell
git clone https://github.com/Gowthamch6s/NeuralLens.git
cd NeuralLens
```

Create the virtual environment:

```powershell
py -3.12 -m venv .venv312
```

Install dependencies:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

Install CPU-compatible PyTorch and Torchvision if needed:

```powershell
.\.venv312\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Run automated tests:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q tests
```

Start the server:

```powershell
.\.venv312\Scripts\python.exe -m uvicorn neuralens.api:app --host 127.0.0.1 --port 8002
```

Open:

**http://127.0.0.1:8002**

API documentation:

**http://127.0.0.1:8002/docs**

## Testing & Validation

The latest local development check reported:

```text
60 passed in 2.78s
```

This verifies the automated test suite in the developer's environment. Some model integration tests use mocked inference results, so passing tests should not be interpreted as independent validation of every pretrained model.

Selected experiments have also been tested manually with live inference.

### Example: Contextual Embeddings

Using BERT, experiments showed that the same word can have different contextual representations.

Observed cosine similarities included:

- Financial context versus another financial context: **0.9332**
- Financial context versus river-related context: **0.5412**

These are demonstration results for specific input sentences, not general benchmarks.

### Example: Sentiment Classification

On a small nine-example educational holdout:

- TF-IDF + Logistic Regression: **8/9 correct**
- Pretrained RoBERTa: **9/9 correct**

The dataset is too small to establish reliable real-world accuracy. The experiment is intended to demonstrate evaluation methods and failure analysis.

## Model Limitations

NeuralLens is an educational project and development MVP, not a production decision-making system.

Important considerations:

- Classification scores are not necessarily calibrated probabilities.
- Pretrained models may fail on unfamiliar images, ambiguous inputs, or distribution shifts.
- OCR performance depends on image quality and text characteristics.
- Object detection is limited to the model's supported categories.
- Visual question answering models may produce plausible but incorrect answers.
- Explanations based on removing words are perturbation measurements, not definitive causal explanations.
- Small teaching datasets should not be treated as production benchmarks.

## Project Status

**Current:** 14 interactive experiments implemented and the automated test suite passing locally.

**In progress:** Broader live inference validation, improved frontend navigation, accessibility review, larger evaluation datasets, and public deployment.

## Roadmap

- [x] NLP experiments and contextual embeddings
- [x] Sentiment comparison and explainability
- [x] Image preprocessing and classification labs
- [x] Object detection and OCR modules
- [x] Visual question answering module
- [x] Automated regression test suite
- [ ] Independent model evaluation benchmarks
- [ ] Improved application dashboard and accessibility testing
- [ ] CI testing using GitHub Actions
- [ ] Public hosted demonstration

## Author

**Gowtham**

AI Engineering | Machine Learning | NLP | Computer Vision

GitHub: https://github.com/Gowthamch6s

---

*NeuralLens is a learning-in-public project focused on understanding AI systems through experiments rather than treating predictions as unquestionable answers.*