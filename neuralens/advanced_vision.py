"""Advanced vision modules. Models load lazily; never execute generated code."""
from io import BytesIO
from functools import lru_cache
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 12_000_000

class VisionInputError(ValueError):
    pass

def load_image(data):
    if not data or len(data) > MAX_BYTES:
        raise VisionInputError("Upload a nonempty image up to 5 MB")
    try:
        with Image.open(BytesIO(data)) as src:
            if src.format not in ("PNG", "JPEG", "WEBP"):
                raise VisionInputError("Only PNG, JPEG and WebP are supported")
            if src.width * src.height > MAX_PIXELS:
                raise VisionInputError("Image exceeds 12 megapixels")
            return ImageOps.exif_transpose(src).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise VisionInputError("Image could not be decoded") from exc

@lru_cache(maxsize=1)
def detection_model():
    import torch
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    )
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).eval()
    return model, weights.meta["categories"]

def detect(data: bytes, threshold: float = 0.5):
    if not 0.1 <= threshold <= 0.99:
        raise VisionInputError("Threshold must be between 0.10 and 0.99")
    img = load_image(data)
    import torch
    from torchvision.transforms.functional import to_tensor
    model, categories = detection_model()
    with torch.inference_mode():
        result = model([to_tensor(img)])[0]
    objects = []
    for box, cls, score in zip(result['boxes'], result['labels'], result['scores']):
        confidence = float(score)
        if confidence < threshold:
            continue
        label_id = int(cls)
        objects.append({"label": categories[label_id] if label_id < len(categories) else str(label_id),
                        "score": round(confidence, 4),
                        "box": [round(float(x), 1) for x in box]})
        if len(objects) >= 30:
            break
    return {"width": img.width, "height": img.height, "objects": objects,
            "threshold": threshold, "model": "Faster R-CNN MobileNetV3 320 FPN (COCO)",
            "caution": "Detection scores are not calibrated probabilities; missed and incorrect detections are possible."}

def read_text(data: bytes):
    import pytesseract
    from pytesseract import Output
    img = load_image(data)
    details = pytesseract.image_to_data(img, output_type=Output.DICT)
    items = []
    for i, raw in enumerate(details["text"]):
        word = raw.strip()
        try: conf = float(details["conf"][i])
        except (ValueError, TypeError): continue
        if not word or conf < 0:
            continue
        items.append({"text": word, "confidence": round(conf / 100, 4),
                      "box": [int(details[k][i]) for k in ("left", "top", "width", "height")]})
    return {"text": " ".join(x["text"] for x in items), "words": items,
            "width": img.width, "height": img.height,
            "engine": "Tesseract OCR", "caution": "OCR may misread handwriting, unusual fonts, and low-quality images. Confidence is not a guarantee."}

@lru_cache(maxsize=1)
def vqa_model():
    from transformers import AutoProcessor, BlipForQuestionAnswering
    model_id = "Salesforce/blip-vqa-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id).eval()
    return processor, model

def answer_image(data: bytes, question: str):
    if not 3 <= len(question.strip()) <= 250:
        raise VisionInputError("Question must contain 3 to 250 characters")
    img = load_image(data)
    processor, model = vqa_model()
    import torch
    inputs = processor(img, question.strip(), return_tensors="pt")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=20)
    answer = processor.decode(ids[0], skip_special_tokens=True).strip()
    return {"question": question, "answer": answer,
            "model": "Salesforce/blip-vqa-base", "width": img.width, "height": img.height,
            "caution": "This is a model-generated answer, not verified visual evidence. Check objects and relationships yourself."}
