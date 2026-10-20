"""ResNet18 ImageNet-1K inference; deterministic preprocessing and blur challenge."""
from functools import lru_cache
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 12_000_000

class ClassificationInputError(ValueError):
    pass

def decode_image(data: bytes):
    if not data or len(data) > MAX_BYTES:
        raise ClassificationInputError('Image must be between 1 byte and 5 MB.')
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ClassificationInputError('Upload PNG, JPEG, or WebP.')
            if source.width * source.height > MAX_PIXELS:
                raise ClassificationInputError('Image exceeds 12 megapixels.')
            return ImageOps.exif_transpose(source).convert('RGB')
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ClassificationInputError('Unable to read the image.') from exc

@lru_cache(maxsize=1)
def get_model():
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model, weights

def classify(data: bytes, blur_radius: float = 0.0):
    if not (0 <= blur_radius <= 8):
        raise ClassificationInputError('Blur radius must be between 0 and 8.')
    img = decode_image(data)
    if blur_radius:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    import torch
    model, weights = get_model()
    transform = weights.transforms()
    tensor = transform(img).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
        scores, indices = probabilities.topk(5)
    categories = weights.meta['categories']
    return {
        'model': 'torchvision/resnet18',
        'weights': str(weights),
        'dataset': 'ImageNet-1K',
        'input_size': list(img.size),
        'blur_radius': blur_radius,
        'preprocessing': 'RGB -> resize shorter side to 256 -> center crop 224x224 -> convert to float tensor -> ImageNet normalization',
        'predictions': [{'label': categories[int(i)], 'score': round(float(s), 6), 'class_id': int(i)} for s, i in zip(scores, indices)],
        'caution': 'Scores are softmax values for ImageNet-1K classes, not guaranteed confidence or proof the depicted object exists. Out-of-distribution images may produce misleading high scores.'
    }
