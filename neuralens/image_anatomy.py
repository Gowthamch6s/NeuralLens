"""Deterministic image preprocessing lab; no learned CV model involved."""
from io import BytesIO
import base64
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 12_000_000

class ImageInputError(ValueError):
    pass

def inspect_image(data: bytes, width: int = 96):
    if not data or len(data) > MAX_BYTES:
        raise ImageInputError('Image must be nonempty and no larger than 5 MB.')
    if not 16 <= width <= 256:
        raise ImageInputError('Preview width must be between 16 and 256 pixels.')
    try:
        with Image.open(BytesIO(data)) as opened:
            if opened.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ImageInputError('Upload a PNG, JPEG, or WebP image.')
            if opened.width * opened.height > MAX_PIXELS:
                raise ImageInputError('Image exceeds the 12-megapixel limit.')
            rgb = ImageOps.exif_transpose(opened).convert('RGB')
            source_dims = list(rgb.size)
            pixels = np.asarray(rgb, dtype=np.uint8)
            h, w = pixels.shape[:2]
            center = [int(x) for x in pixels[h//2, w//2]]
            histogram = {}
            for i, name in enumerate(('red', 'green', 'blue')):
                vals, _ = np.histogram(pixels[:, :, i], bins=16, range=(0, 256))
                histogram[name] = vals.astype(int).tolist()
            scaled = rgb.copy()
            scaled.thumbnail((width, width), Image.Resampling.LANCZOS)
            px = np.asarray(scaled, dtype=np.uint8)
            images = {'original': scaled, 'grayscale': ImageOps.grayscale(scaled).convert('RGB')}
            for i, name in enumerate(('red', 'green', 'blue')):
                channel = np.zeros_like(px)
                channel[:, :, i] = px[:, :, i]
                images[name] = Image.fromarray(channel, 'RGB')
            views = {}
            for name, im in images.items():
                buf = BytesIO()
                im.save(buf, 'PNG')
                views[name] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
            return {'original_size': source_dims, 'preview_size': list(scaled.size),
                    'center_pixel_rgb': center,
                    'normalized_center_rgb': [round(n/255, 4) for n in center],
                    'mean_rgb': [round(float(v), 2) for v in pixels.mean(axis=(0, 1))],
                    'histogram_16_bins': histogram, 'views': views,
                    'note': 'RGB channels are separate color planes; grayscale uses weighted luminance; normalization scales channel values into [0,1].'}
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageInputError('Could not decode the image safely. Try PNG, JPEG, or WebP.') from exc
