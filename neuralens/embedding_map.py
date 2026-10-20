"""Project contextual token representations into 2D using per-dataset PCA."""
import re

DEFAULT_EXAMPLES = [
    {"sentence": "The bank approved my loan.", "label": "Finance · loan"},
    {"sentence": "She opened a savings account at the bank.", "label": "Finance · savings"},
    {"sentence": "I deposited money at the bank.", "label": "Finance · deposit"},
    {"sentence": "The fisherman rested on the river bank.", "label": "Nature · fishing"},
    {"sentence": "The water flooded the bank of the river.", "label": "Nature · flood"},
    {"sentence": "Wildflowers grew along the bank of the stream.", "label": "Nature · flowers"},
]


def pca_map(vectors):
    """Return 2D PCA coordinates and explained-variance proportion, numpy only.

    Coordinates are for visual exploration, not substitutes for high-D cosine similarities.
    """
    import numpy as np
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("Need at least 3 vectors with 2 dimensions")
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variance = s ** 2
    ratio = float((variance[:2].sum() / variance.sum()) if variance.sum() > 0 else 0)
    return coords.tolist(), round(ratio, 4)


def build_embedding_map(examples, target):
    """Extract each target contextual state; compute PCA and full-space cosines."""
    import numpy as np
    from .embedding_service import _load, _target_positions

    if not target.strip() or len(target) > 60 or any(c.isspace() for c in target):
        raise ValueError("Target must be one nonempty word (maximum 60 characters)")
    if not 3 <= len(examples) <= 12:
        raise ValueError("Provide between 3 and 12 sentences")
    torch, tokenizer, model = _load()
    vectors = []
    clean = []
    for index, item in enumerate(examples):
        sentence = item['sentence'].strip()
        label = item['label'].strip() or f"Sample {index + 1}"
        if not sentence or len(sentence) > 500 or len(label) > 80:
            raise ValueError("Sentences must have 1–500 characters and labels at most 80")
        encoded = tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True, truncation=True, max_length=128)
        positions = _target_positions(sentence, target, tokenizer, encoded)
        inputs = {key: value for key, value in encoded.items() if key != 'offset_mapping'}
        with torch.inference_mode():
            vector = model(**inputs).last_hidden_state[0, positions].mean(dim=0)
        vectors.append(vector.detach().cpu().numpy())
        clean.append({'label': label, 'sentence': sentence})
    mat = np.stack(vectors).astype(float)
    coord, explained = pca_map(mat)
    norms = np.linalg.norm(mat, axis=1)
    unit = mat / np.maximum(norms[:, None], 1e-12)
    similarity = unit @ unit.T
    return {
        'model': 'bert-base-uncased', 'target': target,
        'dimensions': int(mat.shape[1]), 'explained_variance_ratio': explained,
        'points': [{**item, 'x': round(float(x), 5), 'y': round(float(y), 5)} for item, (x, y) in zip(clean, coord)],
        'cosine_similarity_matrix': [[round(float(v), 4) for v in row] for row in similarity],
    }
