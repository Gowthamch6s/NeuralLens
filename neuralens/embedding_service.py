"""Contextual target-word comparison using a lazily loaded BERT encoder."""
from functools import lru_cache

MODEL_NAME = "bert-base-uncased"

@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return torch, tokenizer, model


def _target_positions(sentence, target, tokenizer, encoded):
    """Find token indices overlapping an exact, case-insensitive word span."""
    import re
    # Require whole-word match; do not accidentally select 'bank' inside 'banking'.
    matches = list(re.finditer(r"(?<!\w)" + re.escape(target) + r"(?!\w)", sentence, flags=re.I))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one occurrence of '{target}' in each sentence; found {len(matches)}")
    start, end = matches[0].span()
    offsets = encoded['offset_mapping'][0].tolist()
    indices = [i for i,(a,b) in enumerate(offsets) if b > start and a < end and b > a]
    if not indices:
        raise ValueError(f"No tokens overlap '{target}'")
    return indices


def compare_embeddings(sentence_a: str, sentence_b: str, target: str):
    if not target.strip() or any(c.isspace() for c in target):
        raise ValueError("Target must be a single nonempty word")
    torch, tokenizer, model = _load()
    def analyze(sentence):
        encoded = tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True, truncation=True, max_length=128)
        indices = _target_positions(sentence, target, tokenizer, encoded)
        ids = encoded['input_ids'][0].tolist()
        inputs = {k:v for k,v in encoded.items() if k != 'offset_mapping'}
        with torch.inference_mode():
            output = model(**inputs).last_hidden_state[0, indices].mean(dim=0)
            raw = model.get_input_embeddings()(inputs['input_ids'])[0, indices].mean(dim=0)
        return output, raw, [tokenizer.convert_ids_to_tokens([ids[i]])[0] for i in indices], [ids[i] for i in indices]
    a, raw_a, tokens_a, ids_a = analyze(sentence_a)
    b, raw_b, tokens_b, ids_b = analyze(sentence_b)
    import torch.nn.functional as F
    def sim(x,y): return round(F.cosine_similarity(x.unsqueeze(0),y.unsqueeze(0)).item(),4)
    return {
        'model': MODEL_NAME, 'target': target,
        'sentence_a': sentence_a, 'sentence_b': sentence_b,
        'tokens_a': tokens_a, 'tokens_b': tokens_b,
        'token_ids_a': ids_a, 'token_ids_b': ids_b,
        'embedding_dimensions': int(a.numel()),
        'static_similarity': sim(raw_a, raw_b),
        'contextual_similarity': sim(a,b),
        'vector_preview_a': [round(float(x),4) for x in a[:8]],
        'vector_preview_b': [round(float(x),4) for x in b[:8]],
    }
