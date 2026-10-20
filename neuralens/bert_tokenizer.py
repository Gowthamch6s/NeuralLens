"""Real BERT WordPiece tokenizer. Requires one-time Hugging Face download."""
from functools import lru_cache

MODEL_ID = 'google-bert/bert-base-uncased'

@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('This experiment requires a fast tokenizer for offsets.')
    return tokenizer


def analyze_bert(text: str) -> dict:
    tokenizer = get_tokenizer()
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
    ids = encoding['input_ids']
    pieces = tokenizer.convert_ids_to_tokens(ids)
    special_ids = set(tokenizer.all_special_ids)
    tokens = [
        {'text': piece, 'id': token_id, 'start': start, 'end': end,
         'special': token_id in special_ids}
        for piece, token_id, (start, end) in zip(pieces, ids, encoding['offset_mapping'])
    ]
    return {
        'model': MODEL_ID,
        'input': text,
        'token_count': len(tokens),
        'tokens': tokens,
        'attention_mask': encoding['attention_mask'],
        'note': 'Special tokens are added by BERT and have no character span. ## means word continuation. Lowercasing and subword splitting depend on this exact model vocabulary.',
    }
