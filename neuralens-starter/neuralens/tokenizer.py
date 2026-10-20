"""Transparent baseline tokenizers for learning, not transformer tokenizers."""
import re
from dataclasses import asdict, dataclass

TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    kind: str


def tokenize(text: str) -> list[Token]:
    """Split words and punctuation; preserve character positions.

    This tokenizer is a teaching baseline. It doesn't produce BERT token IDs,
    and it is not designed to cover all languages or emoji sequences.
    """
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):
        piece = match.group()
        kind = "word" if piece[0].isalnum() or piece[0] == "_" else "punctuation"
        tokens.append(Token(piece, match.start(), match.end(), kind))
    return tokens


def analyze(text: str) -> dict:
    tokens = tokenize(text)
    return {
        "input": text,
        "token_count": len(tokens),
        "tokens": [asdict(t) for t in tokens],
        "explanation": (
            "This is a regex-based word/punctuation tokenizer. Real transformer "
            "models typically use learned subword tokenization and token IDs."
        ),
    }
