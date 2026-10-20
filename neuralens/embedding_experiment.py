
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

sentences = [
     "The bank approved my loan.",
    "The fisherman rested on the river bank.",
    "She opened a savings account at the bank."
]


def get_bank_embedding(sentence):
    encoded = tokenizer(sentence, return_tensors="pt")

    tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][0].tolist()
    )

    positions = [
        i for i, token in enumerate(tokens)
        if token == "bank"
    ]

    if len(positions) != 1:
        raise ValueError(
            f"Expected one 'bank' token, got: {tokens}"
        )

    position = positions[0]

    with torch.no_grad():
        output = model(**encoded)

    embedding = output.last_hidden_state[0, position]

    return tokens, embedding


results = []

for sentence in sentences:
    tokens, embedding = get_bank_embedding(sentence)
    results.append(embedding)

    print("\nSentence:", sentence)
    print("Tokens:", tokens)
    print("Embedding dimensions:", embedding.shape)
    print("First 5 values:", embedding[:5].tolist())


print("\nCOSINE SIMILARITIES")

pairs = [
    (0, 1),
    (0, 2),
    (1, 2)
]

for a, b in pairs:
    similarity = F.cosine_similarity(
        results[a].unsqueeze(0),
        results[b].unsqueeze(0)
    ).item()

    print(
        f"Sentence {a + 1} vs {b + 1}: "
        f"{similarity:.4f}"
    )
