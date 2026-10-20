from neuralens.tokenizer import analyze, tokenize


def test_simple_sentence():
    assert [t.text for t in tokenize("I love NLP!")] == ["I", "love", "NLP", "!"]


def test_punctuation():
    assert [t.text for t in tokenize("Hello, world?")] == ["Hello", ",", "world", "?"]


def test_apostrophes():
    assert [t.text for t in tokenize("I can't stop.")] == ["I", "can't", "stop", "."]


def test_empty_text():
    assert tokenize("   ") == []


def test_offsets_match_original_text():
    text = "NLP is fun!"
    assert all(text[t.start:t.end] == t.text for t in tokenize(text))


def test_unicode():
    assert [t.text for t in tokenize("Café 😊")] == ["Café", "😊"]


def test_response_count():
    result = analyze("Hi!")
    assert result["token_count"] == len(result["tokens"]) == 2
