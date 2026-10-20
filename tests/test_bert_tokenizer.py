from unittest.mock import Mock, patch
from neuralens.bert_tokenizer import analyze_bert

def test_bert_encoding_includes_special_tokens_and_offsets():
    fake = Mock()
    fake.all_special_ids = [101, 102]
    fake.convert_ids_to_tokens.return_value = ['[CLS]', 'hello', '!', '[SEP]']
    fake.return_value = {
        'input_ids': [101, 7592, 999, 102],
        'offset_mapping': [(0, 0), (0, 5), (5, 6), (0, 0)],
        'attention_mask': [1, 1, 1, 1],
    }
    with patch('neuralens.bert_tokenizer.get_tokenizer', return_value=fake):
        result = analyze_bert('Hello!')
    assert result['token_count'] == 4
    assert result['tokens'][0]['special'] is True
    assert result['tokens'][1]['start'] == 0
    assert result['tokens'][1]['end'] == 5
    assert result['tokens'][2]['id'] == 999
    assert result['tokens'][3]['special'] is True
