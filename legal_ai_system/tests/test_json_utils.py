import pytest

from legal_ai_system.utils.json_utils import extract_json_from_llm_response


def test_extract_plain_json():
    text = '{"a": 1, "b": 2}'
    assert extract_json_from_llm_response(text) == {"a": 1, "b": 2}


def test_extract_json_code_block():
    text = """```json
{"key": "value"}
```"""
    assert extract_json_from_llm_response(text) == {"key": "value"}


def test_extract_json_invalid_returns_empty_dict():
    text = 'not json'
    assert extract_json_from_llm_response(text) == {}
