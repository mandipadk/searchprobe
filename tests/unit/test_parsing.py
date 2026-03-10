"""Tests for JSON parsing utilities."""

import pytest

from searchprobe.utils.parsing import _extract_balanced_json, extract_json_from_llm_response


class TestExtractJsonFromLlmResponse:
    def test_raw_json(self):
        result = extract_json_from_llm_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_llm_response(text)
        assert result == {"key": "value"}

    def test_markdown_code_block_no_json_tag(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json_from_llm_response(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"score": 0.8}\nThat looks good.'
        result = extract_json_from_llm_response(text)
        assert result == {"score": 0.8}

    def test_nested_json(self):
        text = '{"scores": {"relevance": 0.9, "accuracy": 0.8}}'
        result = extract_json_from_llm_response(text)
        assert result["scores"]["relevance"] == 0.9

    def test_json_with_trailing_text_after_brace(self):
        """The key case: greedy regex matches too much, balanced extraction fixes it."""
        text = '{"key": "value"} some trailing text with another {brace}'
        result = extract_json_from_llm_response(text)
        assert result == {"key": "value"}

    def test_non_dict_json_wrapped(self):
        text = "[1, 2, 3]"
        result = extract_json_from_llm_response(text)
        assert result == {"data": [1, 2, 3]}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to extract JSON"):
            extract_json_from_llm_response("not json at all")

    def test_complex_nested_with_trailing_text(self):
        text = 'Analysis: {"scores": {"a": 1}, "modes": ["x"]} The end. {}'
        result = extract_json_from_llm_response(text)
        assert result["scores"]["a"] == 1
        assert result["modes"] == ["x"]

    def test_json_with_escaped_quotes(self):
        text = '{"text": "he said \\"hello\\""}'
        result = extract_json_from_llm_response(text)
        assert result["text"] == 'he said "hello"'


class TestExtractBalancedJson:
    def test_simple_object(self):
        result = _extract_balanced_json('{"a": 1}')
        assert result == '{"a": 1}'

    def test_nested_braces(self):
        text = '{"a": {"b": 1}} trailing'
        result = _extract_balanced_json(text)
        assert result == '{"a": {"b": 1}}'

    def test_braces_in_strings(self):
        text = '{"text": "a { b } c"} after'
        result = _extract_balanced_json(text)
        assert result == '{"text": "a { b } c"}'

    def test_no_brace(self):
        result = _extract_balanced_json("no json here")
        assert result == "no json here"

    def test_prefix_text(self):
        text = 'prefix {"key": 1} suffix'
        result = _extract_balanced_json(text)
        assert result == '{"key": 1}'

    def test_escaped_quote_in_string(self):
        text = '{"k": "v\\"x"} rest'
        result = _extract_balanced_json(text)
        assert result == '{"k": "v\\"x"}'
