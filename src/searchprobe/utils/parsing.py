"""Shared parsing utilities for extracting structured data from LLM responses."""

import json
import re
from typing import Any


def extract_json_from_llm_response(response_text: str) -> dict[str, Any]:
    """Extract JSON from an LLM response that may contain markdown code blocks.

    Handles common patterns:
    - ```json ... ``` blocks
    - ``` ... ``` blocks
    - Raw JSON objects

    Args:
        response_text: Raw text response from an LLM

    Returns:
        Parsed JSON as a dictionary

    Raises:
        ValueError: If no valid JSON could be extracted
    """
    # Try markdown code block first (with or without json tag)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try greedy regex first
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            json_str = json_match.group(0)
            # If greedy match fails to parse, use balanced-brace extraction
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
                return {"data": result}
            except json.JSONDecodeError:
                json_str = _extract_balanced_json(response_text)
        else:
            json_str = response_text.strip()

    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result
        return {"data": result}
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to extract JSON from LLM response: {e}") from e


def _extract_balanced_json(text: str) -> str:
    """Extract JSON object using balanced brace counting.

    Finds the first '{' and matches it with its balanced closing '}'.
    Handles braces inside strings correctly.
    """
    start = text.find("{")
    if start == -1:
        return text.strip()

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Fallback: return from first brace to end
    return text[start:]
