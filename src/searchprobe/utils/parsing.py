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
        # Try to find a raw JSON object
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text.strip()

    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result
        return {"data": result}
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to extract JSON from LLM response: {e}") from e
