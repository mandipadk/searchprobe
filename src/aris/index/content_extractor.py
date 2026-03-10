"""Content extraction from HTML using trafilatura."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from aris.index.document import IndexedDocument

logger = logging.getLogger(__name__)


def extract_content(html: str, url: str = "", source: str = "web") -> IndexedDocument | None:
    """Extract structured content from HTML."""
    try:
        import trafilatura
        content = trafilatura.extract(html) or ""
        metadata_json = trafilatura.extract(
            html, output_format="json", include_links=False
        )
    except ImportError:
        content = _basic_extract(html)
        metadata_json = None

    if not content:
        return None

    title = ""
    published_date = None

    if metadata_json:
        try:
            meta = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
            title = meta.get("title", "")
            date_str = meta.get("date")
            if date_str:
                try:
                    published_date = datetime.fromisoformat(date_str)
                except (ValueError, TypeError):
                    pass
        except (json.JSONDecodeError, AttributeError):
            pass

    if not title:
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()

    # Extract numeric values from content
    numeric_values = _extract_numbers(content)

    return IndexedDocument(
        url=url,
        title=title,
        content=content,
        snippet=content[:300],
        source=source,
        published_date=published_date,
        word_count=len(content.split()),
        numeric_values=numeric_values,
    )


def _basic_extract(html: str) -> str:
    """Basic HTML -> text fallback without trafilatura."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:10000]


def _extract_numbers(text: str) -> dict[str, float]:
    """Extract named numeric values from text (heuristic)."""
    values = {}
    patterns = [
        (r"(\d[\d,]*)\s+employees?", "employees"),
        (r"(\d[\d,]*)\s+(?:github\s+)?stars?", "stars"),
        (r"\$(\d[\d,.]*)\s*(?:million|M)", "revenue_millions"),
        (r"\$(\d[\d,.]*)\s*(?:billion|B)", "revenue_billions"),
        (r"(\d[\d,]*)\s+(?:users?|customers?)", "users"),
        (r"founded?\s+(?:in\s+)?(\d{4})", "founded_year"),
    ]

    for pattern, name in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1).replace(",", ""))
                values[name] = val
            except ValueError:
                pass

    return values
