"""Main-content extraction from HTML, with publication date and language.

Uses trafilatura/htmldate when installed and falls back to a small built-in
extractor so the evidence layer works without optional dependencies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser

from searchprobe.evidence.language import detect_language, normalize_lang_code


@dataclass
class Extracted:
    title: str
    text: str
    language: str | None
    language_confidence: float
    published_date: date | None
    extractor: str


_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"}
_BLOCK_TAGS = {
    "p",
    "div",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
    "tr",
    "section",
    "article",
    "blockquote",
    "pre",
    "td",
    "th",
    "dd",
    "dt",
}


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.skip_depth = 0
        self.in_title = False
        self.html_lang: str | None = None
        self.meta: dict[str, str] = {}
        self.time_values: list[str] = []
        self.jsonld: list[str] = []
        self._in_jsonld = False

    def handle_starttag(self, tag, attrs):  # type: ignore[override]
        a = dict(attrs)
        if tag == "html" and a.get("lang"):
            self.html_lang = a["lang"]
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and a.get("content"):
                self.meta[key] = a["content"]
        if tag == "time" and a.get("datetime"):
            self.time_values.append(a["datetime"])
        if tag == "script" and (a.get("type") or "").lower() == "application/ld+json":
            self._in_jsonld = True
        if tag == "title":
            self.in_title = True
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):  # type: ignore[override]
        if tag == "title":
            self.in_title = False
        if tag == "script":
            self._in_jsonld = False
        if tag in _SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):  # type: ignore[override]
        if self._in_jsonld:
            self.jsonld.append(data)
            return
        if self.in_title:
            self.title_parts.append(data)
            return
        if not self.skip_depth:
            self.parts.append(data)


_DATE_META_KEYS = (
    "article:published_time",
    "datepublished",
    "pubdate",
    "publish-date",
    "date",
    "dc.date",
    "dc.date.issued",
    "og:published_time",
    "sailthru.date",
)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value.strip()
    for candidate in (value, value[:10]):
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).date()
        except ValueError:
            continue
    return None


def _jsonld_date(blobs: list[str]) -> date | None:
    for blob in blobs:
        try:
            data = json.loads(blob)
        except ValueError:
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key in ("datePublished", "dateCreated", "uploadDate"):
                    d = _parse_date(item.get(key)) if isinstance(item.get(key), str) else None
                    if d:
                        return d
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return None


def _clean(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def builtin_extract(html: str) -> Extracted:
    """Dependency-free extraction (used as fallback and in tests)."""
    parser = _TextParser()
    try:
        parser.feed(html)
    except Exception:  # malformed HTML: keep what was parsed
        pass
    text = _clean("".join(parser.parts))
    published = None
    for key in _DATE_META_KEYS:
        published = _parse_date(parser.meta.get(key))
        if published:
            break
    published = published or _jsonld_date(parser.jsonld)
    if not published and parser.time_values:
        published = _parse_date(parser.time_values[0])
    lang, conf = detect_language(text)
    html_lang = normalize_lang_code(parser.html_lang)
    if html_lang and (lang is None or conf < 0.5):
        lang, conf = html_lang, max(conf, 0.6)
    return Extracted(
        title=_clean("".join(parser.title_parts)),
        text=text,
        language=lang,
        language_confidence=conf,
        published_date=published,
        extractor="builtin",
    )


def extract(html: str, url: str | None = None) -> Extracted:
    """Extract title, main text, language and publication date from HTML."""
    fallback = builtin_extract(html)
    try:
        import trafilatura
    except ImportError:
        return fallback

    try:
        doc = trafilatura.bare_extraction(
            html, url=url, with_metadata=True, include_comments=False, favor_recall=True
        )
    except Exception:
        doc = None
    if doc is None:
        return fallback
    data = doc.as_dict() if hasattr(doc, "as_dict") else dict(doc)
    text = _clean(data.get("text") or "")
    if len(text) < 0.2 * len(fallback.text):
        text = fallback.text  # trafilatura dropped most of the page; keep recall
    published = _parse_date(data.get("date")) or fallback.published_date
    if published is None:
        try:
            from htmldate import find_date

            published = _parse_date(find_date(html, original_date=True, url=url))
        except Exception:
            published = None
    lang, conf = detect_language(text)
    if lang is None or conf < 0.5:
        lang, conf = (
            (fallback.language, fallback.language_confidence)
            if fallback.language
            else (
                lang,
                conf,
            )
        )
    return Extracted(
        title=(data.get("title") or fallback.title or "").strip(),
        text=text,
        language=lang,
        language_confidence=conf,
        published_date=published,
        extractor="trafilatura",
    )
