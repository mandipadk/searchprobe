"""Local document cache for avoiding redundant fetches."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from aris.core.models import Document

logger = logging.getLogger(__name__)


class DocumentCache:
    """Simple file-based document cache keyed by URL hash."""

    def __init__(self, cache_dir: str = ".aris/cache") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _path(self, url: str) -> Path:
        return self._dir / f"{self._key(url)}.json"

    def get(self, url: str) -> Document | None:
        path = self._path(url)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return Document(**data)
            except Exception:
                return None
        return None

    def put(self, document: Document) -> None:
        path = self._path(document.url)
        try:
            path.write_text(document.model_dump_json())
        except Exception as e:
            logger.debug("Cache write failed for %s: %s", document.url, e)

    def has(self, url: str) -> bool:
        return self._path(url).exists()
