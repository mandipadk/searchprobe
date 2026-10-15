"""Read and write test-case datasets (JSON Lines, one ``TestCase`` per line)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from searchprobe.cases.models import TestCase

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "cases"


def load_cases(path: str | Path) -> list[TestCase]:
    """Load cases from a ``.jsonl`` file or every ``.jsonl`` file in a directory."""
    path = Path(path)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    cases: list[TestCase] = []
    for file in files:
        with open(file, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(TestCase.model_validate_json(line))
                except ValueError as e:
                    raise ValueError(f"{file}:{lineno}: invalid test case: {e}") from e
    return cases


def save_cases(cases: Iterable[TestCase], path: str | Path) -> int:
    """Write cases to a ``.jsonl`` file (sorted by id for stable diffs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(cases, key=lambda c: (c.category.value, c.id))
    with open(path, "w", encoding="utf-8") as f:
        for case in ordered:
            f.write(json.dumps(case.model_dump(mode="json", exclude_defaults=False)) + "\n")
    return len(ordered)


def merge_cases(existing: list[TestCase], new: list[TestCase]) -> list[TestCase]:
    """Merge by id; a gold case is never overwritten by a non-gold one."""
    by_id = {c.id: c for c in existing}
    for case in new:
        old = by_id.get(case.id)
        if old is not None and old.validation == "gold" and case.validation != "gold":
            continue
        by_id[case.id] = case
    return list(by_id.values())
