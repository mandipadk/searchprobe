"""Run manifests: everything needed to know what produced a number."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from searchprobe import __version__

_PACKAGES = (
    "anthropic",
    "exa-py",
    "tavily-python",
    "google-search-results",
    "httpx",
    "trafilatura",
    "sentence-transformers",
)


def _git_sha() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str]:
    out = {}
    for pkg in _PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


def build_manifest(settings: Any | None = None, **extra: Any) -> dict[str, Any]:
    """Manifest for a run (git SHA, versions, judge config, plus caller-supplied fields)."""
    manifest: dict[str, Any] = {
        "searchprobe_version": __version__,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "packages": _versions(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if settings is not None:
        manifest["judge_model"] = getattr(settings, "judge_model", None)
        manifest["judge_effort"] = getattr(settings, "judge_effort", None)
    manifest.update({k: v for k, v in extra.items() if v is not None})
    return manifest
