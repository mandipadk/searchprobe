"""Uniform evidence: fetch, extract and snapshot every judged document."""

from searchprobe.evidence.fetcher import EvidenceFetcher
from searchprobe.evidence.passages import select_passages
from searchprobe.evidence.store import EvidenceStore, Snapshot

__all__ = ["EvidenceFetcher", "EvidenceStore", "Snapshot", "select_passages"]
