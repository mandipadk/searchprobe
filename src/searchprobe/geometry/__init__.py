"""Embedding geometry analysis for understanding why search fails."""

from searchprobe.geometry.analyzer import EmbeddingGeometryAnalyzer
from searchprobe.geometry.models import (
    CategoryGeometryProfile,
    EmbeddingPair,
    GeometryReport,
    SimilarityResult,
)

__all__ = [
    "EmbeddingGeometryAnalyzer",
    "CategoryGeometryProfile",
    "EmbeddingPair",
    "GeometryReport",
    "SimilarityResult",
]
