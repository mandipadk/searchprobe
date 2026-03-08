"""TOML-based experiment configuration profiles.

Profiles separate reproducible experiment config (sharable, version-controlled)
from secrets (which stay in environment variables / .env files).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SearchProbeProfile(BaseModel):
    """Experiment configuration loaded from a TOML file.

    This defines WHAT to run (providers, categories, analysis toggles, parameters)
    but NOT credentials -- those come from Settings via environment variables.
    """

    name: str = Field(..., description="Human-readable profile name")
    description: str = Field(default="", description="What this experiment investigates")

    # Provider selection
    providers: list[str] = Field(default_factory=lambda: ["exa"])
    exa_modes: list[str] = Field(default_factory=lambda: ["auto"])

    # Query config
    categories: list[str] = Field(
        default_factory=list,
        description="Adversarial categories to test (empty = all)",
    )
    queries_per_category: int = Field(default=5, ge=1, le=100)
    difficulty: str = Field(default="mixed", description="easy, medium, hard, or mixed")

    # Analysis toggles
    run_geometry: bool = False
    run_perturbation: bool = False
    run_validation: bool = False
    run_evolution: bool = False

    # Geometry config
    geometry_models: list[str] = Field(
        default_factory=lambda: ["all-MiniLM-L6-v2"],
    )

    # Perturbation config
    perturbation_operators: list[str] = Field(
        default_factory=lambda: ["word_delete", "word_swap", "synonym_replace"],
    )
    perturbation_variants: int = Field(default=5, ge=1, le=20)

    # Evolution config
    evolution_generations: int = Field(default=20, ge=1)
    evolution_population: int = Field(default=30, ge=5)
    evolution_fitness_mode: str = Field(default="embedding_sim")
    evolution_budget: float = Field(default=5.0, ge=0.0)

    # Evaluation
    judge_model: str = "claude-sonnet-4-6"
    num_results: int = Field(default=10, ge=1, le=100)

    # Budgets & concurrency
    budget_limit: float = Field(default=10.0, ge=0.0)
    max_concurrent: int = Field(default=5, ge=1, le=50)

    @classmethod
    def from_toml(cls, path: str | Path) -> SearchProbeProfile:
        """Load a profile from a TOML file.

        Args:
            path: Path to the .toml file.

        Returns:
            Parsed SearchProbeProfile.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the TOML is invalid or missing required fields.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (e.g. for storing in run metadata)."""
        return self.model_dump()
