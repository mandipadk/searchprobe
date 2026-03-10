"""Aris configuration via Pydantic Settings.

Loads from environment variables with ARIS_ prefix and optional .env file.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class ArisConfig(BaseSettings):
    """Central Aris configuration.

    All secrets come from environment variables (ARIS_ prefix).
    Non-secret defaults are set here.
    """

    model_config = {"env_prefix": "ARIS_", "env_file": ".env", "extra": "ignore"}

    # LLM
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude")
    parser_model: str = Field(default="claude-haiku-4-5-20251001", description="Model for QUE parsing")
    verifier_model: str = Field(
        default="claude-haiku-4-5-20251001", description="Model for constraint verification fallback"
    )

    # Data source API keys
    brave_api_key: str = Field(default="", description="Brave Search API key")
    serpapi_api_key: str = Field(default="", description="SerpAPI key")

    # Retrieval
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence-transformers model for dense retrieval"
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )
    default_num_results: int = Field(default=10, ge=1, le=100)
    max_candidates: int = Field(default=100, ge=10, le=500)

    # Agent
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max search refinement iterations")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Index paths
    index_dir: str = Field(default=".aris/index", description="Directory for index storage")
    cache_dir: str = Field(default=".aris/cache", description="Directory for document cache")
