"""Application settings loaded from environment variables and .env files."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SearchProbe configuration settings.

    All settings can be configured via environment variables with the
    SEARCHPROBE_ prefix, or via a .env file in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SEARCHPROBE_",
        extra="ignore",
    )

    # Search Provider API Keys
    exa_api_key: str | None = None
    tavily_api_key: str | None = None
    brave_api_key: str | None = None
    serpapi_api_key: str | None = None

    # LLM API Keys (for query generation and evaluation)
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Database
    database_path: str = "searchprobe.db"

    # Search Defaults
    default_num_results: int = 10
    default_max_content_chars: int = 5000
    max_concurrent_requests: int = 5

    # Cost Control
    budget_limit_usd: float = 10.0

    # LLM Settings
    judge_model: str = "claude-sonnet-4-20250514"
    judge_temperature: float = 0.0
    generation_model: str = "claude-sonnet-4-20250514"
    generation_temperature: float = 0.9

    @property
    def configured_providers(self) -> list[str]:
        """Return list of providers that have API keys configured."""
        providers = []
        if self.exa_api_key:
            providers.append("exa")
        if self.tavily_api_key:
            providers.append("tavily")
        if self.brave_api_key:
            providers.append("brave")
        if self.serpapi_api_key:
            providers.append("serpapi")
        return providers

    def has_llm_configured(self) -> bool:
        """Check if at least one LLM provider is configured."""
        return bool(self.anthropic_api_key or self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
