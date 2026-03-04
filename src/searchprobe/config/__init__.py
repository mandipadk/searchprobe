"""Configuration management for SearchProbe."""

from searchprobe.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "get_anthropic_client"]


def get_anthropic_client(settings: Settings | None = None) -> "anthropic.Anthropic":
    """Create an Anthropic client using either direct API key or Vertex AI.

    Args:
        settings: Application settings. Uses cached settings if not provided.

    Returns:
        An Anthropic or AnthropicVertex client instance.

    Raises:
        ValueError: If neither direct API key nor Vertex AI is configured.
    """
    import anthropic

    if settings is None:
        settings = get_settings()

    if settings.use_vertex_ai and settings.vertex_project_id:
        from anthropic import AnthropicVertex

        return AnthropicVertex(
            project_id=settings.vertex_project_id,
            region=settings.vertex_region,
        )

    if settings.anthropic_api_key:
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)

    raise ValueError(
        "No Anthropic credentials configured. Either set SEARCHPROBE_ANTHROPIC_API_KEY "
        "or enable Vertex AI with SEARCHPROBE_USE_VERTEX_AI=true and "
        "SEARCHPROBE_VERTEX_PROJECT_ID=your-project-id"
    )
