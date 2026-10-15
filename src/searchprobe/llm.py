"""Shared async helpers for calling Claude with schema-validated JSON output.

Every LLM-backed component (judge, constraint parser, query generator,
verifier) goes through :func:`structured_call`, so model selection, refusal
handling, prompt caching, and cost accounting live in one place.

Notes on the current API surface (2026):
- Current models reject sampling parameters (``temperature`` etc.), so
  determinism comes from structured outputs + caching, not temperature 0.
- Forced ``tool_choice`` is rejected on current models; JSON output is
  obtained with ``output_config.format`` (JSON schema) instead.
- On the first-party API, server-side refusal fallbacks are enabled with
  ``fallbacks="default"`` so a policy decline is retried on a fallback model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from searchprobe.config import Settings, get_settings
from searchprobe.core.exceptions import ConfigurationError, SearchProbeError

# USD per million tokens (input, output). Used for cost tracking only.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(SearchProbeError):
    """An LLM call failed or returned unusable output."""


@dataclass
class StructuredResult:
    """Parsed JSON output plus usage accounting."""

    data: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost of a call; unknown models are priced like Opus."""
    in_price, out_price = MODEL_PRICING.get(model, MODEL_PRICING["claude-opus-5-5"])
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def is_vertex(settings: Settings) -> bool:
    return bool(settings.use_vertex_ai and settings.vertex_project_id)


def get_async_client(settings: Settings | None = None) -> Any:
    """Create an async Anthropic client (direct API or Vertex AI)."""
    import anthropic

    settings = settings or get_settings()
    if is_vertex(settings):
        return anthropic.AsyncAnthropicVertex(
            project_id=settings.vertex_project_id, region=settings.vertex_region
        )
    if settings.anthropic_api_key:
        return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    raise ConfigurationError(
        "No Anthropic credentials configured. Set SEARCHPROBE_ANTHROPIC_API_KEY, or "
        "SEARCHPROBE_USE_VERTEX_AI=true with SEARCHPROBE_VERTEX_PROJECT_ID."
    )


async def structured_call(
    client: Any,
    *,
    model: str,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    max_tokens: int = 16000,
    effort: str | None = None,
    use_fallbacks: bool = True,
) -> StructuredResult:
    """Call Claude and return JSON that conforms to ``schema``.

    Args:
        client: An ``AsyncAnthropic``/``AsyncAnthropicVertex`` client (or a fake
            exposing ``beta.messages.create``/``messages.create``).
        model: Model ID.
        system: Stable system prompt (cached).
        prompt: Per-request user content.
        schema: JSON schema for the output object.
        max_tokens: Output ceiling (thinking counts against it).
        effort: Optional ``output_config.effort`` level.
        use_fallbacks: Enable server-side refusal fallbacks (first-party API only).

    Raises:
        LLMError: On refusal, truncation, or unparseable output.
    """
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
    if effort:
        output_config["effort"] = effort

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": prompt}],
        "output_config": output_config,
    }

    vertex = type(client).__name__.endswith("Vertex")
    if use_fallbacks and not vertex and hasattr(client, "beta"):
        response = await client.beta.messages.create(
            betas=[FALLBACK_BETA], fallbacks="default", **kwargs
        )
    else:
        response = await client.messages.create(**kwargs)

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise LLMError(f"Model refused the request (category={category})")
    if stop_reason == "max_tokens":
        raise LLMError("Model output truncated at max_tokens")

    text = "".join(
        getattr(block, "text", "")
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Unparseable JSON output: {e}") from e
    if not isinstance(data, dict):
        raise LLMError("Structured output was not a JSON object")

    usage = getattr(response, "usage", None)
    return StructuredResult(
        data=data,
        model=getattr(response, "model", None) or model,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        raw_text=text,
    )
