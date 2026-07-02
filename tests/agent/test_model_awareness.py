"""Tests for model self-awareness — the live (post-fallback) model line.

Re-ports the pre-migration ``hermes-swarm`` behaviour: a
``[Current model: X via Y]`` line is appended to the ephemeral per-turn
context so the LLM can answer "what model are you?" truthfully instead of
hallucinating. The line must reflect the LIVE runtime — i.e. after
``try_activate_fallback`` mutates ``agent.model`` / ``agent.provider`` in
place it must show the FALLBACK model, not the configured primary.
"""

from agent.model_awareness import (
    append_current_model_line,
    format_current_model_line,
)


class _FakeAgent:
    """Minimal stand-in exposing the two live runtime fields the loop reads."""

    def __init__(self, model: str, provider: str) -> None:
        self.model = model
        self.provider = provider


def test_format_with_provider():
    assert (
        format_current_model_line("claude-opus-4-8", "anthropic")
        == "[Current model: claude-opus-4-8 via anthropic]"
    )


def test_format_without_provider():
    assert format_current_model_line("glm-4-9b", "") == "[Current model: glm-4-9b]"


def test_format_unknown_model_is_empty():
    # No model configured yet → inject nothing rather than a half-line.
    assert format_current_model_line("", "anthropic") == ""
    assert format_current_model_line("   ", "") == ""


def test_append_injects_line_for_configured_primary():
    agent = _FakeAgent("claude-opus-4-8", "anthropic")
    injections: list[str] = []
    append_current_model_line(injections, agent)
    assert injections == ["[Current model: claude-opus-4-8 via anthropic]"]


def test_append_reflects_fallback_after_simulated_activation():
    # Turn starts on the configured primary.
    agent = _FakeAgent("claude-opus-4-8", "anthropic")
    primary: list[str] = []
    append_current_model_line(primary, agent)
    assert primary == ["[Current model: claude-opus-4-8 via anthropic]"]

    # Simulate try_activate_fallback(): it mutates agent.model / agent.provider
    # in place. A fresh per-API-call assembly must now surface the FALLBACK.
    agent.model = "glm-4.6"
    agent.provider = "zai"
    after_fallback: list[str] = []
    append_current_model_line(after_fallback, agent)
    assert after_fallback == ["[Current model: glm-4.6 via zai]"]
    # The stale configured primary must NOT leak into the degraded turn.
    assert "claude-opus-4-8" not in after_fallback[0]


def test_append_is_noop_when_model_unknown():
    agent = _FakeAgent("", "")
    injections: list[str] = ["existing"]
    append_current_model_line(injections, agent)
    assert injections == ["existing"]
