"""Tests for the intelligent-routing pre_llm_call hook wiring.

The hook:
  - registers on ``pre_llm_call``,
  - is INERT when the per-agent toggle is OFF (returns None -> zero behavior
    change, the opt-in guarantee),
  - when ON, classifies the turn and returns a context injection that extends
    the model-awareness line with the routing reason, e.g.
    ``[Current model: <model> via OpenRouter — routed: mechanical]``,
  - fails open (routes premium / does not claim "mechanical") on uncertainty,
  - never raises into the turn loop.
"""
from unittest.mock import MagicMock, patch

from plugins.intelligent_routing import registration


class FakeCtx:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)


def test_registers_pre_llm_call_hook():
    ctx = FakeCtx()
    registration.register(ctx)
    assert "pre_llm_call" in ctx.hooks


def test_inert_when_disabled():
    """Toggle OFF -> hook returns None (no context injected, no behavior change)."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=False):
        result = registration.on_pre_llm_call(
            user_message="mark card 42 done", model="claude-sonnet-5", platform="cli",
        )
    assert result is None


def test_mechanical_turn_injects_routing_reason_when_enabled():
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="mark card 42 done", model="claude-sonnet-5", platform="cli",
        )
    assert result is not None
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: mechanical" in ctx_text
    assert "Current model:" in ctx_text


def test_judgment_turn_injects_premium_reason_when_enabled():
    long_open_ended = (
        "Think through the tradeoffs and recommend an approach with full reasoning. "
        * 4
    )
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message=long_open_ended, model="claude-sonnet-5", platform="cli",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: judgment" in ctx_text


def test_uncertain_turn_fails_open_to_judgment_reason():
    """An empty message is uncertain -> reason must be judgment (fail open),
    never 'mechanical'."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="", model="claude-sonnet-5", platform="cli",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: mechanical" not in ctx_text
    assert "routed: judgment" in ctx_text


def test_hook_never_raises_on_bad_input():
    """Defensive: bad/missing kwargs must not raise into the turn loop."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        # No user_message, weird model type.
        result = registration.on_pre_llm_call(model=None)
    # Either None or a well-formed injection, but no exception.
    assert result is None or "Current model:" in (
        result["context"] if isinstance(result, dict) else result
    )


def test_background_context_classified_mechanical():
    """A non-interactive/background platform routes mechanical when enabled."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="run nightly digest", model="claude-sonnet-5",
            platform="cron",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: mechanical" in ctx_text
