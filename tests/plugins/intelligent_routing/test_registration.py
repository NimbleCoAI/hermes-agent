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


def test_mechanical_turn_injects_task_type_and_cheap_tier():
    """A mechanical task surfaces its task type AND the cheap tier it routes to."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="grep for TODO in the repo", model="claude-sonnet-5",
            platform="cli",
        )
    assert result is not None
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: mechanical → cheap" in ctx_text
    assert "Current model:" in ctx_text


def test_architecture_turn_routes_premium():
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="design a system for multi-tenant routing across the fleet",
            model="claude-sonnet-5", platform="cli",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: architecture → premium" in ctx_text


def test_code_gen_turn_routes_premium():
    """Code-gen -> premium (DeepSWE-grounded: keep coding on Claude)."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="implement the retry logic and open a PR",
            model="claude-sonnet-5", platform="cli",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: code-gen → premium" in ctx_text


def test_uncertain_turn_fails_open_to_premium_reason():
    """An empty message is uncertain -> must route premium (fail open), never cheap."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="", model="claude-sonnet-5", platform="cli",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "→ cheap" not in ctx_text
    assert "→ premium" in ctx_text


def test_public_facing_never_cheap_routed():
    """Public-facing dominates: reason must show premium even if mechanical-shaped."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="grep the announcement", model="claude-sonnet-5",
            platform="cli", is_public_facing=True,
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "→ premium" in ctx_text
    assert "→ cheap" not in ctx_text


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


def test_background_context_classified_mechanical_cheap():
    """A non-interactive/background platform routes mechanical → cheap when enabled."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="run nightly digest", model="claude-sonnet-5",
            platform="cron",
        )
    ctx_text = result["context"] if isinstance(result, dict) else result
    assert "routed: mechanical → cheap" in ctx_text


# ------------------------------------- route override emission (actuation) ----

def test_cheap_route_emits_route_override():
    """A cheap-tier turn emits a `route` override targeting the configured cheap
    model so the runtime can actuate the swap."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="grep for TODO", model="claude-sonnet-5", platform="cli",
        )
    assert isinstance(result, dict)
    assert result.get("route") == {
        "model": "deepseek/deepseek-v3.2", "provider": "openrouter",
    }


def test_premium_route_emits_no_override():
    """A premium turn must NOT emit a route override (stay on the primary)."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="design a system for multi-tenant routing across the fleet",
            model="claude-sonnet-5", platform="cli",
        )
    assert isinstance(result, dict)
    assert "route" not in result


def test_public_facing_emits_no_override():
    """Public-facing never emits a cheap route override."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True):
        result = registration.on_pre_llm_call(
            user_message="grep the announcement", model="claude-sonnet-5",
            platform="cli", is_public_facing=True,
        )
    assert "route" not in result


def test_route_override_honors_config_target():
    """The emitted override uses the per-agent configured cheap target."""
    with patch("plugins.intelligent_routing.registration.is_intelligent_routing_enabled",
               return_value=True), \
         patch("plugins.intelligent_routing.registration.cheap_tier_target",
               return_value=("qwen/qwen-2.5", "ollama")):
        result = registration.on_pre_llm_call(
            user_message="grep for TODO", model="claude-sonnet-5", platform="cli",
        )
    assert result["route"] == {"model": "qwen/qwen-2.5", "provider": "ollama"}
