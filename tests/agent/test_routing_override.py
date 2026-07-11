"""Tests for the pre-turn routing-override interface extension.

This is the minimal reference implementation of the plugin-interface extension
teknium1 offered on PR #43534: a ``pre_llm_call`` hook result may carry a
``{"route": {"model": ..., "provider": ...}}`` override that the runtime applies
BEFORE the turn's client is built — turning a plugin routing DECISION into an
actual model swap.

Design (see agent/routing_override.py):
  - ``extract_routing_override(results)`` pulls the first well-formed override
    out of the list of pre_llm_call hook results.
  - ``apply_routing_override(agent, override)`` actuates the swap by delegating
    to the agent's existing, tested ``switch_model`` machinery, then makes the
    swap TURN-SCOPED (reverted next turn by ``restore_primary_runtime``) by
    restoring the pre-swap ``_primary_runtime`` snapshot and arming
    ``_fallback_activated``. Fail-safe: any error leaves the agent untouched.
"""
from unittest.mock import MagicMock

import pytest

from agent.routing_override import (
    apply_routing_override,
    extract_routing_override,
)


# ------------------------------------------------ extract_routing_override ----

def test_extract_none_when_no_results():
    assert extract_routing_override([]) is None


def test_extract_none_when_only_context():
    results = [{"context": "[Current model: x]"}, "some string"]
    assert extract_routing_override(results) is None


def test_extract_pulls_route_dict():
    results = [{"context": "line", "route": {"model": "deepseek/deepseek-v3.2",
                                             "provider": "openrouter"}}]
    ov = extract_routing_override(results)
    assert ov == {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"}


def test_extract_requires_model_key():
    # A route dict with no model is not actionable.
    assert extract_routing_override([{"route": {"provider": "openrouter"}}]) is None


def test_extract_first_wins():
    results = [
        {"route": {"model": "a", "provider": "p1"}},
        {"route": {"model": "b", "provider": "p2"}},
    ]
    assert extract_routing_override(results)["model"] == "a"


# ------------------------------------------------- apply_routing_override -----

def _fake_agent():
    agent = MagicMock()
    agent.model = "claude-sonnet-5"
    agent.provider = "anthropic"
    agent._primary_runtime = {"model": "claude-sonnet-5", "provider": "anthropic"}
    agent._fallback_activated = False

    # switch_model mutates model/provider like the real one, and (like the real
    # one) would persist _primary_runtime — we assert the override code re-scopes.
    def _switch(new_model, new_provider, **kw):
        agent.model = new_model
        agent.provider = new_provider
        agent._primary_runtime = {"model": new_model, "provider": new_provider}
        return {"ok": True}

    agent.switch_model.side_effect = _switch
    return agent


def test_apply_actuates_the_swap():
    agent = _fake_agent()
    ok = apply_routing_override(agent, {"model": "deepseek/deepseek-v3.2",
                                        "provider": "openrouter"})
    assert ok is True
    assert agent.model == "deepseek/deepseek-v3.2"
    assert agent.provider == "openrouter"
    agent.switch_model.assert_called_once()


def test_apply_is_turn_scoped():
    """After applying, _primary_runtime must still point at the CONFIGURED primary
    and _fallback_activated must be armed, so restore_primary_runtime reverts the
    swap next turn (proactive routing is turn-scoped, like reactive fallback)."""
    agent = _fake_agent()
    apply_routing_override(agent, {"model": "deepseek/deepseek-v3.2",
                                   "provider": "openrouter"})
    assert agent._primary_runtime == {"model": "claude-sonnet-5", "provider": "anthropic"}
    assert agent._fallback_activated is True


def test_apply_noop_when_target_equals_current():
    """No swap when the override already matches the live model (avoid churn)."""
    agent = _fake_agent()
    ok = apply_routing_override(agent, {"model": "claude-sonnet-5",
                                        "provider": "anthropic"})
    assert ok is False
    agent.switch_model.assert_not_called()
    # _fallback_activated untouched — nothing was swapped.
    assert agent._fallback_activated is False


def test_apply_fail_safe_on_switch_error():
    """If switch_model raises, the agent is left untouched and we return False."""
    agent = _fake_agent()
    agent.switch_model.side_effect = RuntimeError("bad key")
    ok = apply_routing_override(agent, {"model": "deepseek/deepseek-v3.2",
                                        "provider": "openrouter"})
    assert ok is False
    # Original model/provider preserved (switch_model itself rolls back; the
    # applier must not leave a half-state or re-raise into the turn loop).
    assert agent.model == "claude-sonnet-5"


def test_apply_ignores_empty_override():
    agent = _fake_agent()
    assert apply_routing_override(agent, {}) is False
    assert apply_routing_override(agent, None) is False
    agent.switch_model.assert_not_called()
