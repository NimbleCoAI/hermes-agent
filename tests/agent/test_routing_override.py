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


def test_apply_is_turn_scoped_with_dedicated_flag():
    """After applying, scoping uses a DEDICATED flag — NOT reactive-fallback state.

    _primary_runtime stays == cheap (so in-turn recovery paths don't jump tiers),
    the premium snapshot is stashed for the revert, _routing_override_active is
    set, and _fallback_activated is left untouched (arming it would corrupt
    reactive cooldown accounting — audit Major 3)."""
    agent = _fake_agent()
    apply_routing_override(agent, {"model": "deepseek/deepseek-v3.2",
                                   "provider": "openrouter"})
    # _primary_runtime is the CHEAP runtime during the routed turn.
    assert agent._primary_runtime == {"model": "deepseek/deepseek-v3.2",
                                      "provider": "openrouter"}
    # The premium snapshot is stashed for restore_primary_runtime to revert.
    assert agent._routing_override_saved_primary == {"model": "claude-sonnet-5",
                                                     "provider": "anthropic"}
    assert agent._routing_override_active is True
    # Reactive-fallback state is NOT touched.
    assert agent._fallback_activated is False


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


# ---- base_url resolution on cross-provider routes (cyborg 404 regression) ----
#
# The plugin emits {"model": ..., "provider": ...} with NO base_url. switch_model
# keeps the CURRENT base_url when none is supplied (its `if base_url:` guard), so
# a route from an Anthropic primary to an OpenRouter model would dispatch the
# OpenRouter model to api.anthropic.com/chat/completions → 404, silently
# cascading to the bottom of the fallback chain. apply_routing_override must
# resolve the new provider's canonical endpoint up front when the override omits
# it AND the provider changes.


class _FakeClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key


def _stub_resolver(monkeypatch, fn):
    """Inject a fake ``agent.auxiliary_client`` so the lazy
    ``from agent.auxiliary_client import resolve_provider_client`` in
    apply_routing_override picks up ``fn`` without importing the heavy real
    module (and its runtime deps) at unit-test time."""
    import sys
    import types

    mod = types.ModuleType("agent.auxiliary_client")
    mod.resolve_provider_client = fn
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", mod)


def test_apply_resolves_base_url_when_provider_changes(monkeypatch):
    agent = _fake_agent()  # primary: anthropic
    seen = {}

    def _resolve(provider, model=None, **kw):
        seen["provider"] = provider
        seen["model"] = model
        return _FakeClient("https://openrouter.ai/api/v1/", "sk-or-test"), model

    _stub_resolver(monkeypatch, _resolve)

    ok = apply_routing_override(
        agent, {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"}
    )
    assert ok is True
    assert seen["provider"] == "openrouter"
    _, kwargs = agent.switch_model.call_args
    # The resolved endpoint (and key) must be forwarded so switch_model does NOT
    # inherit the Anthropic primary's base_url.
    assert kwargs.get("base_url") == "https://openrouter.ai/api/v1/"
    assert kwargs.get("api_key") == "sk-or-test"


def test_apply_does_not_resolve_when_provider_unchanged(monkeypatch):
    """Same-provider route (e.g. anthropic sonnet → anthropic haiku) must not
    inject base_url — switch_model correctly keeps the same-provider base_url
    and re-derives api_mode. Forcing a resolve here is needless and risky."""
    agent = _fake_agent()  # primary: anthropic

    def _resolve(*a, **k):
        raise AssertionError("resolve_provider_client should not be called")

    _stub_resolver(monkeypatch, _resolve)

    ok = apply_routing_override(
        agent, {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"}
    )
    assert ok is True
    _, kwargs = agent.switch_model.call_args
    assert "base_url" not in kwargs


def test_apply_preserves_explicit_base_url(monkeypatch):
    """An override that DOES carry base_url wins — no resolve, no overwrite."""
    agent = _fake_agent()

    def _resolve(*a, **k):
        raise AssertionError("resolve_provider_client should not be called")

    _stub_resolver(monkeypatch, _resolve)

    apply_routing_override(
        agent,
        {
            "model": "deepseek/deepseek-v3.2",
            "provider": "openrouter",
            "base_url": "https://custom.example/v1",
        },
    )
    _, kwargs = agent.switch_model.call_args
    assert kwargs["base_url"] == "https://custom.example/v1"


def test_apply_resolve_failure_is_fail_safe(monkeypatch):
    """A resolve miss must not break the turn — the swap still proceeds and
    switch_model falls back to provider defaults."""
    agent = _fake_agent()

    def _resolve(*a, **k):
        raise RuntimeError("registry down")

    _stub_resolver(monkeypatch, _resolve)

    ok = apply_routing_override(
        agent, {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"}
    )
    assert ok is True
    agent.switch_model.assert_called_once()
