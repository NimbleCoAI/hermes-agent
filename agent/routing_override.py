"""Pre-turn routing-override interface extension.

This is the minimal reference implementation of the plugin-interface extension
teknium1 offered on upstream PR #43534 ("Happy to support adding to the plugin
interface to make this work"). It lets a ``pre_llm_call`` plugin hook RETURN a
routing decision that the runtime actually ACTUATES for the turn — closing the
gap that the existing hook can only inject context, never change which model
runs (the model is frozen by ``restore_primary_runtime`` before the hook fires).

── Contract ──
A ``pre_llm_call`` hook result may be a dict carrying a ``route`` key::

    {"context": "...optional injected text...",
     "route": {"model": "deepseek/deepseek-v3.2", "provider": "openrouter"}}

``model`` is required; ``provider`` (and optional ``base_url`` / ``api_key`` /
``api_mode``) are passed through to ``switch_model``. ``turn_context`` extracts
the first well-formed override and applies it right after the hook fires, before
the turn's first API call assembles.

── Turn-scoping (the load-bearing detail) ──
Proactive routing is a PER-TURN decision, exactly like reactive fallback — it
must NOT pin the session to the cheap model. We reuse the agent's existing,
tested ``switch_model`` (full atomic client rebuild + rollback), which persists
the swap into ``_primary_runtime``. Crucially we then scope it with a DEDICATED
flag, NOT by overloading reactive-fallback state:

  1. Stash the pre-swap premium ``_primary_runtime`` in
     ``_routing_override_saved_primary``.
  2. Set ``_routing_override_active = True``.
  3. LEAVE ``_primary_runtime`` pointing at the CHEAP runtime for the turn, and
     do NOT touch ``_fallback_activated``.

``restore_primary_runtime`` then reverts the swap at the top of the NEXT turn via
a dedicated block that runs BEFORE the ``_fallback_activated`` and rate-limit
cooldown gates, so a routed turn always reverts regardless of cooldown state.

Why the dedicated flag (three audit-confirmed bugs the old overload caused):
  - Overloading ``_rate_limited_until``/``_fallback_activated`` for scoping let a
    cheap turn LEAK past its turn when a cooldown gate skipped restoration
    (Blocker 1); pointing ``_primary_runtime`` at premium made in-turn transient
    recovery jump the routed turn onto premium (Major 2); and pre-arming
    ``_fallback_activated`` corrupted the reactive cooldown accounting so a cheap
    429 got no cooldown (Major 3). Keeping ``_primary_runtime`` == cheap and using
    a dedicated flag fixes all three: reactive fallback runs normally UNDERNEATH a
    routed turn with the cheap runtime as its "primary".

── Fail-safe ──
Any error (bad override, ``switch_model`` raising on a bad key/network) leaves
the agent untouched and returns ``False`` — a routing miss must degrade to
"answer on the configured primary", never break the turn.

⚠️ AUDIT NOTE: this module + its single call site in ``turn_context.py`` are the
ONLY core-runtime changes for intelligent routing. They touch the
model-selection path (user-visible: they change which model answers a turn).
Everything else lives in the ``intelligent_routing`` plugin. Reviewers should
focus here.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Fields we forward from an override into switch_model (model/provider plus the
# optional endpoint/credential fields switch_model already accepts).
_PASSTHROUGH_FIELDS = ("base_url", "api_key", "api_mode")


def extract_routing_override(results: Iterable[Any]) -> Optional[dict]:
    """Return the first well-formed ``route`` override from hook results, or None.

    A well-formed override is a dict with a non-empty ``model``. First match
    wins (deterministic; matches how the ``pre_llm_call`` context parts are
    consumed in registration order).
    """
    if not results:
        return None
    for r in results:
        if not isinstance(r, dict):
            continue
        route = r.get("route")
        if isinstance(route, dict) and str(route.get("model") or "").strip():
            return route
    return None


def apply_routing_override(agent: Any, override: Optional[dict]) -> bool:
    """Actuate a routing override for THIS turn. Return True iff a swap happened.

    Turn-scoped and fail-safe (see module docstring). No-op (returns False) when
    the override is empty or already matches the live model/provider.
    """
    if not override or not isinstance(override, dict):
        return False

    target_model = str(override.get("model") or "").strip()
    if not target_model:
        return False
    target_provider = str(override.get("provider") or "").strip()

    cur_model = str(getattr(agent, "model", "") or "").strip()
    cur_provider = str(getattr(agent, "provider", "") or "").strip()
    # Avoid churn: nothing to do if we're already on the target.
    if target_model == cur_model and (
        not target_provider or target_provider == cur_provider
    ):
        return False

    # Stash the configured-primary runtime so restore_primary_runtime can revert
    # the swap next turn. switch_model will overwrite _primary_runtime with the
    # cheap runtime — we deliberately KEEP that (so in-turn recovery paths that
    # read _primary_runtime stay on cheap) and use the stash + a dedicated flag
    # for the revert, NOT the reactive-fallback state.
    primary_snapshot = getattr(agent, "_primary_runtime", None)

    kwargs = {
        k: override[k] for k in _PASSTHROUGH_FIELDS if override.get(k)
    }
    # When the route switches provider but the override carries no explicit
    # base_url, resolve the new provider's canonical endpoint up front. Without
    # this, switch_model()'s ``if base_url:`` guard keeps the CURRENT base_url,
    # so the routed model is dispatched to the previous provider's endpoint —
    # e.g. an OpenRouter model sent to api.anthropic.com/chat/completions → 404,
    # silently cascading to the bottom of the fallback chain. Mirrors the
    # reactive-fallback path in chat_completion_helpers. (cyborg routed-turn
    # 404 regression.) Fail-safe: a resolve miss must never break the turn —
    # fall through and let switch_model use provider defaults.
    if (
        "base_url" not in kwargs
        and target_provider
        and target_provider.strip().lower() != cur_provider.strip().lower()
    ):
        try:
            from agent.auxiliary_client import resolve_provider_client

            _client, _ = resolve_provider_client(target_provider, model=target_model)
            if _client is not None:
                _resolved_base = str(getattr(_client, "base_url", "") or "").strip()
                if _resolved_base:
                    kwargs["base_url"] = _resolved_base
                if "api_key" not in kwargs:
                    _resolved_key = getattr(_client, "api_key", None)
                    if _resolved_key:
                        kwargs["api_key"] = _resolved_key
        except Exception:  # noqa: BLE001 — never break the turn on a resolve miss
            logger.debug(
                "intelligent_routing: could not pre-resolve endpoint for %s; "
                "switch_model will fall back to provider defaults",
                target_provider, exc_info=True,
            )
    try:
        agent.switch_model(target_model, target_provider, **kwargs)
    except Exception as exc:  # noqa: BLE001 — a routing miss must never break the turn
        logger.warning(
            "intelligent_routing: switch to %s/%s failed (%s); staying on %s/%s",
            target_model, target_provider or "?", exc, cur_model, cur_provider,
        )
        return False

    # Mark the override turn-scoped with a DEDICATED flag. Do NOT restore the
    # premium snapshot into _primary_runtime (keep it == cheap) and do NOT arm
    # _fallback_activated (that would corrupt reactive cooldown accounting).
    try:
        agent._routing_override_saved_primary = primary_snapshot
        agent._routing_override_active = True
    except Exception:  # noqa: BLE001 — best-effort scoping; swap already applied
        logger.debug("intelligent_routing: could not scope override to turn",
                     exc_info=True)

    logger.info(
        "intelligent_routing: routed this turn to %s/%s (was %s/%s)",
        target_model, target_provider or "?", cur_model, cur_provider,
    )
    return True
