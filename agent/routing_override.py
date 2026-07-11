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
tested ``switch_model`` (full atomic client rebuild + rollback), but
``switch_model`` persists the swap into ``_primary_runtime`` (it's built for the
session-scoped ``/model`` command). So after the swap we:
  1. restore the pre-swap ``_primary_runtime`` snapshot, and
  2. arm ``_fallback_activated``,
which makes ``restore_primary_runtime`` revert the swap at the top of the NEXT
turn — identical turn-scoping to the reactive fallback path, on a different axis.

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

    # Snapshot the configured-primary runtime so we can re-scope the swap to this
    # turn only (switch_model would otherwise persist it across turns).
    primary_snapshot = getattr(agent, "_primary_runtime", None)

    kwargs = {
        k: override[k] for k in _PASSTHROUGH_FIELDS if override.get(k)
    }
    try:
        agent.switch_model(target_model, target_provider, **kwargs)
    except Exception as exc:  # noqa: BLE001 — a routing miss must never break the turn
        logger.warning(
            "intelligent_routing: switch to %s/%s failed (%s); staying on %s/%s",
            target_model, target_provider or "?", exc, cur_model, cur_provider,
        )
        return False

    # Re-scope to this turn: restore the primary snapshot and arm the fallback
    # flag so restore_primary_runtime reverts the swap next turn.
    try:
        if primary_snapshot is not None:
            agent._primary_runtime = primary_snapshot
        agent._fallback_activated = True
    except Exception:  # noqa: BLE001 — best-effort re-scoping; swap already applied
        logger.debug("intelligent_routing: could not re-scope override to turn",
                     exc_info=True)

    logger.info(
        "intelligent_routing: routed this turn to %s/%s (was %s/%s)",
        target_model, target_provider or "?", cur_model, cur_provider,
    )
    return True
