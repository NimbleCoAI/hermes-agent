"""register(ctx) — wire intelligent routing onto the ``pre_llm_call`` hook.

Enabling the plugin (``hermes plugins enable intelligent_routing``) attaches the
hook; the PER-AGENT ``routing.intelligent`` toggle (config.yaml, default OFF)
gates whether it does anything. When the toggle is OFF the hook returns ``None``
— zero behavior change, the opt-in guarantee.

When ON, the hook runs the Option-A heuristic classifier over cheap local
signals derived from the hook kwargs and returns:
  1. a context injection that EXTENDS the model-awareness line
     (``agent/model_awareness.py``, spec B1) with the routing reason, e.g.::

         [Current model: claude-sonnet-5 via anthropic — routed: mechanical → cheap]

     so a user is never confused by a silent tier switch mid-thread; and
  2. for a cheap-tier turn, a ``route`` override
     (``{"model": ..., "provider": ...}``) that the runtime ACTUATES via the
     routing-override interface extension (``agent/routing_override.py``), which
     swaps the model for THIS turn (turn-scoped, fail-safe). Premium / uncertain
     / public-facing turns emit NO override and stay on the configured primary.

This is the clean plugin form of upstream PR #43534, and the interface extension
is teknium1's offered "add to the plugin interface" made concrete: a
``pre_llm_call`` result may now carry a per-turn model/provider override.

The reactive fallback chain (``try_activate_fallback`` / ``FailoverReason``) is
deliberately untouched — intelligent routing is a proactive pre-turn layer on a
different axis; reactive escalation still applies underneath the chosen tier.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .classifier import (
    TIER_CHEAP,
    TurnSignals,
    classify_turn,
    decide_tier,
)
from .config import cheap_tier_target, is_intelligent_routing_enabled

logger = logging.getLogger(__name__)

# Platforms that are non-interactive by nature (cron/scheduled/background jobs).
# A turn arriving on one of these is mechanical unless a stronger judgment
# signal fires. Kept as a small, explicit set rather than guessing.
_NON_INTERACTIVE_PLATFORMS = frozenset({"cron", "background", "scheduler", "batch"})


def _signals_from_kwargs(**kwargs: Any) -> TurnSignals:
    """Derive cheap ``TurnSignals`` from the ``pre_llm_call`` kwargs.

    Only uses signals that are genuinely free to read here: the message text,
    and the platform (to tell a background job from a live human chat). Richer
    signals (tool-call history, kanban shape, public-facing) are wired as they
    become reliably available from the hook context; absent, they default to the
    conservative value that keeps a turn OUT of the cheap tier.
    """
    message = kwargs.get("user_message") or ""
    if not isinstance(message, str):
        message = str(message)
    platform = (kwargs.get("platform") or "")
    platform = platform.strip().lower() if isinstance(platform, str) else ""
    is_background = platform in _NON_INTERACTIVE_PLATFORMS
    return TurnSignals(
        user_message=message,
        is_interactive=not is_background,
        is_background=is_background,
        is_public_facing=bool(kwargs.get("is_public_facing", False)),
    )


def probe_route(message: str, **signals: Any) -> dict:
    """Probe the routing decision for a message in ISOLATION — no config, no hook.

    Returns ``{"tier": "cheap"|"premium", "classification": ...}`` so a live-probe
    or re-audit can feed a batch of representative messages and see exactly what
    each would route to, without mocking config or the pre_llm_call plumbing.
    ``signals`` accepts the same kwargs as the hook (e.g. ``platform="cron"``,
    ``is_public_facing=True``).
    """
    sig = _signals_from_kwargs(user_message=message, **signals)
    return {"tier": decide_tier(sig), "classification": classify_turn(sig)}


def _reason_line(model: Any, provider: Any, reason: str) -> str:
    """Build the extended model-awareness line: ``[... — routed: <reason>]``.

    Reuses ``format_current_model_line`` so the base format stays byte-identical
    to the existing model-awareness injection, then appends the routing reason.
    """
    from agent.model_awareness import format_current_model_line

    base = format_current_model_line(
        str(model or ""), str(provider or "")
    )
    if not base:
        # No known model -> emit a minimal, still-informative line.
        base = "[Current model: unknown]"
    # Splice the reason inside the trailing bracket: "[...]" -> "[... — routed: x]".
    if base.endswith("]"):
        return f"{base[:-1]} — routed: {reason}]"
    return f"{base} — routed: {reason}"


def on_pre_llm_call(**kwargs: Any) -> Optional[dict]:
    """pre_llm_call hook. Inert when disabled; otherwise inject the routing line.

    Never raises into the turn loop — any error is swallowed and treated as
    "inject nothing" (the safe, behavior-preserving default).
    """
    try:
        if not is_intelligent_routing_enabled():
            return None

        sig = _signals_from_kwargs(**kwargs)

        # THE routing decision — deliberately SIMPLE (the classifier is not the
        # contribution; the interface extension is). decide_tier() gates cheap on
        # the CONSERVATIVE binary classifier: cheap ONLY on a confident MECHANICAL,
        # else premium. No catch-all-to-cheap.
        tier = decide_tier(sig)

        # Reason line reflects the ACTUAL decider (the binary classification), so
        # the label never disagrees with the tier.
        classification = classify_turn(sig)  # mechanical | judgment | uncertain
        reason = f"{classification} → {tier}"

        line = _reason_line(kwargs.get("model"), kwargs.get("provider"), reason)
        result: dict = {"context": line}

        # Emit a `route` override ONLY for the cheap tier — this is what the
        # runtime's routing-override extension (agent/routing_override.py)
        # actuates. Premium / fail-open turns emit no override and stay on the
        # configured primary.
        if tier == TIER_CHEAP:
            cheap_model, cheap_provider = cheap_tier_target()
            if cheap_model:
                result["route"] = {"model": cheap_model, "provider": cheap_provider}
        return result
    except Exception as exc:  # noqa: BLE001 — must never break the turn
        logger.warning("intelligent_routing: pre_llm_call failed (inert): %s", exc)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info(
        "intelligent_routing active: pre_llm_call classifier registered "
        "(per-agent toggle routing.intelligent gates behavior; default OFF)"
    )
