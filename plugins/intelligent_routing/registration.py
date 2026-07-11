"""register(ctx) — wire intelligent routing onto the ``pre_llm_call`` hook.

Enabling the plugin (``hermes plugins enable intelligent_routing``) attaches the
hook; the PER-AGENT ``routing.intelligent`` toggle (config.yaml, default OFF)
gates whether it does anything. When the toggle is OFF the hook returns ``None``
— zero behavior change, the opt-in guarantee.

When ON, the hook runs the Option-A heuristic classifier over cheap local
signals derived from the hook kwargs and returns a context injection that
EXTENDS the existing model-awareness line (``agent/model_awareness.py``,
spec B1) with the routing reason, e.g.::

    [Current model: claude-sonnet-5 via anthropic — routed: mechanical]

so a user is never confused by a silent tier switch mid-thread (spec §"Where
this plugs in").

── Honest scope / the hook-surface limitation ──
This hook can only INJECT CONTEXT; its return value cannot override
``agent.model`` / ``agent.provider`` for the turn (the model is frozen by
``restore_primary_runtime`` at ``turn_context.py:174``, ~250 lines before this
hook fires, and the hook receives ``model`` as a read-only value, not the agent).
So this Stage-1 slice computes and SURFACES the routing decision but does not yet
ACTUATE the model swap. Actuation needs the plugin-interface extension teknium1
offered on PR #43534: a per-turn hook whose return value can set
``model``/``provider`` before the client is built. See README for the concrete
ask.

The reactive fallback chain (``try_activate_fallback`` /
``FailoverReason``) is deliberately untouched — intelligent routing is a
proactive pre-turn layer on a different axis.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .classifier import MECHANICAL, TurnSignals, classify_turn
from .config import is_intelligent_routing_enabled

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
    )


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
        classification = classify_turn(sig)
        # route_for semantics: only a confident MECHANICAL reaches cheap; both
        # JUDGMENT and UNCERTAIN surface as "judgment" (fail open — we never tell
        # the user a turn was cheap-routed unless it was confidently mechanical).
        reason = "mechanical" if classification == MECHANICAL else "judgment"

        line = _reason_line(kwargs.get("model"), kwargs.get("provider"), reason)
        return {"context": line}
    except Exception as exc:  # noqa: BLE001 — must never break the turn
        logger.warning("intelligent_routing: pre_llm_call failed (inert): %s", exc)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info(
        "intelligent_routing active: pre_llm_call classifier registered "
        "(per-agent toggle routing.intelligent gates behavior; default OFF)"
    )
