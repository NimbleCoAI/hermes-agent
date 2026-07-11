"""Pre-turn heuristic classifier (Option A) for intelligent routing.

Classifies an incoming turn as ``mechanical`` (route to the cheap-metered tier)
vs. ``judgment`` (route to the premium tier) using ONLY cheap, local signals
available before the LLM call — zero extra API cost, zero added latency. When
the signals don't add up to a confident ``mechanical`` classification, the
result is ``uncertain`` and ``route_for`` FAILS OPEN to premium: a judgment turn
must never be silently downgraded to the cheap model, because a visibly worse
answer to a human is the expensive failure mode (spec open-question #4). This
mirrors the fail-open discipline HSM's ``validateCascadeEntries`` already uses
for credential checks.

Everything here is a PURE function of ``TurnSignals`` — deterministic, no I/O,
no globals — so it is trivially testable and safe to call on the hot path.

"Mechanical" is defined concretely (spec open-question #3, per D-2026-07-08-01):
  - cron / background / non-interactive jobs, OR
  - kanban-triage-shaped requests, OR
  - short, single-tool-call-shaped asks (no multi-step reasoning),
and NEVER anything public-facing.
"Judgment" is everything substantive: open-ended / multi-step reasoning, long
human messages, or anything public-facing.
"""
from __future__ import annotations

from dataclasses import dataclass

# Classification labels.
MECHANICAL = "mechanical"
JUDGMENT = "judgment"
UNCERTAIN = "uncertain"

# Routing tiers.
TIER_CHEAP = "cheap"
TIER_PREMIUM = "premium"

# A short mechanical ask is bounded in characters. Above this, a message carries
# enough substance that we no longer treat it as a one-shot mechanical request
# and defer to the other signals (fail open if nothing else fires). Kept
# deliberately conservative — false-mechanical on a real question is the costly
# error, so the length gate errs toward "not mechanical."
_SHORT_MESSAGE_CHARS = 120

# Short length is NECESSARY but NOT SUFFICIENT for mechanical: plenty of short
# human messages are judgment calls (e.g. "Can you look at the thing we
# discussed and get back to me?"). A short interactive ask is only mechanical
# when it ALSO looks like a direct single action — a bare imperative command or
# a trivial factual/lookup query — and carries none of the deferral / prior-
# context markers that signal an open, context-dependent request.

# Leading imperative verbs that mark a direct single-action command.
_IMPERATIVE_VERBS = frozenset({
    "mark", "add", "remove", "delete", "close", "open", "set", "move", "assign",
    "list", "show", "get", "fetch", "run", "start", "stop", "restart", "rename",
    "tag", "label", "archive", "unarchive", "create", "update", "check", "sort",
    "clear", "reset", "enable", "disable", "send", "post", "pin", "unpin",
})

# Trivial factual/lookup query openers (short "what's X" style asks).
_TRIVIAL_QUERY_PREFIXES = ("what's ", "whats ", "what is ", "how many ", "when is ")

# Markers of a hedged / deferred / context-dependent request — these push a
# short message OUT of mechanical (fail open to judgment). "the thing we
# discussed", "get back to me", "look into", etc.
_DEFERRAL_MARKERS = (
    "get back to me", "look at the", "look into", "we discussed", "we talked",
    "the thing", "figure out", "think about", "your thoughts", "what do you think",
    "not sure", "somehow", "whenever you", "when you get a chance",
)


def _is_short_mechanical_ask(message: str) -> bool:
    """True when a short message is a direct single-action command or trivial query.

    Returns False for hedged/deferred/context-referencing short messages so they
    fail open to judgment rather than being silently cheap-routed.
    """
    text = message.strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _DEFERRAL_MARKERS):
        return False
    if text.startswith(_TRIVIAL_QUERY_PREFIXES):
        return True
    first_word = text.split(maxsplit=1)[0].strip(".,!?:")
    return first_word in _IMPERATIVE_VERBS


@dataclass(frozen=True)
class TurnSignals:
    """Cheap local signals for one turn, all available before the LLM call.

    These map directly onto what the ``pre_llm_call`` hook already receives
    (``user_message``, ``platform``, conversation/tool context) plus a few
    booleans the registration layer derives from the runtime/session context.
    """

    user_message: str = ""
    # Live human chat vs. non-interactive invocation.
    is_interactive: bool = True
    # Cron / scheduled / background job (non-interactive by nature).
    is_background: bool = False
    # The request is kanban-triage-shaped (assign/prioritize/sort a backlog).
    is_kanban_triage: bool = False
    # The turn is expected to require multi-step reasoning.
    expects_multi_step: bool = False
    # The output is public-facing (announcement, published content, DM to a
    # non-operator). Public-facing ALWAYS goes premium.
    is_public_facing: bool = False


def classify_turn(sig: TurnSignals) -> str:
    """Return ``MECHANICAL`` / ``JUDGMENT`` / ``UNCERTAIN`` for one turn.

    Order matters: the strongest *judgment* signals are checked first so they
    can never be overridden by a coincidental mechanical signal (e.g. a short
    but public-facing message must not be classified mechanical).
    """
    message = (sig.user_message or "").strip()

    # No signal at all -> we cannot claim it's mechanical. Fail open.
    if not message and not sig.is_background:
        return UNCERTAIN

    # --- Judgment dominators (checked first, never overridden) ---
    if sig.is_public_facing:
        return JUDGMENT
    if sig.expects_multi_step:
        return JUDGMENT
    if len(message) > _SHORT_MESSAGE_CHARS:
        return JUDGMENT

    # --- Mechanical signals ---
    if sig.is_background or not sig.is_interactive:
        return MECHANICAL
    if sig.is_kanban_triage:
        return MECHANICAL
    if (
        message
        and len(message) <= _SHORT_MESSAGE_CHARS
        and _is_short_mechanical_ask(message)
    ):
        # Short, interactive, not multi-step, not public-facing, AND shaped like
        # a direct single action or trivial lookup: a one-shot mechanical ask.
        return MECHANICAL

    # Nothing decisive -> fail open.
    return UNCERTAIN


def route_for(sig: TurnSignals) -> str:
    """Collapse a classification to a routing tier, failing open to premium.

    ``mechanical`` -> ``TIER_CHEAP``. Everything else (``judgment`` AND
    ``uncertain``) -> ``TIER_PREMIUM``. Only a confident mechanical result ever
    reaches the cheap tier.
    """
    return TIER_CHEAP if classify_turn(sig) == MECHANICAL else TIER_PREMIUM
