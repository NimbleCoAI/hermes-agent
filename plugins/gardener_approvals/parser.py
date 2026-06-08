"""Pure intent parser: a Signal reply + the open decision ids -> a structured
action. No I/O, no subprocess — fully unit-testable. The gate re-validates
everything; this parser only decides what (if anything) to ask the gate to do.

v1 maps to the {resolve} action only (approve/hold). `merge` is intentionally
NOT emitted here yet — it is enabled only after the resolve loop is proven live.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_DEC_ID_RE = re.compile(r"D-\d{4}-\d{2}-\d{2}-\d+")

# Whole-word / phrase keywords. Short ambiguous words ("go", "ok") are excluded
# on purpose — a false trigger resolves a decision the user didn't mean.
_APPROVE = ("approve", "approved", "approves", "yes", "yep", "yeah", "do it",
            "go ahead", "ship it", "lgtm")
_APPROVE_EMOJI = ("\U0001F44D", "✅")  # 👍 ✅
_HOLD = ("hold", "not yet", "wait", "nope", "later")
_HOLD_WORD = ("no",)  # matched only as a standalone word


@dataclass
class ParseResult:
    matched: bool
    action: Optional[str] = None        # "resolve"
    value: Optional[str] = None         # "approve" | "hold"
    decision_id: Optional[str] = None
    ask: Optional[str] = None           # set -> inject this, DO NOT call the gate


def _has_phrase(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def _has_word(text: str, words) -> bool:
    toks = re.findall(r"[a-z]+", text)
    return any(w in toks for w in words)


def parse_reply(text: str, open_ids: List[str]) -> ParseResult:
    if not open_ids:
        return ParseResult(matched=False)
    raw = text or ""
    low = raw.lower()

    approve = _has_phrase(low, _APPROVE) or any(e in raw for e in _APPROVE_EMOJI)
    hold = _has_phrase(low, _HOLD) or _has_word(low, _HOLD_WORD)
    if approve and hold:
        return ParseResult(matched=False)            # conflicting -> no-op
    if not approve and not hold:
        return ParseResult(matched=False)            # no intent -> no-op
    value = "approve" if approve else "hold"

    # Which decision: explicit id wins; else exactly-one-open; else ambiguous.
    m = _DEC_ID_RE.search(raw)
    if m:
        dec_id = m.group(0)
    elif len(open_ids) == 1:
        dec_id = open_ids[0]
    else:
        return ParseResult(
            matched=False,
            ask=f"Which decision? Reply with the D-… id (e.g. {open_ids[0]}).",
        )
    return ParseResult(matched=True, action="resolve", value=value, decision_id=dec_id)
