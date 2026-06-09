from plugins.gardener_approvals.decisions import Decision
from plugins.gardener_approvals.parser import parse_reply

# Helper: plain decisions (no PR) matching the old open_ids interface
def _plain(id_: str) -> Decision:
    return Decision(id=id_, pr_number=None)

def _merge(id_: str, pr: int) -> Decision:
    return Decision(id=id_, pr_number=pr)


ONE = [_plain("D-2026-06-09-01")]
TWO = [_plain("D-2026-06-09-01"), _plain("D-2026-06-09-02")]


def test_bare_approve_one_open_resolves():
    r = parse_reply("Approve", ONE)
    assert r.matched and r.action == "resolve" and r.value == "approve"
    assert r.decision_id == "D-2026-06-09-01" and r.ask is None


def test_bare_hold_one_open():
    r = parse_reply("hold", ONE)
    assert r.matched and r.action == "resolve" and r.value == "hold"
    assert r.decision_id == "D-2026-06-09-01"


def test_explicit_id_used_even_with_many_open():
    r = parse_reply("approve D-2026-06-09-02", TWO)
    assert r.matched and r.decision_id == "D-2026-06-09-02" and r.value == "approve"


def test_intent_but_ambiguous_asks_and_does_not_match():
    r = parse_reply("approve", TWO)
    assert r.matched is False and r.ask is not None and "D-" in r.ask


def test_no_intent_is_noop():
    r = parse_reply("what's the weather", ONE)
    assert r.matched is False and r.ask is None


def test_conflicting_intent_is_noop():
    r = parse_reply("approve but actually hold", ONE)
    assert r.matched is False and r.ask is None


def test_emoji_thumbsup_approves():
    r = parse_reply("\U0001F44D", ONE)
    assert r.matched and r.value == "approve"


def test_no_open_decisions_noop():
    r = parse_reply("approve", [])
    assert r.matched is False and r.ask is None


def test_disapprove_is_noop():
    r = parse_reply("I disapprove D-2026-06-09-01", ONE)
    assert r.matched is False and r.ask is None


def test_not_approved_is_noop():
    r = parse_reply("not approved", ONE)
    assert r.matched is False and r.ask is None


def test_uphold_is_noop():
    r = parse_reply("uphold the plan", ONE)
    assert r.matched is False


def test_yesterday_is_noop():
    r = parse_reply("I saw it yesterday", ONE)
    assert r.matched is False


# --- New tests for merge support ---

def test_approve_on_pr_decision_produces_merge():
    decisions = [_merge("D-2026-06-09-01", 27)]
    r = parse_reply("approve", decisions)
    assert r.matched
    assert r.action == "merge"
    assert r.value == "27"
    assert r.decision_id == "D-2026-06-09-01"


def test_merge_it_on_pr_decision_produces_merge():
    decisions = [_merge("D-2026-06-09-01", 27)]
    r = parse_reply("merge it", decisions)
    assert r.matched
    assert r.action == "merge"
    assert r.value == "27"
    assert r.decision_id == "D-2026-06-09-01"


def test_merge_word_on_pr_decision_produces_merge():
    decisions = [_merge("D-2026-06-09-01", 27)]
    r = parse_reply("merge", decisions)
    assert r.matched
    assert r.action == "merge"
    assert r.value == "27"


def test_hold_on_pr_decision_still_resolves_hold():
    decisions = [_merge("D-2026-06-09-01", 27)]
    r = parse_reply("hold", decisions)
    assert r.matched
    assert r.action == "resolve"
    assert r.value == "hold"
    assert r.decision_id == "D-2026-06-09-01"


def test_plain_decision_approve_still_resolves_approve():
    decisions = [_plain("D-2026-06-09-01")]
    r = parse_reply("approve", decisions)
    assert r.matched
    assert r.action == "resolve"
    assert r.value == "approve"


def test_explicit_id_picks_pr_decision_among_mixed_two():
    # Two open: one with PR, one plain. Explicit id targets the merge one.
    decisions = [
        _merge("D-2026-06-09-01", 27),
        _plain("D-2026-06-09-02"),
    ]
    r = parse_reply("approve D-2026-06-09-01", decisions)
    assert r.matched
    assert r.action == "merge"
    assert r.value == "27"
    assert r.decision_id == "D-2026-06-09-01"


def test_explicit_id_picks_plain_decision_among_mixed_two():
    # Two open: one with PR, one plain. Explicit id targets the plain one.
    decisions = [
        _merge("D-2026-06-09-01", 27),
        _plain("D-2026-06-09-02"),
    ]
    r = parse_reply("approve D-2026-06-09-02", decisions)
    assert r.matched
    assert r.action == "resolve"
    assert r.value == "approve"
    assert r.decision_id == "D-2026-06-09-02"


# --- has_approval_intent tests ---

from plugins.gardener_approvals.parser import has_approval_intent


def test_intent_approve():
    assert has_approval_intent("approve") is True

def test_intent_approved():
    assert has_approval_intent("Approved!") is True

def test_intent_yes():
    assert has_approval_intent("yes") is True

def test_intent_lgtm():
    assert has_approval_intent("lgtm") is True

def test_intent_merge():
    assert has_approval_intent("merge") is True

def test_intent_merge_it():
    assert has_approval_intent("merge it") is True

def test_intent_hold():
    assert has_approval_intent("hold") is True

def test_intent_emoji_thumbsup():
    assert has_approval_intent("\U0001F44D") is True

def test_intent_emoji_check():
    assert has_approval_intent("✅") is True

def test_intent_ship_it():
    assert has_approval_intent("ship it") is True

def test_intent_go_ahead():
    assert has_approval_intent("go ahead") is True

def test_no_intent_unrelated_chatter():
    assert has_approval_intent("hey what's up") is False

def test_no_intent_yesterday():
    assert has_approval_intent("I saw it yesterday") is False

def test_no_intent_disapprove():
    # "disapprove" contains "approve" as substring but word-boundary check should block it
    assert has_approval_intent("I disapprove") is False

def test_no_intent_not_approved():
    # approve + not = conflicting → False per conservative rules
    # NOTE: has_approval_intent is intentionally True if ANY approve/hold/merge
    # keyword is present WITHOUT conflict filtering — re-read spec: it should be
    # conservative: "not approved" → False.
    # The spec says same word-boundary/phrase logic; "not approved" → hold+approve → conflict → noop
    # So has_approval_intent("not approved") → False (conflict logic applies).
    assert has_approval_intent("not approved") is False

def test_no_intent_uphold():
    assert has_approval_intent("uphold the plan") is False

def test_intent_nope():
    # "nope" is in _HOLD_WORDS, so it counts as intent
    assert has_approval_intent("nope") is True

def test_no_intent_empty():
    assert has_approval_intent("") is False
