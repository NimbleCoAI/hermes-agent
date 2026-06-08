from plugins.gardener_approvals.parser import parse_reply

ONE = ["D-2026-06-09-01"]
TWO = ["D-2026-06-09-01", "D-2026-06-09-02"]


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
