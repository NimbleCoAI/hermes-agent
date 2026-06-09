import pytest
from plugins.gardener_approvals.decisions import Decision
from plugins.gardener_approvals.handler import on_pre_llm_call
from plugins.gardener_approvals.config import Config

CFG = Config(enabled=True, gate_path="/g.sh",
             decisions_file="/d.md", approver="Juniper")

# Helpers
def _plain(id_: str) -> Decision:
    return Decision(id=id_, pr_number=None)

def _merge(id_: str, pr: int) -> Decision:
    return Decision(id=id_, pr_number=pr)

ONE = [_plain("D-2026-06-09-01")]


def _call(**over):
    kw = dict(platform="signal", user_message="approve",
              sender_id="+15550001234", sender_id_alt="U-ACI",
              _config=CFG, _open_reader=lambda p: list(ONE),
              _gate_caller=lambda **k: ("OK", "✓ noted, approved — D-2026-06-09-01 resolved"))
    kw.update(over)
    return on_pre_llm_call(**kw)


def test_non_signal_platform_noop():
    assert _call(platform="cli") == ""


def test_no_open_decisions_noop():
    assert _call(_open_reader=lambda p: []) == ""


def test_prefers_uuid_alt_over_number():
    seen = {}
    def gate(**k): seen.update(k); return ("OK", "✓ done")
    _call(_gate_caller=gate)
    assert seen["uuid"] == "U-ACI"   # not the phone number


def test_falls_back_to_sender_id_when_alt_empty():
    seen = {}
    def gate(**k): seen.update(k); return ("OK", "✓ done")
    _call(sender_id_alt="", sender_id="U-FROM-ENVELOPE", _gate_caller=gate)
    assert seen["uuid"] == "U-FROM-ENVELOPE"


def test_ok_injects_relay_instruction_with_message():
    out = _call()
    assert isinstance(out, dict)
    assert "✓ noted, approved" in out["context"]
    assert "relay" in out["context"].lower()


def test_refused_auth_injects_nothing():
    assert _call(_gate_caller=lambda **k: ("REFUSED_AUTH", "(ignored)")) == ""


def test_error_token_is_honest_no_false_success():
    out = _call(_gate_caller=lambda **k: ("ERROR", "boom"))
    assert isinstance(out, dict)
    low = out["context"].lower()
    assert "couldn't" in low or "could not" in low
    assert "approved" not in low


def test_ambiguous_two_open_asks_without_calling_gate():
    called = {"n": 0}
    def gate(**k): called["n"] += 1; return ("OK", "x")
    out = _call(user_message="approve",
                _open_reader=lambda p: [_plain("D-2026-06-09-01"), _plain("D-2026-06-09-02")],
                _gate_caller=gate)
    assert isinstance(out, dict) and "D-" in out["context"]
    assert called["n"] == 0


def test_no_intent_noop():
    assert _call(user_message="hey what's up") == ""


def test_hook_never_raises_on_internal_error():
    def boom(p): raise RuntimeError("disk gone")
    assert _call(_open_reader=boom) == ""   # fail-safe -> empty, no exception


@pytest.mark.parametrize("token,msg", [
    ("ALREADY_DONE", "already handled"),
    ("REFUSED_ALLOWLIST", "do it in Claude Code"),
    ("AMBIGUOUS", "Which decision?"),
])
def test_relay_tokens_produce_relay(token, msg):
    out = _call(_gate_caller=lambda **k: (token, msg))
    assert isinstance(out, dict) and msg in out["context"]


def test_no_sender_id_noop():
    assert _call(sender_id="", sender_id_alt="") == ""


# --- New test: merge path ---

def test_merge_decision_approve_calls_gate_with_merge_action():
    """approve on a PR-bearing decision → gate receives action=merge, value=str(pr#)."""
    seen = {}
    def gate(**k): seen.update(k); return ("OK", "✓ merged PR #27")

    out = on_pre_llm_call(
        platform="signal",
        user_message="approve",
        sender_id_alt="U-ACI",
        _config=CFG,
        _open_reader=lambda p: [_merge("D-2026-06-09-01", 27)],
        _gate_caller=gate,
    )

    assert seen["action"] == "merge"
    assert seen["value"] == "27"
    assert seen["decision_id"] == "D-2026-06-09-01"
    assert isinstance(out, dict)
    assert "merged PR #27" in out["context"]
