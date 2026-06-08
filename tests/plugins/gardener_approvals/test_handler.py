from plugins.gardener_approvals.handler import on_pre_llm_call
from plugins.gardener_approvals.config import Config

CFG = Config(enabled=True, gate_path="/g.sh",
             decisions_file="/d.md", approver="Juniper")
ONE = ["D-2026-06-09-01"]


def _call(**over):
    kw = dict(platform="signal", user_message="approve",
              sender_id="+64221977149", sender_id_alt="U-ACI",
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
                _open_reader=lambda p: ["D-2026-06-09-01", "D-2026-06-09-02"],
                _gate_caller=gate)
    assert isinstance(out, dict) and "D-" in out["context"]
    assert called["n"] == 0


def test_no_intent_noop():
    assert _call(user_message="hey what's up") == ""


def test_hook_never_raises_on_internal_error():
    def boom(p): raise RuntimeError("disk gone")
    assert _call(_open_reader=boom) == ""   # fail-safe -> empty, no exception
