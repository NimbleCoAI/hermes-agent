"""Security tests: R6 inbound turn throttle + daily ceilings (pure module).

An unauthenticated flood of inbound messages (a Discord raid, a compromised
allowlisted account, a runaway bot loop) must not be able to spawn unbounded
LLM turns or unbounded spend. These tests pin the fail-closed contract of
``gateway.inbound_throttle``:

  - per-user and per-scope sliding windows bind independently;
  - the daily turn/spend ceilings persist across process restarts;
  - a corrupt or unreadable ledger DENIES (a broken limiter must never
    become an open gate);
  - unknown per-turn pricing still consumes budget via the fallback cost;
  - malformed env overrides fall back to the conservative defaults instead
    of disabling enforcement.
"""

import json
import os
import sys

import pytest

from gateway.inbound_throttle import InboundThrottle, Verdict
from hermes_constants import get_hermes_home


@pytest.fixture(autouse=True)
def _enable_throttle(monkeypatch):
    """The hermetic suite disables the throttle globally; opt back in here."""
    monkeypatch.setenv("GATEWAY_THROTTLE_ENABLED", "true")
    for var in (
        "GATEWAY_THROTTLE_USER_PER_MINUTE",
        "GATEWAY_THROTTLE_USER_PER_HOUR",
        "GATEWAY_THROTTLE_SCOPE_PER_MINUTE",
        "GATEWAY_THROTTLE_SCOPE_PER_HOUR",
        "GATEWAY_THROTTLE_DAILY_TURN_CEILING",
        "GATEWAY_THROTTLE_DAILY_SPEND_USD",
        "GATEWAY_THROTTLE_FALLBACK_TURN_COST_USD",
        "GATEWAY_THROTTLE_EXEMPT_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _ledger_path():
    return get_hermes_home() / "throttle" / "ledger.json"


def _read_ledger():
    return json.loads(_ledger_path().read_text(encoding="utf-8"))


def _write_ledger(data):
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------


class TestSlidingWindows:
    def test_user_window_allows_n_then_denies(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "3")
        t = InboundThrottle()
        for _ in range(3):
            assert t.check_and_consume("discord", "u1", "g1").allowed
        v = t.check_and_consume("discord", "u1", "g1")
        assert not v.allowed
        assert v.reason == "user_rate_per_minute"

    def test_user_hour_window(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "100")
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_HOUR", "2")
        t = InboundThrottle()
        assert t.check_and_consume("discord", "u1", "g1").allowed
        assert t.check_and_consume("discord", "u1", "g1").allowed
        v = t.check_and_consume("discord", "u1", "g1")
        assert not v.allowed
        assert v.reason == "user_rate_per_hour"

    def test_scope_window_independent_of_user(self, monkeypatch):
        """Different users in the same guild share the SCOPE window: a raid
        by many fresh accounts still hits a per-guild ceiling."""
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "100")
        monkeypatch.setenv("GATEWAY_THROTTLE_SCOPE_PER_MINUTE", "3")
        t = InboundThrottle()
        for i in range(3):
            assert t.check_and_consume("discord", f"user-{i}", "guild-1").allowed
        v = t.check_and_consume("discord", "user-99", "guild-1")
        assert not v.allowed
        assert v.reason == "scope_rate_per_minute"
        # A different guild is unaffected.
        assert t.check_and_consume("discord", "user-99", "guild-2").allowed

    def test_different_users_have_independent_user_windows(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "1")
        monkeypatch.setenv("GATEWAY_THROTTLE_SCOPE_PER_MINUTE", "100")
        t = InboundThrottle()
        assert t.check_and_consume("discord", "u1", "g1").allowed
        assert not t.check_and_consume("discord", "u1", "g1").allowed
        assert t.check_and_consume("discord", "u2", "g1").allowed

    def test_no_scope_skips_scope_window(self, monkeypatch):
        """DMs (no scope) are still bound by the user window."""
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "1")
        t = InboundThrottle()
        assert t.check_and_consume("discord", "u1", None).allowed
        assert not t.check_and_consume("discord", "u1", None).allowed


# ---------------------------------------------------------------------------
# Exemptions
# ---------------------------------------------------------------------------


class TestExemptUsers:
    def test_exempt_user_bypasses_windows_but_not_daily_ceiling(self, monkeypatch):
        """Break-glass operators skip the anti-flood windows, but the daily
        ceilings protect the wallet and bind for EVERYONE."""
        monkeypatch.setenv("GATEWAY_THROTTLE_EXEMPT_USERS", "111,222")
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "1")
        monkeypatch.setenv("GATEWAY_THROTTLE_SCOPE_PER_MINUTE", "1")
        monkeypatch.setenv("GATEWAY_THROTTLE_DAILY_TURN_CEILING", "5")
        t = InboundThrottle()
        for _ in range(5):
            assert t.check_and_consume("discord", "111", "g1").allowed
        v = t.check_and_consume("discord", "111", "g1")
        assert not v.allowed
        assert v.reason == "daily_turn_ceiling"

    def test_exempt_user_does_not_consume_scope_window(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_EXEMPT_USERS", "111")
        monkeypatch.setenv("GATEWAY_THROTTLE_SCOPE_PER_MINUTE", "1")
        t = InboundThrottle()
        assert t.check_and_consume("discord", "111", "g1").allowed
        assert t.check_and_consume("discord", "111", "g1").allowed
        # Non-exempt user still has a full scope budget.
        assert t.check_and_consume("discord", "u2", "g1").allowed


# ---------------------------------------------------------------------------
# Daily ceilings (persistent ledger)
# ---------------------------------------------------------------------------


class TestDailyCeilings:
    def test_daily_turn_ceiling_binds(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_DAILY_TURN_CEILING", "2")
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "100")
        t = InboundThrottle()
        assert t.check_and_consume("discord", "u1", None).allowed
        assert t.check_and_consume("discord", "u2", None).allowed
        v = t.check_and_consume("discord", "u3", None)
        assert not v.allowed
        assert v.reason == "daily_turn_ceiling"

    def test_daily_ceiling_survives_restart(self, monkeypatch):
        """A fresh InboundThrottle (simulated restart) reads the same ledger —
        crash-looping the gateway must not reset the daily budget."""
        monkeypatch.setenv("GATEWAY_THROTTLE_DAILY_TURN_CEILING", "2")
        t1 = InboundThrottle()
        assert t1.check_and_consume("discord", "u1", None).allowed
        assert t1.check_and_consume("discord", "u1", None).allowed

        t2 = InboundThrottle()  # restart
        v = t2.check_and_consume("discord", "u1", None)
        assert not v.allowed
        assert v.reason == "daily_turn_ceiling"

    def test_utc_day_rollover_resets(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_DAILY_TURN_CEILING", "2")
        _write_ledger({"day": "2000-01-01", "turns": 999, "spend_usd": 999.0})
        t = InboundThrottle()
        assert t.check_and_consume("discord", "u1", None).allowed
        assert _read_ledger()["turns"] == 1

    def test_ledger_turns_persisted_on_allow(self):
        t = InboundThrottle()
        assert t.check_and_consume("discord", "u1", None).allowed
        assert _read_ledger()["turns"] == 1
        assert oct(os.stat(_ledger_path()).st_mode & 0o777) == oct(0o600) or (
            sys.platform == "win32"
        )


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_corrupt_ledger_denies(self):
        _ledger_path().parent.mkdir(parents=True, exist_ok=True)
        _ledger_path().write_text("{not json", encoding="utf-8")
        t = InboundThrottle()
        v = t.check_and_consume("discord", "u1", "g1")
        assert not v.allowed
        assert v.reason == "ledger_error"

    def test_wrong_schema_ledger_denies(self):
        _write_ledger({"day": "2026-01-01", "turns": "many", "spend_usd": 0.0})
        # Even a same-day-shaped but schema-invalid ledger denies.
        t = InboundThrottle()
        v = t.check_and_consume("discord", "u1", "g1")
        assert not v.allowed
        assert v.reason == "ledger_error"

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod semantics differ")
    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file modes")
    def test_unreadable_ledger_denies(self):
        _write_ledger({"day": "2026-01-01", "turns": 0, "spend_usd": 0.0})
        os.chmod(_ledger_path(), 0o000)
        try:
            t = InboundThrottle()
            v = t.check_and_consume("discord", "u1", "g1")
            assert not v.allowed
            assert v.reason == "ledger_error"
        finally:
            os.chmod(_ledger_path(), 0o600)


# ---------------------------------------------------------------------------
# Spend accounting
# ---------------------------------------------------------------------------


class TestSpend:
    def test_record_spend_accumulates_and_cap_denies(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_DAILY_SPEND_USD", "0.10")
        t = InboundThrottle()
        t.record_spend(0.06, "estimated")
        assert t.check_and_consume("discord", "u1", None).allowed
        t.record_spend(0.06, "estimated")
        v = t.check_and_consume("discord", "u1", None)
        assert not v.allowed
        assert v.reason == "daily_spend_ceiling"

    def test_none_cost_applies_fallback(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_FALLBACK_TURN_COST_USD", "0.25")
        t = InboundThrottle()
        t.record_spend(None, None)
        assert _read_ledger()["spend_usd"] == pytest.approx(0.25)

    def test_zero_cost_applies_fallback(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_FALLBACK_TURN_COST_USD", "0.25")
        t = InboundThrottle()
        t.record_spend(0.0, "estimated")
        assert _read_ledger()["spend_usd"] == pytest.approx(0.25)

    def test_unknown_status_applies_fallback_even_with_cost(self, monkeypatch):
        """A cost paired with status 'unknown' is not trusted — the fallback
        keeps unpriced turns from being free."""
        monkeypatch.setenv("GATEWAY_THROTTLE_FALLBACK_TURN_COST_USD", "0.25")
        t = InboundThrottle()
        t.record_spend(1.23, "unknown")
        assert _read_ledger()["spend_usd"] == pytest.approx(0.25)

    def test_spend_persists_across_restart(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_DAILY_SPEND_USD", "0.10")
        t1 = InboundThrottle()
        t1.record_spend(0.20, "estimated")
        t2 = InboundThrottle()  # restart
        v = t2.check_and_consume("discord", "u1", None)
        assert not v.allowed
        assert v.reason == "daily_spend_ceiling"


# ---------------------------------------------------------------------------
# Config robustness
# ---------------------------------------------------------------------------


class TestConfig:
    def test_malformed_env_falls_back_to_default_and_enforces(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "banana")
        t = InboundThrottle()
        for _ in range(4):  # default is 4/minute
            assert t.check_and_consume("discord", "u1", None).allowed
        v = t.check_and_consume("discord", "u1", None)
        assert not v.allowed
        assert v.reason == "user_rate_per_minute"

    def test_negative_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_DAILY_SPEND_USD", "-5")
        t = InboundThrottle()
        t.record_spend(5.0, "estimated")
        # Default cap 10.0 still enforced (5 < 10 → allowed).
        assert t.check_and_consume("discord", "u1", None).allowed

    def test_disabled_allows_everything(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_ENABLED", "false")
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "0")
        t = InboundThrottle()
        for _ in range(20):
            assert t.check_and_consume("discord", "u1", "g1").allowed

    def test_notify_cooldown(self):
        t = InboundThrottle()
        assert t.should_notify("discord:u1")
        assert not t.should_notify("discord:u1")
        assert t.should_notify("discord:u2")  # independent per key
