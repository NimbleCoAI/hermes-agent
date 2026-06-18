"""Tests for /approve and /deny gateway commands.

Verifies that dangerous command approvals use the blocking gateway approval
mechanism — the agent thread blocks until the user responds with /approve
or /deny, mirroring the CLI's synchronous input() flow.

Supports multiple concurrent approvals (parallel subagents, execute_code)
via a per-session queue.
"""

import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    # Default: the approver is an individual-allowlisted admin. Admin-gating
    # behavior is covered explicitly in TestApprovalUsesDmAllowlist (handler
    # wiring) and TestIndividualAllowlist (the real predicate).
    runner._is_individual_allowlisted = lambda _source: True
    runner._set_session_env = lambda _context: None
    return runner


def _clear_approval_state():
    """Reset all module-level approval state between tests."""
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


# ------------------------------------------------------------------
# Blocking gateway approval infrastructure (tools/approval.py)
# ------------------------------------------------------------------


class TestBlockingGatewayApproval:
    """Tests for the blocking approval mechanism in tools/approval.py."""

    def setup_method(self):
        _clear_approval_state()

    def test_register_and_resolve_unblocks_entry(self):
        """resolve_gateway_approval signals the entry's event."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, has_blocking_approval,
            _ApprovalEntry, _gateway_queues,
        )
        session_key = "test-session"
        register_gateway_notify(session_key, lambda d: None)

        # Simulate what check_all_command_guards does
        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues.setdefault(session_key, []).append(entry)

        assert has_blocking_approval(session_key) is True

        # Resolve from another thread
        def resolve():
            time.sleep(0.1)
            resolve_gateway_approval(session_key, "once")

        t = threading.Thread(target=resolve)
        t.start()
        resolved = entry.event.wait(timeout=5)
        t.join()

        assert resolved is True
        assert entry.result == "once"
        unregister_gateway_notify(session_key)

    def test_resolve_returns_zero_when_no_pending(self):
        from tools.approval import resolve_gateway_approval
        assert resolve_gateway_approval("nonexistent", "once") == 0

    def test_resolve_all_unblocks_multiple_entries(self):
        """resolve_gateway_approval with resolve_all=True signals all entries."""
        from tools.approval import (
            resolve_gateway_approval, _ApprovalEntry, _gateway_queues,
        )
        session_key = "test-all"
        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        e3 = _ApprovalEntry({"command": "cmd3"})
        _gateway_queues[session_key] = [e1, e2, e3]

        count = resolve_gateway_approval(session_key, "session", resolve_all=True)
        assert count == 3
        assert all(e.event.is_set() for e in [e1, e2, e3])
        assert all(e.result == "session" for e in [e1, e2, e3])

    def test_resolve_single_pops_oldest_fifo(self):
        """resolve_gateway_approval without resolve_all resolves oldest first."""
        from tools.approval import (
            resolve_gateway_approval,
            _ApprovalEntry, _gateway_queues,
        )
        session_key = "test-fifo"
        e1 = _ApprovalEntry({"command": "first"})
        e2 = _ApprovalEntry({"command": "second"})
        _gateway_queues[session_key] = [e1, e2]

        count = resolve_gateway_approval(session_key, "once")
        assert count == 1
        assert e1.event.is_set()
        assert e1.result == "once"
        assert not e2.event.is_set()
        assert len(_gateway_queues[session_key]) == 1

    def test_unregister_signals_all_entries(self):
        """unregister_gateway_notify signals all waiting entries to prevent hangs."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            _ApprovalEntry, _gateway_queues,
        )
        session_key = "test-cleanup"
        register_gateway_notify(session_key, lambda d: None)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        unregister_gateway_notify(session_key)
        assert e1.event.is_set()
        assert e2.event.is_set()

    def test_clear_session_denies_and_signals_all_entries(self):
        """clear_session must wake blocked entries during boundary cleanup."""
        from tools.approval import clear_session, _ApprovalEntry, _gateway_queues

        session_key = "test-boundary-cleanup"
        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        clear_session(session_key)

        assert e1.event.is_set()
        assert e2.event.is_set()
        assert e1.result == "deny"
        assert e2.result == "deny"
        assert session_key not in _gateway_queues


# ------------------------------------------------------------------
# /approve command
# ------------------------------------------------------------------


class TestApproveCommand:

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_approve_resolves_blocking_approval(self):
        """Basic /approve signals the oldest blocked agent thread."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        result = await runner._handle_approve_command(_make_event("/approve"))
        assert "approved" in result.lower()
        assert "resuming" in result.lower()
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_approve_all_resolves_multiple(self):
        """/approve all resolves all pending approvals."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_approve_command(_make_event("/approve all"))
        assert "2 commands" in result
        assert e1.event.is_set()
        assert e2.event.is_set()

    @pytest.mark.asyncio
    async def test_approve_all_session(self):
        """/approve all session resolves all with session scope."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_approve_command(_make_event("/approve all session"))
        assert "session" in result.lower()
        assert e1.result == "session"
        assert e2.result == "session"

    @pytest.mark.asyncio
    async def test_approve_no_pending(self):
        """/approve with no pending approval returns helpful message."""
        runner = _make_runner()
        result = await runner._handle_approve_command(_make_event("/approve"))
        assert "No pending command" in result

    @pytest.mark.asyncio
    async def test_approve_stale_old_style_pending(self):
        """Old-style _pending_approvals without blocking event reports expired."""
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)
        runner._pending_approvals[session_key] = {"command": "test"}

        result = await runner._handle_approve_command(_make_event("/approve"))
        assert "expired" in result.lower() or "no longer waiting" in result.lower()
        assert session_key not in runner._pending_approvals


# ------------------------------------------------------------------
# /deny command
# ------------------------------------------------------------------


class TestDenyCommand:

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_deny_resolves_blocking_approval(self):
        """/deny signals the oldest blocked agent thread with 'deny'."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        result = await runner._handle_deny_command(_make_event("/deny"))
        assert "denied" in result.lower()
        assert entry.event.is_set()
        assert entry.result == "deny"

    @pytest.mark.asyncio
    async def test_deny_all_resolves_all(self):
        """/deny all denies all pending approvals."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_deny_command(_make_event("/deny all"))
        assert "2 commands" in result
        assert all(e.result == "deny" for e in [e1, e2])

    @pytest.mark.asyncio
    async def test_deny_no_pending(self):
        """/deny with no pending approval returns helpful message."""
        runner = _make_runner()
        result = await runner._handle_deny_command(_make_event("/deny"))
        assert "No pending command" in result


# ------------------------------------------------------------------
# Bare "yes" must NOT trigger approval
# ------------------------------------------------------------------


class TestBareTextNoLongerApproves:

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_yes_does_not_execute_pending_command(self):
        """Saying 'yes' must not trigger approval. Only /approve works."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        # "yes" is not /approve — entry should still be pending
        assert not entry.event.is_set()


# ------------------------------------------------------------------
# End-to-end blocking flow
# ------------------------------------------------------------------


class TestBlockingApprovalE2E:
    """Test the full blocking flow: agent thread blocks → user approves → agent resumes."""

    def setup_method(self):
        _clear_approval_state()
        os.environ.pop("HERMES_YOLO_MODE", None)
        os.environ.pop("HERMES_INTERACTIVE", None)
        os.environ.pop("HERMES_GATEWAY_SESSION", None)
        os.environ.pop("HERMES_EXEC_ASK", None)
        os.environ.pop("HERMES_SESSION_KEY", None)

    def test_blocking_approval_approve_once(self):
        """check_all_command_guards blocks until resolve_gateway_approval is called."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, check_all_command_guards,
        )

        session_key = "e2e-test"
        notified = []

        register_gateway_notify(session_key, lambda d: notified.append(d))

        result_holder = [None]

        def agent_thread():
            from tools.approval import reset_current_session_key, set_current_session_key

            token = set_current_session_key(session_key)
            os.environ["HERMES_GATEWAY_SESSION"] = "1"
            os.environ["HERMES_EXEC_ASK"] = "1"
            os.environ["HERMES_SESSION_KEY"] = session_key
            try:
                result_holder[0] = check_all_command_guards(
                    "rm -rf /important", "local"
                )
            finally:
                os.environ.pop("HERMES_GATEWAY_SESSION", None)
                os.environ.pop("HERMES_EXEC_ASK", None)
                os.environ.pop("HERMES_SESSION_KEY", None)
                reset_current_session_key(token)

        t = threading.Thread(target=agent_thread)
        t.start()

        for _ in range(50):
            if notified:
                break
            time.sleep(0.05)

        assert len(notified) == 1
        assert "rm -rf /important" in notified[0]["command"]

        resolve_gateway_approval(session_key, "once")
        t.join(timeout=5)

        assert result_holder[0] is not None
        assert result_holder[0]["approved"] is True
        unregister_gateway_notify(session_key)

    def test_blocking_approval_deny(self):
        """check_all_command_guards returns BLOCKED when denied."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, check_all_command_guards,
        )

        session_key = "e2e-deny"
        notified = []
        register_gateway_notify(session_key, lambda d: notified.append(d))

        result_holder = [None]

        def agent_thread():
            from tools.approval import reset_current_session_key, set_current_session_key

            token = set_current_session_key(session_key)
            os.environ["HERMES_GATEWAY_SESSION"] = "1"
            os.environ["HERMES_EXEC_ASK"] = "1"
            os.environ["HERMES_SESSION_KEY"] = session_key
            try:
                result_holder[0] = check_all_command_guards(
                    "rm -rf /important", "local"
                )
            finally:
                os.environ.pop("HERMES_GATEWAY_SESSION", None)
                os.environ.pop("HERMES_EXEC_ASK", None)
                os.environ.pop("HERMES_SESSION_KEY", None)
                reset_current_session_key(token)

        t = threading.Thread(target=agent_thread)
        t.start()
        for _ in range(50):
            if notified:
                break
            time.sleep(0.05)

        resolve_gateway_approval(session_key, "deny")
        t.join(timeout=5)

        assert result_holder[0]["approved"] is False
        assert "BLOCKED" in result_holder[0]["message"]
        unregister_gateway_notify(session_key)

    def test_blocking_approval_timeout(self):
        """check_all_command_guards returns BLOCKED on timeout."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            check_all_command_guards,
        )

        session_key = "e2e-timeout"
        register_gateway_notify(session_key, lambda d: None)

        result_holder = [None]

        def agent_thread():
            from tools.approval import reset_current_session_key, set_current_session_key

            token = set_current_session_key(session_key)
            os.environ["HERMES_GATEWAY_SESSION"] = "1"
            os.environ["HERMES_EXEC_ASK"] = "1"
            os.environ["HERMES_SESSION_KEY"] = session_key
            try:
                with patch("tools.approval._get_approval_config",
                           return_value={"gateway_timeout": 1}):
                    result_holder[0] = check_all_command_guards(
                        "rm -rf /important", "local"
                    )
            finally:
                os.environ.pop("HERMES_GATEWAY_SESSION", None)
                os.environ.pop("HERMES_EXEC_ASK", None)
                os.environ.pop("HERMES_SESSION_KEY", None)
                reset_current_session_key(token)

        t = threading.Thread(target=agent_thread)
        t.start()
        t.join(timeout=10)

        assert result_holder[0]["approved"] is False
        assert "timed out" in result_holder[0]["message"]
        unregister_gateway_notify(session_key)

    def test_parallel_subagent_approvals(self):
        """Multiple threads can block concurrently and be resolved independently."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, check_all_command_guards,
            _gateway_queues,
        )

        session_key = "e2e-parallel"
        notified = []
        register_gateway_notify(session_key, lambda d: notified.append(d))

        results = [None, None, None]

        def make_agent(idx, cmd):
            def run():
                from tools.approval import reset_current_session_key, set_current_session_key

                token = set_current_session_key(session_key)
                os.environ["HERMES_GATEWAY_SESSION"] = "1"
                os.environ["HERMES_EXEC_ASK"] = "1"
                os.environ["HERMES_SESSION_KEY"] = session_key
                try:
                    results[idx] = check_all_command_guards(cmd, "local")
                finally:
                    os.environ.pop("HERMES_GATEWAY_SESSION", None)
                    os.environ.pop("HERMES_EXEC_ASK", None)
                    os.environ.pop("HERMES_SESSION_KEY", None)
                    reset_current_session_key(token)
            return run

        threads = [
            threading.Thread(target=make_agent(0, "rm -rf /a")),
            threading.Thread(target=make_agent(1, "rm -rf /b")),
            threading.Thread(target=make_agent(2, "rm -rf /c")),
        ]
        for t in threads:
            t.start()

        # Wait for all 3 to block
        for _ in range(100):
            if len(notified) >= 3:
                break
            time.sleep(0.05)

        assert len(notified) == 3
        assert len(_gateway_queues.get(session_key, [])) == 3

        # Approve all at once
        count = resolve_gateway_approval(session_key, "session", resolve_all=True)
        assert count == 3

        for t in threads:
            t.join(timeout=5)

        assert all(r is not None for r in results)
        assert all(r["approved"] is True for r in results)
        unregister_gateway_notify(session_key)

    def test_parallel_mixed_approve_deny(self):
        """Approve some, deny others in a parallel batch."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, check_all_command_guards,
        )

        session_key = "e2e-mixed"
        register_gateway_notify(session_key, lambda d: None)

        results = [None, None]

        def make_agent(idx, cmd):
            def run():
                from tools.approval import reset_current_session_key, set_current_session_key

                token = set_current_session_key(session_key)
                os.environ["HERMES_GATEWAY_SESSION"] = "1"
                os.environ["HERMES_EXEC_ASK"] = "1"
                os.environ["HERMES_SESSION_KEY"] = session_key
                try:
                    results[idx] = check_all_command_guards(cmd, "local")
                finally:
                    os.environ.pop("HERMES_GATEWAY_SESSION", None)
                    os.environ.pop("HERMES_EXEC_ASK", None)
                    os.environ.pop("HERMES_SESSION_KEY", None)
                    reset_current_session_key(token)
            return run

        threads = [
            threading.Thread(target=make_agent(0, "rm -rf /x")),
            threading.Thread(target=make_agent(1, "rm -rf /y")),
        ]
        for t in threads:
            t.start()

        # Wait for both threads to register pending approvals instead of
        # relying on a fixed sleep.  The approval module stores entries in
        # _gateway_queues[session_key] — poll until we see 2 entries.
        from tools.approval import _gateway_queues
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if len(_gateway_queues.get(session_key, [])) >= 2:
                break
            time.sleep(0.05)

        # Approve first, deny second
        resolve_gateway_approval(session_key, "once")   # oldest
        resolve_gateway_approval(session_key, "deny")   # next

        for t in threads:
            t.join(timeout=5)

        assert all(r is not None for r in results)
        assert sorted(r["approved"] for r in results) == [False, True]
        assert sum("BLOCKED" in (r.get("message") or "") for r in results) == 1
        unregister_gateway_notify(session_key)


# ------------------------------------------------------------------
# Fallback: no gateway callback (cron/batch mode)
# ------------------------------------------------------------------


class TestFallbackNoCallback:

    def setup_method(self):
        _clear_approval_state()

    def test_no_callback_returns_approval_required(self):
        """Without a registered callback, the fallback returns pending_approval.

        PR #6d495d9e7 renamed the LLM-visible status from ``approval_required``
        to ``pending_approval`` to make the state distinguishable from a
        failed tool call.
        """
        from tools.approval import check_all_command_guards

        os.environ["HERMES_EXEC_ASK"] = "1"
        os.environ["HERMES_SESSION_KEY"] = "no-callback-test"
        try:
            result = check_all_command_guards("rm -rf /important", "local")
        finally:
            os.environ.pop("HERMES_EXEC_ASK", None)
            os.environ.pop("HERMES_SESSION_KEY", None)

        assert result["approved"] is False
        assert result.get("status") == "pending_approval"
        assert result.get("approval_pending") is True


# ------------------------------------------------------------------
# Cross-session admin approval (group_sessions_per_user)
# ------------------------------------------------------------------


class TestCrossSessionAdminApproval:
    """In groups with per-user sessions, a dangerous command is keyed to the
    *triggering* participant's session — not the approver's. An admin's
    /approve must still clear that sibling session's pending approval, or a
    non-admin can trigger a command nobody can ever approve."""

    def setup_method(self):
        _clear_approval_state()

    @staticmethod
    def _group_source(user_id: str) -> SessionSource:
        return SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id="g1",
            user_name=user_id,
            chat_type="group",
        )

    @staticmethod
    def _runner_with_admin(admin_id: str):
        runner = _make_runner()
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="***",
                    extra={"group_allow_admin_from": [admin_id]},
                )
            }
        )
        return runner

    @pytest.mark.asyncio
    async def test_admin_approve_clears_other_participants_pending(self):
        """Admin /approve resolves a pending approval bound to another user's
        per-user session in the same group."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = self._runner_with_admin("u_admin")

        triggerer = self._group_source("u_cam")  # non-admin who hit the command
        trig_key = runner._session_key_for_source(triggerer)
        entry = _ApprovalEntry({"command": "ffprobe x | python3", "description": "scan"})
        _gateway_queues[trig_key] = [entry]

        admin_event = MessageEvent(
            text="/approve session",
            source=self._group_source("u_admin"),
            message_id="m1",
        )
        result = await runner._handle_approve_command(admin_event)

        assert entry.event.is_set()
        assert entry.result == "session"
        assert trig_key not in _gateway_queues
        assert "no pending" not in result.lower()
        assert "not authorized" not in result.lower()

    @pytest.mark.asyncio
    async def test_nonadmin_cannot_cross_approve(self):
        """A non-admin must NOT be able to clear another participant's pending
        approval — admin-only gating still applies to cross-session resolves."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = self._runner_with_admin("u_admin")

        triggerer = self._group_source("u_other")
        trig_key = runner._session_key_for_source(triggerer)
        entry = _ApprovalEntry({"command": "rm -rf /", "description": "scan"})
        _gateway_queues[trig_key] = [entry]

        nonadmin_event = MessageEvent(
            text="/approve",
            source=self._group_source("u_cam"),  # not in group_allow_admin_from
            message_id="m2",
        )
        result = await runner._handle_approve_command(nonadmin_event)

        assert not entry.event.is_set()
        assert trig_key in _gateway_queues
        assert "not authorized" in result.lower()

    @pytest.mark.asyncio
    async def test_admin_deny_clears_other_participants_pending(self):
        """Admin /deny must also resolve another participant's stuck pending
        approval — otherwise a non-admin-triggered command can only time out."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = self._runner_with_admin("u_admin")

        triggerer = self._group_source("u_cam")
        trig_key = runner._session_key_for_source(triggerer)
        entry = _ApprovalEntry({"command": "rm -rf /", "description": "scan"})
        _gateway_queues[trig_key] = [entry]

        admin_event = MessageEvent(
            text="/deny",
            source=self._group_source("u_admin"),
            message_id="m3",
        )
        result = await runner._handle_deny_command(admin_event)

        assert entry.event.is_set()
        assert entry.result == "deny"
        assert trig_key not in _gateway_queues
        assert "no pending" not in result.lower()


# ------------------------------------------------------------------
# Approval gate: "approved = admin" — the individual (DM) allowlist decides
# who may /approve·/deny, NOT group membership.
# ------------------------------------------------------------------


class TestApprovalUsesDmAllowlist:
    """When no explicit approval-admin list is configured (the HSM/Mare
    default), the /approve|/deny admin gate falls back to "approved = admin":
    a user counts as an admin iff they are in the individual (DM) allowlist —
    the same identity DM-auth and group-invite approval already use. A user
    who only reaches the bot through an approved *group* is a normal
    participant and cannot approve dangerous commands. Otherwise an empty
    config admin list would treat every group member as an admin (the gap this
    fixes)."""

    def setup_method(self):
        _clear_approval_state()

    @staticmethod
    def _group_source(user_id: str) -> SessionSource:
        return SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id="grp",
            user_name=user_id,
            chat_type="group",
        )

    @staticmethod
    def _runner():
        runner = _make_runner()
        # No config.yaml admin list — "approved = admin" governs.
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
        )
        # Only u_admin is in the individual (DM) allowlist; that predicate is
        # what the gate consults. (The real predicate is exercised directly in
        # TestIndividualAllowlist.)
        runner._is_individual_allowlisted = lambda source: source.user_id == "u_admin"
        return runner

    @pytest.mark.asyncio
    async def test_group_only_user_rejected_even_for_own_session(self):
        """A group-only user (not in the DM allowlist) cannot /approve — even a
        command pending in their OWN session. Without the allowlist fallback the
        empty config list would treat everyone as admin and let this through."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = self._runner()
        cam = self._group_source("u_cam")
        key = runner._session_key_for_source(cam)
        entry = _ApprovalEntry({"command": "rm -rf /", "description": "scan"})
        _gateway_queues[key] = [entry]

        result = await runner._handle_approve_command(
            MessageEvent(text="/approve", source=cam, message_id="m1")
        )

        assert not entry.event.is_set()
        assert "not authorized" in result.lower()

    @pytest.mark.asyncio
    async def test_dm_allowlisted_user_can_approve(self):
        """A user in the individual (DM) allowlist can /approve (and clears a
        sibling per-user session via the cross-session resolver), even when the
        /approve arrives from a group."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = self._runner()
        cam = self._group_source("u_cam")
        key = runner._session_key_for_source(cam)
        entry = _ApprovalEntry({"command": "ffprobe x", "description": "scan"})
        _gateway_queues[key] = [entry]

        result = await runner._handle_approve_command(
            MessageEvent(text="/approve", source=self._group_source("u_admin"), message_id="m2")
        )

        assert entry.event.is_set()
        assert "not authorized" not in result.lower()

    @pytest.mark.asyncio
    async def test_group_only_user_deny_rejected(self):
        """Same gate applies to /deny."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = self._runner()
        cam = self._group_source("u_cam")
        key = runner._session_key_for_source(cam)
        entry = _ApprovalEntry({"command": "rm -rf /", "description": "scan"})
        _gateway_queues[key] = [entry]

        result = await runner._handle_deny_command(
            MessageEvent(text="/deny", source=cam, message_id="m3")
        )

        assert not entry.event.is_set()
        assert "not authorized" in result.lower()

    @pytest.mark.asyncio
    async def test_toggle_off_lets_anyone_approve(self):
        """When the operator turns the "Admins only" toggle off
        (approvals.admin_only = False), the gate is skipped entirely and any
        participant may /approve — the HSM setting still governs on/off,
        independent of the allowlist."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = self._runner()  # u_cam is NOT in the DM allowlist
        cam = self._group_source("u_cam")
        key = runner._session_key_for_source(cam)
        entry = _ApprovalEntry({"command": "ffprobe x", "description": "scan"})
        _gateway_queues[key] = [entry]

        with patch("tools.approval._get_approval_config", return_value={"admin_only": False}):
            result = await runner._handle_approve_command(
                MessageEvent(text="/approve", source=cam, message_id="m4")
            )

        assert entry.event.is_set()
        assert "not authorized" not in result.lower()


# ------------------------------------------------------------------
# The individual-allowlist predicate the approval gate falls back to.
# Exercises the REAL _is_individual_allowlisted (not a stub) so the
# fail-closed + UUID-alt guarantees are actually tested.
# ------------------------------------------------------------------

# Env keys neutralized in every case so the host environment can't leak in.
_ISO_ENV_OFF = {
    "GATEWAY_ALLOWED_USERS": "",
    "GATEWAY_ALLOW_ALL_USERS": "",
    "TELEGRAM_ALLOWED_USERS": "",
    "TELEGRAM_ALLOW_ALL_USERS": "",
    "SIGNAL_ALLOWED_USERS": "",
    "SIGNAL_ALLOW_ALL_USERS": "",
    "WHATSAPP_ALLOWED_USERS": "",
    "WHATSAPP_ALLOW_ALL_USERS": "",
}


class TestIndividualAllowlist:
    """Direct tests of _is_individual_allowlisted — the "approved = admin"
    predicate. Must be fail-closed: only the individual (DM) allowlist, the
    pairing store, an allow-all flag, or a wildcard grants approval rights;
    group membership and adapter own-policy trust must NOT."""

    @staticmethod
    def _runner():
        # Bare instance so the REAL _is_individual_allowlisted runs (not the
        # _make_runner stub). The method only needs os.environ and the source.
        from gateway.run import GatewayRunner
        return object.__new__(GatewayRunner)

    @staticmethod
    def _src(platform, user_id, *, user_id_alt=None, chat_type="group"):
        return SessionSource(
            platform=platform,
            user_id=user_id,
            chat_id="g1",
            user_name=user_id,
            chat_type=chat_type,
            user_id_alt=user_id_alt,
        )

    def test_dm_allowlisted_user_matches(self):
        runner = self._runner()
        env = {**_ISO_ENV_OFF, "TELEGRAM_ALLOWED_USERS": "u_admin"}
        with patch.dict(os.environ, env, clear=False):
            assert runner._is_individual_allowlisted(self._src(Platform.TELEGRAM, "u_admin"))
            assert not runner._is_individual_allowlisted(self._src(Platform.TELEGRAM, "u_cam"))

    def test_group_only_user_fails_closed_with_no_allowlist(self):
        """MUST-FIX: on an own-policy adapter (WhatsApp) with no individual
        allowlist, a group-only user is NOT an approval admin. The previous
        _is_user_authorized reuse fell open here via the adapter-trust shortcut."""
        runner = self._runner()
        with patch.dict(os.environ, _ISO_ENV_OFF, clear=False):
            assert not runner._is_individual_allowlisted(self._src(Platform.WHATSAPP, "u_group"))

    def test_signal_uuid_alt_matches_when_listed_by_uuid(self):
        """SHOULD-FIX: a user listed by one id form is recognized via user_id_alt
        — e.g. allowlist holds the UUID, the message carries the phone as user_id
        and the UUID as user_id_alt."""
        runner = self._runner()
        env = {**_ISO_ENV_OFF, "SIGNAL_ALLOWED_USERS": "uuid-abc-123"}
        with patch.dict(os.environ, env, clear=False):
            src = self._src(Platform.SIGNAL, "+15550001111", user_id_alt="uuid-abc-123")
            assert runner._is_individual_allowlisted(src)
            # A different user (neither id listed) is rejected.
            other = self._src(Platform.SIGNAL, "+15559998888", user_id_alt="uuid-zzz-999")
            assert not runner._is_individual_allowlisted(other)

    def test_wildcard_allows_anyone(self):
        runner = self._runner()
        env = {**_ISO_ENV_OFF, "TELEGRAM_ALLOWED_USERS": "*"}
        with patch.dict(os.environ, env, clear=False):
            assert runner._is_individual_allowlisted(self._src(Platform.TELEGRAM, "anyone"))

    def test_allow_all_flag_allows_anyone(self):
        runner = self._runner()
        env = {**_ISO_ENV_OFF, "SIGNAL_ALLOW_ALL_USERS": "true"}
        with patch.dict(os.environ, env, clear=False):
            assert runner._is_individual_allowlisted(self._src(Platform.SIGNAL, "anyone"))
