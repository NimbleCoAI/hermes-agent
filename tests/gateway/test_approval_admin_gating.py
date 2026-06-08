"""Tests for admin-only approval gating.

Verifies that /approve and /deny commands — and Telegram inline button
callbacks — are blocked for non-admin users when ``approvals.admin_only``
is True (the default).
"""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key
from gateway.slash_access import SlashAccessPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADMIN_POLICY = SlashAccessPolicy(
    enabled=True,
    admin_user_ids=frozenset({"admin1"}),
    user_allowed_commands=frozenset(),
)

_DISABLED_POLICY = SlashAccessPolicy(
    enabled=False,
    admin_user_ids=frozenset(),
    user_allowed_commands=frozenset(),
)


def _make_source(user_id: str = "u1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=user_id,
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, user_id: str = "u1") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(user_id=user_id),
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
    runner._set_session_env = lambda _context: None
    return runner


def _clear_approval_state():
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


# ===========================================================================
# /approve admin gating
# ===========================================================================


class TestApproveAdminGating:
    """Non-admin users should be blocked from /approve when admin_only=True."""

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_non_admin_blocked_when_admin_only_true(self):
        """A non-admin user gets the denial message."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source(user_id="nonadmin")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_ADMIN_POLICY):
                result = await runner._handle_approve_command(
                    _make_event("/approve", user_id="nonadmin")
                )

        assert "Not authorized" in result
        assert not entry.event.is_set()

    @pytest.mark.asyncio
    async def test_admin_allowed_when_admin_only_true(self):
        """An admin user can still approve."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source(user_id="admin1")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_ADMIN_POLICY):
                result = await runner._handle_approve_command(
                    _make_event("/approve", user_id="admin1")
                )

        assert "approved" in result.lower() or "resuming" in result.lower()
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_anyone_allowed_when_admin_only_false(self):
        """When admin_only is False, any authorized user can approve."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source(user_id="nonadmin")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval._get_approval_config", return_value={"admin_only": False}):
            result = await runner._handle_approve_command(
                _make_event("/approve", user_id="nonadmin")
            )

        assert "approved" in result.lower() or "resuming" in result.lower()
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_gating_skipped_when_policy_disabled(self):
        """When slash_access policy is disabled, admin check is skipped."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source(user_id="anyone")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_DISABLED_POLICY):
                result = await runner._handle_approve_command(
                    _make_event("/approve", user_id="anyone")
                )

        assert "approved" in result.lower() or "resuming" in result.lower()
        assert entry.event.is_set()


# ===========================================================================
# /deny admin gating
# ===========================================================================


class TestDenyAdminGating:
    """Non-admin users should be blocked from /deny when admin_only=True."""

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_non_admin_blocked_when_admin_only_true(self):
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source(user_id="nonadmin")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_ADMIN_POLICY):
                result = await runner._handle_deny_command(
                    _make_event("/deny", user_id="nonadmin")
                )

        assert "Not authorized" in result
        assert not entry.event.is_set()

    @pytest.mark.asyncio
    async def test_admin_can_deny(self):
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source(user_id="admin1")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval._get_approval_config", return_value={"admin_only": True}):
            with patch("gateway.slash_access.policy_for_source", return_value=_ADMIN_POLICY):
                result = await runner._handle_deny_command(
                    _make_event("/deny", user_id="admin1")
                )

        assert "denied" in result.lower()
        assert entry.event.is_set()
        assert entry.result == "deny"

    @pytest.mark.asyncio
    async def test_anyone_can_deny_when_admin_only_false(self):
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source(user_id="nonadmin")
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        with patch("tools.approval._get_approval_config", return_value={"admin_only": False}):
            result = await runner._handle_deny_command(
                _make_event("/deny", user_id="nonadmin")
            )

        assert "denied" in result.lower()
        assert entry.event.is_set()


# ===========================================================================
# Telegram button callback admin gating
# ===========================================================================


def _ensure_telegram_mock():
    """Wire up minimal mocks to import TelegramAdapter."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramCallbackAdminGating:
    """Telegram inline button clicks should be blocked for non-admin users."""

    @pytest.mark.asyncio
    async def test_non_admin_blocked_on_button_click(self):
        adapter = _make_adapter()
        adapter._approval_state[10] = "agent:main:telegram:group:12345:99"

        query = AsyncMock()
        query.data = "ea:once:10"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat = MagicMock()
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "NonAdmin"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval._get_approval_config", return_value={"admin_only": True}):
                with patch.object(adapter, "_is_user_admin", return_value=False):
                    await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert "admin" in query.answer.call_args[1]["text"].lower()
        query.edit_message_text.assert_not_called()
        # State should NOT be consumed
        assert 10 in adapter._approval_state

    @pytest.mark.asyncio
    async def test_admin_allowed_on_button_click(self):
        adapter = _make_adapter()
        adapter._approval_state[11] = "agent:main:telegram:group:12345:99"

        query = AsyncMock()
        query.data = "ea:once:11"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat = MagicMock()
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = "111"
        query.from_user.first_name = "Admin"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval._get_approval_config", return_value={"admin_only": True}):
                with patch.object(adapter, "_is_user_admin", return_value=True):
                    with patch("tools.approval.resolve_gateway_approval", return_value=1):
                        await adapter._handle_callback_query(update, context)

        # Should have been resolved — state consumed
        assert 11 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_anyone_allowed_when_admin_only_false(self):
        adapter = _make_adapter()
        adapter._approval_state[12] = "agent:main:telegram:group:12345:99"

        query = AsyncMock()
        query.data = "ea:once:12"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat = MagicMock()
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = "999"
        query.from_user.first_name = "Anyone"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval._get_approval_config", return_value={"admin_only": False}):
                with patch("tools.approval.resolve_gateway_approval", return_value=1):
                    await adapter._handle_callback_query(update, context)

        # Should have been resolved — no admin block
        assert 12 not in adapter._approval_state


# ===========================================================================
# _is_user_admin method
# ===========================================================================


class TestIsUserAdmin:
    """Test the _is_user_admin method on TelegramAdapter."""

    def test_first_allowed_user_is_admin(self):
        adapter = _make_adapter()
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "111,222,333"}, clear=False):
            with patch("plugins.swarm_map_policy.is_platform_admin", side_effect=Exception("not installed"), create=True):
                assert adapter._is_user_admin("111") is True
                assert adapter._is_user_admin("222") is False

    def test_wildcard_not_treated_as_admin(self):
        adapter = _make_adapter()
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*,111"}, clear=False):
            # "*" is skipped, first real ID is admin
            assert adapter._is_user_admin("111") is True
            assert adapter._is_user_admin("222") is False

    def test_empty_allowed_users_returns_false(self):
        adapter = _make_adapter()
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}, clear=False):
            assert adapter._is_user_admin("111") is False

    def test_hsm_policy_checked_first(self):
        adapter = _make_adapter()
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "999"}, clear=False):
            with patch("plugins.swarm_map_policy.is_platform_admin", return_value=True, create=True):
                # HSM says admin, even though not first in allowlist
                assert adapter._is_user_admin("222") is True

    def test_empty_user_id_returns_false(self):
        adapter = _make_adapter()
        assert adapter._is_user_admin("") is False
        assert adapter._is_user_admin(None) is False
