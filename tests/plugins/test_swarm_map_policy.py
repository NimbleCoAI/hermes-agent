"""Tests for swarm-map-policy plugin — HSM integration for group access control."""
import pytest
from unittest.mock import patch, MagicMock
from enum import Enum


def _make_event(platform="signal", chat_id="group-123", user_id="user-456"):
    """Create a mock MessageEvent with source attributes."""
    class MockPlatform(Enum):
        SIGNAL = "signal"
        TELEGRAM = "telegram"
        DISCORD = "discord"

    event = MagicMock()
    event.source.chat_id = chat_id
    event.source.user_id = user_id
    # Set platform as an enum with .value
    platform_enum = MockPlatform(platform) if platform in ("signal", "telegram", "discord") else MagicMock(value=platform)
    event.source.platform = platform_enum
    return event


class TestSwarmMapPolicy:

    def test_plugin_registers_hooks(self):
        """Plugin registers on_session_start, pre_tool_call, and pre_gateway_dispatch hooks."""
        from plugins.swarm_map_policy import register
        ctx = MagicMock()
        register(ctx)
        hook_names = [call.args[0] for call in ctx.register_hook.call_args_list]
        assert "on_session_start" in hook_names
        assert "pre_tool_call" in hook_names
        assert "pre_gateway_dispatch" in hook_names

    def test_hsm_url_from_env(self):
        from plugins.swarm_map_policy import _hsm_url
        with patch.dict("os.environ", {"HSM_URL": "http://localhost:3002"}):
            assert _hsm_url() == "http://localhost:3002"

    def test_hsm_url_missing_returns_none(self):
        from plugins.swarm_map_policy import _hsm_url
        with patch.dict("os.environ", {}, clear=True):
            assert _hsm_url() is None

    def test_group_check_fail_closed_on_error(self):
        from plugins.swarm_map_policy import is_group_allowed
        with patch("plugins.swarm_map_policy._hsm_url", return_value="http://dead:9999"):
            with patch("plugins.swarm_map_policy.requests") as mock_req:
                mock_req.get.side_effect = Exception("Connection refused")
                assert is_group_allowed("group-123", "signal") is False

    def test_group_check_fail_closed_no_config(self):
        from plugins.swarm_map_policy import is_group_allowed
        with patch("plugins.swarm_map_policy._hsm_url", return_value=None):
            assert is_group_allowed("group-123", "signal") is False

    def test_tool_check_fail_open(self):
        from plugins.swarm_map_policy import is_tool_allowed
        with patch("plugins.swarm_map_policy._hsm_url", return_value=None):
            assert is_tool_allowed("dangerous_tool", "group-123") is True

    def test_admin_check_fail_closed(self):
        from plugins.swarm_map_policy import is_platform_admin
        with patch("plugins.swarm_map_policy._hsm_url", return_value=None):
            assert is_platform_admin("user-123", "signal") is False


class TestApproveGroupAdd:
    """Tests for approve_group_add — HSM group auto-approval on bot add."""

    def _call(self, mock_resp=None, side_effect=None, group_id="-100123", adder="777"):
        from plugins.swarm_map_policy import approve_group_add
        with patch("plugins.swarm_map_policy._hsm_url", return_value="http://hsm:3002"), \
             patch("plugins.swarm_map_policy._harness_id", return_value="hermes-test"), \
             patch("plugins.swarm_map_policy.requests") as mock_req:
            if side_effect is not None:
                mock_req.post.side_effect = side_effect
            else:
                mock_req.post.return_value = mock_resp
            result = approve_group_add(group_id, adder)
            return result, mock_req

    @staticmethod
    def _resp(status_code=200, body=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = body if body is not None else {}
        return resp

    def test_approved(self):
        """200 + approved:true returns True."""
        result, _ = self._call(self._resp(200, {"approved": True, "restarted": True}))
        assert result is True

    def test_approved_already_allowed(self):
        """200 + approved:true + already_allowed (wildcard/listed) returns True."""
        result, _ = self._call(self._resp(200, {"approved": True, "already_allowed": True}))
        assert result is True

    def test_approved_restart_failed_still_true(self):
        """approved:true with restarted:false (env written, recreate failed) is still approved."""
        result, _ = self._call(self._resp(200, {"approved": True, "restarted": False}))
        assert result is True

    def test_not_approved(self):
        """200 + approved:false returns False."""
        result, _ = self._call(
            self._resp(200, {"approved": False, "reason": "adder is not an admin"})
        )
        assert result is False

    def test_bad_request_denied(self):
        """400 with error body is treated as not approved."""
        result, _ = self._call(self._resp(400, {"error": "missing addedByUserId"}))
        assert result is False

    def test_network_error_fail_closed(self):
        """Network failure denies (fail-closed)."""
        result, _ = self._call(side_effect=Exception("Connection refused"))
        assert result is False

    def test_missing_approved_field_denied(self):
        """200 with no approved field denies (fail-closed)."""
        result, _ = self._call(self._resp(200, {}))
        assert result is False

    def test_non_bool_approved_denied(self):
        """approved must be strictly true — truthy strings deny."""
        result, _ = self._call(self._resp(200, {"approved": "yes"}))
        assert result is False

    def test_no_config_fail_closed(self):
        """Missing HSM_URL denies without any request."""
        from plugins.swarm_map_policy import approve_group_add
        with patch("plugins.swarm_map_policy._hsm_url", return_value=None):
            assert approve_group_add("-100123", "777") is False

    def test_posts_correct_url_and_body(self):
        """Request hits the HSM group endpoint with addedByUserId body."""
        _, mock_req = self._call(self._resp(200, {"approved": True}))
        mock_req.post.assert_called_once_with(
            "http://hsm:3002/api/harnesses/hermes-test/surfaces/telegram/groups/-100123",
            json={"addedByUserId": "777"},
            timeout=5,
        )


class TestSessionContextCaching:
    """Tests for session context caching via pre_gateway_dispatch."""

    def setup_method(self):
        """Clear session context before each test."""
        from plugins.swarm_map_policy import clear_session_context
        clear_session_context()

    def test_pre_gateway_dispatch_caches_context(self):
        """pre_gateway_dispatch extracts and caches platform/chat_id/user_id."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = _make_event(platform="signal", chat_id="group-123", user_id="user-456")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False):
            result = _pre_gateway_dispatch(event=event)
        assert result is None  # Should allow normal dispatch
        ctx = get_session_context()
        assert ctx is not None
        assert ctx["platform"] == "signal"
        assert ctx["chat_id"] == "group-123"
        assert ctx["user_id"] == "user-456"

    def test_pre_gateway_dispatch_returns_none_on_no_event(self):
        """pre_gateway_dispatch returns None when no event provided."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        result = _pre_gateway_dispatch(event=None)
        assert result is None
        assert get_session_context() is None

    def test_get_session_context_none_before_dispatch(self):
        """get_session_context returns None before any dispatch."""
        from plugins.swarm_map_policy import get_session_context
        assert get_session_context() is None

    def test_clear_session_context_resets(self):
        """clear_session_context removes cached data."""
        from plugins.swarm_map_policy import (
            _pre_gateway_dispatch, get_session_context, clear_session_context
        )
        event = _make_event()
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False):
            _pre_gateway_dispatch(event=event)
        assert get_session_context() is not None
        clear_session_context()
        assert get_session_context() is None

    def test_pre_gateway_dispatch_handles_none_source_fields(self):
        """Gracefully handles None platform/chat_id/user_id."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = MagicMock()
        event.source.platform = None
        event.source.chat_id = None
        event.source.user_id = None
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False):
            result = _pre_gateway_dispatch(event=event)
        assert result is None
        ctx = get_session_context()
        assert ctx["platform"] == ""
        assert ctx["chat_id"] == ""
        assert ctx["user_id"] == ""


class TestAdminResolution:
    """Tests for admin identity resolution during pre_gateway_dispatch."""

    def setup_method(self):
        from plugins.swarm_map_policy import clear_session_context
        clear_session_context()

    def test_admin_resolved_on_dispatch(self):
        """Admin status is resolved and cached during pre_gateway_dispatch."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = _make_event(platform="signal", user_id="admin-user")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=True):
            _pre_gateway_dispatch(event=event)
        ctx = get_session_context()
        assert ctx["is_admin"] is True

    def test_non_admin_resolved_on_dispatch(self):
        """Non-admin status is correctly cached."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = _make_event(platform="signal", user_id="regular-user")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False):
            _pre_gateway_dispatch(event=event)
        ctx = get_session_context()
        assert ctx["is_admin"] is False

    def test_admin_resolution_fail_closed(self):
        """Admin resolution defaults to False on HSM failure."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch, get_session_context
        event = _make_event(platform="signal", user_id="user-123")
        with patch("plugins.swarm_map_policy.is_platform_admin", side_effect=Exception("HSM down")):
            _pre_gateway_dispatch(event=event)
        ctx = get_session_context()
        assert ctx["is_admin"] is False

    def test_admin_resolution_called_with_correct_args(self):
        """is_platform_admin called with user_id and platform from event."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch
        event = _make_event(platform="telegram", user_id="tg-user-789")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=False) as mock_admin:
            _pre_gateway_dispatch(event=event)
        mock_admin.assert_called_once_with("tg-user-789", "telegram")


class TestApprovalGating:
    """Tests for approval command admin gating via pre_tool_call."""

    def setup_method(self):
        from plugins.swarm_map_policy import clear_session_context
        clear_session_context()

    def _set_admin_context(self, is_admin=True):
        """Set up session context with admin status."""
        from plugins.swarm_map_policy import _pre_gateway_dispatch
        event = _make_event(platform="signal", user_id="user-1")
        with patch("plugins.swarm_map_policy.is_platform_admin", return_value=is_admin):
            _pre_gateway_dispatch(event=event)

    def test_admin_can_use_approval_tool(self):
        """Admin users can execute approval tools."""
        from plugins.swarm_map_policy import _pre_tool_call
        self._set_admin_context(is_admin=True)
        result = _pre_tool_call(tool_name="approval")
        assert result is None  # Allowed

    def test_non_admin_blocked_from_approval(self):
        """Non-admin users are blocked from approval tools."""
        from plugins.swarm_map_policy import _pre_tool_call
        self._set_admin_context(is_admin=False)
        result = _pre_tool_call(tool_name="approval")
        assert result is not None
        assert result["action"] == "block"
        assert "admin" in result["message"].lower() or "Admin" in result["message"]

    def test_non_admin_blocked_from_pr_approval(self):
        """Non-admin users are blocked from pr_approval tool."""
        from plugins.swarm_map_policy import _pre_tool_call
        self._set_admin_context(is_admin=False)
        result = _pre_tool_call(tool_name="pr_approval")
        assert result is not None
        assert result["action"] == "block"

    def test_no_context_blocks_approval(self):
        """Missing session context blocks approval tools (fail-closed)."""
        from plugins.swarm_map_policy import _pre_tool_call
        # No dispatch happened, so no context
        result = _pre_tool_call(tool_name="approval")
        assert result is not None
        assert result["action"] == "block"

    def test_non_gated_tool_allowed_without_admin(self):
        """Non-gated tools are allowed regardless of admin status."""
        from plugins.swarm_map_policy import _pre_tool_call
        self._set_admin_context(is_admin=False)
        result = _pre_tool_call(tool_name="web_search")
        assert result is None  # Allowed

    def test_non_gated_tool_allowed_without_context(self):
        """Non-gated tools are allowed even without session context."""
        from plugins.swarm_map_policy import _pre_tool_call
        result = _pre_tool_call(tool_name="web_search")
        assert result is None  # Allowed
