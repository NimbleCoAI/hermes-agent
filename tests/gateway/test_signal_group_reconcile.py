"""Tests for Signal adapter group reconciliation (Gap G1: added-at-creation).

The invite handler only auto-approves a group when an invite *envelope*
(groupV2 update with no dataMessage) arrives from an approved user. When an
admin *creates* a group with the agent already in it, no invite envelope is
ever delivered, so that handler never fires and the group stays unapproved —
its messages are silently dropped.

`_reconcile_groups()` closes that gap, envelope-agnostically: on connect it
lists the agent's groups and auto-approves any the agent is already a member
of when an approved user is a group admin. It mirrors the invite handler's
trust model (open-mode / wildcard / approved-admin) and reuses the same
persistence so reconciled groups survive restarts.
"""
import json
import pytest

from gateway.config import PlatformConfig


def _make_signal_adapter(monkeypatch, tmp_path, account="+15551234567", **extra):
    """Create a SignalAdapter with a tmp HERMES_HOME so tests never touch the real one."""
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "gateway.platforms.signal.get_hermes_home", lambda: tmp_path, raising=False
    )

    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    if "allowed_users" in extra:
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", extra.pop("allowed_users"))
    if "group_invite_policy" in extra:
        monkeypatch.setenv("SIGNAL_GROUP_INVITE_POLICY", extra.pop("group_invite_policy"))

    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        **extra,
    }
    return SignalAdapter(config)


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


def _group(gid, *, admin_number=None, admin_uuid=None, is_member=True,
           is_blocked=False, extra_members=None):
    """Build a listGroups-shaped group dict."""
    admins = []
    members = list(extra_members or [])
    if admin_number or admin_uuid:
        admin = {"number": admin_number, "uuid": admin_uuid}
        admins.append(admin)
        members.append({"number": admin_number, "uuid": admin_uuid, "isAdmin": True})
    return {
        "id": gid,
        "name": f"group {gid}",
        "isMember": is_member,
        "isBlocked": is_blocked,
        "members": members,
        "admins": admins,
    }


def _mock_list_groups(adapter, groups):
    """Patch adapter._rpc so listGroups returns `groups`; record calls."""
    calls = []

    async def mock_rpc(method, params=None, rpc_id=None, **kwargs):
        calls.append((method, params))
        if method == "listGroups":
            return groups
        return {"success": True}

    adapter._rpc = mock_rpc
    return calls


# ---------------------------------------------------------------------------
# Approve when an approved user is a group admin (the real use case)
# ---------------------------------------------------------------------------

class TestReconcileApprovesTrustedGroup:

    @pytest.mark.asyncio
    async def test_approves_group_with_approved_admin_by_number(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        _mock_list_groups(adapter, [_group("grp-1", admin_number="+15559999999")])

        await adapter._reconcile_groups()

        assert "grp-1" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_approves_group_with_approved_admin_by_uuid(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        # UUID-only allowlist entry, resolved as DMs are at boot.
        adapter.dm_allow_from_uuids.add("66666666-6666-6666-6666-666666666666")
        _mock_list_groups(
            adapter,
            [_group("grp-2", admin_uuid="66666666-6666-6666-6666-666666666666")],
        )

        await adapter._reconcile_groups()

        assert "grp-2" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_persists_reconciled_group(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        _mock_list_groups(adapter, [_group("grp-3", admin_number="+15559999999")])

        await adapter._reconcile_groups()

        from gateway.platforms.signal import _APPROVED_GROUPS_FILE
        data = json.loads((tmp_path / _APPROVED_GROUPS_FILE).read_text())
        assert "grp-3" in data
        assert data["grp-3"]["added_by_uuid"] or data["grp-3"]["added_by"]


# ---------------------------------------------------------------------------
# Do NOT approve untrusted / ineligible groups
# ---------------------------------------------------------------------------

class TestReconcileSkipsUntrusted:

    @pytest.mark.asyncio
    async def test_skips_group_without_approved_admin(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        _mock_list_groups(adapter, [_group("grp-x", admin_number="+15550000000")])

        await adapter._reconcile_groups()

        assert "grp-x" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_skips_non_member_group(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        _mock_list_groups(
            adapter,
            [_group("grp-nm", admin_number="+15559999999", is_member=False)],
        )

        await adapter._reconcile_groups()

        assert "grp-nm" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_skips_blocked_group(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        _mock_list_groups(
            adapter,
            [_group("grp-bl", admin_number="+15559999999", is_blocked=True)],
        )

        await adapter._reconcile_groups()

        assert "grp-bl" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_already_approved_group_not_repersisted(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999", group_allowed="grp-known"
        )
        _mock_list_groups(adapter, [_group("grp-known", admin_number="+15559999999")])

        await adapter._reconcile_groups()

        from gateway.platforms.signal import _APPROVED_GROUPS_FILE
        # Idempotent: an already-approved group should not create a persistence entry.
        path = tmp_path / _APPROVED_GROUPS_FILE
        if path.exists():
            assert "grp-known" not in json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Policy parity with the invite handler
# ---------------------------------------------------------------------------

class TestReconcilePolicyParity:

    @pytest.mark.asyncio
    async def test_allow_all_policy_approves_any_member_group(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999",
            group_invite_policy="allow-all",
        )
        # No approved admin in the group, but policy is allow-all.
        _mock_list_groups(adapter, [_group("grp-aa", admin_number="+15550000000")])

        await adapter._reconcile_groups()

        assert "grp-aa" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_open_mode_no_allowlist_approves_member_group(self, monkeypatch, tmp_path):
        # No SIGNAL_ALLOWED_USERS configured = open mode (mirrors invite handler).
        adapter = _make_signal_adapter(monkeypatch, tmp_path)
        _mock_list_groups(adapter, [_group("grp-open", admin_number="+15550000000")])

        await adapter._reconcile_groups()

        assert "grp-open" in adapter.group_allow_from


# ---------------------------------------------------------------------------
# Robustness: a malformed listGroups payload must never break startup
# ---------------------------------------------------------------------------

class TestReconcileRobustness:

    @pytest.mark.asyncio
    async def test_malformed_admins_does_not_raise(self, monkeypatch, tmp_path):
        # admins as a list of strings (unexpected shape) must not crash startup.
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        bad = {
            "id": "grp-bad",
            "isMember": True,
            "isBlocked": False,
            "admins": ["+15559999999"],  # strings, not dicts
            "members": [],
        }
        good = _group("grp-good", admin_number="+15559999999")
        _mock_list_groups(adapter, [bad, good])

        await adapter._reconcile_groups()  # must not raise

        # The malformed group is skipped; the well-formed one still approved.
        assert "grp-bad" not in adapter.group_allow_from
        assert "grp-good" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_non_dict_group_entry_does_not_raise(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")
        _mock_list_groups(
            adapter,
            ["i am not a dict", _group("grp-ok", admin_number="+15559999999")],
        )

        await adapter._reconcile_groups()  # must not raise

        assert "grp-ok" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_wildcard_groups_does_not_repersist_every_group(self, monkeypatch, tmp_path):
        # SIGNAL_GROUP_ALLOWED_USERS="*" already allows all groups — reconciliation
        # must not individually approve+persist each member group on every connect.
        adapter = _make_signal_adapter(
            monkeypatch, tmp_path, allowed_users="+15559999999", group_allowed="*"
        )
        _mock_list_groups(
            adapter,
            [_group("grp-a", admin_number="+15559999999"),
             _group("grp-b", admin_number="+15559999999")],
        )

        await adapter._reconcile_groups()

        from gateway.platforms.signal import _APPROVED_GROUPS_FILE
        path = tmp_path / _APPROVED_GROUPS_FILE
        assert not path.exists(), "wildcard allow-all should not persist per-group entries"


# ---------------------------------------------------------------------------
# Wiring: connect() runs reconciliation after a successful health check
# ---------------------------------------------------------------------------

class TestReconcileWiredIntoConnect:

    @pytest.mark.asyncio
    async def test_connect_invokes_reconcile_groups(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = _make_signal_adapter(monkeypatch, tmp_path, allowed_users="+15559999999")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client.aclose = AsyncMock()

        # Replace the network-touching collaborators so connect() stays local.
        monkeypatch.setattr(adapter, "_resolve_allowlist_uuids", AsyncMock())
        monkeypatch.setattr(adapter, "_set_profile_name", AsyncMock())
        monkeypatch.setattr(adapter, "_sse_listener", AsyncMock())
        monkeypatch.setattr(adapter, "_health_monitor", AsyncMock())
        reconcile = AsyncMock()
        monkeypatch.setattr(adapter, "_reconcile_groups", reconcile)
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
        monkeypatch.setattr(adapter, "_release_platform_lock", lambda *a, **k: None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok = await adapter.connect()

        await adapter.disconnect()

        assert ok is True
        reconcile.assert_awaited_once()
