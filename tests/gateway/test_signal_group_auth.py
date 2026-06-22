"""Signal group-sender authorization (SIGNAL_GROUP_ALLOWED_USERS).

Regression guard: Signal previously authorized senders ONLY via
SIGNAL_ALLOWED_USERS in gateway/run.py, so a bot whose SIGNAL_ALLOWED_USERS
listed just the admin would answer *only* the admin in group chats — every
other member was logged "Unauthorized user" and dropped, even when they
@mentioned the bot.

The Signal adapter already gates which *groups* are active
(SIGNAL_GROUP_ALLOWED_USERS, "*" = all) and marks unmentioned messages
observe_only. So group membership — not per-sender allowlisting — should
authorize a group message at the gateway. DMs stay gated by
SIGNAL_ALLOWED_USERS.
"""

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_signal_env(monkeypatch):
    for var in (
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _group_source(user_id="+15550001111", gid="abc123groupid"):
    return SessionSource(
        platform=Platform.SIGNAL,
        chat_id=f"group:{gid}",
        chat_type="group",
        user_id=user_id,
        user_name="Charlotte",
        chat_id_alt=gid,
    )


def _dm_source(user_id="+15550001111"):
    return SessionSource(
        platform=Platform.SIGNAL,
        chat_id=user_id,
        chat_type="dm",
        user_id=user_id,
        user_name="Charlotte",
    )


ADMIN = "+15550112233"


def test_non_admin_group_member_authorized_when_groups_open(monkeypatch):
    """SIGNAL_GROUP_ALLOWED_USERS=* authorizes any sender in a group, even
    when SIGNAL_ALLOWED_USERS only lists the admin."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_group_source(user_id="+15550009999")) is True


def test_group_member_authorized_by_explicit_group_id(monkeypatch):
    """An explicit group id in SIGNAL_GROUP_ALLOWED_USERS matches the
    'group:<id>' chat_id / chat_id_alt forms."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc123groupid,otherid")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_group_source(user_id="+15550009999")) is True


def test_group_member_denied_when_group_not_listed(monkeypatch):
    """A group id not in the allowlist is not authorized (no wildcard)."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "someothergroup")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_group_source(user_id="+15550009999")) is False


def test_dm_from_non_admin_still_denied(monkeypatch):
    """Opening groups must NOT open DMs — SIGNAL_ALLOWED_USERS still gates DMs."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_dm_source(user_id="+15550009999")) is False


def test_dm_from_admin_authorized(monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_dm_source(user_id=ADMIN)) is True


def test_group_sender_denied_when_no_group_allowlist(monkeypatch):
    """With no SIGNAL_GROUP_ALLOWED_USERS and a DM allowlist set, a group
    sender not on the DM allowlist is denied (defense in depth; the adapter
    also drops these upstream)."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_group_source(user_id="+15550009999")) is False


# ---------------------------------------------------------------------------
# Adapter runtime-approved groups (invite-accept / reconcile / persisted) must
# authorize their senders even when the group is NOT in the env var. run.py's
# group-sender auth previously read only SIGNAL_GROUP_ALLOWED_USERS, so groups
# the adapter approved at runtime had their senders rejected as "Unauthorized".
# ---------------------------------------------------------------------------

def test_group_member_authorized_by_adapter_runtime_approval(monkeypatch):
    """Group present in adapter.group_allow_from but NOT in the env var → the
    sender is authorized (run.py honors the adapter's own approval decision)."""
    # A DM allowlist is set, so the "no allowlist → honor adapter" fallback does
    # NOT apply; and the group is deliberately absent from the env group list.
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "someothergroup")
    runner = _make_bare_runner()
    runner.adapters = {Platform.SIGNAL: SimpleNamespace(group_allow_from={"runtimegid"})}
    src = _group_source(user_id="+15550009999", gid="runtimegid")
    assert runner._is_user_authorized(src) is True


def test_group_member_authorized_by_adapter_wildcard(monkeypatch):
    """Adapter group_allow_from == {'*'} authorizes any group sender."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "someothergroup")
    runner = _make_bare_runner()
    runner.adapters = {Platform.SIGNAL: SimpleNamespace(group_allow_from={"*"})}
    src = _group_source(user_id="+15550009999", gid="anygid")
    assert runner._is_user_authorized(src) is True


def test_group_member_denied_when_not_in_env_or_adapter(monkeypatch):
    """A group in neither the env var nor the adapter's approved set still denies."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "someothergroup")
    runner = _make_bare_runner()
    runner.adapters = {Platform.SIGNAL: SimpleNamespace(group_allow_from={"differentgid"})}
    src = _group_source(user_id="+15550009999", gid="runtimegid")
    assert runner._is_user_authorized(src) is False


def test_adapter_runtime_approval_no_adapters_attr_does_not_crash(monkeypatch):
    """A bare runner without `.adapters` must not raise (defensive getattr)."""
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", ADMIN)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "someothergroup")
    runner = _make_bare_runner()  # no .adapters set
    src = _group_source(user_id="+15550009999", gid="runtimegid")
    assert runner._is_user_authorized(src) is False
