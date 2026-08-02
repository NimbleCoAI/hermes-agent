"""Discord authorization: gateway channel-scoped access (R2) + fail-closed allowlists (R1).

Two independent defects, both found in a Discord control-surface audit:

  R2 — ``_is_user_authorized`` had no Discord entry in its chat-scoped allowlist
  map, while the Discord adapter DOES grant channel-scoped access at intake. So
  the adapter admitted a guild message and the gateway then denied it: the bot
  went silent with no operator-visible error, and the obvious operator "fix" is
  ``DISCORD_ALLOWED_USERS=*`` — the worst available state.

  The fix is deliberately OPT-IN via ``DISCORD_CHANNEL_SCOPED_ACCESS``. Simply
  adding Discord to the shared map would widen every existing Discord agent:
  those agents use ``DISCORD_ALLOWED_CHANNELS`` to scope *where the bot speaks*
  while restricting *who may command it* to a short ``DISCORD_ALLOWED_USERS``
  list, and a chat-scoped grant authorizes every sender in the channel.

  R1 — an allowlist var that was SET but parsed to zero usable entries (e.g.
  ``DISCORD_ALLOWED_ROLES=moderators``, a role name where a numeric snowflake is
  required) read downstream as "no allowlist configured". That branch grants
  channel-scoped access to everyone and honors the allow-all flags, so a typo
  WIDENED access instead of denying.
"""

import os
from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_discord_env(monkeypatch):
    """Start every test from a clean Discord env (mirrors test_discord_bot_auth_bypass)."""
    for var in (
        "DISCORD_ALLOW_BOTS",
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_ALLOW_ALL_USERS",
        "DISCORD_CHANNEL_SCOPED_ACCESS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    """GatewayRunner skeleton — object.__new__ skips the heavy __init__ (pitfall #17)."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _discord_channel_msg(user_id="999", chat_id="555"):
    """A human posting in a Discord guild text channel (chat_type 'channel')."""
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="channel",
        user_id=user_id,
        user_name="SomeHuman",
        is_bot=False,
    )


# ---------------------------------------------------------------------------
# R2 — gateway channel-scoped access, opt-in
# ---------------------------------------------------------------------------


def test_channel_scoped_access_is_off_by_default(monkeypatch):
    """Without the opt-in flag, an allowlisted channel does NOT authorize a stranger.

    This is the no-silent-widening guard: every existing Discord agent has
    DISCORD_ALLOWED_CHANNELS set, so if this ever returns True the fix has
    promoted 'the bot may post here' into 'anyone here may drive the bot'.
    """
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_discord_channel_msg()) is False


def test_channel_scoped_access_authorizes_when_opted_in(monkeypatch):
    """With the flag and an explicit matching channel, the sender is authorized."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111,555,222")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_discord_channel_msg(chat_id="555")) is True


def test_channel_scoped_access_ignores_wildcard(monkeypatch):
    """``*`` is a scope statement, not an authorization grant.

    cyborg runs DISCORD_ALLOWED_CHANNELS=* today. Honoring the wildcard here
    (as the Telegram/Signal branches do) would make it a fully open bot.
    """
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "*")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_discord_channel_msg()) is False


def test_channel_scoped_access_wildcard_mixed_with_real_ids(monkeypatch):
    """A stray ``*`` among real ids is dropped, and the real ids still work."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "*,555")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_discord_channel_msg(chat_id="555")) is True
    assert runner._is_user_authorized(_discord_channel_msg(chat_id="777")) is False


def test_channel_scoped_access_denies_unlisted_channel(monkeypatch):
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111,222")
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_discord_channel_msg(chat_id="555")) is False


def _discord_thread_msg(user_id="999", chat_id="888", parent_chat_id="555"):
    """A human posting in a thread under a guild channel (chat_type 'thread').

    With DISCORD_AUTO_THREAD on (the default) most conversation happens in
    threads, so the channel-scoped grant must cover them — this was the exact
    shape of the original incident: an approved parent channel, a stranger in
    one of its threads, silence.
    """
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="thread",
        user_id=user_id,
        user_name="SomeHuman",
        is_bot=False,
        parent_chat_id=parent_chat_id,
    )


def test_channel_scoped_access_covers_thread_under_allowed_parent(monkeypatch):
    """A thread authorizes iff its parent channel would."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111,555,222")
    runner = _make_bare_runner()
    assert (
        runner._is_user_authorized(_discord_thread_msg(parent_chat_id="555")) is True
    )


def test_channel_scoped_access_denies_thread_under_unlisted_parent(monkeypatch):
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111,222")
    runner = _make_bare_runner()
    assert (
        runner._is_user_authorized(_discord_thread_msg(parent_chat_id="555")) is False
    )


def test_channel_scoped_access_denies_thread_without_parent_id(monkeypatch):
    """A thread source missing parent_chat_id must not match on its own id."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111,222")
    runner = _make_bare_runner()
    assert (
        runner._is_user_authorized(_discord_thread_msg(parent_chat_id=None)) is False
    )


def test_channel_scoped_access_thread_off_without_opt_in(monkeypatch):
    """Threads get no wider default than channels: flag off → deny."""
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    runner = _make_bare_runner()
    assert (
        runner._is_user_authorized(_discord_thread_msg(parent_chat_id="555")) is False
    )


def test_channel_scoped_access_thread_directly_allowlisted(monkeypatch):
    """A thread id listed directly in DISCORD_ALLOWED_CHANNELS also matches —
    HSM's group-register writes thread ids into the var for thread-scoped
    agents, and those must keep working."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "888")
    runner = _make_bare_runner()
    assert (
        runner._is_user_authorized(
            _discord_thread_msg(chat_id="888", parent_chat_id="555")
        )
        is True
    )


def test_channel_scoped_access_does_not_apply_to_dms(monkeypatch):
    """DMs have no channel scope to stand on — the flag must not authorize them."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    runner = _make_bare_runner()
    dm = SessionSource(
        platform=Platform.DISCORD,
        chat_id="555",
        chat_type="dm",
        user_id="999",
        user_name="SomeHuman",
        is_bot=False,
    )
    assert runner._is_user_authorized(dm) is False


def test_other_platforms_still_honor_their_wildcard(monkeypatch):
    """Regression guard: the Discord branch must not disturb Telegram semantics."""
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "*")
    runner = _make_bare_runner()
    tg = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="group",
        user_id="999",
        user_name="SomeHuman",
        is_bot=False,
    )
    assert runner._is_user_authorized(tg) is True


# ---------------------------------------------------------------------------
# R1 — adapter fail-closed on a configured-but-unresolvable allowlist
# ---------------------------------------------------------------------------


def _make_bare_adapter(**attrs):
    """DiscordAdapter skeleton for _is_allowed_user (object.__new__, pitfall #17)."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    # ``name`` is a read-only property deriving from ``_platform``
    # (gateway/platforms/base.py:2764), so set the backing attribute.
    adapter._platform = Platform.DISCORD
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._pairing_store = None
    for k, v in attrs.items():
        setattr(adapter, k, v)
    return adapter


def _no_pairing(adapter, monkeypatch):
    monkeypatch.setattr(
        type(adapter), "_is_pairing_approved_user", lambda self, uid: False, raising=False
    )


def test_role_allowlist_set_but_unparseable_fails_closed(monkeypatch):
    """DISCORD_ALLOWED_ROLES=moderators must deny, not fall through to channel access.

    Role names are not snowflakes, so the parse yields nothing. Before the fix
    that looked identical to 'no allowlist configured', which grants
    channel-scoped access to every sender.
    """
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    adapter = _make_bare_adapter(
        _role_allowlist_configured=True,   # var was set...
        _allowed_role_ids=set(),           # ...but parsed to nothing
        _user_allowlist_configured=False,
    )
    _no_pairing(adapter, monkeypatch)
    assert (
        adapter._is_allowed_user("999", guild=object(), channel_ids={"555"})
        is False
    )


def test_user_allowlist_set_but_unparseable_fails_closed(monkeypatch):
    """DISCORD_ALLOWED_USERS=" , " parses to nothing and must deny."""
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    adapter = _make_bare_adapter(
        _user_allowlist_configured=True,
        _allowed_user_ids=set(),
        _role_allowlist_configured=False,
    )
    _no_pairing(adapter, monkeypatch)
    assert (
        adapter._is_allowed_user("999", guild=object(), channel_ids={"555"})
        is False
    )


def test_unparseable_allowlist_does_not_honor_allow_all_flags(monkeypatch):
    """A broken allowlist must not be rescued by GATEWAY_ALLOW_ALL_USERS either."""
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _make_bare_adapter(
        _role_allowlist_configured=True,
        _allowed_role_ids=set(),
        _user_allowlist_configured=False,
    )
    _no_pairing(adapter, monkeypatch)
    assert adapter._is_allowed_user("999", guild=object()) is False


def test_no_allowlist_at_all_still_grants_channel_scoped_access(monkeypatch):
    """Unchanged behavior: genuinely-unset allowlists keep the channel bypass."""
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    adapter = _make_bare_adapter(
        _user_allowlist_configured=False,
        _role_allowlist_configured=False,
    )
    _no_pairing(adapter, monkeypatch)
    monkeypatch.setattr(
        type(adapter),
        "_discord_channel_ids_allowed",
        lambda self, ids: True,
        raising=False,
    )
    assert (
        adapter._is_allowed_user("999", guild=object(), channel_ids={"555"}) is True
    )


def test_valid_user_allowlist_unaffected(monkeypatch):
    """No regression: a real user allowlist still authorizes its members."""
    adapter = _make_bare_adapter(
        _user_allowlist_configured=True,
        _allowed_user_ids={"999"},
    )
    _no_pairing(adapter, monkeypatch)
    assert adapter._is_allowed_user("999", guild=object()) is True
    assert adapter._is_allowed_user("111", guild=object()) is False


def test_allowlist_assigned_after_init_is_still_configured(monkeypatch):
    """A populated set must count as configured even when the flag says False.

    Regression guard: __init__ sets the flags False, and several callers (plus
    tests/gateway/test_discord_slash_auth.py's fixture) then assign
    ``_allowed_user_ids`` directly. Deriving `configured` from the flag ALONE
    left it stale-False, sent control into the "no allowlist" widening branch,
    and denied a legitimately allowlisted user. The flag must be additive to the
    parsed truthiness, never a replacement for it.
    """
    adapter = _make_bare_adapter(
        _user_allowlist_configured=False,  # as __init__ leaves it
        _role_allowlist_configured=False,
    )
    adapter._allowed_user_ids = {"999"}  # assigned afterwards, flag not updated
    _no_pairing(adapter, monkeypatch)
    assert adapter._is_allowed_user("999", guild=object()) is True
    assert adapter._is_allowed_user("111", guild=object()) is False


def test_missing_configured_attrs_fall_back_to_parsed_truthiness(monkeypatch):
    """Fixtures that skip __init__ entirely keep their old behavior.

    Without the getattr fallback, an adapter built via object.__new__ with a
    populated _allowed_user_ids would read as 'not configured' and take the
    widening branch — a fail-open introduced by the fix itself.
    """
    adapter = _make_bare_adapter(_allowed_user_ids={"999"})
    # deliberately no _user_allowlist_configured / _role_allowlist_configured
    assert not hasattr(adapter, "_user_allowlist_configured")
    _no_pairing(adapter, monkeypatch)
    assert adapter._is_allowed_user("999", guild=object()) is True
    assert adapter._is_allowed_user("111", guild=object()) is False


# ---------------------------------------------------------------------------
# R3 — union semantics: the flag admits channel members even when a user
# allowlist is configured (adapter side; without this the flag silently does
# nothing on any agent that also lists users, which is every deployed agent)
# ---------------------------------------------------------------------------


def _union_adapter(monkeypatch):
    adapter = _make_bare_adapter(
        _user_allowlist_configured=True,
        _allowed_user_ids={"111"},
        _role_allowlist_configured=False,
    )
    _no_pairing(adapter, monkeypatch)
    return adapter


def test_adapter_union_admits_stranger_in_allowed_channel(monkeypatch):
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    adapter = _union_adapter(monkeypatch)
    assert adapter._is_allowed_user("999", guild=object(), channel_ids={"555"}) is True


def test_adapter_union_keeps_allowlisted_user_everywhere(monkeypatch):
    """The user allowlist is not narrowed by the flag — union, not replacement."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    adapter = _union_adapter(monkeypatch)
    assert adapter._is_allowed_user("111", guild=object(), channel_ids={"777"}) is True


def test_adapter_union_thread_channel_ids_carry_parent(monkeypatch):
    """on_message passes {thread id, parent id}; the parent match must admit."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    adapter = _union_adapter(monkeypatch)
    assert (
        adapter._is_allowed_user("999", guild=object(), channel_ids={"888", "555"})
        is True
    )


def test_adapter_union_requires_the_flag(monkeypatch):
    """Flag off → configured user allowlist stays a strict gate (no widening)."""
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    adapter = _union_adapter(monkeypatch)
    assert adapter._is_allowed_user("999", guild=object(), channel_ids={"555"}) is False


def test_adapter_union_denies_unlisted_channel(monkeypatch):
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111,222")
    adapter = _union_adapter(monkeypatch)
    assert adapter._is_allowed_user("999", guild=object(), channel_ids={"555"}) is False


def test_adapter_union_ignores_wildcard(monkeypatch):
    """DISCORD_ALLOWED_CHANNELS=* is a scope statement, not a grant (cyborg)."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "*")
    adapter = _union_adapter(monkeypatch)
    assert adapter._is_allowed_user("999", guild=object(), channel_ids={"555"}) is False


def test_adapter_union_gives_no_dm_grant(monkeypatch):
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    adapter = _union_adapter(monkeypatch)
    assert (
        adapter._is_allowed_user("999", guild=None, is_dm=True, channel_ids=None)
        is False
    )


def test_end_to_end_stranger_passes_both_gates_with_user_allowlist_present(
    monkeypatch,
):
    """The incident shape, fixed end to end: a user allowlist exists, the flag is
    on, and a stranger in an approved channel passes BOTH the adapter gate and
    the gateway gate (previously the adapter dropped them and the flag was
    unreachable)."""
    monkeypatch.setenv("DISCORD_CHANNEL_SCOPED_ACCESS", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "555")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
    adapter = _union_adapter(monkeypatch)
    assert adapter._is_allowed_user("999", guild=object(), channel_ids={"555"}) is True
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_discord_channel_msg(user_id="999")) is True
    assert (
        runner._is_user_authorized(_discord_thread_msg(user_id="999", parent_chat_id="555"))
        is True
    )


# ---------------------------------------------------------------------------
# R4 — reconnect must not fail open after the on_ready username rewrite
# ---------------------------------------------------------------------------


def test_resolver_zero_resolution_does_not_clear_env(monkeypatch):
    """If NO username resolves, DISCORD_ALLOWED_USERS must not be rewritten to ''.

    An emptied var reads as 'no allowlist configured' on the next
    connect(is_reconnect=True) parse, clearing _user_allowlist_configured and
    widening access — the reconnect fail-open found in review.
    """
    import asyncio
    from types import SimpleNamespace as NS

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "ghostname")
    adapter = _make_bare_adapter(
        _allowed_user_ids={"ghostname"},
        _client=NS(guilds=[]),
        platform=Platform.DISCORD,  # self.name derives from this in the print path
    )
    asyncio.run(adapter._resolve_allowed_usernames())
    assert os.environ["DISCORD_ALLOWED_USERS"] == "ghostname"
    assert adapter._allowed_user_ids == {"ghostname"}


def test_resolver_partial_resolution_still_rewrites(monkeypatch):
    """Numeric entries (and resolved ids) keep flowing into the env var."""
    import asyncio
    from types import SimpleNamespace as NS

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "12345,ghostname")
    adapter = _make_bare_adapter(
        _allowed_user_ids={"12345", "ghostname"},
        _client=NS(guilds=[]),
        platform=Platform.DISCORD,  # self.name derives from this in the print path
    )
    asyncio.run(adapter._resolve_allowed_usernames())
    assert os.environ["DISCORD_ALLOWED_USERS"] == "12345"
    assert adapter._allowed_user_ids == {"12345"}
