"""B4 audit parity — Letta-brain turns must persist like native turns.

A Letta-brained "door" (Swarm Map v1 B1) keeps all Hermes surfaces, approval,
budget, and audit. That contract is broken if the turn's transcript rows never
reach state.db: the native loop persists its own messages via
``_flush_messages_to_session_db()`` and the gateway therefore appends with
``skip_db=agent_persisted`` (#860 / #42039) — but ``_run_agent_via_letta``
never touches the agent SessionDB, so on a gateway whose ``_session_db`` is
live the default ``agent_persisted = self._session_db is not None`` silently
turns every Letta turn's DB write into a no-op. Letta-brain result dicts must
opt into the gateway-side write by returning ``agent_persisted: False``
(the documented opt-in at the persistence block in gateway/run.py).

Same story for usage: the Letta payload carries real token counts, but the
bridge discarded them, so ``update_session(last_prompt_tokens=...)`` recorded
0 for every brain turn.

These tests run the REAL delegation path — ``_run_agent`` is NOT mocked; the
turn routes through ``_run_agent_via_letta`` against a fake urlopen and hits
the real persistence block in ``_handle_message_with_agent``.

Non-goals (deliberately out of scope for B4): credits_tracker accounting and
tool_calls.log entries for brain turns — the Letta server owns tool execution
and billing for its own loop; this module only covers transcript durability
and prompt-token recording.
"""

import io
import json
import sys
import types
import urllib.error
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource

FAKE_PROMPT_TOKENS = 777
FAKE_COMPLETION_TOKENS = 10


def _letta_turn_response(reply: str) -> dict:
    """Blocking-endpoint response shape (see test_letta_brain.py), with the
    usage block Letta actually returns (LettaUsageStatistics)."""
    return {
        "messages": [
            {
                "id": "message-1",
                "message_type": "reasoning_message",
                "reasoning": "thinking about it",
            },
            {
                "id": "message-2",
                "message_type": "assistant_message",
                "content": reply,
            },
        ],
        "usage": {
            "prompt_tokens": FAKE_PROMPT_TOKENS,
            "completion_tokens": FAKE_COMPLETION_TOKENS,
            "total_tokens": FAKE_PROMPT_TOKENS + FAKE_COMPLETION_TOKENS,
        },
        "stop_reason": "end_turn",
    }


class _FakeUrlopenResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_urlopen(monkeypatch, reply: str, seen: dict):
    def _fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeUrlopenResponse(
            json.dumps(_letta_turn_response(reply)).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)


def _install_500_urlopen(monkeypatch):
    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Internal Server Error", {}, io.BytesIO(b"boom")
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)


def _bootstrap(monkeypatch, tmp_path):
    """GatewayRunner wired like test_42039's fixture, plus a Letta brain bound
    via env. ``_run_agent`` is real — turns route through
    ``_run_agent_via_letta`` (blocking client forced for determinism)."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    monkeypatch.setenv("LETTA_BRAIN_URL", "http://localhost:8283")
    monkeypatch.setenv("LETTA_BRAIN_AGENT_ID", "agent-abc")
    monkeypatch.setenv("LETTA_BRAIN_STREAMING", "off")
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    # A live agent SessionDB — this is exactly the configuration where the
    # gateway defaults to skip_db=True and Letta turns silently vanish
    # from state.db unless the result dict opts in with agent_persisted=False.
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-letta-parity",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _event():
    return MessageEvent(
        text="hello brain",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id="12345",
        ),
        message_id="msg-77",
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _calls_for_role(calls, role):
    matched = []
    for call in calls:
        args = call.args
        if len(args) >= 2 and isinstance(args[1], dict) and args[1].get("role") == role:
            matched.append(call)
    return matched


def _assert_role_calls_durable(calls, role):
    """The turn's rows for ``role`` must be written durably (skip_db falsy) —
    inverse of test_42039's native-loop expectation."""
    role_calls = _calls_for_role(calls, role)
    assert len(role_calls) >= 1, (
        f"Expected at least one {role}-role append_to_transcript call, "
        f"got calls: {[c.args for c in calls if len(c.args) >= 2]}"
    )
    for call in role_calls:
        skip_db = call.kwargs.get("skip_db", False)
        assert not skip_db, (
            f"Letta-brain turn wrote a {role} row with skip_db={skip_db!r} — "
            f"the row never reaches state.db (audit hole). kwargs={call.kwargs}"
        )


# ── 1: success turn persists user AND assistant rows durably ──────────


@pytest.mark.asyncio
async def test_letta_turn_persists_user_and_assistant_durably(
    monkeypatch, tmp_path
):
    runner = _bootstrap(monkeypatch, tmp_path)
    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "hello from letta", seen)

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert seen["url"] == "http://localhost:8283/v1/agents/agent-abc/messages", (
        "turn did not route through the real Letta delegation path"
    )
    calls = runner.session_store.append_to_transcript.call_args_list
    _assert_role_calls_durable(calls, "user")
    _assert_role_calls_durable(calls, "assistant")


# ── 2: first turn writes the session_meta transcript row ──────────────


@pytest.mark.asyncio
async def test_letta_turn_writes_session_meta(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "hello from letta", seen)

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    calls = runner.session_store.append_to_transcript.call_args_list
    meta_calls = _calls_for_role(calls, "session_meta")
    assert len(meta_calls) == 1, (
        f"Expected exactly one first-turn session_meta row (parity with the "
        f"native loop), got {len(meta_calls)}. "
        f"calls={[c.args for c in calls if len(c.args) >= 2]}"
    )
    meta = meta_calls[0].args[1]
    assert meta["platform"] == "telegram"
    assert "model" in meta and "timestamp" in meta
    # The meta row itself must be durable too.
    assert not meta_calls[0].kwargs.get("skip_db", False)


# ── 3: usage tokens from the Letta payload reach update_session ───────


@pytest.mark.asyncio
async def test_letta_turn_records_prompt_tokens(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "hello from letta", seen)

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    token_calls = [
        call
        for call in runner.session_store.update_session.call_args_list
        if "last_prompt_tokens" in call.kwargs
    ]
    assert token_calls, "expected an update_session(last_prompt_tokens=...) call"
    recorded = token_calls[-1].kwargs["last_prompt_tokens"]
    assert recorded == FAKE_PROMPT_TOKENS, (
        f"Letta payload reported prompt_tokens={FAKE_PROMPT_TOKENS} but the "
        f"gateway recorded last_prompt_tokens={recorded} (usage discarded)"
    )


# ── 4: LettaBrainError turn still persists the user message durably ───


@pytest.mark.asyncio
async def test_letta_error_turn_persists_user_message(monkeypatch, tmp_path):
    runner = _bootstrap(monkeypatch, tmp_path)
    _install_500_urlopen(monkeypatch)

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    _assert_role_calls_durable(
        runner.session_store.append_to_transcript.call_args_list, "user"
    )
