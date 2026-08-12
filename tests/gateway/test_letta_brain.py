"""Tests for Letta-backed turn delegation (SLICE-0 PROTOTYPE, Swarm Map v1 B1).

A gateway with LETTA_BRAIN_URL + LETTA_BRAIN_AGENT_ID configured must route
an inbound turn to the Letta agent's REST endpoint instead of the native
AIAgent loop, and return Letta's assistant_message content through the normal
result-dict shape.
"""

import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.letta_brain import (
    LettaBrainError,
    apply_sender_tag,
    build_sender_tag,
    send_message,
)
from gateway.session import SessionSource


def _letta_turn_response(reply: str) -> dict:
    """Shape validated live against a self-hosted Letta server (2026-07-19)."""
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
        "usage": {"completion_tokens": 10},
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
        seen["timeout"] = timeout
        seen["headers"] = dict(req.header_items())
        return _FakeUrlopenResponse(
            json.dumps(_letta_turn_response(reply)).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def test_send_message_extracts_assistant_content(monkeypatch):
    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "hello from letta", seen)

    turn = send_message("http://localhost:8283", "agent-123", "hi there")

    assert turn.text == "hello from letta"
    assert seen["url"] == "http://localhost:8283/v1/agents/agent-123/messages"
    assert seen["body"] == {"messages": [{"role": "user", "content": "hi there"}]}


# ---------------------------------------------------------------------------
# B1.5 door-context sender tagging
# ---------------------------------------------------------------------------


def test_build_sender_tag_sender_and_group():
    # Matches the format validated live 2026-07-21 (agent extracted both facts).
    assert build_sender_tag("Alice", "#family") == "[from Alice in #family]"


def test_build_sender_tag_partial_and_empty():
    assert build_sender_tag("Alice", None) == "[from Alice]"
    assert build_sender_tag(None, "#family") == "[in #family]"
    assert build_sender_tag(None, None) == ""


def test_apply_sender_tag_prepends_on_its_own_line():
    assert apply_sender_tag("hello", "Alice", "#family") == "[from Alice in #family]\nhello"


def test_apply_sender_tag_is_noop_without_identity():
    # No sender/group known → message text is forwarded verbatim.
    assert apply_sender_tag("hello") == "hello"


def test_send_message_forwards_the_tagged_content(monkeypatch):
    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "ok", seen)

    send_message("http://localhost:8283", "agent-1", "who am I?", sender="Alice", group="#family")

    # The brain receives the door-context tag inline with the message.
    assert seen["body"] == {
        "messages": [{"role": "user", "content": "[from Alice in #family]\nwho am I?"}]
    }


def test_send_message_untagged_when_no_identity(monkeypatch):
    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "ok", seen)

    send_message("http://localhost:8283", "agent-1", "plain message")

    assert seen["body"] == {"messages": [{"role": "user", "content": "plain message"}]}


def test_send_message_unreachable_raises_clear_error(monkeypatch):
    def _boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    with pytest.raises(LettaBrainError) as excinfo:
        send_message("http://localhost:8283", "agent-123", "hi")
    assert "unreachable" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Gateway routing
# ---------------------------------------------------------------------------


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(send_typing=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner, adapter


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="42",
        chat_id="-100999",
        user_name="tester",
        chat_type="group",
    )


@pytest.mark.asyncio
async def test_turn_routes_through_letta_delegation(monkeypatch):
    """With a Letta brain bound, _run_agent_inner never touches the native
    AIAgent loop — the turn goes to Letta and its reply comes back."""
    monkeypatch.setenv("LETTA_BRAIN_URL", "http://localhost:8283")
    monkeypatch.setenv("LETTA_BRAIN_AGENT_ID", "agent-abc")
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)

    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "the brain says hi", seen)

    runner, adapter = _make_runner()

    result = await runner._run_agent_inner(
        message="hello brain",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="sess-1",
    )

    assert result["final_response"] == "the brain says hi"
    assert result["api_calls"] == 1
    assert seen["url"] == "http://localhost:8283/v1/agents/agent-abc/messages"
    # Only the NEW message goes over the wire (Letta owns its own history), now
    # carrying the B1.5 door-context tag: _make_source() is a group chat with
    # user_name "tester" and no chat_name, so the group falls back to type:id.
    assert seen["body"] == {
        "messages": [
            {"role": "user", "content": "[from tester in group:-100999]\nhello brain"}
        ]
    }
    adapter.send_typing.assert_awaited_once()


@pytest.mark.asyncio
async def test_dm_turn_tags_sender_only_no_group(monkeypatch):
    """A DM has no group — the door-context tag carries just the sender."""
    monkeypatch.setenv("LETTA_BRAIN_URL", "http://localhost:8283")
    monkeypatch.setenv("LETTA_BRAIN_AGENT_ID", "agent-abc")
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)

    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "hi", seen)
    runner, _adapter = _make_runner()

    dm_source = SessionSource(
        platform=Platform.TELEGRAM, user_id="7", chat_id="7",
        user_name="Alice", chat_type="dm",
    )
    await runner._run_agent_inner(
        message="hello", context_prompt="", history=[], source=dm_source, session_id="s",
    )
    assert seen["body"] == {
        "messages": [{"role": "user", "content": "[from Alice]\nhello"}]
    }


@pytest.mark.asyncio
async def test_group_turn_prefers_chat_name_for_group_label(monkeypatch):
    """When the adapter supplies a human-readable chat_name, use it as the group."""
    monkeypatch.setenv("LETTA_BRAIN_URL", "http://localhost:8283")
    monkeypatch.setenv("LETTA_BRAIN_AGENT_ID", "agent-abc")
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)

    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "hi", seen)
    runner, _adapter = _make_runner()

    src = SessionSource(
        platform=Platform.TELEGRAM, user_id="7", chat_id="-100999",
        user_name="Alice", chat_type="group", chat_name="#family",
    )
    await runner._run_agent_inner(
        message="hello", context_prompt="", history=[], source=src, session_id="s",
    )
    assert seen["body"] == {
        "messages": [{"role": "user", "content": "[from Alice in #family]\nhello"}]
    }


@pytest.mark.asyncio
async def test_letta_failure_returns_clear_error_not_crash(monkeypatch):
    monkeypatch.setenv("LETTA_BRAIN_URL", "http://localhost:8283")
    monkeypatch.setenv("LETTA_BRAIN_AGENT_ID", "agent-abc")

    def _boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    runner, _adapter = _make_runner()

    result = await runner._run_agent_inner(
        message="hello brain",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="sess-1",
    )

    assert result["final_response"].startswith("⚠️")
    assert "unreachable" in result["final_response"]
    assert result["api_calls"] == 0


# ---------------------------------------------------------------------------
# Production hardening (real B1 — Swarm Map v1 slice 4)
# ---------------------------------------------------------------------------


def test_send_message_sends_bearer_auth_when_key_given(monkeypatch):
    """Letta Cloud / secured servers need an Authorization header."""
    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "ok", seen)
    send_message("http://localhost:8283", "agent-1", "hi", api_key="sk-letta-xyz")
    assert seen["headers"].get("Authorization") == "Bearer sk-letta-xyz"


def test_send_message_no_auth_header_by_default(monkeypatch):
    seen: dict = {}
    _install_fake_urlopen(monkeypatch, "ok", seen)
    send_message("http://localhost:8283", "agent-1", "hi")
    assert "Authorization" not in seen["headers"]


def test_tool_only_turn_returns_empty_reply_not_error(monkeypatch):
    """A turn can legitimately end without an assistant_message (the agent only
    ran tools). Production semantics: quiet empty reply, NOT an error the user
    sees (resolves the prototype's TODO)."""

    def _fake_urlopen(req, timeout=None):
        return _FakeUrlopenResponse(
            json.dumps(
                {
                    "messages": [
                        {"message_type": "tool_call_message", "tool_call": {}},
                        {"message_type": "tool_return_message", "tool_return": "..."},
                    ]
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    assert send_message("http://localhost:8283", "agent-1", "do the thing").text == ""


def test_parse_stream_line_assistant_delta():
    from gateway.letta_brain import parse_stream_line

    line = 'data: ' + json.dumps(
        {"message_type": "assistant_message", "content": "Hel"}
    )
    assert parse_stream_line(line) == "Hel"


def test_parse_stream_line_content_parts():
    from gateway.letta_brain import parse_stream_line

    line = 'data: ' + json.dumps(
        {
            "message_type": "assistant_message",
            "content": [{"type": "text", "text": "lo"}],
        }
    )
    assert parse_stream_line(line) == "lo"


def test_parse_stream_line_ignores_non_assistant_and_noise():
    from gateway.letta_brain import parse_stream_line

    assert parse_stream_line("data: [DONE]") is None
    assert (
        parse_stream_line(
            'data: {"message_type": "reasoning_message", "reasoning": "hmm"}'
        )
        is None
    )
    assert parse_stream_line("data: {not json") is None
    assert parse_stream_line(": keepalive comment") is None
    assert parse_stream_line("") is None


def test_get_letta_brain_picks_up_api_key(monkeypatch):
    monkeypatch.setenv("LETTA_BRAIN_URL", "http://localhost:8283")
    monkeypatch.setenv("LETTA_BRAIN_AGENT_ID", "agent-abc")
    monkeypatch.setenv("LETTA_BRAIN_API_KEY", "sk-letta-xyz")
    runner, _ = _make_runner()
    brain = runner._get_letta_brain()
    assert brain["api_key"] == "sk-letta-xyz"


def test_letta_agent_lock_identity():
    """B3: one lock per bound Letta agent — same key, same lock."""
    runner, _ = _make_runner()
    a = runner._letta_agent_lock("http://h:8283|agent-1")
    b = runner._letta_agent_lock("http://h:8283|agent-1")
    c = runner._letta_agent_lock("http://h:8283|agent-2")
    assert a is b
    assert a is not c


@pytest.mark.asyncio
async def test_concurrent_group_turns_serialize_per_agent(monkeypatch):
    """B3: Letta processes an agent's messages sequentially — concurrent group
    messages must queue, never overlapping in flight."""
    import asyncio
    import threading

    monkeypatch.setenv("LETTA_BRAIN_URL", "http://localhost:8283")
    monkeypatch.setenv("LETTA_BRAIN_AGENT_ID", "agent-abc")
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    # Force the blocking client path so in-flight tracking is deterministic.
    monkeypatch.setenv("LETTA_BRAIN_STREAMING", "off")

    state = {"in_flight": 0, "max_in_flight": 0}
    gate = threading.Lock()

    def _slow_urlopen(req, timeout=None):
        with gate:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        import time as _t

        _t.sleep(0.05)
        with gate:
            state["in_flight"] -= 1
        return _FakeUrlopenResponse(
            json.dumps(_letta_turn_response("ok")).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", _slow_urlopen)
    runner, _adapter = _make_runner()

    src = _make_source()
    results = await asyncio.gather(
        runner._run_agent_inner(
            message="one", context_prompt="", history=[], source=src, session_id="s1"
        ),
        runner._run_agent_inner(
            message="two", context_prompt="", history=[], source=src, session_id="s2"
        ),
    )
    assert [r["final_response"] for r in results] == ["ok", "ok"]
    assert state["max_in_flight"] == 1, "turns for one Letta agent overlapped"
