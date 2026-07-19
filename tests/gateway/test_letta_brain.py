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
from gateway.letta_brain import LettaBrainError, send_message
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

    reply = send_message("http://localhost:8283", "agent-123", "hi there")

    assert reply == "hello from letta"
    assert seen["url"] == "http://localhost:8283/v1/agents/agent-123/messages"
    assert seen["body"] == {"messages": [{"role": "user", "content": "hi there"}]}


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
    # Only the NEW message goes over the wire — Letta owns its own history.
    assert seen["body"] == {"messages": [{"role": "user", "content": "hello brain"}]}
    adapter.send_typing.assert_awaited_once()


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
