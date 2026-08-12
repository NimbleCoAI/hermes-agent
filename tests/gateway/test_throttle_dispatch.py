"""Security tests: R6 throttle enforcement at the handle_message choke point.

Every platform's inbound-LLM paths converge on
``BasePlatformAdapter.handle_message`` — direct text, batched-text flush,
slash dispatch, and thread sessions. These tests pin the dispatch contract:

  - a throttled event never reaches ``_start_session_processing`` (no LLM
    turn is spawned and nothing is queued);
  - control commands (/stop, /approve, /new, ...) bypass the throttle so
    operators can steer during the very flood the throttle is suppressing;
  - /retry IS throttled (it spawns LLM work);
  - observe-only events bypass (they never spawn turns);
  - the denial notice is sent at most once per user per cooldown window —
    the notice must not amplify the flood.
"""

import asyncio

import pytest

from gateway import inbound_throttle
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    """Concrete adapter with abstract methods stubbed out."""

    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = _StubAdapter(config, Platform.DISCORD)
    adapter._busy_text_mode = ""
    adapter.sent_responses = []
    adapter.started_sessions = []

    async def _mock_handler(event):
        cmd = event.get_command()
        return f"handled:{cmd}" if cmd else f"handled:text:{event.text}"

    adapter._message_handler = _mock_handler

    async def _mock_send_retry(chat_id, content, **kwargs):
        adapter.sent_responses.append(content)

    adapter._send_with_retry = _mock_send_retry

    def _mock_start(event, session_key):
        adapter.started_sessions.append(session_key)

    adapter._start_session_processing = _mock_start
    return adapter


def _make_event(text="hello", user_id="u1", chat_id="c1", scope_id="g1"):
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        scope_id=scope_id,
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


@pytest.fixture(autouse=True)
def _throttle_env(monkeypatch):
    """Enable the throttle (the hermetic suite disables it globally) and give
    each test a pristine singleton so windows don't leak between tests."""
    monkeypatch.setenv("GATEWAY_THROTTLE_ENABLED", "true")
    monkeypatch.setattr(inbound_throttle, "_singleton", inbound_throttle.InboundThrottle())


class TestThrottleDispatch:
    @pytest.mark.asyncio
    async def test_throttled_event_never_starts_session(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "1")
        adapter = _make_adapter()

        await adapter.handle_message(_make_event("first"))
        await adapter.handle_message(_make_event("second — throttled"))

        assert len(adapter.started_sessions) == 1, (
            "throttled message still reached _start_session_processing"
        )
        assert not adapter._pending_messages, (
            "throttled message was queued instead of dropped"
        )

    @pytest.mark.asyncio
    async def test_control_commands_bypass_throttle_when_exhausted(self, monkeypatch):
        """Operators must be able to /stop and /approve DURING a raid."""
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "0")
        adapter = _make_adapter()
        sk = build_session_key(_make_event().source)
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("/stop"))
        assert any("handled:stop" in r for r in adapter.sent_responses), (
            "/stop was throttled — operators cannot stop the agent mid-flood"
        )

        adapter._active_sessions[sk] = asyncio.Event()
        await adapter.handle_message(_make_event("/approve"))
        assert any("handled:approve" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_retry_is_throttled(self, monkeypatch):
        """/retry spawns a fresh LLM turn — it must NOT be exempt."""
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "0")
        adapter = _make_adapter()

        await adapter.handle_message(_make_event("/retry"))

        assert adapter.started_sessions == []
        assert not adapter._pending_messages

    @pytest.mark.asyncio
    async def test_observe_only_bypasses_throttle(self, monkeypatch):
        """Observe-only events never spawn turns — throttling them would
        drop transcript context for no cost saving."""
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "0")
        adapter = _make_adapter()
        handled = []

        async def _observe_handler(event):
            handled.append(event.text)

        adapter._message_handler = _observe_handler
        event = _make_event("observed")
        event.observe_only = True

        await adapter.handle_message(event)

        assert handled == ["observed"]

    @pytest.mark.asyncio
    async def test_denial_notice_once_per_cooldown_then_silent(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "0")
        adapter = _make_adapter()

        for i in range(5):
            await adapter.handle_message(_make_event(f"flood {i}"))

        assert len(adapter.sent_responses) == 1, (
            f"expected exactly one throttle notice, got {adapter.sent_responses}"
        )

    @pytest.mark.asyncio
    async def test_notice_cooldown_is_per_user(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "0")
        adapter = _make_adapter()

        await adapter.handle_message(_make_event("a", user_id="u1"))
        await adapter.handle_message(_make_event("b", user_id="u2"))

        assert len(adapter.sent_responses) == 2

    @pytest.mark.asyncio
    async def test_disabled_throttle_dispatches_normally(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_THROTTLE_ENABLED", "false")
        monkeypatch.setenv("GATEWAY_THROTTLE_USER_PER_MINUTE", "0")
        adapter = _make_adapter()

        await adapter.handle_message(_make_event("hello"))

        assert len(adapter.started_sessions) == 1
