"""
Tests for Slack Socket Mode 'Session is closed' self-healing.

Covers the stuck-reconnect scenario where slack_sdk's SocketModeClient
enters an infinite retry loop on a closed aiohttp.ClientSession.
The adapter must detect and recover from this state.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock the slack-bolt package if it's not installed
# ---------------------------------------------------------------------------


def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return  # Real library installed

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)

    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app_token = "xapp-fake"
    a._proxy_url = None
    a._running = True
    a.handle_message = AsyncMock()
    return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSlackSocketSessionClosedHeal:
    """Verify the adapter detects and recovers from a closed aiohttp session."""

    @pytest.mark.asyncio
    async def test_start_handler_replaces_closed_aiohttp_session(self, adapter):
        """_start_socket_mode_handler must replace a closed aiohttp_client_session.

        Reproduces: RuntimeError('Session is closed') stuck loop.
        The new AsyncSocketModeHandler's SocketModeClient has a closed session
        (simulates the race condition). The adapter must swap it for a fresh one
        before the start_async task begins.
        """
        closed_session = MagicMock()
        closed_session.closed = True

        fresh_session = MagicMock()
        fresh_session.closed = False

        mock_client = MagicMock()
        mock_client.aiohttp_client_session = closed_session

        mock_handler = MagicMock()
        mock_handler.client = mock_client

        fake_task = MagicMock()
        fake_task.add_done_callback = MagicMock()

        with (
            patch(
                "plugins.platforms.slack.adapter.AsyncSocketModeHandler",
                return_value=mock_handler,
            ),
            patch("plugins.platforms.slack.adapter.aiohttp") as mock_aiohttp,
            patch("asyncio.create_task", return_value=fake_task),
        ):
            mock_aiohttp.ClientSession.return_value = fresh_session
            adapter._start_socket_mode_handler()

        # The closed session must have been replaced with a fresh one
        assert mock_client.aiohttp_client_session is fresh_session, (
            "_start_socket_mode_handler must replace a closed aiohttp_client_session "
            "with a fresh one to avoid the 'Session is closed' stuck-loop"
        )

    @pytest.mark.asyncio
    async def test_start_handler_leaves_open_session_alone(self, adapter):
        """_start_socket_mode_handler must NOT replace an already-open session."""
        open_session = MagicMock()
        open_session.closed = False

        mock_client = MagicMock()
        mock_client.aiohttp_client_session = open_session

        mock_handler = MagicMock()
        mock_handler.client = mock_client

        fake_task = MagicMock()
        fake_task.add_done_callback = MagicMock()

        with (
            patch(
                "plugins.platforms.slack.adapter.AsyncSocketModeHandler",
                return_value=mock_handler,
            ),
            patch("plugins.platforms.slack.adapter.aiohttp") as mock_aiohttp,
            patch("asyncio.create_task", return_value=fake_task),
        ):
            adapter._start_socket_mode_handler()
            mock_aiohttp.ClientSession.assert_not_called()

        assert mock_client.aiohttp_client_session is open_session

    @pytest.mark.asyncio
    async def test_watchdog_triggers_restart_on_closed_session(self, adapter):
        """Watchdog must trigger restart when handler's aiohttp session is closed.

        Reproduces: task.done() == False (start_async is sleeping),
        is_connected() returns None (indeterminate), but session is closed.
        Without the fix, watchdog never calls _restart_socket_mode.
        """
        closed_session = MagicMock()
        closed_session.closed = True

        mock_client = MagicMock()
        mock_client.aiohttp_client_session = closed_session

        mock_handler = MagicMock()
        mock_handler.client = mock_client
        adapter._handler = mock_handler

        mock_task = MagicMock()
        mock_task.done.return_value = False  # task is running (not done)
        adapter._socket_mode_task = mock_task

        restart_calls: list[str] = []

        async def _fake_restart(reason: str) -> None:
            restart_calls.append(reason)
            adapter._running = False  # stop the loop after first restart

        adapter._restart_socket_mode = _fake_restart
        # Transport probe returns None (indeterminate) — this path must NOT trigger restart
        adapter._socket_transport_connected = AsyncMock(return_value=None)
        adapter._socket_watchdog_interval_s = 0.01

        await adapter._socket_watchdog_loop()

        assert restart_calls, (
            "Watchdog must call _restart_socket_mode when aiohttp session is closed"
        )
        assert "aiohttp session closed" in restart_calls[0], (
            f"Expected restart reason 'aiohttp session closed', got: {restart_calls}"
        )

    @pytest.mark.asyncio
    async def test_watchdog_does_not_restart_when_session_open(self, adapter):
        """Watchdog must NOT restart when task is running and session is open."""
        open_session = MagicMock()
        open_session.closed = False

        mock_client = MagicMock()
        mock_client.aiohttp_client_session = open_session

        mock_handler = MagicMock()
        mock_handler.client = mock_client
        adapter._handler = mock_handler

        mock_task = MagicMock()
        mock_task.done.return_value = False
        adapter._socket_mode_task = mock_task

        restart_calls: list[str] = []

        async def _fake_restart(reason: str) -> None:
            restart_calls.append(reason)

        adapter._restart_socket_mode = _fake_restart

        async def _stop_after_one_iteration():
            await asyncio.sleep(0.02)
            adapter._running = False

        adapter._socket_transport_connected = AsyncMock(return_value=None)
        adapter._socket_watchdog_interval_s = 0.01

        stopper = asyncio.ensure_future(_stop_after_one_iteration())
        await adapter._socket_watchdog_loop()
        stopper.cancel()

        assert not restart_calls, (
            f"Watchdog must not restart when session is open. Got: {restart_calls}"
        )
