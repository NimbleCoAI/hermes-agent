"""Tests for Telegram group auto-approval on bot add (my_chat_member).

When an admin adds the bot to a group and TELEGRAM_GROUP_AUTOAPPROVE is
enabled, the adapter asks HSM to approve the group (the adder must be a
platform admin). Approved -> greet and stay; denied -> polite decline and
leave. Flag unset -> log-only no-op, preserving prior behavior.
"""
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter(**extra):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter._bot = SimpleNamespace(
        id=999,
        username="test_bot",
        send_message=AsyncMock(),
        leave_chat=AsyncMock(),
    )
    return adapter


def _make_update(
    *,
    chat_id=-100123,
    chat_type="supergroup",
    old_status="left",
    new_status="member",
    adder_id=777,
):
    return SimpleNamespace(
        my_chat_member=SimpleNamespace(
            chat=SimpleNamespace(id=chat_id, type=chat_type, title="Test Group"),
            from_user=SimpleNamespace(id=adder_id),
            old_chat_member=SimpleNamespace(status=old_status),
            new_chat_member=SimpleNamespace(status=new_status),
        )
    )


_FLAG_ON = {"TELEGRAM_GROUP_AUTOAPPROVE": "1"}


@pytest.mark.asyncio
async def test_added_and_approved_greets_and_stays():
    """Approved add: bot greets the group and does not leave."""
    adapter = _make_adapter()
    update = _make_update()
    with patch.dict(os.environ, _FLAG_ON):
        with patch(
            "plugins.swarm_map_policy.approve_group_add", return_value=True
        ) as mock_approve:
            await adapter._handle_my_chat_member(update, None)

    mock_approve.assert_called_once_with("-100123", "777", platform="telegram")
    adapter._bot.send_message.assert_awaited_once()
    assert adapter._bot.send_message.await_args.kwargs["chat_id"] == -100123
    adapter._bot.leave_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_added_and_denied_declines_and_leaves():
    """Denied add: bot sends one polite decline, then leaves the chat."""
    adapter = _make_adapter()
    update = _make_update()
    with patch.dict(os.environ, _FLAG_ON):
        with patch("plugins.swarm_map_policy.approve_group_add", return_value=False):
            await adapter._handle_my_chat_member(update, None)

    adapter._bot.send_message.assert_awaited_once()
    text = adapter._bot.send_message.await_args.kwargs["text"]
    assert "approved" in text.lower()
    adapter._bot.leave_chat.assert_awaited_once_with(chat_id=-100123)


@pytest.mark.asyncio
async def test_flag_off_is_logged_noop():
    """Flag unset: event is a no-op — no HSM call, no message, no leave."""
    adapter = _make_adapter()
    update = _make_update()
    env = {k: v for k, v in os.environ.items() if k != "TELEGRAM_GROUP_AUTOAPPROVE"}
    with patch.dict(os.environ, env, clear=True):
        with patch(
            "plugins.swarm_map_policy.approve_group_add", return_value=False
        ) as mock_approve:
            await adapter._handle_my_chat_member(update, None)

    mock_approve.assert_not_called()
    adapter._bot.send_message.assert_not_awaited()
    adapter._bot.leave_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_promotion_transition_is_noop():
    """member -> administrator (promotion) must not re-trigger approval."""
    adapter = _make_adapter()
    update = _make_update(old_status="member", new_status="administrator")
    with patch.dict(os.environ, _FLAG_ON):
        with patch(
            "plugins.swarm_map_policy.approve_group_add", return_value=False
        ) as mock_approve:
            await adapter._handle_my_chat_member(update, None)

    mock_approve.assert_not_called()
    adapter._bot.send_message.assert_not_awaited()
    adapter._bot.leave_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_bot_leaving_is_noop():
    """member -> left (bot removed) must not trigger anything."""
    adapter = _make_adapter()
    update = _make_update(old_status="member", new_status="left")
    with patch.dict(os.environ, _FLAG_ON):
        with patch(
            "plugins.swarm_map_policy.approve_group_add", return_value=True
        ) as mock_approve:
            await adapter._handle_my_chat_member(update, None)

    mock_approve.assert_not_called()
    adapter._bot.send_message.assert_not_awaited()
    adapter._bot.leave_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_chat_is_noop():
    """my_chat_member in a private chat (user unblocked bot etc.) is ignored."""
    adapter = _make_adapter()
    update = _make_update(chat_type="private", chat_id=555)
    with patch.dict(os.environ, _FLAG_ON):
        with patch(
            "plugins.swarm_map_policy.approve_group_add", return_value=True
        ) as mock_approve:
            await adapter._handle_my_chat_member(update, None)

    mock_approve.assert_not_called()
    adapter._bot.send_message.assert_not_awaited()
    adapter._bot.leave_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_plain_group_chat_is_enforced():
    """Basic (non-super) groups are gated too."""
    adapter = _make_adapter()
    update = _make_update(chat_type="group", chat_id=-4567)
    with patch.dict(os.environ, _FLAG_ON):
        with patch(
            "plugins.swarm_map_policy.approve_group_add", return_value=True
        ) as mock_approve:
            await adapter._handle_my_chat_member(update, None)

    mock_approve.assert_called_once_with("-4567", "777", platform="telegram")


@pytest.mark.asyncio
async def test_approval_error_fails_closed():
    """Exception in the approval path counts as denied: decline + leave."""
    adapter = _make_adapter()
    update = _make_update()
    with patch.dict(os.environ, _FLAG_ON):
        with patch(
            "plugins.swarm_map_policy.approve_group_add",
            side_effect=Exception("HSM exploded"),
        ):
            await adapter._handle_my_chat_member(update, None)

    adapter._bot.leave_chat.assert_awaited_once_with(chat_id=-100123)


@pytest.mark.asyncio
async def test_decline_send_failure_still_leaves():
    """If the decline message can't be sent, the bot still leaves."""
    adapter = _make_adapter()
    adapter._bot.send_message.side_effect = Exception("send failed")
    update = _make_update()
    with patch.dict(os.environ, _FLAG_ON):
        with patch("plugins.swarm_map_policy.approve_group_add", return_value=False):
            await adapter._handle_my_chat_member(update, None)

    adapter._bot.leave_chat.assert_awaited_once_with(chat_id=-100123)
