"""registration.register(ctx) — wiring: env gate, root handler, sender, teardown."""
import logging
import threading

from plugins.observability.home_log_router import registration
from plugins.observability.home_log_router.handler import HomeLogHandler


class FakeCtx:
    def __init__(self):
        self.dispatched = []
        self.hooks = {}
        self.tool_event = threading.Event()

    def dispatch_tool(self, name, args, **kw):
        self.dispatched.append((name, args))
        self.tool_event.set()
        return "{}"

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)


def _our_handlers():
    return [h for h in logging.getLogger().handlers if isinstance(h, HomeLogHandler)]


def _teardown(ctx):
    for cb in ctx.hooks.get("on_session_end", []):
        cb()


def test_active_when_plugin_enabled_by_default(monkeypatch):
    # No env switch needed: enabling the plugin is the opt-in. register() activates.
    monkeypatch.delenv("HERMES_HOME_LOG_ENABLED", raising=False)
    ctx = FakeCtx()
    try:
        registration.register(ctx)
        assert len(_our_handlers()) == 1
        assert "on_session_end" in ctx.hooks
    finally:
        _teardown(ctx)
    assert _our_handlers() == []


def test_kill_switch_disables(monkeypatch):
    # Optional escape hatch: disable without un-enabling the plugin.
    monkeypatch.setenv("HERMES_HOME_LOG_ENABLED", "0")
    before = len(_our_handlers())
    ctx = FakeCtx()
    registration.register(ctx)
    assert len(_our_handlers()) == before
    assert ctx.hooks == {}


def test_enabled_attaches_handler_and_teardown_hook(monkeypatch):
    monkeypatch.delenv("HERMES_HOME_LOG_ENABLED", raising=False)
    ctx = FakeCtx()
    try:
        registration.register(ctx)
        assert len(_our_handlers()) == 1
        assert "on_session_end" in ctx.hooks
    finally:
        _teardown(ctx)
    assert _our_handlers() == []  # teardown detached it


def test_forwards_record_to_send_message_home(monkeypatch):
    monkeypatch.setenv("HERMES_HOME_LOG_ENABLED", "1")
    ctx = FakeCtx()
    try:
        registration.register(ctx)
        log = logging.getLogger("gateway.platforms.signal")
        log.setLevel(logging.DEBUG)
        log.warning("disk full on %s", "agent-7")
        assert ctx.tool_event.wait(timeout=2.0), "send_message was never dispatched"
        name, args = ctx.dispatched[0]
        assert name == "send_message"
        assert args["target"] == "signal"      # bare platform -> home channel
        assert "disk full on agent-7" in args["message"]
    finally:
        _teardown(ctx)


def test_respects_platform_env(monkeypatch):
    monkeypatch.setenv("HERMES_HOME_LOG_ENABLED", "1")
    monkeypatch.setenv("HERMES_HOME_LOG_PLATFORM", "telegram")
    ctx = FakeCtx()
    try:
        registration.register(ctx)
        log = logging.getLogger("model_tools")
        log.setLevel(logging.DEBUG)
        log.error("provider down")
        assert ctx.tool_event.wait(timeout=2.0)
        _, args = ctx.dispatched[0]
        assert args["target"] == "telegram"
    finally:
        _teardown(ctx)
