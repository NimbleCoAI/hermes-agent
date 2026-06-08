"""register(ctx) — wire the plugin into the running agent.

Enabling the plugin (``hermes plugins enable observability/home_log_router``) is
the opt-in; this then attaches a ``HomeLogHandler`` to the root logger and starts
a ``HomeLogWorker`` that forwards throttled records to the home channel through the
``send_message`` tool. A bare platform target (e.g. ``"signal"``) resolves to
``get_home_channel()``; if no home is configured the tool returns an error which
the worker silently ignores.

``HERMES_HOME_LOG_ENABLED=0`` is an optional kill switch — disable forwarding
without un-enabling the plugin. Everything else defaults sensibly; no knobs needed.
"""
from __future__ import annotations

import logging
import os
import queue
import time

from .guard import ReentrancyGuard
from .handler import HomeLogHandler
from .policy import RoutePolicy
from .throttle import Throttle
from .worker import HomeLogWorker

logger = logging.getLogger(__name__)

# Loggers whose suppressed records are worth surfacing to home, grounded in the
# real module logger names: Signal reconnect/health, model cascade fallbacks, and
# provider errors during model calls. Prefix-matched, so submodules are covered.
DEFAULT_LOGGERS = (
    "gateway.platforms.signal",
    "agent.conversation_loop",
    "model_tools",
)


def _truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in ("0", "false", "no", "off", "")


def _level(name: str) -> int:
    resolved = logging.getLevelName(name.strip().upper())
    return resolved if isinstance(resolved, int) else logging.WARNING


def _enabled() -> bool:
    # Active by default once the plugin is enabled; only an explicit falsey
    # value disables it (kill switch).
    val = os.getenv("HERMES_HOME_LOG_ENABLED")
    return True if val is None else _truthy(val)


def register(ctx) -> None:
    if not _enabled():
        return  # kill switch engaged

    platform = os.getenv("HERMES_HOME_LOG_PLATFORM", "signal").strip() or "signal"
    level = _level(os.getenv("HERMES_HOME_LOG_LEVEL", "WARNING"))
    raw_loggers = os.getenv("HERMES_HOME_LOG_LOGGERS", "")
    loggers = tuple(s.strip() for s in raw_loggers.split(",") if s.strip()) or DEFAULT_LOGGERS
    rate = _int("HERMES_HOME_LOG_RATE", 20)
    window = _float("HERMES_HOME_LOG_WINDOW", 60.0)
    dedup_window = _float("HERMES_HOME_LOG_DEDUP_WINDOW", 300.0)
    queue_max = _int("HERMES_HOME_LOG_QUEUE", 1000)

    guard = ReentrancyGuard()
    out_queue: "queue.Queue[str]" = queue.Queue(maxsize=queue_max)
    handler = HomeLogHandler(RoutePolicy(loggers, level), out_queue, guard)
    throttle = Throttle(rate=rate, window=window, dedup_window=dedup_window, clock=time.monotonic)
    worker = HomeLogWorker(out_queue, throttle, _make_sender(ctx, platform), guard)

    logging.getLogger().addHandler(handler)
    worker.start()
    logger.info(
        "home_log_router active: platform=%s level=%s loggers=%s",
        platform, logging.getLevelName(level), ",".join(loggers),
    )

    def _teardown(**_kwargs):
        logging.getLogger().removeHandler(handler)
        worker.stop()

    ctx.register_hook("on_session_end", _teardown)


def _make_sender(ctx, platform: str):
    def sender(message: str) -> None:
        # Bare platform target -> home channel. Return value (incl. "no home"
        # errors) is intentionally ignored: forwarding is best-effort.
        ctx.dispatch_tool("send_message", {"target": platform, "message": message})

    return sender


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
