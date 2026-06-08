"""RoutePolicy — decides whether a LogRecord should be forwarded to home."""
import logging

from plugins.observability.home_log_router.policy import RoutePolicy


def _record(name: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )


def test_forwards_allowed_logger_at_or_above_floor():
    policy = RoutePolicy(logger_prefixes=["gateway.platforms.signal"], level=logging.WARNING)
    assert policy.should_forward(_record("gateway.platforms.signal", logging.WARNING)) is True


def test_drops_allowed_logger_below_floor():
    policy = RoutePolicy(logger_prefixes=["gateway.platforms.signal"], level=logging.WARNING)
    assert policy.should_forward(_record("gateway.platforms.signal", logging.INFO)) is False


def test_drops_logger_not_in_allowlist():
    policy = RoutePolicy(logger_prefixes=["gateway.platforms.signal"], level=logging.WARNING)
    assert policy.should_forward(_record("some.other.module", logging.ERROR)) is False


def test_forwards_submodule_of_allowed_prefix():
    policy = RoutePolicy(logger_prefixes=["agent.conversation_loop"], level=logging.WARNING)
    assert policy.should_forward(_record("agent.conversation_loop.cascade", logging.ERROR)) is True
