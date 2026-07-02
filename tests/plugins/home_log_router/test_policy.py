"""RoutePolicy — decides whether a LogRecord should be forwarded to home."""
import logging

from plugins.observability.home_log_router.policy import RoutePolicy
from plugins.observability.home_log_router.registration import DEFAULT_LOGGERS


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


def test_default_loggers_route_main_cascade_fallback_warning():
    # The main-cascade "Fallback activated" WARNING rides the dedicated
    # agent.degradation logger (emitted by try_activate_fallback()). It must
    # reach home so a silent main-model degradation (often = Anthropic credits
    # exhausted) is loud, not invisible. Regression guard for
    # [home-log-fallback-gap].
    policy = RoutePolicy(logger_prefixes=DEFAULT_LOGGERS, level=logging.WARNING)
    assert policy.should_forward(
        _record("agent.degradation", logging.WARNING)
    ) is True


def test_default_loggers_do_not_route_chat_helper_lifecycle_noise():
    # agent.chat_completion_helpers also emits stream TTFB/stale timeouts,
    # partial-stream drops, and schema-sanitize WARNINGs. Those must NOT reach
    # the home channel — routing the whole module prefix would spam it and
    # violate the no-lifecycle-spam rule. Only agent.degradation is routed.
    policy = RoutePolicy(logger_prefixes=DEFAULT_LOGGERS, level=logging.WARNING)
    assert policy.should_forward(
        _record("agent.chat_completion_helpers", logging.WARNING)
    ) is False
