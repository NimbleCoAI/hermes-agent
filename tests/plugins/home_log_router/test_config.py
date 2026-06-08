"""_read_config — env parsing with guards against silently-broken values."""
import logging

from plugins.observability.home_log_router import registration
from plugins.observability.home_log_router.registration import DEFAULT_LOGGERS


def _cfg(monkeypatch, **env):
    for k in (
        "HERMES_HOME_LOG_ENABLED", "HERMES_HOME_LOG_PLATFORM", "HERMES_HOME_LOG_LEVEL",
        "HERMES_HOME_LOG_LOGGERS", "HERMES_HOME_LOG_RATE", "HERMES_HOME_LOG_WINDOW",
        "HERMES_HOME_LOG_DEDUP_WINDOW", "HERMES_HOME_LOG_QUEUE",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return registration._read_config()


def test_defaults(monkeypatch):
    c = _cfg(monkeypatch)
    assert c.enabled is True
    assert c.platform == "signal"
    assert c.level == logging.WARNING
    assert c.loggers == DEFAULT_LOGGERS
    assert c.rate == 20 and c.window == 60 and c.dedup_window == 300


def test_enabled_unset_is_active(monkeypatch):
    assert _cfg(monkeypatch).enabled is True


def test_kill_switch_values_disable(monkeypatch):
    assert _cfg(monkeypatch, HERMES_HOME_LOG_ENABLED="0").enabled is False
    assert _cfg(monkeypatch, HERMES_HOME_LOG_ENABLED="false").enabled is False
    assert _cfg(monkeypatch, HERMES_HOME_LOG_ENABLED="1").enabled is True


def test_nonpositive_rate_falls_back_to_default(monkeypatch):
    assert _cfg(monkeypatch, HERMES_HOME_LOG_RATE="0").rate == 20
    assert _cfg(monkeypatch, HERMES_HOME_LOG_RATE="-5").rate == 20


def test_nonpositive_window_falls_back_to_default(monkeypatch):
    assert _cfg(monkeypatch, HERMES_HOME_LOG_WINDOW="0").window == 60


def test_nonpositive_dedup_falls_back_to_default(monkeypatch):
    assert _cfg(monkeypatch, HERMES_HOME_LOG_DEDUP_WINDOW="0").dedup_window == 300


def test_named_level(monkeypatch):
    assert _cfg(monkeypatch, HERMES_HOME_LOG_LEVEL="ERROR").level == logging.ERROR


def test_numeric_level_string(monkeypatch):
    # "10" must mean DEBUG, not silently fall back to WARNING.
    assert _cfg(monkeypatch, HERMES_HOME_LOG_LEVEL="10").level == logging.DEBUG


def test_bogus_level_falls_back_to_warning(monkeypatch):
    assert _cfg(monkeypatch, HERMES_HOME_LOG_LEVEL="nonsense").level == logging.WARNING


def test_logger_allowlist_override(monkeypatch):
    c = _cfg(monkeypatch, HERMES_HOME_LOG_LOGGERS="a.b, c.d ,e")
    assert c.loggers == ("a.b", "c.d", "e")
