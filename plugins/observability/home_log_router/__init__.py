"""home_log_router — forward selected suppressed logs to the Signal home channel.

Hermes agents suppress operational log records (cascade fallbacks, provider
errors, persistent reconnects) to keep chat clean. None of them reach the
agent's home channel — only shutdown/startup/cron resolve ``get_home_channel()``.
This plugin installs a ``logging.Handler`` that forwards a curated, throttled
slice of those records to the home channel via the ``send_message`` tool (a bare
platform target resolves to home). Zero core emit-site edits.

Enabling the plugin activates it; ``HERMES_HOME_LOG_ENABLED=0`` is a kill switch.
With no home channel configured the forward is a silent no-op.

Activation is handled by the Hermes plugin system — standalone plugins only load
when listed in ``plugins.enabled`` (``hermes plugins enable
observability/home_log_router``).
"""
from __future__ import annotations


def register(ctx) -> None:
    # Lazy import: keeps unit submodules independently importable for testing
    # without pulling the whole plugin graph at import time.
    from .registration import register as _register

    _register(ctx)
