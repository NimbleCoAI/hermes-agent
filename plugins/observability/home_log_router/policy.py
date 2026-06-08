"""RoutePolicy — decides whether a LogRecord should be forwarded to home.

Pure decision unit: a record forwards iff its logger name starts with one of
the allowlisted prefixes AND its level is at or above the floor.
"""
from __future__ import annotations

import logging
from typing import Iterable


class RoutePolicy:
    def __init__(self, logger_prefixes: Iterable[str], level: int) -> None:
        # tuple so str.startswith can test all prefixes in one call
        self.prefixes = tuple(logger_prefixes)
        self.level = level

    def should_forward(self, record: logging.LogRecord) -> bool:
        if record.levelno < self.level:
            return False
        return bool(self.prefixes) and record.name.startswith(self.prefixes)
