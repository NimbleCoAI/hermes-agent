"""ReentrancyGuard — process-wide suppression while the worker is sending.

The worker forwards via ``send_message``, which (through ``_run_async``) performs
the actual platform send on a *separate* thread, and the platform adapter emits
its own log records during that send. Those records match the allowlist and would
feed back into the queue. A thread-local guard would miss the adapter's thread,
so this guard is process-wide: while active, the handler drops every record.
"""
from __future__ import annotations

import threading


class ReentrancyGuard:
    def __init__(self) -> None:
        self._depth = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._depth > 0

    def __enter__(self) -> "ReentrancyGuard":
        with self._lock:
            self._depth += 1
        return self

    def __exit__(self, *exc) -> None:
        with self._lock:
            self._depth -= 1
