"""HomeLogHandler — cheap emit(): policy-gate, format, non-blocking enqueue.

emit() must never touch the network and never block: it only filters and does a
``put_nowait`` (dropping on a full queue). A process-wide re-entrancy guard lets
the worker suppress capture while it is itself sending (which emits adapter logs
that would otherwise feed back).
"""
import logging
import queue

from plugins.observability.home_log_router.handler import HomeLogHandler
from plugins.observability.home_log_router.guard import ReentrancyGuard
from plugins.observability.home_log_router.policy import RoutePolicy


def _record(name="gateway.platforms.signal", level=logging.WARNING, msg="boom"):
    return logging.LogRecord(name, level, __file__, 1, msg, (), None)


def _handler(maxsize=10, guard=None):
    policy = RoutePolicy(["gateway.platforms.signal"], logging.WARNING)
    q = queue.Queue(maxsize=maxsize)
    h = HomeLogHandler(policy=policy, out_queue=q, guard=guard or ReentrancyGuard())
    return h, q


def test_forwards_matching_record_to_queue():
    h, q = _handler()
    h.emit(_record(msg="hello"))
    item = q.get_nowait()
    assert "hello" in item
    assert "WARNING" in item


def test_drops_record_policy_rejects():
    h, q = _handler()
    h.emit(_record(name="some.other.logger", level=logging.ERROR))
    assert q.empty()


def test_drops_silently_when_queue_full():
    h, q = _handler(maxsize=1)
    h.emit(_record(msg="first"))   # fills the queue
    h.emit(_record(msg="second"))  # must not raise, must be dropped
    assert q.qsize() == 1
    assert "first" in q.get_nowait()


def test_drops_record_while_guard_active():
    guard = ReentrancyGuard()
    h, q = _handler(guard=guard)
    with guard:
        h.emit(_record(msg="during-send"))
    assert q.empty()
