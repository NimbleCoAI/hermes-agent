"""Throttle — dedup identical messages, rate-cap a window, summarize suppressions.

``admit(msg)`` returns the list of strings that should actually be sent: usually
``[msg]``; ``[]`` when suppressed; and a leading summary line on the first admit
following any suppression.
"""
from plugins.observability.home_log_router.throttle import Throttle


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _make(rate=100, window=60.0, dedup_window=300.0):
    clock = FakeClock()
    return Throttle(rate=rate, window=window, dedup_window=dedup_window, clock=clock), clock


def test_admits_distinct_messages():
    t, _ = _make()
    assert t.admit("A") == ["A"]
    assert t.admit("B") == ["B"]


def test_suppresses_identical_within_dedup_window():
    t, clock = _make(dedup_window=300.0)
    assert t.admit("A") == ["A"]
    clock.t = 1.0
    assert t.admit("A") == []


def test_readmits_identical_after_dedup_window():
    t, clock = _make(dedup_window=300.0)
    t.admit("A")
    clock.t = 301.0
    assert "A" in t.admit("A")


def test_rate_cap_suppresses_excess_within_window():
    t, _ = _make(rate=2, window=60.0)
    assert t.admit("X") == ["X"]
    assert t.admit("Y") == ["Y"]
    assert t.admit("Z") == []  # third distinct message in the window is over the cap


def test_rate_resets_after_window():
    t, clock = _make(rate=1, window=10.0)
    assert t.admit("X") == ["X"]
    assert t.admit("Y") == []
    clock.t = 11.0
    assert "Z" in t.admit("Z")


def test_summary_emitted_after_suppression():
    t, clock = _make(rate=1, window=10.0)
    t.admit("A")          # admitted
    t.admit("B")          # rate-suppressed (count=1)
    clock.t = 11.0
    out = t.admit("C")    # new window: should lead with a summary, then "C"
    assert out[-1] == "C"
    assert len(out) == 2
    assert "1" in out[0]
    assert "suppress" in out[0].lower()


def test_summary_counter_resets_after_emit():
    t, clock = _make(rate=1, window=10.0)
    t.admit("A")
    t.admit("B")          # suppressed (count=1)
    clock.t = 11.0
    t.admit("C")          # emits summary, resets counter
    clock.t = 22.0
    assert t.admit("D") == ["D"]  # no lingering summary


def test_flush_summary_returns_pending_and_resets():
    # A burst that ends with suppression (no following admit) must still be
    # reportable via flush_summary, not lost.
    t, _ = _make(rate=1, window=10.0)
    t.admit("A")
    t.admit("B")  # suppressed
    out = t.flush_summary()
    assert len(out) == 1
    assert "1" in out[0] and "suppress" in out[0].lower()
    assert t.flush_summary() == []  # counter cleared


def test_flush_summary_empty_when_nothing_suppressed():
    t, _ = _make()
    t.admit("A")
    assert t.flush_summary() == []
