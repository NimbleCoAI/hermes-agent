"""Tests for gateway.event_loop_watchdog — freeze detection + self-heal."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from gateway import event_loop_watchdog as elw


# ---------------------------------------------------------------------------
# probe_loop_responsive — real background loop, no mocks. This is the actual
# boundary-crossing mechanism the whole module depends on, so it's worth
# proving against a genuine event loop rather than only unit-testing the
# decision logic around it.
# ---------------------------------------------------------------------------

class TestProbeLoopResponsive:
    @pytest.fixture
    def running_loop(self):
        """A real asyncio loop spinning on its own background thread."""
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        # Give the loop a moment to actually start running.
        for _ in range(50):
            if loop.is_running():
                break
            time.sleep(0.01)
        yield loop
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    def test_true_for_a_responsive_loop(self, running_loop):
        assert elw.probe_loop_responsive(running_loop, timeout=2.0) is True

    def test_false_for_a_genuinely_blocked_loop(self, running_loop):
        # Wedge the loop with real blocking work: call_soon_threadsafe
        # callbacks run in-line on the loop thread, so a synchronous sleep
        # here genuinely blocks every other coroutine — the actual failure
        # mode this module exists to detect, not a simulation of it.
        def _block_the_loop():
            time.sleep(1.0)

        running_loop.call_soon_threadsafe(_block_the_loop)
        assert elw.probe_loop_responsive(running_loop, timeout=0.2) is False

    def test_false_for_a_loop_that_is_not_running(self):
        loop = asyncio.new_event_loop()
        # Never started — run_coroutine_threadsafe raises RuntimeError.
        assert elw.probe_loop_responsive(loop, timeout=1.0) is False
        loop.close()


# ---------------------------------------------------------------------------
# dump_diagnostics
# ---------------------------------------------------------------------------

class TestDumpDiagnostics:
    def test_writes_a_file_with_thread_stack_content(self, tmp_path):
        dump_path = tmp_path / "nested" / "freeze.log"
        elw.dump_diagnostics(dump_path)
        assert dump_path.exists()
        content = dump_path.read_text()
        assert "Event loop freeze detected" in content
        # faulthandler.dump_traceback output includes "Thread" or "Current thread"
        assert "Thread" in content or "thread" in content

    def test_creates_parent_directories(self, tmp_path):
        dump_path = tmp_path / "a" / "b" / "c" / "freeze.log"
        elw.dump_diagnostics(dump_path)
        assert dump_path.exists()


# ---------------------------------------------------------------------------
# EventLoopWatchdog.check_once — pure decision logic, fully injected.
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _make_watchdog(responsive_sequence, clock=None, threshold=90.0):
    """Build a watchdog whose probe_fn pops from `responsive_sequence` each call."""
    seq = list(responsive_sequence)
    calls = {"dump": [], "exit": []}

    def probe_fn(loop, timeout):
        return seq.pop(0)

    def dump_fn(path):
        calls["dump"].append(path)

    def exit_fn(code):
        calls["exit"].append(code)

    watchdog = elw.EventLoopWatchdog(
        loop=None,  # never touched — probe_fn is faked
        unresponsive_threshold=threshold,
        probe_fn=probe_fn,
        dump_fn=dump_fn,
        dump_path_fn=lambda: Path("/tmp/fake-dump.log"),
        exit_fn=exit_fn,
        clock_fn=clock or FakeClock(),
    )
    return watchdog, calls


class TestCheckOnce:
    def test_responsive_probe_never_fires(self):
        watchdog, calls = _make_watchdog([True, True, True])
        for _ in range(3):
            assert watchdog.check_once() is False
        assert calls["dump"] == []
        assert calls["exit"] == []

    def test_first_unresponsive_probe_does_not_fire_immediately(self):
        watchdog, calls = _make_watchdog([False])
        assert watchdog.check_once() is False
        assert calls["dump"] == []
        assert calls["exit"] == []

    def test_unresponsive_under_threshold_does_not_fire(self):
        clock = FakeClock()
        watchdog, calls = _make_watchdog(
            [False, False, False], clock=clock, threshold=90.0
        )
        watchdog.check_once()  # t=0, starts the timer
        clock.advance(30)
        watchdog.check_once()  # t=30, stalled_for=30 < 90
        clock.advance(30)
        watchdog.check_once()  # t=60, stalled_for=60 < 90
        assert calls["dump"] == []
        assert calls["exit"] == []

    def test_unresponsive_past_threshold_fires_dump_and_exit(self):
        clock = FakeClock()
        watchdog, calls = _make_watchdog(
            [False, False, False], clock=clock, threshold=90.0
        )
        watchdog.check_once()  # t=0, starts the timer
        clock.advance(45)
        watchdog.check_once()  # t=45, stalled_for=45 < 90
        clock.advance(50)
        result = watchdog.check_once()  # t=95, stalled_for=95 >= 90
        assert result is True
        assert calls["dump"] == [Path("/tmp/fake-dump.log")]
        assert calls["exit"] == [1]

    def test_a_responsive_probe_resets_the_freeze_timer(self):
        clock = FakeClock()
        watchdog, calls = _make_watchdog(
            [False, True, False, False], clock=clock, threshold=90.0
        )
        watchdog.check_once()  # t=0, unresponsive — starts timer
        clock.advance(80)
        watchdog.check_once()  # t=80, responsive — resets timer
        clock.advance(80)
        watchdog.check_once()  # t=160, unresponsive again — timer restarts here
        clock.advance(80)
        result = watchdog.check_once()  # stalled_for=80 < 90 since reset
        assert result is False
        assert calls["dump"] == []
        assert calls["exit"] == []

    def test_dump_failure_does_not_prevent_exit(self):
        clock = FakeClock()
        seq = [False, False]
        exits = []

        def probe_fn(loop, timeout):
            return seq.pop(0)

        def dump_fn(path):
            raise OSError("disk full")

        watchdog = elw.EventLoopWatchdog(
            loop=None,
            unresponsive_threshold=10.0,
            probe_fn=probe_fn,
            dump_fn=dump_fn,
            dump_path_fn=lambda: Path("/tmp/fake-dump.log"),
            exit_fn=lambda code: exits.append(code),
            clock_fn=clock,
        )
        watchdog.check_once()
        clock.advance(20)
        result = watchdog.check_once()
        assert result is True
        assert exits == [1]


# ---------------------------------------------------------------------------
# EventLoopWatchdog.run — the real-world threading wrapper
# ---------------------------------------------------------------------------

class TestRun:
    def test_stops_when_stop_event_is_set(self):
        watchdog, calls = _make_watchdog([True] * 1000)
        stop_event = threading.Event()
        watchdog._check_interval = 0.01

        thread = threading.Thread(target=watchdog.run, args=(stop_event,))
        thread.start()
        time.sleep(0.05)
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert calls["exit"] == []

    def test_stops_early_once_a_freeze_is_declared(self):
        clock = FakeClock()
        watchdog, calls = _make_watchdog(
            [False] * 1000, clock=clock, threshold=5.0
        )
        watchdog._check_interval = 0.01
        stop_event = threading.Event()

        # Advance the fake clock in step with real time so check_once
        # eventually crosses the threshold without a real 5-second wait.
        def _clock_ticker():
            for _ in range(200):
                clock.advance(1.0)
                time.sleep(0.005)
                if calls["exit"]:
                    return

        ticker = threading.Thread(target=_clock_ticker, daemon=True)
        ticker.start()

        thread = threading.Thread(target=watchdog.run, args=(stop_event,))
        thread.start()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert calls["exit"] == [1]
