"""Regressions for the partial-summary salvage, from an independent audit.

Three classes of defect the first cut of this feature had, each of which made
it emit *confidently wrong* data rather than fail:

**B1 — the published transcript went stale after compression.** `messages` is
not only appended to inside `run_conversation`; it is REBOUND by every
`_compress_context` call (5 sites in that function). An AST scan found 4 of the
7 binding sites did not re-publish, so `agent._session_messages` pointed at the
abandoned pre-compression list. The salvage then reported only pre-compression
turns *and* a tool-call tally of "0 tool call(s): 0 ok, 0 errored" — a fabricated
number stated with the same confidence as a real one. Fixed by re-publishing
inside `_compress_context` itself (the single choke point, so a new call site
cannot miss it).

**B2 — the "(empty)" failure sentinel was salvaged as reasoning.** A child that
gives up after repeated empty-LLM-response retries emits the literal
``(empty)`` plus recovery scaffolding. The success path treats that as failure.
The salvage accepted it, so a stuck child produced a summary of "(empty)" three
times over while its real finding was pushed out of the tail window.

**S1 — the salvaged body was uncapped on the wire.** The progress relay copies
the summary onto the gateway websocket before `_apply_summary_budget` runs.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from tools.delegate_tool import (
    _EMPTY_RESPONSE_SENTINEL,
    _PARTIAL_SUMMARY_MAX_CHARS,
    _salvage_partial_summary,
)


def _child(messages):
    class C:
        pass
    c = C()
    c._session_messages = messages
    return c


# --------------------------------------------------------------------------
# B1 — every rebind of `messages` must leave _session_messages current
# --------------------------------------------------------------------------

class TestPublishedTranscriptCannotGoStale:
    def test_compress_context_republishes_session_messages(self):
        """The choke-point fix — this is what makes the 4 missed sites safe."""
        import run_agent

        src = inspect.getsource(run_agent.AIAgent._compress_context)
        assert "_session_messages" in src, (
            "_compress_context must re-publish the new list. Every caller "
            "rebinds its local `messages` to the returned list; without this, "
            "an external holder of _session_messages keeps the abandoned "
            "pre-compression list and the salvage silently reports stale work."
        )

    def test_every_messages_rebind_in_run_conversation_is_accounted_for(self):
        """AST scan, not regex.

        The original verification used `grep -E '^\\s+messages = '`, which
        structurally cannot match the tuple-unpacking form
        `messages, active_system_prompt = agent._compress_context(...)` — that
        is exactly how all four missed sites were written.
        """
        from agent import conversation_loop

        src = inspect.getsource(conversation_loop)
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "run_conversation"
        )

        lines = src.splitlines()
        unaccounted = []
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            elif isinstance(node, (ast.For, ast.NamedExpr)):
                targets = [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name) and sub.id == "messages":
                        stmt = lines[node.lineno - 1]
                        # Window to the next 40 lines: a fixed-size window is
                        # brittle (an added comment block silently changes the
                        # verdict), so make it generously larger than any
                        # comment while still local to the rebind.
                        window = "\n".join(lines[node.lineno - 1:node.lineno + 40])
                        # Accounted for if it re-publishes locally, OR the RHS
                        # is _compress_context (which re-publishes internally).
                        if (
                            "agent._session_messages = messages" in window
                            or "_compress_context" in stmt
                        ):
                            continue
                        unaccounted.append((node.lineno, stmt.strip()))

        assert not unaccounted, (
            "these rebinds of `messages` leave agent._session_messages "
            "pointing at an abandoned list, so the delegation timeout salvage "
            f"would report stale work: {unaccounted}"
        )


# --------------------------------------------------------------------------
# B2 — the empty-response sentinel is failure, not reasoning
# --------------------------------------------------------------------------

class TestEmptyResponseSentinelIsNotSalvaged:
    def test_bare_empty_sentinel_is_skipped(self):
        msgs = [
            {"role": "assistant", "content": "Found the registry entry for ACME."},
            {"role": "assistant", "content": _EMPTY_RESPONSE_SENTINEL},
            {"role": "assistant", "content": _EMPTY_RESPONSE_SENTINEL},
            {"role": "assistant", "content": _EMPTY_RESPONSE_SENTINEL},
        ]
        summary, _ = _salvage_partial_summary(_child(msgs), 0)

        assert "ACME" in summary, (
            "the real finding must survive — three sentinel turns previously "
            "pushed it out of the tail window"
        )
        assert _EMPTY_RESPONSE_SENTINEL not in summary

    @pytest.mark.parametrize(
        "flag", ["_empty_recovery_synthetic", "_empty_terminal_sentinel"]
    )
    def test_recovery_scaffolding_flags_are_skipped(self, flag):
        msgs = [
            {"role": "assistant", "content": "real analysis here"},
            {"role": "assistant", "content": "scaffolding text", flag: True},
        ]
        summary, _ = _salvage_partial_summary(_child(msgs), 0)
        assert "real analysis here" in summary
        assert "scaffolding text" not in summary

    def test_sentinel_only_transcript_yields_no_summary(self):
        """Nothing real was produced — do not manufacture a partial."""
        msgs = [{"role": "assistant", "content": _EMPTY_RESPONSE_SENTINEL}]
        summary, _ = _salvage_partial_summary(_child(msgs), 0)
        assert summary == ""


# --------------------------------------------------------------------------
# S1 — bounded output
# --------------------------------------------------------------------------

class TestSalvagedBodyIsBounded:
    def test_huge_turns_are_capped(self):
        msgs = [{"role": "assistant", "content": "x" * 500_000} for _ in range(3)]
        summary, _ = _salvage_partial_summary(_child(msgs), 0)

        assert len(summary) < _PARTIAL_SUMMARY_MAX_CHARS + 2000, (
            f"salvaged summary was {len(summary):,} chars — the progress relay "
            "copies this onto the gateway websocket before the model-context "
            "budget runs"
        )
        assert "further chars of salvaged text omitted" in summary

    def test_normal_sized_output_is_not_truncated(self):
        msgs = [{"role": "assistant", "content": "a concise finding"}]
        summary, _ = _salvage_partial_summary(_child(msgs), 0)
        assert "omitted]" not in summary


# --------------------------------------------------------------------------
# S5 — one malformed tool_call must not destroy the whole salvage
# --------------------------------------------------------------------------

class TestMalformedToolCallsDoNotAbortSalvage:
    @pytest.mark.parametrize(
        "tool_calls",
        [[None], ["a string"], [{"function": None}], [{"function": {}}], [42]],
        ids=["none", "str", "null-fn", "empty-fn", "int"],
    )
    def test_prose_survives_a_malformed_tool_call(self, tool_calls):
        msgs = [{
            "role": "assistant",
            "content": "the valuable finding",
            "tool_calls": tool_calls,
        }]
        summary, trace = _salvage_partial_summary(_child(msgs), 0)
        assert "the valuable finding" in summary, (
            "a malformed tool_call previously raised inside the loop and was "
            "swallowed by the function-level except, discarding every "
            "recovered turn"
        )
        assert isinstance(trace, list)


# --------------------------------------------------------------------------
# The tally must describe the transcript it actually read
# --------------------------------------------------------------------------

class TestTallyIsHonest:
    def test_zero_tool_calls_reported_only_when_there_were_none(self):
        msgs = [
            {
                "role": "assistant",
                "content": "searching",
                "tool_calls": [
                    {"id": "a", "function": {"name": "web_search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "results"},
            {"role": "assistant", "content": "found it"},
        ]
        summary, trace = _salvage_partial_summary(_child(msgs), 0)
        assert "1 tool call(s): 1 ok, 0 errored" in summary
        assert len(trace) == 1


class TestCompressRepublishAtRuntime:
    """Prove the mechanism, not just its source text.

    Simulates the exact B1 scenario: the salvage holds _session_messages, the
    turn compresses (rebinding to a NEW list), the child then appends its real
    finding to the new list, and only then times out.
    """

    def test_salvage_sees_post_compression_work(self, monkeypatch):
        import run_agent
        from tools.delegate_tool import _salvage_partial_summary

        pre = [{"role": "assistant", "content": "early scratch reasoning"}]

        class Fake:
            _session_messages = pre

        agent = Fake()

        compressed = [{"role": "assistant", "content": "compacted history"}]

        def fake_compress(_agent, _messages, _system, **kwargs):
            return compressed, "sys"

        monkeypatch.setattr(
            "agent.conversation_compression.compress_context", fake_compress
        )

        new_messages, _sys = run_agent.AIAgent._compress_context(
            agent, pre, "sys", task_id="t"
        )
        assert new_messages is compressed, "sanity: compression returned a new list"

        # The child keeps working on the post-compression list.
        new_messages.append({"role": "assistant", "content": "THE REAL FINDING"})

        summary, _ = _salvage_partial_summary(agent, 0)
        assert "THE REAL FINDING" in summary, (
            "the salvage read the abandoned pre-compression list — this is B1"
        )
        assert "early scratch reasoning" not in summary
