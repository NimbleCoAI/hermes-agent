"""Tests for the intelligent-routing pre-turn classifier (Option A — heuristic).

The classifier is a PURE function: given cheap local signals available before the
LLM call (message text/length, tool-call context from recent turns, whether the
turn is a background/cron/non-interactive job vs a live human chat, and whether
the request is kanban-triage-shaped), it returns one of:

    "mechanical" -> route to the cheap-metered tier (OpenRouter workhorse)
    "judgment"   -> route to the premium (Claude-class) tier
    "uncertain"  -> FAIL OPEN to premium (never silently downgrade a judgment turn)

`route_for()` collapses (mechanical -> cheap, judgment/uncertain -> premium),
mirroring the fail-open discipline of HSM's ``validateCascadeEntries``.
"""
import pytest

from plugins.intelligent_routing.classifier import (
    classify_turn,
    route_for,
    TurnSignals,
    MECHANICAL,
    JUDGMENT,
    UNCERTAIN,
    TIER_CHEAP,
    TIER_PREMIUM,
)


# ---------------------------------------------------------------- mechanical ---

def test_cron_background_job_is_mechanical():
    """Non-interactive cron/background jobs are mechanical by definition."""
    sig = TurnSignals(user_message="run the nightly digest", is_interactive=False,
                      is_background=True)
    assert classify_turn(sig) == MECHANICAL


def test_short_single_tool_ask_is_mechanical():
    """A short, single-tool-call-shaped ask is mechanical."""
    sig = TurnSignals(user_message="mark card 42 done", is_interactive=True,
                      expects_multi_step=False)
    assert classify_turn(sig) == MECHANICAL


def test_kanban_triage_shaped_request_is_mechanical():
    """Kanban-triage-shaped requests are mechanical (per D-2026-07-08-01)."""
    sig = TurnSignals(user_message="triage the backlog and assign priorities",
                      is_interactive=True, is_kanban_triage=True)
    assert classify_turn(sig) == MECHANICAL


def test_short_mechanical_ask_routes_cheap():
    # A short ask with a leading imperative verb is mechanical -> cheap.
    sig = TurnSignals(user_message="list open PRs", is_interactive=True)
    assert route_for(sig) == TIER_CHEAP


def test_trivial_query_prefix_is_not_cheap():
    """"what's/how many/when is" prefixes front real judgment questions — they must
    fail open to premium, not cheap-route just because they're short. (A genuinely
    trivial "what's 2+2" going premium is harmless.)"""
    for msg in (
        "what's 2+2",
        "what's the best architecture for this?",
        "when is it worth adding a cache layer?",
    ):
        sig = TurnSignals(user_message=msg, is_interactive=True)
        assert route_for(sig) == TIER_PREMIUM, msg


# ------------------------------------------------------------------ judgment ---

def test_open_ended_chat_is_judgment():
    """Open-ended, multi-step reasoning is a judgment turn."""
    sig = TurnSignals(
        user_message=(
            "Think through the tradeoffs between our OpenRouter cheap tier and "
            "keeping Sonnet primary, and recommend an approach with reasoning."
        ),
        is_interactive=True,
        expects_multi_step=True,
    )
    assert classify_turn(sig) == JUDGMENT


def test_public_facing_turn_is_judgment():
    """Anything public-facing must go premium (visible quality matters)."""
    sig = TurnSignals(user_message="draft the launch announcement", is_interactive=True,
                      is_public_facing=True)
    assert classify_turn(sig) == JUDGMENT


def test_long_message_is_judgment():
    """A long, substantive human message is judgment, not mechanical."""
    long_msg = "Here is the situation. " * 60  # well over the mechanical length cap
    sig = TurnSignals(user_message=long_msg, is_interactive=True)
    assert classify_turn(sig) == JUDGMENT


def test_judgment_turn_routes_premium():
    sig = TurnSignals(user_message="weigh the options and decide", is_interactive=True,
                      expects_multi_step=True)
    assert route_for(sig) == TIER_PREMIUM


# ------------------------------------------------ uncertain -> fail open -------

def test_empty_message_is_uncertain():
    """No signal to classify on -> uncertain."""
    sig = TurnSignals(user_message="", is_interactive=True)
    assert classify_turn(sig) == UNCERTAIN


def test_uncertain_fails_open_to_premium():
    """Uncertain MUST route premium — never silently downgrade a judgment turn."""
    sig = TurnSignals(user_message="", is_interactive=True)
    assert route_for(sig) == TIER_PREMIUM


def test_ambiguous_midlength_interactive_fails_open():
    """A medium interactive message that trips no mechanical signal fails open."""
    # Not short enough to be mechanical, not clearly multi-step/public-facing.
    sig = TurnSignals(
        user_message="Can you look at the thing we discussed and get back to me?",
        is_interactive=True,
    )
    assert route_for(sig) in (TIER_PREMIUM,)  # fail-open, never cheap on ambiguity


def test_public_facing_beats_short_length():
    """Public-facing dominates even a short message — never cheap-route it."""
    sig = TurnSignals(user_message="ship it", is_interactive=True, is_public_facing=True)
    assert route_for(sig) == TIER_PREMIUM


# -------------------------------------------------------- purity / determinism -

def test_classifier_is_pure_and_deterministic():
    sig = TurnSignals(user_message="mark card 42 done", is_interactive=True)
    assert classify_turn(sig) == classify_turn(sig)


def test_route_for_only_returns_known_tiers():
    for msg in ("", "x", "triage backlog", "a" * 5000):
        assert route_for(TurnSignals(user_message=msg, is_interactive=True)) in (
            TIER_CHEAP,
            TIER_PREMIUM,
        )
