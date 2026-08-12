"""Tests for task-TYPE classification + tier mapping (adopted from PR #43534).

PR #43534 (model-task-router, closed on NousResearch/hermes-agent) routes by
task TYPE — code-gen / hard-architecture / orchestration / research / mechanical
— not a binary mechanical/judgment split, and grounds the mapping in DeepSWE
reliability data (see plugins/intelligent_routing/references/deepswe-routing-data.md).

We ADOPT its five task categories and its reliability principle, but ADAPT the
tier mapping to OUR fleet's actual tiers (NOT #43534's GPT-5.4/5.5):

    premium (Claude sonnet-5 / fable-5 / opus)   — hard-architecture, code-gen, research
    cheap   (OpenRouter deepseek/deepseek-v3.2)   — orchestration, mechanical
    local   (ollama qwen/glm)                     — floor, not auto-selected here

Uncertain -> fail open to premium. Public-facing -> always premium.
"""
import pytest

from plugins.intelligent_routing.classifier import (
    classify_task_type,
    tier_for_task_type,
    route_task,
    TurnSignals,
    TASK_CODE_GEN,
    TASK_ARCHITECTURE,
    TASK_ORCHESTRATION,
    TASK_RESEARCH,
    TASK_MECHANICAL,
    TASK_UNCERTAIN,
    TIER_CHEAP,
    TIER_PREMIUM,
)


# ------------------------------------------------- task-type classification ---

def test_code_gen_detected():
    for msg in (
        "implement the retry logic in agent/chat_completion_helpers.py",
        "fix the bug in the parser",
        "refactor this module and open a PR",
        "add a feature that writes the config file",
    ):
        assert classify_task_type(TurnSignals(user_message=msg)) == TASK_CODE_GEN, msg


def test_architecture_detected():
    for msg in (
        "design a system for multi-tenant routing across the fleet",
        "how would you structure the caching layer",
        "do a security review of the auth flow",
        "help with complex debugging of this distributed race condition",
    ):
        assert classify_task_type(TurnSignals(user_message=msg)) == TASK_ARCHITECTURE, msg


def test_mechanical_detected():
    for msg in (
        "grep for TODO in the repo",
        "run the tests",
        "list all files in plugins/",
        "mark card 42 done",
    ):
        assert classify_task_type(TurnSignals(user_message=msg)) == TASK_MECHANICAL, msg


def test_background_job_is_mechanical_task():
    sig = TurnSignals(user_message="nightly digest", is_background=True,
                      is_interactive=False)
    assert classify_task_type(sig) == TASK_MECHANICAL


def test_kanban_triage_is_mechanical_task():
    sig = TurnSignals(user_message="triage the backlog", is_kanban_triage=True)
    assert classify_task_type(sig) == TASK_MECHANICAL


def test_research_detected():
    for msg in (
        "research the tradeoffs between OpenRouter and z.ai",
        "summarize this document for me",
        "explain how the fallback chain works",
        "look up the pricing for deepseek",
    ):
        assert classify_task_type(TurnSignals(user_message=msg)) == TASK_RESEARCH, msg


def test_orchestration_is_the_catch_all():
    """Tool/shell/nav/diagnostics/planning with no stronger signal -> orchestration."""
    sig = TurnSignals(user_message="check the deploy status and tell me what's up")
    assert classify_task_type(sig) == TASK_ORCHESTRATION


def test_empty_message_is_uncertain_task():
    assert classify_task_type(TurnSignals(user_message="")) == TASK_UNCERTAIN


def test_code_gen_beats_mechanical_priority():
    """Priority (per #43534): code-gen wins over a coincidental mechanical verb."""
    # "run" is a mechanical verb but the task is implementing code.
    sig = TurnSignals(user_message="implement and run the new migration script")
    assert classify_task_type(sig) == TASK_CODE_GEN


# ----------------------------------------------------------- tier mapping -----

def test_architecture_maps_premium():
    assert tier_for_task_type(TASK_ARCHITECTURE) == TIER_PREMIUM


def test_code_gen_maps_premium():
    """Code-gen -> premium: DeepSWE shows DeepSeek coding throughput directionally
    lower; we keep multi-file code-gen on Claude for our fleet."""
    assert tier_for_task_type(TASK_CODE_GEN) == TIER_PREMIUM


def test_research_maps_premium():
    assert tier_for_task_type(TASK_RESEARCH) == TIER_PREMIUM


def test_orchestration_maps_cheap():
    """Orchestration -> cheap: DeepSeek is competitive at CLI/tool orchestration
    (Terminal-Bench 67.9% @ $0.87/1M) — the empirical basis for v3.2 here."""
    assert tier_for_task_type(TASK_ORCHESTRATION) == TIER_CHEAP


def test_mechanical_maps_cheap():
    assert tier_for_task_type(TASK_MECHANICAL) == TIER_CHEAP


def test_uncertain_maps_premium_fail_open():
    assert tier_for_task_type(TASK_UNCERTAIN) == TIER_PREMIUM


# --------------------------------------------------- route_task end-to-end ----

def test_public_facing_forced_premium_even_if_mechanical_shaped():
    """Public-facing dominates: never cheap-route a public-facing turn."""
    sig = TurnSignals(user_message="run the announcement", is_public_facing=True)
    assert route_task(sig) == TIER_PREMIUM


def test_mechanical_turn_routes_cheap():
    assert route_task(TurnSignals(user_message="grep for TODO")) == TIER_CHEAP


def test_architecture_turn_routes_premium():
    sig = TurnSignals(user_message="design the multi-tenant routing system")
    assert route_task(sig) == TIER_PREMIUM


def test_uncertain_turn_fails_open_premium():
    assert route_task(TurnSignals(user_message="")) == TIER_PREMIUM


def test_route_task_only_returns_known_tiers():
    for msg in ("", "x", "implement foo", "grep bar", "design baz", "a" * 4000):
        assert route_task(TurnSignals(user_message=msg)) in (TIER_CHEAP, TIER_PREMIUM)


def test_task_type_classifier_is_deterministic():
    sig = TurnSignals(user_message="refactor the parser and open a PR")
    assert classify_task_type(sig) == classify_task_type(sig)
