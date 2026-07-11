"""Intelligent routing plugin — opt-in pre-turn cost routing (Option A).

When enabled per-agent (``routing.intelligent: true`` in config.yaml, default
OFF), a cheap local heuristic classifies each incoming turn as mechanical vs.
judgment and surfaces the routing decision in the model-awareness prompt line so
users see WHY a tier was picked. It fails open to the premium tier on any
uncertainty.

Actuating the model swap (making a mechanical turn actually go to the cheap
provider) requires a plugin-interface extension the maintainer offered on
PR #43534 — the existing ``pre_llm_call`` hook can inject context but cannot
override model/provider for the turn. See ``registration.py`` for details.
"""
from .registration import register  # noqa: F401

__all__ = ["register"]
