"""gardener_approvals — deterministic two-way Signal approval trigger.

On every Signal turn, parses a reply to an open gardener decision and routes it
through the decision-approve gate with the verified sender UUID. The gate owns
all security; the LLM only relays the gate's confirm line. See the design spec
in nimbleco-egregore: specs/2026-06-09-gardener-approval-trigger-hook-design.md.
"""
from __future__ import annotations


def register(ctx) -> None:
    from .registration import register as _register
    _register(ctx)
