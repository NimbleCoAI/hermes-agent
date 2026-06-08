"""Env-driven config. Active by default once the plugin is enabled in
plugins.enabled; HERMES_GARDENER_APPROVALS_ENABLED=0 is a kill switch.
"""
from __future__ import annotations

import os
from typing import NamedTuple

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_DEFAULT_GATE = "/opt/data/scripts/decision-approve.sh"
_DEFAULT_DECISIONS = "/opt/data/gardener-memory/state/decisions.md"


class Config(NamedTuple):
    enabled: bool
    gate_path: str
    decisions_file: str
    approver: str


def _enabled() -> bool:
    val = os.getenv("HERMES_GARDENER_APPROVALS_ENABLED")
    if val is None:
        return True
    return val.strip().lower() in _TRUTHY


def load_config() -> Config:
    return Config(
        enabled=_enabled(),
        gate_path=os.getenv("HERMES_GARDENER_GATE", _DEFAULT_GATE).strip() or _DEFAULT_GATE,
        decisions_file=os.getenv("HERMES_GARDENER_DECISIONS", _DEFAULT_DECISIONS).strip() or _DEFAULT_DECISIONS,
        approver=os.getenv("HERMES_GARDENER_APPROVER", "Juniper").strip() or "Juniper",
    )
