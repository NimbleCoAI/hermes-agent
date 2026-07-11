"""Per-agent config for intelligent routing (read from config.yaml).

Follows the ``image_gen`` plugin's pattern (``from hermes_cli.config import
load_config``; read a named section; best-effort with ``{}`` on failure). The
toggle lives in its own ``routing:`` namespace — a plugin owns its config
namespace, so this sidesteps needing a core ``routing:`` section (per the spec's
2026-07-09 update: teknium1 steered this to a plugin precisely so it doesn't
fight for space in core).

Defaults, on purpose:
  - ``routing.intelligent`` defaults to **False** — opt-in, because it changes
    which model answers a user's message (visible behavior). A failed/absent
    config reads as OFF, never silently ON.
  - ``routing.mode`` defaults to ``"heuristic"`` (Option A). ``"llm-router"``
    (Option B) is Phase 2 and not implemented here.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_DEFAULT_MODE = "heuristic"


def _load_config() -> Dict[str, Any]:
    """Load the full config dict (``{}`` on any failure). Patched in tests."""
    from hermes_cli.config import load_config

    cfg = load_config()
    return cfg if isinstance(cfg, dict) else {}


def _routing_section() -> Dict[str, Any]:
    try:
        cfg = _load_config()
    except Exception as exc:  # noqa: BLE001 — config is best-effort; fail OFF
        logger.debug("intelligent_routing: could not load config: %s", exc)
        return {}
    section = cfg.get("routing") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return bool(value) if isinstance(value, (int, float)) else False


def is_intelligent_routing_enabled() -> bool:
    """True only when ``routing.intelligent`` is explicitly truthy. Default OFF."""
    return _coerce_bool(_routing_section().get("intelligent", False))


def routing_mode() -> str:
    """``heuristic`` (Option A, default) or ``llm-router`` (Option B, Phase 2)."""
    mode = _routing_section().get("mode", _DEFAULT_MODE)
    mode = str(mode).strip().lower() if mode else _DEFAULT_MODE
    return mode or _DEFAULT_MODE
