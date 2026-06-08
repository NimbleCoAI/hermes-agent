"""pre_llm_call hook: turn a Signal reply into a gated decision action.

Deterministic trigger — parses intent in code and calls the gate with the
verified Signal UUID; the LLM only relays the gate's confirm line. The gate is
the sole security boundary (auth/allowlist/idempotency); this hook's parse is a
convenience the gate re-validates. Fails safe: any error returns "" so a hook
bug can never break the agent turn.
"""
from __future__ import annotations

import logging

from .config import load_config
from .decisions import read_open_ids
from .gate import call_gate
from .parser import parse_reply

logger = logging.getLogger(__name__)

# Tokens whose human message we relay to the user verbatim.
_RELAY = {"OK", "ALREADY_DONE", "AMBIGUOUS", "REFUSED_ALLOWLIST"}


def _relay(msg: str) -> dict:
    # Tell the LLM to relay exactly this line and nothing else.
    return {"context": "[gardener] A Signal decision reply was processed by the "
            "approval gate. Relay this line to the user verbatim and add nothing "
            f"else:\n{msg}"}


def on_pre_llm_call(*, platform: str = "", user_message: str = "",
                    sender_id: str = "", sender_id_alt: str = "",
                    _config=None, _open_reader=None, _gate_caller=None,
                    **_kwargs):
    try:
        cfg = _config or load_config()
        if not cfg.enabled:
            return ""
        if (platform or "").strip().lower() != "signal":
            return ""
        uuid = (sender_id_alt or sender_id or "").strip()
        if not uuid:
            return ""
        open_reader = _open_reader or read_open_ids
        open_ids = open_reader(cfg.decisions_file)
        if not open_ids:
            return ""
        pr = parse_reply(user_message or "", open_ids)
        if pr.ask:
            return _relay(pr.ask)
        if not pr.matched:
            return ""
        gate = _gate_caller or call_gate
        token, msg = gate(
            gate_path=cfg.gate_path, uuid=uuid, decision_id=pr.decision_id,
            action=pr.action, value=pr.value, approver=cfg.approver,
            decisions_file=cfg.decisions_file,
        )
        # msg must be non-empty; empty relay messages are silently dropped (gate protocol)
        if token in _RELAY and msg:
            return _relay(msg)
        if token == "REFUSED_AUTH":
            return ""   # unknown sender -> silence, by design
        if token == "ERROR":
            return {"context": "[gardener] The approval gate couldn't complete "
                    "that. Tell the user it didn't go through and to try again in "
                    "Claude Code. Do not claim success."}
        return ""
    except Exception:
        logger.exception("gardener_approvals hook error (suppressed; turn continues)")
        return ""
