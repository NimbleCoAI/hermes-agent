"""Letta-backed turn delegation — SLICE-0 THROWAWAY PROTOTYPE (Swarm Map v1, B1).

A Hermes gateway stays the messaging "door" (surfaces, pairing/group approval,
audit, budget — all untouched upstream of _run_agent), but instead of running
the native AIAgent loop for a turn, it forwards the user message to a bound
Letta agent over REST and returns Letta's assistant reply.

Endpoints (validated live 2026-07-19 against a self-hosted Letta on :8283):
    POST /v1/agents/{agent_id}/messages
        body: {"messages": [{"role": "user", "content": "..."}]}
        resp: {"messages": [{"message_type": "assistant_message",
                             "content": "..."}, ...], ...}

Deliberately stdlib-only (urllib) — the gateway's aiohttp import is optional
(see _run_agent_via_proxy), so the brain client must not add a hard dep. The
sync call is bridged onto the event loop with asyncio.to_thread at the call
site in gateway/run.py.

TODO(slice-4, real B1):
  - per-Letta-agent message serialization (Letta processes an agent's
    messages sequentially; concurrent group messages need a queue — spec B3)
  - streaming (Letta has an SSE endpoint; this blocks for the full turn)
  - auth header for Letta Cloud / secured servers
  - surface reasoning_message content somewhere (audit log?)
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0


class LettaBrainError(Exception):
    """Raised when the Letta round-trip fails. Message is user-displayable."""


def _extract_assistant_text(payload: dict) -> Optional[str]:
    """Pull assistant_message content out of a Letta turn response.

    Letta returns a typed message list (reasoning_message, tool_call_message,
    assistant_message, ...). We join all assistant_message contents; content
    may be a plain string or a list of {"type": "text", "text": ...} parts.
    """
    parts = []
    for msg in payload.get("messages", []):
        if not isinstance(msg, dict):
            continue
        if msg.get("message_type") != "assistant_message":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for piece in content:
                if isinstance(piece, dict) and isinstance(piece.get("text"), str):
                    parts.append(piece["text"])
    if not parts:
        return None
    return "\n\n".join(p for p in parts if p)


def send_message(
    base_url: str,
    agent_id: str,
    text: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Send one user message to a Letta agent; return the assistant reply text.

    Blocking (urllib). Raises LettaBrainError with a clear, user-displayable
    message on any failure. Letta keeps its own conversation state server-side,
    so only the new message is sent — no history replay.
    """
    url = f"{base_url.rstrip('/')}/v1/agents/{agent_id}/messages"
    body = json.dumps(
        {"messages": [{"role": "user", "content": text}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        raise LettaBrainError(
            f"Letta brain error ({e.code}) from {url}: {detail or e.reason}"
        ) from e
    except Exception as e:
        raise LettaBrainError(
            f"Letta brain unreachable at {base_url} "
            f"(is the Letta server running?): {e}"
        ) from e

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise LettaBrainError(f"Letta brain returned non-JSON response: {e}") from e

    reply = _extract_assistant_text(payload)
    if reply is None:
        # A turn can legitimately end without an assistant_message (e.g. the
        # agent only ran tools). For the prototype, treat as an error so the
        # user sees *something* — TODO(slice-4): decide real semantics.
        raise LettaBrainError(
            "Letta brain returned no assistant_message in its reply "
            f"(message_types: {[m.get('message_type') for m in payload.get('messages', []) if isinstance(m, dict)]})"
        )
    return reply
