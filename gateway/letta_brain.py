"""Letta-backed turn delegation (Swarm Map v1, B1 — the Hermes-front bridge).

A Hermes gateway stays the messaging "door" (surfaces, pairing/group approval,
audit, budget — all untouched upstream of _run_agent), but instead of running
the native AIAgent loop for a turn, it forwards the user message to a bound
Letta agent over REST and returns Letta's assistant reply.

Endpoints (blocking path validated live 2026-07-19 against a self-hosted
Letta on :8283):
    POST /v1/agents/{agent_id}/messages
        body: {"messages": [{"role": "user", "content": "..."}]}
        resp: {"messages": [{"message_type": "assistant_message",
                             "content": "..."}, ...], ...}
    POST /v1/agents/{agent_id}/messages/stream   (SSE; stream_message below)

The blocking client is deliberately stdlib-only (urllib) — the gateway's
aiohttp import is optional (see _run_agent_via_proxy), so the fallback path
must not add a hard dep. The sync call is bridged onto the event loop with
asyncio.to_thread at the call site in gateway/run.py. The streaming client
uses aiohttp when available; callers fall back to the blocking path on any
streaming failure (defensive: the SSE surface is less battle-tested than the
blocking endpoint we validated live).

Grown out of the slice-0 prototype (agent-mt#96) per the v1 spec: "real B1 =
productionize the prototype + reuse proxy mode's streaming consumer."
"""

import json
import logging
import urllib.error
import urllib.request
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0


class LettaBrainError(Exception):
    """Raised when the Letta round-trip fails. Message is user-displayable.

    ``retryable`` is True only when the failure is provably pre-delivery
    (connection never established, endpoint absent, client dep missing) — the
    Letta server cannot have processed the message, so the caller may safely
    retry via the blocking client. Ambiguous failures (5xx, mid-stream drops)
    stay non-retryable: Letta owns server-side history, and a blind retry
    would make the brain process the turn twice.
    """

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def build_sender_tag(
    sender: Optional[str] = None, group: Optional[str] = None
) -> str:
    """Build the compact door-context tag for a shared Letta brain (B1.5).

    A shared Letta agent serves many senders across many groups but, over REST,
    sees only the message text — Letta owns its own server-side history, so the
    rich per-turn context the native loop builds never reaches it. Without a tag
    the brain can't tell WHO is speaking or WHICH group it's in, and multiplayer
    is broken. We prepend a one-line structured tag; Letta's history then
    accumulates identity naturally, turn over turn.

    Current sender + group label ONLY — deliberately NOT group transcripts
    (spec B1.5). Returns "" when neither is known (so tagging is a no-op).

    Validated live 2026-07-21: given ``[from Alice in #family] ...``, a
    self-hosted Letta agent replied "Alice is messaging me from the group
    #family." — the brain extracts both facts from the inline tag.
    """
    parts = []
    if sender:
        parts.append(f"from {sender}")
    if group:
        parts.append(f"in {group}")
    return f"[{' '.join(parts)}]" if parts else ""


def apply_sender_tag(
    text: str, sender: Optional[str] = None, group: Optional[str] = None
) -> str:
    """Prepend the door-context tag to a message (no-op when no identity known)."""
    tag = build_sender_tag(sender, group)
    return f"{tag}\n{text}" if tag else text


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


def _request_headers(api_key: Optional[str] = None) -> dict:
    """Common request headers; Bearer auth for Letta Cloud / secured servers."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def send_message(
    base_url: str,
    agent_id: str,
    text: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    sender: Optional[str] = None,
    group: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Send one user message to a Letta agent; return the assistant reply text.

    Blocking (urllib). Raises LettaBrainError with a clear, user-displayable
    message on any failure. Letta keeps its own conversation state server-side,
    so only the new message is sent — no history replay.

    ``sender``/``group`` add the B1.5 door-context tag so a shared brain knows
    who is speaking and where (see build_sender_tag). The URL has NO trailing
    slash: ``/messages/`` 307-redirects and the POST body is dropped on the
    redirect (confirmed live 2026-07-21).
    """
    url = f"{base_url.rstrip('/')}/v1/agents/{agent_id}/messages"
    content = apply_sender_tag(text, sender, group)
    body = json.dumps(
        {"messages": [{"role": "user", "content": content}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers=_request_headers(api_key),
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
        # A turn can legitimately end without an assistant_message — the agent
        # only ran tools and chose not to speak. Production semantics: a quiet
        # empty reply (delivery is suppressed downstream), never an error the
        # user sees. The message_types are logged for audit (spec B4).
        logger.info(
            "Letta brain turn ended without assistant_message (tool-only turn); "
            "message_types=%s",
            [
                m.get("message_type")
                for m in payload.get("messages", [])
                if isinstance(m, dict)
            ],
        )
        return ""
    return reply


def parse_stream_line(line: str) -> Optional[str]:
    """Parse one SSE line from ``/v1/agents/{id}/messages/stream``.

    Returns the assistant-text delta the line carries, or None for everything
    else — non-data lines, keepalive comments, the ``[DONE]`` terminator,
    non-assistant message types (reasoning/tool chunks), and unparseable JSON.
    Deliberately forgiving: the streaming surface is less battle-tested than
    the blocking endpoint, so unknown shapes are skipped, never fatal.
    """
    line = line.strip()
    if not line.startswith("data: "):
        return None
    data = line[6:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or obj.get("message_type") != "assistant_message":
        return None
    content = obj.get("content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            piece["text"]
            for piece in content
            if isinstance(piece, dict) and isinstance(piece.get("text"), str)
        ]
        return "".join(parts) or None
    return None


async def stream_message(
    base_url: str,
    agent_id: str,
    text: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    sender: Optional[str] = None,
    group: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream one turn from a Letta agent, yielding assistant-text deltas.

    SSE against ``POST /v1/agents/{id}/messages/stream`` with
    ``stream_tokens: true``, mirroring proxy mode's consumer loop. Raises
    LettaBrainError if aiohttp is unavailable or the request fails before any
    delta arrives — callers fall back to the validated blocking send_message.
    A mid-stream error after deltas have been yielded also raises; the caller
    decides whether the partial text is deliverable.
    """
    try:
        from aiohttp import ClientConnectorError, ClientSession, ClientTimeout
    except ImportError as e:
        raise LettaBrainError("Letta streaming requires aiohttp", retryable=True) from e

    url = f"{base_url.rstrip('/')}/v1/agents/{agent_id}/messages/stream"
    content = apply_sender_tag(text, sender, group)
    body = {
        "messages": [{"role": "user", "content": content}],
        "stream_tokens": True,
    }
    try:
        client_timeout = ClientTimeout(total=0, sock_read=timeout)
        async with ClientSession(timeout=client_timeout) as session:
            async with session.post(
                url, json=body, headers=_request_headers(api_key)
            ) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:300]
                    raise LettaBrainError(
                        f"Letta brain stream error ({resp.status}) from {url}: {detail}",
                        # 404/405 = the stream route doesn't exist on this
                        # server — the message was never accepted for a turn.
                        retryable=resp.status in (404, 405),
                    )
                buffer = ""
                async for chunk in resp.content.iter_any():
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        delta = parse_stream_line(line)
                        if delta:
                            yield delta
    except LettaBrainError:
        raise
    except Exception as e:
        raise LettaBrainError(
            f"Letta brain stream failed against {base_url}: {e}",
            # Connection never established → Letta never saw the message.
            retryable=isinstance(e, (ClientConnectorError, ConnectionRefusedError)),
        ) from e
