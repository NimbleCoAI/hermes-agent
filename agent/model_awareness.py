"""Model self-awareness: surface the live (post-fallback) model each turn.

The runtime always *knows* which model is live — ``try_activate_fallback``
mutates ``agent.model`` / ``agent.provider`` in place when it swaps backends —
but the LLM never sees it, so "what model are you?" gets answered by
hallucination. The pre-migration ``hermes-swarm`` repo fixed this by appending
a ``[Current model: X via Y]`` line to the ephemeral per-turn context; that was
lost in the ``hermes-swarm`` → ``hermes-agent-mt`` migration and is re-ported
here.

Sourcing (the load-bearing detail): the line reads ``agent.model`` /
``agent.provider`` — the live runtime fields — at per-API-call assembly time,
NOT the ``_RUNTIME_MAIN_*`` process globals (which are published once at turn
start, before ``restore_primary_runtime``, and are *not* updated by
``try_activate_fallback``, so they go stale on a degraded turn). Reading the
agent fields means a fallback turn shows the fallback backend, not the
configured primary.

Placement: it is appended to the current turn's user message (call-time only,
never persisted to the session DB), mirroring the existing memory-prefetch /
``pre_llm_call`` plugin injection. It deliberately never touches the cached
system prompt — that prefix must stay byte-stable for prompt caching.

Contents are two non-secret slugs (model + provider); no credentials, no PII.
"""

from __future__ import annotations

from typing import Any, List


def format_current_model_line(model: str, provider: str) -> str:
    """Return ``[Current model: <model> via <provider>]``.

    Falls back to ``[Current model: <model>]`` when the provider is unknown,
    and to ``""`` when the model itself is unknown (inject nothing rather than
    a half-formed line).
    """
    model = (model or "").strip()
    provider = (provider or "").strip()
    if not model:
        return ""
    if provider:
        return f"[Current model: {model} via {provider}]"
    return f"[Current model: {model}]"


def append_current_model_line(injections: List[str], agent: Any) -> None:
    """Append the live-model line to ``injections``, read from ``agent``.

    Reads ``agent.model`` / ``agent.provider`` at call time so that when this
    runs after ``try_activate_fallback`` has mutated those fields it reflects
    the fallback backend, not the configured primary. No-op when the model is
    unknown.
    """
    line = format_current_model_line(
        getattr(agent, "model", "") or "",
        getattr(agent, "provider", "") or "",
    )
    if line:
        injections.append(line)
