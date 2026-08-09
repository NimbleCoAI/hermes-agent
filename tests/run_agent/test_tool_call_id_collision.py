"""Cross-round tool_call_id reuse must not delete tool results.

Regression suite for the Matilde P0 (session ``20260809_040444_3aca1c0f``,
Discord thread 1534161039812198430, model ``moonshotai/kimi-k3`` via
OpenRouter).

Some upstream providers mint tool_call ids scoped to the assistant message
that issued them — ``read_file:0``, ``read_file:1``, ``terminal:0``,
``skill_view:0`` — so the ids are unique WITHIN a round and repeat on EVERY
round.  ``sanitize_api_messages`` deduplicated tool_call ids with
conversation-global sets, so only the FIRST result per distinct id ever
reached the model.  In the incident session that collapsed 185 tool results
onto 10 distinct ids: the prompt grew by a constant +23 tokens per API call
(the stripped assistant envelope) regardless of whether the tool returned
286 or 10,533 characters, the model re-read files it could not see, the
anti-repeat guard blocked it, and the turn burned 60/60 api_calls ending with
``response_len=0``.

Numbers quoted in the assertions below come from that session's
``/opt/data/logs/agent.log`` and ``/opt/data/state.db``.
"""

import logging

from agent.agent_runtime_helpers import sanitize_api_messages


def _round(call_ids, tool_name="read_file", result_size=4000):
    """Build one assistant tool-call round plus its tool results."""
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "call_id": cid,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": '{"path":"/opt/data/fde-lean/FDE/Completeness.lean"}',
                },
            }
            for cid in call_ids
        ],
    }
    results = [
        {
            "role": "tool",
            "tool_call_id": cid,
            "name": tool_name,
            "content": f"{cid}-payload-" + ("z" * result_size),
        }
        for cid in call_ids
    ]
    return [assistant, *results]


def test_results_survive_when_provider_reuses_call_ids_across_rounds():
    """Every round's tool results must survive round-scoped id reuse."""
    messages = [{"role": "user", "content": "formalize FDE completeness"}]
    for _ in range(12):
        messages += _round(["read_file:0", "read_file:1"])

    out = sanitize_api_messages(list(messages))

    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 24, (
        f"expected all 24 tool results to survive, got {len(tool_msgs)} — "
        "cross-round id reuse is being treated as duplication"
    )
    assert all("payload" in m["content"] for m in tool_msgs)

    assistants = [m for m in out if m.get("role") == "assistant"]
    assert len(assistants) == 12
    assert all(len(m.get("tool_calls") or []) == 2 for m in assistants), (
        "assistant tool_calls were stripped, leaving empty envelopes"
    )


def test_mixed_tool_names_reusing_the_same_index_all_survive():
    """The incident's real shape: several tools each restarting at ``:0``."""
    messages = [{"role": "user", "content": "go"}]
    for _ in range(6):
        messages += _round(["read_file:0"], tool_name="read_file")
        messages += _round(["terminal:0"], tool_name="terminal")
        messages += _round(["skill_view:0"], tool_name="skill_view")

    out = sanitize_api_messages(list(messages))

    assert len([m for m in out if m.get("role") == "tool"]) == 18
    result_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert len(set(result_ids)) == 18, "wire payload still carries duplicate ids"


def test_uniquified_payload_has_no_duplicate_tool_call_ids():
    """Results survive AND the wire payload keeps tool_call_id unique.

    Strict providers (DeepSeek) reject a duplicate tool_call_id with HTTP 400
    (#58327), so the fix must rename rather than keep the collision.
    """
    messages = [{"role": "user", "content": "go"}]
    for _ in range(3):
        messages += _round(["terminal:0"], tool_name="terminal")

    out = sanitize_api_messages(list(messages))

    call_ids = [
        tc["id"]
        for m in out
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    ]
    result_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert len(call_ids) == 3 and len(set(call_ids)) == 3, call_ids
    assert result_ids == call_ids, (result_ids, call_ids)


def test_uniquification_is_deterministic_and_prefix_stable():
    """Renaming must be stable so the provider-side prompt cache still hits."""
    messages = [{"role": "user", "content": "go"}]
    for _ in range(4):
        messages += _round(["read_file:0"])

    first = sanitize_api_messages(list(messages))
    second = sanitize_api_messages(list(messages))
    assert [m.get("tool_call_id") for m in first] == [
        m.get("tool_call_id") for m in second
    ]

    extended = sanitize_api_messages(list(messages) + _round(["read_file:0"]))
    assert [m.get("tool_call_id") for m in extended][: len(first)] == [
        m.get("tool_call_id") for m in first
    ]


def test_uniquification_does_not_mutate_persisted_messages():
    """The live/persisted trajectory keeps the provider's original ids."""
    messages = [{"role": "user", "content": "go"}]
    for _ in range(3):
        messages += _round(["read_file:0"])

    sanitize_api_messages(list(messages))

    for m in messages:
        if m.get("role") == "assistant":
            assert [tc["id"] for tc in m["tool_calls"]] == ["read_file:0"]
            assert [tc["call_id"] for tc in m["tool_calls"]] == ["read_file:0"]
        elif m.get("role") == "tool":
            assert m["tool_call_id"] == "read_file:0"


def test_dropping_tool_result_content_logs_an_error(caplog):
    """Losing a content-bearing tool result must be LOUD, never silent.

    A second result for the SAME call inside ONE round is a genuine duplicate
    and is still dropped — but the sanitizer must name the numbers so the
    failure shows up in the log instead of presenting as a model that
    inexplicably re-reads the same file forever.
    """
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_X",
                    "call_id": "call_X",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_X", "content": "A" * 10533},
        {"role": "tool", "tool_call_id": "call_X", "content": "B" * 9654},
    ]

    with caplog.at_level(logging.ERROR):
        out = sanitize_api_messages(list(messages))

    assert len([m for m in out if m.get("role") == "tool"]) == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "dropping a tool result must log an ERROR"
    joined = "\n".join(errors)
    assert "9654" in joined, joined  # characters of model-visible context lost
    assert "call_X" in joined, joined  # which call it belonged to


def test_within_round_duplicate_call_ids_are_still_collapsed():
    """Negative control: two calls sharing an id in ONE message stay collapsed.

    Their results are indistinguishable, so uniquifying them would invent a
    pairing that does not exist (#58327 shape).
    """
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_Y", "type": "function",
                 "function": {"name": "foo", "arguments": "{}"}},
                {"id": "call_Y", "type": "function",
                 "function": {"name": "bar", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_Y", "content": "r"},
    ]

    out = sanitize_api_messages(list(messages))
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_Y"]


def test_globally_unique_ids_are_left_untouched():
    """Negative control: providers that already mint unique ids see no rename."""
    messages = [{"role": "user", "content": "go"}]
    for n in range(5):
        messages += _round([f"call_{n:08x}"])

    out = sanitize_api_messages(list(messages))

    result_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert result_ids == [f"call_{n:08x}" for n in range(5)]
