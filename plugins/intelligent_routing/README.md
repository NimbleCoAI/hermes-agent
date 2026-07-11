# intelligent_routing

Opt-in **pre-turn cost routing** for the Hermes main chat turn, built as the
clean answer to how upstream PR #43534 was closed.

teknium1 closed #43534 (model-task-router) with a steer, not a merits rejection:
> *"If you do want it, please create a plugin. Happy to support adding to the
> plugin interface to make this work as one."*

This directory is both halves of that answer:

1. **The plugin** — classifies each turn and, when it's mechanical/orchestration
   work, routes it to a cheap-metered tier; premium/uncertain/public-facing stay
   on the configured primary (fail-open).
2. **The interface extension he offered** — `agent/routing_override.py` (+ one
   call site in `agent/turn_context.py`): a `pre_llm_call` hook result may now
   carry a `route` override that the runtime **actuates** for the turn. Without
   it, `pre_llm_call` could only inject context, never change which model runs.

The classifier is deliberately simple — it is **not** the contribution. The
plugin form + the interface extension are.

## Enable

```bash
hermes plugins enable intelligent_routing
```

Per agent, in `config.yaml` (default **OFF** — opt-in; it changes which model
answers a user's message):

```yaml
routing:
  intelligent: true                       # default false
  mode: heuristic                         # Option A (default); "llm-router" (Option B) = Phase 2, not built
  cheap_model: deepseek/deepseek-v3.2     # cheap-tier target (default)
  cheap_provider: openrouter              # cheap-tier provider (default)
```

Enabled but `routing.intelligent` OFF ⇒ the hook is inert (`None`, zero change).

## How a turn flows

```
turn arrives
  └─ pre_llm_call hook (plugins/intelligent_routing/registration.py)
       ├─ decide_tier(sig) (classifier.py) ── gate cheap on the CONSERVATIVE
       │     binary classifier: cheap ONLY if classify_turn == MECHANICAL,
       │     else PREMIUM (fail open). NO catch-all-to-cheap.
       ├─ inject model-awareness reason line:
       │     [Current model: … — routed: mechanical → cheap]   (or "judgment → premium")
       └─ if cheap: return {"route": {"model": …, "provider": …}}
             └─ turn_context extracts + applies it (agent/routing_override.py)
                  └─ agent.switch_model(...)  ── turn-scoped: reverted next turn
```

**Probe the decision in isolation** (for re-audit / live-probe, no config/plumbing):

```python
from plugins.intelligent_routing.registration import probe_route
probe_route("Write a Python function that merges two sorted linked lists.")
#   -> {"tier": "premium", "classification": "uncertain"}
probe_route("grep for TODO in the repo")
#   -> {"tier": "cheap", "classification": "mechanical"}
probe_route("run the nightly digest", platform="cron")
#   -> {"tier": "cheap", "classification": "mechanical"}
```

## The interface extension (the centerpiece)

`agent/routing_override.py` — a minimal, additive, reference implementation:

- `extract_routing_override(results)` — pull the first `{"route": {...}}` from the
  list of `pre_llm_call` results (first well-formed, `model` required).
- `apply_routing_override(agent, override)` — actuate the swap by delegating to
  the agent's existing, tested `switch_model` (full atomic client rebuild +
  rollback), then **scope it to this turn with a DEDICATED flag**
  (`_routing_override_active`), stashing the premium `_primary_runtime` snapshot
  in `_routing_override_saved_primary`. It deliberately leaves `_primary_runtime`
  pointing at the **cheap** runtime for the turn and does **not** touch
  `_fallback_activated`. `restore_primary_runtime` reverts the swap at the top of
  the next turn via a dedicated block that runs **before** both the
  `_fallback_activated` gate and the rate-limit cooldown gate. Fail-safe: any
  error leaves the agent on its configured primary and never raises into the turn
  loop.

  > **Why the dedicated flag (three audit-confirmed bugs the earlier design had).**
  > An earlier version scoped the override by restoring the premium snapshot into
  > `_primary_runtime` and arming `_fallback_activated`. That overloaded reactive-
  > fallback state and broke three error-adjacent paths: (1) the cheap model
  > **leaked past its turn** when `restore_primary_runtime`'s rate-limit cooldown
  > gate skipped restoration (Blocker 1); (2) in-turn transient-transport recovery
  > (which rebuilds from `_primary_runtime`) **jumped the routed turn onto premium**
  > (Major 2); (3) pre-arming `_fallback_activated` made a cheap-model 429 **skip
  > arming its cooldown** (Major 3). Keeping `_primary_runtime == cheap` + a
  > dedicated flag fixes all three, and lets reactive fallback run correctly
  > UNDERNEATH a routed turn with the cheap runtime as its "primary".

Call site: `agent/turn_context.py`, immediately after the `pre_llm_call` results
are collected (before the turn's first API call assembles). That is the whole
core change — one module + one call site.

**Why this is the right shape for upstream.** It generalizes: any `pre_llm_call`
plugin can now return a per-turn model/provider override, not just this one. It
reuses `switch_model` rather than duplicating the swap logic. It's turn-scoped and
fail-safe by construction. This is what teknium1 offered to add to the interface.

## The routing decision (simplified after a live-deploy bug)

**The decider is the conservative binary classifier, not the task-type table.**
`decide_tier(sig)` routes to cheap **only** when `classify_turn(sig) == MECHANICAL`
(a confident positive: cron/background, kanban-triage, or a short direct
single-action ask). Everything else — architecture, code-gen, open-ended
reasoning, hedged/ambiguous asks, long messages, public-facing — **fails open to
premium**.

> **Why (the live-deploy bug this fixes).** An earlier version routed on the
> 5-way `classify_task_type`, whose catch-all default was `orchestration → cheap`.
> A live deploy on the cyborg agent proved that hard architecture questions,
> multi-file code-gen ("Write a Python function that merges two sorted linked
> lists"), and hedged/uncertain asks ALL fell through the catch-all onto cheap —
> the "fail open to premium" guarantee was false in practice. The fix: gate cheap
> on the conservative binary classifier and make the default premium. There is no
> catch-all-to-cheap. `tests/plugins/intelligent_routing/test_realistic_distribution.py`
> feeds a representative batch through the FULL hook and asserts what actually
> routes — the regression that would have caught this.

`classify_task_type` (the 5-way path) still exists as a descriptive utility and
is unit-tested, but it is **NOT consulted in the routing path**.

Cheap tier defaults to `deepseek/deepseek-v3.2` — v3.2, **not** v4-Pro; grounded
in #43534's data (DeepSeek coding throughput directionally lower → keep code-gen
on Claude; DeepSeek only competitive AND cheap at CLI/tool orchestration). See
`references/prior-art.md`.

## Scope — DONE vs DEFERRED

**Done (Stage 1, our fork):**
- Per-agent `routing.intelligent` toggle (default OFF), `routing.mode`,
  `routing.cheap_model` / `cheap_provider`.
- Task-type heuristic classifier (pure, deterministic, fail-open to premium).
- `pre_llm_call` wiring: injects the routing-reason line AND emits the `route`
  override for cheap turns; inert when disabled; never raises.
- **Interface extension** (`routing_override.py` + `turn_context.py` call site)
  that actuates the swap, turn-scoped and fail-safe.
- Full RGTDD, incl. an end-to-end actuation test (decision → real swap).

**Deferred:**
- **Phase 3 measurement** — before/after per-agent token spend on a pilot agent
  (the interface extension unlocks this; not yet run).
- **Option B (`mode: llm-router`)** — cheap-LLM-as-router. Config key read, not built.
- **HSM UI toggle** (Phase 2) — the Switch beside the cascade editor.
- Richer signals (tool-call history, real kanban/public-facing detection).
- **Live-agent E2E** — the swap is proven against a faithful fake agent in tests,
  NOT yet against a real running agent hitting OpenRouter.

## ⚠️ For the independent audit

The **only** core-runtime changes are `agent/routing_override.py` and the single
call site in `agent/turn_context.py`. They touch the model-selection path and are
**user-visible** (they change which model answers a turn). Everything else is
plugin-local. Audit focus points:
- Turn-scoping correctness: proven against the REAL `restore_primary_runtime` in
  `tests/agent/test_routing_override_revert.py` — the revert runs before both the
  `_fallback_activated` and rate-limit cooldown gates, so it happens regardless of
  cooldown state (the first-round audit's leak repro is now a passing regression
  test there).
- Interaction with reactive fallback within the same turn: `_primary_runtime`
  stays == cheap during the routed turn, so reactive fallback engages with the
  cheap runtime as its "primary" and cooldown accounting is intact (covered by
  `test_reactive_fallback_engages_under_a_routed_turn`).
- Fail-safe coverage: a bad/unreachable cheap provider must degrade to the
  primary, not break the turn.
- **The swap is verified against faithful fake agents + the real
  `restore_primary_runtime`, but NOT yet against a live agent hitting OpenRouter —
  a live-agent run is required before fleet rollout.**
