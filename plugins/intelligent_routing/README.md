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
       ├─ classify task type (classifier.py)  ── code-gen / architecture /
       │                                          research / orchestration / mechanical
       ├─ map to tier: mechanical|orchestration → cheap ; else → premium (fail open)
       ├─ inject model-awareness reason line:
       │     [Current model: … — routed: mechanical → cheap]
       └─ if cheap: return {"route": {"model": …, "provider": …}}
             └─ turn_context extracts + applies it (agent/routing_override.py)
                  └─ agent.switch_model(...)  ── turn-scoped: reverted next turn
```

## The interface extension (the centerpiece)

`agent/routing_override.py` — a minimal, additive, reference implementation:

- `extract_routing_override(results)` — pull the first `{"route": {...}}` from the
  list of `pre_llm_call` results (first well-formed, `model` required).
- `apply_routing_override(agent, override)` — actuate the swap by delegating to
  the agent's existing, tested `switch_model` (full atomic client rebuild +
  rollback), then **re-scope it to this turn**: restore the pre-swap
  `_primary_runtime` snapshot and arm `_fallback_activated`, so
  `restore_primary_runtime` reverts it at the top of the next turn. Identical
  turn-scoping to the reactive fallback path — a different axis, same data
  structure. Fail-safe: any error leaves the agent on its configured primary and
  never raises into the turn loop.

Call site: `agent/turn_context.py`, immediately after the `pre_llm_call` results
are collected (before the turn's first API call assembles). That is the whole
core change — one module + one call site.

**Why this is the right shape for upstream.** It generalizes: any `pre_llm_call`
plugin can now return a per-turn model/provider override, not just this one. It
reuses `switch_model` rather than duplicating the swap logic. It's turn-scoped and
fail-safe by construction. This is what teknium1 offered to add to the interface.

## Task types → tiers

Adopted from #43534 (see `references/prior-art.md`), adapted to our fleet:

| task type      | tier    | why (directional, recalibrate from fleet data) |
|----------------|---------|--------------------------------------------------|
| architecture   | premium | highest reasoning; never cheap-route hard design/security/debug |
| code-gen       | premium | DeepSWE: DeepSeek coding throughput directionally lower — keep on Claude |
| research       | premium | analysis/judgment, often public-facing |
| orchestration  | cheap   | DeepSeek competitive at CLI/tool orchestration (Terminal-Bench 67.9% @ $0.87/1M) |
| mechanical     | cheap   | fast/cheap delegated workhorse |
| uncertain / public-facing | premium | fail open — never silently downgrade |

Cheap tier defaults to `deepseek/deepseek-v3.2` — v3.2, **not** v4-Pro; that
choice is grounded in #43534's data (see `references/prior-art.md`).

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
- Turn-scoping correctness: is the swap really reverted next turn? (relies on
  `restore_primary_runtime`'s `_fallback_activated` gate — same mechanism reactive
  fallback uses).
- Interaction with reactive fallback within the same turn (both mutate
  `agent.model`; proactive runs first, in the prologue; reactive still applies
  underneath on failure).
- Fail-safe coverage: a bad/unreachable cheap provider must degrade to the
  primary, not break the turn.
- The swap is verified against a fake agent only — a live-agent run is required
  before fleet rollout.
