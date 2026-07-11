# intelligent_routing

Opt-in **pre-turn cost routing** for the Hermes main chat turn.

When enabled per-agent, a cheap local heuristic classifies each incoming turn as
**mechanical** (route to the cheap-metered tier — e.g. an OpenRouter workhorse)
vs. **judgment** (route to the premium, Claude-class tier), and surfaces *why* in
the model-awareness prompt line so a user is never confused by a silent quality-
tier switch mid-thread.

This is the plugin form of the routing mechanism four upstream issues have asked
for (#30652, #61371/#61373, #32704) and PR #43534 attempted — built as a plugin
per teknium1's steer on that PR, and motivated by NimbleCo's real fleet cost
(`[intelligent-routing-cost]`: ~$1k/mo, 6/6 agents primary on metered Sonnet with
no cheap rung anywhere in any cascade).

## Enable

```bash
hermes plugins enable intelligent_routing
```

Then, per agent, in `config.yaml` (default **OFF** — this is opt-in because it
changes which model answers a user's message):

```yaml
routing:
  intelligent: true      # default false
  mode: heuristic        # Option A (default). "llm-router" (Option B) = Phase 2, not built.
```

With the plugin enabled but `routing.intelligent` OFF, the hook is **inert** —
it returns `None` and there is zero behavior change.

## What it classifies (Option A — heuristic, no extra LLM call)

`classifier.classify_turn(TurnSignals) -> "mechanical" | "judgment" | "uncertain"`
is a pure, deterministic function of cheap local signals. Definitions
(spec open-question #3, per D-2026-07-08-01):

**mechanical** (→ cheap tier):
- cron / background / non-interactive jobs, OR
- kanban-triage-shaped requests, OR
- a **short, direct single-action ask** — a bare imperative command
  (`mark card 42 done`, `close #17`) or a trivial lookup (`what's 2+2`).

**judgment** (→ premium tier):
- open-ended / multi-step reasoning,
- long substantive human messages,
- anything public-facing,
- **any short-but-hedged / deferred / context-referencing ask**
  (`Can you look at the thing we discussed and get back to me?`) — short length
  is necessary but NOT sufficient for mechanical.

**uncertain** → **fails open to premium.** `route_for()` only sends a turn to the
cheap tier on a *confident* mechanical classification; judgment AND uncertain
both go premium. This mirrors HSM's `validateCascadeEntries` fail-open discipline
and honors the spec's rule: never silently downgrade a judgment turn.

## What it surfaces

The `pre_llm_call` hook injects an extended model-awareness line (built on
`agent.model_awareness.format_current_model_line`, so the base format stays
byte-identical):

```
[Current model: x-ai/grok via openrouter — routed: mechanical]
[Current model: claude-sonnet-5 via anthropic — routed: judgment]
```

## Scope — DONE vs DEFERRED

**Done (this Stage-1 slice):**
- Per-agent `routing.intelligent` toggle, default OFF, with `routing.mode`.
- Option-A heuristic classifier as a pure, fully-tested function (mechanical /
  judgment / uncertain, fail-open to premium).
- `pre_llm_call` wiring that computes the decision and injects the routing-reason
  line; inert when disabled; never raises into the turn loop.
- Full RGTDD coverage (see `tests/plugins/intelligent_routing/`).

**Deferred:**
- **Actuation of the model swap.** See below — needs a plugin-interface
  extension. This slice *surfaces* the decision; it does not yet *change* which
  model runs.
- **Option B (`mode: llm-router`)** — cheap-LLM-as-router. Config key is read but
  not implemented.
- **HSM UI toggle** (Phase 2) — the Switch beside the cascade editor.
- **Phase 3 measurement** — before/after per-agent token spend on a pilot agent.
- Richer signals (tool-call history, kanban-shape detection from real context,
  public-facing detection) — currently derived only from message text + platform;
  absent signals default conservatively (keep the turn OUT of cheap).

## The hook-surface limitation (concrete ask for the interface extension)

The existing `pre_llm_call` hook can only **inject context** — its return value
cannot override `model` / `provider` for the turn. The model is frozen by
`restore_primary_runtime` at `agent/turn_context.py:174`, ~250 lines before this
hook fires (`turn_context.py:436`), and the hook receives `model` as a read-only
value, not the agent object. Mutating `agent.model` in `on_session_start` is
session-scoped, not per-turn, so it can't do per-turn routing either.

**Concrete extension request** (answers teknium1's offer on PR #43534): a
pre-turn hook whose return value may carry a `{"model": ..., "provider": ...}`
override that `turn_context` applies *before* the client is built for the turn —
i.e. select the tier's runtime the same way `restore_primary_runtime` sets it,
but from a plugin decision rather than only the reactive fallback chain. With
that, this plugin's `route_for()` result becomes actuating instead of advisory.
