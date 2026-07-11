# Prior art — model-task-router (PR #43534)

This plugin is a light adaptation of prior community work; the contribution here
is **the plugin + the interface extension**, not the classifier. This doc is a
short citation, not a reproduction of the original data.

## Source

- **PR #43534** — *"feat(skills): add model-task-router — automatic task-to-model
  routing backed by DeepSWE data"* (CLOSED on `NousResearch/hermes-agent`), by
  **Sugumaran Balasubramaniyan** (https://github.com/Sugumaran-Balasubramaniyan),
  MIT-licensed. It shipped as a *skill* (two markdown files) and routed by task
  TYPE (code-gen / hard-architecture / orchestration / research / mechanical).
- Maintainer **teknium1** closed it with a steer, not a rejection on merits:
  *"If you do want it, please create a plugin. Happy to support adding to the
  plugin interface to make this work as one."* — which is exactly what this
  plugin + `agent/routing_override.py` implement.

## What we took (and what we did NOT)

**Adopted:** the five task categories and the principle *route by task type, not
brand loyalty; treat published benchmarks as directional*.

**Adapted:** the tier mapping goes to our fleet's tiers (premium = Claude,
cheap = OpenRouter `deepseek/deepseek-v3.2`, local floor = ollama), NOT #43534's
non-fleet GPT-5.4/5.5 defaults.

## The one data point that is load-bearing for our v3.2 choice

From #43534's DeepSWE reference (and the `datacurve-ai/deep-swe#21` correction
thread — note the original author **retracted** the solve-rate analysis; only the
cost correction remained valid):

- DeepSeek V4-Pro's DeepSWE **coding** score was directionally low (~8%, heavily
  caveated: no effort tuning, OpenRouter guardrail 404s, ~5–8% community range).
  → So we do **not** route hard code-gen to DeepSeek at all (code-gen → premium).
- DeepSeek V4-Pro is **competitive at CLI/tool orchestration** (Terminal-Bench
  67.9% at $0.87/1M output). → So we route DeepSeek only where it's both
  competitive AND cheap: mechanical / orchestration turns.

That asymmetry — cheap-and-fine for orchestration, risky for code-gen — is the
whole reason the cheap tier is `deepseek-v3.2` and the mapping keeps code-gen on
Claude. Treat these numbers as directional; recalibrate from real fleet data
(Phase 3 measurement) before trusting them further.

Full data lives in the original PR; we deliberately do not re-host the table here.
