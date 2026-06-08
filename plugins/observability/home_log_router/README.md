# home_log_router

Forward the operational logs that actually matter to an agent's **home channel**.

## Why

Hermes agents suppress operational log records to keep chat clean — cascade
fallbacks, provider errors, persistent reconnects. None of them reach the home
channel: only shutdown/startup/cron resolve `get_home_channel()`. Suppression
isn't routing, so operators are blind to agent health.

This plugin closes that gap **without touching any emit site**. It installs a
`logging.Handler` on the root logger and forwards a curated, throttled slice of
records to the home channel through the existing `send_message` tool (a bare
platform target resolves to the home channel). It survives upstream rebases.

## Enable

```
hermes plugins enable observability/home_log_router
```

That's the opt-in. It activates immediately. If no home channel is configured
(`SIGNAL_HOME_CHANNEL` / `/sethome`), forwarding is a silent no-op until one is.

## What it forwards

Records at **WARNING+** from these loggers (prefix-matched):

- `gateway.platforms.signal` — reconnects / health (benign churn is DEBUG, so only real problems surface)
- `agent.conversation_loop` — model cascade fallbacks
- `model_tools` — provider errors during model calls

## Storm safety

Identical messages are deduplicated (re-surfacing once per window as a
heartbeat), sends are rate-capped per window, and the first send after any
suppression leads with a short "N suppressed" summary. The capture path is
non-blocking (bounded queue, drop-on-full) and fully decoupled from delivery —
logging is never blocked by a slow send.

## Configuration

The only setting anyone needs is the enable switch above. Advanced overrides
(all optional, sensible defaults):

| Env | Default | Meaning |
|-----|---------|---------|
| `HERMES_HOME_LOG_ENABLED` | (active) | Set to `0` to disable without un-enabling the plugin |
| `HERMES_HOME_LOG_PLATFORM` | `signal` | Platform whose home channel receives forwards |
| `HERMES_HOME_LOG_LEVEL` | `WARNING` | Minimum level to forward |
| `HERMES_HOME_LOG_LOGGERS` | (the three above) | Comma-separated logger-name prefixes |
| `HERMES_HOME_LOG_RATE` | `20` | Max sends per window |
| `HERMES_HOME_LOG_WINDOW` | `60` | Rate window (seconds) |
| `HERMES_HOME_LOG_DEDUP_WINDOW` | `300` | Identical-message dedup window (seconds) |
| `HERMES_HOME_LOG_QUEUE` | `1000` | Bounded capture queue size |

## Design

Five small, independently testable units:

- `RoutePolicy` — pure: does this record forward? (allowlist ∩ level floor)
- `Throttle` — pure (injectable clock): dedup + rate-cap + suppression summary
- `HomeLogHandler` — cheap `emit()`: policy-gate, format, non-blocking enqueue
- `HomeLogWorker` — daemon thread: drain → throttle → guarded send
- `ReentrancyGuard` — process-wide; suppresses capture while the worker is itself
  sending (the platform adapter emits its own logs on another thread that a
  thread-local guard would miss)

Tests: `tests/plugins/home_log_router/`.
