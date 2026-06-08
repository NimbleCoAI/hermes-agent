# Hermes Multi-Tenancy Patches

These patches add multi-tenant memory isolation to [Hermes](https://github.com/NousResearch/hermes-agent). They are generated from the commits on this fork and represent the complete delta from upstream.

## What's Changed

### Memory Scoping via context_id

Adds per-channel/tenant memory isolation. Memory writes are routed to scoped subdirectories based on an opaque `context_id`. Reads merge global and scoped entries. Backwards-compatible: `context_id=None` preserves existing behavior.

**Files changed:** `tools/memory_tool.py`, `agent/agent_init.py`, `run_agent.py`, `gateway/run.py`

### Swarm Map Policy Plugin

[Hermes Swarm Map](https://github.com/NimbleCoAI/hermes-swarm-map) integration plugin for group access control. Uses standard plugin hooks — no core code changes.

**Files changed:** `plugins/swarm_map_policy/`

## Patches

| File | Commit |
|------|--------|
| `0001-feat-add-context_id-memory-scoping-for-per-channel-t.patch` | feat: add context_id memory scoping for per-channel/tenant isolation |
| `0002-fix-sanitize-context_id-prevent-global-entry-mutatio.patch` | fix: sanitize context_id, prevent global entry mutation from scoped contexts |
| `0003-feat-add-swarm-map-policy-plugin-for-HSM-group-acces.patch` | feat: add swarm-map-policy plugin for HSM group access control |

## Applying Manually

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
git am /path/to/patches/*.patch
```

## Regenerating

After rebasing onto a new upstream version:

```bash
git format-patch upstream/main..HEAD -o patches/
```

## Related

- [Hermes Swarm Map](https://github.com/NimbleCoAI/hermes-swarm-map) — Admin GUI for managing Hermes agents
- Upstream Issue: _link pending_
