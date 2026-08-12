# Multi-Tenant Fork Notice

> **⚠️ The owning org was renamed: `NimbleCoOrg` → `cyborg-garden`.**
> (Earlier history: this repo moved from `NimbleCoAI/hermes-agent-mt` to `NimbleCoOrg/hermes-agent-mt`, and before that was renamed from `hermes-agent` to `hermes-agent-mt`.)
> - **Git URLs:** clone/remote URLs **auto-redirect** — existing checkouts keep working, no action needed.
> - **Container image — action required.** CI now publishes to **`ghcr.io/cyborg-garden/hermes-agent-mt`** (the workflow derives the namespace from the repo owner, so it repointed itself on the rename). **GHCR does not redirect the way Git does.** The org rename carried the package namespace with it, so **`ghcr.io/nimblecoorg/hermes-agent-mt` no longer resolves at all** — a pull against it now fails outright. Repoint any compose file, deploy script, or pinned digest to the `cyborg-garden` path.
> - **The older `NimbleCoAI` packages still pull — and that is the dangerous case.** `NimbleCoAI` is a separate user account and was *not* renamed. **`ghcr.io/nimblecoai/hermes-agent-mt`** remains public and is **frozen at its last build, 2026-08-06**; `docker pull` against it still **succeeds** and simply returns pre-move code, with no error. The pre-rename **`ghcr.io/nimblecoai/hermes-agent`** package is likewise public and frozen (last build 2026-07-21), and it too still returns a *successful* pull. Neither is being deleted; both are the intended read-only grace window. If anything is pinned to either, repoint it.

This is **hermes-agent-mt** — a thin fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) patched for multi-tenant deployments managed by [Swarm Map](https://github.com/cyborg-garden/swarm-map).

## What's Different

This fork adds **2 core patches** and **~8 adapter improvements** on top of upstream:

### Core Patches
1. **Memory context scoping** — Memory writes are scoped per-context (group/DM), not global. Each conversation thread maintains isolated memory.
2. **Context ID sanitization** — Platform-specific context IDs are normalized for safe filesystem paths.

### Adapter Improvements
- Signal: UUID-based allowlisting, group invite policy, voice memo detection, profile name setting
- Mattermost: Channel join/leave gating, per-channel allowlist, mention gating
- Telegram: Group session isolation, admin resolution

### Plugins (installed by HSM)
- `swarm_map_policy` — Group access control via HSM policy endpoint
- `boot_md` — Startup checklist execution
- `lifecycle-notify` — Startup notification hook

## Upstream Relationship

- **Upstream:** [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- **Sync:** Weekly automated rebase via CI workflow
- **Goal:** Minimize diff. Patches that benefit upstream are submitted as PRs.
- **Rebase journal:** See `docs/rebase-journal.md`

## Using This Fork

```bash
# Docker (recommended)
docker pull ghcr.io/cyborg-garden/hermes-agent-mt:latest

# Or build from source
git clone https://github.com/cyborg-garden/hermes-agent-mt.git
cd hermes-agent-mt
pip install -e ".[all]"
```

For multi-tenant management, use [Swarm Map](https://github.com/cyborg-garden/swarm-map).

## License

Same as upstream: MIT License.
