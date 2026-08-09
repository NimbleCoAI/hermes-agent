"""Structural guard: every subprocess-env builder must inject a fresh GH_TOKEN.

The defect this file exists to prevent has already happened once. The GH_TOKEN
injection was added to the terminal paths (``_sanitize_subprocess_env``,
``_make_run_env``) and ``hermes_subprocess_env`` — a sibling env builder for
the browser worker, ACP/codex/copilot executors, the TUI Node host, dep-ensure
and the detached gateway — was simply not updated. Nothing caught it, because
the omission does not fail: ``gh`` falls back to the boot-written
``~/.config/gh/hosts.yml``, whose GitHub App token is dead within the hour. The
symptom is a 401 deep inside a tool, weeks later, with a credential file
sitting right there looking perfectly plausible.

A behavioural test per known builder cannot catch the NEXT builder somebody
adds. This one reads the source instead: any function that finalizes HOME via
``apply_subprocess_home_env`` is, by definition, building an environment for a
spawned process, and must therefore also call ``_inject_github_app_gh_token``.

Non-vacuity is asserted explicitly — a test that silently finds zero builders
would pass forever after a rename.
"""
from __future__ import annotations

import ast
from pathlib import Path

import tools.environments.local as local_mod

HOME_FINALIZER = "apply_subprocess_home_env"
INJECTOR = "_inject_github_app_gh_token"

# Builders known to exist today. Kept as a floor, not as the definition: the
# scan finds builders, this set only proves the scan is actually finding them.
KNOWN_BUILDERS = {
    "_sanitize_subprocess_env",
    "hermes_subprocess_env",
    "_make_run_env",
}


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _env_builders() -> dict[str, set[str]]:
    source = Path(local_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    builders: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = _called_names(node)
        if HOME_FINALIZER in calls:
            builders[node.name] = calls
    return builders


def test_scan_finds_the_known_builders():
    """Non-vacuity floor. Without this, a rename of the finalizer would make
    every assertion below pass over an empty set."""
    found = set(_env_builders())
    assert KNOWN_BUILDERS <= found, f"scan lost builders: {KNOWN_BUILDERS - found}"


def test_every_env_builder_injects_a_fresh_gh_token():
    offenders = sorted(
        name for name, calls in _env_builders().items() if INJECTOR not in calls
    )
    assert not offenders, (
        f"{offenders} build a subprocess environment but never call {INJECTOR}(). "
        f"Without it the spawn authenticates to GitHub out of the boot-written "
        f"~/.config/gh/hosts.yml, which holds a GitHub App token that expires "
        f"within the hour and is refreshed on no schedule the spawn controls. "
        f"That failure is silent — the credential is present, merely dead."
    )


def test_injection_runs_after_home_is_final():
    """Ordering is load-bearing: the mint layer derives the agent's .env dir as
    HOME's parent, so injecting before HOME is finalized reads the wrong home
    (or none) and silently declines to inject."""
    source = Path(local_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        finalize_lines = [
            c.lineno
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == HOME_FINALIZER
        ]
        inject_lines = [
            c.lineno
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == INJECTOR
        ]
        if not finalize_lines or not inject_lines:
            continue
        checked += 1
        assert min(inject_lines) > max(finalize_lines), (
            f"{node.name}: {INJECTOR}() must run after {HOME_FINALIZER}()"
        )
    assert checked == len(KNOWN_BUILDERS), (
        f"ordering was only checked in {checked} builders; expected "
        f"{len(KNOWN_BUILDERS)}"
    )
