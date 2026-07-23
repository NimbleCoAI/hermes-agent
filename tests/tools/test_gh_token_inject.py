"""Tests for short-lived GitHub App token injection as GH_TOKEN.

Tier-1 stripping removes every ambient GitHub credential from tool
subprocesses — deliberately: they are long-lived. But that leaves ``gh``
with no auth at all. git is covered by the ``github_app_token`` credential
helper, which mints a fresh installation token per operation; ``gh`` has no
credential-helper concept and falls back to a ``hosts.yml`` written once at
container boot, whose App token dies within the hour (observed live:
hermes-nimbleco's ``gh`` dead from 2026-07-21 onward while git kept working).

The fix mirrors the git path at the env boundary: mint (cache-first) the
same short-lived installation token and inject it as ``GH_TOKEN`` into the
sanitized subprocess env. Exposure is not widened — the helper already
caches this token under the subprocess HOME, readable by any tool.

See hermes_cli/github_app_token.py for the mint/cache layer.
"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.environments.local as local_mod
from tools.environments.local import (
    _make_run_env,
    _sanitize_subprocess_env,
)


MINT_TARGET = "hermes_cli.github_app_token.get_installation_token"


@pytest.fixture(autouse=True)
def _reset_cooldown():
    """Each test starts with a clean mint-failure cooldown."""
    local_mod._GH_TOKEN_MINT_FAILURE_TS = 0.0
    yield
    local_mod._GH_TOKEN_MINT_FAILURE_TS = 0.0


def _base_env(home="/data/agent/home"):
    return {"HOME": home, "PATH": "/usr/bin"}


class TestInjection:
    def test_minted_token_injected_as_gh_token(self):
        with patch(MINT_TARGET, return_value="ghs_minted") as mint:
            env = _sanitize_subprocess_env(_base_env())
        assert env.get("GH_TOKEN") == "ghs_minted"
        mint.assert_called_once()

    def test_home_parent_is_the_hermes_home_dir(self):
        """The mint layer's contract: HOME is ``{HERMES_HOME}/home``, so the
        dir holding ``.env`` is HOME's parent (github_app_token._default_home)."""
        with patch(MINT_TARGET, return_value="ghs_minted") as mint:
            env = _sanitize_subprocess_env(_base_env())
        home = Path(env["HOME"])
        assert mint.call_args.args[0] == home.parent

    def test_run_env_path_also_injects(self):
        """_make_run_env (foreground terminal) mirrors the sanitize path."""
        with patch(MINT_TARGET, return_value="ghs_minted"):
            env = _make_run_env(_base_env())
        assert env.get("GH_TOKEN") == "ghs_minted"

    def test_ambient_github_tokens_still_stripped(self):
        """Injection must not resurrect the long-lived ambient credentials."""
        base = _base_env() | {
            "GITHUB_TOKEN": "ghp_longlived",
            "GITHUB_APP_PRIVATE_KEY_B64": "pem",
        }
        with patch(MINT_TARGET, return_value="ghs_minted"):
            env = _sanitize_subprocess_env(base)
        assert env.get("GH_TOKEN") == "ghs_minted"
        assert "GITHUB_TOKEN" not in env
        assert "GITHUB_APP_PRIVATE_KEY_B64" not in env


class TestFailSoft:
    def test_no_app_creds_no_injection(self):
        """None from the mint layer = this agent has no App creds. Not an
        error: PAT-less / App-less installs see exactly the old behavior."""
        with patch(MINT_TARGET, return_value=None):
            env = _sanitize_subprocess_env(_base_env())
        assert "GH_TOKEN" not in env

    def test_mint_error_swallowed(self):
        with patch(MINT_TARGET, side_effect=RuntimeError("api down")):
            env = _sanitize_subprocess_env(_base_env())  # must not raise
        assert "GH_TOKEN" not in env

    def test_mint_error_starts_cooldown(self):
        """A failed mint must not retry on every spawn while GitHub is down —
        each retry costs a live HTTP timeout on the tool hot path."""
        with patch(MINT_TARGET, side_effect=RuntimeError("api down")) as mint:
            _sanitize_subprocess_env(_base_env())
            _sanitize_subprocess_env(_base_env())
        assert mint.call_count == 1

    def test_cooldown_expires(self):
        with patch(MINT_TARGET, side_effect=RuntimeError("api down")) as mint:
            _sanitize_subprocess_env(_base_env())
        local_mod._GH_TOKEN_MINT_FAILURE_TS = (
            time.time() - local_mod._GH_TOKEN_MINT_COOLDOWN_SECONDS - 1
        )
        with patch(MINT_TARGET, return_value="ghs_recovered") as mint2:
            env = _sanitize_subprocess_env(_base_env())
        assert env.get("GH_TOKEN") == "ghs_recovered"
        assert mint2.call_count == 1

    def test_none_result_does_not_start_cooldown(self):
        """No-creds is a cheap file read, not a failure — never cool down on
        it, or an agent that gains App creds mid-session waits a minute."""
        with patch(MINT_TARGET, return_value=None) as mint:
            _sanitize_subprocess_env(_base_env())
            _sanitize_subprocess_env(_base_env())
        assert mint.call_count == 2

    def test_missing_home_skips_quietly(self):
        with patch(MINT_TARGET, return_value="ghs_minted") as mint:
            env = _sanitize_subprocess_env({"PATH": "/usr/bin"})
        # apply_subprocess_home_env may synthesize a HOME; only assert we
        # didn't crash and the mint layer got a real dir if it was called.
        if mint.called:
            assert mint.call_args.args[0] != Path(".")
        assert isinstance(env, dict)


class TestPrecedence:
    def test_forced_gh_token_wins_over_mint(self):
        """An operator force-passing GH_TOKEN (force prefix) is explicit
        intent — the mint must not clobber it."""
        from tools.environments.local import _HERMES_PROVIDER_ENV_FORCE_PREFIX

        extra = {_HERMES_PROVIDER_ENV_FORCE_PREFIX + "GH_TOKEN": "ghp_forced"}
        with patch(MINT_TARGET, return_value="ghs_minted") as mint:
            env = _sanitize_subprocess_env(_base_env(), extra_env=extra)
        assert env.get("GH_TOKEN") == "ghp_forced"
        mint.assert_not_called()

    def test_opt_out_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_GH_TOKEN_INJECT", "off")
        with patch(MINT_TARGET, return_value="ghs_minted") as mint:
            env = _sanitize_subprocess_env(_base_env())
        assert "GH_TOKEN" not in env
        mint.assert_not_called()
