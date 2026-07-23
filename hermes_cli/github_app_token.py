"""GitHub App installation-token minting for fleet agents.

A NimbleCoOrg-owned GitHub App lets every agent authenticate to org repos with a
short-lived (~1h), auto-rotating installation token instead of a long-lived,
per-user PAT bound to a single resource owner. This module mints and caches that
token.

It runs in two contexts, and BOTH read the App credentials from the agent's
``.env`` FILE — never from ``os.environ``:

* at container boot (``git_credentials_boot``), where the ``.env`` on the mounted
  volume is readable and the tool subprocess hasn't started;
* as a git **credential helper** inside the tool subprocess, where the App
  private key is deliberately blocklisted from the environment
  (``tools/environments/local.py``), so a file read is the only path.

Credentials (all single-line, so they survive HSM's newline-rejecting ``.env``
writer):

* ``GITHUB_APP_ID``
* ``GITHUB_APP_PRIVATE_KEY_B64`` — the PEM, base64-encoded (the raw PEM is
  multi-line and cannot be stored by HSM)
* ``GITHUB_APP_INSTALLATION_ID`` — optional; auto-discovered if absent

Mint flow mirrors the proven pattern in ``tools/skills_hub.py``: sign a short
RS256 JWT (iss=app id, iat-60, exp+600) → ``POST /app/installations/{id}/
access_tokens`` → ``201 {token, expires_at}``.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import httpx

GITHUB_API = "https://api.github.com"

# Required + optional App vars in the agent .env.
_APP_ID_VAR = "GITHUB_APP_ID"
_APP_KEY_VAR = "GITHUB_APP_PRIVATE_KEY_B64"
_APP_INSTALL_VAR = "GITHUB_APP_INSTALLATION_ID"

# Re-mint when fewer than this many seconds remain, so a token never expires
# mid-operation (a `git push` may run minutes after `git credential fill`).
REMINT_MARGIN_SECONDS = 600

_JWT_LIFETIME_SECONDS = 600  # GitHub caps App JWTs at 10 min.
_HTTP_TIMEOUT = 10.0


@dataclass(frozen=True)
class AppCredentials:
    app_id: str
    private_key_pem: str
    installation_id: Optional[str]


# ---------------------------------------------------------------------------
# .env parsing (flat file, no shell expansion — matches git_credentials_boot)
# ---------------------------------------------------------------------------


def _read_env_var(content: str, key: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(key)}=(.*)$", content, re.MULTILINE)
    if not m:
        return None
    val = re.sub(r'^["\']|["\']$', "", m.group(1).strip())
    return val or None


def read_app_credentials(env_path: Path) -> Optional[AppCredentials]:
    """Return ``AppCredentials`` from ``env_path`` or ``None``.

    ``None`` (any required var absent) is the signal for callers to fall back to
    the static-PAT path — App auth is strictly opt-in per agent.
    """
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    app_id = _read_env_var(content, _APP_ID_VAR)
    key_b64 = _read_env_var(content, _APP_KEY_VAR)
    if not app_id or not key_b64:
        return None
    try:
        pem = decode_private_key(key_b64)
    except Exception:
        return None
    return AppCredentials(
        app_id=app_id,
        private_key_pem=pem,
        installation_id=_read_env_var(content, _APP_INSTALL_VAR),
    )


def decode_private_key(b64: str) -> str:
    """Base64-decode the stored PEM back to its multi-line text form."""
    return base64.b64decode(b64).decode("utf-8")


# ---------------------------------------------------------------------------
# JWT + mint
# ---------------------------------------------------------------------------


def build_app_jwt(app_id: str, private_key_pem: str, *, now: Optional[int] = None) -> str:
    """Sign a short-lived RS256 App JWT. ``now`` is injectable for tests."""
    import jwt  # PyJWT[crypto] — already in pyproject.toml

    ts = int(time.time()) if now is None else now
    payload = {"iat": ts - 60, "exp": ts + _JWT_LIFETIME_SECONDS, "iss": app_id}
    return jwt.encode(payload, private_key_pem, algorithm="RS256")


def _app_headers(jwt_token: str) -> dict:
    return {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github.v3+json",
    }


def resolve_installation_id(jwt_token: str, creds: AppCredentials) -> str:
    """Return the installation id, discovering it via the API if unset.

    An App installed on exactly one org has one installation; if the agent
    didn't pin ``GITHUB_APP_INSTALLATION_ID`` we take the first installation the
    App can see.
    """
    if creds.installation_id:
        return creds.installation_id
    resp = httpx.get(
        f"{GITHUB_API}/app/installations", headers=_app_headers(jwt_token), timeout=_HTTP_TIMEOUT
    )
    if resp.status_code != 200:
        raise RuntimeError(f"installation discovery failed: HTTP {resp.status_code}")
    installs = resp.json()
    if not installs:
        raise RuntimeError("GitHub App has no installations")
    return str(installs[0]["id"])


def mint_installation_token(creds: AppCredentials) -> Tuple[str, str]:
    """Mint a fresh installation token. Returns ``(token, expires_at_iso)``."""
    jwt_token = build_app_jwt(creds.app_id, creds.private_key_pem)
    installation_id = resolve_installation_id(jwt_token, creds)
    resp = httpx.post(
        f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
        headers=_app_headers(jwt_token),
        timeout=_HTTP_TIMEOUT,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"token mint failed: HTTP {resp.status_code}")
    body = resp.json()
    return body["token"], body["expires_at"]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_path(home_dir: Path) -> Path:
    """Cache file for one agent home. ``home_dir`` is the HERMES_HOME-shaped dir
    (holds ``.env``); the token cache lives under its ``home/`` subprocess HOME."""
    return home_dir / "home" / ".cache" / "hermes" / "github_app_token.json"


def _expires_epoch(iso: str) -> float:
    # GitHub returns e.g. "2026-01-01T00:00:00Z".
    return _dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=_dt.timezone.utc
    ).timestamp()


def _load_cache(path: Path) -> Optional[Tuple[str, float]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["token"], _expires_epoch(data["expires_at"])
    except Exception:
        return None  # absent or corrupt → caller re-mints


def _store_cache(path: Path, token: str, expires_at_iso: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"token": token, "expires_at": expires_at_iso}), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)  # atomic — a concurrent reader never sees a partial file


def get_installation_token(home_dir: Path, *, force: bool = False) -> Optional[str]:
    """Return a valid installation token for ``home_dir``, minting if needed.

    Cache-first: a cached token with more than ``REMINT_MARGIN_SECONDS`` of life
    left is returned as-is. Returns ``None`` only if the home has no App creds.
    """
    creds = read_app_credentials(home_dir / ".env")
    if creds is None:
        return None

    path = cache_path(home_dir)
    if not force:
        cached = _load_cache(path)
        if cached and cached[1] - time.time() > REMINT_MARGIN_SECONDS:
            return cached[0]

    token, expires_at = mint_installation_token(creds)
    _store_cache(path, token, expires_at)
    return token


# ---------------------------------------------------------------------------
# CLI — `python -m hermes_cli.github_app_token get-credential` IS the git helper
# ---------------------------------------------------------------------------


def _default_home() -> Path:
    """The HERMES_HOME-shaped dir: parent of the subprocess HOME.

    The credential helper runs with ``HOME={HERMES_HOME}/home``, so the dir
    holding ``.env`` is its parent. Overridable with ``--home`` (boot passes it).
    """
    return Path(os.environ.get("HOME", "/opt/data/home")).parent


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes_cli.github_app_token")
    sub = parser.add_subparsers(dest="cmd", required=True)
    # `get` alias: agents reach for the short form first (observed live
    # 2026-07-21 — a fleet agent fumbled the subcommand, got an argparse
    # error, and abandoned the helper for a broken workaround instead).
    gc = sub.add_parser(
        "get-credential",
        aliases=["get"],
        help="emit git credential-helper output (username/password lines)",
    )
    gc.add_argument("--home", type=Path, default=None)
    # git invokes a credential helper as ``<helper> <operation>`` where operation
    # is get/store/erase. Accept and ignore that trailing arg — without it argparse
    # errors ("unrecognized arguments: get") and git can never read the credential.
    gc.add_argument("op", nargs="?", default="get")
    args = parser.parse_args(argv)

    if args.cmd in ("get-credential", "get"):
        if args.op not in (None, "get"):
            return 0  # store/erase are no-ops for a mint-on-demand helper
        home = args.home or _default_home()
        token = get_installation_token(home)
        if not token:
            return 1  # no App creds → git falls through to any other helper
        # git credential protocol: key=value lines, blank line terminates.
        sys.stdout.write(f"username=x-access-token\npassword={token}\n\n")
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
