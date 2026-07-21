"""Tests for hermes_cli.github_app_token — GitHub App installation-token minting.

Mints a short-lived (1h) installation token from a NimbleCoOrg GitHub App so
fleet agents auth to org repos without a long-lived PAT. Runs in TWO contexts:
at container boot (reads the agent .env directly) and as a git credential helper
inside the tool subprocess (where the App private key is blocklisted from env,
so it MUST read the .env file, not os.environ).

Tests use an in-test RSA keypair (via `cryptography`) and a fake GitHub API —
no network, no real key.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pytest

from hermes_cli import github_app_token as gat


# ---------------------------------------------------------------------------
# Fixtures: a throwaway RSA keypair, PEM + base64 form
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    return key, pem


@pytest.fixture
def app_env(tmp_path: Path, rsa_keypair) -> Path:
    """A HERMES_HOME-shaped dir whose .env carries the three App vars."""
    _key, pem = rsa_keypair
    b64 = base64.b64encode(pem.encode("utf-8")).decode("ascii")
    home = tmp_path / "agent"
    home.mkdir()
    (home / ".env").write_text(
        f"GITHUB_APP_ID=123456\n"
        f"GITHUB_APP_INSTALLATION_ID=987654\n"
        f"GITHUB_APP_PRIVATE_KEY_B64={b64}\n"
    )
    return home


# ---------------------------------------------------------------------------
# read_app_credentials + decode_private_key
# ---------------------------------------------------------------------------


def test_read_app_credentials_parses_all_three(app_env, rsa_keypair):
    _key, pem = rsa_keypair
    creds = gat.read_app_credentials(app_env / ".env")
    assert creds is not None
    assert creds.app_id == "123456"
    assert creds.installation_id == "987654"
    assert creds.private_key_pem.strip() == pem.strip()


def test_read_app_credentials_none_when_app_id_missing(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("GITHUB_APP_PRIVATE_KEY_B64=eHg=\nGITHUB_APP_INSTALLATION_ID=1\n")
    assert gat.read_app_credentials(env) is None


def test_read_app_credentials_none_when_key_missing(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("GITHUB_APP_ID=1\nGITHUB_APP_INSTALLATION_ID=1\n")
    assert gat.read_app_credentials(env) is None


def test_read_app_credentials_installation_optional(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("GITHUB_APP_ID=1\nGITHUB_APP_PRIVATE_KEY_B64=eHg=\n")
    creds = gat.read_app_credentials(env)
    assert creds is not None and creds.installation_id is None


def test_decode_private_key_roundtrips(rsa_keypair):
    _key, pem = rsa_keypair
    b64 = base64.b64encode(pem.encode("utf-8")).decode("ascii")
    assert gat.decode_private_key(b64).strip() == pem.strip()


# ---------------------------------------------------------------------------
# build_app_jwt — verifiable with the public key
# ---------------------------------------------------------------------------


def test_build_app_jwt_claims_and_alg(rsa_keypair):
    import jwt as pyjwt

    key, pem = rsa_keypair
    now = 1_700_000_000
    token = gat.build_app_jwt("123456", pem, now=now)
    header = pyjwt.get_unverified_header(token)
    assert header["alg"] == "RS256"
    pub = key.public_key()
    decoded = pyjwt.decode(token, pub, algorithms=["RS256"], options={"verify_exp": False})
    assert decoded["iss"] == "123456"
    assert decoded["iat"] == now - 60
    assert decoded["exp"] == now + 600


# ---------------------------------------------------------------------------
# mint_installation_token — fake GitHub API
# ---------------------------------------------------------------------------


def test_mint_uses_bearer_jwt_and_returns_token(app_env, rsa_keypair, monkeypatch):
    calls = {}

    class FakeResp:
        status_code = 201

        def json(self):
            return {"token": "ghs_minted", "expires_at": "2026-01-01T00:00:00Z"}

    def fake_post(url, headers=None, timeout=None):
        calls["url"] = url
        calls["auth"] = headers.get("Authorization", "")
        return FakeResp()

    monkeypatch.setattr(gat.httpx, "post", fake_post)
    creds = gat.read_app_credentials(app_env / ".env")
    token, expires = gat.mint_installation_token(creds)

    assert token == "ghs_minted"
    assert expires == "2026-01-01T00:00:00Z"
    assert "987654/access_tokens" in calls["url"]
    assert calls["auth"].startswith("Bearer ")


def test_resolve_installation_id_discovers_when_absent(tmp_path: Path, rsa_keypair, monkeypatch):
    _key, pem = rsa_keypair
    b64 = base64.b64encode(pem.encode("utf-8")).decode("ascii")
    env = tmp_path / ".env"
    env.write_text(f"GITHUB_APP_ID=1\nGITHUB_APP_PRIVATE_KEY_B64={b64}\n")

    class FakeResp:
        status_code = 200

        def json(self):
            return [{"id": 555}]

    monkeypatch.setattr(gat.httpx, "get", lambda url, headers=None, timeout=None: FakeResp())
    creds = gat.read_app_credentials(env)
    jwt_tok = gat.build_app_jwt(creds.app_id, creds.private_key_pem)
    assert gat.resolve_installation_id(jwt_tok, creds) == "555"


# ---------------------------------------------------------------------------
# get_installation_token — cache-first with expiry
# ---------------------------------------------------------------------------


def _write_cache(home: Path, token: str, expires_epoch: float) -> None:
    import datetime as dt

    cache = gat.cache_path(home)
    cache.parent.mkdir(parents=True, exist_ok=True)
    iso = dt.datetime.fromtimestamp(expires_epoch, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    cache.write_text(json.dumps({"token": token, "expires_at": iso}))


def test_cache_hit_when_far_from_expiry(app_env, monkeypatch):
    _write_cache(app_env, "ghs_cached", time.time() + 3600)

    def boom(*a, **k):
        raise AssertionError("must not mint on a fresh cache hit")

    monkeypatch.setattr(gat.httpx, "post", boom)
    assert gat.get_installation_token(app_env) == "ghs_cached"


def test_remint_when_within_margin(app_env, monkeypatch):
    _write_cache(app_env, "ghs_stale", time.time() + 120)  # < 600s margin

    class FakeResp:
        status_code = 201

        def json(self):
            return {"token": "ghs_fresh", "expires_at": "2099-01-01T00:00:00Z"}

    monkeypatch.setattr(gat.httpx, "post", lambda *a, **k: FakeResp())
    assert gat.get_installation_token(app_env) == "ghs_fresh"


def test_remint_when_cache_corrupt(app_env, monkeypatch):
    cache = gat.cache_path(app_env)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{not json")

    class FakeResp:
        status_code = 201

        def json(self):
            return {"token": "ghs_recovered", "expires_at": "2099-01-01T00:00:00Z"}

    monkeypatch.setattr(gat.httpx, "post", lambda *a, **k: FakeResp())
    assert gat.get_installation_token(app_env) == "ghs_recovered"


def test_cache_written_0600(app_env, monkeypatch):
    import stat

    class FakeResp:
        status_code = 201

        def json(self):
            return {"token": "ghs_x", "expires_at": "2099-01-01T00:00:00Z"}

    monkeypatch.setattr(gat.httpx, "post", lambda *a, **k: FakeResp())
    gat.get_installation_token(app_env, force=True)
    mode = stat.S_IMODE(gat.cache_path(app_env).stat().st_mode)
    assert mode == 0o600


# ---------------------------------------------------------------------------
# get-credential CLI — git credential-helper protocol
# ---------------------------------------------------------------------------


def test_get_credential_stdout_format(app_env, monkeypatch, capsys):
    monkeypatch.setattr(gat, "get_installation_token", lambda home, force=False: "ghs_helper")
    rc = gat.main(["get-credential", "--home", str(app_env)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "username=x-access-token\n" in out
    assert "password=ghs_helper\n" in out
    assert out.endswith("\n\n")


def test_get_credential_accepts_git_get_operation(app_env, monkeypatch, capsys):
    # git invokes the helper as `<helper> get` — the trailing operation must not
    # crash argparse (regression: "unrecognized arguments: get").
    monkeypatch.setattr(gat, "get_installation_token", lambda home, force=False: "ghs_helper")
    rc = gat.main(["get-credential", "--home", str(app_env), "get"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "password=ghs_helper\n" in out


def test_get_credential_store_and_erase_are_noops(app_env, monkeypatch, capsys):
    # store/erase carry no output and must never mint.
    def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("mint attempted on store/erase")

    monkeypatch.setattr(gat, "get_installation_token", boom)
    for op in ("store", "erase"):
        assert gat.main(["get-credential", "--home", str(app_env), op]) == 0
        assert capsys.readouterr().out == ""
