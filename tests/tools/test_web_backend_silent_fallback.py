"""A discarded ``web.*_backend`` config value must not be dropped silently.

Real incident: a fleet had ``web.search_backend: brave`` set. The registered
provider name is ``brave-free``, so the value was discarded and a *different*
provider (firecrawl) served every search on every agent, indefinitely, with
nothing logged at any level. The config file and the runtime disagreed and
nothing said so — the only symptom was "search returns empty", which is
indistinguishable from a provider outage.

Reporting happens at ONE place — the config layer, in
``_warn_discarded_backend``, which knows which key it read. It covers
``web.backend`` (the key ``hermes tools`` actually writes) as well as the
per-capability keys.

The classification must not lie, and two earlier cuts of this change did:

- a private duplicate of ``_LEGACY_WEB_BACKENDS`` reported *registered
  third-party providers* as typos, with a "valid values" list excluding the
  user's own provider;
- a dispatch-layer warner named ``web.{capability}_backend`` even when the value
  came from ``web.backend``, and asserted "the credential is set" — a fact
  ``_resolve_backend`` never establishes, since it returns any legacy-or-
  registered name without probing availability. It has been removed rather
  than repaired: every bundled provider supports search, and the extract
  dispatcher already emits a precise typed error for a real capability
  mismatch, so it was near-dead as well as wrong.

These tests pin the behaviour that prevents both.
"""
from __future__ import annotations

import logging

import pytest

from tools import web_tools


# NOTE: the dedup-set reset is an autouse fixture in tests/tools/conftest.py so
# it protects every test in this directory, not just this file.


@pytest.fixture
def cfg(monkeypatch):
    def _apply(**web_config):
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: dict(web_config))
    return _apply


@pytest.fixture
def no_registry(monkeypatch):
    """No plugin-registered providers, so only built-ins are recognised."""
    monkeypatch.setattr(web_tools, "_registered_web_provider", lambda b: None)
    monkeypatch.setattr(web_tools, "_list_registered_web_providers", lambda: [])


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# The original incident: a typo in a per-capability key
# ---------------------------------------------------------------------------

class TestTypoInCapabilityKey:
    def test_reports_typo_and_lists_valid_values(self, cfg, no_registry, caplog, monkeypatch):
        cfg(search_backend="brave")
        monkeypatch.setattr(web_tools, "_resolve_backend", lambda: "firecrawl")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        assert web_tools._get_capability_backend("search") == "firecrawl"

        msgs = " ".join(_warnings(caplog))
        assert "search_backend='brave'" in msgs
        assert "not a known backend" in msgs
        assert "NOT being used" in msgs
        assert "brave-free" in msgs, "the valid-values list must name the real provider"


# ---------------------------------------------------------------------------
# Audit S1: web.backend is the key `hermes tools` writes — it must be covered
# ---------------------------------------------------------------------------

class TestSharedBackendKeyIsCovered:
    def test_typo_in_web_backend_is_reported(self, cfg, no_registry, caplog):
        """`hermes tools` writes ONLY web.backend, never the capability keys."""
        cfg(backend="brave")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        resolved = web_tools._get_backend()

        assert resolved != "brave"
        msgs = " ".join(_warnings(caplog))
        assert "backend='brave'" in msgs, (
            "a typo in web.backend was previously 100% silent, and it is the "
            "key the setup tool actually writes"
        )


# ---------------------------------------------------------------------------
# Audit B2: a registered third-party provider is NOT a typo
# ---------------------------------------------------------------------------

class TestRegisteredProviderIsNeverCalledATypo:
    def test_plugin_provider_missing_credential_is_not_a_typo(self, cfg, caplog, monkeypatch):
        class MyCorp:
            name = "mycorp"

            def is_available(self):
                return False

        cfg(search_backend="mycorp")
        monkeypatch.setattr(
            web_tools, "_registered_web_provider",
            lambda b: MyCorp() if b == "mycorp" else None,
        )
        monkeypatch.setattr(web_tools, "_list_registered_web_providers", lambda: [MyCorp()])
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda b: False)
        monkeypatch.setattr(web_tools, "_resolve_backend", lambda: "brave-free")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        web_tools._get_capability_backend("search")

        msgs = " ".join(_warnings(caplog))
        assert "recognised backend but is not available" in msgs, (
            "'mycorp' IS registered — telling the operator it does not exist "
            "would send them to change a correct config"
        )
        assert "not a known backend" not in msgs

    def test_valid_values_includes_registered_providers(self, monkeypatch):
        class MyCorp:
            name = "mycorp"

        monkeypatch.setattr(web_tools, "_list_registered_web_providers", lambda: [MyCorp()])
        names = web_tools._valid_backend_names()
        assert "mycorp" in names, (
            "a valid-values list that omits the user's own provider would tell "
            "them to switch to the wrong backend"
        )
        assert "brave-free" in names


# ---------------------------------------------------------------------------
# Audit B1: silence when the operator configured nothing
# ---------------------------------------------------------------------------

class TestNoConfigMeansNoBlame:
    def test_empty_config_produces_no_warning(self, cfg, no_registry, caplog):
        """A fresh install with no keys must not be told to hunt for a typo."""
        cfg()
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        web_tools._get_capability_backend("search")
        web_tools._get_backend()

        assert not _warnings(caplog), (
            "nothing was configured, so nothing was discarded — warning here "
            "blames the operator for an auto-detected value and contradicts "
            "the tool's own 'No web search provider configured' error"
        )

    def test_available_backend_is_silent(self, cfg, no_registry, caplog, monkeypatch):
        cfg(search_backend="firecrawl")
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda b: True)
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        assert web_tools._get_capability_backend("search") == "firecrawl"
        assert not _warnings(caplog)

    def test_no_warning_when_fallback_equals_request(self, cfg, no_registry, caplog, monkeypatch):
        cfg(search_backend="firecrawl")
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda b: False)
        monkeypatch.setattr(web_tools, "_resolve_backend", lambda: "firecrawl")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        web_tools._get_capability_backend("search")
        assert not _warnings(caplog)


# ---------------------------------------------------------------------------
# Dedup — these run on every dispatch and every `hermes tools` repaint
# ---------------------------------------------------------------------------

class TestDedup:
    def test_warns_once_not_per_dispatch(self, cfg, no_registry, caplog, monkeypatch):
        cfg(search_backend="brave")
        monkeypatch.setattr(web_tools, "_resolve_backend", lambda: "firecrawl")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        for _ in range(25):
            web_tools._get_capability_backend("search")

        assert len(_warnings(caplog)) == 1, (
            "un-deduped, this becomes the spam it exists to surface"
        )

    def test_search_and_extract_report_independently(self, cfg, no_registry, caplog, monkeypatch):
        cfg(search_backend="brave", extract_backend="brave")
        monkeypatch.setattr(web_tools, "_resolve_backend", lambda: "firecrawl")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        web_tools._get_capability_backend("search")
        web_tools._get_capability_backend("extract")

        msgs = _warnings(caplog)
        assert len(msgs) == 2
        assert any("search_backend" in m for m in msgs)
        assert any("extract_backend" in m for m in msgs)


# ---------------------------------------------------------------------------
# No duplicated constant (audit S2)
# ---------------------------------------------------------------------------

class TestValidValuesRespectCapability:
    """A suggestion list must not name backends that cannot do the job."""

    def test_extract_list_excludes_search_only_backends(self, no_registry):
        names = web_tools._valid_backend_names("extract")
        for search_only in ("brave-free", "ddgs", "searxng", "xai"):
            assert search_only not in names, (
                f"{search_only} cannot extract — suggesting it swaps one "
                "broken config for another, and contradicts the extract "
                "dispatcher's own 'firecrawl, tavily, exa, or parallel'"
            )
        assert {"firecrawl", "tavily", "exa", "parallel"} <= set(names)

    def test_search_list_includes_search_only_backends(self, no_registry):
        assert "brave-free" in web_tools._valid_backend_names("search")

    def test_no_dispatch_layer_warner(self):
        """Deliberately removed: it misstated the config key and invented a
        credential fact. Misconfiguration is reported at the config layer."""
        assert not hasattr(web_tools, "_warn_dispatch_capability_mismatch")
