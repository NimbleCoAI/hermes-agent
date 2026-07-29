"""A discarded ``web.*_backend`` config value must not be dropped silently.

Real incident: a fleet had ``web.search_backend: brave`` set. The registered
provider name is ``brave-free``, so ``_is_backend_available("brave")`` fell
through every branch, returned False, and ``_get_capability_backend`` quietly
auto-detected a *different* provider (firecrawl). Every web_search on every
agent was served by a provider nobody configured, indefinitely, with nothing in
the logs at any level. The config file and the actual behavior disagreed and
nothing said so — the only symptom was "search returns empty".

These tests pin that an explicitly-configured-but-discarded backend is reported
once, and that the two distinct operator mistakes are distinguishable:

- unknown name (a typo) -> says it is unrecognised, lists valid values
- known name, missing credential -> says the credential is unset

They also pin the dedup, because ``_get_capability_backend`` runs on every
dispatch and on every ``hermes tools`` repaint.
"""
from __future__ import annotations

import logging

import pytest

from tools import web_tools


@pytest.fixture(autouse=True)
def _reset_warn_dedup():
    web_tools._backend_fallback_warned.clear()
    yield
    web_tools._backend_fallback_warned.clear()


@pytest.fixture
def cfg(monkeypatch):
    """Control the web config and the auto-detect result independently."""
    def _apply(configured: str, resolved: str = "firecrawl"):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"search_backend": configured, "extract_backend": configured},
        )
        monkeypatch.setattr(web_tools, "_get_backend", lambda: resolved)
    return _apply


class TestUnknownBackendName:
    def test_typo_warns_and_names_valid_values(self, cfg, caplog):
        cfg("brave", resolved="firecrawl")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        got = web_tools._get_capability_backend("search")

        assert got == "firecrawl", "the fallback still happens — behavior unchanged"
        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert msgs, "an unrecognised configured backend must be reported"
        joined = " ".join(msgs)
        assert "brave" in joined
        assert "not a recognised backend" in joined
        assert "brave-free" in joined, "the message should list valid values"

    def test_message_says_configured_provider_is_not_used(self, cfg, caplog):
        """The operator's actual misconception is 'my config is in effect'."""
        cfg("brave", resolved="firecrawl")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")
        web_tools._get_capability_backend("search")
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "NOT being used" in joined


class TestKnownButUnavailable:
    def test_missing_credential_is_a_different_message(self, cfg, caplog, monkeypatch):
        """`tavily` is real but has no key — that is not a typo."""
        cfg("tavily", resolved="firecrawl")
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda b: False)
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        web_tools._get_capability_backend("search")

        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "known backend but is not available" in joined
        assert "not a recognised backend" not in joined, (
            "a real backend missing its key must not be reported as a typo"
        )


class TestNoSpuriousWarnings:
    def test_available_backend_is_silent(self, cfg, caplog, monkeypatch):
        cfg("firecrawl", resolved="firecrawl")
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda b: True)
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        assert web_tools._get_capability_backend("search") == "firecrawl"
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_unset_backend_is_silent(self, cfg, caplog):
        """No configured value means nothing was discarded — say nothing."""
        cfg("", resolved="firecrawl")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        assert web_tools._get_capability_backend("search") == "firecrawl"
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_no_warning_when_fallback_equals_request(self, cfg, caplog, monkeypatch):
        """If auto-detect lands on the same provider, nothing was lost."""
        cfg("firecrawl", resolved="firecrawl")
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda b: False)
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        web_tools._get_capability_backend("search")
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestDedup:
    def test_warns_once_not_per_dispatch(self, cfg, caplog):
        cfg("brave", resolved="firecrawl")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        for _ in range(25):
            web_tools._get_capability_backend("search")

        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 1, (
            "this runs on every dispatch and every `hermes tools` repaint; "
            f"an un-deduped warning becomes spam. Got {len(warns)}."
        )

    def test_search_and_extract_report_independently(self, cfg, caplog):
        cfg("brave", resolved="firecrawl")
        caplog.set_level(logging.WARNING, logger="tools.web_tools")

        web_tools._get_capability_backend("search")
        web_tools._get_capability_backend("extract")

        warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warns) == 2, "each capability is a separate misconfiguration"
        assert any("search_backend" in m for m in warns)
        assert any("extract_backend" in m for m in warns)


class TestKnownBackendsStaysInSync:
    """_KNOWN_BACKENDS only exists to classify the error; drift makes it lie."""

    def test_every_known_backend_is_probed_by_is_backend_available(self):
        import inspect
        src = inspect.getsource(web_tools._is_backend_available)
        missing = [b for b in web_tools._KNOWN_BACKENDS if repr(b) not in src and f'"{b}"' not in src]
        assert not missing, (
            f"_KNOWN_BACKENDS lists {missing} but _is_backend_available does not "
            "probe them — a real backend would be misreported as a typo."
        )

    def test_no_unknown_name_claims_to_be_known(self):
        assert "brave" not in web_tools._KNOWN_BACKENDS, (
            "'brave' is the typo from the original incident; the registered "
            "provider is 'brave-free'. If an alias is ever added, it must be "
            "registered in the provider registry, not just listed here."
        )
