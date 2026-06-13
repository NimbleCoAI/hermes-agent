"""Focused tests for EvoLink provider profile wiring."""

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
from hermes_cli.config import OPTIONAL_ENV_VARS
from hermes_cli.models import _PROVIDER_LABELS, normalize_provider, provider_model_ids


def test_evolink_profile_auto_wires_core_surfaces(monkeypatch):
    monkeypatch.delenv("EVOLINK_API_KEY", raising=False)
    monkeypatch.delenv("EVOLINK_BASE_URL", raising=False)

    from agent.model_metadata import _URL_TO_PROVIDER
    from providers import get_provider_profile

    profile = get_provider_profile("evolink")
    assert profile is not None
    assert profile.base_url == "https://direct.evolink.ai/v1"
    assert profile.default_aux_model == "gpt-5.2"
    assert profile.fallback_models[0] == "gpt-5.2"

    assert "evolink" in PROVIDER_REGISTRY
    pconfig = PROVIDER_REGISTRY["evolink"]
    assert pconfig.name == "EvoLink"
    assert pconfig.api_key_env_vars == ("EVOLINK_API_KEY",)
    assert pconfig.base_url_env_var == "EVOLINK_BASE_URL"
    assert pconfig.inference_base_url == "https://direct.evolink.ai/v1"

    assert OPTIONAL_ENV_VARS["EVOLINK_API_KEY"]["category"] == "provider"
    assert OPTIONAL_ENV_VARS["EVOLINK_API_KEY"]["password"] is True
    assert OPTIONAL_ENV_VARS["EVOLINK_BASE_URL"]["password"] is False

    assert _PROVIDER_LABELS["evolink"] == "EvoLink"
    assert normalize_provider("evolink") == "evolink"
    assert resolve_provider("evolink") == "evolink"
    assert _URL_TO_PROVIDER["direct.evolink.ai"] == "evolink"


def test_evolink_provider_model_ids_falls_back_to_profile_models(monkeypatch):
    monkeypatch.delenv("EVOLINK_API_KEY", raising=False)

    assert provider_model_ids("evolink")[:3] == [
        "gpt-5.2",
        "gpt-5.1",
        "gemini-3.1-flash-lite-preview",
    ]
