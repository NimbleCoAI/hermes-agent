"""GLM-5.2 (and 4.6) must be recognised model ids on the first-class `zai` provider.

HSM's cascade editor emits `{provider: "zai", model: "glm-5.2"}`; the runtime
`zai` plugin should carry those ids in its fallback catalog so model selection
and suggestion surfaces know about them. The endpoint (Z.ai cloud, or a future
self-hosted GPU box) is configured via base_url — this only asserts the catalog.
"""

from __future__ import annotations

from providers import get_provider_profile


def _zai():
    profile = get_provider_profile("zai")
    assert profile is not None, "zai provider plugin should be registered"
    return profile


def test_zai_profile_lists_glm_5_2():
    assert "glm-5.2" in _zai().fallback_models


def test_zai_profile_lists_glm_4_6():
    assert "glm-4.6" in _zai().fallback_models


def test_zai_aliases_cover_glm_naming():
    # HSM/users may say "glm"/"zhipu"; the plugin must resolve those to zai.
    aliases = _zai().aliases
    assert "glm" in aliases
    assert "zhipu" in aliases


def test_zai_authenticates_via_glm_key():
    # The canonical credential HSM writes is GLM_API_KEY (plugin also accepts ZAI_*).
    assert "GLM_API_KEY" in _zai().env_vars
