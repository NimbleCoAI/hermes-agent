"""``request_overrides["extra_body"]`` must MERGE on both transport paths.

The intelligent_routing plugin ships ``routing.cheap_extra_body`` into
``agent.request_overrides["extra_body"]`` for a routed turn (see
``agent/routing_override.py``). The provider-profile path merges that dict into
the extra_body the transport already assembled; the legacy (no-profile) path
used ``api_kwargs.update(overrides)``, which REPLACED the assembled extra_body
wholesale — so a cheap tier pointed at an unregistered provider silently lost
its provider preferences and reasoning config.

Both paths must agree: routed keys win on collision, everything else survives.
"""

import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


MSGS = [{"role": "user", "content": "Hi"}]
PREFS = {"sort": "throughput"}
ROUTED_EXTRA_BODY = {"reasoning_effort": "none"}


def _legacy_kwargs(transport, **extra):
    """Legacy path: no provider_profile (custom / unregistered provider)."""
    return transport.build_kwargs(
        model="qwen3.5:9b",
        messages=MSGS,
        is_openrouter=True,
        provider_preferences=PREFS,
        supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "medium"},
        **extra,
    )


def _profile_kwargs(transport, **extra):
    from providers import get_provider_profile

    profile = get_provider_profile("openrouter")
    return transport.build_kwargs(
        model="deepseek/deepseek-v3.2",
        messages=MSGS,
        provider_profile=profile,
        provider_name="openrouter",
        base_url=profile.base_url,
        provider_preferences=PREFS,
        supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "medium"},
        **extra,
    )


class TestLegacyPathMerges:

    def test_baseline_assembles_provider_and_reasoning(self, transport):
        eb = _legacy_kwargs(transport)["extra_body"]
        assert eb["provider"] == PREFS
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_routed_extra_body_does_not_destroy_assembled_keys(self, transport):
        eb = _legacy_kwargs(
            transport, request_overrides={"extra_body": ROUTED_EXTRA_BODY}
        )["extra_body"]

        # The regression: these two used to vanish entirely.
        assert eb["provider"] == PREFS
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}
        assert eb["reasoning_effort"] == "none"

    def test_routed_value_wins_on_collision(self, transport):
        eb = _legacy_kwargs(
            transport,
            request_overrides={"extra_body": {"provider": {"sort": "price"}}},
        )["extra_body"]
        assert eb["provider"] == {"sort": "price"}
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_non_extra_body_overrides_still_land_top_level(self, transport):
        kw = _legacy_kwargs(
            transport,
            request_overrides={
                "service_tier": "priority",
                "extra_body": ROUTED_EXTRA_BODY,
            },
        )
        assert kw["service_tier"] == "priority"
        assert kw["extra_body"]["provider"] == PREFS

    def test_extra_body_only_from_overrides_when_nothing_assembled(self, transport):
        kw = transport.build_kwargs(
            model="qwen3.5:9b",
            messages=MSGS,
            request_overrides={"extra_body": ROUTED_EXTRA_BODY},
        )
        assert kw["extra_body"] == ROUTED_EXTRA_BODY

    def test_non_dict_extra_body_override_is_passed_through_verbatim(self, transport):
        """Malformed value: keep the old last-write-wins behaviour, don't crash."""
        kw = _legacy_kwargs(transport, request_overrides={"extra_body": "nope"})
        assert kw["extra_body"] == "nope"


class TestProfilePathParity:

    def test_profile_path_merges_the_same_way(self, transport):
        eb = _profile_kwargs(
            transport, request_overrides={"extra_body": ROUTED_EXTRA_BODY}
        )["extra_body"]
        assert eb["provider"] == PREFS
        assert eb["reasoning_effort"] == "none"
        assert "reasoning" in eb

    def test_both_paths_preserve_assembled_keys(self, transport):
        legacy = _legacy_kwargs(
            transport, request_overrides={"extra_body": ROUTED_EXTRA_BODY}
        )["extra_body"]
        profile = _profile_kwargs(
            transport, request_overrides={"extra_body": ROUTED_EXTRA_BODY}
        )["extra_body"]
        for eb in (legacy, profile):
            assert eb["provider"] == PREFS
            assert eb["reasoning_effort"] == "none"
