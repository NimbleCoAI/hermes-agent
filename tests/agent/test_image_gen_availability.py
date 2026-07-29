"""``check_image_generation_requirements`` must agree with what dispatch will do.

Regression cover for behaviour that is already correct on main (fixed in
``e13104cfd``) but had **no tests at all**. An audit flagged this function as a live
defect after reading a working tree 23 commits behind main; the behaviour was fine,
the coverage was not. These tests make the next such reading unnecessary — and would
fail loudly if the older logic were ever reintroduced by a merge.

The tool schema is gated on this function (``tools/registry.py``), while the actual
call is routed by ``_dispatch_to_plugin_provider``. Before ``e13104cfd`` those two
used different rules:

* the check returned True if **any** registered provider's ``is_available()`` was
  True;
* dispatch returns ``None`` — falling through to the in-tree FAL path — unless
  ``image_gen.provider`` is *explicitly* set (deliberately, so a user with
  ``OPENAI_API_KEY`` set for other features isn't silently billed for image gen).

So with FAL unconfigured, ``image_gen.provider`` unset, and some provider keyed,
``image_generate`` was advertised in the schema and then failed on every call.
A capability that is advertised but non-functional produces false claims about
what the agent can do, which is worse than not having it.

These tests pin the check to the dispatch chain, including the ``configured ==
"fal"`` short-circuit and the FAL-path behaviour that must NOT be narrowed.
"""
from __future__ import annotations

import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


class _FakeProvider(ImageGenProvider):
    def __init__(self, name: str, available: bool = True):
        self._name = name
        self._available = available

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt, aspect_ratio="landscape", **kw):
        return {"success": True, "image": f"{self._name}://{prompt}"}


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


@pytest.fixture()
def no_fal(monkeypatch):
    """No in-tree FAL backend, and no real plugin discovery.

    Discovery must be stubbed: it registers the eight real providers, and
    ``register_provider`` overwrites by name — so a fake called "openai" is
    silently replaced by the real one, whose ``is_available()`` is False without
    a key. That makes every assertion here pass for the wrong reason. Stubbing
    keeps the registry exactly as each test sets it.
    """
    import hermes_cli.plugins as plugins
    import tools.image_generation_tool as igt
    monkeypatch.setattr(igt, "check_fal_api_key", lambda: False)
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: None)
    return igt


def _set_configured_provider(monkeypatch, igt, value):
    """Stub ``image_gen.provider`` as read from config.yaml."""
    monkeypatch.setattr(igt, "_read_configured_image_provider", lambda: value)


class TestChecksMatchDispatch:
    def test_keyed_provider_but_provider_unset_is_not_available(
        self, no_fal, monkeypatch
    ):
        """The regression. A provider is registered and ready, but nothing
        selected it, so dispatch would fall through to the absent FAL path.
        The check must therefore say False rather than advertise the tool."""
        igt = no_fal
        _set_configured_provider(monkeypatch, igt, None)
        image_gen_registry.register_provider(_FakeProvider("openai", available=True))

        assert igt.check_image_generation_requirements() is False

    def test_selected_and_available_provider_is_available(self, no_fal, monkeypatch):
        igt = no_fal
        _set_configured_provider(monkeypatch, igt, "openai")
        image_gen_registry.register_provider(_FakeProvider("openai", available=True))

        assert igt.check_image_generation_requirements() is True

    def test_selected_but_unavailable_provider_is_not_available(
        self, no_fal, monkeypatch
    ):
        igt = no_fal
        _set_configured_provider(monkeypatch, igt, "openai")
        image_gen_registry.register_provider(_FakeProvider("openai", available=False))

        assert igt.check_image_generation_requirements() is False

    def test_selected_provider_not_registered_is_not_available(
        self, no_fal, monkeypatch
    ):
        """``image_gen.provider`` names a provider that never registered — the
        plugin is missing or failed to load. Dispatch returns an error for this;
        the tool should not be advertised."""
        igt = no_fal
        _set_configured_provider(monkeypatch, igt, "nonexistent")
        image_gen_registry.register_provider(_FakeProvider("openai", available=True))

        assert igt.check_image_generation_requirements() is False

    def test_a_different_provider_being_ready_does_not_count(
        self, no_fal, monkeypatch
    ):
        """Selecting one provider must not be satisfied by a *different* one
        being ready — the exact 'any provider' bug, one step subtler."""
        igt = no_fal
        _set_configured_provider(monkeypatch, igt, "krea")
        image_gen_registry.register_provider(_FakeProvider("openai", available=True))

        assert igt.check_image_generation_requirements() is False


class TestFalPathUnchanged:
    def test_fal_key_alone_is_enough(self, monkeypatch):
        """Shipping with a FAL key must still expose the tool, with no plugin
        registered and no ``image_gen.provider`` set. This is the historical
        behaviour and the fix must not narrow it."""
        import tools.image_generation_tool as igt
        monkeypatch.setattr(igt, "check_fal_api_key", lambda: True)
        monkeypatch.setattr(igt, "_load_fal_client", lambda: object())
        _set_configured_provider(monkeypatch, igt, None)

        assert igt.check_image_generation_requirements() is True

    def test_fal_key_without_sdk_falls_through_to_plugin_rules(self, monkeypatch):
        """``_load_fal_client`` raising ImportError (optional dep absent) must
        not short-circuit to True — it continues to the plugin rules, which
        without a selected provider means False."""
        import hermes_cli.plugins as plugins
        import tools.image_generation_tool as igt

        def _boom():
            raise ImportError("fal-client not installed")

        monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: None)
        monkeypatch.setattr(igt, "check_fal_api_key", lambda: True)
        monkeypatch.setattr(igt, "_load_fal_client", _boom)
        _set_configured_provider(monkeypatch, igt, None)

        assert igt.check_image_generation_requirements() is False
