"""``check_image_generation_requirements`` must agree with what dispatch will do.

Regression cover for behaviour that is already correct on main (fixed in
``e13104cfd``). An audit flagged this function as a live defect after reading a
working tree 23 commits behind main; the behaviour was fine, the *coverage* was not.

There WAS a pre-existing test —
``tests/tools/test_image_generation_plugin_dispatch.py::test_requirements_ignore_unselected_paid_plugin``
— and that is the interesting part: **it passes against the buggy body.** It stubs
``check_fal_api_key`` and ``_read_configured_image_provider`` but never stubs
``_ensure_plugins_discovered`` and never registers a ready provider, so in a keyless
environment all eight real providers report ``is_available() == False`` and the old
any-provider implementation returned ``False`` too. Verified: with the genuine
pre-``e13104cfd`` body spliced in, that file is 6 passed while this one is 3 failed.
A test that cannot fail is not coverage.

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
A capability that is advertised but non-functional produces false claims about what
the agent can do, which is worse than not having it.

These tests pin the check to the dispatch chain, including both halves of the
``if not configured or configured == "fal"`` short-circuit that ``e13104cfd`` added,
and the FAL-path behaviour that must NOT be narrowed.

Mutation-tested against ``tools/image_generation_tool.py``: deleting the
short-circuit, dropping ``== "fal"``, replacing ``get_provider(configured)`` with an
any-provider scan, dropping ``.is_available()``, dropping the ``_load_fal_client``
SDK probe, and making the FAL branch unconditional all turn this file red.

One mutation does NOT: narrowing the guard to ``if configured == "fal"`` alone. That
is not a coverage gap — ``get_provider`` returns None for any non-``str`` name
(``agent/image_gen_registry.py:69-70``), so an unset provider can never resolve and
the ``not configured`` half cannot change the return value. It is redundant defence,
and no test can distinguish it. Said here so nobody "fixes" the gap with a test that
asserts nothing.
"""
from __future__ import annotations

import importlib

import pytest


def _resolve():
    """Resolve the modules under test AT CALL TIME, and build the fake against
    whatever ``ImageGenProvider`` is currently live.

    ``tests/agent/test_empty_tool_name_loop_dampening.py`` purges ``agent.*`` and
    ``tools.*`` from ``sys.modules`` without restoring them. Module-level imports
    here would then bind to a *different* module object than the one
    ``check_image_generation_requirements`` re-imports when called — and a fake
    subclassing the stale ABC fails ``register_provider``'s ``isinstance`` check
    (``agent/image_gen_registry.py:43``). Resolving both together avoids it.

    CI runs each file in its own subprocess so this never bit there, which is
    exactly why it would have sat unnoticed.
    """
    igt = importlib.import_module("tools.image_generation_tool")
    registry = importlib.import_module("agent.image_gen_registry")
    base = importlib.import_module("agent.image_gen_provider").ImageGenProvider

    class _FakeProvider(base):  # type: ignore[misc,valid-type]
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

    return igt, registry, _FakeProvider


@pytest.fixture()
def env():
    """(igt, registry, FakeProvider) with a clean registry either side."""
    igt, registry, fake = _resolve()
    registry._reset_for_tests()
    yield igt, registry, fake
    registry._reset_for_tests()


@pytest.fixture()
def no_fal(env, monkeypatch):
    """No in-tree FAL backend, and no real plugin discovery.

    Discovery must be stubbed: it registers the eight real providers, and
    ``register_provider`` overwrites by name — so a fake called "openai" is
    silently replaced by the real one, whose ``is_available()`` is False without
    a key. That makes assertions pass for the wrong reason. Verified: with the
    buggy body AND this stub removed, the regression test passes.
    """
    igt, _registry, _fake = env
    plugins = importlib.import_module("hermes_cli.plugins")
    monkeypatch.setattr(igt, "check_fal_api_key", lambda: False)
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: None)
    return env


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
        igt, image_gen_registry, _FakeProvider = no_fal
        _set_configured_provider(monkeypatch, igt, None)
        image_gen_registry.register_provider(_FakeProvider("openai", available=True))

        assert igt.check_image_generation_requirements() is False

    def test_selected_and_available_provider_is_available(self, no_fal, monkeypatch):
        igt, image_gen_registry, _FakeProvider = no_fal
        _set_configured_provider(monkeypatch, igt, "openai")
        image_gen_registry.register_provider(_FakeProvider("openai", available=True))

        assert igt.check_image_generation_requirements() is True

    def test_selected_but_unavailable_provider_is_not_available(
        self, no_fal, monkeypatch
    ):
        igt, image_gen_registry, _FakeProvider = no_fal
        _set_configured_provider(monkeypatch, igt, "openai")
        image_gen_registry.register_provider(_FakeProvider("openai", available=False))

        assert igt.check_image_generation_requirements() is False

    def test_selected_provider_not_registered_is_not_available(
        self, no_fal, monkeypatch
    ):
        """``image_gen.provider`` names a provider that never registered — the
        plugin is missing or failed to load. Dispatch returns an error for this;
        the tool should not be advertised."""
        igt, image_gen_registry, _FakeProvider = no_fal
        _set_configured_provider(monkeypatch, igt, "nonexistent")
        image_gen_registry.register_provider(_FakeProvider("openai", available=True))

        assert igt.check_image_generation_requirements() is False

    def test_configured_fal_short_circuits_even_with_a_ready_fal_provider(
        self, no_fal, monkeypatch
    ):
        """`image_gen.provider = "fal"` routes to the IN-TREE FAL path, not the
        plugin registry — so a ready plugin named "fal" must not satisfy the
        check when the in-tree path itself is unavailable.

        This is the test that pins the `configured == "fal"` half of the
        short-circuit. Without it, deleting that clause (or the whole
        short-circuit) leaves the suite green — the one line e13104cfd added
        would be the one line nothing defended.
        """
        igt, image_gen_registry, _FakeProvider = no_fal
        _set_configured_provider(monkeypatch, igt, "fal")
        image_gen_registry.register_provider(_FakeProvider("fal", available=True))

        assert igt.check_image_generation_requirements() is False

    def test_a_different_provider_being_ready_does_not_count(
        self, no_fal, monkeypatch
    ):
        """Selecting one provider must not be satisfied by a *different* one
        being ready — the exact 'any provider' bug, one step subtler."""
        igt, image_gen_registry, _FakeProvider = no_fal
        _set_configured_provider(monkeypatch, igt, "krea")
        image_gen_registry.register_provider(_FakeProvider("openai", available=True))

        assert igt.check_image_generation_requirements() is False


class TestFalPathUnchanged:
    def test_fal_key_alone_is_enough(self, env, monkeypatch):
        """Shipping with a FAL key must still expose the tool, with no plugin
        registered and no ``image_gen.provider`` set. This is the historical
        behaviour and the fix must not narrow it."""
        igt, _registry, _fake = env
        monkeypatch.setattr(igt, "check_fal_api_key", lambda: True)
        monkeypatch.setattr(igt, "_load_fal_client", lambda: object())
        _set_configured_provider(monkeypatch, igt, None)

        assert igt.check_image_generation_requirements() is True

    def test_fal_key_without_sdk_falls_through_to_plugin_rules(self, env, monkeypatch):
        """``_load_fal_client`` raising ImportError (optional dep absent) must
        not short-circuit to True — it continues to the plugin rules, which
        without a selected provider means False."""
        igt, _registry, _fake = env
        plugins = importlib.import_module("hermes_cli.plugins")

        def _boom():
            raise ImportError("fal-client not installed")

        monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: None)
        monkeypatch.setattr(igt, "check_fal_api_key", lambda: True)
        monkeypatch.setattr(igt, "_load_fal_client", _boom)
        _set_configured_provider(monkeypatch, igt, None)

        assert igt.check_image_generation_requirements() is False
