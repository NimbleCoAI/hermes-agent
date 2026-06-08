"""Tests for plugin capability declaration + enforcement.

A plugin can declare in ``plugin.yaml`` which tools it is permitted to dispatch::

    declared_capabilities:
      tools: [read_file, web_search]

When the ``tools`` capability is declared, ``PluginContext.dispatch_tool()`` must
refuse to dispatch any tool outside that set (fail-closed) and must NOT reach the
registry — this closes the lateral-movement path where a loaded plugin's own code
invokes arbitrary tools. When ``declared_capabilities`` is absent (the default),
dispatch is unrestricted, preserving backward compatibility for existing plugins.

Part of an inbound security boundary for shared/third-party plugin artifacts.
"""

import json
from unittest.mock import MagicMock, patch

from hermes_cli.plugins import PluginManifest, PluginContext, PluginManager


def _ctx(declared_capabilities=None) -> PluginContext:
    manifest = PluginManifest(
        name="testplugin",
        declared_capabilities=(
            declared_capabilities if declared_capabilities is not None else {}
        ),
    )
    manager = MagicMock()
    manager._cli_ref = None  # skip parent-agent wiring in dispatch_tool
    return PluginContext(manifest, manager)


class TestCapabilityEnforcement:
    def test_undeclared_capabilities_leave_dispatch_unrestricted(self):
        # No declared_capabilities -> pass through to the registry (backward compat).
        ctx = _ctx()
        with patch("tools.registry.registry") as reg:
            reg.dispatch.return_value = json.dumps({"ok": True})
            out = ctx.dispatch_tool("web_search", {"q": "x"})
        reg.dispatch.assert_called_once()
        assert json.loads(out) == {"ok": True}

    def test_declared_tool_is_allowed_through(self):
        ctx = _ctx({"tools": ["read_file", "web_search"]})
        with patch("tools.registry.registry") as reg:
            reg.dispatch.return_value = json.dumps({"ok": True})
            out = ctx.dispatch_tool("read_file", {"path": "/x"})
        reg.dispatch.assert_called_once()
        assert json.loads(out) == {"ok": True}

    def test_tool_outside_declared_set_is_blocked_before_registry(self):
        ctx = _ctx({"tools": ["read_file"]})
        with patch("tools.registry.registry") as reg:
            out = ctx.dispatch_tool("web_search", {"q": "x"})
        # Fail-closed: the registry must never be reached.
        reg.dispatch.assert_not_called()
        payload = json.loads(out)
        assert "error" in payload
        assert "web_search" in payload["error"]
        assert "testplugin" in payload["error"]

    def test_empty_tools_list_blocks_all_dispatch(self):
        # Declaring an explicit empty allow-list means "no tool dispatch".
        ctx = _ctx({"tools": []})
        with patch("tools.registry.registry") as reg:
            out = ctx.dispatch_tool("read_file", {"path": "/x"})
        reg.dispatch.assert_not_called()
        assert "error" in json.loads(out)


class TestManifestParsing:
    def test_manifest_defaults_to_empty_capabilities(self):
        m = PluginManifest(name="x")
        assert m.declared_capabilities == {}

    def test_declared_capabilities_parsed_from_yaml(self, tmp_path):
        manifest_file = tmp_path / "plugin.yaml"
        manifest_file.write_text(
            "name: capplugin\n"
            "declared_capabilities:\n"
            "  tools: [read_file, web_search]\n"
            "  network: false\n"
        )
        # _parse_manifest does not use instance state; call with a stand-in self.
        manifest = PluginManager._parse_manifest(
            MagicMock(), manifest_file, tmp_path, "user", ""
        )
        assert manifest is not None
        assert manifest.declared_capabilities == {
            "tools": ["read_file", "web_search"],
            "network": False,
        }
