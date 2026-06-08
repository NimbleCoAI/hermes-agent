import importlib
from plugins.gardener_approvals import config as C


def test_defaults(monkeypatch):
    for k in ("HERMES_GARDENER_APPROVALS_ENABLED", "HERMES_GARDENER_GATE",
              "HERMES_GARDENER_DECISIONS", "HERMES_GARDENER_APPROVER"):
        monkeypatch.delenv(k, raising=False)
    cfg = C.load_config()
    assert cfg.enabled is True
    assert cfg.gate_path == "/opt/data/scripts/decision-approve.sh"
    assert cfg.decisions_file == "/opt/data/gardener-memory/state/decisions.md"
    assert cfg.approver == "Juniper"


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("HERMES_GARDENER_APPROVALS_ENABLED", "0")
    assert C.load_config().enabled is False


def test_overrides(monkeypatch):
    monkeypatch.setenv("HERMES_GARDENER_GATE", "/custom/gate.sh")
    monkeypatch.setenv("HERMES_GARDENER_APPROVER", "Juni")
    cfg = C.load_config()
    assert cfg.gate_path == "/custom/gate.sh" and cfg.approver == "Juni"
