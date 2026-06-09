from plugins.gardener_approvals.decisions import read_open_ids, read_open_decisions, Decision

SAMPLE = """# Pending decisions

## D-2026-06-09-01 · First thing  — status: open
- **Decision:** a or b

## D-2026-06-09-02 · Second  — status: resolved
- **Resolved:** 2026-06-08

## D-2026-06-09-03 · Third — status: open
- **Decision:** x
"""

SAMPLE_WITH_PR = """# Pending decisions

## D-2026-06-09-01 · Add feature  — status: open
- **Decision:** ship it
- **Link:** NimbleCoAI/nimbleco-egregore PR #27

## D-2026-06-09-02 · Naming — status: open
- **Decision:** pick a name
"""

SAMPLE_WITH_PULL_URL = """# Pending decisions

## D-2026-06-09-01 · Add feature  — status: open
- **Decision:** ship it
- **Link:** https://github.com/org/repo/pull/42

## D-2026-06-09-02 · Hash shorthand — status: open
- **Decision:** pick
- **Link:** #99
"""


def test_reads_only_open_ids(tmp_path):
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE)
    assert read_open_ids(str(f)) == ["D-2026-06-09-01", "D-2026-06-09-03"]


def test_missing_file_returns_empty(tmp_path):
    assert read_open_ids(str(tmp_path / "nope.md")) == []


def test_format_example_not_matched(tmp_path):
    # The block-format doc example uses NN, never a real dated id.
    f = tmp_path / "d.md"
    f.write_text("## D-YYYY-MM-DD-NN · example  — status: open\n")
    assert read_open_ids(str(f)) == []


# --- New tests for read_open_decisions / Decision ---

def test_decision_with_pr_number(tmp_path):
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE_WITH_PR)
    decisions = read_open_decisions(str(f))
    assert len(decisions) == 2
    pr_decision = next(d for d in decisions if d.id == "D-2026-06-09-01")
    assert pr_decision.pr_number == 27


def test_decision_without_pr_number(tmp_path):
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE_WITH_PR)
    decisions = read_open_decisions(str(f))
    plain_decision = next(d for d in decisions if d.id == "D-2026-06-09-02")
    assert plain_decision.pr_number is None


def test_pr_from_pull_url(tmp_path):
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE_WITH_PULL_URL)
    decisions = read_open_decisions(str(f))
    pull_decision = next(d for d in decisions if d.id == "D-2026-06-09-01")
    assert pull_decision.pr_number == 42


def test_pr_from_hash_shorthand(tmp_path):
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE_WITH_PULL_URL)
    decisions = read_open_decisions(str(f))
    hash_decision = next(d for d in decisions if d.id == "D-2026-06-09-02")
    assert hash_decision.pr_number == 99


def test_read_open_ids_still_returns_ids(tmp_path):
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE_WITH_PR)
    # read_open_ids must still work as before, now backed by read_open_decisions
    assert read_open_ids(str(f)) == ["D-2026-06-09-01", "D-2026-06-09-02"]


def test_only_open_blocks_returned(tmp_path):
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE)
    decisions = read_open_decisions(str(f))
    ids = [d.id for d in decisions]
    assert "D-2026-06-09-02" not in ids   # resolved
    assert "D-2026-06-09-01" in ids
    assert "D-2026-06-09-03" in ids


# --- most_recent_resolved tests ---

from plugins.gardener_approvals.decisions import most_recent_resolved

SAMPLE_RESOLVED_MULTI = """# Decisions

## D-2026-06-07-01 · First — status: resolved
- **Decision:** a
- **Resolved:** 2026-06-07 by Juniper

## D-2026-06-09-01 · Latest — status: resolved
- **Decision:** b
- **Resolved:** 2026-06-09 by Juniper

## D-2026-06-08-01 · Middle — status: resolved
- **Decision:** c
- **Resolved:** 2026-06-08 by Juniper
"""

SAMPLE_NO_RESOLVED = """# Decisions

## D-2026-06-09-01 · Open one — status: open
- **Decision:** pending
"""

SAMPLE_RESOLVED_NO_DATE = """# Decisions

## D-2026-06-09-01 · No date — status: resolved
- **Decision:** a
"""

SAMPLE_RESOLVED_TIE = """# Decisions

## D-2026-06-08-01 · First with same date — status: resolved
- **Decision:** a
- **Resolved:** 2026-06-08

## D-2026-06-08-02 · Second with same date — status: resolved
- **Decision:** b
- **Resolved:** 2026-06-08
"""


def test_most_recent_resolved_returns_latest_by_date(tmp_path):
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE_RESOLVED_MULTI)
    r = most_recent_resolved(str(f))
    assert r is not None
    assert r.id == "D-2026-06-09-01"
    assert r.date == "2026-06-09"


def test_most_recent_resolved_none_when_no_resolved(tmp_path):
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE_NO_RESOLVED)
    assert most_recent_resolved(str(f)) is None


def test_most_recent_resolved_none_when_missing_file(tmp_path):
    assert most_recent_resolved(str(tmp_path / "nope.md")) is None


def test_most_recent_resolved_no_date_still_returned(tmp_path):
    # Single resolved block with no Resolved: date — still returns it (date="")
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE_RESOLVED_NO_DATE)
    r = most_recent_resolved(str(f))
    assert r is not None
    assert r.id == "D-2026-06-09-01"
    assert r.date == ""


def test_most_recent_resolved_tie_prefers_last_in_file(tmp_path):
    # Same date: last one in the file wins
    f = tmp_path / "decisions.md"
    f.write_text(SAMPLE_RESOLVED_TIE)
    r = most_recent_resolved(str(f))
    assert r is not None
    assert r.id == "D-2026-06-08-02"
