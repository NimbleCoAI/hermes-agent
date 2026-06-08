from plugins.gardener_approvals.decisions import read_open_ids

SAMPLE = """# Pending decisions

## D-2026-06-09-01 · First thing  — status: open
- **Decision:** a or b

## D-2026-06-09-02 · Second  — status: resolved
- **Resolved:** 2026-06-08

## D-2026-06-09-03 · Third — status: open
- **Decision:** x
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
