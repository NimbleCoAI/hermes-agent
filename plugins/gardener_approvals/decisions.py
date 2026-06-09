"""Read OPEN decisions from the gardener decisions.md. Mirrors the gate's
`open_ids()` grep so the hook and the gate agree on what 'open' means. Real
dated ids only — the `D-YYYY-MM-DD-NN` format example is never matched.

Each open block is returned as a Decision dataclass carrying the id and,
when the block contains a Link: line that references a PR, the PR number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_OPEN_RE = re.compile(r"^## (D-\d{4}-\d{2}-\d{2}-\d+) .*status: open", re.MULTILINE)

# Matches any of:
#   PR #27   |  #27   |  pull/27   |  /pull/27
_PR_NUM_RE = re.compile(r"(?:PR\s+#|/pull/)(\d+)|#(\d+)|pull/(\d+)", re.IGNORECASE)


@dataclass
class Decision:
    id: str
    pr_number: Optional[int]


def read_open_decisions(decisions_path: str) -> List[Decision]:
    try:
        with open(decisions_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return []

    decisions: List[Decision] = []
    for m in _OPEN_RE.finditer(text):
        dec_id = m.group(1)
        block_start = m.start()
        # Find end of block: next ## header or EOF
        next_header = re.search(r"^## ", text[block_start + 1:], re.MULTILINE)
        block_end = (block_start + 1 + next_header.start()) if next_header else len(text)
        block_text = text[block_start:block_end]

        pr_number: Optional[int] = None
        # Only scan Link: lines for PR numbers
        for line in block_text.splitlines():
            if "link:" in line.lower():
                pm = _PR_NUM_RE.search(line)
                if pm:
                    # groups: (PR #N, #N, pull/N) — first non-None group wins
                    raw = pm.group(1) or pm.group(2) or pm.group(3)
                    pr_number = int(raw)
                    break

        decisions.append(Decision(id=dec_id, pr_number=pr_number))

    return decisions


def read_open_ids(decisions_path: str) -> List[str]:
    return [d.id for d in read_open_decisions(decisions_path)]
