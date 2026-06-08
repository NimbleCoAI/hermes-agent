"""Read OPEN decision ids from the gardener decisions.md. Mirrors the gate's
`open_ids()` grep so the hook and the gate agree on what 'open' means. Real
dated ids only — the `D-YYYY-MM-DD-NN` format example is never matched.
"""
from __future__ import annotations

import re
from typing import List

_OPEN_RE = re.compile(r"^## (D-\d{4}-\d{2}-\d{2}-\d+) .*status: open", re.MULTILINE)


def read_open_ids(decisions_path: str) -> List[str]:
    try:
        with open(decisions_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return []
    return _OPEN_RE.findall(text)
