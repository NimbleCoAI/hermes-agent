#!/usr/bin/env python3
"""Migrate memory context directories onto the platform-qualified id format.

Background
----------
Before the partitioned MemoryStore landed, the gateway scoped memory by the
raw ``chat_id`` (e.g. ``group:<signal-id>``) and, on every ordinary scoped
write, persisted the MERGED global+scoped view into the scoped file. Two
consequences survive the code fix and need a data pass:

1. The context id format changed to ``{platform}:{chat_type}:{chat_id}``.
   Directories under the old name are never loaded again, so their genuinely
   scoped entries drop out of the agent's read view.
2. Those directories still contain copies of global entries. The store's
   self-heal only runs for a context it actually loads, so an orphaned
   directory keeps its copied global content on disk indefinitely.

This script renames each old-format directory to its new id and strips any
entry that is also present in the global file for that target.

Safety
------
Dry-run by default: prints the plan and changes nothing. Pass ``--apply`` to
write. ``--apply`` takes a timestamped backup of each memories directory it
touches before modifying it.

Platform cannot be recovered from the directory name alone, so it must be
supplied per run with ``--platform``. Ids that do not look like they belong to
that platform are reported and skipped rather than guessed at.

Usage
-----
    # inspect every agent on this host
    python3 scripts/migrate_memory_contexts.py ~/.hermes-*/memories --platform signal

    # commit the change
    python3 scripts/migrate_memory_contexts.py ~/.hermes-*/memories --platform signal --apply
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ENTRY_DELIMITER = "\n§\n"
TARGETS = (("memory", "MEMORY.md"), ("user", "USER.md"))

# A directory name already in {platform}:{chat_type}:{chat_id} form. The chat
# id itself may contain ':' (Signal's "group:<id>"), so only the first two
# segments are structural.
_NEW_FORMAT = re.compile(r"^[a-z0-9_]+:[a-z0-9_]+:.+$", re.IGNORECASE)

# Chat types the old gateway could produce a scoped directory for. It only
# ever scoped group/forum/channel; everything else went to global.
_OLD_SCOPED_CHAT_TYPES = ("group", "forum", "channel")


def read_entries(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    if not raw.strip():
        return []
    return [e for e in (x.strip() for x in raw.split(ENTRY_DELIMITER)) if e]


def write_entries(path: Path, entries: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


def infer_chat_type(old_id: str, platform: str) -> Optional[str]:
    """Infer the chat_type the old id was scoped under, or None if unclear.

    The old gateway only created scoped directories for group/forum/channel,
    so a directory that exists at all was one of those. Signal encodes it in
    the id itself; for other platforms we cannot distinguish group from
    channel and say so rather than guessing.
    """
    if old_id.startswith("group:"):
        return "group"
    if platform == "signal":
        # Signal DMs were never scoped, so a non-"group:" Signal directory is
        # not something this script knows how to name.
        return None
    return None


def plan_for_memories_dir(
    mem_dir: Path, platform: str
) -> Tuple[List[Dict], List[Dict]]:
    """Return (actions, skipped) for one memories/ directory."""
    actions: List[Dict] = []
    skipped: List[Dict] = []
    contexts = mem_dir / "contexts"
    if not contexts.is_dir():
        return actions, skipped

    globals_by_target = {
        target: set(read_entries(mem_dir / fname)) for target, fname in TARGETS
    }

    for entry in sorted(contexts.iterdir()):
        if not entry.is_dir():
            continue
        old_id = entry.name

        if _NEW_FORMAT.match(old_id) and not old_id.startswith("group:"):
            skipped.append({"dir": old_id, "reason": "already new format"})
            continue

        chat_type = infer_chat_type(old_id, platform)
        if chat_type is None:
            skipped.append({
                "dir": old_id,
                "reason": f"cannot infer chat_type for platform '{platform}' "
                          f"— rename by hand or rerun with the right --platform",
            })
            continue

        new_id = f"{platform}:{chat_type}:{old_id}"
        safe_new_id = new_id.replace("/", "_").replace("\\", "_").replace("..", "_")

        strip: Dict[str, int] = {}
        keep: Dict[str, List[str]] = {}
        for target, fname in TARGETS:
            entries = read_entries(entry / fname)
            if not entries:
                continue
            gset = globals_by_target[target]
            kept = [e for e in entries if e not in gset]
            strip[fname] = len(entries) - len(kept)
            keep[fname] = kept

        actions.append({
            "mem_dir": mem_dir,
            "old_dir": entry,
            "old_id": old_id,
            "new_id": safe_new_id,
            "new_dir": contexts / safe_new_id,
            "strip": strip,
            "keep": keep,
        })

    return actions, skipped


def apply_action(action: Dict) -> None:
    new_dir: Path = action["new_dir"]
    old_dir: Path = action["old_dir"]

    # Rewrite the files with global-copied entries removed, then move the dir.
    for fname, kept in action["keep"].items():
        write_entries(old_dir / fname, kept)

    if new_dir.exists():
        # Merge into an existing new-format dir rather than clobbering it.
        for fname, kept in action["keep"].items():
            existing = read_entries(new_dir / fname)
            merged = list(dict.fromkeys(existing + kept))
            write_entries(new_dir / fname, merged)
        shutil.rmtree(old_dir)
    else:
        old_dir.rename(new_dir)


def backup(mem_dir: Path, stamp: str) -> Path:
    dest = mem_dir.parent / f"{mem_dir.name}.pre-migration-{stamp}"
    shutil.copytree(mem_dir, dest)
    return dest


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("memories_dirs", nargs="+", type=Path,
                    help="one or more agent memories/ directories")
    ap.add_argument("--platform", required=True,
                    help="platform these agents run on (e.g. signal, telegram)")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default is a dry run)")
    args = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    total_actions = 0
    total_stripped = 0

    for mem_dir in args.memories_dirs:
        if not mem_dir.is_dir():
            print(f"!! {mem_dir}: not a directory, skipping")
            continue

        actions, skipped = plan_for_memories_dir(mem_dir, args.platform)
        if not actions and not skipped:
            continue

        print(f"\n=== {mem_dir}")
        for s in skipped:
            print(f"  -- skip {s['dir']}\n       {s['reason']}")
        for a in actions:
            stripped = sum(a["strip"].values())
            total_stripped += stripped
            print(f"  -> {a['old_id']}")
            print(f"     rename to {a['new_id']}")
            for fname, n in sorted(a["strip"].items()):
                kept = len(a["keep"].get(fname, []))
                print(f"     {fname}: drop {n} global-copied, keep {kept} scoped")
        total_actions += len(actions)

        if args.apply and actions:
            dest = backup(mem_dir, stamp)
            print(f"  backup: {dest}")
            for a in actions:
                apply_action(a)
            print(f"  applied {len(actions)} migration(s)")

    verb = "applied" if args.apply else "planned (dry run — pass --apply to write)"
    print(f"\n{total_actions} directory migration(s) {verb}; "
          f"{total_stripped} global-copied entries removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
