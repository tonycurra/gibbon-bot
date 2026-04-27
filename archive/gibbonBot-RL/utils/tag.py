#!/usr/bin/env python3
"""
Tag MJX checkpoints with notes — your "this one was good" log.

Usage:
    python utils/tag.py "first stable stand, very smooth"
                                # tags the latest checkpoint
    python utils/tag.py mjx_000262144.npz "pre-tweak, baseline"
                                # tag a specific checkpoint
    python utils/tag.py --show  # print all notes
    python utils/tag.py --list  # list all checkpoints + their tags

Notes are appended to runs/ape_stand_mjx/notes.md. Each entry includes the
checkpoint name, its saved-time params snapshot path, and your message.
"""

import argparse
import datetime
import json
import pathlib
import sys

RUNS = pathlib.Path(__file__).parent.parent / "runs" / "ape_stand_mjx"
NOTES = RUNS / "notes.md"


def latest_ckpt():
    ckpts = sorted(RUNS.glob("mjx_*.npz"))
    return ckpts[-1] if ckpts else None


def find_ckpt(name_or_partial):
    """Accept full filename, stem, or partial match like '262144'."""
    direct = RUNS / name_or_partial
    if direct.exists():
        return direct
    matches = sorted(RUNS.glob(f"mjx_*{name_or_partial}*.npz"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"Multiple matches: {[m.name for m in matches]}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def load_params_for(ckpt):
    sidecar = ckpt.with_suffix("").with_suffix(".params.json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return None


def append_note(ckpt, message):
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    if not NOTES.exists():
        NOTES.write_text("# Training Notes\n\n"
                         "Tagged checkpoints worth remembering.\n\n")

    params = load_params_for(ckpt)
    ts = datetime.datetime.now().isoformat(timespec="seconds")

    block = [f"## {ckpt.name}    {ts}"]
    block.append(f"> {message}")
    block.append("")
    if params:
        # Show the params that were active when this checkpoint was saved
        block.append("Params at save time:")
        block.append("```")
        for k, v in sorted(params.items()):
            if k.startswith("_"):
                continue
            block.append(f"  {k:<28} = {v}")
        block.append("```")
    else:
        block.append("_(no params snapshot found — saved before this feature existed)_")
    block.append("")

    with NOTES.open("a") as f:
        f.write("\n".join(block) + "\n")
    print(f"[tag] noted {ckpt.name} → {NOTES}")


def show_notes():
    if not NOTES.exists():
        print("No notes yet. Run `python utils/tag.py \"your note\"` to add one.")
        return
    print(NOTES.read_text())


def list_ckpts():
    ckpts = sorted(RUNS.glob("mjx_*.npz"))
    if not ckpts:
        print("No checkpoints yet.")
        return

    # Build a map of {ckpt_name: [note_lines]} from notes.md
    tag_map = {}
    if NOTES.exists():
        current = None
        for line in NOTES.read_text().splitlines():
            if line.startswith("## mjx_") and ".npz" in line:
                current = line.split()[1]
                tag_map.setdefault(current, [])
            elif current and line.startswith("> "):
                tag_map[current].append(line[2:])

    print(f"{'checkpoint':<32} {'has params':<12} {'tags'}")
    print("-" * 80)
    for c in ckpts:
        sidecar = c.with_suffix("").with_suffix(".params.json")
        has_p = "✓" if sidecar.exists() else "—"
        tags = "; ".join(tag_map.get(c.name, [])) or "(no tags)"
        print(f"{c.name:<32} {has_p:<12} {tags}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("args", nargs="*",
                        help="[ckpt_name_or_partial] message")
    parser.add_argument("--show", action="store_true", help="print all notes")
    parser.add_argument("--list", action="store_true",
                        help="list all checkpoints with their tags")
    a = parser.parse_args()

    if a.show:
        show_notes()
        return
    if a.list:
        list_ckpts()
        return

    if len(a.args) == 0:
        parser.print_help()
        return
    if len(a.args) == 1:
        ckpt = latest_ckpt()
        if ckpt is None:
            print("No checkpoints found.", file=sys.stderr)
            sys.exit(1)
        append_note(ckpt, a.args[0])
    else:
        ckpt = find_ckpt(a.args[0])
        if ckpt is None:
            print(f"No checkpoint matches: {a.args[0]}", file=sys.stderr)
            sys.exit(1)
        message = " ".join(a.args[1:])
        append_note(ckpt, message)


if __name__ == "__main__":
    main()
