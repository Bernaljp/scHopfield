"""Collapse tqdm progress-bar frames in the committed tutorial notebooks.

``docs/conf.py`` sets ``nb_execution_mode = "off"``, so the documentation renders the
outputs recorded in the notebook file rather than re-running the cell. A terminal draws a
progress bar by rewriting one line with a carriage return, but the notebook records every
one of those rewrites as a separate output, and the documentation has no terminal to
collapse them: it prints them all, one under the next.

Run this after re-executing any notebook::

    python docs/clean_tutorial_output.py docs/tutorials/*.ipynb

Each run of a bar collapses to the single frame it ended on, so the reader sees the
completed bar that a terminal would have left behind. Nothing else in the notebook is
touched: code, stdout, results and images are all preserved.

Pass ``--check`` to report without writing, which is what continuous integration wants.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: A tqdm frame is recognized by its percentage and the bar that follows it.
FRAME = re.compile(r"(\d+)%\|")


def _text(output: dict) -> str:
    """Return an output's text, which nbformat stores as a string or a list of lines."""
    text = output.get("text", "")
    return "".join(text) if isinstance(text, list) else (text or "")


def _frame(output: dict) -> tuple[str, int] | None:
    """Return ``(bar description, percent)`` if this output is a tqdm frame, else None.

    A frame is a stderr stream carrying a carriage return and a percentage. The
    description is whatever tqdm was given as ``desc``, and it names the bar, so two
    bars in one cell stay apart.
    """
    if output.get("output_type") != "stream" or output.get("name") != "stderr":
        return None
    text = _text(output)
    if "\r" not in text:
        return None
    match = FRAME.search(text)
    if match is None:
        return None
    last = text.split("\r")[-1]
    match = FRAME.search(last)
    if match is None:
        return None
    return last[: match.start()].strip(), int(match.group(1))


def _clean_cell(outputs: list[dict]) -> list[dict]:
    """Drop tqdm's blank line-clears and every frame but the last of each bar run."""
    # A bar run ends where the same description restarts at a lower percentage, which is
    # what a second call to tqdm looks like. Only the last frame of each run survives.
    last_of_run: set[int] = set()
    highest: dict[str, int] = {}
    latest: dict[str, int] = {}
    for index, output in enumerate(outputs):
        frame = _frame(output)
        if frame is None:
            continue
        description, percent = frame
        if description in highest and percent < highest[description]:
            last_of_run.add(latest[description])
        highest[description] = percent
        latest[description] = index
    last_of_run.update(latest.values())

    kept: list[dict] = []
    for index, output in enumerate(outputs):
        if output.get("output_type") == "stream" and output.get("name") == "stderr":
            text = _text(output)
            if not text.strip():
                continue  # tqdm clearing the line it is about to redraw
            if _frame(output) is not None:
                if index not in last_of_run:
                    continue
                # Keep the frame the bar ended on, as a line of its own.
                output = dict(output, text=text.split("\r")[-1].rstrip("\n") + "\n")
        kept.append(output)
    return kept


def clean(path: Path, write: bool) -> tuple[int, int]:
    """Clean one notebook. Returns ``(outputs before, outputs after)``."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    before = after = 0
    for cell in notebook["cells"]:
        outputs = cell.get("outputs")
        if not outputs:
            continue
        before += len(outputs)
        cell["outputs"] = _clean_cell(outputs)
        after += len(cell["outputs"])
    if write and after != before:
        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return before, after


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would be dropped and exit non-zero if anything would be",
    )
    args = parser.parse_args(argv)

    dirty = 0
    for path in args.notebooks:
        before, after = clean(path, write=not args.check)
        if before != after:
            dirty += 1
            verb = "would drop" if args.check else "dropped"
            print(f"{path}: {verb} {before - after} progress-bar frames of {before} outputs")
    if args.check and dirty:
        print(
            f"\n{dirty} notebook(s) carry uncollapsed progress bars. "
            "Run this script without --check.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
