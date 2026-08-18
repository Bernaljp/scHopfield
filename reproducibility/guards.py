"""Loud failures for the figure scripts, written once and called from all ten.

A figure script has two honest outcomes when something it needs is absent. It can stop
with a message naming the missing thing and how to produce it, or it can degrade in a way
that is still true and say on stderr that it did. What it must never do is draw the panel
anyway, because a panel that draws is read as a result.

Three shapes of silent degradation were found in this tree, and this module names one
guard for each:

``require_file``
    An input that is simply absent. The script stops, naming the path and the command
    that writes it. Most of the tree already does this by opening the file directly and
    letting the ``FileNotFoundError`` name the path; this is for the sites that were
    testing ``os.path.exists`` and drawing an empty panel instead.

``require_distinct``
    Two inputs that a first-existing-path fallback can collapse onto one file. The
    comparison then runs a file against itself and reports no difference, which reads as
    a negative result rather than as a missing input.

``require_dataset_entry``
    A per-dataset registry, keyed by the ``--dataset`` argument, whose miss returns an
    empty default. The panel then draws a zero, or a blank, under a caption that asserts
    something about genes the registry was supposed to name.

``warn_once``
    The fourth outcome, for a degradation that stays true: a different rendering of the
    same quantity. Said on stderr once per process, following the same convention as
    ``scHopfield.pl``'s TikZ fallback.
"""
from __future__ import annotations

import os
import sys

__all__ = ["require_file", "require_distinct", "require_dataset_entry", "warn_once"]


_WARNED: set[str] = set()


def warn_once(key: str, message: str) -> None:
    """Say it on stderr, once per process, so a long run does not drown in repeats."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"WARNING: {message}", file=sys.stderr)


def require_file(path: str, what: str, how: str) -> str:
    """Return ``path``, or stop naming it and the command that produces it.

    Parameters
    ----------
    path
        The file the caller is about to read.
    what
        What the file is, in the figure's own terms, so the message says which panel
        goes missing rather than only which path does.
    how
        How to produce it. A command where one exists, otherwise the module and function
        that writes it. Never left vague: the point of the guard is that the reader does
        not have to go looking.
    """
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"{what} is missing.\n"
        f"  wanted: {path}\n"
        f"  produce it with: {how}"
    )


def require_distinct(first: str, second: str, what: str, how: str) -> None:
    """Stop when two inputs that must differ have resolved to the same file.

    A first-existing-path fallback is convenient and, for a single input, harmless. For
    the two arms of a comparison it is not: when the second arm is absent and falls back
    onto the first, the comparison measures a file against itself and draws a flat zero.
    That is indistinguishable from a real null result on the page.
    """
    if os.path.realpath(first) != os.path.realpath(second):
        return
    raise FileNotFoundError(
        f"{what} would compare a file with itself.\n"
        f"  both arms resolved to: {first}\n"
        f"  produce the missing arm with: {how}"
    )


def require_dataset_entry(registry, dataset: str, registry_name: str, what: str,
                          how: str = ""):
    """Return ``registry[dataset]``, or stop naming the registry and what it feeds.

    For the curated per-dataset tables the figures carry: lineage regulators, Jacobian
    gene pairs, progenitor cluster names. They are curated by hand for the datasets the
    paper draws, and every one of them was reached with ``.get(dataset, <empty>)``, so a
    dataset the curator never covered drew an empty panel rather than stopping.

    The message lists the datasets the registry does cover, because the usual cause is a
    ``--dataset`` argument this panel was never curated for, and the usual fix is to pick
    another dataset rather than to curate a new one.
    """
    entry = registry.get(dataset)
    if entry:
        return entry
    covered = ", ".join(sorted(registry)) or "no datasets at all"
    fix = how or f"add an entry for '{dataset}'"
    # ValueError, not KeyError: KeyError renders its message through repr, which turns a
    # multi-line explanation into one line of escaped newlines at the traceback.
    raise ValueError(
        f"no entry for dataset '{dataset}'.\n"
        f"  needed by: {what}\n"
        f"  registry: {registry_name}, which covers: {covered}\n"
        f"  fix: {fix}, or run this on a dataset the registry covers"
    )
