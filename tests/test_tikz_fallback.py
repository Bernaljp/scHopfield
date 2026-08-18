"""The circuit renderer must never leave a panel empty.

TeX is an optional dependency the package cannot declare, so these tests pin the two
things that make its absence safe rather than silent: a snippet that cannot be compiled
returns None instead of raising, and the drawing entry point falls through to matplotlib
and says so. The no-TeX case is simulated by pointing the probe at a PATH with no
pdflatex on it, so the tests are meaningful on a machine that does have TeX.
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import scHopfield as sch
from scHopfield.plotting import tikz as T


NODES = ["Pdx1", "Neurog3", "Sox9", "Arx"]
POS = {"Pdx1": (0.0, 1.0), "Neurog3": (1.0, 0.0), "Sox9": (0.0, -1.0), "Arx": (-1.0, 0.0)}
EDGES = [("Pdx1", "Neurog3", 0.8), ("Neurog3", "Sox9", -0.5), ("Sox9", "Arx", 0.2)]


@pytest.fixture
def no_tex(monkeypatch):
    """Make the availability probe fail the way a machine without TeX fails."""
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    T._AVAILABLE = None
    T._WARNED.clear()
    yield
    T._AVAILABLE = None
    T._WARNED.clear()


def _axes():
    fig, ax = plt.subplots()
    return fig, ax


def test_body_is_pure_text_and_needs_no_tex():
    """Building a snippet is string work, so it succeeds whatever is installed."""
    body = sch.pl.grn_tikz_body(NODES, POS, EDGES, scale=2.3)
    assert r"\node" in body and r"\draw" in body
    assert "activate" in body and "repress" in body     # sign survives into the style name
    assert body.count(r"\draw") == len(EDGES)


def test_underscore_in_a_gene_symbol_is_escaped():
    body = sch.pl.grn_tikz_body(["A_1", "B"], {"A_1": (0.0, 0.0), "B": (1.0, 0.0)},
                                [("A_1", "B", 1.0)], scale=2.0)
    assert r"A\_1" in body


def test_the_probe_reports_unavailable_when_pdflatex_is_not_on_the_path(no_tex):
    assert sch.pl.tikz_available() is False


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="needs pdflatex to fail at compile time")
def test_a_snippet_that_cannot_compile_returns_none_rather_than_raising():
    """The failure that produced four blank shipped panels: TeX is installed, the snippet
    is bad, and the renderer hands back None. It must not raise, so a caller can fall
    back, and it must not pretend it drew something."""
    assert sch.pl.render_tikz(r"\thisControlSequenceDoesNotExist", dpi=72) is None


def test_draw_grn_draws_something_without_tex(no_tex, capsys):
    """The whole point: no TeX must cost typography, not content."""
    fig, ax = _axes()
    try:
        sch.pl.draw_grn(ax, NODES, POS, EDGES, context="test panel")
        # one arrow per edge, one marker per node, one label per node
        assert len(ax.patches) == len(EDGES)
        assert len(ax.collections) == len(NODES)
        assert {t.get_text() for t in ax.texts} == set(NODES)
    finally:
        plt.close(fig)
    assert "matplotlib" in capsys.readouterr().err


def test_the_warning_is_said_once_not_once_per_panel(no_tex, capsys):
    fig, ax = _axes()
    try:
        for _ in range(5):
            sch.pl.draw_grn(ax, NODES, POS, EDGES)
    finally:
        plt.close(fig)
    assert capsys.readouterr().err.count("no working pdflatex") == 1


def test_sign_picks_the_head_and_the_color():
    fig, ax = _axes()
    try:
        sch.pl.draw_grn_mpl(ax, NODES, POS, EDGES, act_color="#00FF00", rep_color="#FF0000")
        colors = [p.get_edgecolor() for p in ax.patches]
        greens = sum(1 for c in colors if c[1] > 0.9 and c[0] < 0.1)
        reds = sum(1 for c in colors if c[0] > 0.9 and c[1] < 0.1)
    finally:
        plt.close(fig)
    assert (greens, reds) == (2, 1)                     # two positive weights, one negative


def test_an_unsigned_network_is_not_colored_by_sign():
    """A promoter-based prior says a regulator MAY act, not that it activates."""
    fig, ax = _axes()
    try:
        sch.pl.draw_grn_mpl(ax, NODES, POS, EDGES, neutral_color="#808080",
                            act_color="#00FF00", rep_color="#FF0000")
        colors = [p.get_edgecolor() for p in ax.patches]
    finally:
        plt.close(fig)
    assert all(abs(c[0] - c[1]) < 1e-6 and abs(c[1] - c[2]) < 1e-6 for c in colors)


def test_edge_width_runs_with_magnitude():
    fig, ax = _axes()
    try:
        sch.pl.draw_grn_mpl(ax, NODES, POS, EDGES, wmax=1.0, edge_lw=(0.0, 10.0))
        widths = sorted(p.get_linewidth() for p in ax.patches)
    finally:
        plt.close(fig)
    assert widths == pytest.approx([2.0, 5.0, 8.0])     # |0.2|, |-0.5|, |0.8| against wmax 1


def test_non_finite_and_negligible_edges_are_dropped():
    fig, ax = _axes()
    try:
        sch.pl.draw_grn_mpl(ax, NODES, POS,
                            EDGES + [("Arx", "Pdx1", np.nan), ("Arx", "Sox9", 0.0)])
        n = len(ax.patches)
    finally:
        plt.close(fig)
    assert n == len(EDGES)


@pytest.mark.skipif(not sch.pl.tikz_available(), reason="needs a working pdflatex")
def test_render_round_trips_when_tex_is_present():
    img = sch.pl.render_tikz(sch.pl.grn_tikz_body(NODES, POS, EDGES, scale=2.3), dpi=72)
    assert img is not None and img.ndim == 3 and min(img.shape[:2]) > 10


@pytest.mark.skipif(not sch.pl.tikz_available(), reason="needs a working pdflatex")
def test_the_render_is_cached_on_its_source():
    body = sch.pl.grn_tikz_body(NODES, POS, EDGES, scale=2.31)
    first = sch.pl.render_tikz(body, dpi=72)
    assert sch.pl.render_tikz(body, dpi=72) is first    # the same object, not merely equal
