"""TikZ circuit rendering, with a matplotlib fallback that draws the same network.

Regulatory circuits read better when LaTeX draws them: TikZ gives a real flat-bar
repression tip, true self-loops, and typeset gene labels that do not collide. So the
figures compile a standalone TikZ snippet with ``pdflatex``, rasterize it with
``pdftoppm``, and place the image in an Axes.

That buys quality at the cost of a dependency the package does not declare and cannot
require, because a TeX installation is not a Python package. The contract here is that
the dependency is optional and its absence is visible:

``render_tikz``
    returns ``None`` when TeX is missing or the snippet does not compile. It is the
    low-level call and it stays quiet, because a caller may legitimately want to try a
    render and decide for itself.

``draw_grn``
    is the call figures should make. It renders through TeX when it can and draws the
    same network with matplotlib when it cannot, warning on stderr about which path it
    took. It has no third outcome, so a missing TeX installation costs a reader some
    typographic polish and costs them nothing else.

The distinction matters because the failure it replaces was silent. A snippet that did
not compile used to return ``None`` into a caller that placed nothing, so the panel came
out as ordinary white space and four shipped supplementary figures carried an empty
panel that way. A blank panel does not read as an error; it reads as a panel with
nothing in it, which is a different and false statement about the data.

The fallback is a fallback, not a second design. It carries the same encoding as the
TikZ drawing, because the encoding is what the panel means: an arrowhead is activation,
a flat bar is repression, and line width runs with the magnitude of the interaction.

Availability
------------
``tikz_available()`` reports whether a render can succeed, by compiling a trivial
document once and caching the answer. Probing beats checking for the executable,
because a ``pdflatex`` with no ``standalone`` class or no ``pgf`` fails at compile
time rather than at lookup time.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "tikz_available",
    "render_tikz",
    "grn_preamble",
    "grn_tikz_body",
    "draw_grn",
    "draw_grn_mpl",
    "GRN_PREAMBLE",
    "DEFAULT_ACT_HEX",
    "DEFAULT_REP_HEX",
]

#: Activation teal and repression rust, the pair the cell-type GRN panels are drawn in.
#: They are a default, not a house rule: the figures in this repository use three
#: different pairs, so the color travels with the call rather than with the package.
DEFAULT_ACT_HEX = "2A9D8F"
DEFAULT_REP_HEX = "C44536"

_NODE_FILL = "ADADAD"


def grn_preamble(act_hex: str = DEFAULT_ACT_HEX, rep_hex: str = DEFAULT_REP_HEX,
                 node_fill: str = _NODE_FILL) -> str:
    """The preamble a :func:`grn_tikz_body` snippet is compiled against.

    It opens the ``tikzpicture`` environment and defines the two edge styles, so a body
    is a list of ``\\node`` and ``\\draw`` lines and nothing else. Colors are ``RRGGBB``
    without a leading hash, because that is what ``\\definecolor{...}{HTML}`` takes.
    """
    return (
        r"\documentclass[border=3pt]{standalone}"
        r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
        r"\usepackage{tikz}\usetikzlibrary{arrows.meta}"
        r"\definecolor{actcol}{HTML}{" + act_hex + r"}\definecolor{repcol}{HTML}{" + rep_hex + r"}"
        r"\definecolor{nodefill}{HTML}{" + node_fill + r"}"
        r"\begin{document}\begin{tikzpicture}["
        r"gene/.style={circle,draw=black!50,fill=nodefill,line width=0.5pt,inner sep=1.0pt},"
        r"activate/.style={-{Latex[length=3.5pt,width=3pt]},actcol},"
        r"repress/.style={-{Bar[width=4.5pt]},repcol}]" + "\n"
    )


#: Preamble at the default palette. Build your own with :func:`grn_preamble`.
GRN_PREAMBLE = grn_preamble()

#: What is appended after a body compiled against :data:`GRN_PREAMBLE`.
GRN_EPILOGUE = "\\end{tikzpicture}\n\\end{document}\n"

_RENDER_CACHE: Dict[Tuple[str, str, str, int], Optional[np.ndarray]] = {}
_AVAILABLE: Optional[bool] = None
_WARNED = set()


def _warn_once(key: str, message: str) -> None:
    """Say it on stderr, once per process, so a long run does not drown in repeats."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"WARNING: {message}", file=sys.stderr)


def tikz_available(force: bool = False) -> bool:
    """Whether a TikZ snippet can be compiled and rasterized on this machine.

    Probes once by compiling a trivial standalone document, then caches the answer.
    Pass ``force=True`` to re-probe, which tests use to simulate a machine without TeX.
    """
    global _AVAILABLE
    if _AVAILABLE is not None and not force:
        return _AVAILABLE
    if shutil.which("pdflatex") is None or shutil.which("pdftoppm") is None:
        _AVAILABLE = False
        return _AVAILABLE
    probe = _render_uncached(r"\node {x};" + "\n", GRN_PREAMBLE, GRN_EPILOGUE, dpi=72)
    _AVAILABLE = probe is not None
    return _AVAILABLE


def _render_uncached(body: str, preamble: str, epilogue: str, dpi: int) -> Optional[np.ndarray]:
    try:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "c.tex"), "w") as fh:
                fh.write(preamble + body + epilogue)
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "c.tex"],
                               cwd=td, capture_output=True)
            if r.returncode != 0 or not os.path.exists(os.path.join(td, "c.pdf")):
                return None
            subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                            os.path.join(td, "c.pdf"), os.path.join(td, "c")], capture_output=True)
            png = os.path.join(td, "c.png")
            return plt.imread(png) if os.path.exists(png) else None
    except Exception:
        return None


def render_tikz(body: str, preamble: str = GRN_PREAMBLE, epilogue: str = GRN_EPILOGUE,
                dpi: int = 600) -> Optional[np.ndarray]:
    """Compile a standalone LaTeX snippet to an image array, or ``None`` on any failure.

    Parameters
    ----------
    body
        The document body. What it may contain is set by ``preamble``: against the
        default it is a list of TikZ ``\\node`` and ``\\draw`` lines, because the
        default preamble has already opened the ``tikzpicture`` environment.
    preamble
        Everything before the body, up to and including ``\\begin{document}``.
    epilogue
        Everything after it. Must close whatever the preamble opened.
    dpi
        Rasterization resolution. The figures use 600 for a page-size panel.

    Returns ``None`` rather than raising, so a caller can fall back. Results are cached
    on the exact source and dpi, which matters because one figure renders the same
    circuit into several panels.
    """
    key = (body, preamble, epilogue, dpi)
    if key not in _RENDER_CACHE:
        _RENDER_CACHE[key] = _render_uncached(body, preamble, epilogue, dpi)
    return _RENDER_CACHE[key]


def _tex(s: str) -> str:
    """Escape the three characters a gene symbol realistically carries."""
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


_COMPASS = [(0, "east"), (45, "north east"), (90, "north"), (135, "north west"),
            (180, "west"), (225, "south west"), (270, "south"), (315, "south east")]


def _anchor(ang: float) -> str:
    """Anchor a node label so it sits away from the center of the layout."""
    a = (ang + 180) % 360
    return min(_COMPASS, key=lambda c: min(abs(a - c[0]), 360 - abs(a - c[0])))[1]


def grn_tikz_body(nodes: Sequence[str], pos: Mapping[str, Tuple[float, float]],
                  edges: Iterable[Tuple[str, str, float]], scale: float, size: float = 4.5,
                  fills: Optional[Mapping[str, str]] = None,
                  borders: Optional[Mapping[str, str]] = None,
                  bold: Optional[Iterable[str]] = None, lblfont: str = r"\tiny",
                  node_lw: float = 1.0, off_base: float = 0.16,
                  edge_lw: Tuple[float, float] = (0.25, 1.0), shorten: float = 2.0,
                  label_sep: float = 0.6, italic: bool = False, labels: bool = True,
                  wmax: Optional[float] = None, head_scale: float = 1.0) -> str:
    """The TikZ body for a gene regulatory network laid out at ``pos``.

    Parameters
    ----------
    nodes
        Gene names, in drawing order.
    pos
        ``{gene: (x, y)}`` in layout units; ``scale`` converts them to TikZ centimetres.
    edges
        ``(regulator, target, weight)`` triples. The sign of the weight picks the tip:
        positive draws an arrowhead, negative a flat bar.
    fills, borders
        ``{gene: "RRGGBB"}`` overrides for the node face and its border.
    bold
        Genes whose label is set bold, for marking a focal set.
    labels
        ``False`` draws bare nodes, for thumbnails whose gene identities are printed
        once in a separate key.
    wmax
        Fixes the edge-width normalizer, so a set of networks drawn separately stays
        comparable instead of each scaling to its own maximum.
    """
    bold = set(bold or ())
    edges = list(edges)
    idm = {n: f"n{i}" for i, n in enumerate(nodes)}
    cx = np.mean([pos[n][0] for n in nodes])
    cy = np.mean([pos[n][1] for n in nodes])
    L = []
    for i, n in enumerate(nodes):
        if fills and n in fills:
            L.append(f"\\definecolor{{fll{i}}}{{HTML}}{{{fills[n]}}}")
        if borders and n in borders:
            L.append(f"\\definecolor{{brd{i}}}{{HTML}}{{{borders[n]}}}")
    for i, n in enumerate(nodes):
        x, y = pos[n]
        fc = f"fll{i}" if (fills and n in fills) else "nodefill"
        bc = f"brd{i}" if (borders and n in borders) else "white"
        L.append(f"\\node[circle,draw={bc},fill={fc},line width={node_lw:.2f}pt,minimum size={size:.1f}mm,inner sep=0pt] "
                 f"({idm[n]}) at ({scale * x:.2f},{scale * y:.2f}) {{}};")
        if not labels:
            continue
        dx, dy = x - cx, y - cy
        r = float(np.hypot(dx, dy)) or 1.0
        off = size / 20.0 + off_base
        lx, ly = scale * x + off * dx / r, scale * y + off * dy / r
        lab = f"\\textit{{{_tex(n)}}}" if italic else _tex(n)
        lab = f"\\textbf{{{lab}}}" if n in bold else lab
        L.append(f"\\node[font={lblfont},anchor={_anchor(np.degrees(np.arctan2(dy, dx)))},inner sep={label_sep:.1f}pt] "
                 f"at ({lx:.2f},{ly:.2f}) {{{lab}}};")
    wmax = wmax or max((abs(w) for *_, w in edges), default=1.0)
    for u, v, w in edges:
        if u not in idm or v not in idm:
            continue
        st = "activate" if w > 0 else "repress"
        if head_scale != 1.0:
            # A later arrow spec wins over the one in the named style.
            st += (f",-{{Latex[length={3.5 * head_scale:.2f}pt,width={3.0 * head_scale:.2f}pt]}}"
                   if w > 0 else f",-{{Bar[width={4.5 * head_scale:.2f}pt]}}")
        L.append(f"\\draw[{st},line width={edge_lw[0] + edge_lw[1] * abs(w) / wmax:.2f}pt,"
                 f"shorten >={shorten:.1f}pt,shorten <={shorten:.1f}pt] "
                 f"({idm[u]}) to[bend left=6] ({idm[v]});")
    return "\n".join(L)


def draw_grn_mpl(ax, nodes: Sequence[str], pos: Mapping[str, Tuple[float, float]],
                 edges: Iterable[Tuple[str, str, float]], *,
                 wmax: Optional[float] = None,
                 node_size: float = 90.0, node_face: str = "#f0efe9",
                 node_edge: str = "0.35", node_lw: float = 0.6,
                 fontsize: float = 4.2, label_offset: float = 0.0,
                 edge_lw: Tuple[float, float] = (0.5, 3.0),
                 rad: float = 0.17, shrink: float = 8.0,
                 act_color: str = "#" + DEFAULT_ACT_HEX, rep_color: str = "#" + DEFAULT_REP_HEX,
                 neutral_color: Optional[str] = None,
                 fills: Optional[Mapping[str, str]] = None,
                 borders: Optional[Mapping[str, str]] = None,
                 bold: Optional[Iterable[str]] = None,
                 labels: bool = True, pad: float = 0.55, alpha: float = 0.9,
                 linestyle: str = "-", draw_nodes: bool = True,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None):
    """Draw the same network as :func:`grn_tikz_body`, in matplotlib.

    This is what a machine without TeX gets. It is not pixel-equivalent to the TikZ
    drawing and is not meant to be; it carries the same encoding, which is the part that
    means something. An arrowhead is activation, a flat bar is repression, and line
    width runs with ``|weight|`` against ``wmax``.

    The repression head is the honest limit of this fallback. matplotlib has no flat-bar
    arrowstyle, so the nearest option is a bracket, which reads more like punctuation
    than like a repression glyph. That is a reason to install TeX, not a reason to leave
    the panel blank.

    Parameters
    ----------
    neutral_color
        Draw every edge in this color with an arrowhead, ignoring the sign of the
        weight. For a network whose edges are unsigned by construction, such as a
        promoter-based prior, which asserts that a regulator may act on a target and not
        whether it activates or represses. Coloring those by sign would state something
        the prior does not say.
    """
    from matplotlib.patches import FancyArrowPatch

    bold = set(bold or ())
    edges = [(u, v, w) for u, v, w in edges
             if u in pos and v in pos and np.isfinite(w) and abs(w) > 1e-12]
    wmax = wmax or max((abs(w) for *_, w in edges), default=1.0) or 1.0

    for u, v, w in edges:
        lw = edge_lw[0] + edge_lw[1] * min(abs(w) / wmax, 1.0)
        if neutral_color is not None:
            style, ms, color = "-|>", 7, neutral_color
        else:
            style = "-|>" if w > 0 else "-["      # arrowhead = activation, flat bar = repression
            ms = 7 if w > 0 else 5.5
            color = act_color if w > 0 else rep_color
        ax.add_patch(FancyArrowPatch(pos[u], pos[v], connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                                     linestyle=linestyle, alpha=alpha,
                                     shrinkA=shrink, shrinkB=shrink, zorder=2))
    drawn = [n for n in nodes if n in pos] if draw_nodes else []
    cx = np.mean([pos[n][0] for n in drawn]) if drawn else 0.0
    cy = np.mean([pos[n][1] for n in drawn]) if drawn else 0.0
    for n in drawn:
        x, y = pos[n]
        fc = f"#{fills[n]}" if (fills and n in fills) else node_face
        ec = f"#{borders[n]}" if (borders and n in borders) else node_edge
        ax.scatter([x], [y], s=node_size, c=fc, edgecolor=ec, lw=node_lw, zorder=3)
        if not labels:
            continue
        weight = "bold" if n in bold else "normal"
        if label_offset:
            # Radially outward from the middle of the layout, as the TikZ version places
            # them. Centered labels are unreadable once a node carries a real gene symbol.
            dx, dy = x - cx, y - cy
            r = float(np.hypot(dx, dy)) or 1.0
            lx, ly = x + label_offset * dx / r, y + label_offset * dy / r
            ha = "left" if dx > 0.1 * r else ("right" if dx < -0.1 * r else "center")
            va = "bottom" if dy > 0.1 * r else ("top" if dy < -0.1 * r else "center")
            ax.text(lx, ly, n, fontsize=fontsize, ha=ha, va=va, zorder=4, fontweight=weight)
        else:
            ax.text(x, y, n, fontsize=fontsize, ha="center", va="center", zorder=4,
                    fontweight=weight)

    xs = [pos[n][0] for n in nodes if n in pos] or [0.0]
    ys = [pos[n][1] for n in nodes if n in pos] or [0.0]
    ax.set_xlim(*(xlim if xlim is not None else (min(xs) - pad, max(xs) + pad)))
    ax.set_ylim(*(ylim if ylim is not None else (min(ys) - pad, max(ys) + pad)))
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def draw_grn(ax, nodes: Sequence[str], pos: Mapping[str, Tuple[float, float]],
             edges: Iterable[Tuple[str, str, float]], *,
             scale: float = 2.3, dpi: int = 600,
             act_hex: str = DEFAULT_ACT_HEX, rep_hex: str = DEFAULT_REP_HEX,
             preamble: Optional[str] = None,
             transparent: bool = False, context: str = "network",
             tikz_kwargs: Optional[Mapping] = None,
             mpl_kwargs: Optional[Mapping] = None):
    """Draw a gene regulatory network into ``ax``, through TeX when it is available.

    This is the call a figure should make. It has two outcomes and no third: either the
    TikZ render lands in the Axes, or matplotlib draws the same network and says so on
    stderr. It never leaves the Axes empty.

    Both paths take their colors from ``act_hex`` and ``rep_hex``, so the drawn fallback
    cannot disagree with the render it stands in for.

    Parameters
    ----------
    act_hex, rep_hex
        ``RRGGBB`` for activation and repression, without a leading hash.
    preamble
        Overrides the preamble built from ``act_hex``/``rep_hex``. Pass one only to
        reproduce an existing snippet exactly.
    context
        Named in the warning, so a reader of a long log can tell which panel fell back.
    transparent
        Turn the render's white ground transparent, for a network that overlays a plot.
    tikz_kwargs, mpl_kwargs
        Passed to :func:`grn_tikz_body` and :func:`draw_grn_mpl` respectively.
    """
    edges = list(edges)
    if tikz_available():
        body = grn_tikz_body(nodes, pos, edges, scale=scale, **dict(tikz_kwargs or {}))
        img = render_tikz(body, preamble=preamble or grn_preamble(act_hex, rep_hex), dpi=dpi)
        if img is not None:
            ax.imshow(_alpha_on_white(img) if transparent else img)
            ax.set_axis_off()
            return ax
        _warn_once(f"tikz-compile:{context}",
                   f"the TikZ snippet for '{context}' did not compile; "
                   f"drawing the network with matplotlib instead")
    else:
        _warn_once("tikz-missing",
                   "no working pdflatex/pdftoppm found, so regulatory circuits are drawn "
                   "with matplotlib rather than TikZ. The encoding is the same; the "
                   "typography is not. Install a TeX distribution to reproduce the "
                   "published panels exactly.")
    mpl = {"act_color": f"#{act_hex}", "rep_color": f"#{rep_hex}", **dict(mpl_kwargs or {})}
    return draw_grn_mpl(ax, nodes, pos, edges, **mpl)


def _alpha_on_white(img, thresh: float = 0.94):
    """Give a rendered snippet an alpha channel, so its white ground stops hiding what
    it is drawn on top of."""
    img = np.asarray(img, float)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    rgb = img[..., :3]
    rgba = np.dstack([rgb, np.ones(rgb.shape[:2])])
    rgba[(rgb > thresh).all(axis=2), 3] = 0.0
    return rgba
