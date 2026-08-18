"""Figure 1: the scHopfield framework overview.

Portrait, full page, four bracketed lanes read top to bottom: what goes in, the single dynamical system
that is fitted, the four readouts that are projections of that one system, and where each claim is
tested.

DELIBERATELY CONCEPTUAL. No numeric values, no axis ticks, no gene names and no dataset name appear
anywhere in the artwork; the only digits are the "cell type 1 / 2" labels. There is also no explanatory
prose inside the panels: the figure carries titles, term explanations and element labels only, and
everything else belongs in the caption. Every panel that shows data is nonetheless PLOTTED FROM THE
REAL FIT and then stripped of its labels, so the reader sees genuine shapes rather than a cartoon, and
the figure regenerates with the fits instead of drifting from them.

Everything the equations assert was verified against the code before drawing (see CAPTIONS.md):
  * the objective is  L = lam_rec*MSE + lam_scaf*||W o (1-S)|| + lam_bias*||I||_1 + lam_bound*<r(x)>_+,
    the first three always on and the fourth canonical; PLUS two HARD constraints that are not part of
    the objective at all (the transcription-factor column mask on W, and the clamp on gamma).
  * the energy is    E = -1/2 sigma^T W sigma + sum_i gamma_i int_0^{sigma_i} phi_i^{-1} - I^T sigma.
  * the Jacobian is  J = W diag(phi'(x)) - diag(gamma).  The bias I DROPS OUT, because it is constant.
    No arrow may run from the bias into the Jacobian, and none does.

THE ONE SCHEMATIC EXCEPTION, recorded so nobody mistakes it for a measurement: the dashed edges in
panel e are drawn by hand. They state that the fit MAY place interactions off the prior, which is
architecturally true (off-scaffold entries are soft-penalized, not forbidden). Measured on the
canonical fit, such edges exist in number but essentially all sit at negligible magnitude, so no real
one would be legible at this size. Everything else in the panel, signs and widths included, is real.

  a  inputs: expression plus unspliced counts or a pseudotime, giving an estimate of the dynamics.
  b  the optional inputs: a regulatory prior (recommended) and cell-type relationships (optional).
  c  fitted per-gene activation, for a unimodal and a bimodal expression distribution.
  d  the fitted system: the governing equation, its parameters color-keyed, and the objective listed.
  e  the fitted networks for two cell types, on the same node layout as the prior in (b).
  f  the quasi-potential energy landscape.
  g  local stability: the Jacobian, its eigenvalue cloud, and the rotational part over the embedding.
  h  network structure: the fitted interaction matrix as a graph, and its regulators.
  i  in-silico knockout: hold one coordinate at zero, integrate, read the response and the flow.
  j  where each claim is tested, including one honest negative.

Run:  python reproducibility/make_framework_figure.py [--dataset paul15_coarse]
"""
from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers the 3d projection)

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
from paper_plot_style import use_style, save, INK, MUTED       # noqa: E402
import anndata as ad                                           # noqa: E402
from sections import basis_of                                  # noqa: E402
import scHopfield as sch                                       # noqa: E402  (circuit rendering)
from scHopfield import sigmoid                                 # noqa: E402

OUT = paths.FIGURES
FW, FH = 7.087, 9.70                    # portrait, a full page. 7.087 in = 180 mm exactly,
                                        # the double-column maximum; 7.2 was 183 mm, over it.

RULE, SOFT, FAINT = "#c9d2d0", "#7d8a88", "#aab4b2"
ACCENT = "#1b7a6e"                      # flow-of-logic marks only
COOL, WARM = "#2166ac", "#b2182b"       # RdBu_r ends: repression/stable and activation/unstable
PERT = "#e69f00"                        # Okabe-Ito orange, perturbation only

# Parameter identity: these color the SYMBOLS so a reader can trace the same parameter across the
# model, the objective, the energy, the Jacobian and the knockout. Keyed in panel d.
PW, PPHI, PGAM, PBIAS = "#7b3294", "#009e73", "#8c5109", "#56b4e9"
HEXW, HEXPHI, HEXGAM, HEXBIAS = "7B3294", "009E73", "8C5109", "56B4E9"

# Sequential ramps, one per distinct magnitude. Never two quantities on one ramp.
CM_EXPR, CM_UNSPL, CM_PTIME = "Blues", "Purples", "cividis"
CM_ENERGY, CM_ROT = "viridis", "magma"

ACTIVATION = {"paul15_coarse": ("Tmsb10", "Klf1")}     # a unimodal and a bimodal expression distribution
CONTRAST = {"paul15_coarse": ("MEP", "Monocytes")}
# Two genes on purpose: the response curves use the knockout that moves the most cell types
# (five of seven), the flow map the most lineage-localized one (76 percent in one arm). They
# are DIFFERENT perturbations and the caption says so; the panel is not one pipeline.
KO_SIM = {"paul15_coarse": "Gfi1"}
KO_PROJ = {"paul15_coarse": "Klf1"}
SCAFFOLD = {"paul15_coarse": os.path.join(paths.DATASETS, "hematopoiesis/base_GRN.parquet")}
PSEUDOTIME = "Pseudotime"

LANES = [("OBSERVE", "Input: a dynamical observation of single cells"),
         ("ONE FITTED SYSTEM", "A continuous Hopfield system is fitted to that observation"),
         ("FOUR READOUTS", "The same fitted system, read four ways"),
         ("SCOPE", "Capabilities, and the evidence behind each")]

SCOPE_ROWS = [
    ("recovers the true interaction matrix",    ("yes", "yes", "na")),
    ("reproduces the observed dynamics",        ("yes", "yes", "yes")),
    ("recovers established lineage regulators", ("na", "na", "yes")),
    ("energy and stability track the dynamics", ("yes", "no", "yes")),
]
SCOPE_COLS = ["circuits with a\nknown matrix", "simulated\nground truth", "real developmental\nsystems"]


# --------------------------------------------------------------------------- #
# geometry: inches from the TOP-LEFT of the page
# --------------------------------------------------------------------------- #
def fx(x):
    return x / FW


def fy(y_top):
    return (FH - y_top) / FH


def rect(x, y_top, w, h):
    return [fx(x), fy(y_top + h), w / FW, h / FH]


def panel_box(fig, x, y_top, w, h, lw=0.7, edge=RULE, face="none", ls="solid"):
    # zorder -5, NOT 0: Figure.get_children() returns axes before figure patches, and a stable sort on
    # equal zorder therefore paints figure patches OVER the axes, hiding a filled panel's contents.
    fig.patches.append(FancyBboxPatch(
        (fx(x), fy(y_top + h)), w / FW, h / FH,
        boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=-5))


def head(fig, x, y_top, letter, title):
    fig.text(fx(x + 0.07), fy(y_top + 0.145), letter, fontsize=9, fontweight="bold",
             color=INK, ha="left", va="baseline")
    fig.text(fx(x + 0.20), fy(y_top + 0.145), title, fontsize=7.4, fontweight="bold",
             color=INK, ha="left", va="baseline")


def bracket(fig, y0, y1, label, color=SOFT, x=0.20, tick=0.09):
    fig.lines.append(plt.Line2D([fx(x), fx(x), fx(x + tick)], [fy(y0), fy(y1), fy(y1)],
                                transform=fig.transFigure, color=color, linewidth=0.9, zorder=0))
    fig.lines.append(plt.Line2D([fx(x), fx(x + tick)], [fy(y0), fy(y0)],
                                transform=fig.transFigure, color=color, linewidth=0.9, zorder=0))
    fig.text(fx(x - 0.045), fy(0.5 * (y0 + y1)), " ".join(label), fontsize=6.2, fontweight="bold",
             color=color, rotation=90, ha="center", va="center")


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def thin_axes(ax, xlabel=None, ylabel=None, size=5.4):
    ax.set_xticks([]); ax.set_yticks([])
    ax.minorticks_off()                       # log axes keep minor ticks after set_xticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SOFT); ax.spines[s].set_linewidth(0.7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=size, color=FAINT, labelpad=1.2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=size, color=FAINT, labelpad=2.2)
    return ax


def plus(fig, x, y, size=11, color=SOFT):
    """Row-1 ingredients are COMBINED, not passed through stages, so they are joined by a plus."""
    fig.text(fx(x), fy(y), "+", fontsize=size, color=color, ha="center", va="center")


def arrow(fig, x0, y0, x1, y1, color=SOFT, lw=0.9, ms=7, ls="solid"):
    fig.patches.append(FancyArrowPatch((fx(x0), fy(y0)), (fx(x1), fy(y1)),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=ms,
                                       linewidth=lw, color=color, linestyle=ls,
                                       shrinkA=0, shrinkB=0, zorder=0))


def lab(fig, x, y, text, size=5.4, color=INK, ha="center"):
    fig.text(fx(x), fy(y), text, fontsize=size, color=color, ha=ha, va="baseline")


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
TEX_PRE = (
    r"\documentclass[border=2pt]{standalone}"
    r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
    r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{array}"
    r"\usepackage{tikz}\usetikzlibrary{arrows.meta}"
    r"\definecolor{ink}{HTML}{12181A}\definecolor{soft}{HTML}{7D8A88}"
    r"\definecolor{faint}{HTML}{AAB4B2}\definecolor{muted}{HTML}{63716F}"
    r"\definecolor{cool}{HTML}{2166AC}\definecolor{warm}{HTML}{B2182B}"
    r"\definecolor{pW}{HTML}{" + HEXW + r"}\definecolor{pPhi}{HTML}{" + HEXPHI + r"}"
    r"\definecolor{pGam}{HTML}{" + HEXGAM + r"}\definecolor{pI}{HTML}{" + HEXBIAS + r"}"
    r"\begin{document}"
)
def render_tex(body, dpi=600):
    """Typeset one snippet against this figure's preamble.

    The renderer is ``sch.pl.render_tikz``, which caches on the source, so a snippet
    drawn into several panels is compiled once. This figure sets equations and small
    tables as well as circuits, so the preamble does not open a ``tikzpicture``: each
    body opens its own where it wants one.
    """
    return sch.pl.render_tikz(body, preamble=TEX_PRE, epilogue=r"\end{document}" + "\n", dpi=dpi)


def common_tex_scale(bodies, widths, max_h=0.30):
    """The scale, in inches per rendered pixel, at which a ROW of equations shares one type size.

    Matching bounding-box HEIGHT does not do it: an equation carrying a summation and an integral is
    far taller than a one-line equation at the same point size, so equal boxes make it render smaller.
    Every snippet is set from the same LaTeX size at the same dpi, so equal type size means one common
    inches-per-pixel factor.
    """
    ss = []
    for body, w in zip(bodies, widths):
        img = render_tex(body)
        if img is not None:
            ss.append(min(w / img.shape[1], max_h / img.shape[0]))
    return min(ss) if ss else None


def _alpha_on_white(img, thresh=0.94):
    img = np.asarray(img, float)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    rgb = img[..., :3]
    rgba = np.dstack([rgb, np.ones(rgb.shape[:2])])
    rgba[(rgb > thresh).all(axis=2), 3] = 0.0
    return rgba


def tex_panel(fig, x, y, w, h, body, fallback=None, fontsize=8, center=True, scale=None):
    """Typeset ``body`` into a panel, or fall back when there is no TeX to typeset it with.

    ``fallback`` is either a string, set as plain matplotlib math text, or a callable
    taking the Axes, for a panel whose content is a drawing rather than a formula. It
    used to be a string only, and six of the eleven panels here passed nothing at all,
    so on a machine without TeX they came out as white space that reads as a panel with
    nothing in it rather than as a panel that failed to draw.
    """
    img = render_tex(body)
    if img is None:
        ax = fig.add_axes(rect(x, y, w, h)); ax.set_axis_off()
        if callable(fallback):
            fallback(ax)
        elif fallback:
            ax.text(0.5, 0.5, fallback, fontsize=fontsize, ha="center", va="center",
                    color=INK, transform=ax.transAxes)
        return ax
    ih, iw = img.shape[0], img.shape[1]
    if scale is not None:
        ww, hh = scale * iw, scale * ih
    else:
        hh = h
        ww = hh * iw / ih
        if ww > w:
            ww, hh = w, w * ih / iw
    xx = x + (w - ww) / 2 if center else x
    ax = fig.add_axes(rect(xx, y + (h - hh) / 2, ww, hh)); ax.set_axis_off()
    ax.imshow(_alpha_on_white(img))
    ax.set_axis_off()
    return ax


# ---- the shared node layout, used by the prior in (b) and the fitted networks in (e) ------------
REG_XY = [(0.35, 1.55), (1.95, 1.85), (3.55, 1.55)]
TGT_XY = [(-0.15, 0.0), (1.15, -0.18), (2.55, -0.14), (3.95, 0.04)]
EDGES = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (1, 0), (0, 2)]   # real, from the fit
# SCHEMATIC ONLY. Drawn by hand and dashed, to state that the fit may place interactions off the
# prior, which is architecturally true (off-scaffold entries are soft-penalized, not forbidden). On
# the canonical fit such edges exist but sit at negligible magnitude, so no real one would be legible.
EXTRA_EDGES = [[(2, 1, +1), (0, 3, -1)],          # cell type 1
               [(1, 3, +1), (2, 0, -1)]]          # cell type 2, deliberately different

TIKZ_NODES = (
    r"\begin{tikzpicture}["
    r"reg/.style={circle,draw=ink,fill=white,line width=0.8pt,minimum size=5.4mm,inner sep=0pt},"
    r"tgt/.style={circle,draw=soft,fill=faint!70,line width=0.7pt,minimum size=3.9mm,inner sep=0pt}]"
)


# --------------------------------------------------------------------------- #
# Fallbacks for the three panels whose content is a NETWORK rather than a formula.
#
# Each one draws the same graph the TikZ body draws, through sch.pl.draw_grn_mpl, so a
# machine without TeX gets the network instead of white space. They are degradations and
# are meant to look like one: the repression head becomes a bracket, and the regulator
# and target discs come out the same size. What they do not do is state anything the
# TikZ version does not.
# --------------------------------------------------------------------------- #
def _bipartite_nodes():
    """The regulator -> target layout panels b and e share."""
    nodes = [f"r{i}" for i in range(len(REG_XY))] + [f"t{j}" for j in range(len(TGT_XY))]
    pos = {f"r{i}": xy for i, xy in enumerate(REG_XY)}
    pos.update({f"t{j}": xy for j, xy in enumerate(TGT_XY)})
    return nodes, pos


def prior_fallback(in_prior):
    """Panel b's prior. Its edges stay UNSIGNED, as in the TikZ version: a promoter-based
    prior asserts that a regulator MAY act on a target, not whether it activates or
    represses, so coloring them by sign would put a claim in the panel that is not there."""
    nodes, pos = _bipartite_nodes()
    eds = [(f"r{a}", f"t{b}", 1.0) for a, b in EDGES if (a, b) in in_prior]
    return lambda ax: sch.pl.draw_grn_mpl(
        ax, nodes, pos, eds, labels=False, neutral_color=SOFT, node_size=70,
        node_face="white", node_edge=INK, node_lw=0.8,
        edge_lw=(0.9, 0.0), rad=0.0, shrink=5, alpha=1.0)


def fitted_network_fallback(W, reg_idx, tgt_idx, wmax, extra):
    """Panel e's fitted networks. The schematic off-prior edges stay dashed, because that
    dash is what says they are drawn by hand rather than fitted (see EXTRA_EDGES)."""
    nodes, pos = _bipartite_nodes()
    real = [(f"r{a}", f"t{b}", float(W[tgt_idx[b], reg_idx[a]])) for a, b in EDGES]
    sketch = [(f"r{a}", f"t{b}", float(sgn)) for a, b, sgn in extra]
    kw = dict(wmax=wmax, labels=False, act_color=WARM, rep_color=COOL, node_size=70,
              node_face="white", node_edge=INK, node_lw=0.8, rad=0.0, shrink=5, alpha=1.0)

    def draw(ax):
        sch.pl.draw_grn_mpl(ax, nodes, pos, real, edge_lw=(0.45, 1.25), **kw)
        sch.pl.draw_grn_mpl(ax, nodes, pos, sketch, edge_lw=(0.8, 0.0), linestyle=(0, (1.8, 1.4)),
                            draw_nodes=False, **kw)
    return draw


def circular_network_fallback(W, core, out):
    """Panel f's regulator core, on the same circular layout at the same 0.14 threshold."""
    n = len(core)
    sub = W[np.ix_(core, core)]
    smax = float(np.max(np.abs(sub))) if np.any(sub) else 1.0
    nodes = [f"n{i}" for i in range(n)]
    ang = [np.radians(90 + 360.0 * i / n) for i in range(n)]
    pos = {f"n{i}": (2.35 * np.cos(t), 2.35 * np.sin(t)) for i, t in enumerate(ang)}
    eds = [(f"n{j}", f"n{i}", float(sub[i, j]))          # target i, regulator j
           for i in range(n) for j in range(n)
           if i != j and abs(float(sub[i, j])) >= 0.14 * smax]
    return lambda ax: sch.pl.draw_grn_mpl(
        ax, nodes, pos, eds, wmax=smax, labels=False, act_color=WARM, rep_color=COOL,
        node_size=55, node_face="white", node_edge=INK, node_lw=0.7,
        edge_lw=(0.40, 1.25), rad=0.11, shrink=4, alpha=1.0)

TIKZ_LINEAGE = r"""
\begin{tikzpicture}[
  ct/.style={circle,draw=muted,fill=faint!70,line width=0.8pt,minimum size=4.6mm,inner sep=0pt},
  lk/.style={-{Latex[length=3.4pt,width=2.8pt]},muted,line width=0.9pt,shorten >=1.2pt,shorten <=1.2pt}]
  \node[ct] (p) at (0,0) {};  \node[ct] (m) at (1.5,0) {};
  \node[ct] (a) at (3.0,0.75) {};  \node[ct] (b) at (3.0,-0.75) {};
  \node[ct] (a2) at (4.4,1.15) {}; \node[ct] (b2) at (4.4,-1.15) {};
  \draw[lk] (p) -- (m); \draw[lk] (m) -- (a); \draw[lk] (m) -- (b);
  \draw[lk] (a) -- (a2); \draw[lk] (b) -- (b2);
\end{tikzpicture}
"""


def _nodes_tex():
    s = ""
    for i, (px, py) in enumerate(REG_XY):
        s += rf"\node[reg] (r{i}) at ({px},{py}) {{}};" + "\n"
    for i, (px, py) in enumerate(TGT_XY):
        s += rf"\node[tgt] (t{i}) at ({px},{py}) {{}};" + "\n"
    return s


def prior_tex(in_prior):
    """The prior. Its edges are UNSIGNED on purpose: a promoter-based prior asserts that a regulator
    MAY act on a target, not whether it activates or represses."""
    s = TIKZ_NODES + "\n" + _nodes_tex()
    s += (r"\tikzset{edge/.style={-{Latex[length=3.4pt,width=2.8pt]},muted,line width=0.9pt,"
          r"shorten >=1.4pt,shorten <=1.4pt}}" + "\n")
    for a, b in EDGES:
        if (a, b) in in_prior:
            s += rf"\draw[edge] (r{a}) -- (t{b});" + "\n"
    return s + r"\end{tikzpicture}"


def fitted_network_tex(W, reg_idx, tgt_idx, wmax, extra):
    """A fitted network on the SAME node layout as the prior, with real signs and magnitudes.
    Activation takes an arrowhead in the activation color, repression a flat bar in the repression
    color, so sign survives grayscale. The dashed edges are the schematic ones (see EXTRA_EDGES)."""
    s = TIKZ_NODES + "\n" + _nodes_tex()
    for a, b in EDGES:
        w = float(W[tgt_idx[b], reg_idx[a]])
        if abs(w) < 1e-9:
            continue
        lw = 0.45 + 1.25 * min(abs(w) / wmax, 1.0)
        col = "warm" if w > 0 else "cool"
        headsp = (r"-{Latex[length=3.6pt,width=3.0pt]}" if w > 0 else r"-{Bar[width=3.6pt]}")
        s += (rf"\draw[{headsp},{col},line width={lw:.2f}pt,shorten >=1.5pt,shorten <=1.5pt]"
              rf" (r{a}) -- (t{b});" + "\n")
    for a, b, sgn in extra:
        col = "warm" if sgn > 0 else "cool"
        headsp = (r"-{Latex[length=3.6pt,width=3.0pt]}" if sgn > 0 else r"-{Bar[width=3.6pt]}")
        s += (rf"\draw[{headsp},{col},line width=0.8pt,shorten >=1.5pt,shorten <=1.5pt,"
              rf"dash pattern=on 1.8pt off 1.4pt] (r{a}) -- (t{b});" + "\n")
    return s + r"\end{tikzpicture}"


TEX_MODEL = r"""
\begin{tikzpicture}
\node[inner sep=0pt] {$\displaystyle
  \frac{dx_i}{dt} \;=\; \sum_{j} \textcolor{pW}{W^{(c)}_{ij}}\,\textcolor{pPhi}{\varphi_j(x_j)}
  \;-\; \textcolor{pGam}{\gamma_i}\, x_i \;+\; \textcolor{pI}{I_i} $};
\end{tikzpicture}
"""

TEX_KEY = r"""
\begin{tikzpicture}
\node[inner sep=0pt] {\footnotesize
\begin{tabular}{@{}l@{\hspace{7pt}}l@{}}
$\textcolor{pW}{W}$ & \textcolor{muted}{interaction, per cell type}\\[2pt]
$\textcolor{pPhi}{\varphi}$ & \textcolor{muted}{activation, per gene}\\[2pt]
$\textcolor{pGam}{\gamma}$ & \textcolor{muted}{decay, per gene}\\[2pt]
$\textcolor{pI}{I}$ & \textcolor{muted}{external bias}\\
\end{tabular}};
\end{tikzpicture}
"""

TEX_LOSS = r"""
\begin{tikzpicture}
\node[inner sep=0pt] {\footnotesize
\begin{tabular}{@{}l@{\hspace{9pt}}l@{}}
$\lambda_{\mathrm{rec}}\big\|\mathbf{v}-\dot{\mathbf{x}}\big\|^{2}$
  & \textcolor{muted}{fidelity to the observed dynamics}\\[10pt]
$\lambda_{\mathrm{scaf}}\big\|\textcolor{pW}{W}\!\circ\!(1-S)\big\|$
  & \textcolor{muted}{penalize interactions absent from the prior}\\[10pt]
$\lambda_{\mathrm{bias}}\big\|\textcolor{pI}{\mathbf{I}}\big\|_{1}$
  & \textcolor{muted}{keep the external bias sparse}\\[10pt]
$\lambda_{\mathrm{bnd}}\big\langle r(\mathbf{x})\big\rangle_{+}$
  & \textcolor{muted}{keep trajectories bounded}\\
\end{tabular}};
\end{tikzpicture}
"""

TEX_HILL = r"""
\begin{tikzpicture}
\node[inner sep=0pt] {$\displaystyle
  \textcolor{pPhi}{\varphi(x)} = \frac{x^{n}}{x^{n}+k^{n}} $};
\end{tikzpicture}
"""

TEX_HILL_KEY = r"""
\begin{tikzpicture}
\node[inner sep=0pt] {\footnotesize
\begin{tabular}{@{}l@{\hspace{7pt}}l@{}}
$n$ & \textcolor{muted}{Hill coefficient}\\[3pt]
$k$ & \textcolor{muted}{half-saturation threshold}\\
\end{tabular}};
\end{tikzpicture}
"""

TEX_ENERGY = r"""
\begin{tikzpicture}
\node[inner sep=0pt] {$\displaystyle E =
  -\tfrac{1}{2}\,\boldsymbol{\sigma}^{\!\top} \textcolor{pW}{W}\, \boldsymbol{\sigma}
  + \sum_i \textcolor{pGam}{\gamma_i}\!\!\int_0^{\sigma_i}\!\!
        \textcolor{pPhi}{\varphi_i^{-1}}
  - \textcolor{pI}{\mathbf{I}}^{\!\top}\boldsymbol{\sigma} $};
\end{tikzpicture}
"""

TEX_JACOBIAN = r"""
\begin{tikzpicture}
\node[inner sep=0pt] {$\displaystyle J = \textcolor{pW}{W}\,
  \mathrm{diag}\!\big(\textcolor{pPhi}{\varphi'}\big)
  - \mathrm{diag}\!\big(\textcolor{pGam}{\boldsymbol{\gamma}}\big) $};
\end{tikzpicture}
"""

def circular_network_tex(W, core, out):
    """The regulator core as a circular-layout graph, drawn in TikZ so that repression carries the
    house flat-bar head. matplotlib has no bar arrowstyle; its nearest option is a bracket, which
    looks like punctuation rather than a repression glyph.

    That is why TikZ is preferred, and it is not a reason to draw nothing where TeX is missing:
    circular_network_fallback() draws the same graph with the bracket, which is worse typography
    and the same claim. A blank panel is the only option here that says something false."""
    n = len(core)
    sub = W[np.ix_(core, core)]
    smax = float(np.max(np.abs(sub))) if np.any(sub) else 1.0
    omax = float(np.max(out[core])) or 1.0
    s = (r"\begin{tikzpicture}["
         r"gene/.style={circle,draw=ink,fill=white,line width=0.7pt,inner sep=0pt},"
         r"act/.style={-{Latex[length=3.0pt,width=2.6pt]},warm,shorten >=1.6pt,shorten <=1.6pt},"
         r"rep/.style={-{Bar[width=3.0pt]},cool,shorten >=1.8pt,shorten <=1.8pt}]" + "\n")
    for i in range(n):
        ang = 90 + 360.0 * i / n
        mm = 2.4 + 2.4 * (float(out[core[i]]) / omax)
        s += rf"\node[gene,minimum size={mm:.2f}mm] (n{i}) at ({ang:.1f}:2.35) {{}};" + "\n"
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = float(sub[i, j])                        # target i, regulator j
            if abs(v) < 0.14 * smax:
                continue
            lw = 0.40 + 1.25 * min(abs(v) / smax, 1.0)
            sty = "act" if v > 0 else "rep"
            s += rf"\draw[{sty},line width={lw:.2f}pt] (n{j}) to[bend left=11] (n{i});" + "\n"
    return s + r"\end{tikzpicture}"


TEX_KO = r"""
\begin{tikzpicture}
\node[inner sep=0pt] {$\displaystyle \dot{\mathbf{x}} =
  \textcolor{pW}{W}\textcolor{pPhi}{\varphi(\mathbf{x})}
  - \textcolor{pGam}{\boldsymbol{\gamma}}\mathbf{x} + \textcolor{pI}{\mathbf{I}},
  \;\; x_g \equiv 0 $};
\end{tikzpicture}
"""


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def grid_quiver(ax, emb, F, *, n_grid=14, min_count=4, color=INK, size=0.92, width=0.010, zorder=5):
    """Bin an embedding-space field into a square grid, average it, and quiver with the arrow length
    set in DATA coordinates so the 90th-percentile arrow is a fixed fraction of one grid cell."""
    emb = np.asarray(emb, float)[:, :2]
    F = np.asarray(F, float)[:, :2]
    mag = np.linalg.norm(F, axis=1)
    if (mag > 0).any():
        cap = float(np.percentile(mag[mag > 0], 99))
        if cap > 0:
            F = F * np.minimum(1.0, cap / (mag + 1e-12))[:, None]
    (xmn, ymn), (xmx, ymx) = emb.min(0), emb.max(0)
    xe = np.linspace(xmn, xmx, n_grid + 1); ye = np.linspace(ymn, ymx, n_grid + 1)
    ix = np.clip(np.digitize(emb[:, 0], xe) - 1, 0, n_grid - 1)
    iy = np.clip(np.digitize(emb[:, 1], ye) - 1, 0, n_grid - 1)
    gx, gy, gu, gv = [], [], [], []
    for i in range(n_grid):
        for j in range(n_grid):
            m = (ix == i) & (iy == j)
            if m.sum() < min_count:
                continue
            gx.append(0.5 * (xe[i] + xe[i + 1])); gy.append(0.5 * (ye[j] + ye[j + 1]))
            gu.append(F[m, 0].mean()); gv.append(F[m, 1].mean())
    if not gx:
        return
    gu, gv = np.asarray(gu), np.asarray(gv)
    gmag = np.hypot(gu, gv)
    cell = float(np.mean([xmx - xmn, ymx - ymn])) / n_grid
    ref = float(np.percentile(gmag[gmag > 0], 90)) if np.any(gmag > 0) else 1.0
    qs = ref / (0.9 * size * cell) if cell > 0 else 1.0
    ax.quiver(gx, gy, gu, gv, color=color, angles="xy", scale_units="xy", scale=qs,
              width=width, headwidth=4.0, headlength=4.4, headaxislength=3.8, zorder=zorder)
    # An arrow near the edge of the embedding, pointing outward, runs past the autoscaled limits and
    # is drawn outside the panel. Grow the limits to contain the longest arrow.
    pad = float(gmag.max()) / qs if qs > 0 else 0.0
    b = getattr(ax, "_flow_bounds", None)
    nb = (xmn - pad, xmx + pad, ymn - pad, ymx + pad)
    ax._flow_bounds = nb if b is None else (min(b[0], nb[0]), max(b[1], nb[1]),
                                            min(b[2], nb[2]), max(b[3], nb[3]))


def finish_flow(ax, fig):
    """Apply the bounds the quiver arrows need, expanded to an equal data aspect by hand.

    set_aspect("equal", adjustable="datalim") recomputes the limits from the artists and throws away
    anything set beforehand, so the outward-pointing arrows at the edge of an embedding were drawn
    outside the axes. Expanding the shorter dimension to the axes' own width-to-height ratio gives an
    undistorted embedding AND keeps every arrow inside.
    """
    b = getattr(ax, "_flow_bounds", None)
    if b is None:
        ax.set_aspect("equal", adjustable="datalim")
        return
    x0, x1, y0, y1 = b
    pos = ax.get_position()
    aw, ah = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    if dx / dy > aw / max(ah, 1e-9):
        need = dx * ah / aw
        c = 0.5 * (y0 + y1); y0, y1 = c - need / 2, c + need / 2
    else:
        need = dy * aw / ah
        c = 0.5 * (x0 + x1); x0, x1 = c - need / 2, c + need / 2
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("auto")


def _dense(a, key):
    M = a.varp[key]
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)


def _layer(a, key):
    L = a.layers[key]
    return np.asarray(L.todense()) if hasattr(L, "todense") else np.asarray(L, dtype=float)


def scaffold_membership(path, reg_names, tgt_names):
    """Which of the DRAWN (regulator, target) pairs are genuinely in the real prior, so panel b and
    panel e correspond to each other and to the fit."""
    if not path or not os.path.exists(path):
        return None
    g = pd.read_parquet(path)
    cols = {c.lower(): c for c in g.columns}
    sub = g[g["gene_short_name"].isin(list(tgt_names))]
    present = set()
    for ri, r in enumerate(reg_names):
        col = cols.get(str(r).lower())
        if col is None:
            continue
        hit = sub.groupby("gene_short_name")[col].max()
        for ti, t in enumerate(tgt_names):
            v = hit.get(t, 0)
            if float(v if v == v else 0) > 0:
                present.add((ri, ti))
    return present


def choose_targets(a, types, reg_idx, tot, pool=26):
    """Pick the four target genes panel e draws.

    Selecting purely by inflow strength is NOT safe. The fitted networks are roughly 55 percent
    activating overall, but the very strongest edges out of the top regulators are repressive, so a
    top-strength pick drew six and seven of eight edges as repression and would have told the reader
    the inferred networks are mostly inhibitory. That is a selection artifact, not a result.
    """
    from itertools import combinations
    mats = [_dense(a, f"W_{t}") for t in types]
    target_pos = np.mean([np.mean(M[np.abs(M) > 1e-9] > 0) for M in mats])
    inflow = tot[:, reg_idx].sum(1).copy()
    inflow[reg_idx] = -1
    cand = list(np.argsort(-inflow)[:pool])
    best, best_score = None, np.inf
    for combo in combinations(cand, len(TGT_XY)):
        drawn = [[M[combo[b], reg_idx[r]] for r, b in EDGES] for M in mats]
        if min(min(abs(v) for v in d) for d in drawn) < 1e-3:
            continue
        frac = np.mean([np.mean([v > 0 for v in d]) for d in drawn])
        if abs(frac - target_pos) < best_score:
            best, best_score = combo, abs(frac - target_pos)
    return list(best) if best is not None else list(np.argsort(-inflow)[:len(TGT_XY)])


# --------------------------------------------------------------------------- #
# a, b
# --------------------------------------------------------------------------- #
def panel_a(fig, x, y, w, h, a, cache, rng):
    head(fig, x, y, "a", "Inputs")
    spliced = a.uns.get("scHopfield", {}).get("spliced_key", "Ms")
    X = _layer(a, spliced)
    used = np.where(a.var["scHopfield_used"].values.astype(bool))[0]
    pool = used[np.argsort(-X[:, used].std(0))[:40]]
    genes = np.sort(rng.choice(pool, 16, replace=False))
    cells = np.sort(rng.choice(X.shape[0], 24, replace=False))
    blk = X[np.ix_(cells, genes)].T
    blk = blk[np.argsort(np.argmax(blk, axis=1))]
    emb = np.asarray(cache["emb"])[:, :2]

    bw = 0.80
    ax1 = fig.add_axes(rect(x + 0.12, y + 0.36, bw, 0.92)); bare(ax1)
    ax1.imshow(blk, aspect="auto", cmap=CM_EXPR, interpolation="nearest",
               vmin=0, vmax=np.percentile(blk, 98))
    lab(fig, x + 0.12 + bw / 2, y + 0.33, "expression")
    plus(fig, x + 0.12 + bw + 0.13, y + 0.84)

    # A DIFFERENT block of genes and cells, so the unspliced panel is not a recolored copy of the
    # spliced one; they are different measurements and should not look like the same matrix twice.
    ug = np.sort(rng.choice(pool, 16, replace=False))
    uc = np.sort(rng.choice(X.shape[0], 24, replace=False))
    U = _layer(a, "Mu")[np.ix_(uc, ug)].T
    U = U[np.argsort(np.argmax(U, axis=1))]

    mw, lw_ = 0.94, 0.16                        # labels sit to the LEFT, rotated, to save the title rows
    xu = x + 0.12 + bw + 0.26 + lw_
    ax2 = fig.add_axes(rect(xu, y + 0.28, mw, 0.62)); bare(ax2)
    ax2.imshow(U, aspect="auto", cmap=CM_UNSPL, interpolation="nearest",
               vmin=0, vmax=max(float(np.percentile(U, 98)), 1e-9))
    fig.text(fx(xu - 0.05), fy(y + 0.59), "unspliced counts", fontsize=5.4, color=INK,
             rotation=90, ha="center", va="center")
    fig.text(fx(xu + mw / 2), fy(y + 1.02), "or", fontsize=5.8, color=FAINT,
             ha="center", va="center")
    ax3 = fig.add_axes(rect(xu, y + 1.10, mw + 0.10, 0.66)); bare(ax3)
    if PSEUDOTIME in a.obs:
        ax3.scatter(emb[:, 0], emb[:, 1], c=a.obs[PSEUDOTIME].values.astype(float),
                    cmap=CM_PTIME, s=1.5, linewidths=0)
    ax3.set_aspect("equal", adjustable="datalim")
    fig.text(fx(xu - 0.05), fy(y + 1.43), "pseudotime", fontsize=5.4, color=INK,
             rotation=90, ha="center", va="center")

    xa = xu + mw + 0.10
    arrow(fig, xa, y + 0.92, xa + 0.20, y + 0.92)
    xf = xa + 0.28
    wf = w - (xf - x) - 0.12
    axf = fig.add_axes(rect(xf, y + 0.42, wf, h - 0.54)); bare(axf)
    axf.scatter(emb[:, 0], emb[:, 1], s=1.2, c="#dde3e2", linewidths=0, zorder=1)
    grid_quiver(axf, emb, cache["wt_flow"], color=INK, n_grid=14, size=0.95)
    finish_flow(axf, fig)
    lab(fig, xf + wf / 2, y + 0.32, "estimated dynamics", size=5.6)


def panel_b(fig, x, y, w, h, in_prior):
    head(fig, x, y, "b", "Optional inputs")
    sub = (w - 0.26) / 2
    for j, (body, title, badge, fb) in enumerate((
            (prior_tex(in_prior), "regulatory prior", "recommended", prior_fallback(in_prior)),
            (TIKZ_LINEAGE, "cell-type relationships", "optional", "lineage tree"))):
        xx = x + 0.09 + j * (sub + 0.08)
        panel_box(fig, xx, y + 0.28, sub, h - 0.56, lw=0.7, edge=SOFT, ls=(0, (2.2, 1.6)))
        tex_panel(fig, xx + 0.05, y + 0.34, sub - 0.10, h - 0.72, body, fallback=fb, fontsize=5.4)
        lab(fig, xx + sub / 2, y + 0.25, title)
        lab(fig, xx + sub / 2, y + h - 0.09, badge, size=5.0, color=MUTED)


# --------------------------------------------------------------------------- #
# c, d, e
# --------------------------------------------------------------------------- #
def panel_c(fig, x, y, w, h, a, genes):
    head(fig, x, y, "c", "Gene activation")
    tex_panel(fig, x + 0.06, y + 0.20, w - 0.12, 0.30, TEX_HILL,
              fallback=r"$\varphi(x)=x^n/(x^n+k^n)$")
    tex_panel(fig, x + 0.08, y + 0.58, w - 0.16, 0.32, TEX_HILL_KEY,
              fallback=r"$k$ threshold,  $n$ steepness", fontsize=5.4)
    ax = fig.add_axes(rect(x + 0.32, y + 1.06, w - 0.46, h - 1.28))
    spliced = a.uns.get("scHopfield", {}).get("spliced_key", "Ms")
    X = _layer(a, spliced)
    for g, (col, lw) in zip(genes, ((SOFT, 1.6), (PPHI, 2.0))):
        gi = a.var_names.get_loc(g)
        k1 = float(a.var["sigmoid_threshold"].values[gi]); n1 = float(a.var["sigmoid_exponent"].values[gi])
        k2 = float(a.var["sigmoid_threshold2"].values[gi]); n2 = float(a.var["sigmoid_exponent2"].values[gi])
        mix = float(a.var["sigmoid_mix"].values[gi])
        xg = X[:, gi]; xg = xg[np.isfinite(xg)]
        hi = float(np.percentile(xg, 99)) or 1.0
        xs = np.linspace(0, hi, 400)
        ax.plot(xs / hi, mix * sigmoid(xs, k1, n1) + (1 - mix) * sigmoid(xs, k2, n2),
                color=col, lw=lw, solid_capstyle="round")
    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.06)
    thin_axes(ax, "expression", None, size=5.4)
    ax.set_ylabel("activation", fontsize=5.4, color=PPHI, labelpad=2.2)
    ax.annotate("unimodal", (0.48, 0.54), fontsize=5.4, color=SOFT, ha="left")
    ax.annotate("bimodal", (0.20, 0.20), fontsize=5.4, color=PPHI, ha="left")


def panel_d(fig, x, y, w, h):
    """The hub: the governing equation, a key for its parameters, and the objective listed term by
    term. The key and the objective are TYPESET, not matplotlib text, so the symbols and their
    explanations share one typeface."""
    head(fig, x, y, "d", "The fitted system")
    tex_panel(fig, x + 0.08, y + 0.20, w - 0.16, 0.32, TEX_MODEL,
              fallback=r"$dx_i/dt = \sum_j W_{ij}\varphi_j(x_j) - \gamma_i x_i + I_i$")
    tex_panel(fig, x + 0.10, y + 0.60, w - 0.20, 0.54, TEX_KEY,
              fallback="\n".join(("$W$ interactions,  $\\varphi$ activation,",
                                  "$\\gamma$ degradation,  $I$ bias")), fontsize=5.4)
    # the phrase IS the separator; a rule as well was one divider too many
    lab(fig, x + w / 2, y + 1.30, "fitted by minimizing the sum of", size=6.0, color=INK)
    tex_panel(fig, x + 0.10, y + 1.42, w - 0.20, h - 1.54, TEX_LOSS,
              fallback=r"$\|\dot{x} - (W\varphi - \gamma x + I)\|^2 + $ penalties", fontsize=5.4)


def panel_e(fig, x, y, w, h, a, types, reg_idx, tgt_idx):
    """Stacked vertically so each network is as large as the column allows."""
    head(fig, x, y, "e", "Fitted networks")
    mats = [_dense(a, f"W_{t}") for t in types]
    sel = [M[np.ix_(tgt_idx, reg_idx)] for M in mats]
    wmax = float(np.max(np.abs(np.concatenate([m.ravel() for m in sel])))) or 1.0
    nh = (h - 0.80) / 2
    for j, M in enumerate(mats):
        yy = y + 0.28 + j * (nh + 0.18)
        lab(fig, x + w / 2, yy - 0.03, f"cell type {j + 1}")
        tex_panel(fig, x + 0.08, yy, w - 0.16, nh,
                  fitted_network_tex(M, reg_idx, tgt_idx, wmax, EXTRA_EDGES[j]),
                  fallback=fitted_network_fallback(M, reg_idx, tgt_idx, wmax, EXTRA_EDGES[j]))

    ky = y + h - 0.19
    lab(fig, x + 0.12, ky, "activation", size=5.2, color=WARM, ha="left")
    arrow(fig, x + 0.62, ky - 0.02, x + 0.78, ky - 0.02, color=WARM, lw=1.0, ms=6)
    lab(fig, x + 0.90, ky, "repression", size=5.2, color=COOL, ha="left")
    fig.lines.append(plt.Line2D([fx(x + 1.40), fx(x + 1.52)], [fy(ky - 0.02)] * 2,
                                transform=fig.transFigure, color=COOL, linewidth=1.0))
    fig.lines.append(plt.Line2D([fx(x + 1.52)] * 2, [fy(ky - 0.045), fy(ky + 0.005)],
                                transform=fig.transFigure, color=COOL, linewidth=1.2))
    ky2 = ky + 0.15
    lab(fig, x + 0.12, ky2, "off-scaffold", size=5.2, color=MUTED, ha="left")
    fig.lines.append(plt.Line2D([fx(x + 0.70), fx(x + 0.92)], [fy(ky2 - 0.02)] * 2,
                                transform=fig.transFigure, color=SOFT, linewidth=1.0,
                                linestyle=(0, (1.8, 1.4))))


# --------------------------------------------------------------------------- #
# f, g, h, i
# --------------------------------------------------------------------------- #
def panel_f(fig, x, y, w, h, a, basis, eq_s=None):
    head(fig, x, y, "f", "Energy landscape")
    tex_panel(fig, x + 0.03, y + 0.22, w - 0.06, 0.30, TEX_ENERGY, scale=eq_s,
              fallback=r"$E = -\frac{1}{2}\sigma^T W \sigma + \dots$")
    ax = fig.add_axes(rect(x + 0.18, y + 0.44, w - 0.26, h - 0.60), projection="3d")
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    z = a.obs["energy_total"].values.astype(float)
    ax.scatter(emb[:, 0], emb[:, 1], z, c=z, cmap=CM_ENERGY, s=1.5, depthshade=False, linewidths=0)
    ax.set_axis_off()
    ax.set_xlim(emb[:, 0].min(), emb[:, 0].max()); ax.set_ylim(emb[:, 1].min(), emb[:, 1].max())
    ax.set_zlim(z.min(), z.max())
    ax.set_box_aspect((1, 1, 0.62), zoom=1.20)
    ax.view_init(elev=22, azim=-66)
    # The energy axis sits OUTSIDE the point cloud, at the panel's left margin, so it never overlaps
    # the distribution the way an in-cloud spike did.
    # The coordinate frame sits just outside the cloud's lower-left corner: parked in the panel
    # corner it read as a separate object, but pushed any closer its arms run into the points. Every
    # label carries a white halo so it stays legible if the cloud shifts on a re-fit.
    ox_, oy_ = x + 0.20, y + 2.04
    fig.patches.append(FancyArrowPatch((fx(ox_), fy(oy_)), (fx(ox_), fy(oy_ - 0.62)),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                       linewidth=0.9, color=SOFT, shrinkA=0, shrinkB=0, zorder=6))
    arrow(fig, ox_, oy_, ox_ + 0.16, oy_ - 0.06, color=SOFT, lw=0.7, ms=5)
    arrow(fig, ox_, oy_, ox_ + 0.13, oy_ + 0.10, color=SOFT, lw=0.7, ms=5)
    for tx, ty, s, rot, ha, va in ((ox_ + 0.04, oy_ - 0.31, "energy", 90, "left", "center"),
                                   (ox_ + 0.18, oy_ - 0.08, "UMAP1", 0, "left", "center"),
                                   (ox_ + 0.15, oy_ + 0.14, "UMAP2", 0, "left", "center")):
        tt = fig.text(fx(tx), fy(ty), s, fontsize=5.2, color=FAINT, rotation=rot,
                      ha=ha, va=va, zorder=7)
        tt.set_path_effects([pe.withStroke(linewidth=1.7, foreground="white")])


def panel_g(fig, x, y, w, h, a, basis, rng, eq_s=None):
    """Stability AND rotation, since both come from the same Jacobian. The equation is the provenance:
    it is built from the interaction matrix, the activation derivative and the decay, and from nothing
    else. The bias is absent because it is constant and its derivative vanishes."""
    head(fig, x, y, "g", "Local stability: Jacobian")
    tex_panel(fig, x + 0.05, y + 0.22, w - 0.10, 0.30, TEX_JACOBIAN, scale=eq_s,
              fallback=r"$J = W\,\mathrm{diag}(\varphi') - \mathrm{diag}(\gamma)$")
    ax = fig.add_axes(rect(x + 0.30, y + 0.62, w - 0.44, 0.62))
    E = np.asarray(a.obsm["jacobian_eigenvalues"])
    ev = E[rng.choice(E.shape[0], min(900, E.shape[0]), replace=False)]
    keep = np.argsort(-ev.real, axis=1)[:, :min(30, ev.shape[1])]
    sel = np.take_along_axis(ev, keep, axis=1).ravel()
    re, im = sel.real, sel.imag
    vmax = float(np.percentile(np.abs(re), 99.5)) or 1.0
    ax.scatter(re, im, c=re, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=0.9, linewidths=0, alpha=0.55)
    ax.axvline(0, color=SOFT, lw=0.8, ls="--"); ax.axhline(0, color=RULE, lw=0.6, ls=":")
    # Most leading eigenvalues are real, so the y-scale comes from the COMPLEX subpopulation; scaling
    # by all |Im| would collapse the panel onto the real axis and hide the oscillatory modes.
    cx = np.abs(im[np.abs(im) > 1e-9])
    yr = float(np.percentile(cx, 90)) if cx.size else float(np.percentile(np.abs(im), 99.9)) or 1.0
    ax.set_xlim(-vmax * 1.1, vmax * 1.1); ax.set_ylim(-yr * 1.35, yr * 1.35)
    thin_axes(ax, "real part", "imaginary part", size=5.2)
    ax.text(0.03, 1.06, "stable", fontsize=5.2, color=COOL, ha="left", transform=ax.transAxes)
    ax.text(0.97, 1.06, "unstable", fontsize=5.2, color=WARM, ha="right", transform=ax.transAxes)

    # Stability and rotation side by side, both REAL and both over the embedding, since both fall
    # out of the same Jacobian. The stability map uses jacobian_leading_real (the maximum real part).
    # It must NEVER use jacobian_eig1_real, which is the arbitrary index-0 eigenvalue and inverts the
    # conclusion; FINDINGS M20 records that trap.
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    sub_w = (w - 0.32) / 2
    lead = a.obs["jacobian_leading_real"].values.astype(float)
    # Saturate at the 90th percentile, not the 98th: most cells sit at the stability floor, so a
    # wider symmetric range washes the unstable minority out to near-white and the map says nothing.
    lim = float(np.percentile(np.abs(lead), 90)) or 1.0
    axs = fig.add_axes(rect(x + 0.12, y + 1.52, sub_w, h - 1.66)); bare(axs)
    axs.scatter(emb[:, 0], emb[:, 1], c=lead, cmap="RdBu_r", vmin=-lim, vmax=lim,
                s=1.3, linewidths=0)
    axs.set_aspect("equal", adjustable="datalim")
    lab(fig, x + 0.12 + sub_w / 2, y + 1.49, "stability index", size=5.2, color=MUTED)

    rot = a.obs["jacobian_rotational"].values.astype(float)
    axr = fig.add_axes(rect(x + 0.20 + sub_w, y + 1.52, sub_w, h - 1.66)); bare(axr)
    axr.scatter(emb[:, 0], emb[:, 1], c=rot, cmap=CM_ROT, s=1.1, linewidths=0,
                vmin=float(np.percentile(rot, 2)), vmax=float(np.percentile(rot, 98)))
    axr.set_aspect("equal", adjustable="datalim")
    lab(fig, x + 0.20 + sub_w + sub_w / 2, y + 1.49, "rotational part", size=5.2, color=MUTED)


def panel_h(fig, x, y, w, h, a, types):
    """Network structure: the fitted interaction matrix read as a graph. The circular layout shows the
    regulator core; the scatter below shows only REGULATORS, since a non-regulator gene has exactly
    zero out-strength under the transcription-factor column mask and would pile on the axis."""
    head(fig, x, y, "h", "Network structure")
    W = _dense(a, f"W_{types[0]}")
    out = np.abs(W).sum(0)
    reg = np.where(out > 0)[0]
    core = reg[np.argsort(-out[reg])[:12]]

    tex_panel(fig, x + 0.05, y + 0.20, w - 0.10, 1.08, circular_network_tex(W, core, out),
              fallback=circular_network_fallback(W, core, out))

    ax = fig.add_axes(rect(x + 0.29, y + 1.46, w - 0.43, 0.82))
    inn = np.abs(W).sum(1)
    ox, iy = out[reg], np.maximum(inn[reg], 1e-4)
    qo, qi = float(np.percentile(ox, 75)), float(np.percentile(iy, 75))
    # Two wings, named as in the network-structure figure: pure sources with high out-strength and
    # little incoming regulation, against relays that are themselves heavily regulated.
    # A dead band around each threshold: genes sitting on the line are left unmarked, since a
    # cluster of marked points straddling the crossing reads as noise rather than as two populations.
    # A gene is marked when it is clearly clear of the OPPOSITE threshold, however close it sits to
    # its own. That excludes the cluster at the crossing automatically, without discarding genes far
    # out along one axis.
    m_ = 0.16                                          # margin in log units
    master = (ox > qo) & (np.log10(iy) < np.log10(qi) - m_)
    relay = (iy > qi) & (np.log10(ox) < np.log10(qo) - m_)
    rest = ~(master | relay)
    ax.scatter(ox[rest], iy[rest], s=1.4, c=FAINT, linewidths=0, alpha=0.55)
    ax.scatter(ox[relay], iy[relay], s=4.0, c=INK, marker="^", linewidths=0)
    ax.scatter(ox[master], iy[master], s=3.0, c=INK, linewidths=0)
    ax.axvline(qo, color=SOFT, lw=0.6, ls="--"); ax.axhline(qi, color=SOFT, lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    thin_axes(ax, "out-strength", "in-strength", size=5.2)
    ax.text(0.97, 0.06, "master\nregulators", fontsize=5.2, color=INK, ha="right", va="bottom",
            transform=ax.transAxes, linespacing=1.1)
    ax.text(0.03, 0.94, "relays", fontsize=5.2, color=INK, ha="left", va="top",
            transform=ax.transAxes)
    lab(fig, x + w / 2, y + 1.42, "regulators only", size=5.2, color=MUTED)


def panel_i(fig, x, y, w, h, cache, gene_sim, gene_proj, eq_s=None):
    """A knockout is the same fitted system with one coordinate held at zero, integrated forward.
    Read top to bottom: the perturbed system, the integration, the response per cell type, and the
    redirected flow. Gray is the unperturbed field and orange the CHANGE the knockout induces; the
    perturbed field itself retains the developmental component and would land on the unperturbed
    arrows."""
    head(fig, x, y, "i", "In-silico perturbation")
    tex_panel(fig, x + 0.03, y + 0.22, w - 0.06, 0.30, TEX_KO, scale=eq_s,
              fallback=r"$\dot{x} = W\varphi(x) - \gamma x + I,\ x_g \equiv 0$")
    arrow(fig, x + w / 2, y + 0.52, x + w / 2, y + 0.64, color=ACCENT, lw=1.0, ms=7)
    lab(fig, x + w / 2, y + 0.78, "ODE integration", size=5.6, color=INK)
    axt = fig.add_axes(rect(x + 0.34, y + 0.84, w - 0.48, 0.46))
    c = cache.get("cascade")
    # The dataset's own cell-type colors sat too close to the parameter colors used in every
    # equation, so the curves read as the terms of the model. A lightness ramp inside the
    # perturbation hue keeps them a perturbation readout and clashes with nothing.
    if c is not None and len(c):
        sub = c[c["perturbation"].astype(str).str.startswith(str(gene_sim))]
        names = sorted(sub["cluster"].astype(str).unique())
        ramp = LinearSegmentedColormap.from_list(
            "pert", ["#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"])
        shade = {n: ramp(i / max(len(names) - 1, 1)) for i, n in enumerate(names)}
        for name, grp in sub.groupby("cluster"):
            grp = grp.sort_values("t")
            axt.plot(grp["t"].values, grp["mean_abs_delta"].values,
                     color=shade[str(name)], lw=0.85)
    thin_axes(axt, "time", "response", size=5.2)

    ky = y + 1.52                                   # the key sits ABOVE the plot, never on it
    fig.lines.append(plt.Line2D([fx(x + 0.12), fx(x + 0.26)], [fy(ky)] * 2,
                                transform=fig.transFigure, color=SOFT, linewidth=1.0))
    lab(fig, x + 0.30, ky + 0.02, "unperturbed", size=5.1, color=MUTED, ha="left")
    fig.lines.append(plt.Line2D([fx(x + 0.98), fx(x + 1.12)], [fy(ky)] * 2,
                                transform=fig.transFigure, color=PERT, linewidth=1.6))
    lab(fig, x + 1.16, ky + 0.02, "change", size=5.1, color=MUTED, ha="left")

    ax = fig.add_axes(rect(x + 0.08, y + 1.58, w - 0.16, h - 1.68)); bare(ax)
    emb = np.asarray(cache["emb"])[:, :2]
    wt = np.asarray(cache["wt_ode_flow"], float)[:, :2]
    delta = np.asarray(cache["ko_flow"][gene_proj], float)[:, :2] - wt
    ax.scatter(emb[:, 0], emb[:, 1], s=1.0, c="#e3e8e7", linewidths=0, zorder=1)
    grid_quiver(ax, emb, wt, color=SOFT, n_grid=11, size=0.85, width=0.008, zorder=4)
    grid_quiver(ax, emb, delta, color=PERT, n_grid=11, size=0.85, width=0.011, zorder=6)
    finish_flow(ax, fig)


def panel_j(fig, x, y, w, h):
    head(fig, x, y, "j", "Model capabilities")
    ax = fig.add_axes(rect(x + 0.10, y + 0.22, w - 0.20, h - 0.32)); bare(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cols = [0.585, 0.735, 0.885]
    for cx, l in zip(cols, SCOPE_COLS):
        for li, line in enumerate(l.split("\n")):
            ax.text(cx, 0.965 - li * 0.080, line, fontsize=5.3, color=FAINT, ha="center", va="center")
    ax.plot([0.01, 0.99], [0.775, 0.775], color=RULE, lw=0.8)
    n = len(SCOPE_ROWS)
    for r, (claim, marks) in enumerate(SCOPE_ROWS):
        yy = 0.690 - r * (0.500 / max(n - 1, 1))
        ax.text(0.012, yy, claim, fontsize=5.8, color=INK, ha="left", va="center")
        for cx, m in zip(cols, marks):
            if m == "yes":
                ax.scatter([cx], [yy], s=17, c=ACCENT, linewidths=0, zorder=3)
            elif m == "na":
                ax.scatter([cx], [yy], s=17, facecolors="none", edgecolors=FAINT,
                           linewidths=0.9, zorder=3)
            else:
                ax.scatter([cx], [yy], s=17, facecolors="none", edgecolors=INK,
                           linewidths=1.0, zorder=3)
                ax.plot([cx - 0.0105, cx + 0.0105], [yy, yy], color=INK, lw=1.0, zorder=4)
    ky = 0.035
    ax.plot([0.01, 0.99], [0.115, 0.115], color=RULE, lw=0.8)
    ax.scatter([0.014], [ky], s=15, c=ACCENT, linewidths=0)
    ax.text(0.030, ky, "shown here", fontsize=5.1, color=FAINT, ha="left", va="center")
    ax.scatter([0.145], [ky], s=15, facecolors="none", edgecolors=FAINT, linewidths=0.9)
    ax.text(0.161, ky, "not applicable at this tier", fontsize=5.1, color=FAINT, ha="left", va="center")
    ax.scatter([0.395], [ky], s=15, facecolors="none", edgecolors=INK, linewidths=1.0)
    ax.plot([0.3845, 0.4055], [ky, ky], color=INK, lw=1.0)
    ax.text(0.411, ky, "tested, does not transfer", fontsize=5.1, color=FAINT, ha="left", va="center")


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build(dataset="paul15_coarse"):
    use_style()
    rng = np.random.default_rng(0)
    a = ad.read_h5ad(f"{paths.REPORTS}/{dataset}/data/adata_analyzed.h5ad")
    with open(f"{paths.REPORTS}/{dataset}/data/perturb_dynamics.pkl", "rb") as fh:
        cache = pickle.load(fh)
    basis = basis_of(a)

    cts = [k[2:] for k in a.varp if k.startswith("W_") and k != "W_all"]
    tot = None
    for c in cts:
        M = np.abs(_dense(a, f"W_{c}"))
        tot = M if tot is None else tot + M
    out = tot.sum(0)
    reg_pool = np.where(out > 0)[0]
    reg_idx = list(reg_pool[np.argsort(-out[reg_pool])[:len(REG_XY)]])
    tgt_idx = choose_targets(a, CONTRAST[dataset], reg_idx, tot)
    in_prior = scaffold_membership(SCAFFOLD.get(dataset, ""),
                                   [str(a.var_names[i]) for i in reg_idx],
                                   [str(a.var_names[i]) for i in tgt_idx])
    if in_prior is None:
        in_prior = set(EDGES)

    fig = plt.figure(figsize=(FW, FH))
    fig.text(fx(0.50), fy(0.30), "scHopfield", fontsize=13.5, fontweight="bold",
             color=ACCENT, ha="left", va="baseline")
    fig.lines.append(plt.Line2D([fx(0.50), fx(7.10)], [fy(0.42), fy(0.42)],
                                transform=fig.transFigure, color=INK, linewidth=0.9))

    lane_span = [(0.52, 2.52), (2.70, 5.12), (5.18, 8.00), (8.12, 9.54)]
    lane_title_y = [0.62, 2.82, 5.28, 8.22]
    for (y0, y1), ty, (rail, title) in zip(lane_span, lane_title_y, LANES):
        bracket(fig, y0, y1, rail, color=ACCENT if rail == "ONE FITTED SYSTEM" else SOFT)
        fig.text(fx(0.50), fy(ty), title, fontsize=7.2, fontweight="bold", color=INK,
                 ha="left", va="baseline")

    # ---- lane I ----
    ya, ha_ = 0.70, 1.80
    panel_box(fig, 0.50, ya, 4.38, ha_); panel_a(fig, 0.50, ya, 4.38, ha_, a, cache, rng)
    panel_box(fig, 5.06, ya, 2.04, ha_); panel_b(fig, 5.06, ya, 2.04, ha_, in_prior)
    plus(fig, 4.95, ya + ha_ / 2)

    # The two feeders are placed SYMMETRICALLY about the centre arrow rather than at their panel
    # centres, and the translucent funnel is gone: it read as a stray shadow.
    fy_bus = 2.62
    for xs in (3.80 - 2.10, 3.80 + 2.10):
        fig.lines.append(plt.Line2D([fx(xs)] * 2, [fy(ya + ha_ + 0.04), fy(fy_bus)],
                                    transform=fig.transFigure, color=ACCENT, linewidth=0.9, zorder=0))
    fig.lines.append(plt.Line2D([fx(3.80 - 2.10), fx(3.80 + 2.10)], [fy(fy_bus)] * 2,
                                transform=fig.transFigure, color=ACCENT, linewidth=0.9, zorder=0))
    arrow(fig, 3.80, fy_bus, 3.80, 2.80, color=ACCENT, lw=1.1, ms=9)
    fig.text(fx(3.88), fy(2.72), "F I T", fontsize=6.2, fontweight="bold", color=ACCENT,
             ha="left", va="center")

    # ---- lane II ----
    yb, hb = 2.90, 2.20
    panel_box(fig, 0.50, yb, 1.72, hb); panel_c(fig, 0.50, yb, 1.72, hb, a, ACTIVATION[dataset])
    panel_box(fig, 2.34, yb, 2.70, hb, lw=1.5, edge=ACCENT, face="#f4f8f7")
    panel_d(fig, 2.34, yb, 2.70, hb)
    panel_box(fig, 5.16, yb, 1.94, hb)
    panel_e(fig, 5.16, yb, 1.94, hb, a, CONTRAST[dataset], reg_idx, tgt_idx)
    for x0, x1 in ((2.24, 2.32), (5.14, 5.06)):
        arrow(fig, x0, yb + hb / 2, x1, yb + hb / 2)

    # ---- the fan: four projections out of one box ----
    fan_y, drop_y = 5.34, 5.44
    fig.lines.append(plt.Line2D([fx(3.80)] * 2, [fy(yb + hb + 0.04), fy(fan_y)],
                                transform=fig.transFigure, color=ACCENT, linewidth=1.0, zorder=0))
    xs4 = (1.26, 3.02, 4.72, 6.30)
    fig.lines.append(plt.Line2D([fx(xs4[0]), fx(xs4[-1])], [fy(fan_y)] * 2,
                                transform=fig.transFigure, color=ACCENT, linewidth=1.0, zorder=0))
    for xs in xs4:
        arrow(fig, xs, fan_y, xs, drop_y, color=ACCENT, lw=1.0, ms=8)
    fig.text(fx(3.88), fy(5.24), "R E A D   O U T", fontsize=6.2, fontweight="bold", color=ACCENT,
             ha="left", va="center")

    # ---- lane III ----
    yc, hc = 5.48, 2.50
    eq_s = common_tex_scale([TEX_ENERGY, TEX_JACOBIAN, TEX_KO], [1.44, 1.78, 1.56])
    panel_box(fig, 0.50, yc, 1.52, hc); panel_f(fig, 0.50, yc, 1.52, hc, a, basis, eq_s)
    panel_box(fig, 2.09, yc, 1.86, hc); panel_g(fig, 2.09, yc, 1.86, hc, a, basis, rng, eq_s)
    panel_box(fig, 4.02, yc, 1.40, hc); panel_h(fig, 4.02, yc, 1.40, hc, a, CONTRAST[dataset])
    panel_box(fig, 5.49, yc, 1.61, hc); panel_i(fig, 5.49, yc, 1.61, hc, cache, KO_SIM[dataset], KO_PROJ[dataset], eq_s)

    # ---- lane IV ----
    panel_box(fig, 0.50, 8.30, 6.60, 1.22); panel_j(fig, 0.50, 8.30, 6.60, 1.22)

    tag = "" if dataset == "paul15_coarse" else f"-{dataset}"
    save(fig, f"{OUT}/framework-overview{tag}", formats=("pdf", "png"))
    plt.close(fig)
    print(f"wrote {OUT}/framework-overview{tag}.pdf (+ .png)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="paul15_coarse")
    build(p.parse_args().dataset)
