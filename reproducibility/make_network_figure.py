"""Composite figure: cell-type regulatory-network structure (repeatable for any dataset).

Reuses the analyzed report AnnData (reports/<dataset>/data/adata_analyzed.h5ad).
Panels (reading order a..f); per-cell-type panels use the requested cell types:
  a  cell-type GRNs (TikZ): top-25 genes by total interaction weight, top-30 edges,
     cluster (spring) layout; node size = total weight; red arrow = activation, blue
     bar-head = repression
  b  network-similarity heatmap: dendrogram, then cell-type names, then heatmap
  c  cell-type-specific hub genes (degree centrality above the cross-type mean)
  d  regulatory roles: out-strength (regulatory output) vs in-strength (regulation
     received), log-log; master regulators (high out/low in) separate from relays
  e  eigenanalysis: eigenvalue spectra + max-Re and min-Re eigenvector loading heatmaps
  f  small regulatory network (TikZ): a gene set + their top regulators, shell layout,
     uniform node size

Focal genes are a REQUIRED INPUT, not a decoration. Panel f draws a curated set of master
regulators together with their top regulators, so the set must belong to the system being
plotted. They come from the FOCAL_GENES table below, keyed by dataset, or from --genes,
which overrides it. There is no gene default: a dataset absent from the table stops the
run rather than borrow another system's regulators. Borrowing is what once drew pancreatic
transcription factors on hematopoiesis and myogenesis and left panel f empty.

Run (defaults = pancreas / Ductal, Pre-endocrine, Beta / the curated pancreatic cascade):
  python reproducibility/make_network_figure.py
  python reproducibility/make_network_figure.py --dataset murine_nc \
      --celltypes "A,B,C" --genes "G1,G2,G3"

--submission renders the same six panels on one journal page
(180 x 240 mm, no type below 5 pt) and writes Extended Data Fig. 4:
  python reproducibility/make_network_figure.py --submission
Without the flag nothing changes: the poster-size figure the per-dataset reports embed is
still written to reproducibility/figures/, at the same sizes it always was.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")
# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
from paper_plot_style import use_style, save, PALETTE   # noqa: E402
from submission_style import (figure_for as sub_figure_for, panel_letter,   # noqa: E402
                              save as sub_save, use_submission_style)
import anndata as ad                                    # noqa: E402
import networkx as nx                                   # noqa: E402
try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

OUT = paths.FIGURES
SUB_OUT = os.path.join(paths.FIGURES_SPEC, "ExtendedDataFig4.pdf")

# Panel f draws a curated regulatory subnetwork, so the focal genes must be the master regulators
# OF THE SYSTEM BEING PLOTTED. Until this dict existed, --genes defaulted to the pancreatic cascade
# for every dataset: the four hematopoietic and myogenic figures kept none of those genes and panel f
# rendered completely empty (only its titles and legend), and murine_nc / schwann kept exactly one
# accidental gene each (Neurod1, Sox9) and drew a subnetwork around it.
# Each list is priority ordered and the first FOCAL_N present in the fit are used, because the 2000-gene
# selection drops some canonical regulators (Tal1 from paul15, SOX10 from schwann, SIX1 from human_limb).
# Ranking the curated regulators by out-strength instead was tried and rejected: it does not reproduce
# the pancreatic cascade the manuscript names, and it drops Gata1 and Spi1 from hematopoiesis.
# Kept identical to the copy the analysis tree runs, so the same dataset renders the same panel f
# from either tree; change both together.
FOCAL_N = 7
FOCAL_GENES = {
    "pancreas": ["Pdx1", "Nkx6-1", "Sox9", "Neurog3", "Neurod1", "Arx", "Pax4"],
    "paul15": ["Gata1", "Gata2", "Klf1", "Spi1", "Cebpa", "Tal1", "Runx1", "Myb", "Zfpm1",
               "Nfe2", "Irf8", "Gfi1b", "Cebpe", "Fli1"],
    "dynamo_hematopoiesis": ["GATA1", "GATA2", "KLF1", "SPI1", "CEBPA", "TAL1", "RUNX1", "MYB",
                             "ZFPM1", "NFE2", "IRF8", "GFI1B", "CEBPE", "FLI1"],
    "murine_nc": ["Sox10", "Foxd3", "Tfap2a", "Pax3", "Mitf", "Phox2b", "Ascl1", "Sox9", "Ets1",
                  "Tfap2b", "Gata3", "Hand2", "Neurog2", "Isl1"],
    "schwann": ["Sox10", "Sox2", "Egr2", "Pou3f1", "Zeb2", "Tfap2a", "Id2", "Foxd3", "Pax3",
                "Sox9", "Gata3", "Phox2b", "Ets1", "Tfap2b"],
    "human_limb": ["PAX3", "PAX7", "MYF5", "MYOD1", "MYOG", "MEF2C", "SIX1", "MYF6", "LBX1",
                   "PITX2", "MSC", "TCF21", "EYA1", "SIX4"],
}
FOCAL_GENES["paul15_coarse"] = FOCAL_GENES["paul15"]              # same hematopoietic regulators
ACT_HEX, REP_HEX = "2A9D8F", "C44536"                   # activation teal / repression rust
ACT, REP = f"#{ACT_HEX}", f"#{REP_HEX}"
NODE_CMAP = LinearSegmentedColormap.from_list("deg", ["#5E3C99", "#F2F2F2", "#C79A00"])   # signed degree: - purple / + gold
TRIAD = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
# extended colorblind-safe categorical set for datasets with many cell types (first 3 = TRIAD,
# so the default 3-cell-type figure is unchanged); paired with a distinct marker per cell type.
CATCOLORS = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["vermillion"],
             PALETTE["purple"], PALETTE["sky"], "#000000", "#777777"]   # +black, +gray (yellow is illegible on white)
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# --------------------------------------------------------------------------- #
# Two sizings of the same six panels.
#   legacy      the poster-size figure the per-dataset reports embed (unchanged)
#   submission  one journal page, 180 x 240 mm, nothing below the 5 pt floor
# SUB is set once by --submission; every size below is read through sz()/mk(), so the
# default path renders exactly what it rendered before this flag existed.
SUB = False

_SIZES = {                          # point sizes                legacy  submission
    "grn_title":        (10.0, 8.0),        # cell-type title over a network panel
    "grn_legend":       (6.5, 5.2),         # activation / repression key
    "grn_cbar_tick":    (5.5, 5.0),         # in / out degree colorbar ticks
    "grn_cbar_title":   (5.2, 5.0),
    "panel_title":      (9.0, 7.0),         # plot titles inside panels b to e
    "sim_title":        (8.5, 7.0),
    "tick":             (6.0, 5.5),
    "cbar_tick":        (6.0, 5.0),         # similarity-heatmap colorbar
    "load_cbar_tick":   (5.5, 5.0),         # loadings-heatmap colorbar, always a step smaller in
                                            # legacy; folding it into "cbar_tick" raised it to 6.0
    "sim_tick":         (6.5, 5.5),         # cell-type names on the heatmap y axis
    # The x axis carries the same names horizontally in legacy mode, so it has always been one
    # step smaller to keep neighboring labels apart. Folding it into "sim_tick" when this fork
    # was created silently set it to 6.5 and made the labels touch; submission mode rotates them
    # 90 degrees and is unaffected, which is why it went unnoticed.
    "sim_tick_x":       (6.0, 5.5),
    "gene_label":       (6.0, 5.0),         # gene names on a heatmap axis
    "hub_gene":         (6.5, 5.2),
    "axis_label":       (8.0, 6.5),
    "axis_label_small": (6.5, 5.5),
    "legend":           (6.5, 5.2),
    "spec_legend":      (6.0, 5.0),
    "annot":            (6.3, 5.2),         # region names inside panel d
    "annot_small":      (5.6, 5.0),         # eigenvalue over a loadings column
    "roles_label":      (6.2, 5.0),         # marked gene names in panel d
    "letter":           (12.5, 8.0),        # panel letter
}
_MARKS = {                          # marker areas               legacy  submission
    "roles_point":      (16.0, 5.0),
    "roles_marked":     (26.0, 11.0),
    "spec_point":       (13.0, 4.0),
    "spec_star":        (95.0, 26.0),
}


def sz(key):
    return _SIZES[key][1 if SUB else 0]


def msz(key):
    return _MARKS[key][1 if SUB else 0]


def _gene(t):
    """Gene symbols are italic in the submission figure, upright in the poster version."""
    return {"style": "italic"} if SUB else {}


TIKZ_PRE = (
    r"\documentclass[border=3pt]{standalone}"
    r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
    r"\usepackage{tikz}\usetikzlibrary{arrows.meta}"
    r"\definecolor{actcol}{HTML}{" + ACT_HEX + r"}\definecolor{repcol}{HTML}{" + REP_HEX + r"}"
    r"\definecolor{nodefill}{HTML}{ADADAD}"
    r"\begin{document}\begin{tikzpicture}["
    r"gene/.style={circle,draw=black!50,fill=nodefill,line width=0.5pt,inner sep=1.0pt},"
    r"activate/.style={-{Latex[length=3.5pt,width=3pt]},actcol},"
    r"repress/.style={-{Bar[width=4.5pt]},repcol}]" + "\n"
)


def _tex(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _tikz_failed(ax, c):
    """Make a failed network render impossible to miss.

    render_tikz swallows every error and returns None, so a panel whose TikZ did not compile
    used to look like ordinary white space. Four shipped supplementary figures carried an empty
    panel f that way. Shout on stderr and put a visible marker in the panel.
    """
    print(f"WARNING: TikZ render failed for '{c}'; panel left empty", file=sys.stderr)
    ax.text(0.5, 0.5, "network render failed", transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color=REP, style="italic")


def render_tikz(body, dpi=None):
    dpi = dpi if dpi is not None else (600 if SUB else 460)   # 600 dpi at page size
    try:
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "c.tex"), "w").write(TIKZ_PRE + body + "\\end{tikzpicture}\n\\end{document}\n")
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "c.tex"],
                               cwd=td, capture_output=True)
            if r.returncode != 0 or not os.path.exists(os.path.join(td, "c.pdf")):
                return None
            subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                            os.path.join(td, "c.pdf"), os.path.join(td, "c")], capture_output=True)
            p = os.path.join(td, "c.png")
            return plt.imread(p) if os.path.exists(p) else None
    except Exception:
        return None


def _hex(rgb):
    return f"{int(rgb[0] * 255):02X}{int(rgb[1] * 255):02X}{int(rgb[2] * 255):02X}"


_COMPASS = [(0, "east"), (45, "north east"), (90, "north"), (135, "north west"),
            (180, "west"), (225, "south west"), (270, "south"), (315, "south east")]


def _anchor(ang):                                        # anchor so the label sits away from the centroid
    a = (ang + 180) % 360
    return min(_COMPASS, key=lambda c: min(abs(a - c[0]), 360 - abs(a - c[0])))[1]


def community_layout(G, nodes, node2c, ncomm):
    """Group nodes spatially by community: community centers on a ring, spring within each."""
    pos = {}
    for i in range(ncomm):
        members = [n for n in nodes if node2c[n] == i]
        if not members:
            continue
        sub = G.subgraph(members)
        if len(members) > 2 and sub.number_of_edges() > 0:
            sp = nx.spring_layout(sub, seed=0, k=0.9)
        elif len(members) > 1:
            sp = nx.circular_layout(nx.path_graph(members))
        else:
            sp = {members[0]: (0.0, 0.0)}
        cx, cy = np.cos(2 * np.pi * i / max(ncomm, 1)), np.sin(2 * np.pi * i / max(ncomm, 1))
        for nd, (x, y) in sp.items():
            pos[nd] = (1.7 * cx + 0.55 * x, 1.7 * cy + 0.55 * y)
    return pos


def tikz_grn_body(nodes, pos, edges, scale, size=4.5, fills=None, borders=None, bold=None, lblfont=r"\tiny",
                  node_lw=1.0, off_base=0.16, edge_lw=(0.25, 1.0), shorten=2.0, label_sep=0.6,
                  italic=False, labels=True, wmax=None, head_scale=1.0):
    """labels=False draws bare nodes, for thumbnails whose gene identities are printed
    once in a separate key. wmax fixes the edge-width normalizer, so a set of networks
    drawn separately stays comparable instead of each scaling to its own maximum."""
    bold = bold or set()
    idm = {n: f"n{i}" for i, n in enumerate(nodes)}
    cx = np.mean([pos[n][0] for n in nodes]); cy = np.mean([pos[n][1] for n in nodes])
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
        dx, dy = x - cx, y - cy; r = float(np.hypot(dx, dy)) or 1.0
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


def get_ck(dataset):
    try:
        from config import DATASETS
        return DATASETS.get(dataset, {}).get("cluster_key", "clusters")
    except Exception:
        return "clusters"


def get_order(dataset):
    try:
        from config import DATASETS
        return DATASETS.get(dataset, {}).get("order")
    except Exception:
        return None


def W_of(adata, c):
    W = adata.varp[f"W_{c}"]
    return np.asarray(W.todense()) if sp.issparse(W) else np.asarray(W)


def all_spines(ax, lw=0.6):
    for sp_ in ax.spines.values():
        sp_.set_visible(True); sp_.set_linewidth(lw)


def grn_legend(ax):
    lw, ms = (1.4, 4) if SUB else (1.8, 5)
    h = [Line2D([0], [0], color=ACT, lw=lw, marker=">", markersize=ms, label="activation"),
         Line2D([0], [0], color=REP, lw=lw, marker="|", markersize=ms * 1.8, label="repression")]
    ax.legend(handles=h, fontsize=sz("grn_legend"), loc="lower left", framealpha=0.85, borderpad=0.3,
              bbox_to_anchor=(-0.02, -0.02), handlelength=1.3 if SUB else None,
              labelspacing=0.25 if SUB else None, handletextpad=0.4 if SUB else None)


# --------------------------------------------------------------------------- #
def _spread_layout(G, k=3.5, iters=300, seeds=range(1, 12)):
    """Unweighted spring layout across several seeds; keep the one whose closest node
    pair is farthest apart, so dense hub cores get spread out rather than piling up."""
    best, best_score = None, -np.inf
    for s in seeds:
        p = nx.spring_layout(G, seed=s, weight=None, k=k, iterations=iters)
        if len(p) < 2:
            return p
        P = np.array([p[n] for n in G.nodes()])
        d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        score = float(d.min())                           # worst overlap; larger = better
        if score > best_score:
            best, best_score = p, score
    return best


def _declump(pos, min_dist=0.34, iters=120, step=0.55):
    """Push any node pair closer than min_dist apart (iterative repulsion), then rescale
    back into the frame. Breaks up dense hub cores that force-directed layouts leave piled."""
    nodes = list(pos)
    if len(nodes) < 2:
        return pos
    P = np.array([pos[n] for n in nodes], float)
    for _ in range(iters):
        D = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((D ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        close = dist < min_dist
        if not close.any():
            break
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = D / dist[..., None]
        unit[~np.isfinite(unit)] = 0.0
        push = ((min_dist - dist).clip(min=0)[..., None] * unit * close[..., None]).sum(1)
        P = P + step * push
    P -= P.mean(0)
    m = np.abs(P).max() or 1.0
    P /= m                                                # renormalize to ~[-1, 1]
    return {n: P[i] for i, n in enumerate(nodes)}


def draw_grn_a(ax, adata, c, legend=False, min_comp=5):
    # At page size the label ring, not the node, sets how many genes stay readable, so the
    # submission figure draws the same construction over a smaller top-degree set.
    top_k, n_edges = (8, 26) if SUB else (15, 50)
    min_comp = 4 if SUB else min_comp
    names = list(adata.var_names)
    W = W_of(adata, c)
    thr = np.quantile(np.abs(W), 0.98)
    Wt = W.copy(); Wt[np.abs(Wt) < thr] = 0.0
    indeg = np.abs(Wt).sum(1)                             # incoming magnitude (row = target)
    outdeg = np.abs(Wt).sum(0)                            # outgoing magnitude (col = regulator)
    keep = sorted(set(np.argsort(indeg)[::-1][:top_k]) | set(np.argsort(outdeg)[::-1][:top_k]))
    kn0 = [names[i] for i in keep]
    sub = Wt[np.ix_(keep, keep)]
    tri = [(kn0[r], kn0[t], sub[t, r]) for t in range(len(kn0)) for r in range(len(kn0))
           if t != r and sub[t, r] != 0]                 # regulator r -> target t
    edges = sorted(tri, key=lambda e: -abs(e[2]))[:n_edges]
    G = nx.Graph()
    G.add_weighted_edges_from([(u, v, abs(w)) for u, v, w in edges])
    comps = [cc for cc in nx.connected_components(G) if len(cc) >= min_comp]  # drop small fragments
    if not comps and G.number_of_nodes():                # fallback: keep the largest component
        comps = [max(nx.connected_components(G), key=len)]
    keep_nodes = set().union(*comps) if comps else set()
    kn = [n for n in kn0 if n in keep_nodes]
    edges = [(u, v, w) for u, v, w in edges if u in keep_nodes and v in keep_nodes]
    G = G.subgraph(keep_nodes).copy()
    pos = _spread_layout(G, k=3.5, iters=300)            # multi-seed spring, de-overlapped
    pos = _declump(pos)                                  # then push apart any remaining pile-ups
    direction = outdeg - indeg                            # + = net outgoing (regulator), - = net incoming (target)
    dvals = np.array([direction[names.index(n)] for n in kn])
    vmax = np.percentile(np.abs(dvals), 80) or (np.abs(dvals).max() or 1.0)   # clip: keep mid-range nodes colored (per-network)
    fills = {n: _hex(NODE_CMAP(0.5 + 0.5 * np.clip(direction[names.index(n)] / vmax, -1, 1))) for n in kn}
    if SUB:      # drawn near its printed size, so the LaTeX label size is the printed size
        body = tikz_grn_body(kn, pos, edges, scale=2.30, size=2.4, fills=fills,
                             lblfont=r"\fontsize{5.6}{6.2}\selectfont", node_lw=0.5, off_base=0.085,
                             edge_lw=(0.14, 0.52), shorten=1.0, label_sep=0.4, italic=True)
    else:
        body = tikz_grn_body(kn, pos, edges, scale=4.6, size=4.3, fills=fills, lblfont=r"\tiny")
    img = render_tikz(body)
    if img is None:                                   # a failed TikZ compile used to leave a blank panel
        _tikz_failed(ax, c)
    if img is not None:
        ax.imshow(img)
        # Reserve bottom space ONLY where the legend and colorbar actually sit. Doing it on
        # every panel padded all three with blank space and left a void under the whole band.
        if legend or not SUB:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0 + (0.13 if SUB else 0.15) * img.shape[0], y1)
    ax.axis("off")
    # In submission mode the titles are drawn once, at a shared baseline, by the caller.
    # A per-axes set_title() lands at a different height in each panel because imshow forces
    # an equal aspect, so each axes box ends up a different height inside its gridspec cell.
    if not SUB:
        ax.set_title(c, fontsize=sz("grn_title"), fontweight="bold")
    if legend:
        lw, ms = (1.3, 4) if SUB else (1.6, 5)
        h = [Line2D([0], [0], color=ACT, lw=lw, marker=">", markersize=ms, label="activation"),
             Line2D([0], [0], color=REP, lw=lw, marker="|", markersize=ms * 1.8, label="repression")]
        ax.legend(handles=h, fontsize=sz("grn_legend"), loc="lower left", bbox_to_anchor=(-0.01, 0.0),
                  framealpha=0.85, borderpad=0.25 if SUB else 0.3, handlelength=1.2 if SUB else 1.4,
                  labelspacing=0.25 if SUB else None, handletextpad=0.4 if SUB else None)
        sm = matplotlib.cm.ScalarMappable(cmap=NODE_CMAP, norm=matplotlib.colors.Normalize(-1, 1))
        rect = [0.36, 0.055, 0.20, 0.020] if SUB else [0.24, 0.05, 0.14, 0.014]
        cax = ax.inset_axes(rect)                        # side by side, to the right of the legend
        cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([-1, 1]); cb.set_ticklabels(["in", "out"])
        cax.tick_params(labelsize=sz("grn_cbar_tick"), length=2, pad=1)
        cax.set_title("in/out degree", fontsize=sz("grn_cbar_title"), pad=1.5)


def draw_grn_f(ax, adata, c, genes, legend=False, n_total=15):
    names = list(adata.var_names)
    W = W_of(adata, c)                                    # W[target, regulator]
    foc = [g for g in genes if g in names]                # curated genes present in this network
    nodes = list(foc)
    if len(nodes) < n_total:                              # complete to n_total with the top neighbors of the focal set
        fi = [names.index(g) for g in foc]
        score = np.abs(W[fi, :]).sum(0) + np.abs(W[:, fi]).sum(1)   # connection to the focal set (regulators + targets)
        for i in fi:
            score[i] = 0.0
        extra = [names[k] for k in np.argsort(score)[::-1] if names[k] not in nodes and score[k] > 1e-9]
        nodes += extra[:n_total - len(nodes)]
    Gc = nx.Graph(); Gc.add_nodes_from(nodes)
    pos = nx.circular_layout(Gc)                          # circular layout
    idx = {n: names.index(n) for n in nodes}
    edges = []                                            # induced subgraph: every regulator -> target among the nodes
    for u in nodes:
        for v in nodes:
            if u != v and abs(W[idx[v], idx[u]]) > 1e-9:
                edges.append((u, v, W[idx[v], idx[u]]))   # u regulates v
    if SUB:      # same construction, drawn near its printed size (see draw_grn_a)
        body = tikz_grn_body(nodes, pos, edges, scale=1.85, size=3.6, bold=set(foc),
                             lblfont=r"\fontsize{6.0}{6.6}\selectfont", node_lw=0.6, off_base=0.10,
                             edge_lw=(0.16, 0.60), shorten=1.2, label_sep=0.4, italic=True)
    else:
        body = tikz_grn_body(nodes, pos, edges, scale=2.3, size=4.8, bold=set(foc), lblfont=r"\scriptsize")
    img = render_tikz(body)
    if img is None:                                   # a failed TikZ compile used to leave a blank panel
        _tikz_failed(ax, c)
    else:
        ax.imshow(img)
    ax.axis("off"); ax.set_title(c, fontsize=sz("grn_title"), fontweight="bold")
    if legend:
        grn_legend(ax)


# --------------------------------------------------------------------------- #
def draw_similarity(ax_dend, ax_heat, adata, triad):
    nc = adata.uns.get("scHopfield", {}).get("network_correlations", {})
    P = nc.get("pearson")
    if P is None:
        ax_heat.text(0.5, 0.5, "no network_correlations", ha="center"); ax_dend.set_axis_off(); return
    P = P.astype(float)
    d = 1 - P.values; np.fill_diagonal(d, 0.0); d = (d + d.T) / 2
    Z = linkage(squareform(d, checks=False), method="average")
    dn = dendrogram(Z, orientation="left", ax=ax_dend, no_labels=True, color_threshold=0,
                    link_color_func=lambda k: "0.45")
    ax_dend.set_axis_off()
    order = dn["leaves"]; labs = [P.index[i] for i in order]
    im = ax_heat.imshow(P.values[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1,
                        aspect="auto", origin="lower")
    ax_heat.set_xticks(range(len(order)))
    ax_heat.set_xticklabels(labs, rotation=90 if SUB else 0, fontsize=sz("sim_tick_x"))
    ax_heat.set_yticks(range(len(order))); ax_heat.set_yticklabels(labs, fontsize=sz("sim_tick"))  # left (default)
    if SUB:
        ax_heat.tick_params(length=1.5, pad=1.0)
    for group in (ax_heat.get_xticklabels(), ax_heat.get_yticklabels()):
        for t in group:
            g = t.get_text(); c = triad.get(g)
            t.set_bbox(dict(boxstyle=f"round,pad={0.10 if SUB else 0.15}", fc=c or "0.6", ec="none",
                            alpha=0.30 if c else 0.13))
            if c:
                t.set_fontweight("bold")
    all_spines(ax_heat, lw=0.5 if SUB else 0.6)
    if SUB:
        ax_heat.set_title("network similarity\n(Pearson of $W_c$)", fontsize=sz("sim_title"),
                          pad=3, linespacing=1.0)
    else:
        ax_heat.set_title("network similarity (Pearson of $W_c$)", fontsize=sz("sim_title"))
    cb = ax_heat.figure.colorbar(im, ax=ax_heat, fraction=0.055 if SUB else 0.045,
                                 pad=0.04 if SUB else 0.03)
    cb.ax.tick_params(labelsize=sz("cbar_tick"))
    if SUB:
        cb.set_ticks([-1, 0, 1]); cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(length=1.5, pad=1.0)


def draw_hubs(ax, adata, c, all_cts, color, n=10):
    M = pd.DataFrame({cc: adata.var.get(f"degree_centrality_all_{cc}",
                                        pd.Series(0.0, index=adata.var_names)).fillna(0)
                      for cc in all_cts})
    diff = (M[c] - M.mean(axis=1)).sort_values(ascending=False).head(n).iloc[::-1]
    ax.barh(range(len(diff)), diff.values, color=color, height=0.72 if SUB else 0.8)
    ax.set_yticks(range(len(diff)))
    ax.set_yticklabels(diff.index, fontsize=sz("hub_gene"), **_gene(None))
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(c, fontsize=sz("panel_title"), color=color, pad=2 if SUB else None)
    ax.tick_params(labelsize=sz("tick"))
    ax.set_xlabel("degree centrality\nabove cross-type mean", fontsize=sz("axis_label_small"),
                  labelpad=1 if SUB else None, linespacing=1.0 if SUB else 1.2)
    if SUB:
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(length=1.5, pad=1.0)
        ax.margins(y=0.02)


def draw_roles(ax, adata, cts, triad, markers):
    """Regulatory-role map: out-strength (regulatory output, sum|W| over a gene's targets)
    vs in-strength (how much the gene is itself regulated). Master regulators sit at the
    lower right (high out, low in); relays at the upper left (high in, low out); pure targets
    (out-strength 0) are omitted. This replaces degree-vs-betweenness, which is degenerate
    under the directed only_TFs topology: the high-degree hubs are sources with ~0
    betweenness, so degree and betweenness are structurally decoupled."""
    names = list(adata.var_names)
    FLOOR = 0.05                                          # so pure regulators (in=0) render on log-y
    allpts = []                                           # (gene, out, in, ci) for every plotted regulator point
    pool_out, pool_in = [], []
    for ci, (c, mk) in enumerate(zip(cts, markers)):
        W = W_of(adata, c)
        outs = np.abs(W).sum(0); ins = np.abs(W).sum(1)
        reg = outs > 1e-9                                 # regulators only (nonzero TF columns)
        yin = np.clip(ins[reg], FLOOR, None)
        ax.scatter(outs[reg], yin, s=msz("roles_point"), color=triad[c], marker=mk, alpha=0.55,
                   linewidths=0, label=c)
        pool_out.extend(outs[reg]); pool_in.extend(yin)
        for i in np.where(reg)[0]:
            allpts.append((names[i], float(outs[i]), float(max(ins[i], FLOOR)), ci))
    ax.set_xscale("log"); ax.set_yscale("log")           # set log BEFORE labels so they land on markers
    mo, mi = float(np.percentile(pool_out, 98)), float(np.percentile(pool_in, 98))  # 98th-pct split -> role regions
    ax.axvline(mo, color="0.45", ls="--", lw=0.7, zorder=0)
    ax.axhline(mi, color="0.45", ls="--", lw=0.7, zorder=0)
    for rx, ry, rha, rva, rs in [(0.985, 0.035, "right", "bottom", "master\nregulators"),
                                 (0.015, 0.965, "left", "top", "relays")]:  # central-hubs region is empty
        ax.text(rx, ry, rs, transform=ax.transAxes, ha=rha, va=rva, fontsize=sz("annot"),
                style="italic", color="0.4", linespacing=0.9)
    # mark EVERY point past either line (a relay/regulator recurs across cell types); label
    # each gene once, at its most extreme point, so recurring genes are not printed 3 times.
    texts = []; labeled = set()
    for g, xo, yi, ci in sorted(allpts, key=lambda p: -max(p[1] / mo, p[2] / mi)):
        if xo >= mo or yi >= mi:
            ax.scatter([xo], [yi], s=msz("roles_marked"), marker=markers[ci], facecolor=triad[cts[ci]],
                       edgecolors="k", linewidths=0.5 if SUB else 1.0, zorder=6)
            if g not in labeled:
                texts.append(ax.text(xo, yi, g, fontsize=sz("roles_label"), fontweight="bold",
                                     **_gene(None))); labeled.add(g)
    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4),
                    expand=(1.15, 1.35) if SUB else (1.5, 1.9))
    ax.set_xlabel(r"out-strength $\sum|W_{ij}|$ (regulatory output)", fontsize=sz("axis_label"),
                  labelpad=1 if SUB else None)
    ax.set_ylabel("in-strength (regulation received)", fontsize=sz("axis_label"),
                  labelpad=1 if SUB else None)
    handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=triad[c],
                      markersize=4 if SUB else 6, label=c) for c, mk in zip(cts, markers)]
    ax.legend(handles=handles, fontsize=sz("legend"), loc="lower left", framealpha=0.9,
              handlelength=1.0 if SUB else None, handletextpad=0.3 if SUB else None,
              labelspacing=0.2 if SUB else None, borderpad=0.25 if SUB else None,
              borderaxespad=0.2 if SUB else None)
    ax.set_title("Regulatory roles: output vs input", fontsize=sz("panel_title"), pad=2 if SUB else None)
    if SUB:
        ax.tick_params(labelsize=sz("tick"), length=1.5, pad=1.0)
        ax.tick_params(which="minor", length=0.8)


def draw_spectrum(ax, adata, cts, triad, markers):
    for c, mk in zip(cts, markers):
        vals = np.linalg.eigvals(W_of(adata, c))
        ax.scatter(vals.real, vals.imag, s=msz("spec_point"), color=triad[c], marker=mk, alpha=0.7,
                   linewidths=0, label=c)
        re = vals.real
        for k in (int(np.argmax(re)), int(np.argmin(re))):
            ax.scatter(vals.real[k], vals.imag[k], s=msz("spec_star"), marker="*", color=triad[c],
                       edgecolors="k", linewidths=0.4 if SUB else 0.5, zorder=6)
    ax.axhline(0, ls="--", c="k", lw=0.6, alpha=0.5); ax.axvline(0, ls="--", c="k", lw=0.6, alpha=0.5)
    ax.set_xlabel(r"Re($\lambda$)", fontsize=sz("axis_label"), labelpad=1 if SUB else None)
    ax.set_ylabel(r"Im($\lambda$)", fontsize=sz("axis_label"), labelpad=1 if SUB else None)
    handles = [] if SUB else [                            # cell-type key is shared with panel d, alongside
        Line2D([0], [0], marker=mk, color="w", markerfacecolor=triad[c], markersize=6, label=c)
        for c, mk in zip(cts, markers)]
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="0.6", markeredgecolor="k",
                          markersize=6 if SUB else 9, label=r"max/min Re($\lambda$)"))
    ax.legend(handles=handles, fontsize=sz("spec_legend"), loc="upper left", framealpha=0.9,
              handlelength=1.0 if SUB else None, handletextpad=0.2 if SUB else None,
              borderpad=0.25 if SUB else None, borderaxespad=0.15 if SUB else None)
    ax.set_title("Eigenvalue spectra", fontsize=sz("panel_title"), pad=2 if SUB else None)
    if SUB:
        ax.tick_params(labelsize=sz("tick"), length=1.5, pad=1.0)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))


def _eig(W, which):
    vals, vecs = np.linalg.eig(W)
    k = int(np.argmax(vals.real)) if which == "max" else int(np.argmin(vals.real))
    v = np.real(vecs[:, k])
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    return v, vals[k]


def draw_loadings_heatmap(ax, adata, cts, triad, which, title, n=20):
    L, lam = {}, {}
    for c in cts:
        v, ev = _eig(W_of(adata, c), which); L[c] = v; lam[c] = ev
    score = np.zeros(adata.n_vars)
    for c in cts:
        score = np.maximum(score, np.abs(L[c]))
    idx = np.argsort(score)[::-1][:n]
    M = np.array([[L[c][i] for c in cts] for i in idx])
    vmax = np.abs(M).max() or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cts)))
    ax.set_xticklabels(cts, rotation=90 if (SUB or len(cts) > 4) else 0, fontsize=sz("sim_tick"))
    for t in ax.get_xticklabels():
        g = t.get_text(); t.set_bbox(dict(boxstyle=f"round,pad={0.08 if SUB else 0.12}",
                                          fc=triad.get(g, "0.6"), ec="none", alpha=0.30))
        t.set_fontweight("bold")
    for i, c in enumerate(cts):                          # eigenvalue above each column
        ax.text(i, -0.75, f"$\\lambda$={lam[c].real:.2f}", ha="center", va="bottom",
                fontsize=sz("annot_small"), rotation=90 if SUB else 0)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([adata.var_names[i] for i in idx], fontsize=sz("gene_label"), **_gene(None))
    all_spines(ax, lw=0.5 if SUB else 0.6)
    ax.set_title(title, fontsize=sz("sim_title"), pad=17 if SUB else 14, linespacing=1.0 if SUB else 1.2)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.10 if SUB else 0.05, pad=0.06 if SUB else 0.04)
    cb.ax.tick_params(labelsize=sz("load_cbar_tick"))
    if SUB:
        ax.tick_params(length=1.5, pad=1.0)
        cb.ax.tick_params(length=1.5, pad=1.0); cb.outline.set_linewidth(0.4)
        cb.locator = MaxNLocator(3); cb.update_ticks()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pancreas")
    ap.add_argument("--celltypes", default="Ductal,Pre-endocrine,Beta")
    ap.add_argument("--genes", default=None,
                    help="focal genes for panel f; default = this dataset's curated regulators")
    ap.add_argument("--cluster-key", default=None)
    ap.add_argument("--submission", action="store_true",
                    help="render the same six panels on one journal page (180 x 240 mm, 5 pt floor)")
    ap.add_argument("--out", default=SUB_OUT, help="output PDF for --submission")
    args = ap.parse_args()

    global SUB
    SUB = bool(args.submission)

    ds = args.dataset
    ck = args.cluster_key or get_ck(ds)
    if args.genes:
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    elif ds in FOCAL_GENES:
        genes = list(FOCAL_GENES[ds])
    else:
        raise SystemExit(f"no curated focal genes for dataset '{ds}'; pass --genes explicitly "
                         f"(known: {', '.join(sorted(FOCAL_GENES))})")

    adata = ad.read_h5ad(f"{paths.REPORTS}/{ds}/data/adata_analyzed.h5ad")
    all_cts = [k[2:] for k in adata.varp if k.startswith("W_") and k[2:] != "all"]
    if args.celltypes.strip().lower() == "all":                    # every cell type, in config order
        order = get_order(ds) or sorted(all_cts)
        cts = [c for c in order if c in all_cts] or list(all_cts)
        NAME = f"network-structure-{ds}-all"
    else:
        cts = [c.strip() for c in args.celltypes.split(",") if c.strip()]
        cts = [c for c in cts if f"W_{c}" in adata.varp]
        NAME = f"network-structure-{ds}"
    genes = [g for g in genes if g in adata.var_names][:FOCAL_N]   # priority order, first FOCAL_N that survived
    if not genes:                                                  # never ship a silently empty panel f
        raise SystemExit(f"none of the focal genes for '{ds}' are in the fit; panel f would be empty")
    n = len(cts)
    # Cell-type color = the report's canonical per-dataset palette (adata.uns[f"{ck}_colors"], the same
    # dict the energy-landscape figure uses), so one cell type keeps ONE color across every figure and
    # datasets with more than 8 cell types get distinct colors (scanpy default_20/default_102) instead
    # of wrapping an 8-color cycle. CATCOLORS stays only as a fallback for a type absent from the palette.
    from sections import get_colors                        # noqa: E402
    ct_palette = get_colors(adata, ck)
    triad = {c: ct_palette.get(c, CATCOLORS[i % len(CATCOLORS)]) for i, c in enumerate(cts)}
    # marker still advances so (color, marker) stays distinguishable in grayscale.
    markers = [MARKERS[(i + i // len(MARKERS)) % len(MARKERS)] for i in range(n)]
    gset = ", ".join(genes)

    use_style(7 if SUB else 9)
    os.makedirs(OUT, exist_ok=True)

    if SUB:
        # ---------------- one journal page: 180 mm wide, 240 mm tall ----------------
        # Boxes are placed in millimeters from the top left, because every constraint here
        # (a 13 mm cell-type name, a 20-row gene axis, a 5 pt floor) is a millimeter
        # constraint. Rows, top to bottom: a | b c | d e | f.
        PW, PH = 180.0, 227.0
        fig = sub_figure_for("double", height_mm=PH)
        use_submission_style()

        def X(mm):
            return mm / PW

        def Y(mm):
            return 1.0 - mm / PH

        def label(ax, text, dx=0.028, dy=0.010):        # signature kept; letter only
            panel_letter(ax, text.split()[0])

        # a: three cell-type GRNs across the full width
        gs_a = fig.add_gridspec(1, n, left=X(5), right=X(178), top=Y(12.0), bottom=Y(53.5),
                                wspace=0.05)
        ax_a = [fig.add_subplot(gs_a[0, i]) for i in range(n)]
        for i, c in enumerate(cts):
            draw_grn_a(ax_a[i], adata, c, legend=(i == 0))
        # One shared baseline for the three cell-type titles, in figure coordinates, so they
        # align across panels whose axes boxes differ in height (see draw_grn_a).
        fig.canvas.draw()
        for ax, c in zip(ax_a, cts):
            bb = ax.get_position()
            fig.text((bb.x0 + bb.x1) / 2, Y(11.1), c, ha="center", va="bottom",
                     fontsize=sz("grn_title"), fontweight="bold")
        # b: dendrogram + similarity heatmap (the gap between them holds the row names)
        gs_b = fig.add_gridspec(1, 2, left=X(6), right=X(66), top=Y(66), bottom=Y(93),
                                width_ratios=[0.22, 1.0], wspace=0.62)
        ax_dend = fig.add_subplot(gs_b[0, 0]); ax_heat = fig.add_subplot(gs_b[0, 1])
        draw_similarity(ax_dend, ax_heat, adata, triad)
        # c: one hub-gene bar chart per cell type
        gs_c = fig.add_gridspec(1, n, left=X(83), right=X(178), top=Y(66), bottom=Y(95),
                                wspace=0.63)
        ax_c = [fig.add_subplot(gs_c[0, i]) for i in range(n)]
        for ax, c in zip(ax_c, cts):
            draw_hubs(ax, adata, c, all_cts, triad[c])
        # d: regulatory-role map
        gs_d = fig.add_gridspec(1, 1, left=X(16), right=X(62), top=Y(121), bottom=Y(163))
        ax_d = fig.add_subplot(gs_d[0, 0]); draw_roles(ax_d, adata, cts, triad, markers)
        # e: eigenvalue spectra + the two eigenvector-loading heatmaps
        gs_s = fig.add_gridspec(1, 1, left=X(76), right=X(104), top=Y(121), bottom=Y(163))
        ax_spec = fig.add_subplot(gs_s[0, 0]); draw_spectrum(ax_spec, adata, cts, triad, markers)
        gs_l1 = fig.add_gridspec(1, 1, left=X(119), right=X(135), top=Y(121), bottom=Y(163))
        ax_lmax = fig.add_subplot(gs_l1[0, 0])
        draw_loadings_heatmap(ax_lmax, adata, cts, triad, "max",
                              "eigenvector of the\nmaximum real-part $\\lambda$")
        gs_l2 = fig.add_gridspec(1, 1, left=X(155), right=X(171), top=Y(121), bottom=Y(163))
        ax_lmin = fig.add_subplot(gs_l2[0, 0])
        draw_loadings_heatmap(ax_lmin, adata, cts, triad, "min",
                              "eigenvector of the\nminimum real-part $\\lambda$")
        # f: curated developmental-TF subnetwork per cell type
        gs_f = fig.add_gridspec(1, n, left=X(5), right=X(178), top=Y(184), bottom=Y(225),
                                wspace=0.05)
        ax_f = [fig.add_subplot(gs_f[0, i]) for i in range(n)]
        for i, c in enumerate(cts):
            draw_grn_f(ax_f[i], adata, c, genes, legend=(i == 0))
    elif n <= 4:                                                   # single-row layout (approved default)
        fig = plt.figure(figsize=(6.4 * n, 21.0))

        def label(ax, text, dx=0.028, dy=0.010):        # render ONLY the panel letter (description -> caption)
            bb = ax.get_position()
            fig.text(bb.x0 - dx, bb.y1 + dy, text.split()[0], fontweight="bold", fontsize=12.5,
                     va="bottom", ha="left")

        fig.text(0.045, 0.992, f"Regulatory network structure  --  {ds}", ha="left", va="top",
                 fontsize=15, fontweight="bold")
        gs_a = fig.add_gridspec(1, n, top=0.955, bottom=0.745, left=0.05, right=0.97, wspace=0.06)
        ax_a = [fig.add_subplot(gs_a[0, i]) for i in range(n)]
        for i, c in enumerate(cts):
            draw_grn_a(ax_a[i], adata, c, legend=(i == 0))
        gs_bc = fig.add_gridspec(1, 2, top=0.705, bottom=0.55, left=0.05, right=0.97,
                                 width_ratios=[0.95, 1.5], wspace=0.18)
        gsb = gs_bc[0].subgridspec(1, 2, width_ratios=[0.30, 1.0], wspace=0.34)   # small gap holds the row names
        ax_dend = fig.add_subplot(gsb[0, 0]); ax_heat = fig.add_subplot(gsb[0, 1])
        draw_similarity(ax_dend, ax_heat, adata, triad)
        gsc = gs_bc[1].subgridspec(1, n, wspace=0.55)
        ax_c = [fig.add_subplot(gsc[0, i]) for i in range(n)]
        for ax, c in zip(ax_c, cts):
            draw_hubs(ax, adata, c, all_cts, triad[c])
        gs_de = fig.add_gridspec(1, 2, top=0.505, bottom=0.30, left=0.05, right=0.97,
                                 width_ratios=[0.8, 2.0], wspace=0.20)
        ax_d = fig.add_subplot(gs_de[0]); draw_roles(ax_d, adata, cts, triad, markers)
        gse = gs_de[1].subgridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.6)
        ax_spec = fig.add_subplot(gse[0, 0]); draw_spectrum(ax_spec, adata, cts, triad, markers)
        ax_lmax = fig.add_subplot(gse[0, 1]); draw_loadings_heatmap(ax_lmax, adata, cts, triad, "max", "eigenvector of the maximum\nreal-part eigenvalue")
        ax_lmin = fig.add_subplot(gse[0, 2]); draw_loadings_heatmap(ax_lmin, adata, cts, triad, "min", "eigenvector of the minimum\nreal-part eigenvalue")
        gs_f = fig.add_gridspec(1, n, top=0.255, bottom=0.03, left=0.05, right=0.97, wspace=0.06)
        ax_f = [fig.add_subplot(gs_f[0, i]) for i in range(n)]
        for i, c in enumerate(cts):
            draw_grn_f(ax_f[i], adata, c, genes, legend=(i == 0))
    else:                                                         # grid layout for many cell types
        PC = 4
        rows = (n + PC - 1) // PC
        A_H, F_H = rows * 4.4 + (rows - 1) * 0.7, rows * 4.7 + (rows - 1) * 0.7
        BC_H, DE_H, TITLE_H, GAP, BOT = max(4.2, rows * 2.4), 4.3, 1.0, 0.95, 0.6
        H = TITLE_H + A_H + GAP + BC_H + GAP + DE_H + GAP + F_H + BOT
        fig = plt.figure(figsize=(6.2 * PC, H))
        ftop = lambda inch: 1.0 - inch / H                        # inches-from-top -> figure fraction

        def label(ax, text, dx=0.024, dy=0.006):        # render ONLY the panel letter (description -> caption)
            bb = ax.get_position()
            fig.text(bb.x0 - dx, bb.y1 + dy, text.split()[0], fontweight="bold", fontsize=13,
                     va="bottom", ha="left")

        fig.text(0.05, ftop(0.35), f"Regulatory network structure  --  {ds}  (all cell types)",
                 ha="left", va="top", fontsize=16, fontweight="bold")
        cur = TITLE_H
        aT, aB = ftop(cur), ftop(cur + A_H); cur += A_H + GAP
        bcT, bcB = ftop(cur), ftop(cur + BC_H); cur += BC_H + GAP
        deT, deB = ftop(cur), ftop(cur + DE_H); cur += DE_H + GAP
        fT, fB = ftop(cur), ftop(cur + F_H)
        L, R = 0.05, 0.975
        gs_a = fig.add_gridspec(rows, PC, top=aT, bottom=aB, left=L, right=R, wspace=0.06, hspace=0.30)
        ax_a = [fig.add_subplot(gs_a[i // PC, i % PC]) for i in range(n)]
        for i, c in enumerate(cts):
            draw_grn_a(ax_a[i], adata, c, legend=(i == 0))
        gs_bc = fig.add_gridspec(1, 2, top=bcT, bottom=bcB, left=L, right=R,
                                 width_ratios=[0.85, 2.3], wspace=0.13)
        gsb = gs_bc[0].subgridspec(1, 2, width_ratios=[0.30, 1.0], wspace=0.34)
        ax_dend = fig.add_subplot(gsb[0, 0]); ax_heat = fig.add_subplot(gsb[0, 1])
        draw_similarity(ax_dend, ax_heat, adata, triad)
        gsc = gs_bc[1].subgridspec(rows, PC, wspace=0.6, hspace=0.55)
        ax_c = [fig.add_subplot(gsc[i // PC, i % PC]) for i in range(n)]
        for ax, c in zip(ax_c, cts):
            draw_hubs(ax, adata, c, all_cts, triad[c])
        gs_de = fig.add_gridspec(1, 2, top=deT, bottom=deB, left=L, right=R,
                                 width_ratios=[0.8, 2.0], wspace=0.20)
        ax_d = fig.add_subplot(gs_de[0]); draw_roles(ax_d, adata, cts, triad, markers)
        gse = gs_de[1].subgridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.6)
        ax_spec = fig.add_subplot(gse[0, 0]); draw_spectrum(ax_spec, adata, cts, triad, markers)
        ax_lmax = fig.add_subplot(gse[0, 1]); draw_loadings_heatmap(ax_lmax, adata, cts, triad, "max", "eigenvector of the maximum\nreal-part eigenvalue")
        ax_lmin = fig.add_subplot(gse[0, 2]); draw_loadings_heatmap(ax_lmin, adata, cts, triad, "min", "eigenvector of the minimum\nreal-part eigenvalue")
        gs_f = fig.add_gridspec(rows, PC, top=fT, bottom=fB, left=L, right=R, wspace=0.06, hspace=0.30)
        ax_f = [fig.add_subplot(gs_f[i // PC, i % PC]) for i in range(n)]
        for i, c in enumerate(cts):
            draw_grn_f(ax_f[i], adata, c, genes, legend=(i == 0))

    label(ax_a[0], "a   Cell-type-specific GRNs")
    label(ax_dend, "b   Network similarity", dx=0.018)
    label(ax_c[0], "c   Cell-type-specific hub genes")
    label(ax_d, "d   Regulatory roles")
    label(ax_spec, "e   Network eigenanalysis")
    label(ax_f[0], f"f   Regulatory network of {gset}  (+ neighbors)")

    if SUB:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        sub_save(fig, args.out)                          # raises if any text is below 5 pt
        print(f"wrote {args.out}   (cell types: {cts}; genes: {genes})")
    else:
        outdir = f"{OUT}/extended" if NAME.endswith("-all") else OUT   # -all variants live in extended/
        os.makedirs(outdir, exist_ok=True)
        save(fig, f"{outdir}/{NAME}", formats=("pdf", "png"))
        print(f"wrote {outdir}/{NAME}.pdf + .png   (cell types: {cts}; genes: {genes})")
    plt.close(fig)


if __name__ == "__main__":
    main()
