"""Combinatorial (double-knockout) in-silico perturbation figure, restructured into two halves:
LEFT = curated (known-TF) based, RIGHT = discovery (data-driven) based. Reads the 4-block screen cache
from `_double_ko_compute.py --only screen` (reports/<ds>/data/double_ko_screen.pkl).

Per half (curated / discovery) and per lineage decision:
  a  anchor-partner SYNERGY bars: for the two anchors (one biasing each lineage), the non-additive
     synergy against the screened partners (green super-additive / purple cancellation); big ones marked.
  b  square all-pairs matrix over the 10 selected genes (5 biasing each lineage): LOWER triangle = the
     double-KO fate shift (RdBu_r), UPPER triangle = the synergy (PRGn), diagonal = single-KO shift.
  c  per-cell fate-shift map of the best (top |synergy|) pairs (RdBu_r).
  d  KO-induced flow change of the best pairs, alignment with development (BrBG; needs --only screenflow).
  e  Jacobian commitment push of the best pairs (PuOr, prediction vs the RdBu_r fate result).
  f  literature tier tables (one per decision) for the best pairs.

Colormaps are all distinct: fate = RdBu_r, synergy = PRGn, flow-vs-development = BrBG, push = PuOr.

Run:  python reproducibility/make_double_perturbation.py [--dataset pancreas]

``--submission`` renders the SAME six panels re-laid out for a journal
page (180 mm wide, type at or above the 5 pt floor) and writes them to
reproducibility/figures/submission/Figure6.pdf. Without the flag nothing about the poster-sized
report figure changes: the per-dataset reports still embed it.
"""
from __future__ import annotations
import argparse, os, sys, pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
from paper_plot_style import use_style, save, PALETTE            # noqa: E402
from make_perturbation_dynamics import draw_fate_map, _grid_arrows   # noqa: E402  (reuse conventions)
import anndata as ad                                             # noqa: E402

OUT = paths.FIGURES
PROCESS = {"pancreas": "endocrinogenesis", "paul15": "hematopoiesis", "paul15_coarse": "hematopoiesis",
           "dynamo_hematopoiesis": "hematopoiesis", "murine_nc": "neural crest", "human_limb": "myogenesis",
           "schwann": "neural crest"}
SYN_POS = PALETTE["green"]                                       # super-additive synergy
SYN_NEG = PALETTE["purple"]                                      # cancellation

# Panel f tier classification of the candidate pairs (per process). T1 known/validated, T2 evidenced but
# joint role not dissected, T3 novel for this process. Verified with lit-novelty (see the tiers notes).
# Tiers for the anchor-partner screen's top-synergy pairs (verified with lit-novelty / verify_papers.py;
# see double-perturbation-tiers.md). The fixed anchors are canonical (Arx/Pax4 antagonism, Collombat
# 2003; Ngn3, Gradwohl 2000), so their known-gene partners recover endocrine biology (T2), while the
# Jacobian-coupling screen also surfaces genuinely novel candidates (T3): Rorc, Klf13, Zbtb7c, Vdr,
# Mef2a, Atf6, and the glycolytic hub Eno1 (a high-out-strength non-TF, flagged for cautious reading).
TIERS = {
    # Rebuilt 2026-07-31 for the pairs the CANONICAL screen nominates. The previous table was keyed on
    # pairs from a 2026-07-26 cache that the recomputed screen no longer produces (Arx/Foxo1,
    # Pax4/Foxo1, Zbtb7c, Atf6 and the Tead2/Foxa3, Creb3l1/Mef2a anchors); those tiers were NOT
    # transferred, because a tier is a claim about a specific pair. Every reference below was checked
    # against Crossref as a real journal article with a matching title. Two errors were caught in that
    # check and are corrected here: the Klf10 paper is the SEI-1/p21Cip1 study, not a SUR1/SERCA2b one,
    # and the Prox1 evidence is Wang 2005, since the "Paul 2016" reference did not resolve.
    "pancreas": {
        # differentiated versus progenitor (endocrine specification)
        ("Klf10", "Rfx6"): ("T3", "Rfx6 KO ablates islets (Smith 2010); Klf10 adult islet only; joint untested"),
        ("Fev", "Rfx6"): ("T1", "RFX6 loss lowers FEV in iPSC islets (Aldous 2024); Rfx6 KO ablates islets (Smith 2010)"),
        ("Klf10", "Neurod1"): ("T3", "Neurod1 KO blocks endocrine specification (Naya 1997); Klf10 adult only; untested"),
        ("Klf10", "Tead2"): ("T3", "Klf10 adult islet mass (Wu 2015); Tead2 pancreas expression only (Escot 2018)"),
        # alpha versus beta (subtype allocation)
        ("Eno1", "Neurod1"): ("T3", "Neurod1 KO loses alpha and beta (Chao 2007); Eno1 glycolytic, no fate role"),
        ("Neurod1", "Prox1"): ("T2", "Neurod1 KO loses alpha and beta (Chao 2007) + Prox1 endocrine (Wang 2005); untested"),
        ("Eno1", "Vdr"): ("T3", "Eno1 beta insulin gene (Luo 2024); Vdr adult islet KO (Zeitz 2003); no fate role"),
        ("Vdr", "Xbp1"): ("T3", "Xbp1 KO drives beta-to-alpha shift (Lee 2022); Vdr no islet-development role"),
    },
}
TIER_COLOR = {"T1": PALETTE["green"], "T2": PALETTE["orange"], "T3": PALETTE["vermillion"], "?": "0.6"}


def tier_of(ds, pair):
    t = TIERS.get(ds, {})
    return t.get(tuple(pair)) or t.get(tuple(pair[::-1])) or ("?", "needs literature")


# One identity color per LINEAGE, distinct per lineage PAIR (decision), so panels a,b,c,d that point
# "toward A vs B" carry the same color for the same lineage. Decision 0 (e.g. differentiated/progenitor)
# = red/blue; decision 1 (e.g. alpha/beta) = gold/indigo. DEC_POLES[k] = (warm A, cool B); DEC_CMAPS[k] =
# the matching diverging fate map (cool B -> white -> warm A). Synergy (super-add/cancel) is NOT a lineage
# direction and stays PRGn; the Jacobian push (e) stays its own distinct warm/cool (orange/teal).
DEC_POLES = [("#B4423A", "#2B5D8A"), ("#C0871A", "#4B3D9E")]    # dec0 red/blue, dec1 gold/indigo
DEC_CMAPS = [LinearSegmentedColormap.from_list(f"fate{i}", [b, "#f5f5f5", a])
             for i, (a, b) in enumerate(DEC_POLES)]
LINE_A, LINE_B = DEC_POLES[0]                                   # generic fallback poles
CMAP_PUSH = LinearSegmentedColormap.from_list("pushAB", ["#1E7B6F", "#f4f4f4", "#C25A1C"])  # teal B -> orange A


A_FS = dict(label=4.3, axis=6.0, title=7.0, tick=5.0, legend=5.0, marker=42.0, star=60.0)


def draw_selection_scatter(ax, block, poles, fs=None, title=True):
    """Panel a: how the ~20 partners are chosen. Each candidate sits in the driver-score plane (toward A
    vs toward B); its MARKER SIZE is the Jacobian coupling to the anchor of the lineage it leans to (how
    similarly it perturbs the fitted system). The partner score is driver bias x this coupling, so the
    selected partners are those far off the diagonal AND large: the 10+10 kept are filled warm (toward A)
    / cool (toward B) and labeled, the two fixed anchors starred. ``poles`` = (A color, B color).
    ``fs`` overrides the type sizes (submission mode raises them to the 5 pt floor); ``title=False``
    drops the panel title where a shared column header already names the decision."""
    fs = fs or A_FS
    cA, cB = poles
    cand = block["candidates"]; partners = set(block["partners"]); anchors = list(block["anchors"])
    matrix = set(block["genes"])                                # the 5+5 carried into the matrix
    An, Bn = block["An"], block["Bn"]
    coup = np.where(cand["bias"].values >= 0, cand.get("coupA", pd.Series(0.0, cand.index)).values,
                    cand.get("coupB", pd.Series(0.0, cand.index)).values)
    size = 6 + fs["marker"] * np.clip(coup, 0, None) / (np.nanmax(coup) or 1.0)   # size ~ Jacobian coupling
    ax.scatter(cand["score_A"], cand["score_B"], s=size, c="0.82", linewidths=0, zorder=1)
    pidx = pd.Index(list(partners))
    for gset, col in [(cand.index[cand["bias"] >= 0].intersection(pidx), cA),
                      (cand.index[cand["bias"] < 0].intersection(pidx), cB)]:
        if len(gset) == 0:
            continue
        sub = cand.loc[list(gset)]
        sc = size[[cand.index.get_loc(g) for g in gset]]
        # matrix genes get a heavier outline so the 10 labeled ones stand out from the other selected
        ew = [1.0 if g in matrix else 0.3 for g in gset]
        ax.scatter(sub["score_A"], sub["score_B"], s=sc, c=col, edgecolor="k", linewidths=ew, zorder=3)
    for anc in anchors:
        if anc in cand.index:
            ax.scatter(cand.loc[anc, "score_A"], cand.loc[anc, "score_B"], s=fs["star"], marker="*",
                       c="k", zorder=5)
    if fs.get("nbins"):                                         # fewer ticks on a small panel
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(fs["nbins"]))
        ax.yaxis.set_major_locator(MaxNLocator(fs["nbins"]))
    if fs.get("headroom"):                                      # a clear strip for the one-row legend
        ax.margins(fs.get("xmargin", 0.05))
        y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + fs["headroom"] * (y1 - y0))
    try:
        from adjustText import adjust_text
    except Exception:
        adjust_text = None
    lbl = list(dict.fromkeys(list(matrix) + anchors))           # label only the 10 matrix genes + anchors
    texts = [ax.text(cand.loc[g, "score_A"], cand.loc[g, "score_B"], g, fontsize=fs["label"])
             for g in lbl if g in cand.index]
    if adjust_text and texts:                                   # avoid BOTH other labels and the markers
        adjust_text(texts, x=cand["score_A"].to_numpy(), y=cand["score_B"].to_numpy(), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.3),
                    expand=fs.get("expand", (1.3, 1.7)), **fs.get("adjkw", {}))
    ax.set_xlabel(f"driver score ({An})", fontsize=fs["axis"], labelpad=fs.get("labelpad", 4.0))
    ax.set_ylabel(f"driver score ({Bn})", fontsize=fs["axis"], labelpad=1.5)
    if title:
        ax.set_title(f"{An} vs {Bn}: driver x Jacobian coupling (size)", fontsize=fs["title"])
    ax.tick_params(labelsize=fs["tick"], pad=fs.get("tickpad", 3.5))
    from matplotlib.lines import Line2D
    ms = fs.get("legmark", 4.5)
    lead = "" if fs.get("legshort") else "toward "               # the axis labels already say "toward"
    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=cA, markeredgecolor="k",
                      markeredgewidth=0.3, markersize=ms, label=f"{lead}{An}"),
               Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=cB, markeredgecolor="k",
                      markeredgewidth=0.3, markersize=ms, label=f"{lead}{Bn}"),
               Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="k", markeredgecolor="k",
                      markersize=ms + 2.5, label="anchor")]
    ax.legend(handles=handles, fontsize=fs["legend"], loc=fs.get("legloc", "best"), frameon=True,
              framealpha=0.95, edgecolor="0.5", facecolor="white", **fs.get("legkw", {}))


B_FS = dict(tick=4.6, ylabel=6.0, ytick=6.0, legend=5.8, label="synergy (double KO vs sum of singles)")


def draw_synergy_bars(ax, block, poles, fs=None, max_partners=None):
    """Panel b: for each of the two anchors, the SYNERGY (double minus the two singles) against the ~20
    screened partners. Bars colored by the anchor's LINEAGE (``poles`` = (A color, B color)); sign is read
    from the direction (+ super-additive up / - cancellation down). The 5+5 partners with the strongest
    |synergy| are kept for the matrix (panel c) and shown in bold. ``max_partners`` keeps only the
    leading ones of the ranking, which is how the page-sized version fits legible gene names."""
    fs = fs or B_FS
    anchors = list(block["anchors"]); a_syn = block["a_synergy"]  # {anchor: Series over partners}
    matrix_genes = set(block["genes"]); acol = list(poles)

    def maxsyn(p):                                              # combined |synergy| across the two anchors
        return max((abs(float(a_syn[anc].get(p, 0.0))) for anc in anchors), default=0.0)

    partners = sorted(block["partners"], key=maxsyn, reverse=True)   # max -> min
    if max_partners:
        partners = partners[:max_partners]
    x = np.arange(len(partners)); w = 0.8 / max(len(anchors), 1)
    for ai, anc in enumerate(anchors):
        s = a_syn.get(anc, pd.Series(dtype=float))
        vals = np.array([s.get(p, np.nan) for p in partners], float)
        off = (ai - (len(anchors) - 1) / 2) * w
        ax.bar(x + off, np.nan_to_num(vals), w, color=acol[ai % 2], edgecolor="k",
               linewidth=0.3, label=f"with {anc}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(partners, rotation=90, fontsize=fs["tick"])
    for t in ax.get_xticklabels():                              # bold the genes that go into the matrix
        if t.get_text() in matrix_genes:
            t.set_fontweight("bold")
    ax.set_ylabel(fs["label"], fontsize=fs["ylabel"], labelpad=fs.get("labelpad", 4.0))
    ax.tick_params(axis="y", labelsize=fs["ytick"])
    ax.tick_params(axis="x", pad=fs.get("tickpad", 3.5))
    ax.legend(fontsize=fs["legend"], framealpha=0.9, edgecolor="0.5", loc="best", **fs.get("legkw", {}))


C_FS = dict(tick=5.2, fate_pole=4.2, syn_pole=3.9, title=6.8, cbx=1.05, cbw=0.05,
            poles=None, short=False)


def draw_square_matrix(fig, ax, block, cmap_fate, best=None, fs=None):
    """Panel c: one square heatmap over the 10 screened genes. LOWER triangle + diagonal = double-KO
    fate shift (``cmap_fate``, this decision's warm A / cool B; diagonal = single KO); UPPER triangle =
    synergy (PRGn). No in-cell numbers; the anchors' row/column and the best (top |synergy|) pairs are
    outlined. Each matrix is scaled to its own max (the shift magnitudes differ by ~30x across blocks),
    with per-matrix colorbars carrying the lineage direction (fate -> A/B; synergy super-add/cancel).
    ``fs`` overrides the type sizes and the colorbar geometry for the page-sized version."""
    fs = fs or C_FS
    genes = block["genes"]; n = len(genes); shift = block["shift"]; syn = block["synergy"]
    tril = np.tril(np.ones((n, n), bool)); triu = np.triu(np.ones((n, n), bool), 1)
    lower = np.where(tril, shift, np.nan); upper = np.where(triu, syn, np.nan)
    vs = float(np.nanmax(np.abs(lower))) or 1e-6; vy = float(np.nanmax(np.abs(upper))) or 1e-6
    cmL = cmap_fate.copy(); cmL.set_bad(alpha=0.0)               # lower = fate shift (this decision's A/B == panel d)
    cmU = plt.get_cmap("PRGn").copy(); cmU.set_bad(alpha=0.0)     # upper = synergy (super-add / cancel, NOT lineage)
    ax.imshow(ma.masked_invalid(lower), cmap=cmL, vmin=-vs, vmax=vs, aspect="equal")
    ax.imshow(ma.masked_invalid(upper), cmap=cmU, vmin=-vy, vmax=vy, aspect="equal")
    # dashed staircase separating the fate-shift (lower + diagonal) from the synergy (upper) triangle
    xs, ys = [-0.5], [-0.5]
    for i in range(n):
        xs += [i + 0.5, i + 0.5]; ys += [i - 0.5, i + 0.5]
    ax.plot(xs, ys, color="0.35", lw=0.7, ls="--")
    half = n // 2                                               # genes are [5 toward A | 5 toward B]
    if 0 < half < n:                                            # divider between the two lineage groups
        ax.axvline(half - 0.5, color="k", lw=0.8); ax.axhline(half - 0.5, color="k", lw=0.8)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(genes, rotation=90, fontsize=fs["tick"])
    ax.set_yticklabels(genes, fontsize=fs["tick"])
    ax.tick_params(length=0, pad=fs.get("tickpad", 3.5))
    anchors = set(block["anchors"])
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):   # bold the anchor ticks
        if t.get_text() in anchors:
            t.set_fontweight("bold")
    for (g1, g2) in (best or []):
        if g1 in genes and g2 in genes:
            i, j = genes.index(g1), genes.index(g2)
            for r, c in [(min(i, j), max(i, j)), (max(i, j), min(i, j))]:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="#111", lw=1.1))
    ax.add_patch(plt.Rectangle((-0.5, -0.5), n, n, fill=False, ec="k", lw=1.0))   # full box border
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    # per-matrix colorbars (the shift/synergy scales differ ~30x across blocks): lower half = fate shift
    # (this decision's warm An / cool Bn), upper half = synergy (PRGn); poles labeled by meaning, not "super"
    import matplotlib as mpl
    cbx, cbw = fs["cbx"], fs["cbw"]
    caxL = ax.inset_axes([cbx, 0.0, cbw, 0.46]); caxU = ax.inset_axes([cbx, 0.54, cbw, 0.46])
    cbL = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-vs, vs), cmap_fate), cax=caxL)
    cbU = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-vy, vy), plt.get_cmap("PRGn")), cax=caxU)
    syn_hi, syn_lo = (("synergy", "antagonism") if fs["short"] else
                      ("synergy (>sum)", "antagonism (<sum)"))
    poles = fs["poles"] or (block["An"], block["Bn"])
    for cb, hi, lo, pt in [(cbL, poles[0], poles[1], fs["fate_pole"]),
                           (cbU, syn_hi, syn_lo, fs["syn_pole"])]:
        cb.set_ticks([cb.mappable.norm.vmin, 0, cb.mappable.norm.vmax])
        cb.ax.set_yticklabels([lo, "0", hi], fontsize=pt)
        cb.ax.tick_params(length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    if fs["short"]:                                             # the column header already names the decision
        ax.set_title(f"shift$\\pm${vs:.1g}, syn$\\pm${vy:.1g}", fontsize=fs["title"], pad=2.5)
    else:
        ax.set_title(f"{block['An']} vs {block['Bn']}   (shift$\\pm${vs:.1g}, syn$\\pm${vy:.1g})",
                     fontsize=fs["title"])


def draw_flow_ip_double(ax, C, pair, basis, ngrid=30, min_count=4):
    """Panel d: KO-specific residual displacement of the joint double KO (arrows) colored by its cosine
    alignment with development (BrBG; distinct from fate RdBu_r, synergy PRGn, push PuOr)."""
    from scipy.ndimage import gaussian_filter
    emb = np.asarray(C["emb"]); flow = np.asarray(C["flow"][tuple(pair)])[:, :2]
    wt_ode = np.asarray(C["wt_ode_flow"]); wt_flow = np.asarray(C["wt_flow"])
    resid = flow - wt_ode
    rmag = np.linalg.norm(resid, axis=1)
    ip = (resid * wt_flow).sum(1) / (rmag * np.linalg.norm(wt_flow, axis=1) + 1e-12)
    x, y = emb[:, 0], emb[:, 1]
    xe = np.linspace(x.min(), x.max(), ngrid + 1); ye = np.linspace(y.min(), y.max(), ngrid + 1)
    ix = np.clip(np.digitize(x, xe) - 1, 0, ngrid - 1); iy = np.clip(np.digitize(y, ye) - 1, 0, ngrid - 1)
    S = np.zeros((ngrid, ngrid)); Wg = np.zeros((ngrid, ngrid)); N = np.zeros((ngrid, ngrid))
    np.add.at(S, (ix, iy), ip * rmag); np.add.at(Wg, (ix, iy), rmag); np.add.at(N, (ix, iy), 1.0)
    grid = np.where(N >= min_count, S / np.maximum(Wg, 1e-12), np.nan)
    V = np.nan_to_num(grid, nan=0.0); Mk = (~np.isnan(grid)).astype(float)
    Vs = gaussian_filter(V * Mk, sigma=1.0); Ms = gaussian_filter(Mk, sigma=1.0)
    smooth = np.where(Ms > 0.2, Vs / np.maximum(Ms, 1e-6), np.nan)
    cmap = plt.get_cmap("BrBG").copy(); cmap.set_bad(alpha=0.0)
    ax.scatter(x, y, c="0.9", s=2, linewidths=0, zorder=0)
    pc = ax.imshow(ma.masked_invalid(smooth.T), cmap=cmap, vmin=-1, vmax=1, origin="lower",
                   extent=[x.min(), x.max(), y.min(), y.max()], interpolation="bilinear",
                   aspect="auto", zorder=1)
    n = len(C["clusters"]); ck = C["cluster_key"]
    adm = ad.AnnData(np.zeros((n, 1), dtype=np.float32),
                     obs=pd.DataFrame({ck: pd.Categorical(C["clusters"])},
                                      index=[str(i) for i in range(n)]))
    adm.obsm[f"X_{basis}"] = emb.astype(float)
    adm.obsm[f"perturbation_flow_{basis}"] = resid.astype(float)
    _grid_arrows(ax, adm, f"perturbation_flow_{basis}", basis, color="k")
    ax.set_axis_off()
    return pc


def draw_tier_block(ax, ds, block, pairs):
    """Panel f: tier table for this block's (already globally deduplicated) best pairs (T1/T2/T3 +
    evidence). Tightly packed rows; wide evidence column so the one-line note does not overflow."""
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rows = [(f"{p[0]} + {p[1]}", *tier_of(ds, p)) for p in pairs]
    ax.text(0.0, 0.99, f"{block['An']} vs {block['Bn']}", fontsize=6.6, fontweight="bold", color="0.35",
            va="top")
    tx, ex = 0.34, 0.42                                         # tier / evidence column x (wide evidence)
    for xx, h in [(0.0, "pair"), (tx, "tier"), (ex, "evidence")]:
        ax.text(xx, 0.86, h, fontsize=5.4, fontweight="bold", va="top")
    ax.plot([0, 1], [0.83, 0.83], color="0.5", lw=0.5)
    top, rowh = 0.76, 0.105                                     # fixed compact row height (packed at top)
    for i, (pr, t, note) in enumerate(rows):
        yc = top - i * rowh
        if i % 2:
            ax.axhspan(yc - rowh / 2, yc + rowh / 2, color="0.96", zorder=0)
        ax.text(0.0, yc, pr, fontsize=5.2, va="center")
        ax.text(tx, yc, t, fontsize=5.2, fontweight="bold", va="center", color=TIER_COLOR.get(t, "0.6"))
        ax.text(ex, yc, note, fontsize=4.2, va="center", color="0.3")


# =======================================================================================
# Submission mode: the same six panels, re-laid out for one journal page.
#
# The report figure is 483 x 371 mm, which is 2.7x too wide. Type is raised to the 5 pt
# floor FIRST and the layout is then built around it, so nothing here is the poster figure
# scaled down. What changed, and why, is recorded in the docstrings of each helper.
# =======================================================================================
SUB_W, SUB_H = 180.0, 220.0          # mm; the legend goes under this on the same 247 mm page
SUB_SHORT = {"differentiated": "diff", "progenitor": "prog"}   # inline colorbar poles


class _Page:
    """Millimeters on the page, since every band position here is a physical measurement."""

    def __init__(self, w=SUB_W, h=SUB_H):
        self.w, self.h = w, h

    def rect(self, x, y, w, h):
        """A figure-fraction rectangle from (x, y) in mm with y measured DOWN from the top."""
        return [x / self.w, 1.0 - (y + h) / self.h, w / self.w, h / self.h]

    def fx(self, x):
        return x / self.w

    def fy(self, y):
        return 1.0 - y / self.h


def _text_mm(fig, s, fontsize, weight="normal"):
    """Rendered width of a string in mm, so the tier notes can be wrapped to a real column."""
    t = fig.text(0.5, 0.5, s, fontsize=fontsize, fontweight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4
    t.remove()
    return w


def _wrap_mm(fig, s, fontsize, max_mm):
    """Greedy word wrap against the measured width; keeps the evidence notes at 5 pt intact."""
    words, lines, cur = s.split(), [], ""
    for wd in words:
        trial = f"{cur} {wd}".strip()
        if cur and _text_mm(fig, trial, fontsize) > max_mm:
            lines.append(cur); cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def sub_fate_map(ax, emb, vals, vmax, title, eps, cmap, fs=5.5):
    """Panels d and e at page size: the same per-cell map as ``draw_fate_map`` with marker sizes and
    title type set for a 19 mm panel, and the dense layers rasterized (the vector version of sixteen
    3,700-cell scatters is what makes the report PDF heavy)."""
    ax.scatter(emb[:, 0], emb[:, 1], c="0.85", s=0.9, linewidths=0, zorder=0, rasterized=True)
    m = np.abs(vals) >= eps
    idx = np.where(m)[0][np.argsort(np.abs(vals[m]))]            # strong values drawn last
    sc = ax.scatter(emb[idx, 0], emb[idx, 1], c=vals[idx], cmap=cmap, vmin=-vmax, vmax=vmax,
                    s=1.8, linewidths=0, zorder=1, rasterized=True)
    ax.set_title(title, fontsize=fs, pad=1.5)
    ax.set_axis_off()
    return sc


def sub_hcbar(fig, page, x, y, w, cmap, vlo, vhi, warm, cool, title, fs=5.0):
    """A horizontal colorbar under a decision's pair of maps, poles labeled by meaning at the two ends.

    Horizontal rather than vertical because a vertical bar plus its pole labels costs 8 mm of width in
    a row that has eight panels in it, while this costs 5 mm of height in a row that has spare height.
    """
    import matplotlib as mpl
    bar_w = min(0.5 * w, 22.0)
    cax = fig.add_axes(page.rect(x + (w - bar_w) / 2, y + 2.6, bar_w, 1.3))
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(vlo, vhi), cmap), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([]); cb.outline.set_linewidth(0.4)
    fig.text(page.fx(x + w / 2), page.fy(y + 1.9), title, fontsize=fs, fontweight="bold",
             ha="center", va="bottom", color="0.25")
    fig.text(page.fx(x + (w - bar_w) / 2 - 0.7), page.fy(y + 3.25), SUB_SHORT.get(cool, cool),
             fontsize=fs, ha="right", va="center")
    fig.text(page.fx(x + (w + bar_w) / 2 + 0.7), page.fy(y + 3.25), SUB_SHORT.get(warm, warm),
             fontsize=fs, ha="left", va="center")


def sub_tier_table(fig, page, ds, x, y, w, groups, fs=5.0):
    """Panel f at page size: pair, tier and the verified one-line evidence for the pairs panels d and e
    show, grouped by lineage decision.

    The report version tables six pairs per block, of which the ones past the tiered set read
    "? needs literature"; those rows carry no information a reader can use and are what makes the panel
    tall. Here every row is a pair with an assigned tier, and the count of further nominated but
    untiered pairs is stated instead of being listed.
    """
    pair_w, tier_w = 17.0, 5.5
    ex = x + pair_w + tier_w
    ev_w = w - pair_w - tier_w
    yy = y
    for gname, pairs in groups:
        fig.text(page.fx(x), page.fy(yy), gname, fontsize=fs, fontweight="bold", color="0.35",
                 va="top")
        yy += 3.4
        for i, pr in enumerate(pairs):
            tier, note = tier_of(ds, pr)
            lines = _wrap_mm(fig, note, fs, ev_w)
            rh = max(3.3, 1.2 + 1.55 * len(lines))
            if i % 2:
                fig.patches.append(plt.Rectangle(
                    (page.fx(x - 0.5), page.fy(yy + rh)), page.fx(w + 1.0), rh / page.h,
                    transform=fig.transFigure, color="0.955", zorder=0, linewidth=0))
            yc = yy + rh / 2
            fig.text(page.fx(x), page.fy(yc), f"{pr[0]} + {pr[1]}", fontsize=fs, va="center")
            fig.text(page.fx(x + pair_w), page.fy(yc), tier, fontsize=fs, fontweight="bold",
                     va="center", color=TIER_COLOR.get(tier, "0.6"))
            fig.text(page.fx(ex), page.fy(yc), "\n".join(lines), fontsize=fs, va="center",
                     color="0.3", linespacing=1.15)
            yy += rh
        yy += 1.2
    return yy - y


def render_submission(C, ds, out_path):
    """Figure 6 on one 180 mm page: six bands, four block columns, curated left / data-driven right."""
    from submission_style import figure_for, panel_letter, save as sub_save
    import matplotlib as mpl

    lps = C["lineage_pairs"]; blocks = C["blocks"]; best_sel = C["best_sel"]
    emb = np.asarray(C["emb"]); fate_map = C["fate_map"]; push = C["push"]
    cvmax = float(np.percentile(np.concatenate([np.abs(v) for v in fate_map.values()]), 99))
    evmax = float(np.percentile(np.concatenate([np.abs(v) for v in push.values()]), 99))

    use_style(7)                                     # the Helvetica-class family and the ink colors
    page = _Page()
    fig = figure_for("double", height_mm=SUB_H)      # refuses anything off-page
    fig.canvas.draw()                                # a renderer, for the measured text wrapping

    # ---- columns ----------------------------------------------------------------------
    OL, OR, MG, CG = 5.0, 3.0, 7.0, 5.0               # outer, mid-figure and between-column gaps
    half_w = (SUB_W - OL - OR - MG) / 2
    col_w = (half_w - CG) / 2
    half_x = {"curated": OL, "discovered": OL + half_w + MG}
    HALF_NAME = {"curated": "curated anchors", "discovered": "data-driven anchors"}

    # ---- bands, in mm from the top ------------------------------------------------------
    hdr_y = 1.0                                      # half name / decision name
    a_y, a_h = 13.5, 31.0                            # selection scatter
    b_y, b_h = 53.5, 21.0                            # synergy bars (+ rotated gene names under)
    c_y, c_h = 89.0, 21.5                            # square matrices (+ rotated gene names under)
    d_y, d_h = 122.0, 20.0                           # propagated fate shift
    e_y, e_h = 152.5, 20.0                           # first-order Jacobian push
    f_y = 183.0                                      # literature tiers
    YL = 6.5                                         # room for the y ticks and label of a and b

    A_SUB = dict(label=5.0, axis=5.5, title=6.0, tick=5.0, legend=5.0, marker=26.0, star=26.0,
                 expand=(1.12, 1.30), labelpad=1.6, tickpad=1.4, legmark=2.2, nbins=4,
                 headroom=0.30, xmargin=0.08, legshort=True, legloc="upper center",
                 adjkw=dict(force_text=(0.35, 0.55), force_static=(0.2, 0.35), min_arrow_len=2.0,
                            max_move=(24, 24), expand_axes=True),
                 legkw=dict(ncol=3, handletextpad=0.15, borderpad=0.2, labelspacing=0.15,
                            columnspacing=0.6, handlelength=0.7, borderaxespad=0.15))
    B_SUB = dict(tick=5.0, ylabel=5.3, ytick=5.0, legend=5.0, label="synergy (double vs sum)",
                 labelpad=1.4, tickpad=1.2,
                 legkw=dict(handletextpad=0.3, borderpad=0.25, labelspacing=0.22,
                            handlelength=0.9, borderaxespad=0.2))
    C_SUB = dict(tick=5.0, fate_pole=5.0, syn_pole=5.0, title=5.5, cbx=1.055, cbw=0.045,
                 poles=None, short=True, tickpad=1.4)

    for origin in ("curated", "discovered"):
        hx = half_x[origin]
        fig.text(page.fx(hx + half_w / 2), page.fy(hdr_y), HALF_NAME[origin], ha="center", va="top",
                 fontsize=8.0, fontweight="bold", color="0.45")
        obl = [(k, blocks[(k, origin)]) for k in range(len(lps)) if (k, origin) in blocks]

        for ci, (k, b) in enumerate(obl):
            cx = hx + ci * (col_w + CG)
            poles = DEC_POLES[k % len(DEC_POLES)]
            fig.text(page.fx(cx + col_w / 2), page.fy(hdr_y + 5.0),
                     f"{b['An']} vs {b['Bn']}", ha="center", va="top", fontsize=6.0, color="0.25")

            # a  candidate genes in the driver-score plane, marker size = Jacobian coupling
            axa = fig.add_axes(page.rect(cx + YL, a_y, col_w - YL, a_h))
            draw_selection_scatter(axa, b, poles, fs=A_SUB, title=False)
            if origin == "curated" and ci == 0:
                panel_letter(axa, "a", dx=-0.06)

            # b  synergy of every anchor-partner pair, ranked
            axb = fig.add_axes(page.rect(cx + YL, b_y, col_w - YL, b_h))
            draw_synergy_bars(axb, b, poles, fs=B_SUB, max_partners=14)
            axb.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
            # Fold the shared power into the axis label rather than leaving it as floating
            # offset text. Matplotlib hangs that text off the top left of the axes, which is
            # where the panel letter goes, so the two collided; and at 5 pt a detached
            # multiplier is hard to associate with its axis anyway.
            fig.canvas.draw()                                    # offset text is empty until a draw
            off = axb.yaxis.get_offset_text()
            exp = off.get_text()
            if exp:
                off.set_visible(False)
                axb.set_ylabel(f"{axb.get_ylabel()} ({exp})",
                               fontsize=axb.yaxis.label.get_fontsize())
            if origin == "curated" and ci == 0:
                panel_letter(axb, "b", dx=-0.06)

            # c  the pairwise matrix; kept square, so the band height sets its side
            axc = fig.add_axes(page.rect(cx + 7.0, c_y, c_h, c_h))
            draw_square_matrix(fig, axc, b, DEC_CMAPS[k % len(DEC_CMAPS)], best=b["best"][:4],
                               fs=C_SUB)
            if origin == "curated" and ci == 0:
                panel_letter(axc, "c", dx=-0.30, dy=1.05)

        # d, e  the selected pairs, grouped by decision so each group carries its own scale
        best_o = [(pr, k) for (pr, k, o) in best_sel if o == origin]
        dgroups = []
        for (pr, k) in best_o:
            if dgroups and dgroups[-1][0] == k:
                dgroups[-1][1].append(pr)
            else:
                dgroups.append([k, [pr]])
        npairs = sum(len(prs) for _k, prs in dgroups)
        GG, IG = 4.0, 1.2                                        # group gap, in-group gap
        u = (half_w - (len(dgroups) - 1) * GG - (npairs - len(dgroups)) * IG) / max(npairs, 1)
        gx = hx
        for k, prs in dgroups:
            cm = DEC_CMAPS[k % len(DEC_CMAPS)]; An, Bn = lps[k][2], lps[k][3]
            gw = len(prs) * u + (len(prs) - 1) * IG
            for pi, pr in enumerate(prs):
                px = gx + pi * (u + IG)
                tag = f"{pr[0]}+{pr[1]}"
                for yy, store, vmax in [(d_y, fate_map, cvmax), (e_y, push, evmax)]:
                    ax = fig.add_axes(page.rect(px, yy, u, e_h))
                    v = store.get((pr, k, origin))
                    if v is None:
                        ax.set_axis_off(); continue
                    sub_fate_map(ax, emb, np.asarray(v), vmax, tag, max(0.06 * vmax, 1e-4), cm)
                    if origin == "curated" and k == dgroups[0][0] and pi == 0:
                        panel_letter(ax, "d" if yy == d_y else "e", dx=-0.02, dy=1.16)
            sub_hcbar(fig, page, gx, d_y + d_h, gw, cm, -cvmax, cvmax, An, Bn, "fate shift")
            sub_hcbar(fig, page, gx, e_y + e_h, gw, cm, -evmax, evmax, An, Bn, "Jacobian push")
            gx += gw + GG

    # f  literature evidence tiers for the pairs panels d and e show
    n_shown, n_nominated = 0, 0
    for origin in ("curated", "discovered"):
        groups = []
        for k in range(len(lps)):
            b = blocks.get((k, origin))
            if b is None:
                continue
            prs = [pr for (pr, kk, o) in best_sel if o == origin and kk == k]
            n_nominated += len(b["best"])
            n_shown += len(prs)
            groups.append((f"{b['An']} vs {b['Bn']}", prs))
        sub_tier_table(fig, page, ds, half_x[origin], f_y, half_w, groups)
    fig.text(page.fx(half_x["curated"]), page.fy(SUB_H - 2.0),
             f"T1 known and validated  |  T2 evidenced, joint role not dissected  |  T3 novel for this "
             f"process.  {n_nominated - n_shown} further nominated pairs are untiered (Supplementary).",
             fontsize=5.0, color="0.35", style="italic", va="bottom")
    fig.text(page.fx(half_x["curated"] - 4.4), page.fy(f_y - 1.0), "f", fontsize=8.0,
             fontweight="bold", va="bottom", ha="left")

    # the divider that separates the curated half from the data-driven half
    import matplotlib.lines as mlines
    xm = page.fx(half_x["discovered"] - MG / 2)
    fig.add_artist(mlines.Line2D([xm, xm], [page.fy(SUB_H - 4.0), page.fy(hdr_y - 0.5)],
                                 color="0.75", lw=0.6, transform=fig.transFigure))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub_save(fig, out_path)
    print(f"wrote {out_path}  ({SUB_W:.0f} x {SUB_H:.0f} mm)")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pancreas")
    ap.add_argument("--variant", default="", help="fit-cache tag (e.g. 'bimodal'): read the _<tag> screen "
                    "cache and write double-perturbation-<ds>_<tag>.")
    ap.add_argument("--submission", action="store_true",
                    help="re-lay out the same six panels for one journal page and write --out; the "
                         "report-sized figure is left untouched.")
    ap.add_argument("--out", default=os.path.join(paths.FIGURES_SPEC, "Figure6.pdf"),
                    help="submission-mode output path")
    args = ap.parse_args()
    ds = args.dataset
    suf = f"_{args.variant}" if args.variant else ""
    with open(f"{paths.REPORTS}/{ds}/data/double_ko_screen{suf}.pkl", "rb") as fh:
        C = pickle.load(fh)

    if args.submission:
        render_submission(C, ds, args.out)
        return

    basis = C["basis"]; emb = np.asarray(C["emb"]); lps = C["lineage_pairs"]
    blocks = C["blocks"]; best_sel = C["best_sel"]
    fate_map = C["fate_map"]; push = C["push"]; have_flow = "flow" in C and "wt_ode_flow" in C
    origins = ["curated", "discovered"]
    cvals = [np.abs(v) for v in fate_map.values()]
    cvmax = float(np.percentile(np.concatenate(cvals), 99)) if cvals else 1e-3
    evals = [np.abs(v) for v in push.values()]
    evmax = float(np.percentile(np.concatenate(evals), 99)) if evals else 1e-3

    use_style(9)
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(19.0, 14.6))     # slightly taller: more vertical room for the six bands
    proc = f"  ({PROCESS[ds]})" if ds in PROCESS else ""
    fig.text(0.05, 0.992, f"Combinatorial in-silico perturbation of {ds}{proc}", ha="left", va="top",
             fontsize=15, fontweight="bold")

    L, R = 0.05, 0.955; gapM = 0.09           # wider mid gap so the discovery y-labels clear the divider
    Lx = (L, (L + R - gapM) / 2); Rx = ((L + R + gapM) / 2, R)
    halves = {"curated": Lx, "discovered": Rx}
    # shared y-bands (filled to the bottom, no trailing whitespace). The b->c and c->d gaps are LARGE
    # because the rotated gene-name x-tick labels under b and c extend well below their axes and must
    # clear the next panel's title. a selection, b synergy, c matrix, d fate, e push, f tiers
    a_top, a_bot = 0.945, 0.850
    b_top, b_bot = 0.822, 0.750
    c_top, c_bot = 0.694, 0.514          # b->c gap holds the synergy x-labels + the matrix title
    d_top, d_bot = 0.458, 0.350          # c->d gap holds the matrix x-labels + the panel-d pair titles
    e_top, e_bot = 0.320, 0.212
    f_top, f_bot = 0.182, 0.012

    def letter(x, y, s):
        fig.text(x, y, s, fontweight="bold", fontsize=12, va="bottom", ha="left")

    import matplotlib as mpl
    SHORT = {"differentiated": "diff", "progenitor": "prog"}    # short pole labels for the inline colorbars

    def inline_cbar(cell, cmap, vlo, vhi, warm, cool, title):
        """A thin vertical colorbar placed in its own (narrow) grid cell right after the two UMAPs it
        serves, titled by the quantity (two lines, centered over the bar) with the lineage poles labeled
        warm A / cool B."""
        cell.set_axis_off()
        bx0, bw = 0.30, 0.15                                    # bar x-position + width (cell fraction)
        cax = cell.inset_axes([bx0, 0.08, bw, 0.68])
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(vlo, vhi), cmap), cax=cax)
        cb.set_ticks([vlo, 0, vhi])
        cb.ax.set_yticklabels([SHORT.get(cool, cool), "0", SHORT.get(warm, warm)], fontsize=4.4)
        cb.ax.tick_params(length=1.5, pad=1); cb.outline.set_linewidth(0.4)
        cell.text(bx0 + bw / 2, 0.99, title, fontsize=5.4, fontweight="bold", ha="center", va="top",
                  ma="center", transform=cell.transAxes)

    f_seen = set()                                             # global dedup for the panel-f tier tables
    for origin in origins:
        x0, x1 = halves[origin]
        fig.text((x0 + x1) / 2, 0.955, origin.upper(), ha="center", va="bottom", fontsize=12,
                 fontweight="bold", color="0.5")
        obl = [(k, blocks[(k, origin)]) for k in range(len(lps)) if (k, origin) in blocks]
        ndec = max(len(obl), 1)

        gs_a = fig.add_gridspec(1, ndec, left=x0, right=x1, top=a_top, bottom=a_bot, wspace=0.30)
        for ci, (k, b) in enumerate(obl):
            draw_selection_scatter(fig.add_subplot(gs_a[0, ci]), b, DEC_POLES[k % len(DEC_POLES)])

        gs_b = fig.add_gridspec(1, ndec, left=x0, right=x1, top=b_top, bottom=b_bot, wspace=0.32)
        for ci, (k, b) in enumerate(obl):
            draw_synergy_bars(fig.add_subplot(gs_b[0, ci]), b, DEC_POLES[k % len(DEC_POLES)])

        gs_c = fig.add_gridspec(1, ndec, left=x0, right=x1, top=c_top, bottom=c_bot, wspace=0.42)
        for ci, (k, b) in enumerate(obl):
            draw_square_matrix(fig, fig.add_subplot(gs_c[0, ci]), b, DEC_CMAPS[k % len(DEC_CMAPS)],
                               best=b["best"][:4])

        best_o = [(pr, k) for (pr, k, o) in best_sel if o == origin]
        # group the half's best pairs by decision (order preserved), so each decision's 2 UMAPs are
        # immediately followed by their OWN inline colorbar (a thin extra grid column per group)
        dgroups = []
        for (pr, k) in best_o:
            if dgroups and dgroups[-1][0] == k:
                dgroups[-1][1].append(pr)
            else:
                dgroups.append([k, [pr]])
        widths = []
        for k, prs in dgroups:
            widths += [1.0] * len(prs) + [0.42]          # umaps, then a thin colorbar column
        gs_d = fig.add_gridspec(1, len(widths), left=x0, right=x1, top=d_top, bottom=d_bot,
                                wspace=0.10, width_ratios=widths)
        gs_e = fig.add_gridspec(1, len(widths), left=x0, right=x1, top=e_top, bottom=e_bot,
                                wspace=0.10, width_ratios=widths)
        col = 0
        for k, prs in dgroups:
            cm = DEC_CMAPS[k % len(DEC_CMAPS)]; An, Bn = lps[k][2], lps[k][3]
            for pr in prs:
                tag = f"{pr[0]}+{pr[1]}"
                axd = fig.add_subplot(gs_d[0, col])
                v = fate_map.get((pr, k, origin))
                if v is not None:
                    draw_fate_map(axd, emb, np.asarray(v), cvmax, tag, max(0.06 * cvmax, 1e-4), cmap=cm)
                else:
                    axd.set_axis_off()
                axe = fig.add_subplot(gs_e[0, col])
                pv = push.get((pr, k, origin))
                if pv is not None:
                    draw_fate_map(axe, emb, np.asarray(pv), evmax, tag, max(0.06 * evmax, 1e-4), cmap=cm)
                else:
                    axe.set_axis_off()
                col += 1
            inline_cbar(fig.add_subplot(gs_d[0, col]), cm, -cvmax, cvmax, An, Bn, "fate\nshift")
            inline_cbar(fig.add_subplot(gs_e[0, col]), cm, -evmax, evmax, An, Bn, "jacobian\npush")
            col += 1

        gs_f = fig.add_gridspec(1, ndec, left=x0, right=x1, top=f_top, bottom=f_bot, wspace=0.28)
        for ci, (k, b) in enumerate(obl):
            picked = []                                        # globally dedup so no pair repeats across tables
            for pr in b["best"]:
                key = tuple(sorted(pr))
                if key in f_seen:
                    continue
                f_seen.add(key); picked.append(pr)
                if len(picked) >= 6:
                    break
            draw_tier_block(fig.add_subplot(gs_f[0, ci]), ds, b, picked)

    # vertical divider separating the curated (left) and discovery (right) halves
    import matplotlib.lines as mlines
    xmid = (L + R) / 2
    fig.add_artist(mlines.Line2D([xmid, xmid], [f_bot, 0.95], color="0.6", lw=1.2,
                                 transform=fig.transFigure))

    for yb, s in [(a_top, "a"), (b_top, "b"), (c_top, "c"), (d_top, "d"), (e_top, "e"), (f_top, "f")]:
        letter(L - 0.045, yb + 0.003, s)         # far enough left to clear panel b's long rotated y-label

    NAME = f"double-perturbation-{ds}{suf}"
    outdir = OUT if ds == "pancreas" else f"{OUT}/extended"
    os.makedirs(outdir, exist_ok=True)
    save(fig, f"{outdir}/{NAME}", formats=("pdf", "png"))
    print(f"wrote {outdir}/{NAME}.pdf + .png  (blocks={list(blocks)}; flow={'yes' if have_flow else 'pending'})")
    plt.close(fig)


if __name__ == "__main__":
    main()
