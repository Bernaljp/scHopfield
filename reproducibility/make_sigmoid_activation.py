"""Composite figure: the learned activation functions, and the two-component (bimodal) Hill upgrade.

The Hopfield model's nonlinearity phi_i(x_i) is fit per gene from the expression CDF (Methods). This
figure shows that (i) a single Hill misfits switch-like genes, (ii) a two-component Hill captures them
without over-firing on unimodal genes, and (iii) the two components correspond to the low/high
(progenitor/terminal) expression regimes, so the per-cell regime split tracks cell type. The activation
is fit ONCE per gene over all cells (no per-cell-type parameters); each cell then uses the Hill component
its own expression is closer to.

Panels:
  a  mechanism: single Hill vs the two-component mixture, with the per-cell regime crossover.
  b  example gene fits: empirical CDF + single-Hill + two-component fit (bimodal switch genes + a unimodal one).
  c  fit-quality gain: single-Hill vs bimodal MSE per gene (unimodal genes fall back to the diagonal).
  d  the bimodal-gene population: mixing weight and the two thresholds (k1 vs k2).
  e  two-regime map: cells colored by Hill component for representative bimodal marker genes.
  f  regime/cell-type correlation: fraction of each cell type in the high-expression component.
  g  biology: bimodal genes are enriched for lineage regulators/markers.
  h  downstream safety: per-cell energy and per-cell-type stability are preserved under the upgrade.

Run:  python reproducibility/make_sigmoid_activation.py [--dataset pancreas]
      python reproducibility/make_sigmoid_activation.py --submission   # journal page size

Two layouts, one set of panels. The default (poster) layout is what the per-dataset reports
embed and is left untouched. `--submission` re-lays the same eight panels onto a Nature
Machine Intelligence page (180 mm wide, one page tall) with the type raised to the 5 pt floor,
and writes reproducibility/figures/submission/ExtendedDataFig2.pdf.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
import guards                                                    # noqa: E402
from paper_plot_style import use_style, save, PALETTE            # noqa: E402
import anndata as ad                                             # noqa: E402
from sections import basis_of, get_colors                        # noqa: E402
from scHopfield import sigmoid                                    # noqa: E402
from scHopfield.preprocessing import hill_regime, fit_sigmoid, fit_sigmoid_bimodal  # noqa: E402

OUT = paths.FIGURES
SUB_OUT = os.path.join(paths.FIGURES_SPEC, "ExtendedDataFig2.pdf")

# --------------------------------------------------------------------------- #
# Type sizes, in one place so the two layouts cannot drift.
#
# S holds the poster-layout sizes exactly as they were hardcoded, so the default render is
# unchanged. S_SUBMISSION replaces them for the journal page, where every size has to clear
# the 5 pt floor that submission_style.save() enforces.
# --------------------------------------------------------------------------- #
S = dict(
    mech_note=6.0, mech_lab=8.0, mech_leg=5.8, mech_title=8.5, mech_tick=6.5,
    ex_title=6.6, ex_lab=7.0, ex_tick=6.0, ex_leg=5.6,
    mse_lab=8.0, mse_title=8.0, mse_note=5.8, mse_leg=6.0, mse_tick=6.5,
    pop_lab=7.5, pop_tick=6.5, pop_title=7.6,
    pr_xtick=6.6, pr_ytick=6.0, pr_ann=5.0, pr_title=7.6, pr_cblab=6.2, pr_cbtick=6.0,
    um_title=7.5, um_cbtick=6.5, um_cblab=6.0, um_leg=5.8,
    rc_tick=5.6, rc_title=7.6, rc_cbtick=5.6,
    bio_val=7.0, bio_lab=8.0, bio_title=7.8, bio_tick=6.8,
    saf_lab=7.5, saf_title=7.6, saf_tick=6.5, saf_xtick=5.6, saf_leg=6.0,
)
# Labels that carry a subscript or a superscript are set at 7.2 pt, not at the 5 to 6 pt the
# rest of the page uses: mathtext draws a sub- or superscript at 0.7x its base, so 7.2 is the
# smallest base whose subscripts still clear the 5 pt floor (7.2 x 0.7 = 5.04).
MATH_PT = 7.2
S_SUBMISSION = dict(
    mech_note=MATH_PT, mech_lab=6.6, mech_leg=5.4, mech_title=7.0, mech_tick=5.6,
    ex_title=5.8, ex_lab=6.2, ex_tick=5.4, ex_leg=5.2,
    mse_lab=MATH_PT, mse_title=7.2, mse_note=5.4, mse_leg=5.4, mse_tick=MATH_PT,
    pop_lab=6.2, pop_tick=5.4, pop_title=6.6,
    pr_xtick=MATH_PT, pr_ytick=5.4, pr_ann=5.0, pr_title=6.4, pr_cblab=5.2, pr_cbtick=5.2,
    um_title=6.4, um_cbtick=5.2, um_cblab=MATH_PT, um_leg=5.2,
    rc_tick=5.2, rc_title=6.4, rc_cbtick=5.2,
    bio_val=6.0, bio_lab=6.4, bio_title=6.4, bio_tick=5.6,
    saf_lab=6.2, saf_title=6.6, saf_tick=5.4, saf_xtick=5.2, saf_leg=5.4,
)
SUB = False        # True inside --submission, so panels can shorten their own long titles
_TIGHT_LEG: dict = {}     # extra legend kwargs, populated only in submission mode
_TIGHT_TICK: dict = {}    # extra tick kwargs (shorter ticks, tighter pad), submission mode only

# Okabe-Ito accents (house palette): component 1 = blue, component 2 = orange/vermillion.
C1 = PALETTE.get("blue", "#0072B2")
C2 = PALETTE.get("orange", "#E69F00")
CFIT = PALETTE.get("vermillion", "#D55E00")
CGREY = "0.6"

# Curated lineage regulators + hormone markers per dataset, for the enrichment panel (g) and the
# row order of panel e. Panel g's title asserts that the bimodal genes enrich for switch-like
# lineage regulators, so the comparison is only meaningful against a set curated for the dataset in
# hand: an absent set made the enrichment bar read zero percent under that title. Curated for the
# dataset this figure is published on; main() stops rather than drawing panel g for any other.
LINEAGE_BY_DATASET = {
    "pancreas": ["Neurog3", "Neurod1", "Pax4", "Arx", "Nkx2-2", "Pax6", "Isl1", "Pdx1", "Mafa",
                 "Mafb", "Fev", "Rfx6", "Insm1", "Ins1", "Ins2", "Gcg", "Sst", "Ghrl", "Ppy",
                 "Iapp", "Chga", "Chgb", "Pcsk1", "Pcsk2"],
}
LINEAGE = []       # the resolved set for this run, filled by main() before anything draws


def _to_dense(a, key):
    L = a.layers[key]
    return np.asarray(L.todense()) if hasattr(L, "todense") else np.asarray(L, dtype=float)


def _flagged(ab):
    """Boolean per-gene mask of genes that actually use a second Hill component (mix < 1)."""
    used = ab.var["scHopfield_used"].values if "scHopfield_used" in ab.var else np.ones(ab.n_vars, bool)
    mix = ab.var["sigmoid_mix"].values.astype(float)
    return used & np.isfinite(mix) & (mix < 1 - 1e-6)


# --------------------------------------------------------------------------- #
# a: mechanism schematic
# --------------------------------------------------------------------------- #
def draw_mechanism(ax, ab, gene):
    gi = ab.var_names.get_loc(gene)
    k1 = float(ab.var["sigmoid_threshold"].values[gi]); n1 = float(ab.var["sigmoid_exponent"].values[gi])
    k2 = float(ab.var["sigmoid_threshold2"].values[gi]); n2 = float(ab.var["sigmoid_exponent2"].values[gi])
    a_ = float(ab.var["sigmoid_mix"].values[gi])
    x = _to_dense(ab, ab.uns.get("scHopfield", {}).get("spliced_key", "Ms"))[:, gi]
    xmax = float(np.nanpercentile(x[np.isfinite(x)], 99.5)) or max(k1, k2) * 2
    xs = np.linspace(0, xmax, 400)
    lo, hi = (k1, k2) if k1 <= k2 else (k2, k1)
    cross = 0.5 * (lo + hi)                                       # nearest-component crossover
    ax.axvspan(0, cross, color=C1, alpha=0.06); ax.axvspan(cross, xmax, color=C2, alpha=0.06)
    ax.plot(xs, sigmoid(xs, k1, n1), color=C1, lw=2.0, label="component 1 (low regime)")
    ax.plot(xs, sigmoid(xs, k2, n2), color=C2, lw=2.0, label="component 2 (high regime)")
    ax.plot(xs, a_ * sigmoid(xs, k1, n1) + (1 - a_) * sigmoid(xs, k2, n2), color=CFIT, lw=1.3, ls="--",
            label="fitted CDF mixture")
    # on the journal page the rule stops short of the note, which now hangs inside the axes
    ax.axvline(cross, color="0.35", lw=0.9, ls=":", **({"ymax": 0.85} if SUB else {}))
    # cells below x* are assigned to component 1, above to component 2 (hill_regime nearest-threshold rule)
    # the note sits above the curves; on the journal page it hangs INSIDE the axes instead,
    # where the two lines cannot crowd the title
    ax.text(cross, 1.12 if SUB else 1.03, "component crossover\n$x^*=(k_1+k_2)/2$",
            fontsize=S["mech_note"], ha="center", va="top" if SUB else "bottom", color="0.3")
    ax.set_xlabel(f"{gene} expression", fontsize=S["mech_lab"])
    ax.set_ylabel("activation  $\\varphi(x)$", fontsize=S["mech_lab"])
    ax.set_ylim(-0.02, 1.14); ax.set_xlim(0, xmax)
    # bottom-right gap: below component 2's late rise, with the blue plateau and dashed mixture overhead
    ax.legend(fontsize=S["mech_leg"], loc="lower right", bbox_to_anchor=(0.99, 0.02), frameon=False,
              **_TIGHT_LEG)
    ax.set_title("two-component activation", fontsize=S["mech_title"])
    ax.tick_params(labelsize=S["mech_tick"])


# --------------------------------------------------------------------------- #
# b: example gene CDF fits
# --------------------------------------------------------------------------- #
def draw_example_fits(fig, gs_cell, ab, genes, min_th=0.05):
    sub = gs_cell.subgridspec(1, len(genes), wspace=0.42)
    axes = []
    x_all = _to_dense(ab, ab.uns.get("scHopfield", {}).get("spliced_key", "Ms"))
    for j, g in enumerate(genes):
        ax = fig.add_subplot(sub[0, j]); axes.append(ax)
        gi = ab.var_names.get_loc(g)
        xg = x_all[:, gi]; xg = np.sort(xg[np.isfinite(xg)])
        gmax = float(xg.max()) or 1.0
        val = xg[xg > min_th * gmax]
        if val.size < 8:
            ax.set_axis_off(); continue
        y = np.linspace(0, 1, val.size)
        k1, n1, off, mse_s = fit_sigmoid(val)
        k1b, n1b, k2b, n2b, ab_, offb, mse_b, isbi = fit_sigmoid_bimodal(val)
        xs = np.linspace(val.min(), val.max(), 300)
        ax.plot(val, y, color="0.35", lw=0, marker="o", ms=1.3, alpha=0.5, label="empirical CDF")
        ax.plot(xs, sigmoid(xs, k1, n1), color=C1, lw=1.6, label="single Hill")
        ax.plot(xs, ab_ * sigmoid(xs, k1b, n1b) + (1 - ab_) * sigmoid(xs, k2b, n2b),
                color=CFIT, lw=1.6, ls="--", label="two-component")
        tag = "bimodal" if isbi else "unimodal"
        ax.set_title(f"{g}  ({tag})\nMSE {mse_s:.1e}$\\to${mse_b:.1e}", fontsize=S["ex_title"])
        ax.set_xlabel("expression", fontsize=S["ex_lab"]); ax.tick_params(labelsize=S["ex_tick"])
        if j == 0:
            ax.set_ylabel("cumulative fraction", fontsize=S["ex_lab"])
            ax.legend(fontsize=S["ex_leg"], loc="lower right", frameon=False, **_TIGHT_LEG)
    return axes[0]


# --------------------------------------------------------------------------- #
# c: fit-quality gain (single vs bimodal MSE)
# --------------------------------------------------------------------------- #
def draw_mse_gain(ax, asingle, ab):
    genes = [g for g in ab.var_names if g in asingle.var_names]
    gi_b = [ab.var_names.get_loc(g) for g in genes]
    gi_s = [asingle.var_names.get_loc(g) for g in genes]
    mse_s = asingle.var["sigmoid_mse"].values[gi_s].astype(float)
    mse_b = ab.var["sigmoid_mse"].values[gi_b].astype(float)
    fl = _flagged(ab)[gi_b]
    eps = 1e-6
    ax.scatter(mse_s[~fl] + eps, mse_b[~fl] + eps, s=5, c=CGREY, alpha=0.4, linewidths=0,
               label=f"unimodal ({int((~fl).sum())})")
    ax.scatter(mse_s[fl] + eps, mse_b[fl] + eps, s=8, c=CFIT, alpha=0.75, linewidths=0,
               label=f"bimodal ({int(fl.sum())})")
    lim = [eps, max(mse_s.max(), mse_b.max()) * 1.3]
    ax.plot(lim, lim, color="0.4", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("single-Hill MSE", fontsize=S["mse_lab"])
    ax.set_ylabel("two-component MSE", fontsize=S["mse_lab"])
    med_gain = 100 * (1 - np.median(mse_b[fl] / np.clip(mse_s[fl], eps, None)))
    title = (f"fit gain on bimodal genes\n(median $-${med_gain:.0f}% MSE)" if SUB
             else f"fit gain on bimodal genes (median $-${med_gain:.0f}% MSE)")
    ax.set_title(title, fontsize=S["mse_title"])
    ax.text(0.03, 0.97, "specificity: single-Hill\ntoggle control flags 0 genes",
            transform=ax.transAxes, fontsize=S["mse_note"], va="top", color="0.3")
    ax.legend(fontsize=S["mse_leg"], loc="lower right", frameon=False, **_TIGHT_LEG)
    ax.tick_params(labelsize=S["mse_tick"])


# --------------------------------------------------------------------------- #
# d: bimodal-gene population (mix weight + thresholds)
# --------------------------------------------------------------------------- #
def draw_population(fig, gs_cell, ab, wspace=0.85):
    sub = gs_cell.subgridspec(1, 2, wspace=wspace, width_ratios=[1.0, 0.9])
    fl = _flagged(ab)
    a_ = ab.var["sigmoid_mix"].values[fl].astype(float)
    ax0 = fig.add_subplot(sub[0, 0])
    ax0.hist(a_, bins=20, color=C1, alpha=0.85, edgecolor="0.3", linewidth=0.4)
    ax0.set_xlabel("mixing weight $a$ (component 1)", fontsize=S["pop_lab"])
    ax0.set_ylabel("bimodal genes", fontsize=S["pop_lab"]); ax0.tick_params(labelsize=S["pop_tick"])
    ax0.set_title(f"{int(fl.sum())} of {int(ab.var['scHopfield_used'].sum())} genes bimodal",
                  fontsize=S["pop_title"])
    draw_param_ratio(fig, sub[0, 1], ab)         # parameter-ratio heatmap (the 'how different are k, n' view)
    return ax0


# --------------------------------------------------------------------------- #
# e: two-regime UMAP for representative bimodal marker genes
# --------------------------------------------------------------------------- #
def _param_ratios(ab, gene):
    """Per-gene comparison of the two Hill components: the smaller parameter as a fraction of the larger
    (in (0,1]), for the threshold k and the steepness n. 1 = the two components are identical (barely
    bimodal); toward 0 = well-separated switch points. Both are dimensionless, so ONE colormap serves all
    genes (k is in each gene's own expression units and is not otherwise comparable across genes; n, the
    Hill coefficient, is already dimensionless)."""
    gi = ab.var_names.get_loc(gene)
    k1 = abs(float(ab.var["sigmoid_threshold"].values[gi])); k2 = abs(float(ab.var["sigmoid_threshold2"].values[gi]))
    n1 = float(ab.var["sigmoid_exponent"].values[gi]); n2 = float(ab.var["sigmoid_exponent2"].values[gi])
    kr = min(k1, k2) / max(k1, k2) if max(k1, k2) > 0 else 1.0
    nr = min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 1.0
    return kr, nr


def draw_param_ratio(fig, gs_cell, ab, n_max_rows=20):
    """Panel e: heatmap of the two-component parameter ratios per gene (rows), threshold and steepness
    (columns), one shared colormap. Rows = the curated lineage genes that are bimodal, filled out with the
    most-separated other bimodal genes, sorted most-separated first."""
    fl = _flagged(ab)
    lineage = [g for g in LINEAGE if g in ab.var_names and fl[ab.var_names.get_loc(g)]]
    # lead with the curated lineage genes (the recognizable ones); only if too few, top up with the
    # most-separated other bimodal genes.
    if len(lineage) < 10:
        others = sorted([g for g in ab.var_names[fl] if g not in lineage],
                        key=lambda g: _param_ratios(ab, g)[0])
        genes = (lineage + others)[:n_max_rows]
    else:
        genes = lineage[:n_max_rows]
    M = np.array([_param_ratios(ab, g) for g in genes])
    order = np.argsort(M[:, 0]); M = M[order]; genes = [genes[i] for i in order]
    is_lin = [g in lineage for g in genes]
    ax = fig.add_subplot(gs_cell)
    im = ax.imshow(M, aspect="auto", cmap="Purples_r", vmin=0, vmax=1)   # low ratio (separated) = dark
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["threshold\n$k_{\\min}/k_{\\max}$", "steepness\n$n_{\\min}/n_{\\max}$"],
                       fontsize=S["pr_xtick"])
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=S["pr_ytick"])
    ax.tick_params(**_TIGHT_TICK)
    for tick, lin in zip(ax.get_yticklabels(), is_lin):                # bold the curated lineage genes
        tick.set_fontweight("bold" if lin else "normal")
    for i in range(M.shape[0]):                                          # annotate the fraction values
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=S["pr_ann"],
                    color="w" if M[i, j] < 0.5 else "k")
    ax.set_title("two-component parameter ratio\n(1 = identical, lower = separated)",
                 fontsize=S["pr_title"])
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_ticks([0, 0.5, 1])
    cb.set_label("smaller / larger", fontsize=S["pr_cblab"]); cb.ax.tick_params(labelsize=S["pr_cbtick"])
    return ax


def draw_regime_umaps(fig, gs_cell, ab, basis, genes, umap_wspace=0.10, dot=3, cb_w=0.006,
                      cb_gap=0.008, cb_orientation="vertical"):
    """Panel e: per-gene BINARY regime map that also carries the two thresholds' ratio. Each cell is hard-
    assigned to its nearer Hill component (hill_regime), so every UMAP has exactly TWO colors. The high-
    threshold component is fixed at 1 (the same color in every gene); the low-threshold component is colored
    by k_min/k_max, which differs per gene. So the shared color marks the high regime, the other color
    reports how separated that gene's two switch points are, and one colorbar compares the ratio across
    genes (absolute thresholds span >10^4 and are not otherwise comparable, but the ratio is)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    horizontal_cb = (cb_orientation == "horizontal")
    emb = np.asarray(ab.obsm[f"X_{basis}"])[:, :2]
    X = _to_dense(ab, ab.uns.get("scHopfield", {}).get("spliced_key", "Ms"))
    HIGH_COLOR = "#333333"                            # fixed dark tone for the high-threshold component
    RCMAP = plt.get_cmap("autumn")                   # low component's ratio: warm, avoids viridis' broad blue

    def _ratio(g):
        gi = ab.var_names.get_loc(g)
        a = abs(float(ab.var["sigmoid_threshold"].values[gi])); b = abs(float(ab.var["sigmoid_threshold2"].values[gi]))
        return min(a, b) / max(a, b) if max(a, b) > 0 else 1.0

    rs = [_ratio(g) for g in genes]                  # data-tight, shared scale so the low colors spread + compare
    norm = Normalize(max(0.0, min(rs) - 0.02), min(1.0, max(rs) + 0.02))
    sub = gs_cell.subgridspec(1, len(genes), wspace=umap_wspace)
    axes = []
    for j, g in enumerate(genes):
        ax = fig.add_subplot(sub[0, j]); axes.append(ax)
        gi = ab.var_names.get_loc(g)
        x = X[:, gi].astype(float)
        k1 = float(ab.var["sigmoid_threshold"].values[gi]); k2 = float(ab.var["sigmoid_threshold2"].values[gi])
        reg = hill_regime(x, k1, k2)                 # 0 -> component 1 (k1), 1 -> component 2 (k2)
        high_mask = (reg == 1) if abs(k2) >= abs(k1) else (reg == 0)   # cells in the high-threshold component
        rcol = RCMAP(norm(_ratio(g)))                # the low component's single ratio color for this gene
        ax.scatter(emb[~high_mask, 0], emb[~high_mask, 1], color=rcol, s=dot, linewidths=0,
                   rasterized=SUB)
        ax.scatter(emb[high_mask, 0], emb[high_mask, 1], color=HIGH_COLOR, s=dot, linewidths=0,
                   rasterized=SUB)
        ax.set_title(g, fontsize=S["um_title"], pad=2 if SUB else None); ax.set_axis_off()
    sm = ScalarMappable(norm=norm, cmap=RCMAP); sm.set_array([])
    if horizontal_cb:
        # journal page: the bar lies under the row, which costs 6 mm of height instead of the
        # 20 mm of width a vertical bar plus its rotated label would take out of the panel block
        b0, b1 = axes[0].get_position(), axes[-1].get_position()
        span = b1.x1 - b0.x0
        cax = fig.add_axes([b0.x0 + 0.34 * span, b0.y0 - cb_gap, 0.40 * span, cb_w])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("low component  $k_{\\min}/k_{\\max}$", fontsize=S["um_cblab"], labelpad=1)
        # the high-component key sits beside the bar, never over the cells
        fig.legend(handles=[Patch(facecolor=HIGH_COLOR, label="high component")],
                   loc="center right", bbox_to_anchor=(b0.x0 + 0.30 * span, b0.y0 - cb_gap + cb_w / 2),
                   bbox_transform=fig.transFigure, fontsize=S["um_leg"], frameon=False,
                   handlelength=1.0, handletextpad=0.4, borderaxespad=0.0)
    else:
        bb = axes[-1].get_position()
        cax = fig.add_axes([bb.x1 + cb_gap, bb.y0 + bb.height * 0.22, cb_w, bb.height * 0.56])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("low component\n$k_{\\min}/k_{\\max}$", fontsize=S["um_cblab"])
        axes[0].legend(handles=[Patch(facecolor=HIGH_COLOR, label="high comp")], fontsize=S["um_leg"],
                       loc="lower left", frameon=False, handlelength=1.0, handletextpad=0.4)
    cb.ax.tick_params(labelsize=S["um_cbtick"], **_TIGHT_TICK)
    return axes[0]


# --------------------------------------------------------------------------- #
# f: regime / cell-type correlation
# --------------------------------------------------------------------------- #
def draw_regime_celltype(fig, gs_cell, ab, ck, order, top_n=16):
    fl_genes = [g for g in ab.var_names[_flagged(ab)]]
    cl = ab.obs[ck].astype(str).values
    x_all = _to_dense(ab, ab.uns.get("scHopfield", {}).get("spliced_key", "Ms"))
    rows, spread = [], []
    for g in fl_genes:
        gi = ab.var_names.get_loc(g)
        k1 = float(ab.var["sigmoid_threshold"].values[gi]); k2 = float(ab.var["sigmoid_threshold2"].values[gi])
        reg = hill_regime(x_all[:, gi], k1, k2)
        frac = np.array([reg[cl == c].mean() if (cl == c).any() else np.nan for c in order])
        rows.append(frac); spread.append(np.nanmax(frac) - np.nanmin(frac))
    spread = np.array(spread)
    idx = np.argsort(-spread)[:top_n]                            # most cell-type-associated bimodal genes
    M = np.array(rows)[idx]; sel_genes = [fl_genes[i] for i in idx]
    ax = fig.add_subplot(gs_cell)
    im = ax.imshow(M, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=S["rc_tick"])
    ax.set_yticks(range(len(sel_genes))); ax.set_yticklabels(sel_genes, fontsize=S["rc_tick"])
    ax.tick_params(**_TIGHT_TICK)
    title = ("fraction of each cell type in the\nhigh-expression component" if SUB
             else "fraction of each cell type in the high-expression component")
    ax.set_title(title, fontsize=S["rc_title"])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=S["rc_cbtick"], **_TIGHT_TICK)
    return ax


# --------------------------------------------------------------------------- #
# g: biology of bimodal genes (enrichment among lineage regulators)
# --------------------------------------------------------------------------- #
def draw_biology(ax, ab):
    fl = _flagged(ab)
    used = ab.var["scHopfield_used"].values
    lineage = [g for g in LINEAGE if g in ab.var_names and used[ab.var_names.get_loc(g)]]
    lin_idx = [ab.var_names.get_loc(g) for g in lineage]
    frac_all = fl[used].mean()
    frac_lin = fl[lin_idx].mean() if lin_idx else 0.0
    bars = ax.bar(["all fitted\ngenes", "lineage\nregulators"], [frac_all, frac_lin],
                  color=[CGREY, CFIT], edgecolor="0.3", linewidth=0.5, width=0.6)
    for b, v in zip(bars, [frac_all, frac_lin]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{100*v:.0f}%", ha="center",
                fontsize=S["bio_val"])
    ax.set_ylabel("fraction flagged bimodal", fontsize=S["bio_lab"])
    ax.set_ylim(0, max(frac_lin, frac_all) * 1.35 + 0.05)
    title = ("bimodal genes enrich for\nswitch-like lineage regulators" if SUB
             else "bimodal genes enrich for switch-like\nlineage regulators")
    ax.set_title(title, fontsize=S["bio_title"])
    ax.tick_params(labelsize=S["bio_tick"], **_TIGHT_TICK)


# --------------------------------------------------------------------------- #
# h: downstream safety (energy + stability preserved)
# --------------------------------------------------------------------------- #
def draw_safety(fig, gs_cell, asingle, ab, ck, order, colors, wspace=0.42):
    sub = gs_cell.subgridspec(1, 2, wspace=wspace)
    # h1: per-cell total energy single vs bimodal
    ax0 = fig.add_subplot(sub[0, 0])
    es = asingle.obs["energy_total"].values.astype(float); eb = ab.obs["energy_total"].values.astype(float)
    r = np.corrcoef(es, eb)[0, 1]
    cl = ab.obs[ck].astype(str).values
    ax0.scatter(es, eb, s=3, c=[colors.get(c, "0.6") for c in cl], alpha=0.5, linewidths=0,
                rasterized=SUB)
    lim = [min(es.min(), eb.min()), max(es.max(), eb.max())]
    ax0.plot(lim, lim, color="0.4", lw=0.8, ls="--")
    ax0.set_xlabel("single-Hill total energy", fontsize=S["saf_lab"])
    ax0.set_ylabel("bimodal total energy", fontsize=S["saf_lab"])
    ax0.set_title(f"per-cell energy preserved (r={r:.2f})", fontsize=S["saf_title"])
    ax0.tick_params(labelsize=S["saf_tick"], **_TIGHT_TICK)
    # h2: per-cell-type leading real eigenvalue single vs bimodal
    ax1 = fig.add_subplot(sub[0, 1])
    def ct_mean(a, col):
        clv = a.obs[ck].astype(str).values; v = a.obs[col].values.astype(float)
        return np.array([np.nanmean(v[clv == c]) if (clv == c).any() else np.nan for c in order])
    ls = ct_mean(asingle, "jacobian_leading_real"); lb = ct_mean(ab, "jacobian_leading_real")
    xp = np.arange(len(order)); w = 0.38
    ax1.bar(xp - w / 2, ls, w, color=CGREY, edgecolor="0.3", linewidth=0.4, label="single")
    ax1.bar(xp + w / 2, lb, w, color=CFIT, edgecolor="0.3", linewidth=0.4, label="bimodal")
    ax1.set_xticks(xp); ax1.set_xticklabels(order, rotation=60, ha="right", fontsize=S["saf_xtick"])
    ax1.set_ylabel("leading eig (Re)", fontsize=S["saf_lab"])
    ax1.tick_params(axis="y", labelsize=S["saf_tick"]); ax1.tick_params(**_TIGHT_TICK)
    ax1.set_title("stability ordering preserved", fontsize=S["saf_title"])
    ax1.legend(fontsize=S["saf_leg"], frameon=False, **_TIGHT_LEG)
    return ax0


# --------------------------------------------------------------------------- #
# Journal-page layout (--submission): the same eight panels on 180 mm x 213 mm.
#
# The poster layout puts one wide row per pair of panels on a 381 x 401 mm canvas. Shrinking
# that to a page would land the type at 1.3 to 3.3 pt, so the page version instead raises the
# type to the 5 pt floor first and then buys the space back: four rows of two panel blocks,
# the two-regime colorbar laid flat under its row instead of standing beside it, and the
# cell-type heatmap cut from 16 to 14 genes. No panel is dropped.
# --------------------------------------------------------------------------- #
PAGE_H_MM = 209.0

def _band(top_mm, height_mm, h_mm=PAGE_H_MM):
    """(top, bottom) in figure fractions for a row whose axes span top_mm..top_mm+height_mm."""
    return 1.0 - top_mm / h_mm, 1.0 - (top_mm + height_mm) / h_mm


def layout_submission(fig, D):
    """Draw all eight panels onto a page-sized figure. Returns the (axis, letter, dy_mm) list."""
    ab, asingle, ck, order, colors, basis = D["ab"], D["asingle"], D["ck"], D["order"], D["colors"], D["basis"]
    H = PAGE_H_MM
    L = 0.058

    # row 1: a (mechanism) | b (three example CDF fits)
    t, b = _band(11.0, 30.0)
    r1 = fig.add_gridspec(1, 2, top=t, bottom=b, left=L, right=0.978,
                          width_ratios=[1.0, 1.60], wspace=0.30)
    ax_a = fig.add_subplot(r1[0, 0]); draw_mechanism(ax_a, ab, D["schematic_gene"])
    ax_b = draw_example_fits(fig, r1[0, 1], ab, D["ex_genes"])

    # row 2: c (MSE gain) | d (mixing weight + the two-component parameter ratio)
    t, b = _band(60.0, 32.0)
    r2 = fig.add_gridspec(1, 2, top=t, bottom=b, left=L, right=0.905,
                          width_ratios=[1.0, 1.60], wspace=0.34)
    ax_c = fig.add_subplot(r2[0, 0]); draw_mse_gain(ax_c, asingle, ab)
    ax_d = draw_population(fig, r2[0, 1], ab, wspace=0.60)

    # row 3: e (two-regime maps) | f (regime fraction per cell type)
    t, b = _band(110.0, 32.0)
    r3 = fig.add_gridspec(1, 2, top=t, bottom=b, left=L, right=0.935,
                          width_ratios=[1.45, 1.0], wspace=0.28)
    ax_e = draw_regime_umaps(fig, r3[0, 0], ab, basis, D["regime_genes"], umap_wspace=0.08,
                             dot=1.6, cb_w=0.010, cb_gap=0.030, cb_orientation="horizontal")
    ax_f = draw_regime_celltype(fig, r3[0, 1], ab, ck, order, top_n=14)

    # row 4: g (enrichment) | h (energy and stability preserved)
    t, b = _band(166.0, 30.0)
    r4 = fig.add_gridspec(1, 2, top=t, bottom=b, left=L, right=0.978,
                          width_ratios=[0.62, 2.0], wspace=0.34)
    ax_g = fig.add_subplot(r4[0, 0]); draw_biology(ax_g, ab)
    ax_h = draw_safety(fig, r4[0, 1], asingle, ab, ck, order, colors, wspace=0.50)

    # dy_mm clears each panel's own title: 7.5 mm over a two-line title, 3.5 mm over one line.
    return [(ax_a, "a", 3.5), (ax_b, "b", 7.5), (ax_c, "c", 7.5), (ax_d, "d", 3.5),
            (ax_e, "e", 3.5), (ax_f, "f", 7.5), (ax_g, "g", 7.5), (ax_h, "h", 3.5)]


def render_submission(D, out_path=SUB_OUT):
    global SUB, _TIGHT_LEG, _TIGHT_TICK
    from submission_style import figure_for, save as save_submission, TYPE_PANEL_LETTER
    SUB = True
    S.update(S_SUBMISSION)
    _TIGHT_LEG = dict(labelspacing=0.25, handlelength=1.2, handletextpad=0.4, borderpad=0.15,
                      borderaxespad=0.2)
    _TIGHT_TICK = dict(length=1.6, pad=1.4)
    use_style(7)                                   # house font family and palette
    fig = figure_for("double", height_mm=PAGE_H_MM)   # then the page geometry and type floor
    placed = layout_submission(fig, D)
    for ax, letter, dy_mm in placed:
        bb = ax.get_position()
        fig.text(bb.x0 - 5.0 / 180.0, bb.y1 + dy_mm / PAGE_H_MM, letter, fontweight="bold",
                 fontsize=TYPE_PANEL_LETTER, va="bottom", ha="left")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_submission(fig, out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pancreas")
    ap.add_argument("--submission", action="store_true",
                    help="render at journal page size to " + SUB_OUT)
    args = ap.parse_args()
    ds = args.dataset
    from config import DATASETS
    cfg = DATASETS[ds]; ck = cfg["cluster_key"]

    # Panels e and g are drawn against the dataset's curated lineage regulators. Resolve the set
    # before anything opens a file, so an uncurated --dataset stops here rather than at the point
    # where panel g would have drawn a zero percent bar under an enrichment claim.
    global LINEAGE
    LINEAGE = guards.require_dataset_entry(
        LINEAGE_BY_DATASET, ds, "LINEAGE_BY_DATASET in make_sigmoid_activation.py",
        "Extended Data Fig. 2 panels e and g",
        how="curate the dataset's lineage regulators and hormone markers, the way 'pancreas' is")

    def _pick(*cands):                                       # first existing path (robust to canonical state)
        return next((p for p in cands if os.path.exists(p)), cands[-1])
    base = f"{paths.REPORTS}/{ds}/data"
    # bimodal fit: the explicit tagged cache, or the canonical adata once bimodal is promoted.
    p_bi = _pick(f"{base}/adata_analyzed_bimodal.h5ad", f"{base}/adata_analyzed.h5ad")
    # single-Hill fit for the comparison panels (c, h): the backup made when bimodal became canonical,
    # else the canonical adata (pre-promotion it is still single-Hill). Once bimodal IS canonical that
    # fallback lands on the same file as the bimodal arm, and panels c and h then compare a fit with
    # itself and draw a flat zero improvement. Which era we are in is not worth guessing: the two arms
    # collapsing onto one path is the defect itself, whichever fit the canonical object holds.
    p_single = _pick(f"{base}/adata_analyzed_singlehill.h5ad", f"{base}/adata_analyzed.h5ad")
    guards.require_distinct(
        p_bi, p_single,
        "Extended Data Fig. 2 panels c and h, the two-component versus single-Hill comparison",
        how=(f"re-fit {ds} with config.BIMODAL_HILL = False and save the result as "
             f"{base}/adata_analyzed_singlehill.h5ad"))
    ab = ad.read_h5ad(p_bi)
    asingle = ad.read_h5ad(p_single)
    basis = basis_of(ab); colors = get_colors(ab, ck)
    present = [c for c in ab.obs[ck].astype(str).unique()]
    order = [c for c in (cfg.get("order") or present) if c in present]

    # pick example + regime-map genes: top bimodal genes by MSE improvement, that are lineage markers if possible
    fl = _flagged(ab)
    fl_genes = list(ab.var_names[fl])
    gain = {}
    for g in fl_genes:
        gi_b = ab.var_names.get_loc(g)
        if g in asingle.var_names:
            ms = float(asingle.var["sigmoid_mse"].values[asingle.var_names.get_loc(g)])
            mb = float(ab.var["sigmoid_mse"].values[gi_b])
            gain[g] = ms - mb
    ranked = sorted(gain, key=lambda g: -gain[g])
    lineage_bi = [g for g in LINEAGE if g in fl_genes]
    ex_genes = (lineage_bi[:2] + [g for g in ranked if g not in lineage_bi][:1])[:3]
    unimodal = [g for g in ab.var_names if not fl[ab.var_names.get_loc(g)]
                and ab.var["scHopfield_used"].values[ab.var_names.get_loc(g)]]
    ex_genes = (ex_genes + unimodal[:1])[:3] if len(ex_genes) < 3 else ex_genes[:2] + unimodal[:1]
    regime_genes = (lineage_bi[:4] or ranked[:4])[:4]
    schematic_gene = ex_genes[0]
    print(f"[{ds}] flagged bimodal {int(fl.sum())}/{int(ab.var['scHopfield_used'].sum())}; "
          f"examples={ex_genes}; regime={regime_genes}", flush=True)

    if args.submission:
        render_submission(dict(ab=ab, asingle=asingle, ck=ck, order=order, colors=colors,
                               basis=basis, ex_genes=ex_genes, regime_genes=regime_genes,
                               schematic_gene=schematic_gene))
        return

    use_style(9)
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(15.0, 15.8))     # shorter than wide: the panels read squarer, not elongated
    L, R = 0.06, 0.96
    fig.text(0.05, 0.995, f"Activation-function fits and the two-component (bimodal) Hill: {ds}",
             ha="left", va="top", fontsize=14, fontweight="bold")

    def label(ax, letter, dx=0.052, dy=0.014):
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + dy, letter, fontweight="bold", fontsize=13, va="bottom", ha="left")

    # row 1: a (schematic) | b (example fits)
    r1 = fig.add_gridspec(1, 2, top=0.955, bottom=0.775, left=L, right=R, width_ratios=[1.0, 1.5], wspace=0.24)
    ax_a = fig.add_subplot(r1[0, 0]); draw_mechanism(ax_a, ab, schematic_gene)
    ax_b = draw_example_fits(fig, r1[0, 1], ab, ex_genes)

    # row 2: c (MSE gain) | d (population: mixing weight + parameter-ratio heatmap)
    r2 = fig.add_gridspec(1, 2, top=0.725, bottom=0.545, left=L, right=R, width_ratios=[1.0, 1.5], wspace=0.24)
    ax_c = fig.add_subplot(r2[0, 0]); draw_mse_gain(ax_c, asingle, ab)
    ax_d = draw_population(fig, r2[0, 1], ab)

    # row 3: e (per-gene UMAP, expression relative to the two switch points) | f (regime/celltype heatmap)
    r3 = fig.add_gridspec(1, 2, top=0.495, bottom=0.290, left=L, right=R, width_ratios=[1.5, 1.0], wspace=0.22)
    ax_e = draw_regime_umaps(fig, r3[0, 0], ab, basis, regime_genes)
    ax_f = draw_regime_celltype(fig, r3[0, 1], ab, ck, order)

    # row 4: g (biology) | h (safety)
    r4 = fig.add_gridspec(1, 2, top=0.235, bottom=0.045, left=L, right=R, width_ratios=[0.7, 1.6], wspace=0.28)
    ax_g = fig.add_subplot(r4[0, 0]); draw_biology(ax_g, ab)
    ax_h = draw_safety(fig, r4[0, 1], asingle, ab, ck, order, colors)

    for ax, lt in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"),
                   (ax_e, "e"), (ax_f, "f"), (ax_g, "g"), (ax_h, "h")]:
        label(ax, lt)

    save(fig, f"{OUT}/sigmoid-activation-{ds}", formats=("pdf", "png"))
    print(f"wrote {OUT}/sigmoid-activation-{ds}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
