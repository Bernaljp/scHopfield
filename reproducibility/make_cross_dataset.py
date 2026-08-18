"""Cross-dataset generality figure: the same regulatory-dynamical readouts, summarized as one point per
dataset across seven single-cell developmental datasets spanning four biological systems. Each panel is
the cross-dataset counterpart of one per-dataset analysis (the per-dataset figures are left untouched).

  a  energy-basin depth       -- interaction-energy depth per cell (within-dataset z), progenitor vs terminal
                                (terminals sit in deeper basins in every dataset)
  b  regulator recovery       -- rank-percentile of each system's curated lineage TFs among all genes (driver score)
  c  identifiability          -- effective dimensionality of the expression manifold, as a percent of the modeled
                                genes (small everywhere, so the [W|I] inverse problem is under-determined)
  d  regulatory concentration -- share of total TF out-strength held by the top ten regulators (hub-dominated)
  e  activation bimodality    -- fraction of genes with a two-component Hill, and its enrichment among lineage TFs
  f  velocity reconstruction  -- median per-cell cosine, cell-type-specific fit against one global fit (in sample)
  g  identifiability          -- split-half correlation of the unconstrained W against neighbor fraction, 4 datasets

Reads the canonical (bimodal) report AnnData + driver-score CSVs. Run:
  python reproducibility/make_cross_dataset.py                # poster-size figure, figures_named/
  python reproducibility/make_cross_dataset.py --submission   # journal page size, figure_variants/spec/

The default (no flag) output is what the per-dataset reports still use, so it is left exactly as it
was; --submission re-lays panels a to e out for a 180 mm journal page and adds f and g, which
exist only in the submission variant.
"""
from __future__ import annotations
import argparse, json, os, sys
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

OUT = paths.FIGURES
SPEC_OUT = paths.FIGURES_SPEC                    # journal-page variant, --submission only
CACHE = os.path.join(paths.CACHE, "_cross_dataset_cache.json")  # scalars only; the AnnData reads take minutes
# Panel f: median per-cell velocity cosine per dataset, under the cell-type-specific fit and
# under the single global one. Written by reproducibility/compute/make_recon_cache.py; the global arm is
# stored as varp["W_all"] inside the same canonical fits, so neither arm needs a refit.
RECON_CACHE = os.path.join(paths.CACHE, "_recon_cache.json")
# Panel g: split-half stability of the unconstrained W, four datasets by neighbor fraction.
SPLITHALF = os.path.join(paths.IDENTIFIABILITY, "multi.json")
SPLITHALF_SYSTEM = {"hematopoiesis": "hematopoiesis", "pancreas": "pancreas",
                    "murine_NC": "neural crest", "human_limb": "myogenesis"}

# 7 datasets across 4 biological systems (hematopoiesis is sampled 3 ways -> also a robustness point).
# Ordered so datasets of the SAME biological system are adjacent in every panel (hematopoiesis x3,
# pancreas, neural crest x2, myogenesis); previously the two neural-crest sets were split by myogenesis.
DATASETS = ["paul15", "paul15_coarse", "dynamo_hematopoiesis", "pancreas", "murine_nc", "schwann", "human_limb"]
SYSTEM = {"pancreas": "pancreas", "paul15": "hematopoiesis", "paul15_coarse": "hematopoiesis",
          "dynamo_hematopoiesis": "hematopoiesis", "murine_nc": "neural crest", "schwann": "neural crest",
          "human_limb": "myogenesis"}
SYSCOLOR = {"hematopoiesis": PALETTE.get("vermillion", "#D55E00"), "pancreas": PALETTE.get("blue", "#0072B2"),
            "neural crest": PALETTE.get("green", "#009E73"), "myogenesis": PALETTE.get("orange", "#E69F00")}
LABEL = {"pancreas": "pancreas", "paul15": "paul15", "paul15_coarse": "paul15\ncoarse",
         "dynamo_hematopoiesis": "dynamo\nhema", "murine_nc": "murine\nNC", "human_limb": "human\nlimb",
         "schwann": "schwann"}

# Progenitor / root cell types per dataset; everything else present is treated as terminal / committed.
from config import PROGENITORS       # single source of truth; see the note beside it in config.py


def _ck(ds):
    from config import DATASETS as CFG
    return CFG[ds]["cluster_key"]


def _curated_tfs(ds):
    # Panel b ranks these against every gene, so an empty list is not an empty panel, it is a
    # nan the reader cannot tell from a dataset where the curated TFs simply rank badly.
    from _perturb_dynamics_compute import TFS_BY_DATASET
    return list(guards.require_dataset_entry(
        TFS_BY_DATASET, ds, "TFS_BY_DATASET in compute/_perturb_dynamics_compute.py",
        "the curated transcription factors panel b ranks"))


def _to_dense(a, key):
    L = a.layers[key]
    return np.asarray(L.todense()) if hasattr(L, "todense") else np.asarray(L, float)


def _z(v):
    v = np.asarray(v, float); s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s else 1.0)


def _gini(x):
    x = np.sort(np.asarray(x, float)); x = x[x >= 0]
    if x.sum() == 0:
        return 0.0
    n = x.size; return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def collect(cache: str | None = None):
    """One row per dataset with the five scalars/vectors each panel needs.

    Pass a path to reuse a cached read: the seven report AnnData files are several GB in
    total, so re-reading them for every layout tweak is the slow part. The default (poster)
    figure passes nothing and always recomputes, exactly as before.
    """
    if cache and os.path.exists(cache):
        with open(cache) as fh:
            rows = json.load(fh)
        if all(d in rows for d in DATASETS):
            print(f"[cache] reusing {cache}", flush=True)
            return rows
    rows = {}
    for ds in DATASETS:
        a = ad.read_h5ad(f"{paths.REPORTS}/{ds}/data/adata_analyzed.h5ad")
        ck = _ck(ds); cl = a.obs[ck].astype(str).values
        used = a.var["scHopfield_used"].values if "scHopfield_used" in a.var else np.ones(a.n_vars, bool)
        # a: interaction-energy basin DEPTH per cell (within-dataset z-score), split progenitor vs terminal.
        # Energy depth is the clean cross-dataset commitment signal; the velocity/settling term is confounded
        # by quiescent stem/progenitor compartments (also low-velocity), so it is deliberately NOT used here.
        eint = a.obs["energy_interaction"].values.astype(float)
        att = _z(-eint)
        # An uncurated dataset would mark every cell terminal and draw the progenitor arm as nan.
        prog = np.isin(cl, list(guards.require_dataset_entry(
            PROGENITORS, ds, "PROGENITORS in config.py",
            "panel a, the progenitor versus terminal split")))
        att_prog = float(np.nanmean(att[prog])) if prog.any() else np.nan
        att_term = float(np.nanmean(att[~prog])) if (~prog).any() else np.nan
        # b: rank-percentile of curated TFs among all genes (best of the lineage-pair driver CSVs)
        pcts = []
        for tf in _curated_tfs(ds):
            best = np.nan
            for k in (1, 2, 3):
                p = f"{paths.REPORTS}/{ds}/data/driver_scores_{k}.csv"
                if os.path.exists(p):
                    df = pd.read_csv(p, index_col=0)
                    if tf in df.index:
                        n = len(df)
                        r = min(df.loc[tf, "rank_A"], df.loc[tf, "rank_B"])   # best rank across the two arms
                        best = np.nanmax([best, 100.0 * (1 - (r - 1) / max(n - 1, 1))])
            if np.isfinite(best):
                pcts.append(best)
        # c: identifiability -- effective dimensionality (participation ratio) of the expression manifold,
        #    as a fraction of the used genes. Small everywhere => the [W|I] inverse problem is heavily
        #    under-determined in every system, which is what motivates the transcription-factor scaffold.
        Xc = _to_dense(a, "Ms")[:, used]
        Xc = Xc - Xc.mean(0)
        s = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
        lam = s ** 2
        pr = float((lam.sum() ** 2) / (lam ** 2).sum()) if lam.sum() > 0 else np.nan
        fit_corr = pr / int(used.sum())            # effective-dim fraction (kept var name for the row dict)
        # d: regulatory concentration AMONG regulators -- share of total out-strength held by the top-10 TFs
        #    (Gini of all genes would be trivially ~1 under only_TFs, since non-TFs have zero out-strength).
        W = np.abs(np.asarray(a.varp["W_all"].todense() if hasattr(a.varp["W_all"], "todense") else a.varp["W_all"]))
        outs = np.sort(W.sum(0)[used])[::-1]
        outs = outs[outs > 0]
        gini = float(outs[:10].sum() / outs.sum()) if outs.sum() > 0 else np.nan
        # e: bimodal fraction + lineage-TF enrichment
        mix = a.var["sigmoid_mix"].values.astype(float)
        fl = used & np.isfinite(mix) & (mix < 1 - 1e-6)
        frac_all = float(fl[used].mean())
        lin = [g for g in _curated_tfs(ds) if g in a.var_names and used[a.var_names.get_loc(g)]]
        frac_lin = float(np.mean([fl[a.var_names.get_loc(g)] for g in lin])) if lin else np.nan
        rows[ds] = dict(att_prog=att_prog, att_term=att_term, pcts=pcts, fit_corr=fit_corr,
                        gini=gini, frac_all=frac_all, frac_lin=frac_lin)
        print(f"[{ds}] attr prog {att_prog:.2f}->term {att_term:.2f} | reg pct med "
              f"{np.median(pcts) if pcts else float('nan'):.0f} | fitcorr {fit_corr:.2f} | gini {gini:.2f} | "
              f"bimod {frac_all:.2f} (lin {frac_lin:.2f})", flush=True)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def main():
    R = collect()
    order = DATASETS
    xs = np.arange(len(order))
    cols = [SYSCOLOR[SYSTEM[d]] for d in order]
    use_style(9)
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 8.2))
    fig.text(0.02, 0.985, "scHopfield readouts reproduce across developmental systems",
             ha="left", va="top", fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, top=0.90, bottom=0.10, left=0.07, right=0.98, hspace=0.55, wspace=0.34)

    def label(ax, lt):
        bb = ax.get_position(); fig.text(bb.x0 - 0.035, bb.y1 + 0.028, lt, fontweight="bold", fontsize=13,
                                         va="bottom", ha="left")

    # a: stability slopegraph (progenitor -> terminal)
    ax = fig.add_subplot(gs[0, 0])
    for i, d in enumerate(order):
        ax.plot([0, 1], [R[d]["att_prog"], R[d]["att_term"]], "-o", color=cols[i], lw=1.4, ms=4, alpha=0.9)
    ax.axhline(0, color="0.6", lw=0.6, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["progenitor", "terminal"], fontsize=8)
    ax.set_xlim(-0.25, 1.25); ax.set_ylabel("energy-basin depth (z)", fontsize=8.5)
    ax.set_title("terminals sit in deeper energy basins", fontsize=8.3); ax.tick_params(labelsize=7)
    label(ax, "a")

    # b: regulator recovery (percentile of curated TFs), strip per dataset
    ax = fig.add_subplot(gs[0, 1])
    for i, d in enumerate(order):
        p = R[d]["pcts"]
        ax.scatter(np.full(len(p), i) + np.random.uniform(-0.12, 0.12, len(p)), p, s=14, c=[cols[i]],
                   alpha=0.8, linewidths=0)
        if p:
            ax.plot([i - 0.22, i + 0.22], [np.median(p)] * 2, color="k", lw=1.4)
    ax.axhline(90, color="0.6", lw=0.6, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[d] for d in order], fontsize=6.2)
    ax.set_ylabel("driver-score percentile", fontsize=8.5); ax.set_ylim(0, 101)
    ax.set_title("known-regulator recovery", fontsize=9); ax.tick_params(axis="y", labelsize=7)
    label(ax, "b")

    # c: GRN fit quality (median reconstruction correlation)
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(xs, [100 * R[d]["fit_corr"] for d in order], color=cols, edgecolor="0.3", linewidth=0.4, width=0.7)
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[d] for d in order], fontsize=6.2)
    ax.set_ylabel("effective dim (% of genes)", fontsize=8.5)
    ax.set_title("expression low-rank (GRN under-determined)", fontsize=8.3); ax.tick_params(axis="y", labelsize=7)
    label(ax, "c")

    # d: regulatory concentration (Gini of TF out-strength)
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(xs, [R[d]["gini"] for d in order], color=cols, edgecolor="0.3", linewidth=0.4, width=0.7)
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[d] for d in order], fontsize=6.2)
    ax.set_ylabel("top-10 TF out-strength share", fontsize=8.5); ax.set_ylim(0, 1)
    ax.set_title("regulation concentrates in hubs", fontsize=9); ax.tick_params(axis="y", labelsize=7)
    label(ax, "d")

    # e: bimodality fraction + lineage-TF enrichment
    ax = fig.add_subplot(gs[1, 1])
    w = 0.38
    ax.bar(xs - w / 2, [R[d]["frac_all"] for d in order], w, color="0.7", edgecolor="0.3", linewidth=0.4,
           label="all genes")
    ax.bar(xs + w / 2, [R[d]["frac_lin"] for d in order], w, color=cols, edgecolor="0.3", linewidth=0.4,
           label="lineage TFs")
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[d] for d in order], fontsize=6.2)
    ax.set_ylabel("fraction bimodal", fontsize=8.5); ax.set_ylim(0, 1)
    ax.set_title("activation bimodality", fontsize=9); ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    label(ax, "e")

    # f: system legend
    ax = fig.add_subplot(gs[1, 2]); ax.set_axis_off()
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=c, lw=0, ms=9, label=s) for s, c in SYSCOLOR.items()]
    ax.legend(handles=handles, title="biological system", fontsize=8, title_fontsize=8.5, frameon=False,
              loc="center")
    ax.text(0.5, 0.10, "7 datasets  /  4 systems\n(hematopoiesis sampled 3 ways)", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.8, color="0.4")

    save(fig, f"{OUT}/cross-dataset-generality", formats=("pdf", "png"))
    print(f"wrote {OUT}/cross-dataset-generality.pdf + .png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# Submission variant: the same five panels on a journal page.
# ------------------------------------------------------------------------------------------

def main_submission(height_mm: float = 140.0, out: str | None = None):
    """Extended Data Fig. 3 at journal page size: 180 mm wide, one page, nothing below 5 pt.

    The poster layout spreads a-e over a 2 x 3 grid of 100 mm panels. Scaled to 180 mm that
    lands the tick labels at about 3 pt, so the panels are redistributed instead. Four of the
    five panels (b-e) share one x axis, the seven datasets, so they are stacked into a single
    column and share one set of dataset labels: the height that buys back is what keeps every
    panel, and reading one dataset down through four readouts is easier than hunting for it in
    four separate grids. Panel a has its own axis (progenitor vs terminal) and keeps the left
    column, with the system key under it in the space the sixth grid cell used to waste.

    No panel is dropped and no color changes; only the geometry and the type sizes differ.
    """
    from submission_style import figure_for, panel_letter, save as save_spec

    R = collect(cache=CACHE)
    order = DATASETS
    xs = np.arange(len(order))
    cols = [SYSCOLOR[SYSTEM[d]] for d in order]
    # Where one biological system gives way to the next along the shared x axis. Drawn as a
    # light grey rule in every stacked panel, so the grouping survives without a color.
    bounds = [i + 0.5 for i in range(len(order) - 1) if SYSTEM[order[i]] != SYSTEM[order[i + 1]]]

    TITLE, LAB, TICK, KEY, FINE = 7.5, 7.0, 6.0, 6.0, 5.5
    use_style(7)                                   # font family, palette, spines
    fig = figure_for("double", height_mm=height_mm)   # sets the submission type sizes
    os.makedirs(SPEC_OUT, exist_ok=True)

    # Left column: panel a, then g, then the system key. Right column: b-f sharing one x axis.
    # g cannot join the stack: it covers four datasets against neighbor fraction, not the seven
    # datasets the stack is indexed by.
    gs_a = fig.add_gridspec(1, 1, left=0.085, right=0.325, top=0.960, bottom=0.715)
    gs_g = fig.add_gridspec(1, 1, left=0.085, right=0.325, top=0.630, bottom=0.385)
    gs_k = fig.add_gridspec(1, 1, left=0.055, right=0.345, top=0.320, bottom=0.030)
    gs_r = fig.add_gridspec(5, 1, left=0.455, right=0.995, top=0.960, bottom=0.075, hspace=0.32)

    def _stack_axis(row, title, letter):
        ax = fig.add_subplot(gs_r[row, 0])
        for b in bounds:
            ax.axvline(b, color="0.88", lw=0.5, ls=(0, (1, 1.5)), zorder=0)
        ax.set_xlim(-0.65, len(order) - 0.35)
        ax.set_xticks(xs)
        ax.set_title(title, fontsize=TITLE, loc="left", pad=2.5)
        ax.tick_params(axis="y", labelsize=TICK, pad=1.5)
        ax.tick_params(axis="x", length=0)
        panel_letter(ax, letter, dx=-0.055, dy=1.02)
        return ax

    # a: energy-basin depth, progenitor -> terminal, one line per dataset
    ax = fig.add_subplot(gs_a[0, 0])
    for i, d in enumerate(order):
        ax.plot([0, 1], [R[d]["att_prog"], R[d]["att_term"]], "-o", color=cols[i], lw=1.1, ms=2.6,
                alpha=0.9, clip_on=False)
    ax.axhline(0, color="0.6", lw=0.5, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["progenitor", "terminal"], fontsize=TICK)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("energy-basin depth (z)", fontsize=LAB, labelpad=2)
    # One line, so this letter and title sit on the same baseline as b-e; it runs a few
    # millimetres into the gutter, which is empty.
    ax.set_title("terminals sit in deeper energy basins", fontsize=TITLE, loc="left", pad=2.5)
    ax.tick_params(axis="y", labelsize=TICK, pad=1.5); ax.tick_params(axis="x", length=0, pad=2)
    panel_letter(ax, "a", dx=-0.10, dy=1.02)

    # b: known-regulator recovery, curated lineage TFs as driver-score percentiles
    ax = _stack_axis(0, "known-regulator recovery", "b")
    rng = np.random.default_rng(0)             # fixed jitter, so the variant is reproducible
    for i, d in enumerate(order):
        p = R[d]["pcts"]
        ax.scatter(np.full(len(p), i) + rng.uniform(-0.13, 0.13, len(p)), p, s=5, c=[cols[i]],
                   alpha=0.85, linewidths=0, zorder=3)
        if p:
            ax.plot([i - 0.26, i + 0.26], [np.median(p)] * 2, color="k", lw=1.1, zorder=4)
    ax.axhline(90, color="0.6", lw=0.5, ls="--", zorder=1)
    ax.set_ylim(0, 108); ax.set_yticks([0, 50, 100])
    ax.set_ylabel("driver-score\npercentile", fontsize=LAB, labelpad=2)
    ax.set_xticklabels([])

    # c: identifiability, effective dimensionality as a percent of the modeled genes
    ax = _stack_axis(1, "expression is low rank (the GRN fit is under-determined)", "c")
    vals = [100 * R[d]["fit_corr"] for d in order]
    ax.bar(xs, vals, color=cols, edgecolor="0.3", linewidth=0.4, width=0.66)
    # The point of the panel is how small these are, which leaves the four smallest bars hard
    # to tell apart, so each one carries its value.
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=FINE, color="0.35")
    ax.set_ylim(0, 2.05); ax.set_yticks([0, 1, 2])
    ax.set_ylabel("effective dim\n(% of genes)", fontsize=LAB, labelpad=2)
    ax.set_xticklabels([])

    # d: regulatory concentration, share of out-strength held by the top ten TFs
    ax = _stack_axis(2, "regulation concentrates in hubs", "d")
    ax.bar(xs, [R[d]["gini"] for d in order], color=cols, edgecolor="0.3", linewidth=0.4, width=0.66)
    ax.set_ylim(0, 0.58); ax.set_yticks([0, 0.25, 0.5])
    ax.set_ylabel("top-10 TF\nout-strength share", fontsize=LAB, labelpad=2)
    ax.set_xticklabels([])

    # e: activation bimodality, all genes vs the curated lineage TFs
    ax = _stack_axis(3, "activation bimodality", "e")
    w = 0.33
    ax.bar(xs - w / 2, [R[d]["frac_all"] for d in order], w, color="0.7", edgecolor="0.3",
           linewidth=0.4, label="all genes")
    ax.bar(xs + w / 2, [R[d]["frac_lin"] for d in order], w, color=cols, edgecolor="0.3",
           linewidth=0.4, label="lineage TFs")
    ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("fraction\nbimodal", fontsize=LAB, labelpad=2)
    ax.set_xticklabels([])
    ax.legend(fontsize=FINE, frameon=False, loc="upper left", ncol=2, handlelength=1.0,
              handletextpad=0.4, columnspacing=1.0, borderaxespad=0.15)

    # f: velocity reconstruction, the single global interaction matrix against the
    # cell-type-specific ones. Both arms are scored on the cells they were fitted to, and the
    # cell-type-specific model carries one matrix per type, so it has many times more parameters:
    # this is a contrast in FIT quality, not a test of prediction. The title says "fit" rather
    # than "predict" for that reason and the Inventory caption states it in full. Do not retitle
    # this to imply held-out performance.
    with open(RECON_CACHE) as fh:
        RC = json.load(fh)
    ax = _stack_axis(4, "cell-type-specific systems fit the velocity better", "f")
    for i, d in enumerate(order):
        lo, hi = RC[d]["cos_global"], RC[d]["cos_celltype"]
        ax.plot([i, i], [lo, hi], color=cols[i], lw=1.0, alpha=0.55, zorder=2, solid_capstyle="round")
        ax.scatter([i], [lo], s=11, facecolors="white", edgecolors="0.45", linewidths=0.7, zorder=3)
        ax.scatter([i], [hi], s=13, c=[cols[i]], linewidths=0, zorder=4)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("median per-cell\nvelocity cosine", fontsize=LAB, labelpad=2)
    ax.set_xticklabels([LABEL[d] for d in order], fontsize=TICK)
    h_gl = plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="0.45", mew=0.7, ms=3.2)
    h_ct = plt.Line2D([], [], marker="o", ls="none", color="0.35", ms=3.4)
    ax.legend([h_gl, h_ct], ["one global system", "cell-type-specific"], fontsize=FINE,
              frameon=False, loc="lower left", ncol=2, handlelength=1.0, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.15)

    # g: split-half stability of the UNCONSTRAINED interaction matrix. Fit the same sample twice
    # on disjoint halves and correlate the two off-diagonals. The axis runs to 1.0, which is what
    # two identical halves would give, because the claim is how far below that these sit: velocity
    # alone does not determine W, which is what the scaffold prior is for. Four datasets only, one
    # per biological system.
    with open(SPLITHALF) as fh:
        SH = json.load(fh)
    axg = fig.add_subplot(gs_g[0, 0])
    for name, sysname in SPLITHALF_SYSTEM.items():
        bf = SH[name]["by_frac"]
        fr = sorted(bf, key=float)
        axg.plot([float(f) for f in fr], [bf[f]["splithalf_W"] for f in fr], "-o",
                 color=SYSCOLOR[sysname], lw=1.1, ms=2.6, alpha=0.9, clip_on=False)
    axg.axhline(1.0, color="0.6", lw=0.5, ls="--")
    axg.text(0.40, 1.0, "identical halves", fontsize=FINE, color="0.45", va="bottom", ha="right")
    axg.axhline(0.0, color="0.85", lw=0.5)
    axg.set_ylim(-0.08, 1.12); axg.set_yticks([0, 0.5, 1])
    axg.set_xticks([0.0, 0.1, 0.2, 0.4])
    axg.set_xlabel("neighboring-cell fraction", fontsize=LAB, labelpad=2)
    axg.set_ylabel("split-half correlation of $W$", fontsize=LAB, labelpad=2)
    axg.set_title("velocity alone does not determine $W$", fontsize=TITLE, loc="left", pad=2.5)
    axg.tick_params(axis="both", labelsize=TICK, pad=1.5)
    panel_letter(axg, "g", dx=-0.10, dy=1.02)

    # System key, under panel a: the four systems, and which datasets belong to each. Laid out
    # on an explicit row grid (y is one text line) so the entries cannot drift into each other.
    axk = fig.add_subplot(gs_k[0, 0]); axk.set_axis_off()
    members: dict[str, list[str]] = {}
    for d in order:
        members.setdefault(SYSTEM[d], []).append(LABEL[d].replace("\n", " "))
    rows = 12.0
    axk.set_xlim(0, 1); axk.set_ylim(rows, 0.0)     # inverted: row 0 is the top line
    axk.text(0.0, 0.2, "biological system", fontsize=KEY, fontweight="bold", va="center", ha="left")
    axk.text(0.0, 1.05, "7 datasets, 4 systems", fontsize=FINE, color="0.45", va="center", ha="left")
    y = 2.5
    for s in ["hematopoiesis", "pancreas", "neural crest", "myogenesis"]:
        axk.plot([0.04], [y], marker="o", ms=3.2, color=SYSCOLOR[s], clip_on=False)
        axk.text(0.12, y, s, fontsize=KEY, va="center", ha="left")
        axk.text(0.12, y + 0.85, ", ".join(members[s]), fontsize=FINE, color="0.45",
                 va="center", ha="left")
        y += 2.05

    path = out or f"{SPEC_OUT}/ExtendedDataFig3.pdf"
    save_spec(fig, path)
    print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", action="store_true",
                    help="render the journal-page variant into reproducibility/figures/submission/")
    ap.add_argument("--height-mm", type=float, default=140.0, help="submission canvas height")
    args = ap.parse_args()
    if args.submission:
        main_submission(height_mm=args.height_mm)
    else:
        main()
