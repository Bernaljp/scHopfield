"""Composite figure: scHopfield on the dyngen ground-truth GRN benchmark.

Three backbones (linear, bifurcating, cycle). Panels:
  a  simulated cells (UMAP), colored by (normalized) simulation time  [one shared colorbar]
  b  ground-truth signed interaction matrices W_true  [one shared colorbar]
  c  edge detection: AUROC, AUPRC and sign accuracy (three plots in a row)
  d  "the scaffold makes the difference": AUROC across no/TF/true scaffold, recovered W at each
     stage (linear) as insets, and the actual signed matrix W_true (bottom right)
  e  soft-mode scaffold sweep: AUROC (vs the soft penalty, with the hard-mask level dotted) and
     the off-scaffold weight fraction
  f  Hopfield energy vs simulation time
  g  local stability (leading Jacobian eigenvalue) vs simulation time

Refit artifacts cached by `reproducibility/compute/_dyngen_compute.py` -> reproducibility/data/dyngen/fig_fits.npz

Run:  python reproducibility/make_dyngen_benchmark.py
      python reproducibility/make_dyngen_benchmark.py --submission

The default run writes the poster-sized working figure used by the per-dataset reports and is
unchanged. `--submission` re-lays the same panels, with the same letters and the same colors,
onto one journal page (180 mm wide, under 247 mm tall) at 6 pt or larger
type, and writes it to reproducibility/figures/submission/Figure3.pdf.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
import anndata as ad                                    # noqa: E402

ROOT = paths.DYNGEN
OUT = paths.FIGURES
NAME = "dyngen-benchmark"
BB = ["linear", "bifurcating", "cycle"]
NICE = {"linear": "linear", "bifurcating": "bifurcating", "cycle": "cycle"}

MCOL = {"no-scaffold": "#999999", "tf-scaffold": PALETTE["blue"],
        "true-topology": PALETTE["green"], "genie3": PALETTE["vermillion"]}
MLAB = {"no-scaffold": "no scaffold", "tf-scaffold": "TF scaffold",
        "true-topology": "true topology", "genie3": "GENIE3"}
BCOL = {"linear": PALETTE["sky"], "bifurcating": PALETTE["orange"], "cycle": PALETTE["purple"]}
REG_GRID = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]

SUM = {d["dataset"]: d for d in json.load(open(f"{ROOT}/benchmark_summary.json"))}
FITS = np.load(f"{ROOT}/fig_fits.npz")
SS = json.load(open(f"{ROOT}/scaffold_sweep.json"))   # specific-edge sweep (only_TFs=True)
SS_REG_MIN, SS_REG_MAX = 5e-3, 20.0                   # panel d plots the 5e-3..20 window

# --------------------------------------------------------------------------- #
# Two renderings of the same panels. SUB is False for the working (poster-sized) figure the
# per-dataset reports embed, and True for the journal-page figure. Every size the panels draw
# with is looked up here, so the default values below reproduce the working figure exactly and
# only the submission column is re-tuned.
# --------------------------------------------------------------------------- #
SUB = False
FS = {
    "t_ab": 8.0, "lab_ab": 7.0, "umap_s": 6.0, "tpad": 6.0,      # rows a, b
    "cb_lab": 7.0, "cb_tick": 6.0,                               # shared colorbars
    "t_cd": 9.0, "xtick_cat": 6.5, "dot_ms": 2.4,                # rows c, d
    "leg": 6.0, "leg_small": 5.6, "inset_t": 5.5, "lab_d": 7.5,
    "ms_d": 4.0, "lw_d": 1.4, "ms_s": 3.2, "lw_s": 1.4, "ms_n": 3.0, "lw_n": 1.1,
    "t_ef": 8.5, "lab_ef": 7.5, "tick_ef": 6.0, "off_ef": 5.5, "sc_ef": 7.0,   # rows e, f
    "letter": 13.0, "header": 13.0,
}
FS_SUBMISSION = {
    "t_ab": 7.0, "lab_ab": 6.5, "umap_s": 1.6, "tpad": 2.5,
    "cb_lab": 6.5, "cb_tick": 6.0,
    "t_cd": 7.0, "xtick_cat": 6.0, "dot_ms": 1.8,
    "leg": 6.0, "leg_small": 6.0, "inset_t": 6.0, "lab_d": 6.5,
    "ms_d": 2.6, "lw_d": 1.0, "ms_s": 2.2, "lw_s": 1.0, "ms_n": 2.2, "lw_n": 0.9,
    "t_ef": 7.0, "lab_ef": 6.5, "tick_ef": 6.0, "off_ef": 6.0, "sc_ef": 2.0,
    "letter": 8.0, "header": 8.0,
}


def metric(bb, method, m):
    d = SUM[bb]
    if method == "genie3":
        return d["genie3"].get(m, np.nan)
    if method == "tf-scaffold":
        return d["schopfield_tf_scaffold"]["high"].get(m, np.nan)
    key = {"no-scaffold": "schopfield_no_scaffold", "true-topology": "schopfield_true_scaffold"}[method]
    return d[key].get(m, np.nan)


def heat_W(ax, W, pct=99.5):
    v = np.percentile(np.abs(W), pct) or 1.0
    ax.imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------------------- #
# a: UMAPs (square, frameless, normalized time); b: W_true (full box)
# --------------------------------------------------------------------------- #
def draw_umap(ax, bb):
    a = ad.read_h5ad(f"{ROOT}/{bb}/adata.h5ad")
    u = a.obsm["X_umap"]; t = a.obs["sim_time"].values.astype(float)
    tn = (t - t.min()) / (t.max() - t.min() + 1e-9)
    ax.scatter(u[:, 0], u[:, 1], c=tn, cmap="viridis", vmin=0, vmax=1, s=FS["umap_s"],
               linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_box_aspect(1)
    # On the page the cell count moves to the legend, which leaves the title readable at 7 pt.
    title = f"{NICE[bb]}, {a.n_vars} genes" if SUB else f"{NICE[bb]}  (n={a.n_obs}, {a.n_vars} genes)"
    ax.set_title(title, fontsize=FS["t_ab"], pad=FS["tpad"])


def draw_wtrue(ax, bb, vmax):
    W = np.load(f"{ROOT}/{bb}/W_true.npy")
    ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)
    ax.set_xlabel("regulator", fontsize=FS["lab_ab"], labelpad=1.5 if SUB else 4.0)
    ax.set_ylabel("target", fontsize=FS["lab_ab"], labelpad=1.5 if SUB else 4.0)
    # The backbone name is already the title of the UMAP directly above this column.
    title = f"{int((W != 0).sum())} edges" if SUB else f"{NICE[bb]}  ({int((W != 0).sum())} edges)"
    ax.set_title(title, fontsize=FS["t_ab"], pad=FS["tpad"])


# --------------------------------------------------------------------------- #
# c: three metric bar charts
# --------------------------------------------------------------------------- #
def draw_metric_bars(ax, mname, methods, title, ymax, chance=False):
    x = np.arange(len(methods))
    for i, m in enumerate(methods):
        vals = [metric(b, m, mname) for b in BB]
        ax.bar(i, np.nanmean(vals), 0.64, color=MCOL[m], alpha=0.85, edgecolor="none", zorder=1)
        ax.plot([i] * len(BB), vals, "o", color="0.15", ms=FS["dot_ms"], zorder=3)
    if chance:
        ax.axhline(0.5, ls="--", c="k", lw=0.7, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MLAB[m].replace(" ", "\n") for m in methods], fontsize=FS["xtick_cat"])
    ax.tick_params(axis="x", pad=1.0 if SUB else 3.5)
    ax.set_ylim(0, ymax); ax.set_title(title, fontsize=FS["t_cd"], pad=FS["tpad"])


# --------------------------------------------------------------------------- #
# d: scaffold effect + recovered-W insets + actual W_true
# --------------------------------------------------------------------------- #
def draw_scaffold_effect(ax):
    stages = ["no-scaffold", "tf-scaffold", "true-topology"]; xs = [0, 1, 2]
    # The insets sit in the head-room above the curves; on the page they need a larger share of
    # the panel to stay legible, so the axis is taller there and the inset boxes are bigger.
    ytop = 1.7 if SUB else 1.4
    ins_y, ins_w, ins_h = (0.62, 0.21, 0.36) if SUB else (0.72, 0.18, 0.24)
    for b in BB:
        ys = [metric(b, s, "auroc") for s in stages]
        ax.plot(xs, ys, "-o", color=BCOL[b], ms=FS["ms_d"], lw=FS["lw_d"], label=NICE[b], zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(["no\nscaffold", "TF\nscaffold", "true\ntopology"], fontsize=FS["xtick_cat"])
    ax.tick_params(axis="x", pad=1.0 if SUB else 3.5)
    ax.set_ylabel("edge-detection AUROC", labelpad=1.5 if SUB else 4.0)
    ax.set_ylim(0, ytop); ax.set_xlim(-0.35, 2.35)
    if SUB:
        ax.set_yticks([0, 0.5, 1.0])   # the head-room is for the insets, not for AUROC above 1
    ax.legend(fontsize=FS["leg"], loc="lower left", framealpha=0.9, borderpad=0.3,
              handlelength=1.2, labelspacing=0.25 if SUB else 0.5)
    ax.set_title("The scaffold makes the difference", fontsize=FS["t_cd"], pad=FS["tpad"])
    insW = {"no-scaffold": FITS["lin_W_none"], "tf-scaffold": FITS["lin_W_tf"],
            "true-topology": FITS["lin_W_true"]}
    for s, xfi, xd in zip(stages, [0.02, 0.36, 0.70], xs):
        ia = ax.inset_axes([xfi, ins_y, ins_w, ins_h]); heat_W(ia, insW[s], pct=99.5)
        ax.annotate("", xy=(xd, metric("linear", s, "auroc")), xytext=(xd, ins_y * ytop),
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6), zorder=1)
    ia = ax.inset_axes([0.76 if SUB else 0.77, 0.05 if SUB else 0.07, ins_w, ins_h])
    heat_W(ia, np.load(f"{ROOT}/linear/W_true.npy"), pct=100)
    ia.set_title("actual $W$", fontsize=FS["inset_t"], pad=1)


# --------------------------------------------------------------------------- #
# e: soft-mode sweep (two plots)
# --------------------------------------------------------------------------- #
def _ss(b, prior, key):
    """reg values (>0, <= SS_REG_MAX) and metric for backbone b, prior in {true_edge, noisy}."""
    rows = [c for c in SS[b]["sweeps"][prior]
            if SS_REG_MIN * (1 - 1e-6) <= c["reg"] <= SS_REG_MAX * (1 + 1e-6)]
    return [c["reg"] for c in rows], [c[key] for c in rows]


def draw_ss_auroc(ax):
    for b in BB:
        rt, vt = _ss(b, "true_edge", "auroc"); rn, vn = _ss(b, "noisy", "auroc")
        ax.plot(rt, vt, "-o", color=BCOL[b], ms=FS["ms_s"], lw=FS["lw_s"], label=NICE[b])
        ax.plot(rn, vn, "--s", color=BCOL[b], ms=FS["ms_n"], mfc="none", lw=FS["lw_n"])
    ax.axhline(0.5, ls="--", c="k", lw=0.7, alpha=0.4)
    ax.set_xscale("log"); ax.set_xlim(5e-3, 20)
    # On the page the backbone key lives in the panel to the left and "only TFs" in the legend.
    ax.set_xlabel("scaffold regularization" if SUB else "scaffold regularization (only $TF$s)",
                  fontsize=FS["lab_d"], labelpad=1.5 if SUB else 4.0)
    ax.set_ylabel("edge-detection AUROC", labelpad=1.5 if SUB else 4.0); ax.set_ylim(0.5, 1.01)
    if not SUB:
        leg = ax.legend(fontsize=FS["leg"], loc="lower right", framealpha=0.9, borderpad=0.3,
                        handlelength=1.4)
        ax.add_artist(leg)
    style = [Line2D([0], [0], color="0.35", ls="-", marker="o", ms=3, lw=1.4),
             Line2D([0], [0], color="0.35", ls="--", marker="s", ms=3, mfc="none", lw=1.1)]
    ax.legend(style, ["true-edge prior", "noisy prior"], fontsize=FS["leg_small"], loc="lower right"
              if SUB else "upper left", framealpha=0.9, borderpad=0.3, handlelength=1.9,
              labelspacing=0.25 if SUB else 0.5)
    ax.set_title("AUROC vs regularization" if SUB else "Edge recovery vs regularization",
                 fontsize=FS["t_cd"], pad=FS["tpad"])


def draw_ss_frac(ax):
    for b in BB:
        rt, vt = _ss(b, "true_edge", "offscaffold_frac"); rn, vn = _ss(b, "noisy", "offscaffold_frac")
        ax.plot(rt, vt, "-o", color=BCOL[b], ms=FS["ms_s"], lw=FS["lw_s"])
        ax.plot(rn, vn, "--s", color=BCOL[b], ms=FS["ms_n"], mfc="none", lw=FS["lw_n"])
    ax.set_xscale("log"); ax.set_xlim(5e-3, 20)
    ax.set_xlabel("scaffold regularization" if SUB else "scaffold regularization (only $TF$s)",
                  fontsize=FS["lab_d"], labelpad=1.5 if SUB else 4.0)
    ax.set_ylabel("off-scaffold $|W|$ fraction" if SUB else "|W| fraction off-scaffold",
                  labelpad=1.5 if SUB else 4.0)
    ax.set_ylim(0, 1.0)
    ax.set_title("Off-scaffold weight", fontsize=FS["t_cd"], pad=FS["tpad"])


# --------------------------------------------------------------------------- #
# f, g
# --------------------------------------------------------------------------- #
def draw_energy(ax, bb, xlabel=True, title=True):
    t, E, _ = FITS[f"traj_{bb}"]
    ax.scatter(t, E, s=FS["sc_ef"], alpha=0.55, color=BCOL[bb], linewidths=0)
    if xlabel:
        ax.set_xlabel("sim. time", fontsize=FS["lab_ef"], labelpad=1.5 if SUB else 4.0)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("total energy", fontsize=FS["lab_ef"], labelpad=1.5 if SUB else 4.0)
    if title:
        ax.set_title(NICE[bb], fontsize=FS["t_ef"], pad=FS["tpad"])
    ax.tick_params(labelsize=FS["tick_ef"])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(FS["off_ef"])


def draw_eig(ax, bb, xlabel=True, title=True):
    t, _, lead = FITS[f"traj_{bb}"]
    ax.scatter(t, lead, s=FS["sc_ef"], alpha=0.55, color=BCOL[bb], linewidths=0)
    ax.axhline(0, ls="--", c="k", lw=0.7, alpha=0.5)
    if xlabel:
        ax.set_xlabel("sim. time", fontsize=FS["lab_ef"], labelpad=1.5 if SUB else 4.0)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    # A 25 mm rotated label does not fit a 22 mm tall panel, so the page version carries the
    # standard symbol and the caption names it in full.
    ax.set_ylabel(r"leading Re $\lambda$" if SUB else "leading Jacobian eig (Re)",
                  fontsize=FS["lab_ef"], labelpad=1.5 if SUB else 4.0)
    if title:
        ax.set_title(NICE[bb], fontsize=FS["t_ef"], pad=FS["tpad"])
    ax.tick_params(labelsize=FS["tick_ef"])


# --------------------------------------------------------------------------- #
# Submission rendering: the same panels on one journal page
# --------------------------------------------------------------------------- #
SUB_OUT = os.path.join(paths.FIGURES_SPEC, "Figure3.pdf")

# Every position below is in millimeters on the 180 mm page, measured from the top left, so the
# page budget is readable in one place. Rows a, b, c, e and f share one three-column grid, one
# column per backbone, so a reader tracks a backbone straight down the page.
L = dict(
    height=220.0,                              # leaves about 27 mm of the page for the legend
    col_x=[12.0, 67.0, 122.0], col_w=48.0,     # wide panels (rows c, e, f)
    sq=29.0,                                   # square panels (rows a, b), on the same left edges
    cb_x=154.5, cb_w=3.0,                      # shared colorbars for rows a and b
    y_a=8.0, y_b=43.5,                         # square rows
    y_c=84.5, h_c=23.5,                        # metric bars
    y_d=122.5, h_d=28.0,                       # scaffold priors and the regularization sweep
    # The two sweep plots end at 168 mm, inside the 171 mm right edge the gridspec rows
    # use (right=0.95 of the 180 mm canvas). At 40 mm the third one ran to 172 mm and its
    # tick labels pushed past every other panel, so the row read as overflowing.
    d_x=[12.0, 80.0, 130.0], d_w=[58.0, 38.0, 38.0],
    y_e=167.0, h_e=20.0,                       # energy vs simulation time
    y_f=191.5, h_f=20.0,                       # leading eigenvalue vs simulation time
)


def _ax_mm(fig, x, y, w, h, **kw):
    """One axes placed in millimeters, with y measured down from the top of the canvas."""
    fw, fh = (v * 25.4 for v in fig.get_size_inches())
    return fig.add_axes([x / fw, 1.0 - (y + h) / fh, w / fw, h / fh], **kw)


def _letter_mm(fig, letter, x, y):
    """A panel letter at (x, y) mm, sitting on the line above the top left of its panel."""
    fw, fh = (v * 25.4 for v in fig.get_size_inches())
    fig.text(x / fw, 1.0 - y / fh, letter, fontsize=FS["letter"], fontweight="bold",
             ha="left", va="bottom")


def main_submission():
    global SUB
    SUB = True
    FS.update(FS_SUBMISSION)
    from submission_style import figure_for, save as sub_save   # noqa: E402

    use_style(7)
    fig = figure_for("double", height_mm=L["height"])
    os.makedirs(os.path.dirname(SUB_OUT), exist_ok=True)

    # --- a: simulated cells; b: the networks they were generated from ---
    ax_a = [_ax_mm(fig, x, L["y_a"], L["sq"], L["sq"]) for x in L["col_x"]]
    ax_b = [_ax_mm(fig, x, L["y_b"], L["sq"], L["sq"]) for x in L["col_x"]]
    for ax, b in zip(ax_a, BB):
        draw_umap(ax, b)
    vmaxW = max(np.abs(np.load(f"{ROOT}/{b}/W_true.npy")).max() for b in BB)
    for ax, b in zip(ax_b, BB):
        draw_wtrue(ax, b, vmaxW)
    cba = fig.colorbar(mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(0, 1)),
                       cax=_ax_mm(fig, L["cb_x"], L["y_a"], L["cb_w"], L["sq"]))
    cba.set_label("sim. time (normalized)", fontsize=FS["cb_lab"], labelpad=2)
    cba.ax.tick_params(labelsize=FS["cb_tick"], pad=1.5)
    cba.set_ticks([0, 0.5, 1.0])
    cbb = fig.colorbar(mpl.cm.ScalarMappable(cmap="RdBu_r",
                                             norm=mpl.colors.Normalize(-vmaxW, vmaxW)),
                       cax=_ax_mm(fig, L["cb_x"], L["y_b"], L["cb_w"], L["sq"]))
    cbb.set_label("edge weight, repression to activation", fontsize=FS["cb_lab"], labelpad=2)
    cbb.ax.tick_params(labelsize=FS["cb_tick"], pad=1.5)
    cbb.set_ticks([-1, 0, 1])

    # --- c: edge detection ---
    ax_c = [_ax_mm(fig, x, L["y_c"], L["col_w"], L["h_c"]) for x in L["col_x"]]
    draw_metric_bars(ax_c[0], "auroc", ["no-scaffold", "tf-scaffold", "true-topology", "genie3"],
                     "AUROC", 1.02, chance=True)
    draw_metric_bars(ax_c[1], "auprc", ["no-scaffold", "tf-scaffold", "true-topology", "genie3"],
                     "AUPRC", 0.1)
    draw_metric_bars(ax_c[2], "sign_acc", ["no-scaffold", "tf-scaffold", "true-topology"],
                     "sign accuracy", 1.02)

    # --- d: the structural prior, then the regularization sweep ---
    ax_d = [_ax_mm(fig, x, L["y_d"], w, L["h_d"]) for x, w in zip(L["d_x"], L["d_w"])]
    draw_scaffold_effect(ax_d[0])
    draw_ss_auroc(ax_d[1])
    draw_ss_frac(ax_d[2])

    # --- e, f: energy and local stability against simulation time, one column per backbone ---
    ax_e = [_ax_mm(fig, x, L["y_e"], L["col_w"], L["h_e"]) for x in L["col_x"]]
    ax_f = [_ax_mm(fig, x, L["y_f"], L["col_w"], L["h_f"], sharex=a) for x, a in
            zip(L["col_x"], ax_e)]
    for ax, b in zip(ax_e, BB):
        draw_energy(ax, b, xlabel=False)
    for ax, b in zip(ax_f, BB):
        draw_eig(ax, b, title=False)

    for letter, (x, y) in [("a", (L["col_x"][0] - 8.5, L["y_a"] - 1.0)),
                           ("b", (L["col_x"][0] - 8.5, L["y_b"] - 1.0)),
                           ("c", (L["col_x"][0] - 8.5, L["y_c"] - 1.0)),
                           ("d", (L["d_x"][0] - 8.5, L["y_d"] - 1.0)),
                           ("e", (L["col_x"][0] - 8.5, L["y_e"] - 1.0)),
                           ("f", (L["col_x"][0] - 8.5, L["y_f"] - 1.0))]:
        _letter_mm(fig, letter, x, y)

    sub_save(fig, SUB_OUT)
    print(f"wrote {SUB_OUT}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    use_style(9)
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(13, 17.0))

    # --- section 1: a UMAPs + b W_true, one shared colorbar each ---
    gs1 = fig.add_gridspec(2, 4, top=0.95, bottom=0.71, left=0.06, right=0.95,
                           width_ratios=[1, 1, 1, 0.05], hspace=0.4, wspace=0.3)
    ax_a = [fig.add_subplot(gs1[0, i]) for i in range(3)]
    ax_b = [fig.add_subplot(gs1[1, i]) for i in range(3)]
    for ax, b in zip(ax_a, BB):
        draw_umap(ax, b)
    vmaxW = max(np.abs(np.load(f"{ROOT}/{b}/W_true.npy")).max() for b in BB)
    for ax, b in zip(ax_b, BB):
        draw_wtrue(ax, b, vmaxW)
    cba = fig.colorbar(mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(0, 1)),
                       cax=fig.add_subplot(gs1[0, 3]))
    cba.set_label("sim. time (normalized)", fontsize=7); cba.ax.tick_params(labelsize=6)
    cbb = fig.colorbar(mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(-vmaxW, vmaxW)),
                       cax=fig.add_subplot(gs1[1, 3]))
    cbb.set_label("edge weight", fontsize=7); cbb.ax.tick_params(labelsize=6)

    # --- section 2: c (3 metrics) then d,e1,e2 ---
    gs2 = fig.add_gridspec(2, 3, top=0.663, bottom=0.42, left=0.06, right=0.95,
                           hspace=0.62, wspace=0.34)
    ax_c = [fig.add_subplot(gs2[0, i]) for i in range(3)]
    draw_metric_bars(ax_c[0], "auroc", ["no-scaffold", "tf-scaffold", "true-topology", "genie3"],
                     "AUROC", 1.02, chance=True)
    draw_metric_bars(ax_c[1], "auprc", ["no-scaffold", "tf-scaffold", "true-topology", "genie3"],
                     "AUPRC", 0.1)
    draw_metric_bars(ax_c[2], "sign_acc", ["no-scaffold", "tf-scaffold", "true-topology"],
                     "sign accuracy", 1.02)
    ax_d = fig.add_subplot(gs2[1, 0]); draw_scaffold_effect(ax_d)
    ax_e1 = fig.add_subplot(gs2[1, 1]); draw_ss_auroc(ax_e1)
    ax_e2 = fig.add_subplot(gs2[1, 2]); draw_ss_frac(ax_e2)

    # --- section 3: f energy + g eig ---
    gs3 = fig.add_gridspec(2, 3, top=0.37, bottom=0.06, left=0.06, right=0.95,
                           hspace=0.45, wspace=0.32)
    ax_f = [fig.add_subplot(gs3[0, i]) for i in range(3)]
    ax_g = [fig.add_subplot(gs3[1, i]) for i in range(3)]
    for ax, b in zip(ax_f, BB):
        draw_energy(ax, b)
    for ax, b in zip(ax_g, BB):
        draw_eig(ax, b)

    # panel letters (top-left of each block, aligned in the left column)
    for ax, s in [(ax_a[0], "a"), (ax_b[0], "b"), (ax_c[0], "c"), (ax_d, "d"),
                  (ax_f[0], "e"), (ax_g[0], "f")]:
        bb = ax.get_position()
        fig.text(bb.x0 - 0.032, bb.y1 + 0.008, s, fontweight="bold", fontsize=13, va="bottom", ha="right")

    fig.text(0.05, 0.965, "Simulated datasets and ground-truth networks", ha="left",
             va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.05, 0.683, "GRN recovery", ha="left", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.05, 0.39, "Energy and local stability vs simulation time", ha="left",
             va="bottom", fontsize=13, fontweight="bold")

    save(fig, f"{OUT}/{NAME}", formats=("pdf", "png"))
    print(f"wrote {OUT}/{NAME}.pdf + .png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--submission", action="store_true",
                    help=f"render the journal-page version to {SUB_OUT}")
    args = ap.parse_args()
    main_submission() if args.submission else main()
