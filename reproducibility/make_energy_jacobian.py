"""Composite figure 3: energy landscape and Jacobian dynamical structure of a differentiation process
(default = pancreatic endocrinogenesis). Reads the analyzed report AnnData directly
(reports/<ds>/data/adata_analyzed.h5ad), whose obs already carries the per-cell energy decomposition
and Jacobian spectral statistics, so no new heavy compute is needed (panel f runs a light element-wise
Jacobian finite-difference on a copy).

  a  total-energy 3D landscape over the embedding (Methods 2.2, total energy).
  b  energy distribution by cell type, all four components (total, interaction, degradation, bias).
  c  Jacobian leading real eigenvalue and the number of positive eigenvalues, per cell type.
  d  Jacobian rotational magnitude (antisymmetric part), per cell type.
  e  Jacobian eigenvalue spectra (real vs imaginary), colored by cell type.
  f  element-wise Jacobian dynamics for selected developmental regulator/target pairs, per cell type.
  g  stability vs oscillation score per cell type (proposed definitions, refine later).
  h  attractor vs transient score per cell type (proposed definition, refine later).

Run:  python reproducibility/make_energy_jacobian.py [--dataset pancreas]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3d projection)

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
from paper_plot_style import use_style, save, PALETTE            # noqa: E402
import anndata as ad                                             # noqa: E402
import scHopfield as sch                                         # noqa: E402  (circuit rendering)
from sections import basis_of, get_colors, present_clusters      # noqa: E402

OUT = paths.FIGURES
PROCESS = {"pancreas": "endocrinogenesis", "paul15": "hematopoiesis",
           "murine_nc": "neural crest", "human_limb": "myogenesis", "schwann": "neural crest"}
ENERGY_COMPONENTS = [("energy_total", "total"), ("energy_interaction", "interaction"),
                     ("energy_degradation", "degradation"), ("energy_bias", "bias")]
# element-wise Jacobian pairs to feature (target, regulator): the alpha/beta antagonists and the
# endocrine specification cascade; edit as the biology dictates.
JAC_PAIRS = {"pancreas": [("Pax4", "Arx"), ("Arx", "Pax4"), ("Arx", "Nkx2-2"),
                          ("Neurod1", "Neurog3"), ("Pax6", "Neurod1"), ("Arx", "Pax6")]}

# Node layout order for the per-cell-type mini-networks (edges are the JAC_PAIRS above).
NET_GENES = {"pancreas": ["Neurog3", "Neurod1", "Pax6", "Arx", "Pax4", "Nkx2-2"]}

# Manual placement of each cell type's mini-network, as (right, up) offsets from the default bottom row,
# in units of one network size (equal in x and y on the page). Empty = the plain bottom row below the
# embedding; the user then supplies per-network offsets to move them onto the embedding near their cluster.
NET_OFFSETS = {
    "pancreas": {"Ductal": (-0.5, 1.25), "Ngn3 low EP": (0.25, 0.75), "Ngn3 high EP": (1.0, 0.0),
                 "Pre-endocrine": (2.5, 0.25), "Alpha": (3.5, 2.5), "Beta": (-1.0, 3.0),
                 "Delta": (1.0, 1.0), "Epsilon": (-3.5, 2.0)},
}
# Cell types whose connector arrow should point at the TITLE (top of the network) rather than the nearest
# edge; and small per-network nudges of the edge-arrow target, in units of the network size.
NET_TITLE_ARROW = {"pancreas": {"Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine", "Delta"}}
NET_ARROW_NUDGE = {"pancreas": {"Beta": (0.0, 0.4)}}


def _transparent_white(img, thresh=0.93):
    """Add an alpha channel to a TikZ raster so its white background becomes transparent (the network can
    then overlay the embedding without a white box)."""
    img = np.asarray(img, float)
    if img.ndim == 2:
        img = np.dstack([img, img, img])
    rgb = img[..., :3]
    rgba = np.dstack([rgb, np.ones(rgb.shape[:2])])
    rgba[(rgb > thresh).all(axis=2), 3] = 0.0
    return rgba

# Differentiation path for the panel-g arrows (progenitor chain, then branch to terminal fates). Datasets
# without an entry fall back to the linear chain implied by the config cell-type order.
DIFF_PATH = {
    "pancreas": [("Ductal", "Ngn3 low EP"), ("Ngn3 low EP", "Ngn3 high EP"),
                 ("Ngn3 high EP", "Pre-endocrine"), ("Pre-endocrine", "Alpha"),
                 ("Pre-endocrine", "Beta"), ("Pre-endocrine", "Delta"), ("Pre-endocrine", "Epsilon")],
}


def _velocity_speed(a):
    return np.linalg.norm(np.asarray(a.layers["velocity_S"]), axis=1)


def _per_cell_attractor(a):
    """Per-cell attractor index = z(-velocity magnitude) + z(-interaction energy): settled + deep basin.
    Positive = attractor-like (terminal), negative = transient (fast, shallow). Same construction as the
    per-cell-type attractor score in _scores, painted per cell."""
    speed = _velocity_speed(a)
    eint = a.obs["energy_interaction"].values.astype(float)
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
    return z(-speed) + z(-eint)


def _per_cell_settling(a):
    """Per-cell settling = 1 - minmax(velocity magnitude): low velocity -> settled (1), fast-moving (0)."""
    s = _velocity_speed(a)
    return 1.0 - (s - s.min()) / (s.max() - s.min() + 1e-12)


def _per_cell_oscillation(a):
    """Per-cell oscillation = minmax(Jacobian rotational magnitude): the local rotational character."""
    r = a.obs["jacobian_rotational"].values.astype(float)
    return (r - np.nanmin(r)) / (np.nanmax(r) - np.nanmin(r) + 1e-12)


def _boxen_by_type(ax, a, ck, col, order, colors):
    """Per-cell-type boxen (letter-value) plot of a per-cell obs field; horizontal x labels and the full
    data range (no symmetric clipping, so heavy-tailed groups such as Delta are not cut off)."""
    import seaborn as sns
    v = a.obs[col].values.astype(float); cl = a.obs[ck].astype(str).values
    m = np.isfinite(v)
    df = pd.DataFrame({"v": v[m], "c": cl[m]})
    pal = {c: colors.get(c, "0.6") for c in order}
    sns.boxenplot(data=df, x="c", y="v", order=order, hue="c", palette=pal, legend=False, ax=ax,
                  linewidth=0.3, linecolor="0.25", showfliers=False)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=0, ha="center", fontsize=4.2)
    ax.tick_params(axis="y", labelsize=5.6); ax.tick_params(axis="x", length=0)


def draw_dynamical_readouts(fig, gs_cell, a, basis, ck, order, colors):
    """Combined panel (former c + d): four per-cell dynamical readouts, TOP row = UMAP projections, BOTTOM
    row = per-cell-type violins. Columns: local stability (leading real eigenvalue, + = locally unstable);
    attractor index (settling + interaction-energy depth); oscillation score (rotational); settling score
    (low input RNA velocity). Leading-Re next to the attractor index makes the quasi-potential paradox
    explicit; oscillation and settling are the two axes of the character map in the settling/oscillation
    panel."""
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    for col, fn in [("_attractor", _per_cell_attractor), ("_oscillation", _per_cell_oscillation),
                    ("_settling", _per_cell_settling)]:
        if col not in a.obs:
            a.obs[col] = fn(a)
    specs = [("jacobian_leading_real", "RdBu_r", "leading eig (Re): + locally unstable", True),
             ("_attractor", "PuOr", "attractor index: + attractor / - transient", True),
             ("_oscillation", "magma", "oscillation score (rotational)", False),
             ("_settling", "viridis", "settling score (low velocity)", False)]
    sub = gs_cell.subgridspec(2, 4, wspace=0.24, hspace=0.30, height_ratios=[1.25, 1.0])
    ax0 = None
    for j, (col, cmap, title, sym) in enumerate(specs):
        axu = fig.add_subplot(sub[0, j]); ax0 = ax0 or axu
        draw_umap_value(axu, emb, a.obs[col].values, cmap, title, symmetric=sym)
        _boxen_by_type(fig.add_subplot(sub[1, j]), a, ck, col, order, colors)
    return ax0


def draw_spectra_by_type(fig, gs_cell, a, ck, order, colors, cells_per_type=200, top_k=40):
    """Panel (spectra, moved up after a/b): Jacobian eigenvalue spectra split BY CELL TYPE into a 2xN grid
    with SHARED axes and NO forced symmetry about Re = 0 (the leading eigenvalues sit mostly at Re > 0).
    Per cell the top-k leading eigenvalues (largest Re) are kept; the near-zero bulk is dropped."""
    E = np.asarray(a.obsm["jacobian_eigenvalues"]); cl = a.obs[ck].astype(str).values
    rng = np.random.default_rng(0)
    per = {}
    for c in order:
        idx = np.where(cl == c)[0]
        if len(idx) == 0:
            per[c] = np.array([], complex); continue
        if len(idx) > cells_per_type:
            idx = rng.choice(idx, cells_per_type, replace=False)
        ev = E[idx]; k = min(top_k, ev.shape[1])
        keep = np.argsort(-ev.real, axis=1)[:, :k]
        per[c] = np.take_along_axis(ev, keep, axis=1).ravel()
    allsel = np.concatenate([v for v in per.values() if len(v)]) if any(len(v) for v in per.values()) else np.array([0j])
    xlo = float(np.nanpercentile(allsel.real, 0.5)); xhi = float(np.nanpercentile(allsel.real, 99.5))
    yhi = float(np.nanpercentile(np.abs(allsel.imag), 99.5)) or 1e-3
    xpad = 0.05 * (xhi - xlo + 1e-9)
    nc = int(np.ceil(len(order) / 2)); nr = 2
    sub = gs_cell.subgridspec(nr, nc, wspace=0.10, hspace=0.30)
    ax0 = None
    for k2, c in enumerate(order):
        ax = fig.add_subplot(sub[k2 // nc, k2 % nc]); ax0 = ax0 or ax
        sel = per[c]
        if len(sel):
            ax.scatter(sel.real, sel.imag, s=2.0, c=colors.get(c, "0.6"), alpha=0.55, linewidths=0)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_xlim(xlo - xpad, xhi + xpad); ax.set_ylim(-yhi * 1.12, yhi * 1.12)
        _ct_title(ax, c, colors, fontsize=6.2, pad=2, alpha=0.45); ax.tick_params(labelsize=5)
        if k2 % nc != 0:
            ax.set_yticklabels([])
        if k2 // nc != nr - 1:
            ax.set_xticklabels([])
    return ax0


def _order_colors(a, ck, cfg):
    present = present_clusters(a, ck)
    order = [c for c in (cfg.get("order") or present) if c in present]
    colors = get_colors(a, ck)
    return order, {c: colors.get(c, "0.6") for c in order}


def _by_type(a, ck, col, order):
    return [a.obs[col].values[a.obs[ck].astype(str).values == c] for c in order]


def _ct_title(ax, c, colors, fontsize=6.5, pad=3, alpha=0.85):
    """Cell-type subtitle: black bold text on a rounded, semi-transparent background of the cell color."""
    ax.set_title(c, fontsize=fontsize, color="black", fontweight="bold", pad=pad,
                 bbox=dict(boxstyle="round,pad=0.22", facecolor=colors.get(c, "0.75"),
                           edgecolor="none", alpha=alpha))


def draw_umap_value(ax, emb, vals, cmap, title, symmetric=False, pct=99, cbar=True):
    """Paint a per-cell scalar on the embedding (the report's section-4.1 idiom). Symmetric maps
    (leading real eigenvalue) center on 0 with RdBu_r so stable/unstable read by hue."""
    v = np.asarray(vals, float)
    if symmetric:
        m = float(np.nanpercentile(np.abs(v), pct)) or 1.0
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=v, cmap=cmap, vmin=-m, vmax=m, s=3, linewidths=0)
    else:
        lo = float(np.nanpercentile(v, 100 - pct)); hi = float(np.nanpercentile(v, pct))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=v, cmap=cmap, vmin=lo, vmax=hi, s=3, linewidths=0)
    ax.set_title(title, fontsize=8); ax.set_axis_off()
    if cbar:
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=5.5)
    return sc


def _box_by_type(ax, a, ck, col, order, colors, symmetric=False, title="", ylabel=""):
    data = _by_type(a, ck, col, order)
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(colors.get(c, "0.6")); patch.set_alpha(0.85); patch.set_linewidth(0.4)
    for med in bp["medians"]:
        med.set_color("k"); med.set_linewidth(0.8)
    if symmetric:
        mx = float(np.nanpercentile(np.abs(a.obs[col].values), 99)) or 1.0
        ax.set_ylim(-mx, mx)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=5.4)
    if title:
        ax.set_title(title, fontsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(axis="y", labelsize=5.6)


def draw_energy_landscape(ax, a, basis, colors, ck):
    """Panel a: total-energy landscape as a 3D scatter over the 2D embedding (z = energy_total), each
    cell colored by cell type. No interpolated surface: an earlier plot_trisurf triangulated across the
    empty gap between the lineage arms and produced spurious spikes, so only the real per-cell heights
    are shown. The embedding (x, y) axes, background panes, and grid are removed for clarity; only the
    energy (z) axis is kept."""
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    z = a.obs["energy_total"].values.astype(float)
    cl = a.obs[ck].astype(str).values
    ax.scatter(emb[:, 0], emb[:, 1], z, c=[colors.get(c, "0.6") for c in cl], s=4, depthshade=False)
    ax.set_axis_off()                                            # remove ALL default 3D axes/ticks/panes
    xmn, xmx = float(emb[:, 0].min()), float(emb[:, 0].max())
    ymn, ymx = float(emb[:, 1].min()), float(emb[:, 1].max())
    zmn, zmx = float(z.min()), float(z.max())
    ax.set_xlim(xmn, xmx); ax.set_ylim(ymn, ymx); ax.set_zlim(zmn, zmx)
    # a single coordinate frame: three spike-arrows from ONE common origin corner (the base of the energy
    # spike): UMAP1 (+x), UMAP2 (+y), total energy (+z), all tick-free; the energy spike carries manual
    # value labels
    lx, ly = 0.30 * (xmx - xmn), 0.30 * (ymx - ymn)
    ax.quiver(xmn, ymn, zmn, lx, 0, 0, color="0.3", arrow_length_ratio=0.16, lw=1.4)   # UMAP1 (arrow)
    ax.quiver(xmn, ymn, zmn, 0, ly, 0, color="0.3", arrow_length_ratio=0.16, lw=1.4)   # UMAP2 (arrow)
    ax.plot([xmn, xmn], [ymn, ymn], [zmn, zmx], color="0.3", lw=1.4)                    # energy spike (no arrow)
    ax.text(xmn + lx * 1.18, ymn - 0.05 * (ymx - ymn), zmn, "UMAP1", fontsize=7, color="0.25",
            ha="left", va="top")
    ax.text(xmn - 0.05 * (xmx - xmn), ymn + ly * 1.15, zmn, "UMAP2", fontsize=7, color="0.25",
            ha="right", va="center")
    ax.text(xmn, ymn, zmx * 1.03, "total energy", fontsize=7.5, color="0.25", ha="center", va="bottom")
    from matplotlib.ticker import MaxNLocator                    # manual energy ticks: a short mark ON the
    tk = 0.05 * (xmx - xmn)                                      # spike plus a value label just left of it
    for zt in MaxNLocator(4).tick_values(zmn, zmx):
        if zmn <= zt <= zmx:
            ax.plot([xmn, xmn - tk], [ymn, ymn], [zt, zt], color="0.3", lw=1.0)   # tick mark on the spike
            ax.text(xmn - tk * 1.4, ymn, zt, f"{zt:.0f}", fontsize=5, color="0.4", ha="right", va="center")
    ax.set_box_aspect((1, 1, 0.72)); ax.view_init(elev=20, azim=-60)
    ax.set_title("total-energy landscape", fontsize=9, y=0.97)


def draw_energy_components(fig, gs_cell, a, ck, order, colors):
    """Panel b: per-cell-type boxplots of each energy component (all four), in a 2x2 grid (matching the
    report's sch.pl.plot_energy_boxplots layout; same obs["energy_*"] data, outliers hidden for clarity)."""
    sub = gs_cell.subgridspec(2, 2, wspace=0.32, hspace=0.62)
    axes = []
    for j, (col, lab) in enumerate(ENERGY_COMPONENTS):
        ax = fig.add_subplot(sub[j // 2, j % 2]); axes.append(ax)
        data = _by_type(a, ck, col, order)
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(colors.get(c, "0.6")); patch.set_alpha(0.85); patch.set_linewidth(0.4)
        for med in bp["medians"]:
            med.set_color("k"); med.set_linewidth(0.8)
        ax.axhline(0, color="0.5", lw=0.6, ls="--")
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=60, ha="right", fontsize=5.6)
        ax.set_title(lab, fontsize=8); ax.tick_params(axis="y", labelsize=6)
        if j % 2 == 0:
            ax.set_ylabel("energy", fontsize=8)
    return axes[0]


def draw_leading_positive(fig, gs_cell, a, basis, ck, order, colors):
    """Panel c: LEFT = local Jacobian stability (leading real eigenvalue; + = locally unstable), RIGHT =
    the per-cell ATTRACTOR index (settling + interaction-energy depth). Shown side by side, they make the
    quasi-potential paradox explicit: terminal cells are locally UNSTABLE (leading Re > 0) yet are the
    true ATTRACTORS (deep + settled). The full eigenvalue spectrum is in panel e; the count of positive
    eigenvalues is dropped here as redundant with the leading real eigenvalue. TOP = per-cell UMAP
    projections, BOTTOM = the per-cell-type boxplots."""
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    if "_attractor" not in a.obs:
        a.obs["_attractor"] = _per_cell_attractor(a)
    sub = gs_cell.subgridspec(2, 2, wspace=0.28, hspace=0.32, height_ratios=[1.35, 1.0])
    ax0 = fig.add_subplot(sub[0, 0])
    draw_umap_value(ax0, emb, a.obs["jacobian_leading_real"].values, "RdBu_r",
                    "leading eig (Re): + locally unstable", symmetric=True)
    ax1 = fig.add_subplot(sub[0, 1])
    draw_umap_value(ax1, emb, a.obs["_attractor"].values, "PuOr",
                    "attractor index: + attractor / - transient", symmetric=True)
    axb0 = fig.add_subplot(sub[1, 0])
    _box_by_type(axb0, a, ck, "jacobian_leading_real", order, colors, symmetric=True,
                 ylabel="leading eig (Re)")
    axb1 = fig.add_subplot(sub[1, 1])
    _box_by_type(axb1, a, ck, "_attractor", order, colors, symmetric=True, ylabel="attractor index")
    return ax0


def draw_rotational(fig, gs_cell, a, basis, ck, order, colors):
    """Panel d: Jacobian rotational magnitude (antisymmetric part). LEFT = per-cell UMAP projection
    (magma); RIGHT = the per-cell-type boxplot."""
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    sub = gs_cell.subgridspec(1, 2, wspace=0.28, width_ratios=[1.1, 1.0])
    ax0 = fig.add_subplot(sub[0, 0])
    draw_umap_value(ax0, emb, a.obs["jacobian_rotational"].values, "magma", "rotational magnitude")
    axb = fig.add_subplot(sub[0, 1])
    _box_by_type(axb, a, ck, "jacobian_rotational", order, colors, ylabel="rotational magnitude")
    return ax0


def draw_spectra(ax, a, ck, order, colors, cells_per_type=200, top_k=40):
    """Panel e: Jacobian eigenvalue spectra (real vs imaginary), colored by cell type. Only the top-k
    LEADING eigenvalues per cell (largest real part, i.e. the least-stable / stability-determining modes)
    are shown; the large bulk of strongly negative (degradation-dominated) eigenvalues is dropped so the
    near-zero stability edge and its off-axis (oscillatory) structure are visible instead of a strip at
    the far-negative wall."""
    E = np.asarray(a.obsm["jacobian_eigenvalues"])
    cl = a.obs[ck].astype(str).values
    rng = np.random.default_rng(0)
    shown = []
    for c in order:
        idx = np.where(cl == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) > cells_per_type:
            idx = rng.choice(idx, cells_per_type, replace=False)
        ev = E[idx]                                             # (n_cells, n_eig) complex
        k = min(top_k, ev.shape[1])
        keep = np.argsort(-ev.real, axis=1)[:, :k]             # top-k LEADING (largest Re) per cell
        sel = np.take_along_axis(ev, keep, axis=1).ravel()
        ax.scatter(sel.real, sel.imag, s=2.5, c=colors.get(c, "0.6"), alpha=0.5, linewidths=0)
        shown.append(sel)
    ax.axvline(0, color="k", lw=0.8, ls="--"); ax.axhline(0, color="0.6", lw=0.5, ls=":")
    ax.set_xlabel("Re(eigenvalue)", fontsize=8); ax.set_ylabel("Im(eigenvalue)", fontsize=8)
    ax.set_title("Jacobian eigenvalue spectra (leading modes)", fontsize=9); ax.tick_params(labelsize=6)
    if shown:
        s = np.concatenate(shown)
        xr = float(np.nanpercentile(np.abs(s.real), 99.5)); yr = float(np.nanpercentile(np.abs(s.imag), 99.5))
        ax.set_xlim(-xr * 1.05, xr * 1.05); ax.set_ylim(-yr * 1.1 - 1e-6, yr * 1.1 + 1e-6)


# The activation / repression pair is this figure's, and it is the same pair panel a of the
# network figure uses, because the two draw the same kind of object and this figure's own
# legends already import it from there (see the three `from make_network_figure import ACT`
# below). It used to be a separate blue / red pair defined here, which meant the drawn
# fallback contradicted the legend printed beside it on any machine without TeX.
from make_network_figure import ACT, REP, ACT_HEX, REP_HEX          # noqa: E402

#: These networks are rasterized at the poster figure's resolution, not the page one.
_GRN_DPI = 460


def _draw_mini_network(ax, genes, pos, edges, meanJ, mask, smax):
    """One cell-type regulatory mini-network, drawn with matplotlib.

    This is what the panel falls back to when the TikZ render is unavailable. A directed
    edge runs regulator -> target per JAC_PAIR, colored by the SIGN of the cell-type-mean
    J[target<-reg] (activation with an arrowhead, repression with a FLAT BAR head,
    following the -{Bar}/-{Latex} circuit convention), width ~ |magnitude|.
    """
    eds = [(r, t, meanJ(t, r, mask)) for (t, r) in edges]
    return sch.pl.draw_grn_mpl(ax, genes, pos, eds, wmax=smax, act_color=ACT, rep_color=REP,
                               xlim=(-1.55, 1.55), ylim=(-1.6, 1.5))


def draw_jacobian_networks(fig, gs_cell, a, basis, ck, order, colors, ds):
    """Panel: per-cell-type regulatory mini-networks ANCHORED to the embedding. The UMAP (cells colored by
    type, cell-type centroids marked) sits on top; below it a row of mini-networks, one per cell type,
    each joined to its centroid by a thin line. Each mini-network shows the JAC_PAIRS circuit with edges
    colored by the SIGN of that cell type's mean Jacobian element (blue activation / red repression), so
    the circuit's character can be read off against where the cells sit in the trajectory."""
    import scHopfield as sch
    from matplotlib.patches import ConnectionPatch
    edges = [(t, r) for (t, r) in JAC_PAIRS.get(ds, []) if t in a.var_names and r in a.var_names]
    genes = list(dict.fromkeys([g for g in NET_GENES.get(ds, []) if g in a.var_names]
                               + [g for e in edges for g in e]))
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    if len(edges) == 0:
        ax = fig.add_subplot(gs_cell); ax.scatter(emb[:, 0], emb[:, 1], s=3, c="0.8"); ax.set_axis_off()
        return ax
    b = a.copy()
    try:
        sch.tl.compute_jacobian_elements(b, gene_pairs=edges, cluster_key=ck, store_in_obs=True)
    except Exception as e:
        print(f"[f] jacobian elements failed: {e}", flush=True)
    cl = b.obs[ck].astype(str).values
    cols = {(t, r): _find_element_col(b, t, r) for (t, r) in edges}

    def meanJ(t, r, mask):
        col = cols.get((t, r))
        return float(np.nanmean(b.obs[col].values[mask])) if col else np.nan

    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(genes), endpoint=False)
    pos = {g: (float(np.cos(t)), float(np.sin(t))) for g, t in zip(genes, ang)}
    mags = [abs(meanJ(t, r, cl == c)) for c in order for (t, r) in edges]
    smax = float(np.nanpercentile([m for m in mags if np.isfinite(m)], 95)) or 1.0
    cent = {c: emb[cl == c].mean(0) for c in order if (cl == c).any()}

    # The circuit renderer is package API now (proper -{Latex} activation / -{Bar} repression),
    # so this panel no longer reaches sideways into another figure script for it.
    grn_preamble = sch.pl.grn_preamble(ACT_HEX, REP_HEX)

    ncol = max(len(order), 1)
    bb = gs_cell.get_position(fig)
    # One network = S wide in figure fraction. The figure is taller than wide, so a SQUARE network on the
    # page has height Sy = S*(fig_w/fig_h), and "one network size" up = Sy, right = S (equal on the page).
    S = 0.120; asp = fig.get_figwidth() / fig.get_figheight(); Sy = S * asp
    # The embedding fills the top of the panel; a clean default row of (big) networks sits below it, clear
    # of the scatter. User offsets (NET_OFFSETS) then move each network up onto the embedding.
    row_h = Sy + 0.012
    ax_um = fig.add_axes([bb.x0, bb.y0 + row_h, bb.width, bb.height - row_h])
    ax_um.scatter(emb[:, 0], emb[:, 1], c=[colors.get(c, "0.8") for c in cl], s=16, alpha=0.75, linewidths=0)
    for c, xy in cent.items():
        ax_um.scatter([xy[0]], [xy[1]], s=60, c=colors.get(c, "0.6"), edgecolor="k", lw=0.9, zorder=8)
    xr, yr = float(np.ptp(emb[:, 0])), float(np.ptp(emb[:, 1]))   # fix the data limits so transData is exact
    ax_um.set_xlim(emb[:, 0].min() - 0.03 * xr, emb[:, 0].max() + 0.03 * xr)
    ax_um.set_ylim(emb[:, 1].min() - 0.03 * yr, emb[:, 1].max() + 0.03 * yr)
    ax_um.set_axis_off()
    inv = fig.transFigure.inverted()
    offs = NET_OFFSETS.get(ds, {})
    for k, c in enumerate(order):
        mask = cl == c
        eds = [(r, t, meanJ(t, r, mask)) for (t, r) in edges if np.isfinite(meanJ(t, r, mask))]
        # Thicker edges than the poster figure uses. A network is 21.6 mm here against 48.8 mm
        # there, and the TikZ render is scaled into that box, so the same line width comes out
        # less than half as heavy on the page.
        # The TikZ picture is roughly 5.5 cm across and is scaled into a 21.6 mm box, about
        # 0.4x, so a label set at \tiny (5 pt) lands near 2 pt on the page. 13 pt inside the
        # picture comes out near 5 pt printed. off_base is raised to keep the larger labels
        # clear of the nodes.
        img = sch.pl.render_tikz(
            sch.pl.grn_tikz_body(genes, pos, eds, scale=2.4, size=5.5,
                                 lblfont=r"\fontsize{13}{15}\selectfont",
                                 off_base=0.26, label_sep=0.9,
                                 edge_lw=(0.55, 2.0), head_scale=1.7),
            preamble=grn_preamble, dpi=_GRN_DPI) if eds else None
        x0 = bb.x0 + (k + 0.5) * bb.width / ncol - S / 2       # default row baseline, then user offset
        y0 = bb.y0 + 0.004
        dxu, dyu = offs.get(c, (0.0, 0.0))
        nx, ny = x0 + dxu * S, y0 + dyu * Sy
        pad = 0.008                                            # keep the whole network box inside the figure
        nx = float(np.clip(nx, pad, 1.0 - S - pad))            # (Ductal overflowed left, Alpha right)
        ny = float(np.clip(ny, pad, 1.0 - Sy - pad))
        axn = fig.add_axes([nx, ny, S, Sy]); axn.set_axis_off()
        if img is not None:
            axn.imshow(_transparent_white(img))               # transparent bg so it overlays the embedding
            y0, y1 = axn.get_ylim()
            axn.set_ylim(y0, y1 - 0.16 * img.shape[0])        # headroom under the cell-type chip
        else:
            _draw_mini_network(axn, genes, pos, edges, meanJ, mask, smax)
        _ct_title(axn, c, colors, fontsize=6.0, pad=1.4, alpha=0.6)
        if c in cent:                                         # black arrow: to the title, or to the nearest edge
            # shrinkA clears the centroid marker so the line emerges at its edge (the marker's high zorder
            # also covers any residual start inside the disc).
            akw = dict(color="k", lw=0.7, arrowstyle="-|>", mutation_scale=7, shrinkA=5, shrinkB=1, zorder=6)
            if c in NET_TITLE_ARROW.get(ds, set()):
                akw_t = {**akw, "shrinkB": 7}                 # leave a gap so the head does not touch the title
                fig.add_artist(ConnectionPatch(xyA=(cent[c][0], cent[c][1]), coordsA=ax_um.transData,
                                               xyB=(0.5, 1.02), coordsB=axn.transAxes, **akw_t))
            else:
                fx, fy = inv.transform(ax_um.transData.transform((cent[c][0], cent[c][1])))
                ndx, ndy = NET_ARROW_NUDGE.get(ds, {}).get(c, (0.0, 0.0))
                tx = float(np.clip(fx, nx, nx + S)) + ndx * S
                ty = float(np.clip(fy, ny, ny + Sy)) + ndy * Sy
                fig.add_artist(ConnectionPatch(xyA=(fx, fy), coordsA=fig.transFigure, xyB=(tx, ty),
                                               coordsB=fig.transFigure, **akw))
    return ax_um


def _find_element_col(a, tgt, reg):
    for c in a.obs.columns:
        cl = c.lower()
        if "jac" in cl and tgt.lower() in cl and reg.lower() in cl:
            return c
    return None


def _scores(a, ck, order):
    """Per-cell-type dynamical scores. Terminal-state (attractor) identity is read from ENERGY DEPTH plus
    dynamical SETTLING (low input RNA velocity magnitude), NOT the local Jacobian leading eigenvalue:
      settling    = min-max normalized LOW velocity magnitude (settled = 1, fast-moving = 0);
      oscillation = mean Jacobian rotational magnitude, min-max normalized (the complementary character);
      attractor   = z(-mean velocity magnitude) + z(-mean interaction energy)  (settled + deep = attractor).

    CAVEAT (a result, not a bug). Because the inferred interaction matrix is ASYMMETRIC, the energy is a
    quasi-potential (an effective landscape), NOT a strict Lyapunov function, so local Jacobian stability
    (leading real eigenvalue < 0) and energy depth (a deep basin) need not coincide. Pancreatic Beta is
    the concrete case: it sits in a deep (strongly negative interaction-energy) basin AND is dynamically
    settled (low velocity) yet has a positive leading real eigenvalue, so an attractor index built on the
    leading eigenvalue mislabels this terminal cell type as 'transient'. Terminal identity is therefore
    read from the deep-and-settled signals here, with the Jacobian spectrum reserved for the complementary
    rotational/oscillatory character (panel d) and the leading eigenvalue reported on its own (panel c)."""
    cl = a.obs[ck].astype(str).values
    speed = np.linalg.norm(np.asarray(a.layers["velocity_S"]), axis=1)   # INPUT RNA velocity magnitude, not the fitted field
    rot = a.obs["jacobian_rotational"].values.astype(float)
    eint = a.obs["energy_interaction"].values.astype(float)              # basin depth (deep = attractor)
    rows = {}
    for c in order:
        m = cl == c
        if m.sum() == 0:
            continue
        rows[c] = dict(speed=float(speed[m].mean()), oscillation=float(rot[m].mean()),
                       menergy=float(eint[m].mean()), n=int(m.sum()))
    df = pd.DataFrame(rows).T
    mm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    df["settling"] = 1.0 - mm(df["speed"])                              # low velocity -> settled (attractor)
    df["oscillation_n"] = mm(df["oscillation"])
    z = lambda s: (s - s.mean()) / (s.std() + 1e-12)
    df["attractor"] = z(-df["speed"]) + z(-df["menergy"])               # settled + deep interaction basin
    return df


def draw_stability_oscillation(ax, df, colors, ds=None, order=None):
    """Panel g: per-cell-type settling (x, low input RNA velocity = settled attractor) vs oscillation (y),
    a dynamical-character map. Settled + low-oscillation = terminal attractors; fast + oscillatory =
    transitional. Grey arrows trace the differentiation path (progenitors -> terminal fates), so the
    trajectory reads as a drift from the low-settling (fast) progenitor corner into the settled
    attractors."""
    path = DIFF_PATH.get(ds)
    if path is None and order is not None:
        path = list(zip(order[:-1], order[1:]))
    for s, t in (path or []):                                   # differentiation-path arrows (under points)
        if s in df.index and t in df.index:
            ax.annotate("", xy=(df.loc[t, "settling"], df.loc[t, "oscillation_n"]),
                        xytext=(df.loc[s, "settling"], df.loc[s, "oscillation_n"]),
                        arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1.0, alpha=0.8,
                                        shrinkA=7, shrinkB=7), zorder=1)
    texts = []
    for c, r in df.iterrows():                                  # uniform marker size (size carried no
        ax.scatter(r["settling"], r["oscillation_n"], s=110,    # documented meaning, so it is not encoded)
                   c=colors.get(c, "0.6"), edgecolor="k", linewidth=0.5, zorder=3)
        texts.append(ax.text(r["settling"], r["oscillation_n"], c, fontsize=6.5, zorder=4))
    try:                                                        # push labels off the dots and the arrows
        from adjustText import adjust_text
        adjust_text(texts, x=df["settling"].to_numpy(), y=df["oscillation_n"].to_numpy(), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4), expand=(1.4, 1.6))
    except Exception:
        pass
    ax.set_xlabel("settling score  (low input RNA velocity = settled)", fontsize=8)
    ax.set_ylabel("oscillation score  (rotational, normalized)", fontsize=8)
    ax.set_title("settling vs oscillation per cell type", fontsize=9)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.tick_params(labelsize=6.5)


def draw_attractor_transient(ax, df, colors):
    """Panel h: per-cell-type attractor (positive) vs transient (negative) index, ordered."""
    d = df.sort_values("attractor")
    ax.barh(range(len(d)), d["attractor"].values,
            color=[colors.get(c, "0.6") for c in d.index], edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=6.5)
    ax.set_xlabel("attractor index (+)  /  transient (-)", fontsize=8)
    ax.set_title("attractor vs transient per cell type", fontsize=9); ax.tick_params(axis="x", labelsize=6.5)


# ======================================================================================
# Submission mode (--submission): the same six panels, re-laid out for one Nature Machine
# Intelligence page (180 mm wide, and short enough that the printed legend still fits under
# it). Nothing below this line runs without the flag, so the poster-size figure that the
# per-dataset reports embed is byte-for-byte untouched.
#
# The height came out of the type, not the other way round: every label is set first at 5.5
# to 8 pt and the bands are then sized to hold it. What changed relative to the poster:
#   b, d  per-cell-type distributions are drawn horizontally, so the cell-type names read
#         straight instead of rotated, and are printed once (panel b) rather than four times;
#   c     the cell-type name moves inside each spectrum as a tag, which buys back the title
#         band of both rows;
#   d     each colorbar is horizontal, under its map, and is labeled by meaning at the two
#         poles rather than by numbers;
#   e     the mini-networks move from an overlay to a row directly above the embedding, each
#         one over its own cluster and joined to its centroid, because a TikZ network shrunk
#         to 18 mm would carry 3 pt gene labels. The gene identities are printed once, in the
#         circuit key beside the embedding, and every network uses that same node layout.
# No colormap, no cell-type color and no panel letter changes.
# ======================================================================================

SUB_W_MM = 180.0                        # the double-column hard maximum
SUB_H_MM = 204.0                        # the circuits band moved to its own figure
# Panel bands, as (top edge, height) in millimeters measured down from the top of the canvas.
# The c-to-d gap is 10.5 mm rather than the 4.5 mm it started at: band c carries tick labels
# and an axis label below its axes, band d carries titles above its maps, and at 4.5 mm the
# two ran together with no clear line between the panels.
# The circuits panel is no longer part of this figure: it answers a different question from
# the rest (which regulatory wiring, where) and now stands alone, drawn by
# render_submission_circuits(). What was band f is band e here.
SUB_BANDS = {
    "ab": (4.0, 54.0),          # + ~10 mm below for panel b's 45-degree cell-type names
    "c": (72.0, 32.0),
    "d": (116.0, 41.0),
    "e": (164.0, 35.0),
}
CIRCUITS_H_MM = 108.0                   # the standalone circuits figure's own canvas
FS_TITLE, FS_LABEL, FS_TICK, FS_TAG = 7.0, 6.5, 5.5, 5.5


def _sub_rect(top_mm, h_mm, x0_mm, x1_mm):
    """Gridspec bounds in figure fraction, from millimeters measured down from the top left."""
    return dict(top=1.0 - top_mm / SUB_H_MM, bottom=1.0 - (top_mm + h_mm) / SUB_H_MM,
                left=x0_mm / SUB_W_MM, right=x1_mm / SUB_W_MM)


def _sub_letter(fig, x_mm, y_mm, letter):
    """A panel letter for a group of axes, in figure coordinates but typographically
    identical to submission_style.panel_letter() (bold, lowercase, 8 pt, above top left)."""
    from submission_style import TYPE_PANEL_LETTER
    fig.text(x_mm / SUB_W_MM, 1.0 - y_mm / SUB_H_MM, letter, fontsize=TYPE_PANEL_LETTER,
             fontweight="bold", ha="left", va="bottom")


def _sub_tag(ax, c, colors, fontsize=FS_TAG, inside=False):
    """Cell-type name on a rounded chip of that cell type's own color."""
    bb = dict(boxstyle="round,pad=0.18", facecolor=colors.get(c, "0.75"), edgecolor="none",
              alpha=0.75)
    if inside:
        return ax.text(0.03, 0.94, c, transform=ax.transAxes, fontsize=fontsize, ha="left",
                       va="top", bbox=bb, zorder=6)
    return ax.set_title(c, fontsize=fontsize, color="black", pad=1.6, bbox=bb)


def _sub_landscape(ax, a, basis, colors, ck):
    """Panel a, as in draw_energy_landscape but sized for the page: the same per-cell 3D
    scatter (no interpolated surface), the same single corner frame, smaller type, and a
    box aspect that follows the embedding, which is wider than it is tall."""
    from matplotlib.ticker import MaxNLocator
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    z = a.obs["energy_total"].values.astype(float)
    cl = a.obs[ck].astype(str).values
    ax.scatter(emb[:, 0], emb[:, 1], z, c=[colors.get(c, "0.6") for c in cl], s=0.7,
               depthshade=False, linewidths=0, rasterized=True)
    ax.set_axis_off()
    xmn, xmx = float(emb[:, 0].min()), float(emb[:, 0].max())
    ymn, ymx = float(emb[:, 1].min()), float(emb[:, 1].max())
    zmn, zmx = float(z.min()), float(z.max())
    ax.set_xlim(xmn, xmx); ax.set_ylim(ymn, ymx); ax.set_zlim(zmn, zmx)
    lx, ly = 0.30 * (xmx - xmn), 0.30 * (ymx - ymn)
    ax.quiver(xmn, ymn, zmn, lx, 0, 0, color="0.3", arrow_length_ratio=0.16, lw=0.8)
    ax.quiver(xmn, ymn, zmn, 0, ly, 0, color="0.3", arrow_length_ratio=0.16, lw=0.8)
    # clip_on=False: with zoom > 1 the frame corner lies outside the axes box, where Line3D
    # is clipped but Text is not. Without this the energy spine and its ticks disappear while
    # their labels stay, which reads as a missing axis rather than as clipping.
    ax.plot([xmn, xmn], [ymn, ymn], [zmn, zmx], color="0.3", lw=0.8, clip_on=False)
    ax.text(xmn + lx * 1.25, ymn - 0.05 * (ymx - ymn), zmn, "UMAP1", fontsize=FS_TICK,
            color="0.25", ha="left", va="top")
    ax.text(xmn - 0.04 * (xmx - xmn), ymn + ly * 1.20, zmn, "UMAP2", fontsize=FS_TICK,
            color="0.25", ha="right", va="center")
    ax.text(xmn, ymn, zmx * 1.06, "total energy", fontsize=FS_LABEL, color="0.25",
            ha="center", va="bottom")
    tk = 0.05 * (xmx - xmn)
    for zt in MaxNLocator(3).tick_values(zmn, zmx):
        if zmn <= zt <= zmx:
            ax.plot([xmn, xmn - tk], [ymn, ymn], [zt, zt], color="0.3", lw=0.6,
                    clip_on=False)
            ax.text(xmn - tk * 1.5, ymn, zt, f"{zt:.0f}", fontsize=FS_TICK, color="0.4",
                    ha="right", va="center")
    # zoom fills the panel: a 3D axes reserves large internal margins, so at 0.92 the
    # landscape sat in the middle of a mostly empty box.
    ax.set_box_aspect((1.25, 1.0, 0.80), zoom=1.16)
    ax.view_init(elev=20, azim=-60)


def _sub_clean(ax):
    """House style: no top or right spine, so only the data-carrying rules survive."""
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _sub_hbox(ax, data, order, colors, ynames=True, boxen=False, vertical=False):
    """One horizontal per-cell-type distribution column, first cell type at the top.

    ``boxen`` draws a letter-value plot instead of a plain box, matching the poster figure.
    It matters wherever the groups differ sharply in spread: on the leading real eigenvalue
    four of the eight pancreatic cell types sit almost exactly at zero, and a plain box
    renders each of those as an unreadable sliver against the spine, while the nested
    letter values still show their shape.
    """
    y = np.arange(len(order))[::-1]
    if boxen or vertical:
        import seaborn as sns
        df = pd.DataFrame({"v": np.concatenate([np.asarray(d, float) for d in data]),
                           "c": np.repeat(list(order), [len(d) for d in data])})
        pal = {c: colors.get(c, "0.6") for c in order}
        kw = dict(data=df, order=list(order), hue="c", palette=pal, legend=False, ax=ax,
                  linewidth=0.22, linecolor="0.25", showfliers=False, width=0.86)
        draw = sns.boxenplot if boxen else sns.boxplot
        if not boxen:
            kw.update(linewidth=0.35, fliersize=0)
        if vertical:
            draw(x="c", y="v", **kw)
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(list(order) if ynames else [], fontsize=FS_TAG,
                               rotation=45, ha="right", va="top",
                               rotation_mode="anchor")
            ax.tick_params(axis="x", length=0, pad=1.0)
            ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.2)
        else:
            draw(y="c", x="v", **kw)
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(list(order) if ynames else [], fontsize=FS_TAG)
            ax.tick_params(axis="y", length=0, pad=1.2)
            ax.tick_params(axis="x", labelsize=FS_TICK, pad=1.2)
        return
    bp = ax.boxplot(data, orientation="horizontal", positions=y, patch_artist=True,
                    showfliers=False, widths=0.62)
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(colors.get(c, "0.6")); patch.set_alpha(0.9); patch.set_linewidth(0.3)
    for part in ("whiskers", "caps"):
        for ln in bp[part]:
            ln.set_linewidth(0.4)
    for med in bp["medians"]:
        med.set_color("k"); med.set_linewidth(0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(order if ynames else [], fontsize=FS_TAG)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.tick_params(axis="x", labelsize=FS_TICK, pad=1.2)


def _sub_energy_components(fig, gs_cell, a, ck, order, colors):
    """Panel b: the four energy components per cell type, as a 2x2 of horizontal boxplots.
    The cell-type names are printed once per row, on the left column."""
    from matplotlib.ticker import MaxNLocator
    sub = gs_cell.subgridspec(2, 2, wspace=0.32, hspace=0.30)
    axes = []
    for j, (col, lab) in enumerate(ENERGY_COMPONENTS):
        ax = fig.add_subplot(sub[j // 2, j % 2]); axes.append(ax)
        # Cell-type names only under the bottom row: the two rows share one category axis.
        _sub_hbox(ax, _by_type(a, ck, col, order), order, colors, ynames=(j // 2 == 1),
                  vertical=True)
        ax.axhline(0, color="0.5", lw=0.5, ls="--"); _sub_clean(ax)
        ax.set_title(f"{lab} energy", fontsize=FS_TITLE, pad=2.0)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.get_offset_text().set_fontsize(FS_TICK)
    return axes[0]


def _sub_spectra(fig, gs_cell, a, ck, order, colors, cells_per_type=200, top_k=40):
    """Panel c: the leading-mode eigenvalue spectrum of every cell type, 2 x 4, on shared
    axes with no forced symmetry about Re = 0. The cell-type chip sits inside the panel so
    the two rows do not each pay for a title band."""
    E = np.asarray(a.obsm["jacobian_eigenvalues"]); cl = a.obs[ck].astype(str).values
    rng = np.random.default_rng(0)
    per = {}
    for c in order:
        idx = np.where(cl == c)[0]
        if len(idx) == 0:
            per[c] = np.array([], complex); continue
        if len(idx) > cells_per_type:
            idx = rng.choice(idx, cells_per_type, replace=False)
        ev = E[idx]; k = min(top_k, ev.shape[1])
        keep = np.argsort(-ev.real, axis=1)[:, :k]
        per[c] = np.take_along_axis(ev, keep, axis=1).ravel()
    allsel = np.concatenate([v for v in per.values() if len(v)])
    xlo = float(np.nanpercentile(allsel.real, 0.5)); xhi = float(np.nanpercentile(allsel.real, 99.5))
    yhi = float(np.nanpercentile(np.abs(allsel.imag), 99.5)) or 1e-3
    xpad = 0.05 * (xhi - xlo + 1e-9)
    nc, nr = 4, 2
    sub = gs_cell.subgridspec(nr, nc, wspace=0.09, hspace=0.16)
    ax0 = None
    for k2, c in enumerate(order):
        ax = fig.add_subplot(sub[k2 // nc, k2 % nc]); ax0 = ax0 or ax
        sel = per[c]
        if len(sel):
            ax.scatter(sel.real, sel.imag, s=0.7, c=colors.get(c, "0.6"), alpha=0.55,
                       linewidths=0, rasterized=True)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlim(xlo - xpad, xhi + xpad); ax.set_ylim(-yhi * 1.35, yhi * 1.35)
        _sub_tag(ax, c, colors, inside=True)
        ax.tick_params(labelsize=FS_TICK, pad=1.0)
        ax.locator_params(axis="x", nbins=3); ax.locator_params(axis="y", nbins=3)
        if k2 % nc != 0:
            ax.set_yticklabels([])
        if k2 // nc != nr - 1:
            ax.set_xticklabels([])
    return ax0


def _sub_umap_value(fig, ax, emb, vals, cmap, title, poles, symmetric=False, pct=99):
    """One per-cell readout painted on the embedding, with a horizontal colorbar under it
    whose two ends are labeled by meaning rather than by number (FIGURE_SPEC rule 6)."""
    v = np.asarray(vals, float)
    if symmetric:
        m = float(np.nanpercentile(np.abs(v), pct)) or 1.0
        lo, hi = -m, m
    else:
        lo = float(np.nanpercentile(v, 100 - pct)); hi = float(np.nanpercentile(v, pct))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=v, cmap=cmap, vmin=lo, vmax=hi, s=0.7,
                    linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=FS_TITLE, pad=1.8); ax.set_axis_off()
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.062, pad=0.015,
                      aspect=26)
    cb.set_ticks([lo, hi]); cb.set_ticklabels(list(poles))
    cb.ax.tick_params(labelsize=FS_TICK, length=0, pad=1.0)
    cb.outline.set_linewidth(0.4)
    for t, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
        t.set_ha(ha)
    return sc


def _sub_readouts(fig, gs_cell, a, basis, ck, order, colors):
    """Panel d: the four per-cell dynamical readouts on the embedding (top) and summarized
    per cell type (bottom). The summary boxes keep panel b's order and colors, so they need
    no second copy of the cell-type names."""
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    for col, fn in [("_attractor", _per_cell_attractor), ("_oscillation", _per_cell_oscillation),
                    ("_settling", _per_cell_settling)]:
        if col not in a.obs:
            a.obs[col] = fn(a)
    specs = [("jacobian_leading_real", "RdBu_r", "leading eigenvalue (Re)",
              ("stable", "unstable"), True),
             ("_attractor", "PuOr", "attractor index", ("transient", "attractor"), True),
             ("_oscillation", "magma", "oscillation score", ("low", "high"), False),
             ("_settling", "viridis", "settling score", ("fast", "settled"), False)]
    sub = gs_cell.subgridspec(2, 4, wspace=0.14, hspace=0.30, height_ratios=[1.55, 1.0])
    ax0 = None
    for j, (col, cmap, title, poles, sym) in enumerate(specs):
        axu = fig.add_subplot(sub[0, j]); ax0 = ax0 or axu
        _sub_umap_value(fig, axu, emb, a.obs[col].values, cmap, title, poles, symmetric=sym)
        axb = fig.add_subplot(sub[1, j])
        _sub_hbox(axb, _by_type(a, ck, col, order), order, colors, ynames=False,
                  boxen=True, vertical=True)
        axb.axhline(0, color="0.5", lw=0.5, ls="--"); _sub_clean(axb)
        axb.locator_params(axis="y", nbins=3)
        axb.yaxis.get_offset_text().set_fontsize(FS_TICK)
    return ax0


def _sub_circle_positions(genes):
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(genes), endpoint=False)
    return {g: (float(np.cos(t)), float(np.sin(t))) for g, t in zip(genes, ang)}


def _sub_spread(targets, w, lo, hi, gap):
    """Nudge the thumbnails' desired left edges apart so none overlap, keeping them inside
    [lo, hi] and as near their requested positions as possible.

    Sorted by requested position and unsorted afterwards, so the row reads left to right in
    the same order as the clusters it points at and each thumbnail stays over its own.
    """
    t = np.asarray(targets, dtype=float)
    idx = np.argsort(t)
    x = np.clip(t[idx], lo, hi)
    step = w + gap
    for i in range(1, len(x)):                       # push right to clear overlaps
        x[i] = max(x[i], x[i - 1] + step)
    if len(x) and x[-1] > hi:                        # ran off the right edge: pack back left
        x[-1] = hi
        for i in range(len(x) - 2, -1, -1):
            x[i] = min(x[i], x[i + 1] - step)
        x = np.maximum(x, lo)
    out = np.empty_like(x)
    out[idx] = x
    return out


def _tikz_mini(genes, pos, tik_edges, wmax, labels, size, node_lw, edge_lw, lblfont=r"\tiny"):
    """Render one circuit through the TikZ pipeline shared with make_network_figure.

    TikZ, not matplotlib: matplotlib has no flat-bar arrowhead, and its nearest option
    (arrowstyle "-[") draws a bracket that reads as punctuation rather than as repression.
    The TikZ styles give -{Latex} for activation and -{Bar} for repression.
    """
    from make_network_figure import render_tikz, tikz_grn_body
    body = tikz_grn_body(genes, pos, tik_edges, scale=2.4, size=size, node_lw=node_lw,
                         edge_lw=edge_lw, shorten=1.0, off_base=0.10, label_sep=0.5,
                         lblfont=lblfont, italic=True, labels=labels, wmax=wmax,
                         fills={g: "F0EFE9" for g in genes},
                         borders={g: "5A5A5A" for g in genes})   # tikz_grn_body draws white
                                                                 # borders by default, which
                                                                 # made the nodes vanish
    return render_tikz(body, dpi=900)


def _sub_mini_network(ax, genes, pos, edges, vals, smax, act, rep):
    """One cell-type circuit thumbnail, drawn in TikZ. A node per gene at the shared layout,
    and an edge regulator -> target wherever that cell type's mean Jacobian element is
    nonzero; sign sets the head and color, magnitude the width. Elements that are numerically
    zero in this cell type get no edge at all, which is what the fitted model says.

    Nodes are unlabeled here: the gene identities are printed once, at legible size, in the
    circuit key beside the embedding. Widths are normalized by the shared ``smax`` so the
    eight thumbnails stay comparable with one another.
    """
    ax.set_axis_off()
    tik = [(r, t, float(vals[(t, r)])) for (t, r) in edges
           if np.isfinite(vals.get((t, r), np.nan)) and abs(vals.get((t, r), 0.0)) > 1e-12]
    if not tik:
        return
    img = _tikz_mini(genes, pos, tik, wmax=smax, labels=False, size=2.6, node_lw=0.45,
                     edge_lw=(0.45, 1.5))
    if img is not None:
        ax.imshow(img)


def _sub_circuit_key(ax, genes, pos, edges, act, rep):
    """The circuit key: which node is which gene, printed once for all eight thumbnails,
    plus the edge-sign legend. Same TikZ pipeline and the same node layout as the
    thumbnails, so the key reads directly onto them. Gene symbols italic per FIGURE_SPEC.
    """
    from matplotlib.lines import Line2D
    ax.set_axis_off()
    # Nodes only, no edges. tikz_grn_body picks the head and color from the weight's sign, so
    # drawing the key's edges at a nominal +1 rendered every one as a teal activation arrow,
    # asserting a sign the key has no business asserting. Sign and strength belong to the
    # thumbnails, which carry them per cell type; the key answers "which node is which gene".
    img = _tikz_mini(genes, pos, [], wmax=1.0, labels=True, size=2.2, node_lw=0.5,
                     edge_lw=(0.35, 0.0), lblfont=r"\fontsize{5.2}{5.8}\selectfont")
    if img is not None:
        ax.imshow(img)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0 + 0.34 * img.shape[0], y1 - 0.05 * img.shape[0])
    ax.text(0.5, 1.0, "circuit key", transform=ax.transAxes, fontsize=FS_TAG,
            color="0.35", ha="center", va="bottom")
    h = [Line2D([0], [0], color=act, lw=1.1, marker=">", markersize=3, label="activation"),
         Line2D([0], [0], color=rep, lw=1.1, marker="|", markersize=5, label="repression")]
    ax.legend(handles=h, fontsize=FS_TAG, loc="lower center", frameon=False,
              handlelength=1.1, handletextpad=0.35, labelspacing=0.25, borderpad=0.0)


def _sub_networks(fig, a, basis, ck, order, colors, ds, top_mm, h_mm):
    """Panel e: one circuit thumbnail per cell type, in a row directly above the embedding
    and over its own cluster, joined to that cluster's centroid. Overlaying them on the
    embedding, as the poster version does, needs 45 mm networks to stay readable; at page
    width the row plus connectors keeps the same 'which circuit, where' reading with 18 mm
    thumbnails, and the gene identities move into the key at the left."""
    from matplotlib.patches import ConnectionPatch
    from make_network_figure import ACT as act, REP as rep
    edges = [(t, r) for (t, r) in JAC_PAIRS.get(ds, []) if t in a.var_names and r in a.var_names]
    genes = list(dict.fromkeys([g for g in NET_GENES.get(ds, []) if g in a.var_names]
                               + [g for e in edges for g in e]))
    emb = np.asarray(a.obsm[f"X_{basis}"])[:, :2]
    cl = a.obs[ck].astype(str).values

    b = a.copy()
    try:
        import scHopfield as sch
        sch.tl.compute_jacobian_elements(b, gene_pairs=edges, cluster_key=ck, store_in_obs=True)
    except Exception as e:
        print(f"[e] jacobian elements failed: {e}", flush=True)
    cols = {(t, r): _find_element_col(b, t, r) for (t, r) in edges}
    vals = {c: {(t, r): (float(np.nanmean(b.obs[cols[(t, r)]].values[cl == c]))
                         if cols.get((t, r)) else np.nan) for (t, r) in edges} for c in order}
    mags = [abs(v) for d in vals.values() for v in d.values() if np.isfinite(v)]
    smax = float(np.nanpercentile(mags, 95)) if mags else 1.0
    smax = smax or 1.0

    net_mm, tag_mm, gap_mm, emb_w_mm = 18.0, 3.6, 2.4, 96.0
    S, Sy = net_mm / SUB_W_MM, net_mm / SUB_H_MM
    emb_top = top_mm + tag_mm + net_mm + gap_mm
    emb_h = h_mm - (tag_mm + net_mm + gap_mm)
    ax_um = fig.add_axes([(SUB_W_MM - emb_w_mm) / 2 / SUB_W_MM, 1 - (emb_top + emb_h) / SUB_H_MM,
                          emb_w_mm / SUB_W_MM, emb_h / SUB_H_MM])
    ax_um.scatter(emb[:, 0], emb[:, 1], c=[colors.get(c, "0.8") for c in cl], s=1.2, alpha=0.8,
                  linewidths=0, rasterized=True)
    cent = {c: emb[cl == c].mean(0) for c in order if (cl == c).any()}
    for c, xy in cent.items():
        ax_um.scatter([xy[0]], [xy[1]], s=9, c=colors.get(c, "0.6"), edgecolor="k", lw=0.45,
                      zorder=8)
    xr, yr = float(np.ptp(emb[:, 0])), float(np.ptp(emb[:, 1]))
    ax_um.set_xlim(emb[:, 0].min() - 0.03 * xr, emb[:, 0].max() + 0.03 * xr)
    ax_um.set_ylim(emb[:, 1].min() - 0.03 * yr, emb[:, 1].max() + 0.03 * yr)
    ax_um.set_axis_off()

    inv = fig.transFigure.inverted()
    tgt = np.array([inv.transform(ax_um.transData.transform(cent[c]))[0] - S / 2
                    if c in cent else 0.5 for c in order])
    left = _sub_spread(tgt, S, 2.0 / SUB_W_MM, 1.0 - (2.0 + net_mm) / SUB_W_MM, 0.7 / SUB_W_MM)
    pos = _sub_circle_positions(genes)
    ny = 1 - (top_mm + tag_mm + net_mm) / SUB_H_MM
    for k, c in enumerate(order):
        axn = fig.add_axes([left[k], ny, S, Sy])
        _sub_mini_network(axn, genes, pos, edges, vals[c], smax, act, rep)
        _sub_tag(axn, c, colors)
        if c in cent:
            fig.add_artist(ConnectionPatch(
                xyA=(0.5, -0.01), coordsA=axn.transAxes,
                xyB=(cent[c][0], cent[c][1]), coordsB=ax_um.transData,
                color="0.35", lw=0.4, arrowstyle="-|>", mutation_scale=4,
                shrinkA=0.5, shrinkB=3.0, zorder=6))
    key_w = float(left.min() * SUB_W_MM) - 3.0            # the gutter the row leaves free
    ax_key = fig.add_axes([1.5 / SUB_W_MM, 1 - (top_mm + h_mm) / SUB_H_MM,
                           max(key_w, 22.0) / SUB_W_MM, h_mm / SUB_H_MM])
    _sub_circuit_key(ax_key, genes, pos, edges, act, rep)
    return ax_um


def _sub_character_map(ax, df, colors, ds, order):
    """Panel f: the settling-versus-oscillation character map, with the differentiation path
    drawn under the points."""
    path = DIFF_PATH.get(ds) or list(zip(order[:-1], order[1:]))
    for s, t in path:
        if s in df.index and t in df.index:
            ax.annotate("", xy=(df.loc[t, "settling"], df.loc[t, "oscillation_n"]),
                        xytext=(df.loc[s, "settling"], df.loc[s, "oscillation_n"]),
                        arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.55, alpha=0.9,
                                        shrinkA=3.5, shrinkB=3.5, mutation_scale=5), zorder=1)
    texts = []
    for c, r in df.iterrows():
        ax.scatter(r["settling"], r["oscillation_n"], s=22, c=colors.get(c, "0.6"),
                   edgecolor="k", linewidth=0.35, zorder=3)
        texts.append(ax.text(r["settling"], r["oscillation_n"], c, fontsize=FS_TAG, zorder=4))
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=df["settling"].to_numpy(), y=df["oscillation_n"].to_numpy(), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.35), expand=(1.25, 1.5))
    except Exception:
        pass
    ax.set_xlabel("settling score (low input RNA velocity = settled)", fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel("oscillation score\n(rotational, normalized)", fontsize=FS_LABEL, labelpad=1.5)
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.10, 1.14)
    ax.tick_params(labelsize=FS_TICK, pad=1.2); _sub_clean(ax)


def _sub_celltype_key(ax, order, colors):
    """The figure's cell-type key, which panels a, c, e and f all read from."""
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, len(order) + 0.6)
    for i, c in enumerate(order):
        y = len(order) - 0.4 - i
        ax.add_patch(plt.Rectangle((0.02, y - 0.30), 0.13, 0.60, facecolor=colors.get(c, "0.6"),
                                   edgecolor="0.35", lw=0.3, clip_on=False))
        ax.text(0.20, y, c, fontsize=FS_LABEL, va="center", ha="left")


def render_submission(a, ds, ck, order, colors, basis, out_path):
    """Draw the page-size Figure 4 and write it, refusing to save if any type is under 5 pt."""
    from submission_style import figure_for, panel_letter, save
    fig = figure_for("double", height_mm=SUB_H_MM)

    top, h = SUB_BANDS["ab"]
    # x0 is 14 mm, not 0: a 3D axes draws its z label and tick labels well outside its own
    # box, and those are not picked up by the tight bounding box, so at a smaller x0 they
    # were silently cut off at the page edge ("otal energy", "00").
    gs_a = fig.add_gridspec(1, 1, **_sub_rect(top - 1.0, h + 2.0, 14.0, 68.0))
    ax_a = fig.add_subplot(gs_a[0, 0], projection="3d")
    _sub_landscape(ax_a, a, basis, colors, ck)
    gs_b = fig.add_gridspec(1, 1, **_sub_rect(top, h, 74.0, 179.0))
    ax_b = _sub_energy_components(fig, gs_b[0, 0], a, ck, order, colors)
    _sub_letter(fig, 1.0, top - 1.0, "a")
    panel_letter(ax_b, "b", dx=-0.30, dy=1.10)

    top, h = SUB_BANDS["c"]
    gs_c = fig.add_gridspec(1, 1, **_sub_rect(top, h, 8.0, 179.0))
    ax_c = _sub_spectra(fig, gs_c[0, 0], a, ck, order, colors)
    panel_letter(ax_c, "c", dx=-0.13, dy=1.04)
    # va="top" so the label hangs BELOW this y. With va="bottom" it grew upward from
    # top+h+4.2 and landed on the tick labels of the middle two columns.
    fig.text(93.5 / SUB_W_MM, 1 - (top + h + 4.4) / SUB_H_MM, "Re(eigenvalue)",
             fontsize=FS_LABEL, ha="center", va="top")
    fig.text(1.6 / SUB_W_MM, 1 - (top + h / 2) / SUB_H_MM, "Im(eigenvalue)", fontsize=FS_LABEL,
             ha="left", va="center", rotation=90)

    top, h = SUB_BANDS["d"]
    gs_d = fig.add_gridspec(1, 1, **_sub_rect(top, h, 5.0, 179.0))
    ax_d = _sub_readouts(fig, gs_d[0, 0], a, basis, ck, order, colors)
    panel_letter(ax_d, "d", dx=-0.04, dy=1.10)

    top, h = SUB_BANDS["e"]
    gs_f = fig.add_gridspec(1, 1, **_sub_rect(top, h - 1.0, 14.0, 122.0))
    ax_f = fig.add_subplot(gs_f[0, 0])
    _sub_character_map(ax_f, _scores(a, ck, order), colors, ds, order)
    panel_letter(ax_f, "e", dx=-0.10, dy=1.02)
    gs_k = fig.add_gridspec(1, 1, **_sub_rect(top, h, 131.0, 179.0))
    _sub_celltype_key(fig.add_subplot(gs_k[0, 0]), order, colors)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save(fig, out_path)
    print(f"wrote {out_path}")
    plt.close(fig)


def render_submission_circuits(a, ds, ck, order, colors, basis, out_path):
    """The circuits panel as a figure in its own right, drawn by the ORIGINAL panel function.

    Lifted out of Figure 4 because it answers a different question from the rest of that
    figure: energy, spectra and the readout maps describe the fitted dynamics per cell, while
    this describes the regulatory wiring behind them and where it sits along the embedding.

    It calls draw_jacobian_networks() unchanged, so the circuits land exactly where they do in
    the poster figure: on the embedding itself, at the hand-tuned NET_OFFSETS, not in a row
    above it. That placement is the point of the panel.

    The canvas is chosen so the two ratios that set the layout are the same as in the poster
    figure, since draw_jacobian_networks sizes a network as a fraction of FIGURE width and the
    offsets are in units of that size:
        network width / panel width  = 0.120 / 0.89          (fixed by using the same margins)
        network height / panel height = 0.120 * aspect / 0.94 = 0.060 / 0.255  (sets the height)
    """
    from submission_style import figure_for, save, use_submission_style
    from matplotlib.lines import Line2D
    from make_network_figure import ACT as act, REP as rep

    L, R, TOP, BOT = 0.06, 0.95, 0.97, 0.03
    aspect = (0.060 / 0.255) * (TOP - BOT) / 0.120         # 1.843
    use_submission_style()
    fig = figure_for("double", height_mm=180.0 / aspect)
    gs = fig.add_gridspec(1, 1, left=L, right=R, top=TOP, bottom=BOT)
    draw_jacobian_networks(fig, gs[0, 0], a, basis, ck, order, colors, ds)
    fig.legend(handles=[Line2D([0], [0], color=act, lw=1.4, label="activation"),
                        Line2D([0], [0], color=rep, lw=1.4, label="repression")],
               loc="lower left", bbox_to_anchor=(0.012, 0.012), fontsize=6.0,
               frameon=False, ncol=1, handlelength=1.4, handletextpad=0.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save(fig, out_path, check=False)      # the TikZ circuits are rasters, not Text artists
    print(f"wrote {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pancreas")
    ap.add_argument("--variant", default="",
                    help="fit-cache tag to read (e.g. 'bimodal' -> adata_analyzed_bimodal.h5ad); "
                         "the output name is suffixed to sit alongside the canonical figure.")
    ap.add_argument("--submission", action="store_true",
                    help="render the journal-page version (180 mm wide) to --out instead of the "
                         "poster-size figure the per-dataset reports embed.")
    ap.add_argument("--out", default=os.path.join(paths.FIGURES_SPEC, "Figure4.pdf"),
                    help="output PDF for --submission.")
    ap.add_argument("--circuits", action="store_true",
                    help="render the standalone cell-type circuits figure instead of Figure 4.")
    ap.add_argument("--circuits-out", default=os.path.join(paths.FIGURES_SPEC, "Figure4e_circuits.pdf"),
                    help="output PDF for --circuits.")
    args = ap.parse_args()
    ds = args.dataset
    var = args.variant
    from config import DATASETS
    cfg = DATASETS[ds]
    ck = cfg["cluster_key"]
    suf = f"_{var}" if var else ""
    a = ad.read_h5ad(f"{paths.REPORTS}/{ds}/data/adata_analyzed{suf}.h5ad")
    basis = basis_of(a)
    order, colors = _order_colors(a, ck, cfg)

    if args.circuits:
        render_submission_circuits(a, ds, ck, order, colors, basis, args.circuits_out)
        return
    if args.submission:
        render_submission(a, ds, ck, order, colors, basis, args.out)
        return
    else:
        use_style(9)
        os.makedirs(OUT, exist_ok=True)
        fig = plt.figure(figsize=(16.0, 32.0))
        L, R = 0.06, 0.95
        proc = f"  ({PROCESS[ds]})" if ds in PROCESS else ""
        vtag = f"  [{var} sigmoid fit]" if var else ""
        fig.text(0.05, 0.996, f"Energy landscape and Jacobian dynamics of {ds}{proc}{vtag}", ha="left", va="top",
                 fontsize=15, fontweight="bold")

        def label(ax, text, dx=0.045, dy=0.012):
            # render ONLY the panel letter; the description stays in the call as a caption reference
            bb = ax.get_position()
            fig.text(bb.x0 - dx, bb.y1 + dy, text.split()[0], fontweight="bold", fontsize=13, va="bottom",
                     ha="left")

        # a (3D landscape) + b (energy 2x2) on the top band
        gs_top = fig.add_gridspec(1, 2, top=0.965, bottom=0.815, left=L, right=R, width_ratios=[1.0, 1.3],
                                  wspace=0.20)
        ax_a = fig.add_subplot(gs_top[0, 0], projection="3d")
        draw_energy_landscape(ax_a, a, basis, colors, ck)
        ax_b0 = draw_energy_components(fig, gs_top[0, 1], a, ck, order, colors)

        # c: eigenvalue spectra split by cell type (2xN), moved up to sit right after a/b
        gs_c = fig.add_gridspec(1, 1, top=0.788, bottom=0.650, left=L, right=R)
        ax_c0 = draw_spectra_by_type(fig, gs_c[0, 0], a, ck, order, colors)

        # d: combined dynamical readouts (leading Re, attractor, oscillation, settling; UMAP + violin rows)
        gs_d = fig.add_gridspec(1, 1, top=0.620, bottom=0.435, left=L, right=R)
        ax_d0 = draw_dynamical_readouts(fig, gs_d[0, 0], a, basis, ck, order, colors)

        # e: per-cell-type regulatory mini-networks anchored to the embedding + edge-sign legend
        gs_e = fig.add_gridspec(1, 1, top=0.405, bottom=0.150, left=L, right=R)
        ax_e0 = draw_jacobian_networks(fig, gs_e[0, 0], a, basis, ck, order, colors, ds)
        from matplotlib.lines import Line2D
        from make_network_figure import ACT as _ACTC, REP as _REPC
        fig.legend(handles=[Line2D([0], [0], color=_ACTC, lw=2.6, label="activation"),
                            Line2D([0], [0], color=_REPC, lw=2.6, label="repression")],
                   loc="upper right", bbox_to_anchor=(R, 0.428), fontsize=7.2, frameon=False, ncol=2)

        # f: settling vs oscillation character map + differentiation arrows (panel h removed; the attractor
        # ranking now lives in panel d)
        df = _scores(a, ck, order)                               # no cell-type legend: names are on the scatter
        gs_f = fig.add_gridspec(1, 1, top=0.120, bottom=0.030, left=L, right=R)
        ax_f = fig.add_subplot(gs_f[0, 0]); draw_stability_oscillation(ax_f, df, colors, ds=ds, order=order)

        import string
        lt = list(string.ascii_lowercase)
        label(ax_a, f"{lt[0]}   Total-energy 3D landscape", dx=0.04)
        label(ax_b0, f"{lt[1]}   Energy by cell type (all components)")
        label(ax_c0, f"{lt[2]}   Eigenvalue spectra by cell type (leading modes)")
        label(ax_d0, f"{lt[3]}   Local stability, attractor index, oscillation, settling")
        label(ax_e0, f"{lt[4]}   Per-cell-type regulatory mini-networks on the embedding")
        label(ax_f, f"{lt[5]}   Settling vs oscillation character map")

        NAME = f"energy-jacobian-{ds}{suf}"
        outdir = OUT if ds == "pancreas" else f"{OUT}/extended"
        os.makedirs(outdir, exist_ok=True)
        save(fig, f"{outdir}/{NAME}", formats=("pdf", "png"))
        print(f"wrote {outdir}/{NAME}.pdf + .png")
        plt.close(fig)


if __name__ == "__main__":
    main()
