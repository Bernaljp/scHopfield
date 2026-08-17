"""Composite figure: in-silico perturbation dynamics of a differentiation process
(default = pancreatic endocrinogenesis). Reads the cached ODE analyses from
`_perturb_dynamics_compute.py` (reports/<ds>/data/perturb_dynamics.pkl) so it iterates cheaply.

  a    embedding + RNA-velocity streamlines (the report's dynamo streamline plot)
  b,c  driver-score scatter per lineage decision (Pareto rounds; discovery = threshold wings)
  d    per-cell fate-probability shift map after each KO (projection-free spatial decision)
  e    single-KO fate-probability shift per lineage pair (bar chart; biological figure only)
  f    dose-response of the fate-probability shift (dose 0 = the panel-e KO), the pair's own TFs,
       with the two half-planes shaded by which lineage they favor
  g    short-time cascade relative to WT: mean |x_KO(t) - x_WT(t)| per cell type vs ODE time

The fate-probability readouts (d, e, f) are projection-free: each measures a change in terminal-state
absorption probability, so a pure sink gene (no out-edges, e.g. Malat1) gives ~0 by construction.
They are documented in the Methods (fate-probability lineage effect); panel g's WT-relative cascade
(|x_KO(t) - x_WT(t)|) and the fate readouts are all defined there.

Run:  python reproducibility/make_perturbation_dynamics.py [--dataset pancreas] [--mode biological|discovery]

``--submission`` renders the SAME nine panels on one journal page
(180 x <=247 mm, no type below 5 pt) through ``submission_style.figure_for``/``save``, and
writes to reproducibility/figures/submission/Figure5.pdf. It is an additional code path: without the
flag nothing about the poster-sized figure the per-dataset reports use changes.
"""
from __future__ import annotations
import argparse, os, sys, pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
from paper_plot_style import use_style, save, PALETTE, _resolve_sans   # noqa: E402
import submission_style as sub                                   # noqa: E402  (journal page rules)
import anndata as ad                                             # noqa: E402
from sections import _flow_grid                                  # noqa: E402  (report's 5.1.3 flow plotter)

_SANS, _CHOSEN, _LAST_RESORT = _resolve_sans()
_MATHSET = "dejavusans" if _LAST_RESORT else "stixsans"

OUT = paths.FIGURES
TF_PALETTE = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
              PALETTE["vermillion"], PALETTE["purple"], PALETTE["sky"]]
GRP = {"A": "#B4423A", "B": "#2B5D8A"}                           # lineage A warm / B cool
PROCESS = {"pancreas": "endocrinogenesis", "paul15": "hematopoiesis",
           "paul15_coarse": "hematopoiesis", "dynamo_hematopoiesis": "hematopoiesis",
           "schwann": "neural crest", "murine_nc": "neural crest", "human_limb": "myogenesis"}

# Evidence tier of each DISCOVERED (data-driven) candidate for its lineage decision, for the discovery
# figure's classification table. Tiers verified with the lit-novelty skill (verify_papers.py); a gene
# with no entry falls back to "?" (its literature has not been checked yet). T1 known and validated in
# this decision; T2 evidenced (expression/binding) but the fate role not dissected; T3 novel for this
# process (surfaced here, a candidate to explore). Unlike the curated double-KO pairs (mostly T1), the
# discovered candidates span T1..T3, which is where genuine novelty is expected.
DISCOVERY_TIERS = {
    "pancreas": {
        "Prox1": ("T1", "endocrine-progenitor / secondary-transition TF; sets beta maturation (Prox1, Diabetes 2016)"),
        "Foxa3": ("T2", "Foxa paralog tied to Ngn3-high progenitors; Foxa1/2 carry the essential roles"),
        "Tead2": ("T2", "TEAD/YAP gate the progenitor->endocrine switch (Cebola 2015); TEAD2 role not dissected"),
        "Vdr": ("T2", "maintains beta identity / blocks beta dedifferentiation (VDR, Diabetes 2020); fate role open"),
        "Mef2a": ("T2", "negative regulator of beta maturation; overexpression drives dedifferentiation (recent)"),
        "Creb3l1": ("T3", "ER-stress / secretory bZIP in beta cells; no known role in the alpha/beta fate choice"),
    },
}
TIER_COLOR = {"T1": PALETTE["green"], "T2": PALETTE["orange"], "T3": PALETTE["vermillion"], "?": "0.6"}


def draw_disc_tier_table(ax, ds, groups):
    """Discovery figure classification table (panel i), laid out in TWO columns to use the full width and
    keep the rows uncramped: each lineage decision's discovered candidates form one column (a
    single-decision dataset splits its genes into two columns). Per row: gene, evidence tier (T1/T2/T3
    colored chip), and the one-line verified evidence; the decision is the column title; alternating row
    shading; a legend at the foot."""
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    dt = DISCOVERY_TIERS.get(ds, {})
    items = [(gname, [(g, *(dt.get(g, ("?", "needs literature")))) for g in genes])
             for gname, genes in groups.items()]
    if len(items) == 1:                                          # single decision -> split its genes in two
        gname, gl = items[0]; half = (len(gl) + 1) // 2
        blocks = [(gname, gl[:half]), (gname, gl[half:])]
    else:
        blocks = items[:2]
    for bi, (gname, gl) in enumerate(blocks):
        if not gl:
            continue
        x0 = 0.0 if bi == 0 else 0.52
        gx, tx, ex = x0, x0 + 0.085, x0 + 0.14
        ax.text(x0, 0.95, gname, fontsize=7.6, fontweight="bold", color="0.35", va="center")
        for x, h in [(gx, "gene"), (tx, "tier"), (ex, "evidence")]:
            ax.text(x, 0.85, h, fontsize=6.4, fontweight="bold", va="center")
        ax.plot([x0, x0 + 0.47], [0.815, 0.815], color="0.5", lw=0.6)
        top, bot = 0.75, 0.14; m = len(gl); dy = (top - bot) / max(m, 1)
        for i, (g, t, note) in enumerate(gl):
            yc = top - (i + 0.5) * dy
            if i % 2:
                ax.add_patch(plt.Rectangle((x0 - 0.006, yc - dy / 2), 0.49, dy, color="0.955",
                                           zorder=0, linewidth=0))
            ax.text(gx, yc, g, fontsize=6.9, va="center")
            ax.text(tx, yc, t, fontsize=6.9, fontweight="bold", va="center", color=TIER_COLOR.get(t, "0.6"))
            ax.text(ex, yc, note, fontsize=5.6, va="center", color="0.3")
    ax.text(0.0, 0.03, "T1 known and validated    |    T2 evidenced but fate role not dissected    |    "
            "T3 novel for this process (candidate to explore)", fontsize=5.9, color="0.35",
            style="italic", va="center")


def draw_fate_map(ax, emb, vals, vmax, title, eps, cmap="RdBu_r", s_bg=3, s_fg=6,
                  title_fs=8.5, title_pad=None, raster=False):
    """Per-cell map: each cell colored by a signed per-cell value on a symmetric diverging scale; cells
    with |value| < eps stay gray (committed/inert) so only the cells the knockout moves show color.
    ``cmap`` distinguishes DIFFERENT quantities: RdBu_r for the propagated fate shift (the result),
    PuOr for the first-order commitment push (the prediction). Strong values drawn on top.
    The size/type keywords default to the poster figure's values, so the report figure is unchanged;
    the submission page passes smaller marks and rasterizes the two dense scatter layers."""
    ax.scatter(emb[:, 0], emb[:, 1], c="0.85", s=s_bg, linewidths=0, zorder=0,
               rasterized=raster)                                             # all cells, faint gray
    m = np.abs(vals) >= eps
    idx = np.where(m)[0][np.argsort(np.abs(vals[m]))]                         # colored cells, strong on top
    sc = ax.scatter(emb[idx, 0], emb[idx, 1], c=vals[idx], cmap=cmap, vmin=-vmax, vmax=vmax,
                    s=s_fg, linewidths=0, zorder=1, rasterized=raster)
    ax.set_title(title, fontsize=title_fs, pad=title_pad); ax.set_axis_off()
    return sc


def _grid_arrows(ax, adm, flow_key, basis, color="#333333", n_grid=25, min_mass=25, size=1.0):
    """The grid-averaged displacement quiver of the report's `_flow_grid` (same outlier-clip + scale),
    WITHOUT the cluster cell coloring, so it can be overlaid on an inner-product-colored scatter."""
    from scHopfield.tools.flow import calculate_grid_flow
    Fl = np.asarray(adm.obsm[flow_key])[:, :2].astype(float)
    mag = np.linalg.norm(Fl, axis=1); pos = mag > 0
    if pos.any():
        cap = float(np.percentile(mag[pos], 99))
        if cap > 0:
            Fl = Fl * np.minimum(1.0, cap / (mag + 1e-12))[:, None]
    adm.obsm[flow_key] = Fl
    grid = calculate_grid_flow(adm, flow_key=flow_key, basis=basis, n_grid=n_grid, min_mass=min_mass, n_jobs=4)
    coords = np.asarray(grid["grid_coords"]); valid = ~np.asarray(grid["mass_filter"])
    gflow = np.asarray(grid["grid_flow"])
    if not valid.any():
        return
    gmag = np.linalg.norm(gflow[valid], axis=1)
    span = float(np.mean(coords.max(0) - coords.min(0))); cell = span / max(n_grid, 1)
    ref = float(np.percentile(gmag[gmag > 0], 90)) if np.any(gmag > 0) else 1.0
    qscale = ref / (0.9 * size * cell) if cell > 0 else 1.0
    ax.quiver(coords[valid, 0], coords[valid, 1], gflow[valid, 0], gflow[valid, 1],
              color=color, scale=qscale, scale_units="xy", angles="xy", width=0.005, zorder=2)


def draw_flow_ip(ax, C, g, wt_flow, wt_ode_flow, basis, ngrid=30, min_count=4):
    """Flow row: BOTH the arrows and the color are the KO-SPECIFIC residual displacement
    $(\\Delta x_{KO} - \\Delta x_{WT})$ (projected), so they show the same per-knockout vector and
    agree with each other. Black arrows = the grid-averaged residual flow; the GRID-SMOOTHED PuOr
    field = the cosine alignment of that residual with the WT developmental velocity (+ aligned /
    - opposed with development). The 2-D embedding cosine (it colors an embedding visualization); the
    projection-free fate metric carries the quantitative claims. PuOr, not the fate map's RdBu_r."""
    emb = np.asarray(C["emb"]); flow = np.asarray(C["ko_flow"][g])[:, :2]
    resid = flow - wt_ode_flow                                 # KO-specific residual: arrows AND color
    rmag = np.linalg.norm(resid, axis=1)
    ip = (resid * wt_flow).sum(1) / (rmag * np.linalg.norm(wt_flow, axis=1) + 1e-12)
    x, y = emb[:, 0], emb[:, 1]
    xe = np.linspace(x.min(), x.max(), ngrid + 1); ye = np.linspace(y.min(), y.max(), ngrid + 1)
    ix = np.clip(np.digitize(x, xe) - 1, 0, ngrid - 1); iy = np.clip(np.digitize(y, ye) - 1, 0, ngrid - 1)
    S = np.zeros((ngrid, ngrid)); Wg = np.zeros((ngrid, ngrid)); N = np.zeros((ngrid, ngrid))
    np.add.at(S, (ix, iy), ip * rmag); np.add.at(Wg, (ix, iy), rmag); np.add.at(N, (ix, iy), 1.0)
    grid = np.where(N >= min_count, S / np.maximum(Wg, 1e-12), np.nan)   # residual-weighted mean alignment
    import numpy.ma as ma
    from scipy.ndimage import gaussian_filter
    V = np.nan_to_num(grid, nan=0.0); Mk = (~np.isnan(grid)).astype(float)
    Vs = gaussian_filter(V * Mk, sigma=1.0); Ms = gaussian_filter(Mk, sigma=1.0)
    smooth = np.where(Ms > 0.2, Vs / np.maximum(Ms, 1e-6), np.nan)
    cmap = plt.get_cmap("PuOr").copy(); cmap.set_bad(alpha=0.0)
    ax.scatter(x, y, c="0.9", s=2, linewidths=0, zorder=0)              # faint manifold context
    pc = ax.imshow(ma.masked_invalid(smooth.T), cmap=cmap, vmin=-1, vmax=1, origin="lower",
                   extent=[x.min(), x.max(), y.min(), y.max()], interpolation="bilinear",
                   aspect="auto", zorder=1)
    adm = min_adata_flow(C, g)                                 # arrows = the SAME residual vector
    adm.obsm[f"perturbation_flow_{basis}"] = resid.astype(float)
    _grid_arrows(ax, adm, f"perturbation_flow_{basis}", basis, color="k")
    ax.set_axis_off()
    return pc


def min_adata_flow(C, gene):
    """Minimal AnnData carrying the KO displacement flow for one gene, so the report's own grid-flow
    plotter (`sections._flow_grid`, used in report section 5.1.3) can render it exactly as in the
    reports: obsm has the embedding and the per-cell perturbation flow, obs has the cluster labels."""
    n = len(C["clusters"]); basis = C["basis"]; ck = C["cluster_key"]
    adm = ad.AnnData(np.zeros((n, 1), dtype=np.float32),
                     obs=pd.DataFrame({ck: pd.Categorical(C["clusters"])},
                                      index=[str(i) for i in range(n)]))
    adm.obsm[f"X_{basis}"] = np.asarray(C["emb"]).astype(float)
    adm.obsm[f"perturbation_flow_{basis}"] = np.asarray(C["ko_flow"][gene])[:, :2].astype(float)
    return adm


DISCOVERY_Q = 95           # percentile per lineage axis for the discovery figure's threshold lines


PARETO_FS = dict(bg_s=7, front_s=40, disc_s=62, lab=7, disc_lab=7.5, axlab=8, tick=7, title=9,
                 n_lab=8, raster=False)


def draw_pareto(ax, csv_path, focus, An, Bn, discovery=False, q=DISCOVERY_Q, fs=None):
    """Driver-score scatter for the two lineages of a decision (from the report's driver_scores
    CSV). Two styles:
      - biological (default): grey = all genes, viridis_r = the successive Pareto fronts, the
        featured known regulators labeled in bold; returns the colored scatter for a shared colorbar.
      - discovery: dashed lines at the qth percentile of each lineage's driver score, with only the
        `focus` (selected discovery) genes highlighted; returns None (no colorbar).
    ``fs`` overrides mark and type sizes (PARETO_FS holds the poster defaults, so the report
    figure is untouched); the submission page passes smaller marks, 5 pt labels and rasterizes
    the 2000-point background."""
    f = dict(PARETO_FS); f.update(fs or {})
    try:
        from adjustText import adjust_text
    except Exception:
        adjust_text = None
    df = pd.read_csv(csv_path, index_col=0)
    ax.scatter(df.score_A, df.score_B, s=f["bg_s"], c="#d3d3d3", linewidths=0, zorder=1,
               rasterized=f["raster"])
    sc = None
    if discovery:
        thrA = float(np.percentile(df.score_A.values, q)); thrB = float(np.percentile(df.score_B.values, q))
        ax.axvline(thrA, color="0.6", ls="--", lw=0.8, zorder=1)
        ax.axhline(thrB, color="0.6", ls="--", lw=0.8, zorder=1)
        foc = [g for g in focus if g in df.index]
        ax.scatter(df.loc[foc, "score_A"], df.loc[foc, "score_B"], s=f["disc_s"], c="#8f3200",
                   edgecolor="k", linewidth=0.6, zorder=3)   # darker vermillion for the selected wing genes
        texts = [ax.text(df.loc[g, "score_A"], df.loc[g, "score_B"], g, fontsize=f["disc_lab"],
                         fontweight="bold", color="k", zorder=4) for g in foc]
    else:
        pr = df.dropna(subset=["pareto_rank"])
        sc = ax.scatter(pr.score_A, pr.score_B, s=f["front_s"], c=pr.pareto_rank, cmap="viridis_r",
                        vmin=0, vmax=5, edgecolor="k", linewidth=0.3, zorder=2)
        lab = list(dict.fromkeys([g for g in focus if g in df.index]
                                 + list(pr.sort_values("pareto_rank").head(f["n_lab"]).index)))
        texts = [ax.text(df.loc[g, "score_A"], df.loc[g, "score_B"], g, fontsize=f["lab"],
                         fontweight="bold" if g in focus else "normal",
                         color="k" if g in focus else "0.35") for g in lab if g in df.index]
    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4),
                    expand=(1.1, 1.3))
    ax.set_xlabel(f"driver score ({An})", fontsize=f["axlab"])
    ax.set_ylabel(f"driver score ({Bn})", fontsize=f["axlab"]); ax.tick_params(labelsize=f["tick"])
    ax.set_title(f"{An} vs {Bn}", fontsize=f["title"])
    return sc


BIAS_FS = dict(tick=6.5, ylab=8, title=9, legend=6.3, edge_lw=1.3, headroom=0.22, rot=60,
               leg_kw={})


def draw_bias(ax, sb, focus, An, Bn, fs=None):
    """Per-KO FATE-PROBABILITY shift toward lineage A (+) / B (-): the change in terminal-state
    absorption probability (WT vs KO) aggregated per lineage arm. Fate-based replacement for the old
    projected-cosine lineage bias, which was fooled by the high-expression/sink projection artifact
    (a pure sink like Malat1 now gives exactly 0). One bar per candidate; the pair's TFs outlined.
    ``fs`` overrides the type sizes and the headroom kept free for the legend (BIAS_FS = the
    poster defaults)."""
    f = dict(BIAS_FS); f.update(fs or {})
    order = sb.sort_values()
    ax.bar(range(len(order)), order.values,
           color=[GRP["A"] if v >= 0 else GRP["B"] for v in order.values],
           edgecolor=["k" if g in focus else "none" for g in order.index],
           linewidth=[f["edge_lw"] if g in focus else 0 for g in order.index])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order.index, rotation=f["rot"], ha="right", fontsize=f["tick"])
    for t, g in zip(ax.get_xticklabels(), order.index):
        if g in focus:
            t.set_fontweight("bold")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, 0.0) + (hi - lo) * f["headroom"])  # headroom above the bars for the legend
    ax.set_ylabel("fate shift", fontsize=f["ylab"])
    ax.set_title(f"{An} vs {Bn}", fontsize=f["title"])
    ax.legend(handles=[Patch(fc=GRP["A"], label=f"shifts toward {An}"),
                       Patch(fc=GRP["B"], label=f"shifts toward {Bn}"),
                       Patch(fc="0.7", ec="k", lw=f["edge_lw"], label="the pair's TFs")],
              fontsize=f["legend"], loc="upper left", framealpha=0.9, **f["leg_kw"])


DOSE_FS = dict(ms=4, lw=None, side=7, axlab=8, ylab=8, title=9, legend=7, tick=None,
               leg_loc="lower center", leg_kw={})


def draw_fate_dose(ax, C, An, Bn, genes, tf_color, fs=None):
    """Fate split-fraction shift vs dose (panel f). For each gene, the mean shift in the A-vs-B fate
    split fraction as the gene is held from 0 (knockout) through its natural level (=1) to
    overexpression; dose=0 is exactly the panel-e KO value, so panel e is the dose-0 slice of this
    panel. This is the fate-based replacement for the projected-cosine dose-response. Half-planes
    shaded by the sign of the shift; the dashed vertical line marks dose=1 (natural, unperturbed).
    ``fs`` overrides the mark and type sizes (DOSE_FS = the poster defaults)."""
    f = dict(DOSE_FS); f.update(fs or {})
    dd = C.get("fate_dose", {}).get((An, Bn), {})
    for g in genes:
        dr = dd.get(g)
        if dr is None or not len(dr):
            continue
        ax.plot(dr.level_frac, dr.fate_bias, "-o", ms=f["ms"], lw=f["lw"], color=tf_color[g], label=g)
    ax.axvline(1, color="0.5", ls="--", lw=0.8); ax.axhline(0, color="k", lw=0.7)
    lo, hi = ax.get_ylim()
    ax.axhspan(0, max(hi, 0), color=GRP["A"], alpha=0.07, zorder=0)   # shade by the actual shift sign
    ax.axhspan(min(lo, 0), 0, color=GRP["B"], alpha=0.07, zorder=0)
    ax.set_ylim(lo, hi)
    ax.text(0.015, 0.97, f"toward {An}", transform=ax.transAxes, ha="left", va="top",
            fontsize=f["side"], color=GRP["A"], style="italic")
    ax.text(0.015, 0.03, f"toward {Bn}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=f["side"], color=GRP["B"], style="italic")
    ax.set_xlabel("expression (fraction of natural max; 1 = unperturbed)", fontsize=f["axlab"])
    ax.set_ylabel("fate shift", fontsize=f["ylab"])
    ax.set_title(f"{An} vs {Bn}", fontsize=f["title"])
    if f["tick"]:
        ax.tick_params(labelsize=f["tick"])
    ax.legend(fontsize=f["legend"], ncol=3, framealpha=0.9, loc=f["leg_loc"], **f["leg_kw"])


CASC_FS = dict(lw=1.5, title=9, weight="bold", axlab=8, tick=6.5, legend=5.8, title_pad=None,
               leg_kw={})


def draw_cascade(ax, casc, cond, colors, order, ymax, show_legend=False, show_ylabel=False,
                 fs=None):
    """One knockout's short-time cascade: mean |x_KO(t) - x_WT(t)| per cell type against ODE time.
    ``fs`` overrides line and type sizes (CASC_FS = the poster defaults)."""
    f = dict(CASC_FS); f.update(fs or {})
    sub = casc[casc.perturbation == cond]
    for cl in order:
        row = sub[sub.cluster == cl].sort_values("t")
        if len(row):
            ax.plot(row.t, row.mean_abs_delta, "-", lw=f["lw"], color=colors.get(cl, "0.6"), label=cl)
    ax.set_ylim(0, ymax * 1.05); ax.margins(x=0.02)
    ax.set_title(cond, fontsize=f["title"], fontweight=f["weight"], pad=f["title_pad"])
    ax.set_xlabel("ODE time", fontsize=f["axlab"]); ax.tick_params(labelsize=f["tick"])
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    if show_legend:
        ax.legend(fontsize=f["legend"], ncol=1, loc="upper left", framealpha=0.85, **f["leg_kw"])


RESP_FS = dict(title=8, blank=6, ytick=6, xtick=5.5, title_pad=None, exponent=None)


def draw_ko_response(ax, resp, reg, gene, topn=6, fs=None):
    """Predicted first-order KO response (Jacobian), gene space: the regulator targets a knockout is
    predicted to up- (red, production rises = loss of repression) or down- (blue) regulate,
    r_i = -J_ig x_g, restricted to regulators (out-strength > 0) so high-in-strength sinks do not
    dominate. Top ``topn`` up + top ``topn`` down targets as a horizontal bar.
    ``fs['exponent']``, used only on the submission page, divides the bars by a common power of ten
    and prints that power as the axis label, so three plain tick labels fit a 17 mm wide panel
    instead of seven-character decimals (RESP_FS = the poster defaults)."""
    f = dict(RESP_FS); f.update(fs or {})
    ax.set_title(f"{gene} KO", fontsize=f["title"], pad=f["title_pad"])

    def _blank(msg):
        # Never leave a bare title over empty axes (that reads as a broken panel): say why it is empty.
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=f["blank"], style="italic", color="0.45")
        ax.set_axis_off()

    if resp is None or not len(resp):
        _blank("no Jacobian response\ncached"); return
    s = resp
    if reg is not None:
        s = s[reg.reindex(s.index).fillna(0).values > 0]
    s = s[s.abs() > 0]
    if not len(s):
        _blank("no regulator targets"); return
    top = pd.concat([s.nlargest(topn), s.nsmallest(topn)]).drop_duplicates().sort_values()
    vals = top.values
    if f["exponent"]:
        k = int(np.floor(np.log10(np.max(np.abs(vals))))) if np.max(np.abs(vals)) > 0 else 0
        vals = vals / (10.0 ** k)
        ax.set_xlabel(f"$\\times10^{{{k}}}$", fontsize=f["exponent"], labelpad=1)
    ax.barh(range(len(top)), vals,
            color=[PALETTE["vermillion"] if v > 0 else PALETTE["blue"] for v in vals],
            edgecolor="none")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=f["ytick"])
    ax.tick_params(axis="x", labelsize=f["xtick"]); ax.margins(y=0.02)
    if f["exponent"]:
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))


def draw_gene_umap_band(fig, tfs, npair, pair_genes, lps, valuemap, emb, top, bot, L, R, col_w,
                        cbar_title, cmap="RdBu_r", header_dy=0.008):
    """A row of per-gene UMAP maps (draw_fate_map), grouped by lineage decision, each group on its own
    symmetric scale with a colorbar named by the two lineages (committed-stable cells gray). ``cmap``
    keeps the push PREDICTION (PuOr) visually distinct from the fate RESULT (RdBu_r) even though both
    are 'toward A / toward B'. ``valuemap`` = {(An,Bn): {gene: per-cell array}}. Returns the first axis."""
    gs = fig.add_gridspec(1, len(tfs), top=top, bottom=bot, left=L, right=R, wspace=0.06)
    ax0 = None; i0 = 0
    for k in range(npair):
        _, _, An, Bn = lps[k]
        genes_k = pair_genes(k)
        vk = valuemap.get((An, Bn), {})
        pooled = [np.abs(np.asarray(vk[g])) for g in genes_k if g in vk]
        vmax = float(np.percentile(np.concatenate(pooled), 99)) if pooled else 1e-6
        vmax = vmax or 1e-6
        eps = max(0.06 * vmax, 1e-4)
        sc_k = None
        for gi, g in enumerate(genes_k):
            ax = fig.add_subplot(gs[0, i0 + gi]); ax0 = ax0 or ax
            vals = vk.get(g)
            if vals is not None:
                sc_k = draw_fate_map(ax, emb, np.asarray(vals), vmax, f"{g} KO", eps, cmap=cmap)
            else:
                ax.set_axis_off()
        cx = L + (i0 + len(genes_k) / 2) * col_w
        fig.text(cx, top + header_dy, f"{An} vs {Bn}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="0.4")
        if sc_k is not None:
            cb_w = min(0.16, len(genes_k) * col_w * 0.7)
            cbax = fig.add_axes([cx - cb_w / 2, bot - 0.012, cb_w, 0.006])
            cb = fig.colorbar(sc_k, cax=cbax, orientation="horizontal")
            cb.set_ticks([-vmax, 0, vmax]); cb.ax.set_xticklabels(["", "", ""]); cb.ax.tick_params(length=2)
            cbax.set_title(f"{cbar_title}  (±{vmax:.2g})", fontsize=5.6)
            cbax.text(-0.03, 0.5, Bn, transform=cbax.transAxes, ha="right", va="center", fontsize=6)
            cbax.text(1.03, 0.5, An, transform=cbax.transAxes, ha="left", va="center", fontsize=6)
        i0 += len(genes_k)
    return ax0


# ======================================================================================
# Submission layout (journal page: 180 mm wide, <= 247 mm tall with the
# legend area, nothing below 5 pt).
#
# The poster figure hands each band the full width of a 394 mm canvas, so shrinking it to a
# page would land the type at 1.3 to 3.3 pt. What follows is a re-layout, not a rescale: the
# same nine panels a-i, the same colormaps, the same data, on a millimeter grid. The changes
# that buy the height are listed in the module docstring of the --submission path below.
# ======================================================================================
SUB_W, SUB_H = 180.0, 244.0                # the page canvas in mm
SUB_L, SUB_R = 8.0, 178.0                  # the column band the six-wide rows fill
SUB_FS = dict(title=6.5, axlab=6.0, tick=5.0, small=5.0)


def _mm_axes(fig, x, y, w, h):
    """One axes placed in millimeters measured from the TOP LEFT of the submission canvas."""
    return fig.add_axes([x / SUB_W, 1.0 - (y + h) / SUB_H, w / SUB_W, h / SUB_H])


def _mm_text(fig, x, y, s, **kw):
    """Figure-level text at (x, y) millimeters from the top left."""
    return fig.text(x / SUB_W, 1.0 - y / SUB_H, s, **kw)


def _letter(fig, x, y, ch):
    """One panel letter: bold lowercase at the spec's 8 pt, above the panel's top left corner."""
    return _mm_text(fig, x, y, ch, fontsize=sub.TYPE_PANEL_LETTER, fontweight="bold",
                    ha="left", va="bottom")


def _mm_slots(n, w, x0=SUB_L, x1=SUB_R):
    """``n`` equal columns across [x0, x1], each holding an ``w`` mm wide axes, centered in its
    slot. The knockout rows d, e, f and g share this grid so their columns line up."""
    step = (x1 - x0) / n
    return [(x0 + i * step + (step - w) / 2.0, w) for i in range(n)]


def draw_velocity_stream(ax, emb, clusters, colors, flow, n_grid=30, density=0.9, pt=1.2,
                         lw=(0.2, 1.0)):
    """Panel a, drawn natively at submission size: cell-type-colored cells under speed-scaled
    streamlines of the input RNA velocity. The report's stored PNG is not reused here because its
    baked-in cluster labels would land near 2 pt once the panel is a third of a page wide, and
    raster type cannot be checked by save(). Same construction as the report's streamline panel
    (reproducibility/sections.py::_streamplot): Gaussian-weighted KNN interpolation of the
    per-cell flow onto a regular grid, with the sparse regions blanked."""
    from sklearn.neighbors import NearestNeighbors
    emb = np.asarray(emb)[:, :2]
    F = np.asarray(flow)[:, :2]
    cvec = [colors.get(str(c), "#cccccc") for c in np.asarray(clusters).astype(str)]
    ax.scatter(emb[:, 0], emb[:, 1], c=cvec, s=pt, alpha=0.55, linewidths=0, zorder=1,
               rasterized=True)
    xmin, xmax = float(emb[:, 0].min()), float(emb[:, 0].max())
    ymin, ymax = float(emb[:, 1].min()), float(emb[:, 1].max())
    gx = np.linspace(xmin, xmax, n_grid); gy = np.linspace(ymin, ymax, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    nn = NearestNeighbors(n_neighbors=min(60, len(emb))).fit(emb)
    dist, idx = nn.kneighbors(np.column_stack([GX.ravel(), GY.ravel()]))
    sigma = np.mean([xmax - xmin, ymax - ymin]) / n_grid
    wgt = np.exp(-(dist ** 2) / (2 * sigma ** 2)); wsum = wgt.sum(1)
    U = (wgt * F[idx, 0]).sum(1) / (wsum + 1e-9)
    V = (wgt * F[idx, 1]).sum(1) / (wsum + 1e-9)
    low = wsum < np.percentile(wsum, 35)                      # blank out the sparse regions
    U[low] = 0.0; V[low] = 0.0
    U = U.reshape(n_grid, n_grid); V = V.reshape(n_grid, n_grid)
    speed = np.sqrt(U ** 2 + V ** 2)
    ax.streamplot(gx, gy, U, V, density=density, color="#222222", arrowsize=0.35, zorder=2,
                  linewidth=lw[0] + (lw[1] - lw[0]) * speed / (np.nanmax(speed) + 1e-9))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_axis_off()


def sub_umap_row(fig, lps, pair_genes, valuemap, emb, slots, y_map, h_map, y_cbar,
                 cbar_title, cmap):
    """One six-column row of per-gene UMAP maps at page size (panels e and g), in millimeters.
    Each lineage decision keeps its own symmetric scale and its own colorbar, whose poles are
    labeled by the two lineages rather than by numbers. ``cmap`` keeps the push PREDICTION (PuOr)
    apart from the fate RESULT (RdBu_r), exactly as in the poster figure."""
    i0 = 0
    for k in range(len(lps)):
        _, _, An, Bn = lps[k]
        genes_k = pair_genes(k)
        vk = valuemap.get((An, Bn), {})
        pooled = [np.abs(np.asarray(vk[g])) for g in genes_k if g in vk]
        vmax = float(np.percentile(np.concatenate(pooled), 99)) if pooled else 1e-6
        vmax = vmax or 1e-6
        eps = max(0.06 * vmax, 1e-4)
        sc_k = None
        for gi, g in enumerate(genes_k):
            x, w = slots[i0 + gi]
            ax = _mm_axes(fig, x, y_map, w, h_map)
            vals = vk.get(g)
            if vals is None:
                ax.set_axis_off(); continue
            sc_k = draw_fate_map(ax, emb, np.asarray(vals), vmax, f"{g} KO", eps, cmap=cmap,
                                 s_bg=0.7, s_fg=1.6, title_fs=SUB_FS["title"], title_pad=1.5,
                                 raster=True)
        if sc_k is not None:
            x_lo = slots[i0][0]
            x_hi = slots[i0 + len(genes_k) - 1][0] + slots[i0 + len(genes_k) - 1][1]
            cb_w = 32.0
            cax = _mm_axes(fig, (x_lo + x_hi) / 2 - cb_w / 2, y_cbar, cb_w, 1.3)
            cb = fig.colorbar(sc_k, cax=cax, orientation="horizontal")
            cb.set_ticks([-vmax, 0, vmax]); cb.ax.set_xticklabels(["", "", ""])
            cb.ax.tick_params(length=1.5, width=0.4)
            cb.outline.set_linewidth(0.4)
            cax.text(-0.03, 0.5, Bn, transform=cax.transAxes, ha="right", va="center",
                     fontsize=SUB_FS["small"])
            cax.text(1.03, 0.5, An, transform=cax.transAxes, ha="left", va="center",
                     fontsize=SUB_FS["small"])
            cax.text(0.5, -1.6, f"{cbar_title} (±{vmax:.2g})", transform=cax.transAxes,
                     ha="center", va="top", fontsize=SUB_FS["small"], color="0.25")
        i0 += len(genes_k)


def render_submission(C, ds, out_path, suf=""):
    """Panels a-i of the perturbation figure on one journal page.

    What buys the height, none of it a rescale and none of it a dropped panel:
      - panel a is redrawn natively instead of embedding the report's raster streamline plot,
        so it costs a third of a row rather than a full one and carries live 5 pt type;
      - the six knockout columns of d, e, f and g share one column grid, so each row is read
        against the same six genes and no row repeats a header;
      - the per-cell maps of e and g are set to the embedding's own 1.83 aspect ratio, which
        is where most of the poster figure's white space was;
      - the cell-type key is drawn once, under panel a, and serves panel f as well.
    """
    clusters = C["clusters"]; colors = C["colors"]
    order = [c for c in C["cluster_order"] if c in set(clusters)]
    tfs = C["tfs"]; lps = C["lineage_pairs"]; emb = C["emb"]
    groups = C.get("tf_groups", {}); gnames = list(groups.keys())
    tf_color = {g: TF_PALETTE[i % len(TF_PALETTE)] for i, g in enumerate(tfs)}
    casc = C["cascade"]

    def pair_genes(k):
        return groups.get(gnames[k], tfs) if k < len(gnames) else tfs

    fig = sub.figure_for("double", height_mm=SUB_H)
    # Font choices only. Do NOT reset the spine or line settings that figure_for() applied,
    # or every panel gains a full box instead of the two spines that carry an axis.
    matplotlib.rcParams.update({"font.family": "sans-serif",
                                "font.sans-serif": _SANS,
                                "mathtext.fontset": _MATHSET,
                                "mathtext.default": "regular",
                                "legend.frameon": True,
                                "savefig.dpi": 600,
                                "axes.spines.top": False,
                                "axes.spines.right": False})

    # ---- band a/b/c: embedding with velocity streamlines, and the two driver-score panels ----
    # Panel a reuses the report's pre-rendered streamline plot rather than redrawing the
    # field natively. That is deliberate: a native quiver or streamplot over this embedding
    # looks bad at any density that is legible, and the rendered PNG is 683 x 477 px, which
    # is 358 dpi across the 48.5 mm panel, comfortably above the 300 dpi floor.
    ax_a = _mm_axes(fig, SUB_L, 5.0, 48.5, 26.5)
    ax_a.set_axis_off()
    _vpath = f"{paths.REPORTS}/{ds}/plots/A2_input_velocity.png"
    if os.path.exists(_vpath):
        ax_a.imshow(mpimg.imread(_vpath))
    else:
        draw_velocity_stream(ax_a, emb, clusters, colors, C["wt_flow"])
    _letter(fig, 5.5, 4.2, "a")
    lax = _mm_axes(fig, 6.0, 33.5, 54.0, 9.0); lax.set_axis_off()
    from matplotlib.lines import Line2D
    lax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=2.2, color=colors.get(c, "0.6"),
                               label=c) for c in order],
               loc="center", ncol=3, fontsize=SUB_FS["small"], frameon=False,
               handletextpad=0.3, columnspacing=0.9, labelspacing=0.35, borderpad=0.0)

    pfs = dict(bg_s=1.6, front_s=9, lab=5.0, axlab=SUB_FS["axlab"], tick=SUB_FS["tick"],
               title=SUB_FS["title"], n_lab=7, raster=True)
    sc_p = None
    for k, (letter, x0) in enumerate([("b", 74.0), ("c", 131.0)]):
        if k >= len(lps):
            break
        A, B, An, Bn = lps[k]
        axp = _mm_axes(fig, x0, 5.5, 43.0, 27.0)
        _letter(fig, x0 - 8.0, 4.2, letter)
        csvp = f"{paths.REPORTS}/{ds}/data/driver_scores_{k + 1}{suf}.csv"
        if os.path.exists(csvp):
            s = draw_pareto(axp, csvp, pair_genes(k), An, Bn, fs=pfs)
            sc_p = s or sc_p
    if sc_p is not None:                          # the Pareto-front key, shared by b and c
        cax = _mm_axes(fig, 107.0, 41.0, 34.0, 1.3)
        cb = fig.colorbar(sc_p, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 5]); cb.ax.tick_params(labelsize=SUB_FS["small"], length=1.5,
                                                width=0.4, pad=1)
        cb.outline.set_linewidth(0.4)
        cb.set_label("Pareto front (0 = best)", fontsize=SUB_FS["small"], labelpad=1)

    # ---- band d: predicted first-order knockout response, one panel per knockout -----------
    resp = C.get("jac_response", {}); reg = C.get("out_strength")
    d_slots = _mm_slots(len(tfs), 17.5)
    d_slots = [(x + 4.6, w) for x, w in d_slots]        # room for the target-gene labels
    _letter(fig, 2.0, 48.6, "d")
    i0 = 0
    for k in range(len(lps)):
        for gi, g in enumerate(pair_genes(k)):
            x, w = d_slots[i0 + gi]
            ax = _mm_axes(fig, x, 52.5, w, 25.0)
            draw_ko_response(ax, resp.get(g), reg, g,
                             fs=dict(title=SUB_FS["title"], title_pad=1.5, blank=5.0,
                                     ytick=SUB_FS["tick"], xtick=SUB_FS["tick"], exponent=5.0))
        i0 += len(pair_genes(k))

    # ---- band e: predicted commitment push, per cell (the PREDICTION) ----------------------
    push_map = {}
    for g, (An, Bn, arr) in C.get("commit_push", {}).items():
        push_map.setdefault((An, Bn), {})[g] = arr
    m_slots = _mm_slots(len(tfs), 25.6)
    _letter(fig, 2.0, 86.0, "e")
    sub_umap_row(fig, lps, pair_genes, push_map, emb, m_slots, 90.0, 14.0, 106.5,
                 "predicted push", "PuOr")

    # ---- band f: short-time cascade relative to the wild type ------------------------------
    conds = [f"{tf} KO" for tf in tfs]
    ymax = float(casc["mean_abs_delta"].max()) if len(casc) else 1.0
    f_slots = _mm_slots(len(conds), 22.0)
    _letter(fig, 2.0, 114.0, "f")
    for k, cond in enumerate(conds):
        x, w = f_slots[k]
        ax = _mm_axes(fig, x, 118.0, w, 18.0)
        draw_cascade(ax, casc, cond, colors, order, ymax, show_legend=False,
                     show_ylabel=(k == 0),
                     fs=dict(lw=0.8, title=SUB_FS["title"], weight="normal", title_pad=1.5,
                             axlab=SUB_FS["axlab"], tick=SUB_FS["tick"]))
        if k == 0:
            ax.set_ylabel("mean |$x_{KO}-x_{WT}$|", fontsize=SUB_FS["axlab"], labelpad=1)

    # ---- band g: propagated per-cell fate shift (the RESULT) -------------------------------
    fate_map = {(An, Bn): C.get("fate_map", {}).get((An, Bn), {}).get("shift", {})
                for _, _, An, Bn in lps}
    _letter(fig, 2.0, 146.0, "g")
    sub_umap_row(fig, lps, pair_genes, fate_map, emb, m_slots, 150.0, 14.0, 166.5,
                 "post-KO fate shift", "RdBu_r")

    # ---- band h: fate-probability shift per regulator over the transitional cells ----------
    _letter(fig, 2.0, 174.5, "h")
    hi_x = [16.0, 104.0]
    for k, (A, B, An, Bn) in enumerate(lps[:2]):
        ax = _mm_axes(fig, hi_x[k], 178.5, 72.0, 18.0)
        sb = C.get("fate_bias", {}).get((An, Bn))
        if sb is not None and len(sb):
            draw_bias(ax, sb, pair_genes(k), An, Bn,
                      fs=dict(tick=SUB_FS["tick"], ylab=SUB_FS["axlab"], title=SUB_FS["title"],
                              legend=SUB_FS["small"], edge_lw=0.7, headroom=0.42, rot=55,
                              leg_kw=dict(handlelength=1.0, handletextpad=0.4, labelspacing=0.25,
                                          borderpad=0.3, borderaxespad=0.2)))
            ax.yaxis.label.set_size(SUB_FS["axlab"])
            ax.tick_params(axis="y", labelsize=SUB_FS["tick"], pad=1)

    # ---- band i: dose-response of the fate shift -------------------------------------------
    _letter(fig, 2.0, 208.5, "i")
    for k, (A, B, An, Bn) in enumerate(lps[:2]):
        ax = _mm_axes(fig, hi_x[k], 212.5, 72.0, 19.0)
        draw_fate_dose(ax, C, An, Bn, pair_genes(k), tf_color,
                       fs=dict(ms=1.8, lw=0.8, side=SUB_FS["small"], axlab=SUB_FS["axlab"],
                               ylab=SUB_FS["axlab"], title=SUB_FS["title"], tick=SUB_FS["tick"],
                               legend=SUB_FS["small"], leg_loc="lower center",
                               leg_kw=dict(handlelength=1.0, handletextpad=0.3,
                                           columnspacing=0.9, borderpad=0.3)))
        ax.xaxis.labelpad = 1.5; ax.yaxis.labelpad = 1.5

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub.save(fig, out_path)
    plt.close(fig)
    print(f"wrote {out_path}  ({SUB_W:.0f} x {SUB_H:.0f} mm, floor {sub.TYPE_FLOOR:.0f} pt)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pancreas")
    ap.add_argument("--mode", default="biological", choices=["biological", "discovery"],
                    help="biological = known regulators (Pareto rounds, all panels a-g); "
                         "discovery = above-threshold driver genes (threshold lines, no panel e)")
    ap.add_argument("--variant", default="", help="fit-cache tag (e.g. 'bimodal'): read the _<tag> perturb "
                    "cache + driver-score CSVs and write perturbation-dynamics-<ds>_<tag>.")
    ap.add_argument("--submission", action="store_true",
                    help="render the same panels a-i on one journal page (180 x 244 mm, 5 pt "
                         "floor) and write it to --out; the default output is untouched.")
    ap.add_argument("--out", default=os.path.join(paths.FIGURES_SPEC, "Figure5.pdf"),
                    help="output path for --submission")
    args = ap.parse_args()
    ds = args.dataset
    discovery = args.mode == "discovery"
    suf = f"_{args.variant}" if args.variant else ""
    cache = f"perturb_dynamics_discovery{suf}.pkl" if discovery else f"perturb_dynamics{suf}.pkl"
    with open(f"{paths.REPORTS}/{ds}/data/{cache}", "rb") as fh:
        C = pickle.load(fh)

    if args.submission:
        if discovery:
            raise SystemExit("--submission renders the biological figure (panels a-i); the "
                             "discovery variant has a tenth panel and is not a main figure.")
        render_submission(C, ds, args.out, suf=suf)
        return

    clusters = C["clusters"]; colors = C["colors"]; basis = C["basis"]; ck = C["cluster_key"]
    order = [c for c in C["cluster_order"] if c in set(clusters)]
    tfs = C["tfs"]; lps = C["lineage_pairs"]
    tf_color = {g: TF_PALETTE[i % len(TF_PALETTE)] for i, g in enumerate(tfs)}
    groups = C.get("tf_groups", {})
    gnames = list(groups.keys())                                # [progenitor->endocrine, alpha/beta]
    casc = C["cascade"]; Tfin = C.get("cascade_time", 20.0)

    def pair_genes(k):                                          # the TF group tied to lineage pair k
        return groups.get(gnames[k], tfs) if k < len(gnames) else tfs

    use_style(9)
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(max(15.5, 2.0 * len(tfs) + 3.5), 23.0))

    def label(ax, text, dx=0.03, dy=0.008):
        # render ONLY the panel letter; the per-panel description stays in the call as a caption
        # reference (papers put the description in the caption, not as a title on the panel)
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + dy, text.split()[0], fontweight="bold", fontsize=13, va="bottom",
                 ha="left")

    proc = f"  ({PROCESS[ds]})" if ds in PROCESS else ""
    ttl = (f"In-silico perturbation of candidate driver genes in {ds}{proc}" if discovery
           else f"In-silico perturbation of {ds}{proc}")
    fig.text(0.05, 0.996, ttl, ha="left", va="top", fontsize=15, fontweight="bold")

    L, R = 0.05, 0.955
    npair = len(lps)
    col_w = (R - L) / len(tfs)
    emb = C["emb"]

    # Walk the bands top-to-bottom with per-band top/bottom padding for the letter/header/colorbar/
    # x-labels and an even inter-band gap, so no title collides and the stack fills to the bottom
    # (no trailing whitespace). Each spec = (name, height, pad_top, pad_bot).
    # taller bands + smaller pads so the fill-gap between rows is small (tight rows, no bottom whitespace).
    specs = [("top", 0.098, 0.000, 0.022), ("d", 0.082, 0.014, 0.020),
             ("e", 0.088, 0.022, 0.014), ("f", 0.082, 0.014, 0.024)]
    specs += [("g", 0.088, 0.022, 0.014)]
    if not discovery:
        specs += [("h", 0.082, 0.016, 0.028), ("i", 0.104, 0.014, 0.026)]
    else:
        specs += [("i", 0.104, 0.014, 0.026), ("tier", 0.115, 0.026, 0.010)]
    total_h = sum(h + pt + pb for _, h, pt, pb in specs)
    gap = max(0.012, (0.965 - 0.045 - total_h) / max(len(specs) - 1, 1))
    pos = {}; cur = 0.965
    for name, h, pt, pb in specs:
        top = cur - pt; pos[name] = (top, top - h); cur = top - h - pb - gap

    # a: velocity embedding + one driver-score panel per lineage pair
    t, b = pos["top"]
    gs_top = fig.add_gridspec(1, 1 + npair, top=t, bottom=b, left=L, right=R,
                              width_ratios=[1.5] + [1.0] * npair, wspace=0.22)
    ax_a = fig.add_subplot(gs_top[0, 0]); ax_a.set_axis_off()
    vpath = f"{paths.REPORTS}/{ds}/plots/A2_input_velocity.png"
    if os.path.exists(vpath):
        ax_a.imshow(mpimg.imread(vpath))
    ax_paretos = []; sc_p = None
    for k in range(npair):
        A, B, An, Bn = lps[k]
        axp = fig.add_subplot(gs_top[0, 1 + k]); ax_paretos.append(axp)
        csvp = f"{paths.REPORTS}/{ds}/data/driver_scores_{k + 1}{suf}.csv"
        if os.path.exists(csvp):
            s = draw_pareto(axp, csvp, pair_genes(k), An, Bn, discovery=discovery)
            if s is not None:
                sc_p = s
    if sc_p is not None:
        pcax = fig.add_axes([0.965, b + 0.005, 0.006, (t - b) * 0.55])
        pcb = fig.colorbar(sc_p, cax=pcax); pcb.set_ticks([0, 5])
        pcb.set_label("Pareto front (0 = best)", fontsize=6.5); pcb.ax.tick_params(labelsize=6)

    # d: predicted first-order KO response (Jacobian), one barplot per gene (gene space; no UMAP)
    resp = C.get("jac_response", {}); reg = C.get("out_strength")
    t, b = pos["d"]
    gs_r = fig.add_gridspec(1, len(tfs), top=t, bottom=b, left=L, right=R, wspace=0.55)
    ax_r0 = None; i0 = 0
    for k in range(npair):
        for gi, g in enumerate(pair_genes(k)):
            ax = fig.add_subplot(gs_r[0, i0 + gi]); ax_r0 = ax_r0 or ax
            draw_ko_response(ax, resp.get(g), reg, g)
        i0 += len(pair_genes(k))

    # e: predicted commitment push (Jacobian on the fate axis), one UMAP per gene -> the PREDICTION
    push = C.get("commit_push", {})
    push_map = {}
    for g, (An, Bn, arr) in push.items():
        push_map.setdefault((An, Bn), {})[g] = arr
    t, b = pos["e"]
    ax_p0 = draw_gene_umap_band(fig, tfs, npair, pair_genes, lps, push_map, emb, t, b, L, R, col_w,
                                "predicted push", cmap="PuOr")

    # f: short-time cascade relative to WT (the first, coarse KO result: magnitude, not direction)
    conds = [f"{tf} KO" for tf in tfs]
    ymax = float(casc["mean_abs_delta"].max()) if len(casc) else 1.0
    t, b = pos["f"]
    gs_c = fig.add_gridspec(1, len(conds), top=t, bottom=b, left=L, right=R, wspace=0.10)
    ax_c0 = None
    for k, cond in enumerate(conds):
        ax = fig.add_subplot(gs_c[0, k]); ax_c0 = ax_c0 or ax
        draw_cascade(ax, casc, cond, colors, order, ymax, show_legend=(k == 0), show_ylabel=(k == 0))
        if k == 0:
            ax.set_ylabel("mean |$x_{KO}(t) - x_{WT}(t)$|", fontsize=8)

    # g: per-cell fate shift (propagated result; the ACTUAL counterpart to the panel-e push PREDICTION)
    fate_map = {(An, Bn): C.get("fate_map", {}).get((An, Bn), {}).get("shift", {})
                for _, _, An, Bn in lps}
    t, b = pos["g"]
    ax_g0 = draw_gene_umap_band(fig, tfs, npair, pair_genes, lps, fate_map, emb, t, b, L, R, col_w,
                                "post-KO fate shift")

    # h: single-KO fate-probability bias (biological only)
    ax_h0 = None
    if not discovery:
        t, b = pos["h"]
        gs_h = fig.add_gridspec(1, len(lps), top=t, bottom=b, left=L, right=R, wspace=0.16)
        for k, (A, B, An, Bn) in enumerate(lps):
            ax = fig.add_subplot(gs_h[0, k]); ax_h0 = ax_h0 or ax
            sb = C.get("fate_bias", {}).get((An, Bn))
            if sb is not None and len(sb):
                draw_bias(ax, sb, pair_genes(k), An, Bn)

    # i: dose-response of the fate shift
    t, b = pos["i"]
    gs_i = fig.add_gridspec(1, len(lps), top=t, bottom=b, left=L, right=R, wspace=0.16)
    ax_i0 = None
    for k, (A, B, An, Bn) in enumerate(lps):
        ax = fig.add_subplot(gs_i[0, k]); ax_i0 = ax_i0 or ax
        draw_fate_dose(ax, C, An, Bn, pair_genes(k), tf_color)

    # tier table (discovery only)
    ax_t0 = None
    if discovery:
        t, b = pos["tier"]
        gs_t = fig.add_gridspec(1, 1, top=t, bottom=b, left=L, right=R)
        ax_t0 = fig.add_subplot(gs_t[0, 0])
        draw_disc_tier_table(ax_t0, ds, groups)

    import string
    lt = list(string.ascii_lowercase); li = 0
    # panel a's imshow shifts ax_a's bbox (aspect), so anchor its letter to the fixed left edge L (the
    # same x the row-band letters land at: their first axis is at L and label() offsets by dx=0.03).
    fig.text(L - 0.03, pos["top"][0] + 0.010, lt[li],          # letter only (description -> caption)
             fontweight="bold", fontsize=13, va="bottom", ha="left"); li += 1
    for k, axp in enumerate(ax_paretos):
        label(axp, f"{lt[li]}   Driver scores", dx=0.05, dy=0.010); li += 1
    if ax_r0 is not None:
        label(ax_r0, f"{lt[li]}   Predicted knockout response (Jacobian, gene space)", dy=0.012); li += 1
    if ax_p0 is not None:
        label(ax_p0, f"{lt[li]}   Predicted commitment push (Jacobian on the fate axis)", dy=0.016); li += 1
    label(ax_c0, f"{lt[li]}   Short-time cascade of the knockout (deviation from WT)", dy=0.012); li += 1
    if ax_g0 is not None:
        label(ax_g0, f"{lt[li]}   Per-cell fate shift after knockout (propagated)", dy=0.016); li += 1
    if ax_h0 is not None:
        label(ax_h0, f"{lt[li]}   Fate-probability shift after single-gene knockout", dy=0.016); li += 1
    label(ax_i0, f"{lt[li]}   Dose-response of the fate-probability shift", dy=0.012); li += 1
    if ax_t0 is not None:
        label(ax_t0, f"{lt[li]}   Evidence-tier classification of the discovered candidates", dy=0.014); li += 1

    NAME = (f"perturbation-dynamics-discovery-{ds}{suf}" if discovery
            else f"perturbation-dynamics-{ds}{suf}")
    outdir = OUT if (ds == "pancreas" and not discovery) else f"{OUT}/extended"
    os.makedirs(outdir, exist_ok=True)
    save(fig, f"{outdir}/{NAME}", formats=("pdf", "png"))
    print(f"wrote {outdir}/{NAME}.pdf + .png  (mode={args.mode}; tfs={tfs}; "
          f"pairs={[(An,Bn) for _,_,An,Bn in lps]})")
    plt.close(fig)


if __name__ == "__main__":
    main()
