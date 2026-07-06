"""Figure sections for the per-dataset scHopfield reports (notebooks 01-08 pipeline).

Each ``section_*`` drives the canonical ``sch.pl.*`` / ``sch.tl.*`` functions, saves
figures and appends captioned, explained subsections to a Report. Consistent cell-type
colors (from ``uns[f'{ck}_colors']``) are threaded through every cluster-colored plot;
velocity / perturbation flows are drawn as clean streamlines (not huge quiver arrows).
Any single figure that fails is logged and skipped so one bad panel never aborts a report.
"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

import scHopfield as sch
from rutils import save, present_clusters, ROOT as ROOT_

FORCE_FIGS = os.environ.get("REPORT_FORCE_FIGS", "") == "1"

# gene-group colors (match notebook 06): erythroid / myeloid / *-biased
GRP = dict(A="#E74C3C", B="#3498DB", A_bias="#F39C12", B_bias="#9B59B6")


def basis_of(adata):
    for b in ("draw_graph_fa", "umap", "pca", "diffmap"):
        if f"X_{b}" in adata.obsm:
            return b
    return "umap"


def get_colors(adata, ck):
    """Stable {cluster: color} dict stored in uns[f'{ck}_colors'] and reused everywhere."""
    if not hasattr(adata.obs[ck], "cat"):
        adata.obs[ck] = adata.obs[ck].astype("category")
    cats = list(adata.obs[ck].cat.categories)
    key = f"{ck}_colors"
    cols = adata.uns.get(key)
    if cols is None or len(cols) != len(cats):
        pal = sc.pl.palettes.default_20 if len(cats) <= 20 else sc.pl.palettes.default_102
        cols = [pal[i % len(pal)] for i in range(len(cats))]
        adata.uns[key] = cols
    return dict(zip([str(c) for c in cats], list(cols)))


def order_by_trajectory(adata, ck, clusters, lineages=None, explicit=None):
    """Order cell types along the differentiation trajectory. An explicit config order
    wins; otherwise left->right by pseudotime, or (with two lineages) terminal-B ...
    progenitors ... terminal-A (middle-to-sides)."""
    if explicit:
        present = set(clusters)
        ordered = [c for c in explicit if c in present]
        ordered += [c for c in clusters if c not in set(ordered)]  # any leftover
        return ordered
    pt = next((c for c in ["Pseudotime", "dpt_pseudotime", "latent_time", "palantir_pseudotime"]
               if c in adata.obs), None)
    if pt is None:
        return clusters
    lab = adata.obs[ck].astype(str)
    mpt = {c: float(np.nanmean(adata.obs.loc[lab == c, pt].values)) for c in clusters}
    if lineages:
        A = [c for c in clusters if c in set(lineages.get("A", []))]
        B = [c for c in clusters if c in set(lineages.get("B", []))]
        mids = [c for c in clusters if c not in A and c not in B]
        return (sorted(B, key=lambda c: -mpt[c]) + sorted(mids, key=lambda c: mpt[c])
                + sorted(A, key=lambda c: mpt[c]))
    return sorted(clusters, key=lambda c: mpt[c])


def _streamplot(ax, adata, flow_key, basis, ck=None, colors=None, n_grid=30, density=1.3,
                stream_color="#222222", title=None, bg=True):
    """Clean streamline rendering of an embedding flow (replaces huge quiver arrows).

    Interpolates the per-cell flow onto a grid (Gaussian-weighted KNN), masks low-density
    regions, and draws streamlines with speed-scaled linewidth over a cell-type-colored
    background."""
    from sklearn.neighbors import NearestNeighbors
    emb = np.asarray(adata.obsm[f"X_{basis}"])[:, :2]
    F = np.asarray(adata.obsm[flow_key])[:, :2]
    if bg:
        if ck is not None and colors is not None:
            cvec = [colors.get(str(c), "#cccccc") for c in adata.obs[ck].astype(str)]
        else:
            cvec = "#dddddd"
        ax.scatter(emb[:, 0], emb[:, 1], c=cvec, s=7, alpha=0.45, linewidths=0, zorder=1)
    xmin, xmax = emb[:, 0].min(), emb[:, 0].max()
    ymin, ymax = emb[:, 1].min(), emb[:, 1].max()
    gx = np.linspace(xmin, xmax, n_grid); gy = np.linspace(ymin, ymax, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])
    nn = NearestNeighbors(n_neighbors=min(60, len(emb))).fit(emb)
    dist, idx = nn.kneighbors(grid)
    sigma = np.mean([xmax - xmin, ymax - ymin]) / n_grid
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    wsum = w.sum(1)
    U = (w * F[idx, 0]).sum(1) / (wsum + 1e-9)
    V = (w * F[idx, 1]).sum(1) / (wsum + 1e-9)
    low = wsum < np.percentile(wsum, 35)          # blank out sparse regions
    U[low] = 0.0; V[low] = 0.0
    U = U.reshape(n_grid, n_grid); V = V.reshape(n_grid, n_grid)
    speed = np.sqrt(U ** 2 + V ** 2)
    lw = 0.4 + 2.2 * speed / (np.nanmax(speed) + 1e-9)
    ax.streamplot(gx, gy, U, V, density=density, color=stream_color, linewidth=lw,
                  arrowsize=0.8, zorder=2)
    ax.set(xticks=[], yticks=[])
    if title:
        ax.set_title(title)
    return ax


def fitted_velocity(adata, ck):
    """Per-cell fitted Hopfield velocity W.sigma(x) + I - gamma*x (cluster-specific W/I/gamma)."""
    x = np.asarray(adata.layers["Ms"]); sig = np.asarray(adata.layers["sigmoid"])
    lab = adata.obs[ck].astype(str); pred = np.zeros_like(x)
    for c in lab.unique():
        if f"W_{c}" not in adata.varp:
            continue
        m = (lab == c).values; W = np.asarray(adata.varp[f"W_{c}"])
        I = np.asarray(adata.var[f"I_{c}"]) if f"I_{c}" in adata.var else 0.0
        g = np.asarray(adata.var[f"gamma_{c}"]) if f"gamma_{c}" in adata.var else np.asarray(adata.var["gamma"])
        pred[m] = sig[m] @ W.T + I - g * x[m]
    return pred.astype(np.float32)


def _stream(ax, adata, ck, colors, basis, velocity, title=None, color_key=None):
    """Project a per-cell velocity to the embedding and draw scVelo streamlines. This uses
    the transition-probability embedding (like scVelo/dynamo), which gives the clean
    progenitor->terminal flow that a raw KNN quiver projection does not."""
    import scvelo as scv
    a = adata
    if not hasattr(a.obs[ck], "cat"):
        a.obs[ck] = a.obs[ck].astype("category")
    a.uns[f"{ck}_colors"] = [colors.get(str(c), "#888888") for c in a.obs[ck].cat.categories]
    a.layers["velocity"] = np.asarray(velocity, dtype=np.float32)
    scv.tl.velocity_graph(a, n_jobs=4, sqrt_transform=False)
    try:
        scv.pl.velocity_embedding_stream(
            a, basis=basis, color=(color_key or ck), ax=ax, show=False, legend_loc="none",
            title=title or "", density=1.3, linewidth=1.0, arrow_size=1.1, alpha=0.5)
    except Exception:
        # scVelo streamplot needs a regular grid; on the rare embedding where it fails,
        # fall back to a grid quiver of the embedded velocity.
        scv.tl.velocity_embedding(a, basis=basis)
        _flow_grid(ax, a, f"velocity_{basis}", basis, ck, colors)
        ax.set_title(title or "")
    ax.set_axis_off()
    return ax


DYN_PY = "/home/bernaljp/Documents/SCH/.venv-dyn/bin/python"


def _dyn_stream(name, fname, adata, ck, colors, basis, velocity, title=""):
    """Render a velocity field with dynamo's streamline_plot via the isolated .venv-dyn.
    Projects in the main env (scVelo velocity_embedding), hands a clean minimal AnnData to
    the dynamo subprocess. Returns the plots-relative path, or None (caller falls back)."""
    import subprocess
    import anndata as ad
    import scvelo as scv
    rel = f"plots/{fname}"; path = f"{ROOT_}/{name}/{rel}"
    if not FORCE_FIGS and os.path.exists(path):
        return rel
    if not os.path.exists(DYN_PY):
        return None
    try:
        adata.layers["velocity"] = np.asarray(velocity, dtype=np.float32)
        scv.tl.velocity_graph(adata, n_jobs=4, sqrt_transform=False)
        scv.tl.velocity_embedding(adata, basis=basis)
        m = ad.AnnData(X=np.asarray(adata.layers["Ms"]),
                       obs=adata.obs[[ck]].astype(str).astype("category"))
        m.obsm["X_umap"] = np.asarray(adata.obsm[f"X_{basis}"])[:, :2]
        m.obsm["velocity_umap"] = np.asarray(adata.obsm[f"velocity_{basis}"])[:, :2]
        cats = list(m.obs[ck].cat.categories)
        m.uns[f"{ck}_colors"] = [colors.get(str(c), "#888888") for c in cats]
        hp = f"{ROOT_}/{name}/data/_handoff_{fname}.h5ad"
        m.write(hp)
        r = subprocess.run([DYN_PY, "analyses/reports/dyn_streamline.py", "--input", hp,
                            "--cluster", ck, "--out", path, "--title", title, "--basis", "umap"],
                           capture_output=True, text=True, timeout=400)
        try:
            os.remove(hp)
        except OSError:
            pass
        if r.returncode == 0 and os.path.exists(path):
            return rel
        print(f"  [dyn_stream {fname}] rc={r.returncode}: {r.stderr[-300:]}", flush=True)
    except Exception as exc:
        print(f"  [dyn_stream {fname}] {type(exc).__name__}: {exc}", flush=True)
    return None


def _flow_panel(report, name, fname, adata, ck, colors, basis, velocity, title, caption):
    """Embed a velocity field: dynamo streamlines if available, else a scVelo stream."""
    rel = _dyn_stream(name, fname, adata, ck, colors, basis, velocity, title)
    if rel:
        report.img(rel, caption)
        return
    def _fig():
        fig, ax = plt.subplots(figsize=(7, 6))
        _stream(ax, adata, ck, colors, basis, velocity, title=title)
        return fig
    _try(report, name, fname, _fig, caption)


def _flow_grid(ax, adata, flow_key, basis, ck, colors, color="black", scale=None,
               n_grid=25, min_mass=25):
    """Grid-averaged flow via the package plot_flow (on_grid), with magnitude outliers
    clipped so a few cells don't dominate, and no axes."""
    F = np.asarray(adata.obsm[flow_key]); mag = np.linalg.norm(F, axis=1)
    pos = mag > 0
    if pos.any():
        cap = float(np.percentile(mag[pos], 99))
        if cap > 0:
            adata.obsm[flow_key] = F * np.minimum(1.0, cap / (mag + 1e-12))[:, None]
    sch.pl.plot_flow(adata, flow_key=flow_key, basis=basis, ax=ax, on_grid=True,
                     n_grid=n_grid, min_mass=min_mass, scale=scale, color=color,
                     cluster_key=ck, colors=colors)
    ax.set_axis_off()
    return ax


def _try(report, name, fname, fn, caption):
    rel = f"plots/{fname}"
    if not FORCE_FIGS and os.path.exists(f"{ROOT_}/{name}/{rel}"):
        report.img(rel, caption); return True
    try:
        fig = fn()
        rel = save(fig, name, fname)
        report.img(rel, caption)
        return True
    except Exception as exc:
        import traceback
        print(f"  [fig FAIL] {fname}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        report.text(f"> _(figure `{fname}` skipped: {type(exc).__name__})_")
        return False


def _top_by_centrality(adata, ck, clusters, metric="degree_centrality_out", n=4):
    scores = pd.Series(0.0, index=adata.var_names)
    for c in clusters:
        col = f"{metric}_{c}"
        if col in adata.var.columns:
            scores = scores.add(adata.var[col].fillna(0), fill_value=0)
    return list(scores.nlargest(n).index)


# --------------------------------------------------------------------------- #
# Section A: overview -- embedding, velocities, Hill fitting
# --------------------------------------------------------------------------- #
def section_A(adata, name, ck, report, cfg):
    basis = basis_of(adata)
    colors = get_colors(adata, ck)
    report.section(
        "1. Overview: embedding, dynamics, and Hill fitting",
        f"Dataset **{name}** ({adata.n_obs} cells x {adata.n_vars} genes, "
        f"{len(present_clusters(adata, ck))} cell types). Dynamics from **{cfg['velocity_mode']}**."
    )

    report.sub("1.1 Cell types on the embedding",
               "These cell-type colors are reused throughout the report.")
    def _ct():
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        sc.pl.embedding(adata, basis=basis, color=ck, ax=ax, show=False, frameon=False,
                        legend_loc="on data", legend_fontsize=7, size=25, title=f"{name}")
        return fig
    _try(report, name, "A1_celltypes.png", _ct, f"Cell types on the {basis} embedding.")

    report.sub("1.2 Dynamics on the embedding",
               "Velocity fields drawn with dynamo's streamline plotter (the fitted velocity is "
               "projected to the embedding with scVelo, then rendered by dynamo). The fitted "
               "Hopfield velocity should reproduce the input -- progenitors flow to the terminal "
               "states.")
    vlab = "pseudotime" if cfg["velocity_mode"] == "pseudotime" else "RNA"
    _flow_panel(report, name, "A2_input_velocity.png", adata, ck, colors, basis,
                np.asarray(adata.layers["velocity_S"]), f"{name}: input {vlab} velocity",
                f"Input {vlab} velocity field (dynamo streamlines).")
    _flow_panel(report, name, "A2_fitted_velocity.png", adata, ck, colors, basis,
                fitted_velocity(adata, ck), f"{name}: fitted Hopfield velocity",
                "Fitted Hopfield velocity field -- reproduces the input dynamics.")

    report.sub("1.3 Hill (sigmoid) fitting and goodness of fit",
               "Per-gene Hill activation fit to the expression CDF; goodness of fit is "
               "`sigmoid_mse` (lower = better). Worst fits are typically genes with two "
               "expression regimes (double sigmoid) or very sharp switches.")
    used = adata.var["scHopfield_used"].values if "scHopfield_used" in adata.var else np.ones(adata.n_vars, bool)
    mse = adata.var.loc[used, "sigmoid_mse"].values if "sigmoid_mse" in adata.var else None
    if mse is not None:
        def _mse():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(mse, bins=40, color="#3b6ea5")
            ax.set(xlabel="sigmoid_mse (CDF fit)", ylabel="# genes",
                   title=f"Hill goodness-of-fit (median {np.median(mse):.4f})")
            return fig
        _try(report, name, "A3_hill_gof.png", _mse,
             f"Hill-fit MSE across genes (median {np.median(mse):.4f}).")
        names = adata.var_names[used]
        order = np.argsort(mse)
        examples = list(names[order[:2]]) + list(names[order[-2:]])
        def _examples():
            fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
            for ax, g in zip(axes, examples):
                try:
                    sch.pl.plot_sigmoid_fit(adata, g, cluster_key=ck, ax=ax)
                except Exception:
                    ax.set_visible(False)
            fig.suptitle("Hill fits: two best (left) and two worst (right) genes", y=1.02)
            return fig
        _try(report, name, "A4_hill_examples.png", _examples,
             "Example Hill fits: two best- and two worst-fit genes by MSE.")

        report.sub("1.4 Double-sigmoid (two-component) Hill fits",
                   "Some worst-fit genes have two expression regimes (a double-sigmoid CDF) "
                   "that a single Hill cannot capture. Here the four worst genes are fit with a "
                   "single Hill and with a two-component mixture a*H(k1,n1)+(1-a)*H(k2,n2); the "
                   "bimodal fit is shown when it improves the MSE by >15%.")
        worst = list(names[order[-4:]])
        def _bimodal():
            from scHopfield._utils.math import fit_sigmoid as _fs, fit_sigmoid_bimodal as _fb, sigmoid as _sg
            fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
            Ms = np.asarray(adata.layers["Ms"])
            vnames = list(adata.var_names)
            for ax, g in zip(axes, worst):
                gi = vnames.index(g)
                x = np.sort(Ms[:, gi].ravel()); x = x[x > 0.05 * (x.max() if x.size else 1)]
                if x.size < 8:
                    ax.set_visible(False); continue
                y = np.linspace(0, 1, x.size)
                k1, n1, _o, m1 = _fs(x)
                k1b, n1b, k2b, n2b, a, _ob, m2, isbi = _fb(x)
                ax.plot(x, y, ".", ms=2, color="#aaaaaa", label="ECDF")
                ax.plot(x, _sg(x, k1, n1), color="#2a6f97", lw=2, label=f"single (mse {m1:.3g})")
                if isbi:
                    mix = a * _sg(x, k1b, n1b) + (1 - a) * _sg(x, k2b, n2b)
                    ax.plot(x, mix, color="#c1121f", lw=2, label=f"bimodal (mse {m2:.3g})")
                ax.set_title(g, fontsize=9); ax.legend(fontsize=6); ax.set_xlabel("expression")
            fig.suptitle("Worst single-Hill fits: single vs two-component Hill", y=1.02)
            return fig
        _try(report, name, "A5_hill_bimodal.png", _bimodal,
             "Single vs two-component Hill on the worst-fit genes (bimodal shown when it "
             "improves MSE >15%).")


# --------------------------------------------------------------------------- #
# Section B: energy landscape
# --------------------------------------------------------------------------- #
def section_B(adata, name, ck, report, cfg):
    basis = basis_of(adata)
    colors = get_colors(adata, ck)
    clusters = present_clusters(adata, ck)
    order = order_by_trajectory(adata, ck, clusters, cfg.get("lineages"), explicit=cfg.get("order"))
    report.section(
        "2. Energy landscape",
        "The fitted model defines a Lyapunov energy per cell, decomposed into interaction, "
        "degradation, and bias terms (bias ~0 after the scaffold + L1 fit). Cell types are "
        "ordered along the differentiation trajectory."
    )

    report.sub("2.1 Energy by cell type (all components)",
               "Cell types ordered along the trajectory (progenitors toward the middle when "
               "two lineages are present); colored by the shared cell-type palette.")
    _try(report, name, "B1_energy_boxplots.png",
         lambda: sch.pl.plot_energy_boxplots(adata, cluster_key=ck, plot_energy="all",
                                             order=order, colors=colors),
         "Total / interaction / degradation / bias energy per cell type (trajectory-ordered).")

    report.sub("2.2 Energy landscape on the embedding (all components)")
    _try(report, name, "B2_energy_scatters.png",
         lambda: sch.pl.plot_energy_scatters(adata, cluster_key=ck, basis=basis, plot_energy="all",
                                             order=order, colors=colors),
         "Each energy component painted on the embedding.")

    report.sub("2.3 Energy-gene correlation (total energy)",
               "Correlation of each gene's expression with the total energy, per cell type "
               "(descriptive: which genes track landscape depth). Colored by cell type.")
    _try(report, name, "B3_corr_grid_total.png",
         lambda: sch.pl.plot_correlations_grid(adata, cluster_key=ck, energy="total",
                                               order=order, colors=colors),
         "Gene-vs-total-energy correlations across cell types.")


# --------------------------------------------------------------------------- #
# Section C: regulatory network structure
# --------------------------------------------------------------------------- #
def section_C(adata, name, ck, report, cfg):
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    colors = get_colors(adata, ck)
    clusters = present_clusters(adata, ck)
    order = order_by_trajectory(adata, ck, clusters, cfg.get("lineages"), explicit=cfg.get("order"))
    report.section(
        "3. Regulatory network structure",
        "Cell-type-specific GRNs (`W_c`): similarity across cell types, hub genes, spectral "
        "structure, and the wiring itself."
    )

    report.sub("3.1 Cell-type network similarity",
               "Pearson correlation of the flattened `W_c`: dendrogram + a hierarchically "
               "ordered heatmap.")
    nc = adata.uns.get("scHopfield", {}).get("network_correlations", {})
    P = nc.get("pearson")
    if P is not None:
        P = P.loc[[c for c in clusters if c in P.index], [c for c in clusters if c in P.columns]].astype(float)
        def _dend():
            d = 1 - P.values; np.fill_diagonal(d, 0.0); d = (d + d.T) / 2
            Z = linkage(squareform(d, checks=False), method="average")
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
            dendrogram(Z, labels=list(P.index), ax=a1, leaf_rotation=90, color_threshold=0)
            a1.set_title("Network correlation dendrogram"); a1.set_ylabel("1 - Pearson(W)")
            leaves = dendrogram(Z, no_plot=True)["leaves"]
            Po = P.values[np.ix_(leaves, leaves)]
            im = a2.imshow(Po, cmap="RdBu_r", vmin=-1, vmax=1)
            a2.set_xticks(range(len(leaves))); a2.set_xticklabels([P.index[i] for i in leaves], rotation=90, fontsize=7)
            a2.set_yticks(range(len(leaves))); a2.set_yticklabels([P.index[i] for i in leaves], fontsize=7)
            a2.set_title("Ordered network-similarity heatmap"); fig.colorbar(im, ax=a2, fraction=0.046)
            return fig
        _try(report, name, "C1_network_similarity.png", _dend,
             "Cell-type networks cluster by lineage; the ordered heatmap shows the block structure.")

    report.sub("3.2 Cell-type-specific hub genes",
               "A few global hubs (e.g. Myc, Gata1) are central in every cell type, so raw "
               "top-centrality lists look alike. Here each panel shows the genes whose degree "
               "centrality is highest *relative to the cross-cell-type mean* -- the hubs that "
               "are specific to that cell type. One panel per cell type, in its color.")
    def _cent_panels():
        metric = "degree_centrality_all"
        M = pd.DataFrame({c: adata.var.get(f"{metric}_{c}",
                                           pd.Series(0.0, index=adata.var_names)).fillna(0)
                          for c in order})
        mean = M.mean(axis=1)
        ncl = len(order); ncol = min(4, ncl); nrow = int(np.ceil(ncl / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
        for k, c in enumerate(order):
            ax = axes[k // ncol][k % ncol]
            diff = (M[c] - mean).sort_values(ascending=False).head(10).iloc[::-1]
            ax.barh(range(len(diff)), diff.values, color=colors.get(c, "#3b6ea5"))
            ax.set_yticks(range(len(diff))); ax.set_yticklabels(diff.index, fontsize=7)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(c, fontsize=9, color=colors.get(c, "k")); ax.tick_params(labelsize=6)
        for k in range(ncl, nrow * ncol):
            axes[k // ncol][k % ncol].set_visible(False)
        fig.suptitle("Cell-type-specific hubs (centrality above the cross-cell-type mean)",
                     fontweight="bold")
        fig.tight_layout()
        return fig
    _try(report, name, "C2_centrality_by_celltype.png", _cent_panels,
         "Cell-type-specific hub genes (degree centrality relative to the cross-type mean).")

    report.sub("3.3 Centrality-measure comparison",
               "Eigenvector centrality (global regulatory influence) vs betweenness centrality "
               "(bottleneck / information-flow role). These two capture complementary notions "
               "of importance -- an influential hub is not necessarily a bottleneck -- so they "
               "separate broad master regulators from pathway connectors better than in-vs-out "
               "degree. Colored by cell type.")
    _try(report, name, "C3_centrality_scatter.png",
         lambda: sch.pl.plot_centrality_scatter(adata, x_metric="eigenvector_centrality",
                                                y_metric="betweenness_centrality",
                                                cluster_key=ck, order=order, colors=colors),
         "Eigenvector vs betweenness centrality per cell type.")

    report.sub("3.4 Network eigenanalysis",
               "Spectral structure of `W_c`: dominant eigenvalue loadings and eigenvalue spectra.")
    def _eig_grid():
        out = sch.pl.plot_eigenanalysis_grid(adata, cluster_key=ck, order=order, n_genes=10, colors=colors)
        fig = out if hasattr(out, "savefig") else (out.figure if hasattr(out, "figure") else plt.gcf())
        for ax in fig.axes:
            lo, hi = ax.get_ylim(); m = max(abs(lo), abs(hi))
            if m > 0:
                ax.set_ylim(-m, m)
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        return fig
    _try(report, name, "C4_eigen_grid.png", _eig_grid,
         "Leading eigenvector gene loadings per cell type (symmetric y-axis).")
    _try(report, name, "C5_eigenvalue_spectrum.png",
         lambda: sch.pl.plot_eigenvalue_spectrum(adata, cluster_key=ck, colors=colors),
         "Eigenvalue spectra of the cell-type interaction matrices.")

    report.sub("3.5 GRN visualization",
               "The inferred wiring for representative cell types (top-weight edges).")
    show = [order[0]]
    lin = cfg.get("lineages")
    if lin:
        for grp in (lin["A"], lin["B"]):
            for c in grp:
                if c in clusters:
                    show.append(c); break
    show = list(dict.fromkeys(show))[:3]
    for c in show:
        _try(report, name, f"C6_grn_{c}.png".replace("/", "_"),
             lambda cc=c: sch.pl.plot_grn_network(adata, cc, cluster_key=ck, topn=30, w_quantile=0.98),
             f"Inferred GRN for '{c}' (top edges).")


# --------------------------------------------------------------------------- #
# Section D: local stability (Jacobian)
# --------------------------------------------------------------------------- #
def section_D(adata, name, ck, report, cfg):
    basis = basis_of(adata)
    colors = get_colors(adata, ck)
    clusters = present_clusters(adata, ck)
    order = order_by_trajectory(adata, ck, clusters, cfg.get("lineages"), explicit=cfg.get("order"))
    report.section(
        "4. Local stability (Jacobian analysis)",
        "The Jacobian at each cell governs local dynamics; its leading eigenvalue measures "
        "(in)stability, the imaginary part and the antisymmetric (rotational) part measure "
        "rotational flow."
    )

    report.sub("4.1 Stability over the embedding")
    for col, lab, cmap in [("jacobian_leading_real", "leading eig (Re)", "RdBu_r"),
                           ("jacobian_positive_evals", "# positive eigenvalues", "viridis"),
                           ("jacobian_rotational", "rotational magnitude", "magma")]:
        if col not in adata.obs:
            continue
        def _emb(c=col, l=lab, cm=cmap):
            fig, ax = plt.subplots(figsize=(6, 5))
            kw = {}
            if c == "jacobian_leading_real":
                lim = float(np.nanpercentile(np.abs(adata.obs[c].values), 98))
                kw = dict(vmin=-lim, vmax=lim)  # symmetric about 0
            sc.pl.embedding(adata, basis=basis, color=c, ax=ax, show=False, frameon=False,
                            cmap=cm, size=25, title=f"{name}: {l}", **kw)
            return fig
        _try(report, name, f"D1_{col}.png", _emb, f"{lab} over the embedding.")

    report.sub("4.2 Jacobian eigenvalue spectra and statistics",
               "Per-cell-type spectra (real vs imaginary) and summary statistics, colored by "
               "cell type; the min/max real-eigenvalue axis is symmetric about zero.")
    _try(report, name, "D2_jac_eig_spectrum.png",
         lambda: sch.pl.plot_jacobian_eigenvalue_spectrum(adata, cluster_key=ck, order=order, colors=colors),
         "Jacobian eigenvalue spectra per cell type (real vs imaginary part).")

    def _stats():
        out = sch.pl.plot_jacobian_stats_boxplots(adata, cluster_key=ck, order=order, colors=colors)
        # symmetric y-axis on any min/max real-eigenvalue panels
        fig = out if hasattr(out, "savefig") else (out.figure if hasattr(out, "figure") else plt.gcf())
        for ax in fig.axes:
            t = (ax.get_title() or "").lower()
            if "eig" in t and ("min" in t or "max" in t or "real" in t):
                lo, hi = ax.get_ylim(); m = max(abs(lo), abs(hi))
                ax.set_ylim(-m, m); ax.axhline(0, color="k", lw=0.6, ls="--")
        return fig
    _try(report, name, "D3_jac_stats_box.png", _stats,
         "Per-cell-type Jacobian statistics (leading eig with symmetric axis, trace, #unstable, rotational).")

    report.sub("4.3 Element-wise Jacobian dynamics",
               "How specific regulator->target sensitivities vary across the trajectory, for "
               "the strongest hub genes.")
    import itertools as _it
    hubs = _top_by_centrality(adata, ck, clusters, n=8)
    vn = list(adata.var_names)

    def _edge(a_, b_):
        gi, gj = vn.index(a_), vn.index(b_)
        return any(abs(np.asarray(adata.varp[f"W_{c}"])[gi, gj]) > 1e-8
                   for c in clusters if f"W_{c}" in adata.varp)
    pairs = [p for p in _it.permutations(hubs, 2) if _edge(*p)][:9] if len(hubs) >= 4 else []
    if pairs:
        try:
            sch.tl.compute_jacobian_elements(adata, gene_pairs=pairs, cluster_key=ck)
            # plot_jacobian_element_grid hard-codes X_umap; alias it to the report's basis
            # so paul15 (draw_graph_fa) uses the same embedding as the rest of the report.
            if "X_umap" not in adata.obsm:
                adata.obsm["X_umap"] = adata.obsm[f"X_{basis}"]
            def _je():
                out = sch.pl.plot_jacobian_element_grid(adata, gene_pairs=pairs, cluster_key=ck)
                fig = out if hasattr(out, "savefig") else (out.figure if hasattr(out, "figure") else plt.gcf())
                for ax in fig.axes:
                    if ax.collections:
                        ax.set_xticks([]); ax.set_yticks([])
                        for sp in ax.spines.values():
                            sp.set_visible(False)
                return fig
            _try(report, name, "D4_jac_elements.png", _je,
                 f"Jacobian element (regulator->target sensitivity) dynamics for {len(pairs)} hub pairs.")
        except Exception as exc:
            print(f"  [jac elements] {exc}", flush=True)


# --------------------------------------------------------------------------- #
# Section E: perturbation analysis
# --------------------------------------------------------------------------- #
def pareto_fronts(df, xcol="score_A", ycol="score_B", n_rounds=6):
    remaining = df[[xcol, ycol]].copy()
    ranks = {}
    for r in range(n_rounds):
        if remaining.empty:
            break
        X = remaining.values
        dominated = np.zeros(len(X), bool)
        for i in range(len(X)):
            dominated[i] = np.any(np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1))
        front = remaining.index[~dominated]
        for g in front:
            ranks[g] = r
        remaining = remaining.drop(index=front)
    return ranks


def _perturb(adata, cond, ck, basis):
    pert = sch.dyn.simulate_perturbation(adata, perturb_condition=cond, cluster_key=ck,
                                         n_propagation=3, verbose=False)
    sch.tl.calculate_flow(pert, source="delta", basis=basis, cluster_key=ck, verbose=False)
    return pert


def _lineage_pairs(adata, name, ck, cfg):
    """Return a list of (A,B,An,Bn) lineage pairs. Explicit config lineage_pairs win (e.g.
    differentiated-vs-progenitor for multifurcating datasets); else a single configured
    pair; else the two most network-distinct cluster pairs."""
    present = present_clusters(adata, ck)
    cfg_pairs = cfg.get("lineage_pairs")
    if cfg_pairs:
        out = []
        for lp in cfg_pairs:
            A = [c for c in lp["A"] if c in present]; B = [c for c in lp["B"] if c in present]
            if A and B:
                out.append((A, B, lp["A_name"], lp["B_name"]))
        if out:
            return out
    lin = cfg.get("lineages")
    if lin:
        from rutils import resolve_lineages
        A, B, An, Bn = resolve_lineages(adata, name)
        return [(A, B, An, Bn)]
    Ws = {c: np.asarray(adata.varp[f"W_{c}"]) for c in present if f"W_{c}" in adata.varp}
    cl = list(Ws)
    pairs = []
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            r = np.corrcoef(Ws[cl[i]].ravel(), Ws[cl[j]].ravel())[0, 1]
            pairs.append((r, cl[i], cl[j]))
    pairs.sort(key=lambda t: t[0])
    used = set(); out = []
    for r, a, b in pairs:
        if a in used or b in used:
            continue
        out.append(([a], [b], str(a), str(b))); used.update([a, b])
        if len(out) == 2:
            break
    return out or [([cl[0]], [cl[-1]], str(cl[0]), str(cl[-1]))]


def _perturb_section(adata, name, ck, report, cfg, A, B, An, Bn, tag):
    import itertools
    basis = basis_of(adata)
    colors = get_colors(adata, ck)
    wt_key = f"original_velocity_flow_{basis}"
    if wt_key not in adata.obsm:
        sch.tl.calculate_flow(adata, source="original", basis=basis, cluster_key=ck, verbose=False)

    # ---- pre-perturbation scores + Pareto selection ----
    report.sub(f"5.{tag}.1 Driver scores and Pareto selection ({An} vs {Bn})",
               "Structural driver score per lineage; candidates are the successive Pareto-optimal "
               "TFs plus the most lineage-biased genes (not the largest cluster).")
    tf = sch.tl.score_driver_tfs(adata, A, B, cluster_key=ck)
    ranks = pareto_fronts(tf, "score_A", "score_B", 6)
    tf["pareto_rank"] = tf.index.map(lambda g: ranks.get(g, np.nan))
    A_cand = list(tf[tf.lineage_bias > 0].sort_values("score_A", ascending=False).head(6).index)
    B_cand = list(tf[tf.lineage_bias <= 0].sort_values("score_B", ascending=False).head(6).index)
    candidates = [g for g in dict.fromkeys(A_cand + B_cand + (cfg.get("anchors") or [])) if g in adata.var_names]
    # explicitly featured perturbation genes (e.g. paul15: Gata1, Stat3 + two others),
    # filled up to four with top candidates when fewer are configured
    forced = [g for g in (cfg.get("perturb_genes") or []) if g in adata.var_names]
    featured = list(dict.fromkeys(forced + [g for g in (A_cand + B_cand) if g not in forced]))[:4] if forced else None

    def _pareto():
        from adjustText import adjust_text
        fig, ax = plt.subplots(figsize=(6.4, 5.8))
        ax.scatter(tf.score_A, tf.score_B, s=12, c="#cccccc", label="all genes")
        pr = tf.dropna(subset=["pareto_rank"])
        sctr = ax.scatter(pr.score_A, pr.score_B, s=48, c=pr.pareto_rank, cmap="viridis_r",
                          edgecolor="k", linewidth=0.3)
        texts = [ax.text(tf.loc[g, "score_A"], tf.loc[g, "score_B"], g, fontsize=8)
                 for g in candidates if g in tf.index]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        ax.set(xlabel=f"score_A ({An})", ylabel=f"score_B ({Bn})",
               title=f"{name}: driver scores + Pareto fronts")
        fig.colorbar(sctr, ax=ax, label="Pareto front (0=best)")
        return fig
    _try(report, name, f"E{tag}_1_pareto.png", _pareto,
         f"Driver scores with Pareto fronts; labels de-overlapped. A={A_cand}, B={B_cand}.")
    tf.to_csv(f"{ROOT_}/{name}/data/driver_scores_{tag}.csv")

    # ---- single-KO screen, barplot.png style ----
    report.sub(f"5.{tag}.2 Single-KO lineage bias",
               "Post-KO lineage bias for every candidate, sorted; top-2 per lineage highlighted.")
    single_bias = {}
    try:
        single_bias, _ = sch.dyn.run_ko_screen(adata, candidates, A, B, basis, wt_key,
                                               cluster_key=ck, verbose=False)
        sb = pd.Series({g: v["lineage_bias"] for g, v in single_bias.items()}).sort_values()
        topA = list(sb[sb > 0].tail(2).index); topB = list(sb[sb < 0].head(2).index)
        def _bar():
            fig, ax = plt.subplots(figsize=(max(7, 0.32 * len(sb)), 4.6))
            cols = []
            for g, v in sb.items():
                if g in topA: cols.append("#8B2E2E")
                elif g in topB: cols.append("#1F3B73")
                elif v >= 0: cols.append(GRP["A"])
                else: cols.append(GRP["B"])
            ax.bar(range(len(sb)), sb.values, color=cols)
            ax.set_xticks(range(len(sb))); ax.set_xticklabels(sb.index, rotation=60, ha="right", fontsize=7)
            ax.axhline(0, color="k", lw=0.8, ls="--")
            for g in topA + topB:
                i = list(sb.index).index(g)
                ax.annotate(g, (i, sb[g]), ha="center",
                            va="bottom" if sb[g] >= 0 else "top", fontsize=8, fontweight="bold")
            import matplotlib.patches as mp
            ax.legend(handles=[mp.Patch(color="#8B2E2E", label=f"top-2 {An}-biasing"),
                               mp.Patch(color="#1F3B73", label=f"top-2 {Bn}-biasing"),
                               mp.Patch(color=GRP["A"], label=f"other {An}-biasing"),
                               mp.Patch(color=GRP["B"], label=f"other {Bn}-biasing")], fontsize=7)
            ax.set(ylabel=f"lineage bias of KO (+{An} / -{Bn})", title=f"{name}: single-KO lineage bias")
            return fig
        _try(report, name, f"E{tag}_2_single_ko_bias.png", _bar, "Single-KO lineage bias (sorted).")
    except Exception as exc:
        print(f"  [E{tag}.2] {exc}", flush=True)

    # ---- KO vs OE grid (nb05 style) for the strongest candidate ----
    g0 = ((featured or [])[:1] or A_cand[:1] or candidates[:1])[0]
    report.sub(f"5.{tag}.3 KO vs OE flows and velocity alignment ({g0})",
               "Cell types, reference velocity, KO and OE perturbation flows (streamlines), and "
               "their inner product with the developmental velocity (nb05-style panel).")
    try:
        gmax = float(np.percentile(np.asarray(adata[:, g0].layers["Ms"]).ravel(), 99))
        ko = _perturb(adata, {g0: 0.0}, ck, basis)
        oe = _perturb(adata, {g0: 2 * gmax}, ck, basis)
        for pert in (ko, oe):
            pert.obsm[wt_key] = adata.obsm[wt_key]
            sch.tl.calculate_inner_product(pert, flow_key_1=f"perturbation_flow_{basis}",
                                           flow_key_2=wt_key, store_key="ip")
        def _grid():
            fig, ax = plt.subplots(3, 2, figsize=(12, 15))
            emb = np.asarray(adata.obsm[f"X_{basis}"])[:, :2]
            ax[0, 0].scatter(emb[:, 0], emb[:, 1], c=[colors.get(str(c), "#ccc") for c in adata.obs[ck].astype(str)], s=8, alpha=0.6)
            ax[0, 0].set_title("cell types"); ax[0, 0].set_axis_off()
            _stream(ax[0, 1], adata, ck, colors, basis, fitted_velocity(adata, ck), title="reference velocity")
            _stream(ax[1, 0], ko, ck, colors, basis, np.asarray(ko.layers["delta_X"]), title=f"{g0} KO flow")
            _stream(ax[1, 1], oe, ck, colors, basis, np.asarray(oe.layers["delta_X"]), title=f"{g0} OE flow")
            for a2, pert, lab in [(ax[2, 0], ko, "KO"), (ax[2, 1], oe, "OE")]:
                try:
                    sch.pl.plot_inner_product(pert, basis=basis, ax=a2, inner_product_key="ip",
                                              on_grid=True, n_grid=40, min_mass=25)
                except TypeError:
                    sch.pl.plot_inner_product(pert, basis=basis, ax=a2, inner_product_key="ip")
                a2.set_title(f"{g0} {lab} inner product"); a2.set_axis_off()
            fig.tight_layout()
            return fig
        _try(report, name, f"E{tag}_3_ko_oe_grid.png", _grid,
             f"{g0}: cell types, reference velocity, KO/OE flows, and their velocity alignment.")
    except Exception as exc:
        print(f"  [E{tag}.3] {exc}", flush=True)

    # ---- per-gene KO flow grid, colored by group (nb06 style) ----
    report.sub(f"5.{tag}.4 KO flows for top candidates (per gene)",
               "KO-induced flow for the top candidates of each lineage, colored by group "
               f"(red={An}, blue={Bn}, orange/purple=lineage-biased).")
    if featured:
        genes_grp = [(g, GRP["A"] if tf.loc[g, "lineage_bias"] > 0 else GRP["B"]) for g in featured]
    else:
        genes_grp = ([(g, GRP["A"]) for g in A_cand[:3]] + [(g, GRP["B"]) for g in B_cand[:3]])
    genes_grp = [(g, c) for g, c in genes_grp if g in adata.var_names]
    def _grid_flows():
        n = len(genes_grp); ncol = 3; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.6 * nrow), squeeze=False)
        for k, (g, col) in enumerate(genes_grp):
            axx = axes[k // ncol][k % ncol]
            p = sch.dyn.simulate_perturbation(adata, {g: 0.0}, cluster_key=ck, n_propagation=3, verbose=False)
            _stream(axx, p, ck, colors, basis, np.asarray(p.layers["delta_X"]), title=f"{g} KO")
        for k in range(n, nrow * ncol):
            axes[k // ncol][k % ncol].set_visible(False)
        fig.tight_layout()
        return fig
    _try(report, name, f"E{tag}_4_grid_ko_flows.png", _grid_flows,
         "KO flows for the top candidates of each lineage.")

    # ---- pairwise KO + combined epistasis/cancellation heatmap (heatmap.png style) ----
    report.sub(f"5.{tag}.5 Pairwise KO and epistasis",
               "Combined matrix over top shifters: upper triangle = double-KO lineage bias, "
               "lower triangle = cancellation error (epistasis). Genes ordered by lineage.")
    try:
        AA, BB = A_cand[:5], B_cand[:5]
        genes = AA + BB
        pairs = list(itertools.combinations(genes, 2))
        pair_bias, _ = sch.dyn.run_pairwise_ko_screen(adata, pairs, A, B, basis, wt_key,
                                                      cluster_key=ck, verbose=False)
        epi = sch.dyn.compute_epistasis(pair_bias, single_bias, lineage_A_genes=AA, lineage_B_genes=BB)
        epi.to_csv(f"{ROOT_}/{name}/data/epistasis_{tag}.csv")
        idx = {g: i for i, g in enumerate(genes)}
        n = len(genes)
        bias_m = np.full((n, n), np.nan); canc_m = np.full((n, n), np.nan)
        for g in genes:
            if g in single_bias:
                bias_m[idx[g], idx[g]] = single_bias[g]["lineage_bias"]
        for _, r in epi.iterrows():
            if r.geneA in idx and r.geneB in idx:
                i, j = idx[r.geneA], idx[r.geneB]
                bias_m[min(i, j), max(i, j)] = r["lineage_bias"]     # upper
                canc_m[max(i, j), min(i, j)] = r["cancellation_error"]  # lower
        def _combined():
            fig, ax = plt.subplots(figsize=(8.5, 7.5))
            lb = np.nanmax(np.abs(bias_m)) or 1.0; lc = np.nanmax(np.abs(canc_m)) or 1.0
            im1 = ax.imshow(np.where(np.triu(np.ones((n, n)), 0) > 0, bias_m, np.nan),
                            cmap="RdBu_r", vmin=-lb, vmax=lb)
            im2 = ax.imshow(np.where(np.tril(np.ones((n, n)), -1) > 0, canc_m, np.nan),
                            cmap="PRGn", vmin=-lc, vmax=lc)
            ax.axhline(len(AA) - 0.5, color="k", lw=1.5); ax.axvline(len(AA) - 0.5, color="k", lw=1.5)
            # dashed staircase along the diagonal separating the two triangles
            xs, ys = [], []
            for i in range(n):
                xs += [i - 0.5, i + 0.5, i + 0.5]; ys += [i - 0.5, i - 0.5, i + 0.5]
            ax.plot(xs, ys, color="k", lw=1.0, ls="--")
            ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
            ax.set_xticks(range(n)); ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(n)); ax.set_yticklabels(genes, fontsize=7)
            for t, g in zip(ax.get_xticklabels(), genes):
                t.set_color(GRP["A"] if g in AA else GRP["B"])
            for t, g in zip(ax.get_yticklabels(), genes):
                t.set_color(GRP["A"] if g in AA else GRP["B"])
            ax.set_title(f"{name}: perturbation matrix (upper=lineage bias, lower=cancellation)")
            fig.subplots_adjust(right=0.80)
            cax1 = fig.add_axes([0.83, 0.54, 0.022, 0.34])
            cax2 = fig.add_axes([0.83, 0.12, 0.022, 0.34])
            fig.colorbar(im1, cax=cax1).set_label("lineage bias (upper)", fontsize=8)
            fig.colorbar(im2, cax=cax2).set_label("cancellation error (lower)", fontsize=8)
            return fig
        _try(report, name, f"E{tag}_5_epistasis_matrix.png", _combined,
             "Double-KO lineage bias (upper) and cancellation-error epistasis (lower) in one matrix.")
    except Exception as exc:
        print(f"  [E{tag}.5] {exc}", flush=True)

    # ---- energy change ----
    report.sub(f"5.{tag}.6 Energy change under KO",
               f"Per-cell-type change in total energy when {g0} is knocked out.")
    def _de():
        order2 = order_by_trajectory(adata, ck, present_clusters(adata, ck), cfg.get("lineages"), explicit=cfg.get("order"))
        k = sch.dyn.simulate_perturbation(adata, {g0: 0.0}, cluster_key=ck, n_propagation=3, verbose=False)
        e0 = adata.obs["energy_total"].values
        kk = k.copy(); kk.layers["Ms"] = kk.layers["simulated_count"]
        sch.pp.compute_sigmoid(kk, spliced_key="Ms")  # recompute activation from perturbed state
        sch.tl.compute_energies(kk, cluster_key=ck)
        de = kk.obs["energy_total"].values - e0
        lab = adata.obs[ck].astype(str)
        vals = [float(np.nanmean(de[(lab == c).values])) for c in order2]
        fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(order2), 4))
        ax.bar(range(len(order2)), vals, color=[colors.get(c, "#8d6a9f") for c in order2])
        ax.set_xticks(range(len(order2))); ax.set_xticklabels(order2, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(ylabel="mean delta energy", title=f"{g0} KO: energy change per cell type")
        return fig
    _try(report, name, f"E{tag}_6_energy_change.png", _de, f"{g0} KO energy change per cell type.")

    # ---- dose-response for several genes ----
    report.sub(f"5.{tag}.7 Dose-response (several genes)",
               "Fractional dose-response (0 = KO, 1 = natural, 2 = strong OE) for several drivers.")
    dose_genes = [g for g in dict.fromkeys((cfg.get("anchors") or []) + A_cand[:2] + B_cand[:1]) if g in adata.var_names][:4]
    def _dose():
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        for g in dose_genes:
            dr = sch.dyn.run_dose_response(adata, g, lineage_A_clusters=A, lineage_B_clusters=B,
                                           basis=basis, wt_flow_key=wt_key,
                                           fractions=[0, 0.25, 0.5, 0.75, 1, 1.5, 2],
                                           cluster_key=ck, verbose=False)
            ax.plot(dr.level_frac, dr.lineage_bias, "-o", ms=4, label=g)
        ax.axhline(0, color="k", lw=0.6); ax.axvline(1, color="grey", ls="--", lw=0.8)
        ax.set(xlabel="dose (fraction of natural max)", ylabel=f"lineage bias (+{An}/-{Bn})",
               title=f"{name}: dose-response"); ax.legend(fontsize=8)
        return fig
    if dose_genes:
        _try(report, name, f"E{tag}_7_dose_response.png", _dose,
             f"Dose-response of the lineage bias for {dose_genes}.")

    # ---- anchor-partner double KOs (nb07 style: multiple anchors, more partners) ----
    report.sub(f"5.{tag}.8 Anchor-partner double-KO recipes",
               "For each anchor driver, its top regulatory partners (grn_partner_weights) are "
               "screened as anchor+partner double KOs -- a directed search for effective "
               "genetic interactions (nb07 recipe).")
    anchors = [a for a in (cfg.get("anchors") or (A_cand[:1] + B_cand[:1])) if a in adata.var_names][:2]
    def _recipe():
        fig, axes = plt.subplots(1, len(anchors), figsize=(6.2 * len(anchors), 4.8), squeeze=False)
        for j, anchor in enumerate(anchors):
            pw = sch.tl.grn_partner_weights(adata, anchor)
            partners = [g for g in pw.abs().mean(axis=1).sort_values(ascending=False).index if g != anchor][:8]
            apairs = [(anchor, p) for p in partners]
            ab, _ = sch.dyn.run_pairwise_ko_screen(adata, apairs, A, B, basis, wt_key, cluster_key=ck, verbose=False)
            s = pd.Series({p: ab[(anchor, p)]["lineage_bias"] for p in partners if (anchor, p) in ab}).sort_values()
            ax = axes[0][j]
            ax.barh(range(len(s)), s.values, color=[GRP["A"] if v >= 0 else GRP["B"] for v in s.values])
            ax.set_yticks(range(len(s))); ax.set_yticklabels([f"{anchor}+{p}" for p in s.index], fontsize=8)
            ax.axvline(0, color="k", lw=0.8); ax.set(xlabel="double-KO lineage bias", title=f"anchor {anchor}")
        fig.tight_layout()
        return fig
    if anchors:
        _try(report, name, f"E{tag}_8_anchor_partner.png", _recipe,
             f"Anchor-partner double KOs for anchors {anchors} (8 partners each).")


def section_E(adata, name, ck, report, cfg):
    report.section(
        "5. Perturbation analysis",
        "In-silico perturbations on the fitted GRN. Candidate genes and cell types are chosen "
        "principledly (lineage driver score + Pareto optimality), not by picking the largest "
        "cluster. For datasets without an obvious two-lineage split, two different lineage "
        "pairs are analyzed."
    )
    pairs = _lineage_pairs(adata, name, ck, cfg)
    for t, (A, B, An, Bn) in enumerate(pairs, start=1):
        _perturb_section(adata, name, ck, report, cfg, A, B, An, Bn, tag=t)
