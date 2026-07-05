"""Figure sections for the per-dataset scHopfield reports (notebooks 01-08 pipeline).

Each ``section_*`` takes the analyzed AnnData, the dataset name, its cluster key, and a
Report, drives the canonical ``sch.pl.*`` / ``sch.tl.*`` functions, saves the figures,
and appends captioned, explained subsections to the report. Robust: any single figure
that fails is logged and skipped so one bad panel doesn't abort the report.
"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

import scHopfield as sch
from rutils import save, present_clusters, ROOT as ROOT_


def basis_of(adata):
    for b in ("draw_graph_fa", "umap", "pca", "diffmap"):
        if f"X_{b}" in adata.obsm:
            return b
    return "umap"


FORCE_FIGS = os.environ.get("REPORT_FORCE_FIGS", "") == "1"


def _try(report, name, fname, fn, caption):
    """Run a figure-producing fn (returns a Matplotlib fig/ax or None -> gcf), save, embed.
    Skips regeneration if the PNG already exists (unless REPORT_FORCE_FIGS=1), so the
    report can be reassembled cheaply without re-running slow simulations."""
    rel = f"plots/{fname}"
    if not FORCE_FIGS and os.path.exists(f"{ROOT_}/{name}/{rel}"):
        report.img(rel, caption)
        return True
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


# --------------------------------------------------------------------------- #
# Section A: overview -- embedding, velocities, Hill fitting
# --------------------------------------------------------------------------- #
def section_A(adata, name, ck, report, cfg):
    basis = basis_of(adata)
    report.section(
        "1. Overview: embedding, dynamics, and Hill fitting",
        f"Dataset **{name}** ({adata.n_obs} cells x {adata.n_vars} genes, {len(present_clusters(adata, ck))} "
        f"cell types). Dynamics from **{cfg['velocity_mode']}** "
        f"({'pseudotime-inferred velocity' if cfg['velocity_mode']=='pseudotime' else 'RNA velocity'})."
    )

    report.sub("1.1 Cell types on the embedding")
    def _ct():
        fig, ax = plt.subplots(figsize=(6, 5.2))
        sc.pl.embedding(adata, basis=basis, color=ck, ax=ax, show=False, frameon=False,
                        legend_loc="right margin", size=25)
        return fig
    _try(report, name, "A1_celltypes.png", _ct,
         f"Cell types on the {basis} embedding.")

    report.sub("1.2 Dynamics on the embedding",
               "Hopfield velocity field (from the fitted model) projected to the embedding.")
    def _flow():
        sch.tl.calculate_flow(adata, source="original", basis=basis, cluster_key=ck, verbose=False)
        fig, ax = plt.subplots(figsize=(6, 5.2))
        sch.pl.plot_flow(adata, flow_key=f"original_velocity_flow_{basis}", basis=basis,
                         ax=ax, on_grid=True, cluster_key=ck)
        ax.set_title(f"{name}: model velocity field")
        return fig
    _try(report, name, "A2_velocity_flow.png", _flow,
         "Model-inferred velocity field on the embedding (grid-averaged).")

    report.sub("1.3 Hill (sigmoid) fitting and goodness of fit",
               "Each gene's activation is a Hill function fit to its expression CDF. "
               "Goodness of fit is the per-gene CDF MSE (`sigmoid_mse`, lower = better).")
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
             f"Distribution of the Hill-fit MSE across genes (median {np.median(mse):.4f}; "
             "most genes fit well).")
        # example fits: best + worst
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
             "Example Hill fits: the two best- and two worst-fit genes by MSE.")


# --------------------------------------------------------------------------- #
# Section B: energy landscape
# --------------------------------------------------------------------------- #
def section_B(adata, name, ck, report, cfg):
    basis = basis_of(adata)
    report.section(
        "2. Energy landscape",
        "The fitted model defines a Lyapunov energy per cell, decomposed into interaction "
        "(-1/2 sWs), degradation, and bias (-I^T s) terms. After the scaffold + L1 fit the "
        "bias term is negligible, so the landscape is carried by interactions and degradation."
    )

    report.sub("2.1 Energy by cell type (all components)")
    _try(report, name, "B1_energy_boxplots.png",
         lambda: sch.pl.plot_energy_boxplots(adata, cluster_key=ck, plot_energy="all"),
         "Per-cell-type distribution of total / interaction / degradation / bias energy.")

    report.sub("2.2 Energy landscape on the embedding (all components)")
    _try(report, name, "B2_energy_scatters.png",
         lambda: sch.pl.plot_energy_scatters(adata, cluster_key=ck, basis=basis, plot_energy="all"),
         "Each energy component painted on the embedding.")

    report.sub("2.3 Energy-gene correlations",
               "Pearson correlation of each gene's expression with the energy, per cell type. "
               "It is a descriptive readout (which genes track landscape depth), not a causal "
               "driver score, and it feeds the structural driver score in Section 5.")
    for etype in ("total", "interaction", "bias"):
        _try(report, name, f"B3_corr_grid_{etype}.png",
             lambda e=etype: sch.pl.plot_correlations_grid(adata, cluster_key=ck, energy=e),
             f"Gene-vs-{etype}-energy correlations across cell types.")


# --------------------------------------------------------------------------- #
# Section C: regulatory network structure
# --------------------------------------------------------------------------- #
def _top_by_centrality(adata, ck, clusters, metric="degree_centrality_out", n=4):
    import pandas as pd
    scores = pd.Series(0.0, index=adata.var_names)
    k = 0
    for c in clusters:
        col = f"{metric}_{c}"
        if col in adata.var.columns:
            scores = scores.add(adata.var[col].fillna(0), fill_value=0); k += 1
    return list(scores.nlargest(n).index)


def section_C(adata, name, ck, report, cfg):
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    clusters = present_clusters(adata, ck)
    report.section(
        "3. Regulatory network structure",
        "Cell-type-specific GRNs (`W_c`): how similar the networks are across cell types, "
        "which genes are hubs, the spectral structure, and the wiring itself."
    )

    # 3.1 network-similarity dendrogram + a clustermap alternative
    report.sub("3.1 Cell-type network similarity",
               "How related the fitted networks are across cell types (Pearson correlation of "
               "the flattened `W_c`). Left: the classic dendrogram; right: a hierarchically "
               "ordered heatmap (a clearer alternative that also shows the pairwise structure).")
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
            order = dendrogram(Z, no_plot=True)["leaves"]
            Po = P.values[np.ix_(order, order)]
            im = a2.imshow(Po, cmap="RdBu_r", vmin=-1, vmax=1)
            a2.set_xticks(range(len(order))); a2.set_xticklabels([P.index[i] for i in order], rotation=90, fontsize=7)
            a2.set_yticks(range(len(order))); a2.set_yticklabels([P.index[i] for i in order], fontsize=7)
            a2.set_title("Ordered network-similarity heatmap"); fig.colorbar(im, ax=a2, fraction=0.046)
            return fig
        _try(report, name, "C1_network_similarity.png", _dend,
             "Cell-type networks cluster by lineage; the ordered heatmap shows the block structure.")

    # 3.2 symmetricity
    report.sub("3.2 Network asymmetry (directedness)",
               "GRNs are directed; the asymmetry index ||W-W^T||/||W+W^T|| quantifies how far "
               "each cell-type network is from symmetric.")
    def _sym():
        sym = [float(np.linalg.norm(adata.varp[f"W_{c}"] - adata.varp[f"W_{c}"].T) /
                     (np.linalg.norm(adata.varp[f"W_{c}"] + adata.varp[f"W_{c}"].T) + 1e-12)) for c in clusters]
        fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(clusters), 4))
        ax.bar(range(len(clusters)), sym, color="#e76f51")
        ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=7)
        ax.set(ylabel="asymmetry index", title="GRN asymmetry per cell type")
        return fig
    _try(report, name, "C2_asymmetry.png", _sym, "Per-cell-type network asymmetry.")

    # 3.3 centrality
    report.sub("3.3 Network centrality",
               "Hub genes per cell type (igraph degree / eigenvector / betweenness, "
               "regulator->target orientation, weighted by |W|).")
    _try(report, name, "C3_centrality_rank.png",
         lambda: sch.pl.plot_network_centrality_rank(adata, metric="degree_centrality_all",
                                                     cluster_key=ck, n_genes=30),
         "Genes ranked by total degree centrality, per cell type.")
    _try(report, name, "C4_centrality_scatter.png",
         lambda: sch.pl.plot_centrality_scatter(adata, x_metric="degree_centrality_in",
                                                y_metric="degree_centrality_out", cluster_key=ck),
         "In- vs out-degree centrality per cell type (hubs vs targets).")

    # 3.4 eigenanalysis
    report.sub("3.4 Network eigenanalysis",
               "Spectral structure of `W_c`: dominant eigenvalues (amplification) and their "
               "eigenvector gene loadings.")
    _try(report, name, "C5_eigen_grid.png",
         lambda: sch.pl.plot_eigenanalysis_grid(adata, cluster_key=ck, n_genes=10),
         "Leading eigenvector gene loadings per cell type.")
    _try(report, name, "C6_eigenvalue_spectrum.png",
         lambda: sch.pl.plot_eigenvalue_spectrum(adata, cluster_key=ck),
         "Eigenvalue spectra of the cell-type interaction matrices.")

    # 3.5 GRN visualization
    report.sub("3.5 GRN visualization",
               "The inferred wiring for representative cell types (top-weight edges).")
    lin = cfg.get("lineages")
    show = [clusters[0]]
    if lin:
        for grp in (lin["A"], lin["B"]):
            for c in grp:
                if c in clusters:
                    show.append(c); break
    show = list(dict.fromkeys(show))[:3]
    for c in show:
        _try(report, name, f"C7_grn_{c}.png".replace("/", "_"),
             lambda cc=c: sch.pl.plot_grn_network(adata, cc, cluster_key=ck, topn=30, w_quantile=0.98),
             f"Inferred GRN for '{c}' (top edges).")


# --------------------------------------------------------------------------- #
# Section D: local stability (Jacobian)
# --------------------------------------------------------------------------- #
def section_D(adata, name, ck, report, cfg):
    basis = basis_of(adata)
    clusters = present_clusters(adata, ck)
    report.section(
        "4. Local stability (Jacobian analysis)",
        "The Jacobian J = W diag(sigma') - diag(gamma) at each cell governs local dynamics. "
        "Its leading eigenvalue measures (in)stability; the imaginary part and the "
        "antisymmetric (rotational) part measure rotational flow along the trajectory."
    )

    report.sub("4.1 Stability over the embedding",
               "Leading Jacobian eigenvalue (max real part; >0 = locally unstable, expected "
               "for cells in motion), positive-eigenvalue count, and rotational magnitude.")
    for col, lab, cmap in [("jacobian_leading_real", "leading eig (Re)", "RdBu_r"),
                           ("jacobian_positive_evals", "# positive eigenvalues", "viridis"),
                           ("jacobian_rotational", "rotational magnitude", "magma")]:
        if col not in adata.obs:
            continue
        def _emb(c=col, l=lab, cm=cmap):
            fig, ax = plt.subplots(figsize=(6, 5))
            sc.pl.embedding(adata, basis=basis, color=c, ax=ax, show=False, frameon=False,
                            cmap=cm, size=25, title=f"{name}: {l}")
            return fig
        _try(report, name, f"D1_{col}.png", _emb, f"{lab} over the embedding.")

    report.sub("4.2 Jacobian eigenvalue spectra and statistics",
               "Per-cell-type eigenvalue spectra (real vs imaginary) and summary statistics.")
    _try(report, name, "D2_jac_eig_spectrum.png",
         lambda: sch.pl.plot_jacobian_eigenvalue_spectrum(adata, cluster_key=ck),
         "Jacobian eigenvalue spectra per cell type (real vs imaginary part).")
    _try(report, name, "D3_jac_stats_box.png",
         lambda: sch.pl.plot_jacobian_stats_boxplots(adata, cluster_key=ck),
         "Per-cell-type Jacobian statistics (leading eig, trace, #unstable, rotational).")

    report.sub("4.3 Element-wise Jacobian dynamics",
               "How specific regulator->target sensitivities vary across the trajectory, for "
               "the strongest hub genes.")
    hubs = _top_by_centrality(adata, ck, clusters, n=4)
    pairs = [(hubs[0], hubs[1]), (hubs[2], hubs[3]), (hubs[0], hubs[3])] if len(hubs) >= 4 else []
    if pairs:
        try:
            sch.tl.compute_jacobian_elements(adata, gene_pairs=pairs, cluster_key=ck)
            _try(report, name, "D4_jac_elements.png",
                 lambda: sch.pl.plot_jacobian_element_grid(adata, gene_pairs=pairs, cluster_key=ck),
                 f"Jacobian element dynamics for hub pairs {pairs}.")
        except Exception as exc:
            print(f"  [jac elements] {exc}", flush=True)


# --------------------------------------------------------------------------- #
# Section E: perturbation analysis
# --------------------------------------------------------------------------- #
def pareto_fronts(df, xcol="score_A", ycol="score_B", n_rounds=6):
    """Successive non-dominated (Pareto) fronts on (xcol, ycol); returns
    {gene: front_rank} for genes on the first n_rounds fronts (rank 0 = best)."""
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
    """Simulate a perturbation and return the pert AnnData with its embedding delta flow."""
    pert = sch.dyn.simulate_perturbation(adata, perturb_condition=cond, cluster_key=ck,
                                         n_propagation=3, verbose=False)
    sch.tl.calculate_flow(pert, source="delta", basis=basis, cluster_key=ck, verbose=False)
    return pert


def section_E(adata, name, ck, report, cfg):
    import pandas as pd
    from rutils import resolve_lineages
    basis = basis_of(adata)
    A, B, An, Bn = resolve_lineages(adata, name)
    wt_key = f"original_velocity_flow_{basis}"
    if wt_key not in adata.obsm:
        sch.tl.calculate_flow(adata, source="original", basis=basis, cluster_key=ck, verbose=False)

    report.section(
        "5. Perturbation analysis",
        f"In-silico perturbations on the fitted GRN, with lineage choice **{An}** (A) vs "
        f"**{Bn}** (B). Candidate genes and cell types are chosen *principledly* -- by lineage "
        f"driver score and Pareto optimality -- not by picking the largest cluster."
    )

    # ---- 5.1 pre-perturbation scores + Pareto selection ----
    report.sub("5.1 Pre-perturbation driver scores and Pareto selection",
               "Structural driver score (`score_driver_tfs`: standardized W-norm + out-degree + "
               "energy correlation) for each lineage. Candidates are the successive Pareto-optimal "
               "TFs (non-dominated in both lineages) plus the most lineage-biased genes.")
    tf = sch.tl.score_driver_tfs(adata, A, B, cluster_key=ck)
    ranks = pareto_fronts(tf, "score_A", "score_B", n_rounds=6)
    tf["pareto_rank"] = tf.index.map(lambda g: ranks.get(g, np.nan))
    A_cand = list(tf[tf.lineage_bias > 0].sort_values("score_A", ascending=False).head(6).index)
    B_cand = list(tf[tf.lineage_bias <= 0].sort_values("score_B", ascending=False).head(6).index)
    candidates = list(dict.fromkeys(A_cand + B_cand + (cfg.get("anchors") or [])))
    candidates = [g for g in candidates if g in adata.var_names]

    def _pareto():
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        ax.scatter(tf.score_A, tf.score_B, s=12, c="#c9c9c9", label="all genes")
        pr = tf.dropna(subset=["pareto_rank"])
        sctr = ax.scatter(pr.score_A, pr.score_B, s=45, c=pr.pareto_rank, cmap="viridis_r",
                          edgecolor="k", linewidth=0.3, label="Pareto fronts")
        for g in candidates:
            if g in tf.index:
                ax.annotate(g, (tf.loc[g, "score_A"], tf.loc[g, "score_B"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel=f"score_A ({An})", ylabel=f"score_B ({Bn})",
               title=f"{name}: driver scores + Pareto fronts")
        fig.colorbar(sctr, ax=ax, label="Pareto front (0=best)"); ax.legend(fontsize=8)
        return fig
    _try(report, name, "E1_pareto_scores.png", _pareto,
         f"Structural driver scores; Pareto-optimal TFs (colored) are the principled candidates. "
         f"Selected: A={A_cand}, B={B_cand}.")
    tf.to_csv(f"figure_packs/reports/{name}/data/driver_scores.csv")

    def _bias_bar():
        top = tf.loc[candidates].sort_values("lineage_bias")
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.barh(range(len(top)), top.lineage_bias,
                color=["#2a6f97" if v >= 0 else "#bc4749" for v in top.lineage_bias])
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlabel=f"structural lineage bias (+{An}/-{Bn})", title="Candidate driver bias")
        return fig
    _try(report, name, "E2_candidate_bias.png", _bias_bar, "Structural lineage bias of the candidates.")

    # ---- 5.2 KO/OE flows + inner product for a top candidate ----
    g0 = A_cand[0] if A_cand else candidates[0]
    report.sub(f"5.2 Perturbation flows and velocity alignment ({g0})",
               "Velocity field vs the KO- and OE-induced delta_X flows, and their inner product "
               "with the developmental velocity (red = aligns with / green-blue = opposes fate).")
    try:
        gmax = float(np.percentile(np.asarray(adata[:, g0].layers["Ms"]).ravel(), 99))
        ko = _perturb(adata, {g0: 0.0}, ck, basis)
        oe = _perturb(adata, {g0: 2 * gmax}, ck, basis)
        for tag, pert in [("KO", ko), ("OE", oe)]:
            pert.obsm[wt_key] = adata.obsm[wt_key]
            sch.tl.calculate_inner_product(pert, flow_key_1=f"perturbation_flow_{basis}",
                                           flow_key_2=wt_key, store_key="ip")
        def _grid():
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            sch.pl.plot_flow(adata, flow_key=wt_key, basis=basis, ax=axes[0, 0], on_grid=True, cluster_key=ck)
            axes[0, 0].set_title("velocity field")
            sch.pl.plot_flow(ko, flow_key=f"perturbation_flow_{basis}", basis=basis, ax=axes[0, 1], on_grid=True, cluster_key=ck)
            axes[0, 1].set_title(f"{g0} KO delta_X flow")
            sch.pl.plot_inner_product(ko, basis=basis, ax=axes[1, 0], inner_product_key="ip")
            axes[1, 0].set_title(f"{g0} KO . velocity")
            sch.pl.plot_inner_product(oe, basis=basis, ax=axes[1, 1], inner_product_key="ip")
            axes[1, 1].set_title(f"{g0} OE . velocity")
            return fig
        _try(report, name, "E3_flows_innerproduct.png", _grid,
             f"{g0}: velocity, KO flow, and KO/OE alignment with the developmental velocity.")
    except Exception as exc:
        print(f"  [E3] {exc}", flush=True)

    # ---- 5.3 KO vs OE per-cluster symmetry ----
    report.sub("5.3 KO vs OE per-cluster symmetry",
               "Per-cell-type response magnitude to KO vs OE of each candidate. Points on the "
               "diagonal respond symmetrically; off-diagonal genes act mainly in one direction.")
    def _symm():
        order = present_clusters(adata, ck)
        rows = []
        for g in candidates[:6]:
            gm = float(np.percentile(np.asarray(adata[:, g].layers["Ms"]).ravel(), 99))
            k = sch.dyn.simulate_perturbation(adata, {g: 0.0}, cluster_key=ck, n_propagation=3, verbose=False)
            o = sch.dyn.simulate_perturbation(adata, {g: 2 * gm}, cluster_key=ck, n_propagation=3, verbose=False)
            ke = sch.tl.compute_cluster_effects(k, order, cluster_key=ck)
            oe_ = sch.tl.compute_cluster_effects(o, order, cluster_key=ck)
            for c in order:
                rows.append((g, c, float(ke.get(c, np.nan)), float(oe_.get(c, np.nan))))
        d = pd.DataFrame(rows, columns=["gene", "cluster", "KO", "OE"])
        fig, ax = plt.subplots(figsize=(5.8, 5.6))
        for g in d.gene.unique():
            s = d[d.gene == g]
            ax.scatter(s.KO, s.OE, s=30, alpha=0.8, label=g)
        m = np.nanmax([d.KO.max(), d.OE.max()])
        ax.plot([0, m], [0, m], "k--", lw=0.8)
        ax.set(xlabel="mean |delta_X| (KO)", ylabel="mean |delta_X| (OE)",
               title="KO vs OE response per cell type"); ax.legend(fontsize=7, ncol=2)
        d.to_csv(f"figure_packs/reports/{name}/data/ko_oe_symmetry.csv", index=False)
        return fig
    _try(report, name, "E4_ko_oe_symmetry.png", _symm, "KO vs OE per-cluster response symmetry.")

    # ---- 5.4 grid of top-candidate KO flows ----
    report.sub("5.4 KO delta_X flows for top candidates per lineage",
               "The KO-induced flow on the embedding for the top candidates of each lineage.")
    def _grid_flows():
        genes = (A_cand[:2] + B_cand[:2])
        genes = [g for g in genes if g in adata.var_names][:4]
        fig, axes = plt.subplots(1, len(genes), figsize=(5 * len(genes), 4.6))
        if len(genes) == 1:
            axes = [axes]
        for ax, g in zip(axes, genes):
            p = _perturb(adata, {g: 0.0}, ck, basis)
            sch.pl.plot_flow(p, flow_key=f"perturbation_flow_{basis}", basis=basis, ax=ax, on_grid=True, cluster_key=ck)
            ax.set_title(f"{g} KO")
        return fig
    _try(report, name, "E5_grid_ko_flows.png", _grid_flows,
         "KO flows for the top erythroid and myeloid candidates.")

    # ---- 5.5 single-KO screen ----
    report.sub("5.5 Single-gene KO screen (lineage bias)",
               "Post-perturbation lineage bias for each candidate (flow alignment metric).")
    try:
        single_bias, _ = sch.dyn.run_ko_screen(adata, candidates, A, B, basis, wt_key,
                                               cluster_key=ck, verbose=False)
        sb = pd.Series({g: v["lineage_bias"] for g, v in single_bias.items()}).sort_values()
        def _sb():
            fig, ax = plt.subplots(figsize=(6, 4.6))
            ax.barh(range(len(sb)), sb.values, color=["#2a6f97" if v >= 0 else "#bc4749" for v in sb.values])
            ax.set_yticks(range(len(sb))); ax.set_yticklabels(sb.index, fontsize=8)
            ax.axvline(0, color="k", lw=0.8)
            ax.set(xlabel=f"post-KO lineage bias (+{An}/-{Bn})", title="Single-KO lineage bias")
            return fig
        _try(report, name, "E6_single_ko_bias.png", _sb, "Single-KO lineage bias per candidate.")
    except Exception as exc:
        print(f"  [E6] {exc}", flush=True); single_bias = {}

    # ---- 5.6-5.7 pairwise KO + epistasis heatmaps ----
    report.sub("5.6 Pairwise KO screen and epistasis",
               "Cross-lineage KO pairs: double-KO lineage bias and the cancellation error / "
               "synergy relative to the additive expectation.")
    try:
        import itertools
        pairs = [(a, b) for a in A_cand[:3] for b in B_cand[:3]]
        pair_bias, _ = sch.dyn.run_pairwise_ko_screen(adata, pairs, A, B, basis, wt_key,
                                                      cluster_key=ck, verbose=False)
        epi = sch.dyn.compute_epistasis(pair_bias, single_bias, lineage_A_genes=A_cand, lineage_B_genes=B_cand)
        epi.to_csv(f"figure_packs/reports/{name}/data/epistasis.csv")
        AA, BB = A_cand[:3], B_cand[:3]
        def _heat(col, ttl, cmap, fname, cap):
            M = np.full((len(AA), len(BB)), np.nan)
            for _, r in epi.iterrows():
                if r.geneA in AA and r.geneB in BB:
                    M[AA.index(r.geneA), BB.index(r.geneB)] = r[col]
                elif r.geneB in AA and r.geneA in BB:
                    M[AA.index(r.geneB), BB.index(r.geneA)] = r[col]
            lim = np.nanmax(np.abs(M)) or 1.0
            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
            ax.set_xticks(range(len(BB))); ax.set_xticklabels(BB, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(AA))); ax.set_yticklabels(AA, fontsize=8)
            ax.set(xlabel=Bn, ylabel=An, title=ttl); fig.colorbar(im, ax=ax, fraction=0.046)
            return _try(report, name, fname, lambda: fig, cap)
        _heat("lineage_bias", f"{name}: double-KO lineage bias", "RdBu_r", "E7_double_bias_heatmap.png",
              "Double-KO lineage bias across cross-lineage pairs.")
        _heat("cancellation_error", f"{name}: cancellation error", "PRGn", "E8_cancellation_heatmap.png",
              "Cancellation error (double bias minus additive expectation): epistasis.")
    except Exception as exc:
        print(f"  [E7/E8] {exc}", flush=True)

    # ---- 5.9 energy change ----
    report.sub("5.9 Energy change under perturbation",
               "Per-cell-type change in total Hopfield energy when the top candidate is knocked out.")
    def _de():
        order = present_clusters(adata, ck)
        k = sch.dyn.simulate_perturbation(adata, {g0: 0.0}, cluster_key=ck, n_propagation=3, verbose=False)
        # energy of WT vs perturbed state (recompute energies on simulated_count)
        e0 = adata.obs["energy_total"].values
        kk = k.copy(); kk.layers["Ms"] = kk.layers["simulated_count"]
        sch.tl.compute_energies(kk, cluster_key=ck)
        de = kk.obs["energy_total"].values - e0
        lab = adata.obs[ck].astype(str)
        vals = [float(np.nanmean(de[(lab == c).values])) for c in order]
        fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(order), 4))
        ax.bar(range(len(order)), vals, color="#8d6a9f")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(ylabel="mean delta energy", title=f"{g0} KO: energy change per cell type")
        return fig
    _try(report, name, "E9_energy_change.png", _de, f"{g0} KO energy change per cell type.")

    # ---- 5.10 dose-response ----
    report.sub("5.10 Dose-response",
               "Fractional dose-response (0 = KO, 1 = natural, 2 = strong OE) of the lineage bias.")
    dose_genes = (cfg.get("anchors") or [g0])[:2]
    for g in dose_genes:
        if g not in adata.var_names:
            continue
        def _dose(gene=g):
            dr = sch.dyn.run_dose_response(adata, gene, lineage_A_clusters=A, lineage_B_clusters=B,
                                           basis=basis, wt_flow_key=wt_key,
                                           fractions=[0, 0.25, 0.5, 0.75, 1, 1.5, 2],
                                           cluster_key=ck, verbose=False)
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.plot(dr.level_frac, dr.lineage_bias, "-o", color="#0f4c5c")
            ax.axhline(0, color="k", lw=0.6); ax.axvline(1, color="grey", ls="--", lw=0.8)
            ax.set(xlabel=f"{gene} dose (fraction of max)", ylabel=f"lineage bias (+{An}/-{Bn})",
                   title=f"{name}: {gene} dose-response")
            return fig
        _try(report, name, f"E10_dose_{g}.png", _dose, f"{g} fractional dose-response of lineage bias.")

    # ---- 5.11 anchor-partner double-KO recipe (formerly '4+4+4+4') ----
    report.sub("5.11 Anchor-partner double-KO recipe",
               "For an anchor driver, rank candidate partners by their regulatory coupling "
               "(`grn_partner_weights`) and screen anchor+partner double KOs -- a directed way to "
               "find the most effective genetic-interaction pairs (this is the analysis previously "
               "called '4+4+4+4').")
    anchor = (cfg.get("anchors") or A_cand)[0]
    if anchor in adata.var_names:
        try:
            pw = sch.tl.grn_partner_weights(adata, anchor)
            partners = [g for g in pw.abs().mean(axis=1).sort_values(ascending=False).index
                        if g != anchor][:6]
            apairs = [(anchor, p) for p in partners]
            ab, _ = sch.dyn.run_pairwise_ko_screen(adata, apairs, A, B, basis, wt_key,
                                                   cluster_key=ck, verbose=False)
            s = pd.Series({p: ab[(anchor, p)]["lineage_bias"] for p in partners if (anchor, p) in ab})
            def _recipe():
                fig, ax = plt.subplots(figsize=(6, 4.2))
                ax.barh(range(len(s)), s.values, color="#457b9d")
                ax.set_yticks(range(len(s))); ax.set_yticklabels([f"{anchor}+{p}" for p in s.index], fontsize=8)
                ax.axvline(0, color="k", lw=0.8)
                ax.set(xlabel="double-KO lineage bias", title=f"{anchor} anchor: partner double-KOs")
                return fig
            _try(report, name, "E11_anchor_partner.png", _recipe,
                 f"{anchor}-anchored double KOs with its top regulatory partners.")
        except Exception as exc:
            print(f"  [E11] {exc}", flush=True)
