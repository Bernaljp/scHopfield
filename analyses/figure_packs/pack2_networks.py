"""Figure pack 2: regulatory-network analysis (centrality, similarity, structure).

For every re-fitted dataset: per-cell-type centrality (who the hubs are), how the
cell-type-specific networks relate to each other (edge Jaccard / W correlation /
spectral distance), and global GRN structure (spectral radius, symmetry, degree). CPU.

    figure_packs/pack2_networks/<dataset>/{plots,data}/ + FIGURE_GUIDE.md

Run:  .venv/bin/python analyses/figure_packs/pack2_networks.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

import _common as C

OUT = "figure_packs/pack2_networks"
CENT_TYPES = ["degree_centrality_out", "degree_centrality_in",
              "eigenvector_centrality", "betweenness_centrality"]


def _cent(adata, cluster, kind):
    col = f"{kind}_{cluster}"
    if col in adata.var.columns:
        return pd.Series(np.asarray(adata.var[col]), index=adata.var_names)
    # fall back to out-strength of W
    W = C.W_of(adata, cluster)
    return pd.Series(np.abs(W).sum(0), index=adata.var_names)


def fig_hub_heatmap(adata, clusters, out, top=15):
    """1: union of top hub genes x cell types (out-degree centrality)."""
    hubs = []
    for c in clusters:
        hubs += list(_cent(adata, c, "degree_centrality_out").nlargest(top).index)
    genes = list(dict.fromkeys(hubs))
    M = np.array([[_cent(adata, c, "degree_centrality_out")[g] for c in clusters] for g in genes])
    fig, ax = plt.subplots(figsize=(1.1 + 0.5 * len(clusters), 0.28 * len(genes) + 1))
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=6)
    ax.set_title("Hub genes (out-degree) per cell type")
    fig.colorbar(im, ax=ax, fraction=0.046, label="out-degree centrality")
    fig.tight_layout(); fig.savefig(f"{out}/plots/01_hub_heatmap.png", dpi=140); plt.close(fig)
    return genes


def fig_percelltype_hubs(adata, clusters, out, top=10):
    """2: one panel per cell type -- its strongest regulators."""
    ncl = len(clusters)
    ncol = min(4, ncl); nrow = int(np.ceil(ncl / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow), squeeze=False)
    for k, c in enumerate(clusters):
        ax = axes[k // ncol][k % ncol]
        s = _cent(adata, c, "degree_centrality_out").nlargest(top).iloc[::-1]
        ax.barh(range(len(s)), s.values, color="#3b6ea5")
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=6)
        ax.set_title(c, fontsize=8); ax.tick_params(labelsize=6)
    for k in range(ncl, nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    fig.suptitle("Top regulators per cell type (out-degree)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{out}/plots/02_percelltype_hubs.png", dpi=140); plt.close(fig)


def fig_centrality_agreement(adata, cluster, out, top=12):
    """3: do centrality measures agree on the hubs (largest cell type)?"""
    fig, axes = plt.subplots(1, len(CENT_TYPES), figsize=(3.2 * len(CENT_TYPES), 3.6))
    for ax, kind in zip(axes, CENT_TYPES):
        s = _cent(adata, cluster, kind).nlargest(top).iloc[::-1]
        ax.barh(range(len(s)), s.values, color="#6a4c93")
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=6)
        ax.set_title(kind.replace("_centrality", "").replace("_", " "), fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Centrality measures on '{cluster}'", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{out}/plots/03_centrality_agreement.png", dpi=140); plt.close(fig)


def fig_network_similarity(adata, clusters, out, thr=1e-3):
    """4: pairwise similarity of the cell-type networks (corr / Jaccard / spectral)."""
    Ws = {c: C.W_of(adata, c) for c in clusters}
    n = len(clusters)
    corr = np.zeros((n, n)); jacc = np.zeros((n, n)); spec = np.zeros((n, n))
    binE = {c: (np.abs(Ws[c]) > thr) for c in clusters}
    eig = {c: np.sort(np.abs(np.linalg.eigvals(Ws[c])))[::-1] for c in clusters}
    for i, a in enumerate(clusters):
        for j, b in enumerate(clusters):
            corr[i, j] = np.corrcoef(Ws[a].ravel(), Ws[b].ravel())[0, 1]
            inter = (binE[a] & binE[b]).sum(); union = (binE[a] | binE[b]).sum()
            jacc[i, j] = inter / union if union else 0.0
            m = min(len(eig[a]), len(eig[b]))
            spec[i, j] = np.linalg.norm(eig[a][:m] - eig[b][:m])
    # order cell types by hierarchical clustering on correlation distance
    order = list(range(n))
    if n > 2:
        from scipy.spatial.distance import squareform
        d = 1 - corr; np.fill_diagonal(d, 0.0); d = (d + d.T) / 2
        order = list(leaves_list(linkage(squareform(d, checks=False), method="average")))
    cl = [clusters[i] for i in order]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, M, ttl, cmap in [(axes[0], corr, "W correlation", "RdBu_r"),
                             (axes[1], jacc, "edge Jaccard", "viridis"),
                             (axes[2], spec, "spectral distance", "magma_r")]:
        Mo = M[np.ix_(order, order)]
        im = ax.imshow(Mo, cmap=cmap, vmin=(-1 if ttl == "W correlation" else None),
                       vmax=(1 if ttl == "W correlation" else None))
        ax.set_xticks(range(n)); ax.set_xticklabels(cl, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(cl, fontsize=6)
        ax.set_title(ttl); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Cell-type network similarity", fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{out}/plots/04_network_similarity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return dict(clusters=clusters, corr=corr.tolist(), jaccard=jacc.tolist())


def fig_structure(adata, clusters, out):
    """5: spectral radius and asymmetry of each cell-type network."""
    rad, sym = [], []
    for c in clusters:
        W = C.W_of(adata, c)
        rad.append(float(np.max(np.real(np.linalg.eigvals(W)))))
        sym.append(float(np.linalg.norm(W - W.T) / (np.linalg.norm(W + W.T) + 1e-12)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.bar(range(len(clusters)), rad, color="#2a9d8f")
    a1.set_xticks(range(len(clusters))); a1.set_xticklabels(clusters, rotation=45, ha="right", fontsize=7)
    a1.set(ylabel="max Re(eig W)", title="GRN spectral radius (amplification)")
    a1.axhline(0, color="k", lw=0.6)
    a2.bar(range(len(clusters)), sym, color="#e76f51")
    a2.set_xticks(range(len(clusters))); a2.set_xticklabels(clusters, rotation=45, ha="right", fontsize=7)
    a2.set(ylabel="||W-W^T|| / ||W+W^T||", title="GRN asymmetry (directed-ness)")
    fig.tight_layout(); fig.savefig(f"{out}/plots/05_grn_structure.png", dpi=140); plt.close(fig)
    return dict(clusters=clusters, spectral_radius=rad, asymmetry=sym)


def fig_degree_dist(adata, clusters, out):
    """6: in/out degree distributions pooled across cell types."""
    outd, ind = [], []
    for c in clusters:
        W = C.W_of(adata, c)
        outd.append(np.abs(W).sum(0)); ind.append(np.abs(W).sum(1))
    outd = np.concatenate(outd); ind = np.concatenate(ind)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.hist(outd, bins=40, alpha=0.6, label="out-strength", color="#3b6ea5")
    ax.hist(ind, bins=40, alpha=0.6, label="in-strength", color="#e09f3e")
    ax.set(xlabel="summed |W|", ylabel="# gene-cluster", title="Degree (interaction-strength) distribution")
    ax.legend(); ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(f"{out}/plots/06_degree_distribution.png", dpi=140); plt.close(fig)


GUIDE_HEAD = """# Figure pack 2: regulatory-network analysis

Cell-type-specific GRNs (`W_{cluster}`) and their centrality/structure, per re-fitted
dataset. Regenerated by `analyses/figure_packs/pack2_networks.py`. Targets paper section
R4 (robustness of driver identification) and the network-analysis methods.

Datasets: {datasets}

Per dataset (`<dataset>/plots/`):
1. `01_hub_heatmap` -- union of top hub genes x cell types (out-degree centrality).
2. `02_percelltype_hubs` -- one panel per cell type, its strongest regulators.
3. `03_centrality_agreement` -- do out-degree / eigenvector / betweenness agree on hubs?
4. `04_network_similarity` -- pairwise W correlation, edge Jaccard, spectral distance
   between cell-type networks (hierarchically ordered).
5. `05_grn_structure` -- per-cell-type spectral radius (amplification) and asymmetry.
6. `06_degree_distribution` -- pooled in/out interaction-strength distribution.
"""


def run_dataset(name, cluster_key):
    adata = C.load(name)
    clusters = C.present_clusters(adata, cluster_key)
    out = f"{OUT}/{name}"
    os.makedirs(f"{out}/plots", exist_ok=True); os.makedirs(f"{out}/data", exist_ok=True)
    print(f"[pack2] {name}: {len(clusters)} clusters", flush=True)
    data = {}
    fig_hub_heatmap(adata, clusters, out)
    fig_percelltype_hubs(adata, clusters, out)
    fig_centrality_agreement(adata, clusters[0], out)
    data["similarity"] = fig_network_similarity(adata, clusters, out)
    data["structure"] = fig_structure(adata, clusters, out)
    fig_degree_dist(adata, clusters, out)
    json.dump(data, open(f"{out}/data/pack2_{name}.json", "w"), indent=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    avail = C.available()
    for name, ck in avail.items():
        try:
            run_dataset(name, ck)
        except Exception as exc:
            import traceback; print(f"FAILED {name}: {exc}"); traceback.print_exc()
    open(f"{OUT}/FIGURE_GUIDE.md", "w").write(GUIDE_HEAD.replace("{datasets}", ", ".join(avail)))
    print(f"wrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
