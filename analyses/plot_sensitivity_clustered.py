"""Clustered (reordered) versions of the sensitivity heatmaps."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

D = "benchmark_results/network_reg_sensitivity"
OUT = f"{D}/plots"
os.makedirs(OUT, exist_ok=True)
res = json.load(open(f"{D}/results.json"))
tags = list(res)
ns = json.load(open(f"{D}/no_scaffold_compare.json")) if os.path.exists(f"{D}/no_scaffold_compare.json") else None
saved = []


def jaccard_matrix(key):
    m = json.load(open(f"{D}/jaccard_{key}.json"))
    labels = list(m)
    mat = np.array([[m[a][b] for b in labels] for a in labels])
    if ns:
        km = {"top_pert": "pert", "top_score_ery": "score_ery", "top_score_mye": "score_mye"}[key]
        col = np.array([ns["jaccard_vs_scaffold"][t][km] for t in labels])
        mat = np.vstack([np.column_stack([mat, col]), np.append(col, 1.0)])
        labels = labels + ["NO_SCAFFOLD"]
    return mat, labels


# ---- clustered symmetric Jaccard heatmaps ----
for key, lab in [("top_pert", "perturbation"), ("top_score_ery", "score-ery"), ("top_score_mye", "score-mye")]:
    mat, labels = jaccard_matrix(key)
    dist = 1 - mat
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    g = sns.clustermap(mat, row_linkage=Z, col_linkage=Z, cmap="magma", vmin=0, vmax=1,
                       annot=True, fmt=".2f", annot_kws={"size": 7},
                       xticklabels=labels, yticklabels=labels, figsize=(7, 6.5),
                       cbar_kws={"label": "Jaccard"})
    g.fig.suptitle(f"Clustered Jaccard: top-15 {lab} genes", y=1.02)
    g.savefig(f"{OUT}/C_jaccard_{key}.png", dpi=140, bbox_inches="tight")
    plt.close(g.fig)
    saved.append(f"C_jaccard_{key}")

# ---- biclustered KO bias-value heatmap (genes x settings) ----
bias = {t: {gg: b for gg, b in res[t]["top_pert"]} for t in tags}
if ns:
    bias["NO_SCAFFOLD"] = {gg: b for gg, b in ns["no_scaffold"]["top_pert"]}
cols = list(bias)
genes = sorted({gg for t in cols for gg in bias[t]})
mat = np.array([[bias[t].get(gg, np.nan) for t in cols] for gg in genes])
mat_filled = np.nan_to_num(mat, nan=0.0)  # 0 = not a top driver, for clustering
vlim = np.nanmax(np.abs(mat))
g = sns.clustermap(mat_filled, cmap="RdBu_r", vmin=-vlim, vmax=vlim, center=0,
                   xticklabels=cols, yticklabels=genes, figsize=(8, 0.32 * len(genes) + 2),
                   annot=True, fmt=".2f", annot_kws={"size": 6},
                   cbar_kws={"label": "KO lineage bias (+ery / -mye)"})
g.fig.suptitle("Biclustered KO lineage-bias (0 = not in top-15)", y=1.01)
g.savefig(f"{OUT}/C_bias_biclustered.png", dpi=140, bbox_inches="tight")
plt.close(g.fig)
saved.append("C_bias_biclustered")

# ---- biclustered recurrence (binary presence in top-10 perturbation) ----
lists = {t: [gg for gg, _ in res[t]["top_pert"]][:10] for t in tags}
if ns:
    lists["NO_SCAFFOLD"] = [gg for gg, _ in ns["no_scaffold"]["top_pert"]][:10]
cols = list(lists)
genes = sorted({gg for t in cols for gg in lists[t]})
mat = np.array([[1 if gg in lists[t] else 0 for t in cols] for gg in genes], dtype=float)
g = sns.clustermap(mat, cmap="Greens", vmin=0, vmax=1, xticklabels=cols, yticklabels=genes,
                   figsize=(7.5, 0.32 * len(genes) + 2), linewidths=0.3, linecolor="lightgrey",
                   cbar_kws={"label": "in top-10"})
g.fig.suptitle("Biclustered perturbation-driver presence (top-10)", y=1.01)
g.savefig(f"{OUT}/C_recurrence_biclustered.png", dpi=140, bbox_inches="tight")
plt.close(g.fig)
saved.append("C_recurrence_biclustered")

print(f"generated {len(saved)} clustered plots:")
for s in saved:
    print("  ", s + ".png")
