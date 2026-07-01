"""Exploratory plot pack for the network x reg sensitivity results.

Generates many views of the same data (for exploration, not all paper-bound):
Jaccard heatmaps, gene-recurrence maps, lineage-bias heatmaps and trajectories,
setting dendrograms, score-vs-perturbation overlap, consensus frequency, and the
no-scaffold contrast. Everything is derived from results.json + jaccard_*.json +
no_scaffold_compare.json.
"""
import itertools
import json
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

D = "benchmark_results/network_reg_sensitivity"
OUT = f"{D}/plots"
os.makedirs(OUT, exist_ok=True)
res = json.load(open(f"{D}/results.json"))
tags = list(res)
ns = json.load(open(f"{D}/no_scaffold_compare.json")) if os.path.exists(f"{D}/no_scaffold_compare.json") else None
saved = []


def save(fig, name):
    p = f"{OUT}/{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    saved.append(name)


def heat(matrix, labels, title, name, cmap="viridis", vmin=0, vmax=1, annot=True):
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    if annot:
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v < (vmin + vmax) / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=10)
    save(fig, name)


# ---- 1-3: Jaccard heatmaps (with no-scaffold appended) ----
for key, lab in [("top_pert", "perturbation"), ("top_score_ery", "score (erythroid)"),
                 ("top_score_mye", "score (myeloid)")]:
    jf = f"{D}/jaccard_{key}.json"
    m = json.load(open(jf))
    labels = list(m)
    mat = np.array([[m[a][b] for b in labels] for a in labels])
    if ns:  # append no-scaffold row/col
        nsj = ns["jaccard_vs_scaffold"]
        keymap = {"top_pert": "pert", "top_score_ery": "score_ery", "top_score_mye": "score_mye"}[key]
        col = np.array([nsj[t][keymap] for t in labels])
        mat = np.vstack([np.column_stack([mat, col]), np.append(col, 1.0)])
        labels = labels + ["NO_SCAFFOLD"]
    heat(mat, labels, f"Jaccard of top-15 {lab} genes", f"1_jaccard_{key}", cmap="magma")

# ---- 4-6: gene recurrence maps (presence in top-10) ----
for key, lab in [("top_pert", "perturbation"), ("top_score_ery", "score-ery"), ("top_score_mye", "score-mye")]:
    lists = {t: (res[t][key] if key != "top_pert" else [g for g, _ in res[t]["top_pert"]]) for t in tags}
    cols = list(tags)
    if ns:
        nsl = ns["no_scaffold"][key] if key != "top_pert" else [g for g, _ in ns["no_scaffold"]["top_pert"]]
        lists["NO_SCAFFOLD"] = nsl; cols = tags + ["NO_SCAFFOLD"]
    genes = sorted({g for t in cols for g in lists[t][:10]},
                   key=lambda g: -sum(g in lists[t][:10] for t in cols))
    mat = np.array([[1 if g in lists[t][:10] else 0 for t in cols] for g in genes])
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2, 0.32 * len(genes) + 1.5))
    ax.imshow(mat, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=7)
    if ns:
        ax.axvline(len(tags) - 0.5, color="crimson", lw=2)
    ax.set_title(f"Top-10 {lab} genes: presence per setting")
    save(fig, f"2_recurrence_{key}")

# ---- 7: lineage-bias value heatmap for perturbation genes ----
bias = {t: {g: b for g, b in res[t]["top_pert"]} for t in tags}
if ns:
    bias["NO_SCAFFOLD"] = {g: b for g, b in ns["no_scaffold"]["top_pert"]}
cols = list(bias)
genes = sorted({g for t in cols for g in bias[t]},
               key=lambda g: -np.nanmean([abs(bias[t].get(g, np.nan)) for t in cols if g in bias[t]]))
mat = np.array([[bias[t].get(g, np.nan) for t in cols] for g in genes])
fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2, 0.34 * len(genes) + 1.5))
vlim = np.nanmax(np.abs(mat))
im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vlim, vmax=vlim)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=7)
for i in range(len(genes)):
    for j in range(len(cols)):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.046, label="lineage bias (+ery / -mye)")
ax.set_title("KO lineage-bias value per setting (blank = not in top-15)")
save(fig, "3_bias_value_heatmap")

# ---- 8: bias trajectories for canonical genes across settings ----
canon = ["Gata1", "Spi1", "Klf1", "E2f4", "Stat3", "Irf8", "Myc", "Gata2"]
fig, ax = plt.subplots(figsize=(10, 5))
xt = list(tags)
for g in canon:
    y = [bias[t].get(g, np.nan) for t in xt]
    ax.plot(range(len(xt)), y, marker="o", label=g)
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("KO lineage bias"); ax.legend(fontsize=8, ncol=2)
ax.set_title("Canonical-gene KO bias across settings")
save(fig, "4_bias_trajectories")

# ---- 9: setting dendrogram (cluster settings by perturbation-list similarity) ----
mj = json.load(open(f"{D}/jaccard_top_pert.json")); L = list(mj)
dist = 1 - np.array([[mj[a][b] for b in L] for a in L])
np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), method="average")
fig, ax = plt.subplots(figsize=(8, 4))
dendrogram(Z, labels=L, ax=ax, leaf_rotation=40)
ax.set_ylabel("1 - Jaccard (top-pert)"); ax.set_title("Settings clustered by perturbation-driver similarity")
save(fig, "5_settings_dendrogram")

# ---- 10: score vs perturbation overlap within each setting ----
fig, ax = plt.subplots(figsize=(9, 4.5))
overlap_ery = [len(set(res[t]["top_score_ery"]) & {g for g, _ in res[t]["top_pert"]}) for t in tags]
overlap_mye = [len(set(res[t]["top_score_mye"]) & {g for g, _ in res[t]["top_pert"]}) for t in tags]
x = np.arange(len(tags)); w = 0.38
ax.bar(x - w / 2, overlap_ery, w, label="score-ery ∩ pert", color="#8E44AD")
ax.bar(x + w / 2, overlap_mye, w, label="score-mye ∩ pert", color="#16A085")
ax.set_xticks(x); ax.set_xticklabels(tags, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("# shared genes (of top-15)"); ax.legend(fontsize=8)
ax.set_title("Overlap between static-score and perturbation top genes (per setting)")
save(fig, "6_score_vs_pert_overlap")

# ---- 11: consensus frequency of perturbation genes ----
cnt = Counter(g for t in tags for g in [x for x, _ in res[t]["top_pert"]][:10])
genes, freqs = zip(*sorted(cnt.items(), key=lambda kv: -kv[1]))
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(range(len(genes)), freqs, color=["#2E8B57" if f == len(tags) else "#95A5A6" for f in freqs], edgecolor="k")
ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=7)
ax.set_ylabel(f"# of {len(tags)} settings in top-10"); ax.axhline(len(tags), ls="--", lw=0.6, color="grey")
ax.set_title("Perturbation-driver recurrence across the 6 scaffold settings")
save(fig, "7_consensus_frequency")

# ---- 12: within-network vs cross-network vs vs-no-scaffold Jaccard ----
fig, ax = plt.subplots(figsize=(7, 4.5))
mj = json.load(open(f"{D}/jaccard_top_pert.json"))
within = [mj[a][b] for a, b in itertools.combinations(L, 2) if a.split(":")[0] == b.split(":")[0]]
cross = [mj[a][b] for a, b in itertools.combinations(L, 2) if a.split(":")[0] != b.split(":")[0]]
groups = ["within\nnetwork", "cross\nnetwork"]
vals = [np.mean(within), np.mean(cross)]
if ns:
    groups.append("vs\nno-scaffold"); vals.append(ns["mean_jaccard_vs_scaffold"]["pert"])
b = ax.bar(groups, vals, color=["#27AE60", "#2980B9", "#C0392B"][:len(groups)], edgecolor="k")
for bar, v in zip(b, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_ylim(0, 1); ax.set_ylabel("mean Jaccard (top-pert)")
ax.set_title("Perturbation-driver stability by comparison type")
save(fig, "8_stability_by_type")

# ---- 13: regularization effect (within each network: free->low->high pairwise) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for net in ["scATAC_atlas", "promoter"]:
    pairs = [("free", "low"), ("low", "high"), ("free", "high")]
    y = [mj[f"{net}:{a}"][f"{net}:{b}"] for a, b in pairs]
    ax.plot(range(len(pairs)), y, marker="o", label=net)
ax.set_xticks(range(3)); ax.set_xticklabels(["free vs low", "low vs high", "free vs high"])
ax.set_ylim(0, 1); ax.set_ylabel("Jaccard (top-pert)"); ax.legend(fontsize=8)
ax.set_title("Effect of scaffold regularization within each network")
save(fig, "9_reg_effect")

print(f"generated {len(saved)} plots in {OUT}:")
for s in saved:
    print("  ", s + ".png")
