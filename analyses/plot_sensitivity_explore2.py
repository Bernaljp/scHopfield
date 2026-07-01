"""More exploratory plots for network x reg sensitivity (per-setting detail)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "benchmark_results/network_reg_sensitivity"
OUT = f"{D}/plots"
os.makedirs(OUT, exist_ok=True)
res = json.load(open(f"{D}/results.json"))
tags = list(res)
ns = json.load(open(f"{D}/no_scaffold_compare.json")) if os.path.exists(f"{D}/no_scaffold_compare.json") else None
saved = []


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    saved.append(name)


# ---- 10: per-setting top-10 perturbation bar grid ----
allt = tags + (["NO_SCAFFOLD"] if ns else [])
def pert_of(t):
    return ns["no_scaffold"]["top_pert"] if t == "NO_SCAFFOLD" else res[t]["top_pert"]
ncol = 3; nrow = int(np.ceil(len(allt) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
for k, t in enumerate(allt):
    ax = axes[k // ncol][k % ncol]
    gb = pert_of(t)[:10][::-1]
    genes = [g for g, _ in gb]; vals = [b for _, b in gb]
    ax.barh(range(len(genes)), vals, color=["#E74C3C" if v > 0 else "#3498DB" for v in vals], edgecolor="k")
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=7)
    ax.axvline(0, color="k", lw=0.6); ax.set_title(t, fontsize=9)
for k in range(len(allt), nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")
fig.suptitle("Top-10 perturbation drivers per setting (red=erythroid, blue=myeloid bias)", y=1.01)
save(fig, "10_pert_bars_per_setting")

# ---- 11: rank bump chart of top perturbation genes across settings ----
union = sorted({g for t in tags for g in [x for x, _ in res[t]["top_pert"]][:10]})
rank = {t: {g: i for i, (g, _) in enumerate(res[t]["top_pert"])} for t in tags}
fig, ax = plt.subplots(figsize=(11, 7))
cmap = plt.get_cmap("tab20")
for idx, g in enumerate(union):
    y = [rank[t].get(g, np.nan) for t in tags]
    ax.plot(range(len(tags)), y, marker="o", color=cmap(idx % 20), lw=1.5)
    # label at first finite point
    for j, yy in enumerate(y):
        if not np.isnan(yy):
            ax.text(j - 0.05, yy, g, ha="right", va="center", fontsize=6, color=cmap(idx % 20)); break
ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("rank in top perturbation list (0 = strongest)"); ax.invert_yaxis()
ax.set_title("Rank trajectories of perturbation drivers across settings")
save(fig, "11_rank_bump")

# ---- 12: no-scaffold Jaccard vs each setting (3 metrics) ----
if ns:
    nsj = ns["jaccard_vs_scaffold"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(tags)); w = 0.27
    for off, key, c, lab in [(-w, "pert", "#2E8B57", "perturbation"), (0, "score_ery", "#B0B0B0", "score-ery"), (w, "score_mye", "#7F8C8D", "score-mye")]:
        ax.bar(x + off, [nsj[t][key] for t in tags], w, color=c, edgecolor="k", label=lab)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Jaccard (no-scaffold vs setting)"); ax.legend(fontsize=8)
    ax.set_title("No-scaffold overlap with each scaffold setting")
    save(fig, "12_no_scaffold_vs_each")

# ---- 13: driver-score (A vs B) scatter for a representative setting ----
# (uses only ranks we have; show |top-15 ery| vs |top-15 mye| membership overlap as venn-ish counts)
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = tags
share = [len(set(res[t]["top_score_ery"]) & set(res[t]["top_score_mye"])) for t in labels]
ax.bar(range(len(labels)), share, color="#D35400", edgecolor="k")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("# genes in BOTH top-15 score lists")
ax.set_title("Erythroid vs myeloid driver-score overlap per setting")
save(fig, "13_score_ery_mye_overlap")

print(f"generated {len(saved)} more plots:")
for s in saved:
    print("  ", s + ".png")
