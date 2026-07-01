"""Figure: stability of top score vs perturbation genes across network x reg."""
import itertools
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "benchmark_results/network_reg_sensitivity"
res = json.load(open(f"{D}/results.json"))
tags = list(res)


def mean_jac(fname):
    m = json.load(open(f"{D}/{fname}.json"))
    return np.mean([m[a][b] for a, b in itertools.combinations(tags, 2)])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                               gridspec_kw={"width_ratios": [1, 1.6]})

# panel 1: mean pairwise Jaccard stability
labels = ["Driver score\n(erythroid)", "Driver score\n(myeloid)", "Perturbation\n(KO |bias|)"]
vals = [mean_jac("jaccard_top_score_ery"), mean_jac("jaccard_top_score_mye"), mean_jac("jaccard_top_pert")]
cols = ["#B0B0B0", "#B0B0B0", "#2E8B57"]
b = ax1.bar(labels, vals, color=cols, edgecolor="k")
for bar, v in zip(b, vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
ax1.set_ylim(0, 1)
ax1.set_ylabel("Mean pairwise Jaccard of top-15 lists")
ax1.set_title("Top-gene stability across\n6 settings (2 networks x 3 reg)")
ax1.axhline(1.0, ls="--", lw=0.6, color="grey")

# panel 2: recurrence of top perturbation genes across settings (presence matrix)
pert_lists = {t: [g for g, _ in res[t]["top_pert"]] for t in tags}
genes = sorted({g for lst in pert_lists.values() for g in lst[:10]},
               key=lambda g: -sum(g in pert_lists[t][:10] for t in tags))
mat = np.array([[1 if g in pert_lists[t][:10] else 0 for t in tags] for g in genes])
ax2.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=1)
ax2.set_xticks(range(len(tags)))
ax2.set_xticklabels(tags, rotation=40, ha="right", fontsize=8)
ax2.set_yticks(range(len(genes)))
ax2.set_yticklabels(genes, fontsize=8)
ax2.set_title("Top-10 perturbation drivers: presence per setting\n(canonical drivers recur in all settings)")
for i in range(len(genes)):
    for j in range(len(tags)):
        if mat[i, j]:
            ax2.text(j, i, "x", ha="center", va="center", fontsize=7, color="white")

fig.suptitle("scHopfield: perturbation drivers are robust to network/regularization; static scores are not",
             y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(f"{D}/sensitivity.png", dpi=150, bbox_inches="tight")
print(f"wrote {D}/sensitivity.png")
