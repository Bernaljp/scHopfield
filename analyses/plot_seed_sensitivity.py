"""Plot the seed-sensitivity / reproducibility result (supplementary figure)."""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return json.load(open(os.path.join(path, "results.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="benchmark_results/seed_sensitivity_real")
    ap.add_argument("--out", default="benchmark_results/seed_sensitivity_real/reproducibility.png")
    args = ap.parse_args()

    r = load(args.real)
    seeded = r["pairs"]["seed0_a_vs_b"]
    unseeded = r["pairs"]["unseeded_a_vs_b"]
    cs = r["pairs"]["cross_seed"]
    cs_w = [v["W_pearson"] for v in cs.values()]
    cs_c = [v["centrality_spearman"] for v in cs.values()]

    conds = ["Seeded\n(same seed)", "Unseeded\n(2 runs)", "Cross-seed\n(5 seeds)"]
    w_vals = [seeded["W_pearson"], unseeded["W_pearson"], np.mean(cs_w)]
    c_vals = [seeded["centrality_spearman"], unseeded["centrality_spearman"], np.mean(cs_c)]
    w_err = [0, 0, np.std(cs_w)]
    c_err = [0, 0, np.std(cs_c)]

    x = np.arange(len(conds))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    b1 = ax.bar(x - width / 2, w_vals, width, yerr=w_err, capsize=4,
                label="W matrix (Pearson)", color="#2980B9")
    b2 = ax.bar(x + width / 2, c_vals, width, yerr=c_err, capsize=4,
                label="Centrality ranking (Spearman)", color="#E67E22")
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("Agreement between runs")
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_title(f"scHopfield reproducibility (real pancreas, {r['shape'][0]}x{r['shape'][1]}, GPU)")
    ax.legend(loc="lower left", fontsize=9)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
