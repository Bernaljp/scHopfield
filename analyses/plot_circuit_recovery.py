"""Plot synthetic-circuit recovery vs observation noise (validation figure)."""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="benchmark_results/circuit_recovery/summary.json")
    ap.add_argument("--out", default="benchmark_results/circuit_recovery/recovery.png")
    args = ap.parse_args()

    summary = json.load(open(args.summary))
    # index by circuit -> scaffold -> sorted (noise, fro_mean, fro_sd, corr_mean, sign_mean)
    data = defaultdict(lambda: defaultdict(list))
    for e in summary:
        data[e["circuit"]][e["scaffold"]].append(
            (e["noise"], e["frobenius_distance_mean"], e["frobenius_distance_sd"],
             e["edge_correlation_mean"], e["edge_sign_accuracy_mean"]))
    for c in data:
        for s in data[c]:
            data[c][s].sort()

    circuits = list(data.keys())
    fig, axes = plt.subplots(1, len(circuits), figsize=(5.4 * len(circuits), 4.3), squeeze=False)
    colors = {"full": "#2980B9", "partial": "#E67E22", "none": "#7F8C8D"}
    for ax, c in zip(axes[0], circuits):
        for s in ["full", "partial", "none"]:
            if s not in data[c]:
                continue
            arr = np.array(data[c][s])
            noise, fro, fro_sd = arr[:, 0], arr[:, 1], arr[:, 2]
            ax.errorbar(noise, fro, yerr=fro_sd, marker="o", capsize=3,
                        color=colors[s], label=f"scaffold={s}")
        # annotate that sign-accuracy is perfect throughout
        signmin = min(r[4] for s in data[c] for r in data[c][s])
        corrmin = min(r[3] for s in data[c] for r in data[c][s])
        ax.set_title(f"{c}\nsign-acc = {signmin:.2f}, min corr = {corrmin:.4f} (all noise)")
        ax.set_xlabel("observation noise sigma")
        ax.set_ylabel("relative Frobenius distance  |W_hat - W_true| / |W_true|")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("scHopfield recovers ground-truth GRNs on synthetic circuits", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
