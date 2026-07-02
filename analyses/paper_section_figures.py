"""Generate exploratory plots for the paper sections that have result JSON but few
figures, mirroring the network_reg_sensitivity treatment. Reads the saved JSONs in
benchmark_results/<section>/ and writes plots to benchmark_results/<section>/plots/.
No fitting; CPU-only.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BR = "benchmark_results"


def outdir(section):
    d = f"{BR}/{section}/plots"
    os.makedirs(d, exist_ok=True)
    return d


def load(p):
    return json.load(open(p))


# ---------------- circuit recovery (Fig 2) ----------------
def circuit_recovery():
    s = load(f"{BR}/circuit_recovery/summary.json")
    d = outdir("circuit_recovery")
    circuits = sorted({r["circuit"] for r in s})
    scaffolds = sorted({r["scaffold"] for r in s})
    # edge sign accuracy at noise=0, circuit x scaffold
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(circuits)); w = 0.8 / max(len(scaffolds), 1)
    for i, sc in enumerate(scaffolds):
        vals = [next((r["edge_sign_accuracy_mean"] for r in s
                      if r["circuit"] == c and r["scaffold"] == sc and r["noise"] == 0.0), np.nan)
                for c in circuits]
        ax.bar(x + i * w, vals, w, label=f"scaffold={sc}")
    ax.set_xticks(x + w * (len(scaffolds) - 1) / 2); ax.set_xticklabels(circuits, rotation=20, ha="right")
    ax.set_ylabel("edge sign accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Synthetic circuit recovery: signed-edge accuracy (noise=0)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{d}/1_edge_sign_accuracy.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # edge correlation vs noise (full scaffold)
    noises = sorted({r["noise"] for r in s})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in circuits:
        ys = [next((r["edge_correlation_mean"] for r in s
                    if r["circuit"] == c and r["scaffold"] == "full" and r["noise"] == n), np.nan) for n in noises]
        ax.plot(noises, ys, "o-", label=c)
    ax.set_xlabel("noise"); ax.set_ylabel("edge correlation (full scaffold)")
    ax.set_title("Recovery robustness to noise"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{d}/2_edge_corr_vs_noise.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("circuit_recovery: 2 plots", flush=True)


# ---------------- GENIE3 baseline (Fig 2 / supp) ----------------
def grn_baseline():
    g = load(f"{BR}/grn_baseline/genie3_vs_schopfield.json")
    d = outdir("grn_baseline")
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ["auroc", "auprc"]; x = np.arange(len(metrics)); w = 0.35
    sch = [g["schopfield_auroc_mean"], g["schopfield_auprc_mean"]]
    gen = [g["genie3_auroc_mean"], g["genie3_auprc_mean"]]
    schsd = [g.get("schopfield_auroc_sd", 0), g.get("schopfield_auprc_sd", 0)]
    gensd = [g.get("genie3_auroc_sd", 0), g.get("genie3_auprc_sd", 0)]
    ax.bar(x - w / 2, sch, w, yerr=schsd, capsize=4, label="scHopfield", color="#2a6f97")
    ax.bar(x + w / 2, gen, w, yerr=gensd, capsize=4, label="GENIE3", color="#adb5bd")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["AUROC", "AUPRC"]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("score"); ax.legend()
    ax.set_title(f"GRN recovery vs GENIE3 baseline ({g['n_networks']} nets, {g['n_genes']} genes)")
    fig.tight_layout(); fig.savefig(f"{d}/1_genie3_bars.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("grn_baseline: 1 plot", flush=True)


# ---------------- Hill vs linear (Fig 3) ----------------
def ablations():
    h = load(f"{BR}/ablations/hill_vs_linear.json")
    d = outdir("ablations")
    circuits = list(h.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(circuits)); w = 0.35
    axes[0].bar(x - w / 2, [h[c]["hill_recon_r2"] for c in circuits], w, label="Hill", color="#2a9d8f")
    axes[0].bar(x + w / 2, [h[c]["linear_recon_r2"] for c in circuits], w, label="linear", color="#e76f51")
    axes[0].set_xticks(x); axes[0].set_xticklabels(circuits, rotation=15, ha="right")
    axes[0].set_ylabel("velocity reconstruction R2"); axes[0].set_title("Hill vs linear activation"); axes[0].legend()
    axes[1].bar(x - w / 2, [h[c]["hill_stable_fixedpoints"] for c in circuits], w, label="Hill", color="#2a9d8f")
    axes[1].bar(x + w / 2, [h[c]["linear_stable_fixedpoints"] for c in circuits], w, label="linear", color="#e76f51")
    axes[1].plot(x, [h[c]["true_stable_fixedpoints"] for c in circuits], "k*", ms=14, label="truth")
    axes[1].set_xticks(x); axes[1].set_xticklabels(circuits, rotation=15, ha="right")
    axes[1].set_ylabel("# stable fixed points"); axes[1].set_title("Recovered attractor structure"); axes[1].legend()
    fig.suptitle("The Hill nonlinearity is necessary for correct dynamics", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{d}/1_hill_vs_linear.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("ablations: 1 plot", flush=True)


# ---------------- identifiability across datasets (supp) ----------------
def identifiability():
    m = load(f"{BR}/real_identifiability/multi.json")
    d = outdir("real_identifiability")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for name, rec in m.items():
        fr = sorted(rec["by_frac"].keys(), key=float)
        axes[0].plot([float(f) for f in fr], [rec["by_frac"][f]["eff_rank"] for f in fr], "o-", label=name)
        axes[1].plot([float(f) for f in fr], [rec["by_frac"][f]["splithalf_W"] for f in fr], "o-", label=name)
    axes[0].set_xlabel("neighbour fraction"); axes[0].set_ylabel("effective rank of sigma(X)")
    axes[0].set_title("Broadening raises effective rank"); axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("neighbour fraction"); axes[1].set_ylabel("split-half W correlation")
    axes[1].set_title("...but split-half W stays ~0 (scaffold essential)"); axes[1].legend(fontsize=8)
    fig.suptitle("Identifiability across 4 developmental systems (M13)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{d}/3_identifiability_multi.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("real_identifiability: 1 plot", flush=True)


# ---------------- KO panels (Fig 4) ----------------
def ko_panel(section, jname, title):
    p = f"{BR}/{section}/{jname}"
    if not os.path.exists(p):
        return
    j = load(p); d = outdir(section)
    rows = j["rows"]
    genes = [r["gene"] for r in rows]
    bias = [r["lineage_bias"] for r in rows]
    correct = [r.get("correct", None) for r in rows]
    col = ["#2a9d8f" if c else "#d1495b" for c in correct]
    fig, ax = plt.subplots(figsize=(max(7, len(genes) * 0.6), 4.5))
    ax.bar(range(len(genes)), bias, color=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=40, ha="right")
    ax.set_ylabel("predicted lineage bias")
    acc = j.get("directional_accuracy", np.nan)
    ax.set_title(f"{title} -- directional accuracy {acc:.0%} (green=correct, red=miss)")
    fig.tight_layout(); fig.savefig(f"{d}/1_ko_panel.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"{section}: 1 plot", flush=True)


# ---------------- Jacobian regularizer (M11) ----------------
def jacobian_reg():
    j = load(f"{BR}/jacobian_reg/validation.json")
    d = outdir("jacobian_reg")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (circ, rows) in zip(axes, j.items()):
        lam = [r["lambda"] for r in rows]
        ax.plot(lam, [r["sign_acc"] for r in rows], "o-", label="edge sign acc")
        ax.plot(lam, [r["auroc"] for r in rows], "s-", label="AUROC")
        ax.set_xscale("symlog"); ax.set_xlabel("jacobian_lambda"); ax.set_title(circ); ax.legend(fontsize=8)
    fig.suptitle("Jacobian-consistency regularizer: no improvement (M11, honest negative)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{d}/1_jacobian_reg.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("jacobian_reg: 1 plot", flush=True)


if __name__ == "__main__":
    circuit_recovery()
    grn_baseline()
    ablations()
    identifiability()
    ko_panel("hemato_ko", "schopfield_ko_panel.json", "Hematopoiesis known-driver KO")
    ko_panel("nc_ko", "panel.json", "Neural-crest known-driver KO")
    jacobian_reg()
    print("done", flush=True)
