"""Plots for the reprogramming bias validation (M18), from the saved arrays.
No fitting -- reads reprogramming_staged_arrays.npz (per-stage bias vectors) and
writes figures to benchmark_results/bias_penalty/figures/.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OSKM = ["Pou5f1", "Sox2", "Klf4", "Myc"]
STAGES = [("I_MEF", "MEF"), ("I_transitional", "transitional"), ("I_iPSC", "iPSC")]
FIG = "benchmark_results/bias_penalty/figures"
SCOL = {"MEF": "#8ecae6", "transitional": "#ffb703", "iPSC": "#fb8500"}


def null_pct(absI, genes, markers, n=5000):
    present = [g for g in markers if g in set(genes)]
    obs = absI[np.isin(genes, present)].mean()
    rng = np.random.default_rng(0)
    null = np.array([absI[rng.choice(len(genes), len(present), replace=False)].mean() for _ in range(n)])
    return obs, null, float((null < obs).mean() * 100)


def main():
    os.makedirs(FIG, exist_ok=True)
    d = np.load(f"{FIG}/../reprogramming_staged_arrays.npz", allow_pickle=True)
    genes = d["genes"].astype(str)
    stageI = {lab: np.abs(d[key]) for key, lab in STAGES}
    natI = np.abs(d["nat_I"]); natgenes = d["nat_genes"].astype(str)
    is_oskm = np.isin(genes, OSKM)

    # ---- G: OSKM percentile vs random-4 null, per stage + natural control ----
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts, labs, cols = [], [], []
    for _, lab in STAGES:
        _, _, pct = null_pct(stageI[lab], genes, OSKM)
        pcts.append(pct); labs.append(lab); cols.append(SCOL[lab])
    rng = np.random.default_rng(1)
    _, _, npct = null_pct(natI, natgenes, list(rng.choice(natgenes, 4, replace=False)))
    pcts.append(npct); labs.append("pancreas\n(control)"); cols.append("#adb5bd")
    ax.bar(labs, pcts, color=cols)
    ax.axhline(50, color="k", ls="--", lw=0.8, label="random expectation")
    ax.axhline(95, color="crimson", ls=":", lw=0.8, label="95th pct")
    ax.set_ylabel("OSKM |I| percentile vs random 4-gene sets")
    ax.set_title("OSKM bias sits at the ~90th percentile in every reprogramming stage;\nno localization in the natural control")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/G_oskm_percentile.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- H: OSKM mean|I| vs rest, per stage ----
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(STAGES)); w = 0.38
    om = [stageI[l][is_oskm].mean() for _, l in STAGES]
    rm = [stageI[l][~is_oskm].mean() for _, l in STAGES]
    ax.bar(xs - w / 2, om, w, label="OSKM", color="#d1495b")
    ax.bar(xs + w / 2, rm, w, label="rest", color="#9bb7c4")
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in STAGES])
    ax.set_ylabel("mean |bias I|"); ax.set_title("OSKM carry ~2-3x the bias of the average gene, in every stage")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/H_oskm_vs_rest.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- I: null histograms with OSKM observed, per stage ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, (_, lab) in zip(axes, STAGES):
        obs, null, pct = null_pct(stageI[lab], genes, OSKM)
        ax.hist(null, bins=40, color="#c8d6dd")
        ax.axvline(obs, color="#d1495b", lw=2, label=f"OSKM (pct={pct:.0f})")
        ax.set_title(lab); ax.set_xlabel("mean |I| of random 4-gene set"); ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("OSKM bias vs the null of random 4-gene sets (n=5000)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/I_null_histograms.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- J: individual OSKM factor |I| across stages ----
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(OSKM)); w = 0.25
    for i, (_, lab) in enumerate(STAGES):
        vals = [stageI[lab][genes == g][0] if g in set(genes) else 0 for g in OSKM]
        ax.bar(xs + (i - 1) * w, vals, w, label=lab, color=SCOL[lab])
    ax.set_xticks(xs); ax.set_xticklabels(OSKM)
    ax.set_ylabel("|bias I|")
    ax.set_title("Per-factor bias: Myc strongest, then Pou5f1/Klf4; Sox2 the outlier\n(consistent with Sox2 being the dispensable reprogramming factor)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/J_oskm_per_factor.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- K: |I| rank curve per stage, OSKM marked ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, lab in STAGES:
        s = np.sort(stageI[lab])[::-1]
        ax.plot(np.arange(len(s)), s, color=SCOL[lab], label=lab, alpha=0.8)
    for g in OSKM:
        if g in set(genes):
            for _, lab in STAGES:
                r = int((stageI[lab] > stageI[lab][genes == g][0]).sum())
                ax.scatter(r, stageI[lab][genes == g][0], color="#d1495b", s=25, zorder=5)
    ax.set_xlabel("gene rank (by |I|)"); ax.set_ylabel("|bias I|")
    ax.set_title("Ranked bias per stage; red dots = OSKM factors"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/K_rank_curve.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    print(f"wrote reprogramming figures to {FIG}/ (G-K)", flush=True)


if __name__ == "__main__":
    main()
