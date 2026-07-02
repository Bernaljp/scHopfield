"""Generate a rich set of plots for the bias-penalty results (synthetic recovery,
M16). Reruns the controlled recovery on real pancreas sigma(x) on CPU (fast, 60
genes), collecting sweep metrics AND per-gene bias vectors, then writes many
figures to benchmark_results/bias_penalty/figures/.
"""
import os

import numpy as np
import anndata as ad
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import scHopfield as sch
from scHopfield.inference.optimizer import ScaffoldOptimizer
from scHopfield.inference.datasets import CustomDataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, K = 60, 6
LAMBDAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
PENS = ["l1", "l2", "elastic"]
COL = {"l1": "#2a9d8f", "l2": "#e76f51", "elastic": "#e9c46a"}
FIG = "benchmark_results/bias_penalty/figures"


def load():
    a = ad.read_h5ad("data/Pancreas/pancreas_scvelo_ready.h5ad")
    a.var["scHopfield_used"] = True
    if "sigmoid" not in a.layers:
        sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values)
        sch.pp.compute_sigmoid(a)
    lab = a.obs["clusters"].astype(str)
    cl = lab.value_counts().index[0]
    vmag = np.abs(np.asarray(a.layers["velocity_S"])).mean(0).ravel()
    g = np.argsort(vmag)[::-1][:N]
    m = (lab == cl).values
    return (np.asarray(a.layers["sigmoid"])[m][:, g].astype(np.float32),
            np.asarray(a.layers["Ms"])[m][:, g].astype(np.float32))


def fit(sig, x, v, gamma, pen, blam, epochs=500):
    m = ScaffoldOptimizer(g=gamma, scaffold=np.zeros((N, N), np.float32), device=DEV,
                          refit_gamma=False, scaffold_regularization=0.1,
                          reconstruction_regularization=1.0, bias_regularization=blam,
                          bias_penalty=pen, elastic_ratio=0.5)
    dl = DataLoader(CustomDataset(sig, v, x, DEV), batch_size=256, shuffle=True)
    m.train_model(dl, epochs=epochs, learning_rate=0.05, criterion="MSE", verbose=False)
    return m.I.detach().cpu().numpy()


def main():
    os.makedirs(FIG, exist_ok=True)
    sch.set_seed(0)
    sig, x = load()
    gamma = np.full(N, 0.1, np.float32)
    rng = np.random.default_rng(101)
    Wt = (rng.random((N, N)) < 0.15) * rng.normal(0, 1, (N, N)); np.fill_diagonal(Wt, 0)
    nat = sig @ Wt.T.astype(np.float32) - gamma * x
    scale = np.abs(nat).mean()
    forced = rng.choice(N, K, replace=False)
    It = np.zeros(N, np.float32)
    It[forced] = (rng.choice([-1., 1.], K) * rng.uniform(3, 6, K) * scale).astype(np.float32)
    noise = rng.normal(0, 0.1 * scale, nat.shape).astype(np.float32)
    v_forced = (nat + It[None, :] + noise).astype(np.float32)
    v_nat = (nat + noise).astype(np.float32)
    nonf = ~np.isin(np.arange(N), forced)
    y = np.zeros(N); y[forced] = 1
    print(f"device={DEV}; sweeping {len(PENS)}x{len(LAMBDAS)} ...", flush=True)

    # sweep: metrics + store I at a representative lambda
    M = {p: {"contrast": [], "auroc": [], "corr": [], "natres": []} for p in PENS}
    Irep = {}
    rep_lam = 0.1
    for p in PENS:
        for bl in LAMBDAS:
            If = fit(sig, x, v_forced, gamma, p, bl)
            In = fit(sig, x, v_nat, gamma, p, bl)
            M[p]["contrast"].append(np.abs(If[forced]).mean() / (np.abs(If[nonf]).mean() + 1e-9))
            M[p]["auroc"].append(roc_auc_score(y, np.abs(If)))
            M[p]["corr"].append(np.corrcoef(If, It)[0, 1])
            M[p]["natres"].append(np.abs(In).mean())
            if abs(bl - rep_lam) < 1e-9:
                Irep[p] = (If, In)
        print(f"  {p} done", flush=True)

    # ---------- Fig A: metric-vs-lambda (2x2) ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("contrast", "forced/non-forced |I| contrast (higher=sparser)", True),
              ("auroc", "AUROC: |I| identifies forced genes", False),
              ("corr", "corr(I_rec, I_true): magnitude+sign", False),
              ("natres", "NATURAL residual mean|I| (lower=better; L2 blows up)", True)]
    for ax, (k, title, logy) in zip(axes.flat, panels):
        for p in PENS:
            ax.plot(LAMBDAS, M[p][k], "o-", color=COL[p], label=p)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("bias_lambda"); ax.set_title(title); ax.legend()
    fig.suptitle("Bias penalty sweep (synthetic recovery on real pancreas sigma)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/A_metrics_vs_lambda.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---------- Fig B: the "does both" scatter ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    for p in PENS:
        ax.plot(M[p]["natres"], M[p]["contrast"], "o-", color=COL[p], label=p, alpha=0.8)
        for i, bl in enumerate(LAMBDAS):
            ax.annotate(f"{bl:g}", (M[p]["natres"][i], M[p]["contrast"][i]), fontsize=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("natural residual mean|I|  (want small ->)")
    ax.set_ylabel("forced contrast  (want large ^)")
    ax.set_title("Only L1 reaches the top-left: sparse forced bias AND no natural takeover")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/B_does_both_scatter.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---------- Fig C: per-gene recovered |I| (truth vs penalties) ----------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    gi = np.arange(N); col = np.where(np.isin(gi, forced), "#d1495b", "#9bb7c4")
    series = [("ground truth", np.abs(It))] + [(p, np.abs(Irep[p][0])) for p in PENS]
    for ax, (title, vals) in zip(axes, series):
        ax.bar(gi, vals, color=col); ax.set_title(title); ax.set_xlabel("gene"); ax.set_ylabel("|I|")
    fig.suptitle("Recovered bias per gene (forced scenario, bias_lambda=0.1). Red=forced", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/C_per_gene_recovery.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---------- Fig D: I_rec vs I_true scatter ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, p in zip(axes, PENS):
        If = Irep[p][0]
        ax.scatter(It[nonf], If[nonf], s=18, color="#9bb7c4", label="natural")
        ax.scatter(It[forced], If[forced], s=45, color="#d1495b", label="forced")
        lim = max(np.abs(It).max(), np.abs(If).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax.set_xlabel("true I"); ax.set_ylabel("recovered I"); ax.set_title(p); ax.legend(fontsize=8)
    fig.suptitle("Recovery of the true bias (identity = perfect)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/D_Irec_vs_Itrue.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---------- Fig E: |I| distribution on non-forced genes (sparsity) ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [np.abs(Irep[p][0][nonf]) for p in PENS]
    parts = ax.violinplot(data, showmedians=True)
    for pc, p in zip(parts["bodies"], PENS):
        pc.set_facecolor(COL[p]); pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(PENS)
    ax.set_ylabel("|I| on NON-forced genes (want ~0)")
    ax.set_title("L1 drives non-forced biases to ~0 (sparse); L2 leaves them spread")
    fig.tight_layout(); fig.savefig(f"{FIG}/E_nonforced_distribution.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---------- Fig F: sparsity curve (fraction ~0 vs threshold) ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    ths = np.linspace(0, np.abs(Irep["l2"][0][nonf]).max(), 50)
    for p in PENS:
        an = np.abs(Irep[p][0][nonf])
        ax.plot(ths, [(an <= t).mean() for t in ths], color=COL[p], label=p)
    ax.set_xlabel("|I| threshold"); ax.set_ylabel("fraction of non-forced genes below threshold")
    ax.set_title("Sparsity: L1 has far more near-zero biases"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/F_sparsity_curve.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    print(f"wrote synthetic figures to {FIG}/ (A-F)", flush=True)


if __name__ == "__main__":
    main()
