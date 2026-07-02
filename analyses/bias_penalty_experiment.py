"""Controlled recovery test for the bias-term penalty (L2 vs L1 vs elastic).

Question: the model fits v = W.sigma(x) + I - gamma.x. The bias I is a free
per-gene intercept that is confounded with W.sigma (sigma is near-constant within
a cluster). We want I to be ~0 under natural GRN control but large on genes under
a real external input. That is a *sparsity* requirement -> L1 should beat the
current L2-norm penalty.

Design: use real sigma(x), x from one cluster (realistic confounding). Build a
ground-truth W, gamma, and a SPARSE I_true (K forced genes, rest 0). Generate
v = W_true.sigma + I_true - gamma.x + noise. Refit (W, I) with each bias penalty
and measure how well I_true is recovered. Two scenarios:
  - natural: I_true = 0 everywhere (bias should stay ~0, no hallucination)
  - forced : I_true sparse (bias should localize to the forced genes)
"""
import json
import os

import numpy as np
import anndata as ad
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import scHopfield as sch
from scHopfield.inference.optimizer import ScaffoldOptimizer
from scHopfield.inference.datasets import CustomDataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_GENES = 60
K_FORCED = 6
GT_SEEDS = [0, 1]
PENALTIES = ["l2", "l1", "elastic"]
BIAS_LAMBDAS = [0.1, 1.0, 10.0]
OUT = "benchmark_results/bias_penalty"


def load_cluster():
    a = ad.read_h5ad("data/Pancreas/pancreas_scvelo_ready.h5ad")
    if "sigmoid" not in a.layers:
        a.var["scHopfield_used"] = True
        sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values)
        sch.pp.compute_sigmoid(a)
    lab = a.obs["clusters"].astype(str)
    cl = lab.value_counts().index[0]
    vmag = np.abs(np.asarray(a.layers["velocity_S"])).mean(0).ravel()
    genes = np.argsort(vmag)[::-1][:N_GENES]
    m = (lab == cl).values
    sig = np.asarray(a.layers["sigmoid"])[m][:, genes].astype(np.float32)
    x = np.asarray(a.layers["Ms"])[m][:, genes].astype(np.float32)
    return sig, x, cl


def fit_recover(sig, x, v, gamma, penalty, blam, epochs=500):
    scaffold = np.zeros((N_GENES, N_GENES), np.float32)  # no prior -> W is L1+L2 regularized
    m = ScaffoldOptimizer(
        g=gamma, scaffold=scaffold, device=DEV, refit_gamma=False,
        scaffold_regularization=0.1, reconstruction_regularization=1.0,
        bias_regularization=blam, bias_penalty=penalty, elastic_ratio=0.5)
    ds = CustomDataset(sig, v, x, DEV)
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)
    m.train_model(dl, epochs=epochs, learning_rate=0.05, criterion="MSE", verbose=False)
    return m.I.detach().cpu().numpy()


def run():
    os.makedirs(OUT, exist_ok=True)
    sch.set_seed(0)
    sig, x, cl = load_cluster()
    print(f"cluster '{cl}': {sig.shape[0]} cells x {N_GENES} genes; device={DEV}", flush=True)
    gamma = np.full(N_GENES, 0.1, np.float32)

    rows = []
    for scenario in ["natural", "forced"]:
        for seed in GT_SEEDS:
            rng = np.random.default_rng(100 + seed)
            Wtrue = (rng.random((N_GENES, N_GENES)) < 0.15) * rng.normal(0, 1, (N_GENES, N_GENES))
            np.fill_diagonal(Wtrue, 0.0)
            nat = sig @ Wtrue.T.astype(np.float32) - gamma * x            # natural velocity part
            scale = np.abs(nat).mean()
            Itrue = np.zeros(N_GENES, np.float32)
            forced = np.array([], int)
            if scenario == "forced":
                forced = rng.choice(N_GENES, K_FORCED, replace=False)
                Itrue[forced] = (rng.choice([-1.0, 1.0], K_FORCED)
                                 * rng.uniform(3.0, 6.0, K_FORCED) * scale).astype(np.float32)
            noise = rng.normal(0, 0.1 * scale, nat.shape).astype(np.float32)
            v = (nat + Itrue[None, :] + noise).astype(np.float32)

            y = np.zeros(N_GENES)
            y[forced] = 1.0
            nonforced = ~np.isin(np.arange(N_GENES), forced)
            for pen in PENALTIES:
                for blam in BIAS_LAMBDAS:
                    Irec = fit_recover(sig, x, v, gamma, pen, blam)
                    med_nonforced = float(np.median(np.abs(Irec[nonforced])))
                    rec = {"scenario": scenario, "seed": seed, "penalty": pen,
                           "bias_lambda": blam, "med_abs_nonforced": med_nonforced,
                           "mean_abs_all": float(np.abs(Irec).mean())}
                    if scenario == "forced":
                        rec["auroc"] = float(roc_auc_score(y, np.abs(Irec)))
                        rec["corr"] = float(np.corrcoef(Irec, Itrue)[0, 1])
                        rec["contrast"] = float(np.abs(Irec[forced]).mean()
                                                / (np.abs(Irec[nonforced]).mean() + 1e-9))
                    rows.append(rec)
                    tag = (f"AUROC={rec.get('auroc', float('nan')):.3f} corr={rec.get('corr', float('nan')):.3f} "
                           f"contrast={rec.get('contrast', float('nan')):.1f}") if scenario == "forced" \
                        else f"mean|I|={rec['mean_abs_all']:.4f}"
                    print(f"  [{scenario:7s} s{seed}] {pen:8s} blam={blam:6.2f}  {tag}", flush=True)

    json.dump(rows, open(f"{OUT}/recovery.json", "w"), indent=2)

    # ---- summary: best AUROC per penalty (forced) + false-bias per penalty (natural) ----
    print("\n=== FORCED: mean over seeds, best bias_lambda per penalty ===", flush=True)
    for pen in PENALTIES:
        best = None
        for blam in BIAS_LAMBDAS:
            sub = [r for r in rows if r["scenario"] == "forced" and r["penalty"] == pen and r["bias_lambda"] == blam]
            auroc = np.mean([r["auroc"] for r in sub])
            corr = np.mean([r["corr"] for r in sub])
            contrast = np.mean([r["contrast"] for r in sub])
            if best is None or auroc > best[1]:
                best = (blam, auroc, corr, contrast)
        print(f"  {pen:8s} best blam={best[0]:6.2f}: AUROC={best[1]:.3f} corr={best[2]:.3f} contrast={best[3]:.1f}", flush=True)
    print("\n=== NATURAL (I_true=0): residual bias per penalty (mean|I|, lower=better) ===", flush=True)
    for pen in PENALTIES:
        for blam in BIAS_LAMBDAS:
            sub = [r for r in rows if r["scenario"] == "natural" and r["penalty"] == pen and r["bias_lambda"] == blam]
            print(f"  {pen:8s} blam={blam:6.2f}: mean|I|={np.mean([r['mean_abs_all'] for r in sub]):.4f}", flush=True)
    print(f"\nwrote {OUT}/recovery.json", flush=True)


if __name__ == "__main__":
    run()
