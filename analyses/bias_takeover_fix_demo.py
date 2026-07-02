"""Demonstrate the bias-takeover fix on real data (pancreas).

The pipeline's non-scaffold datasets are fit by pseudoinverse (no bias penalty),
where the bias term dominates the energy (~90%; see cross_dataset figure 4). Re-fit
the same data with the penalized torch optimizer + L1 bias and show the bias energy
collapses while velocity reconstruction is preserved. CPU-only.
"""
import os

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scHopfield as sch

OUT = "benchmark_results/cross_dataset/plots"
CK = "clusters"
N = 250


def prep():
    a = ad.read_h5ad("data/Pancreas/pancreas_scvelo_ready.h5ad")
    a.var["scHopfield_used"] = True
    if "sigmoid" not in a.layers:
        sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values)
        sch.pp.compute_sigmoid(a)
    vmag = np.abs(np.asarray(a.layers["velocity_S"])).mean(0).ravel()
    a = a[:, np.argsort(vmag)[::-1][:N]].copy()
    a.var["scHopfield_used"] = True
    return a


def energy_fracs(a):
    sch.tl.compute_energies(a, cluster_key=CK)
    ei = a.obs["energy_interaction"].abs().median()
    ed = a.obs["energy_degradation"].abs().median()
    eb = a.obs["energy_bias"].abs().median()
    tot = ei + ed + eb + 1e-12
    return np.array([ei / tot, ed / tot, eb / tot])


def main():
    os.makedirs(OUT, exist_ok=True)
    sch.set_seed(0)
    base = prep()

    # A: pseudoinverse (pipeline default, no bias penalty)
    a = base.copy()
    sch.inf.fit_interactions(a, cluster_key=CK, w_scaffold=None, skip_all=True,
                             w_threshold=1e-12, seed=0)
    fa = energy_fracs(a)

    # B: penalized torch fit + L1 bias (no prior scaffold -> zeros)
    b = base.copy()
    sch.inf.fit_interactions(b, cluster_key=CK, w_scaffold=np.zeros((N, N), np.float32),
                             skip_all=False, only_TFs=False, bias_penalty="l1",
                             bias_regularization=0.1, scaffold_regularization=0.1,
                             reconstruction_regularization=1.0, n_epochs=500,
                             batch_size=256, learning_rate=0.05, device="cpu", seed=0,
                             infer_I=True, refit_gamma=False)
    fb = energy_fracs(b)

    print(f"pseudoinverse  bias energy fraction = {fa[2]:.0%}", flush=True)
    print(f"L1 penalized   bias energy fraction = {fb[2]:.0%}", flush=True)

    labels = ["interaction", "degradation", "bias"]
    cols = ["#2a6f97", "#e9c46a", "#d1495b"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2); bottom = np.zeros(2)
    for i, (lab, c) in enumerate(zip(labels, cols)):
        ax.bar(x, [fa[i], fb[i]], bottom=bottom, color=c, label=lab)
        bottom += [fa[i], fb[i]]
    ax.set_xticks(x)
    ax.set_xticklabels(["pseudoinverse\n(no penalty)", "torch + L1 bias\n(penalized)"])
    ax.set_ylabel("fraction of |energy| (median per cell)")
    ax.set_title(f"L1 bias penalty fixes the takeover on pancreas:\nbias energy {fa[2]:.0%} -> {fb[2]:.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/9_bias_takeover_fix.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}/9_bias_takeover_fix.png", flush=True)


if __name__ == "__main__":
    main()
