"""Demonstrate the M10 identifiability fix on real data (hematopoiesis).

M10 showed (on synthetic circuits) that broad state-space coverage, not model
expressiveness, is what identifies the interaction matrix. On real data the same
lever is neighbour augmentation (adding cells connected to a cluster but lying off
its manifold). Here, at a FIXED cell count, we vary the fraction of neighbour cells
and measure two identifiability readouts:

  (1) effective rank of the sigmoid design matrix sigma(X) (participation ratio of
      its singular values) -- how many independent directions the data spans;
  (2) split-half stability of the unconstrained least-squares W (Pearson correlation
      between W fit on two disjoint halves) -- how well-determined W is.

Holding the cell count fixed isolates the effect of broadening (composition) from
simply having more cells.
"""
import json
import os

import numpy as np
import anndata as ad
from scipy import sparse

CLUSTER_KEY = "paul15_clusters"
N_GENES = 100
N_TOTAL = 240
GAMMA = 0.1
FRACS = [0.0, 0.1, 0.2, 0.4]
SEEDS = [0, 1, 2]


def eff_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-9]
    return float((s.sum() ** 2) / (np.sum(s ** 2) + 1e-12))  # participation ratio


def fit_W(sig, v, x):
    A = np.hstack([sig, np.ones((sig.shape[0], 1))])
    WI = np.linalg.lstsq(A, v + GAMMA * x, rcond=None)[0]
    return WI[:-1].T


def main():
    adata = ad.read_h5ad("data/hematopoiesis/base_preprocessed.h5ad")
    vmag = np.abs(adata.layers["velocity_S"]).mean(0)
    keep = np.argsort(np.asarray(vmag).ravel())[::-1][:N_GENES]
    sig_all = np.asarray(adata.layers["sigmoid"])[:, keep]
    x_all = np.asarray(adata.layers["Ms"])[:, keep]
    v_all = np.asarray(adata.layers["velocity_S"])[:, keep]
    conn = adata.obsp["connectivities"]
    if not sparse.issparse(conn):
        conn = sparse.csr_matrix(conn)
    labels = adata.obs[CLUSTER_KEY].values

    # clusters with enough cells
    vc = adata.obs[CLUSTER_KEY].value_counts()
    clusters = [c for c in vc.index if vc[c] >= int(N_TOTAL * 0.7)][:4]
    print(f"clusters: {clusters}", flush=True)

    results = {f: {"eff_rank": [], "splithalf_W": []} for f in FRACS}
    for cl in clusters:
        cidx = np.where(labels == cl)[0]
        # neighbour pool: cells NOT in cluster but connected to some cluster cell
        nb_mask = np.asarray((conn[cidx].sum(0) > 0)).ravel()
        nb_pool = np.where(nb_mask & (labels != cl))[0]
        if len(nb_pool) < 10:
            continue
        for f in FRACS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                n_nb = int(round(f * N_TOTAL))
                n_cl = N_TOTAL - n_nb
                if n_cl > len(cidx) or n_nb > len(nb_pool):
                    continue
                sel_cl = rng.choice(cidx, n_cl, replace=False)
                sel_nb = rng.choice(nb_pool, n_nb, replace=False) if n_nb else np.array([], int)
                sel = np.concatenate([sel_cl, sel_nb]).astype(int)
                S, X, V = sig_all[sel], x_all[sel], v_all[sel]
                results[f]["eff_rank"].append(eff_rank(S))
                # split-half W stability
                perm = rng.permutation(len(sel)); h = len(sel) // 2
                W1 = fit_W(S[perm[:h]], V[perm[:h]], X[perm[:h]])
                W2 = fit_W(S[perm[h:2*h]], V[perm[h:2*h]], X[perm[h:2*h]])
                off = ~np.eye(W1.shape[0], dtype=bool)
                r = np.corrcoef(W1[off], W2[off])[0, 1]
                results[f]["splithalf_W"].append(float(r))

    summary = {}
    print("\nneighbour_frac | eff_rank(sigma) | split-half W corr", flush=True)
    for f in FRACS:
        er = np.mean(results[f]["eff_rank"]); sh = np.mean(results[f]["splithalf_W"])
        er_sd = np.std(results[f]["eff_rank"]); sh_sd = np.std(results[f]["splithalf_W"])
        summary[str(f)] = {"eff_rank_mean": round(er, 2), "eff_rank_sd": round(er_sd, 2),
                           "splithalf_W_mean": round(sh, 3), "splithalf_W_sd": round(sh_sd, 3),
                           "n": len(results[f]["eff_rank"])}
        print(f"  {f:.2f}          | {er:5.1f} +/- {er_sd:4.1f}   | {sh:.3f} +/- {sh_sd:.3f}", flush=True)

    os.makedirs("benchmark_results/real_identifiability", exist_ok=True)
    json.dump({"n_genes": N_GENES, "n_total": N_TOTAL, "clusters": list(clusters),
               "summary": summary}, open("benchmark_results/real_identifiability/results.json", "w"), indent=2)
    print("wrote benchmark_results/real_identifiability/results.json", flush=True)


if __name__ == "__main__":
    main()
