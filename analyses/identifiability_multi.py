"""Generalize the identifiability finding (M12) across multiple real datasets.

For each dataset, at a FIXED cell count, vary the fraction of neighbour (off-manifold)
cells added to a cluster and measure (1) effective rank of sigma(X) and (2) split-half
stability of the unconstrained W. Shows the M6/M10/M12 story holds across developmental
systems: broadening raises rank, but real data is too low-rank to identify W without a
scaffold.
"""
import json
import os

import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse

import scHopfield as sch

N_GENES, N_TOTAL, GAMMA = 100, 240, 0.1
FRACS = [0.0, 0.1, 0.2, 0.4]
SEEDS = [0, 1, 2]
DATASETS = [
    ("hematopoiesis", "data/hematopoiesis/base_preprocessed.h5ad", "paul15_clusters"),
    ("pancreas", "data/Pancreas/pancreas_scvelo_ready.h5ad", "clusters"),
    ("murine_NC", "data/generalize/murine_nc.h5ad", "celltype_update"),
    ("human_limb", "data/generalize/human_limb.h5ad", "leiden"),
]


def eff_rank(M):
    s = np.linalg.svd(M, compute_uv=False); s = s[s > 1e-9]
    return float((s.sum() ** 2) / (np.sum(s ** 2) + 1e-12))


def fit_W(sig, v, x):
    A = np.hstack([sig, np.ones((sig.shape[0], 1))])
    return np.linalg.lstsq(A, v + GAMMA * x, rcond=None)[0][:-1].T


def prep(path, cluster_key):
    a = sc.read_h5ad(path)
    if "sigmoid" not in a.layers:
        a.var["scHopfield_used"] = True
        sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values)
        sch.pp.compute_sigmoid(a)
    if "connectivities" not in a.obsp:
        sc.pp.neighbors(a, n_neighbors=30)
    return a


def run_dataset(name, path, ckey):
    a = prep(path, ckey)
    vmag = np.abs(a.layers["velocity_S"]).mean(0)
    keep = np.argsort(np.asarray(vmag).ravel())[::-1][:N_GENES]
    sig = np.asarray(a.layers["sigmoid"])[:, keep]
    x = np.asarray(a.layers["Ms"])[:, keep]
    v = np.asarray(a.layers["velocity_S"])[:, keep]
    conn = a.obsp["connectivities"]
    if not sparse.issparse(conn):
        conn = sparse.csr_matrix(conn)
    labels = a.obs[ckey].astype(str).values
    vc = a.obs[ckey].value_counts()
    clusters = [c for c in vc.index.astype(str) if vc[c] >= int(N_TOTAL * 0.7)][:4]
    res = {f: {"er": [], "sh": []} for f in FRACS}
    for cl in clusters:
        cidx = np.where(labels == cl)[0]
        nb_mask = np.asarray((conn[cidx].sum(0) > 0)).ravel()
        nb_pool = np.where(nb_mask & (labels != cl))[0]
        if len(nb_pool) < 10:
            continue
        for f in FRACS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                n_nb = int(round(f * N_TOTAL)); n_cl = N_TOTAL - n_nb
                if n_cl > len(cidx) or n_nb > len(nb_pool):
                    continue
                sel = np.concatenate([rng.choice(cidx, n_cl, replace=False),
                                      rng.choice(nb_pool, n_nb, replace=False) if n_nb else np.array([], int)]).astype(int)
                S, X, V = sig[sel], x[sel], v[sel]
                res[f]["er"].append(eff_rank(S))
                perm = rng.permutation(len(sel)); h = len(sel) // 2
                W1 = fit_W(S[perm[:h]], V[perm[:h]], X[perm[:h]])
                W2 = fit_W(S[perm[h:2*h]], V[perm[h:2*h]], X[perm[h:2*h]])
                off = ~np.eye(W1.shape[0], dtype=bool)
                res[f]["sh"].append(float(np.corrcoef(W1[off], W2[off])[0, 1]))
    return {str(f): {"eff_rank": round(float(np.mean(res[f]["er"])), 2),
                     "splithalf_W": round(float(np.mean(res[f]["sh"])), 3),
                     "n": len(res[f]["er"])} for f in FRACS}, clusters


def main():
    out = {}
    for name, path, ckey in DATASETS:
        if not os.path.exists(path):
            print(f"skip {name} (missing)", flush=True); continue
        summ, clusters = run_dataset(name, path, ckey)
        out[name] = {"clusters": list(clusters), "by_frac": summ}
        er0 = summ["0.0"]["eff_rank"]; er4 = summ["0.4"]["eff_rank"]
        sh = summ["0.0"]["splithalf_W"]
        print(f"{name:14s}: eff_rank {er0}->{er4} (neighbours 0->0.4), split-half W ~{sh}", flush=True)
    os.makedirs("benchmark_results/real_identifiability", exist_ok=True)
    json.dump(out, open("benchmark_results/real_identifiability/multi.json", "w"), indent=2)
    print("wrote benchmark_results/real_identifiability/multi.json", flush=True)


if __name__ == "__main__":
    main()
