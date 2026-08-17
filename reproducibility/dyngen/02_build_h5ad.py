"""Assemble an AnnData per dyngen dataset from the R exports: spliced/unspliced counts,
the ground-truth signed GRN (W_true), the TF mask, and a scVelo velocity target. Saves
<name>/adata.h5ad, W_true.npy, tf_mask.npy, gene_names.npy.

Run:  python reproducibility/dyngen/02_build_h5ad.py
"""
import os, sys, glob
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scipy.io, scipy.sparse as sp
import anndata as ad
import scanpy as sc
import scvelo as scv

# Anchored to this file rather than to the working directory, so the two-command
# reproduction runs from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                                     # noqa: E402

ROOT = paths.DYNGEN


def build(name):
    d = os.path.join(ROOT, name)
    if not os.path.exists(os.path.join(d, "spliced.mtx")):
        print(f"[skip {name}] no export"); return
    spliced = scipy.io.mmread(os.path.join(d, "spliced.mtx")).tocsr().astype(np.float32)
    unspliced = scipy.io.mmread(os.path.join(d, "unspliced.mtx")).tocsr().astype(np.float32)
    genes = [l.strip() for l in open(os.path.join(d, "genes.txt"))]
    cells = [l.strip() for l in open(os.path.join(d, "cells.txt"))]

    a = ad.AnnData(X=spliced.copy(), obs=pd.DataFrame(index=cells),
                   var=pd.DataFrame(index=genes))
    a.layers["spliced"] = spliced
    a.layers["unspliced"] = unspliced

    # gene metadata (TFs / housekeeping)
    fi = pd.read_csv(os.path.join(d, "feature_info.csv")).set_index("feature_id")
    a.var["is_tf"] = fi.reindex(genes)["is_tf"].fillna(False).astype(bool).values
    a.var["is_hk"] = fi.reindex(genes)["is_hk"].fillna(False).astype(bool).values
    if "module_id" in fi.columns:
        a.var["module_id"] = fi.reindex(genes)["module_id"].astype(str).values

    # cell metadata (simulation time)
    ci = pd.read_csv(os.path.join(d, "cell_info.csv"))
    if "sim_time" in ci.columns:
        a.obs["sim_time"] = ci.set_index("cell_id").reindex(cells)["sim_time"].values
    a.obs["cluster"] = "all"          # one global GRN (ground truth is a single network)
    a.obs["cluster"] = a.obs["cluster"].astype("category")

    # ground-truth signed interaction matrix: W_true[target i, regulator j] = effect
    idx = {g: k for k, g in enumerate(genes)}
    fn = pd.read_csv(os.path.join(d, "feature_network.csv"))
    N = len(genes); W = np.zeros((N, N), np.float32)
    for frm, to, eff in zip(fn["from"], fn["to"], fn["effect"]):
        if frm in idx and to in idx:
            W[idx[to], idx[frm]] = float(eff)      # regulator frm -> target to
    tf_mask = a.var["is_tf"].values.copy()

    # scVelo velocity target (realistic pipeline: moments -> steady-state velocity).
    # Keep ALL genes (the ground truth spans all genes); just per-cell normalize.
    scv.pp.normalize_per_cell(a)
    scv.pp.moments(a, n_pcs=30, n_neighbors=min(30, a.n_obs - 1))
    scv.tl.velocity(a, mode="steady_state")
    a.layers["velocity"] = np.nan_to_num(np.asarray(a.layers["velocity"], dtype=np.float32))
    # degradation rate gamma (scVelo steady-state slope) for the Hopfield model v = W.sig - gamma.x + I
    gk = "velocity_gamma" if "velocity_gamma" in a.var else "fit_gamma"
    a.var["gamma"] = np.nan_to_num(np.asarray(a.var[gk], dtype=np.float32), nan=1.0)
    a.var.loc[a.var["gamma"] <= 0, "gamma"] = float(np.nanmedian(a.var["gamma"][a.var["gamma"] > 0]))
    sc.tl.pca(a, n_comps=min(30, a.n_vars - 1))
    sc.pp.neighbors(a, n_neighbors=min(30, a.n_obs - 1))
    sc.tl.umap(a)

    # keep only genes that survived scVelo filtering; re-align W/tf_mask
    kept = list(a.var_names)
    keep_idx = [idx[g] for g in kept]
    W = W[np.ix_(keep_idx, keep_idx)]
    tf_mask = a.var["is_tf"].values.copy()

    a.write_h5ad(os.path.join(d, "adata.h5ad"))
    np.save(os.path.join(d, "W_true.npy"), W)
    np.save(os.path.join(d, "tf_mask.npy"), tf_mask)
    np.save(os.path.join(d, "gene_names.npy"), np.array(kept, dtype=object))
    n_edge = int((np.abs(W) > 0).sum())
    print(f"[{name}] {a.n_obs} cells x {a.n_vars} genes | {n_edge} GT edges "
          f"({int((W>0).sum())}+/{int((W<0).sum())}-) | {int(tf_mask.sum())} TFs")


if __name__ == "__main__":
    names = sorted([os.path.basename(os.path.dirname(p))
                    for p in glob.glob(os.path.join(ROOT, "*", "spliced.mtx"))])
    print("datasets:", names)
    for nm in names:
        try:
            build(nm)
        except Exception as e:
            print(f"[FAIL {nm}] {type(e).__name__}: {e}")
