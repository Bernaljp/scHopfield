"""General velocity-preprocessing to make a dataset scHopfield-ready.

Produces layers['Ms'], layers['velocity_S'], layers['sigmoid'], var['gamma'],
obsp['connectivities'], var['scHopfield_used']. Handles datasets that already have
moments (Ms/Mu) and raw spliced/unspliced datasets.
"""
import argparse
import os

import numpy as np
import scanpy as sc
import scvelo as scv

import scHopfield as sch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-top", type=int, default=2000)
    args = ap.parse_args()

    scv.settings.verbosity = 1
    a = sc.read_h5ad(args.inp)
    print(f"loaded {args.inp}: {a.shape}; layers={list(a.layers)}", flush=True)

    if "Ms" not in a.layers:
        scv.pp.filter_genes(a, min_shared_counts=20)
        scv.pp.normalize_per_cell(a)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=min(args.n_top, a.n_vars))
        a = a[:, a.var["highly_variable"]].copy()
        scv.pp.moments(a, n_pcs=30, n_neighbors=30)
    elif "connectivities" not in a.obsp:
        scv.pp.moments(a, n_pcs=30, n_neighbors=30)

    scv.tl.velocity(a, mode="steady_state")
    a.layers["velocity_S"] = a.layers["velocity"]
    a.var["gamma"] = np.asarray(a.var["velocity_gamma"]).astype(np.float32)

    finite_g = np.isfinite(a.var["gamma"].values) & (a.var["gamma"].values > 0)
    vel = np.asarray(a.layers["velocity_S"])
    finite_v = np.isfinite(vel).all(axis=0)
    keep = finite_g & finite_v
    a = a[:, keep].copy()
    print(f"usable genes: {a.n_vars}", flush=True)

    a.var["scHopfield_used"] = True
    sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values)
    sch.pp.compute_sigmoid(a)

    assert "connectivities" in a.obsp, "no connectivities"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    a.write(args.out)
    print(f"wrote {args.out}: {a.shape}", flush=True)


if __name__ == "__main__":
    main()
