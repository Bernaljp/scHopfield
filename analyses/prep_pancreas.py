"""Produce a scHopfield-ready pancreas dataset via scVelo steady-state velocity.

Downloads the pancreatic endocrinogenesis dataset (Bastidas-Ponce 2019) through
scVelo, runs standard preprocessing + steady-state velocity, and maps the layers
to the names scHopfield expects: layers['Ms'], layers['velocity_S'], var['gamma'].
"""
import os
import numpy as np
import scvelo as scv


def main(out="data/Pancreas/pancreas_scvelo_ready.h5ad", n_top_genes=2000):
    scv.settings.verbosity = 1
    adata = scv.datasets.pancreas()
    print("raw:", adata.shape)

    import scanpy as sc
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="steady_state")  # -> layers['velocity'], var['velocity_gamma']

    # Map to scHopfield conventions
    adata.layers["velocity_S"] = adata.layers["velocity"]
    adata.var["gamma"] = adata.var["velocity_gamma"].astype(np.float32)

    # Keep genes with finite gamma and finite velocity everywhere
    finite_gamma = np.isfinite(adata.var["gamma"].values)
    vel = np.asarray(adata.layers["velocity_S"])
    finite_vel = np.isfinite(vel).all(axis=0)
    keep = finite_gamma & finite_vel & (adata.var["gamma"].values > 0)
    print(f"genes with usable steady-state fit: {keep.sum()} / {adata.n_vars}")
    adata = adata[:, keep].copy()

    # sanity: required keys
    assert "Ms" in adata.layers, "Ms missing"
    assert "velocity_S" in adata.layers, "velocity_S missing"
    assert "gamma" in adata.var, "gamma missing"
    assert "clusters" in adata.obs, "clusters missing"

    os.makedirs(os.path.dirname(out), exist_ok=True)
    adata.write(out)
    print(f"wrote {out}: {adata.shape}; clusters={list(adata.obs['clusters'].cat.categories)}")


if __name__ == "__main__":
    main()
