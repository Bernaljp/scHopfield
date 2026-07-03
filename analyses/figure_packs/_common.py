"""Shared helpers for the scHopfield figure packs.

All packs read the re-fitted adatas under ``figure_packs/_fits/<dataset>/`` (produced by
``analyses/run_full_pipeline.py --out-root figure_packs/_fits``) and write into
``figure_packs/<pack>/`` -- the whole ``figure_packs/`` tree is gitignored.
"""
import os

import numpy as np
import anndata as ad

FITS = "figure_packs/_fits"

# dataset -> cluster key (matches run_full_pipeline.DATASETS)
DATASETS = {
    "hematopoiesis": "paul15_clusters",
    "pancreas": "clusters",
    "murine_nc": "celltype_update",
    "human_limb": "leiden_R_celltype",
    "schwann": "location",
}

# lineage definitions for the perturbation packs (A vs B), where known.
LINEAGES = {
    "hematopoiesis": dict(A=["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"],
                          B=["9GMP", "10GMP", "11DC", "14Mo", "15Mo", "16Neu", "17Neu"],
                          A_name="erythroid", B_name="myeloid"),
    "pancreas": dict(A=["Alpha"], B=["Beta"], A_name="alpha", B_name="beta"),
}


def available():
    """Return {dataset: cluster_key} for datasets whose fit exists on disk."""
    out = {}
    for name, ck in DATASETS.items():
        if os.path.exists(f"{FITS}/{name}/adata_fitted.h5ad"):
            out[name] = ck
    return out


def load(name):
    return ad.read_h5ad(f"{FITS}/{name}/adata_fitted.h5ad")


def present_clusters(adata, cluster_key, min_cells=20):
    lab = adata.obs[cluster_key].astype(str)
    vc = lab.value_counts()
    return [c for c in vc.index if vc[c] >= min_cells]


def W_of(adata, cluster):
    """Interaction matrix for a cluster: W[target, regulator]."""
    return np.asarray(adata.varp[f"W_{cluster}"])


def basis_of(adata):
    """Best available 2D embedding key (without the X_ prefix)."""
    for b in ("draw_graph_fa", "umap", "pca"):
        if f"X_{b}" in adata.obsm:
            return b
    return "umap"
