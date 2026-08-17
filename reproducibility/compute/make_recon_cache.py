"""Median per-cell velocity cosine per dataset, cell-type-specific fit against one global fit.

Feeds panel f of Extended Data Fig. 3. Neither arm needs a refit: the canonical call already
stores the single all-cells matrix as ``varp["W_all"]`` beside the per-cell-type ones, so both
fields are read from the same file and scored by the same statistic.

Both arms are IN SAMPLE, and the cell-type-specific model carries one matrix per cell type, so
the contrast is fit quality and not generalization. The panel title says so.

Run:  python reproducibility/compute/make_recon_cache.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
# The reproducibility tree is flat one level up; this directory holds the compute
# helpers. Both are anchored to this file rather than to the working directory, so a
# script runs the same from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402

OUT = os.path.join(paths.CACHE, "_recon_cache.json")
DATASETS = ["paul15", "paul15_coarse", "dynamo_hematopoiesis", "pancreas",
            "murine_nc", "schwann", "human_limb"]


def _dense(M):
    """AnnData layers may be sparse, and np.asarray on a sparse matrix gives a 0-d object
    array that fails silently later. Densify explicitly."""
    return np.asarray(M.todense()) if sp.issparse(M) else np.asarray(M)


def _row_cosine(A, B):
    n = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return (A * B).sum(1) / np.where(n > 0, n, np.nan)


def main() -> int:
    import anndata as ad
    from config import DATASETS as CFG

    out = {}
    for ds in DATASETS:
        path = f"{paths.REPORTS}/{ds}/data/adata_analyzed.h5ad"
        if not os.path.exists(path):
            print(f"  [skip {ds}: no fitted object]")
            continue
        a = ad.read_h5ad(path)
        ck = CFG[ds]["cluster_key"]
        vk = CFG[ds].get("velocity_key", "velocity_S")
        V = _dense(a.layers[vk] if vk in a.layers else a.layers["velocity"])
        x = _dense(a.layers["Ms"])
        sig = _dense(a.layers["sigmoid"])
        lab = a.obs[ck].astype(str)
        used = (a.var["scHopfield_used"].values.astype(bool)
                if "scHopfield_used" in a.var else np.ones(a.n_vars, bool))

        pred = np.zeros_like(x)                                  # cell-type-specific field
        for c in lab.unique():
            if f"W_{c}" not in a.varp:
                continue
            m = (lab == c).values
            W = _dense(a.varp[f"W_{c}"])
            I = np.asarray(a.var[f"I_{c}"]) if f"I_{c}" in a.var else 0.0
            g = (np.asarray(a.var[f"gamma_{c}"]) if f"gamma_{c}" in a.var
                 else np.asarray(a.var["gamma"]))
            pred[m] = sig[m] @ W.T + I - g * x[m]

        Wa = _dense(a.varp["W_all"])                             # one global field
        Ia = np.asarray(a.var["I_all"]) if "I_all" in a.var else 0.0
        ga = (np.asarray(a.var["gamma_all"]) if "gamma_all" in a.var
              else np.asarray(a.var["gamma"]))
        pred_all = sig @ Wa.T + Ia - ga * x

        ct = float(np.nanmedian(_row_cosine(pred[:, used], V[:, used])))
        gl = float(np.nanmedian(_row_cosine(pred_all[:, used], V[:, used])))
        out[ds] = {"cos_celltype": ct, "cos_global": gl}
        print(f"  {ds:22s} cell-type {ct:.4f}   global {gl:.4f}   gain {ct - gl:+.4f}")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
