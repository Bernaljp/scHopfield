"""Rebuild the pancreas input object that the report pipeline fits.

Pancreatic endocrinogenesis (Bastidas-Ponce et al. 2019, GEO GSE132188) is the one dataset
behind the paper's figures whose whole chain is reproducible from a package call: scVelo
downloads the raw object itself, and this script applies the preprocessing recipe the fitted
object was made with. It writes

    <SCHOPFIELD_DATA>/Pancreas/pancreas_scvelo_ready.h5ad

which is the path ``config.py`` names as the pancreas input, so running this once is enough
for ``rutils.prepare_and_fit("pancreas")`` to find its input. Five of the six figures that
read the report tree are drawn from the pancreas branch of it.

The recipe is written out here rather than inherited from a library default, including the
parameters that happen to agree with ``sch.pp.prepare_dataset``'s defaults today. A default
that moves later would otherwise silently change what this script produces, and the object
below is the one the published fits were run on. The values are also the ones the object
itself records: ``uns['velocity_params']['mode'] == 'steady_state'`` and
``uns['neighbors']['params']`` with ``n_neighbors = 30``, ``n_pcs = 30``.

Both paths are anchored to ``paths.py``, so the working directory does not matter. That
includes the download: ``scvelo.datasets.pancreas()`` writes to ``data/Pancreas/`` relative
to the working directory unless it is told otherwise, which would leave a second copy of the
raw object wherever the script happened to be run from.

Run:

    python reproducibility/prep_pancreas.py                  # about a minute, CPU
    python reproducibility/prep_pancreas.py --out other.h5ad

Needs scVelo, which is an extra rather than a hard dependency: ``pip install '.[velocity]'``.
"""
from __future__ import annotations
import argparse
import os

import numpy as np

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402

# ----------------------------------------------------------------------------- #
# The recipe. Changing any of these produces a different object from the one the
# paper's pancreas fits were run on.
# ----------------------------------------------------------------------------- #
MIN_SHARED_COUNTS = 20      # scVelo gene filter
N_TOP_GENES = 2000          # highly variable genes kept, matching config.N_GENES
N_PCS = 30                  # moment smoothing and neighbor graph
N_NEIGHBORS = 30
VELOCITY_MODE = "steady_state"

RAW = os.path.join(paths.DATASETS, "Pancreas", "endocrinogenesis_day15.h5ad")
OUT = os.path.join(paths.DATASETS, "Pancreas", "pancreas_scvelo_ready.h5ad")


def main(raw: str = RAW, out: str = OUT) -> None:
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 1
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    # Downloads on the first run and reads the cached copy afterwards.
    adata = scv.datasets.pancreas(file_path=raw)
    print(f"raw: {adata.shape}", flush=True)

    scv.pp.filter_genes(adata, min_shared_counts=MIN_SHARED_COUNTS)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES)
    adata = adata[:, adata.var["highly_variable"]].copy()
    scv.pp.moments(adata, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS)
    scv.tl.velocity(adata, mode=VELOCITY_MODE)   # layers['velocity'], var['velocity_gamma']

    # Map scVelo's names onto the ones scHopfield reads.
    adata.layers["velocity_S"] = adata.layers["velocity"]
    adata.var["gamma"] = adata.var["velocity_gamma"].astype(np.float32)

    # Keep genes whose steady-state fit is usable: a finite positive degradation rate and a
    # finite velocity in every cell.
    finite_gamma = np.isfinite(adata.var["gamma"].values)
    vel = np.asarray(adata.layers["velocity_S"])
    finite_vel = np.isfinite(vel).all(axis=0)
    keep = finite_gamma & finite_vel & (adata.var["gamma"].values > 0)
    print(f"genes with usable steady-state fit: {keep.sum()} / {adata.n_vars}", flush=True)
    adata = adata[:, keep].copy()

    # What the fit reads. Failing here means the object is not scHopfield-ready.
    assert "Ms" in adata.layers, "Ms missing"
    assert "velocity_S" in adata.layers, "velocity_S missing"
    assert "gamma" in adata.var, "gamma missing"
    assert "clusters" in adata.obs, "clusters missing"

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    adata.write(out)
    print(f"wrote {out}: {adata.shape}; "
          f"clusters={list(adata.obs['clusters'].cat.categories)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=RAW,
                    help="where scVelo's download is read from or written to")
    ap.add_argument("--out", default=OUT, help="the prepared object config.py reads")
    args = ap.parse_args()
    main(raw=args.raw, out=args.out)
