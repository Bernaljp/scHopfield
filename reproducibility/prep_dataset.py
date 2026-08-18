"""Make an existing single-cell object scHopfield-ready, and record how it was done.

This is the preparation step for the datasets that arrive already normalized and smoothed
rather than as raw counts. It is a thin wrapper: the work is ``sch.pp.prepare_dataset``,
which is public API. What the wrapper adds is a record of the parameters the paper's objects
were prepared with, and a file rather than an interactive call.

Two of the seven datasets were prepared this way and then read from disk,
``<SCHOPFIELD_DATA>/generalize/murine_nc.h5ad`` and
``<SCHOPFIELD_DATA>/generalize/human_limb.h5ad``. Their inputs are objects that had already
been filtered, normalized, smoothed into ``layers['Ms']`` and given a neighbor graph
elsewhere, so this step adds only the velocity layer, the degradation rates and the per-gene
sigmoid fits. That earlier preprocessing is not part of this repository and is not
reproducible from it; see ``DATA_SOURCES.md``, which says per dataset what a reader can and
cannot rebuild.

    python reproducibility/prep_dataset.py --inp murine_nc_preprocessed.h5ad \
                                           --out <SCHOPFIELD_DATA>/generalize/murine_nc.h5ad

The same preparation can run inside the pipeline instead of ahead of it: a ``config.py``
dataset entry with ``prepare=True`` calls ``sch.pp.prepare_dataset`` on load, which is how
the Schwann-cell dataset is handled. The two routes do the same thing; this one writes the
prepared object out so repeated fits do not repeat the work.

The parameters below are pinned rather than left to the package defaults, which they match
today. A default that moves later would otherwise silently change what this script produces.
For an object that already carries ``layers['Ms']`` and a neighbor graph, the filtering,
smoothing and graph parameters do not apply: both objects above came in that state, one of
them with a 50-neighbor graph built upstream, and only the velocity, degradation and sigmoid
steps ran.

Needs scVelo, which is an extra rather than a hard dependency: ``pip install '.[velocity]'``.
"""
from __future__ import annotations
import argparse
import os

# ----------------------------------------------------------------------------- #
# The recipe, stated rather than inherited from sch.pp.prepare_dataset's defaults.
# ----------------------------------------------------------------------------- #
N_TOP_GENES = 2000          # matching config.N_GENES; applies only from raw counts
VELOCITY_MODE = "steady_state"
N_PCS = 30                  # moment smoothing and neighbor graph, from raw counts
N_NEIGHBORS = 30
MIN_SHARED_COUNTS = 20      # scVelo gene filter, from raw counts
FIT_SIGMOIDS = True


def main() -> None:
    import scanpy as sc
    import scHopfield as sch

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inp", required=True, help="the object to prepare")
    ap.add_argument("--out", required=True, help="where to write the prepared object")
    ap.add_argument("--n-top", type=int, default=N_TOP_GENES,
                    help="highly variable genes kept when starting from raw counts")
    args = ap.parse_args()

    a = sc.read_h5ad(args.inp)
    before = set(a.layers) | {f"var:{c}" for c in a.var.columns}
    print(f"loaded {args.inp}: {a.shape}; layers={sorted(a.layers)}", flush=True)

    sch.pp.prepare_dataset(
        a,
        n_top_genes=args.n_top,
        velocity_mode=VELOCITY_MODE,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        min_shared_counts=MIN_SHARED_COUNTS,
        fit_sigmoids=FIT_SIGMOIDS,
    )
    added = sorted((set(a.layers) | {f"var:{c}" for c in a.var.columns}) - before)
    print(f"usable genes: {a.n_vars}; added {added}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    a.write(args.out)
    print(f"wrote {args.out}: {a.shape}", flush=True)


if __name__ == "__main__":
    main()
