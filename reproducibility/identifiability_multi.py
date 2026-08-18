"""Real-data identifiability sweep: velocity alone does not determine W.

Writes the JSON that Extended Data Fig. 3 panel g is drawn from:

    <reproducibility>/data/real_identifiability/multi.json

For each of four datasets, one per biological system, and at a FIXED cell count, this
varies the fraction of neighboring (off-manifold) cells mixed into a cluster and measures
two things:

  eff_rank      the effective rank of the activation matrix phi(X), a participation
                ratio over its singular values, in units of genes
  splithalf_W   the split-half stability of the UNCONSTRAINED interaction matrix. The
                same sample is fit twice on disjoint halves and the two off-diagonals
                are correlated

Broadening the sample raises the rank, and the split-half correlation stays near zero
anyway. That is the concrete, real-data reason a transcription-factor scaffold prior is
needed, and it is the claim the panel carries.

The baseline here is deliberately NOT the configuration the paper reports. It is an
unconstrained least-squares fit on 100 velocity-selected genes with a single fixed
gamma = 0.1, no scaffold, no bias penalty and no boundedness term, because the point is
what velocity alone determines. It is the contrast to the scaffolded fit, not a version
of it.

## The pinned sigmoid recipe, which is load-bearing

The sweep needs an activation matrix, so it fits the sigmoids itself, and it does so for
every dataset with the recipe stated in ``SIGMOID_FIT`` below rather than at the library
defaults. Two reasons, both measured rather than assumed:

1. ``sch.pp.fit_all_sigmoids`` was reimplemented after this artifact was produced. Run at
   today's defaults, the pancreas arm moves from an effective rank of 13.87 to 16.64 and
   a split-half correlation of 0.017 to 0.073, which would put pancreas outside the range
   the manuscript quotes across the four systems. The recipe below reproduces the shipped
   values exactly. Three of its arguments are load-bearing rather than one: pinning only
   ``bimodal=False`` still gives 13.85 and 0.059 on pancreas, because ``refine`` moves it
   too, and the Hill ceiling moves mouse hematopoiesis even though it leaves pancreas
   untouched. Checking a single dataset is how a parameter gets wrongly called inert.
2. Three of the four objects arrive carrying a ``sigmoid`` layer of their own, fit when a
   different recipe was the default. Reading it would make this artifact depend on which
   era each object on disk came from, which is invisible in the output and would drift
   again the next time those objects are re-prepared. Fitting unconditionally removes
   that dependence, and it costs nothing: refitting all four under the pinned recipe
   reproduces every value of the shipped JSON, both statistics at all four fractions, on
   all four datasets, and selects the same clusters.

## What a reader needs

The four objects, which is where this script is honest about its limits. Only pancreas
rebuilds from a package call, through ``prep_pancreas.py``. Mouse hematopoiesis rebuilds
in every field this sweep reads, since it never touches the pseudotime column that does
not rebuild. The neural crest and limb objects were preprocessed outside this project and
cannot be rebuilt from it. ``DATA_SOURCES.md`` carries the per-dataset record.

A dataset that is not present is skipped with a message rather than faked. Extended Data
Fig. 3 panel g draws one line per dataset and needs all four, so a partial JSON produces a
partial panel; the summary at the end says plainly which datasets were written.

Run:

    python reproducibility/identifiability_multi.py           # a few minutes, CPU
    python reproducibility/identifiability_multi.py --out other.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scanpy as sc
from scipy import sparse

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
from config import DATASETS as CONF                              # noqa: E402

import scHopfield as sch                                         # noqa: E402

# ----------------------------------------------------------------------------- #
# The recipe. Changing any of these produces different numbers from the ones
# Extended Data Fig. 3 panel g is drawn from.
# ----------------------------------------------------------------------------- #
N_GENES = 100          # genes kept, ranked by mean absolute velocity
N_TOTAL = 240          # cells per replicate, held fixed so only composition varies
GAMMA = 0.1            # one fixed degradation rate for every gene and dataset
FRACS = [0.0, 0.1, 0.2, 0.4]      # neighboring-cell fraction
SEEDS = [0, 1, 2]                 # replicates per cluster and fraction
MIN_CLUSTER_FRACTION = 0.7        # a cluster is usable at 0.7 * N_TOTAL cells
MAX_CLUSTERS = 4                  # largest four usable clusters per dataset
N_NEIGHBORS = 30                  # only used when an object carries no neighbor graph

#: Stated in full, including the arguments that agree with the library defaults today.
#: See the module docstring: this call is why the artifact moved once already.
#:
#: ``n_max`` is load-bearing and 20.0 is the value that reproduces the committed JSON.
#: The Methods text names a ceiling of 8, which does not: it changes nothing on pancreas,
#: where no gene reaches the ceiling, but it binds on mouse hematopoiesis and moves that
#: arm from an effective rank of 5.97 to 5.81 at zero neighbors and 7.10 to 6.92 at 0.4.
#: Checking one dataset is not enough to call this parameter inert.
SIGMOID_FIT = dict(spliced_key="Ms", min_th=0.05, n_min=1.001, n_max=20.0,
                   refine=False, bimodal=False)

# Dataset name in the output JSON -> the config.py entry whose path it reads.
# Panel g labels these by biological system, so the four are one per system.
SOURCE = {"hematopoiesis": "paul15", "pancreas": "pancreas",
          "murine_NC": "murine_nc", "human_limb": "human_limb"}

# The cluster column, which is NOT taken from config.py. Three of the four agree with the
# entry's cluster_key, but human_limb deliberately does not: config.py fits it on
# "leiden_R_celltype" while this sweep partitions on the "leiden" clusters themselves.
# Reading the entry's key here would silently change the human_limb arm.
CLUSTER_KEY = {"hematopoiesis": "paul15_clusters", "pancreas": "clusters",
               "murine_NC": "celltype_update", "human_limb": "leiden"}

OUT = os.path.join(paths.IDENTIFIABILITY, "multi.json")


def eff_rank(M):
    """Participation ratio over the singular values, in units of genes."""
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-9]
    return float((s.sum() ** 2) / (np.sum(s ** 2) + 1e-12))


def fit_W(sig, v, x):
    """Unconstrained least-squares fit of W from (phi(x), v) at fixed gamma."""
    A = np.hstack([sig, np.ones((sig.shape[0], 1))])
    return np.linalg.lstsq(A, v + GAMMA * x, rcond=None)[0][:-1].T


def prep(path):
    """Load an object and give it an activation matrix under the pinned recipe."""
    a = sc.read_h5ad(path)
    # Unconditional: any sigmoid layer already on the object was fit under whatever
    # recipe was current when the object was prepared, which is not necessarily this one.
    a.var["scHopfield_used"] = True
    sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values, **SIGMOID_FIT)
    sch.pp.compute_sigmoid(a)
    if "connectivities" not in a.obsp:
        sc.pp.neighbors(a, n_neighbors=N_NEIGHBORS)
    return a


def run_dataset(path, ckey):
    a = prep(path)
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
    clusters = [c for c in vc.index.astype(str)
                if vc[c] >= int(N_TOTAL * MIN_CLUSTER_FRACTION)][:MAX_CLUSTERS]

    res = {f: {"er": [], "sh": []} for f in FRACS}
    for cl in clusters:
        cidx = np.where(labels == cl)[0]
        # The neighbor pool is cells outside the cluster that the graph connects to it,
        # so "broadening" stays local rather than sampling the whole dataset.
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
                sel = np.concatenate([
                    rng.choice(cidx, n_cl, replace=False),
                    rng.choice(nb_pool, n_nb, replace=False) if n_nb else np.array([], int),
                ]).astype(int)
                S, X, V = sig[sel], x[sel], v[sel]
                res[f]["er"].append(eff_rank(S))
                # Split half: fit the same sample twice on disjoint halves, correlate the
                # two off-diagonals. Identical halves would give 1.0.
                perm = rng.permutation(len(sel))
                h = len(sel) // 2
                W1 = fit_W(S[perm[:h]], V[perm[:h]], X[perm[:h]])
                W2 = fit_W(S[perm[h:2 * h]], V[perm[h:2 * h]], X[perm[h:2 * h]])
                off = ~np.eye(W1.shape[0], dtype=bool)
                res[f]["sh"].append(float(np.corrcoef(W1[off], W2[off])[0, 1]))

    summary = {str(f): {"eff_rank": round(float(np.mean(res[f]["er"])), 2),
                        "splithalf_W": round(float(np.mean(res[f]["sh"])), 3),
                        "n": len(res[f]["er"])} for f in FRACS}
    return summary, clusters


def main(out: str = OUT) -> None:
    result, missing = {}, []
    for name, entry in SOURCE.items():
        path = CONF[entry]["path"]
        if not os.path.exists(path):
            print(f"skip {name}: {path} not present", flush=True)
            missing.append(name)
            continue
        summary, clusters = run_dataset(path, CLUSTER_KEY[name])
        result[name] = {"clusters": list(clusters), "by_frac": summary}
        print(f"{name:14s}: eff_rank {summary['0.0']['eff_rank']}->{summary['0.4']['eff_rank']} "
              f"(neighbors 0->0.4), split-half W ~{summary['0.0']['splithalf_W']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out}: {len(result)} of {len(SOURCE)} datasets", flush=True)
    if missing:
        # Said plainly rather than left to be discovered when the panel comes out short.
        print(f"Extended Data Fig. 3 panel g draws one line per dataset and will be "
              f"missing {', '.join(missing)}. See DATA_SOURCES.md for where each object "
              f"comes from.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT,
                    help="the JSON Extended Data Fig. 3 panel g reads")
    main(out=ap.parse_args().out)
