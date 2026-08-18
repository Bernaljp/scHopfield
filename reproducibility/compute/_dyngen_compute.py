"""Compute + cache the dyngen fits the composite figure needs (not in the committed JSONs).

Produces reproducibility/data/dyngen/fig_fits.npz with:
  lin_W_none, lin_W_true, lin_W_tf  : linear recovered W under the 3 scaffolds (panel d)
  lin_scaffold                      : the TF scaffold used (tile(tf_mask))  (panel d inset)
  traj_<backbone>                   : 3xN array [sim_time; energy; leading Re eig]  (panels f,g)

Fit recipe matches the benchmark and energy-trajectory scripts of the analysis pipeline,
which are not part of this repository. It is pinned in full below rather than inherited
from the package defaults, so that this script keeps reproducing the cache that ships
beside it after a default is revised. Verified on 2026-08-17 against the committed
artifacts: the four recovered W matrices come back bitwise identical, the three
trajectories to 1.6e-07 absolute (5e-12 relative, the Jacobian eigenvalue accumulation),
and the six committed benchmark JSONs match on all 90 metrics.

The committed fig_fits.npz also carries a traj_trifurcating array, left over from an
earlier run. The figure reads only linear, bifurcating and cycle, so a fresh run writes
a strict subset of the committed keys and every panel still draws.

Run once (GPU): python reproducibility/compute/_dyngen_compute.py
"""
import os
import warnings

# The reproducibility tree is flat one level up; this directory holds the compute
# helpers. Both are anchored to this file rather than to the working directory, so a
# script runs the same from anywhere.
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402

import numpy as np
import anndata as ad

warnings.filterwarnings("ignore")
import scHopfield as sch  # noqa: E402
import torch  # noqa: E402

ROOT = paths.DYNGEN
CK = "cluster"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONES = ["linear", "bifurcating", "cycle"]


# The benchmark recipe, stated in full rather than left to the package defaults.
#
# This script reproduces artifacts computed on 2026-07-07 and 2026-07-14. The package
# defaults have moved since, so a bare call no longer recomputes what ships beside it:
# the canonical defaults are tuned for real single-cell data, and on this synthetic
# benchmark they lower every arm (the unscaffolded pseudoinverse falls close to chance).
# Every value below is therefore pinned to the one in force when the artifacts were
# written, which reproduces the shipped benchmark JSONs metric for metric.
#
# Pinning the whole recipe, rather than only the arguments that happen to have moved so
# far, is the point: a benchmark that reads a default is a benchmark that silently
# changes the next time a default is revised.
SIGMOID_KWARGS = dict(
    n_max=8.0,        # canonical is 20.0; the benchmark was fitted at the ceiling of 8
    bimodal=False,    # canonical is True; the benchmark used a single-component Hill
)
FIT_KWARGS = dict(
    reconstruction_regularization=100.0,
    bias_regularization=1.0,
    bias_penalty="l1",
    refit_gamma=True,
    seed=0,
    # Below here: values that are no longer the package default. Each one changes the
    # fitted W on this data, so none of them may be dropped.
    batch_size=64,                 # canonical 128
    use_plateau_scheduler=False,   # canonical True
    plateau_patience=50,           # canonical 100
    plateau_factor=0.5,            # canonical 0.1
    include_neighbors=False,       # canonical True
    neighbor_fraction=0.0,         # canonical 0.2
    boundedness_lambda=0.0,        # canonical 0.1 (the C1 radial hinge)
    gamma_min=0.0,                 # canonical 0.01
)


def prep(b):
    a = ad.read_h5ad(f"{ROOT}/{b}/adata.h5ad")
    N = a.n_vars
    sch.pp.fit_all_sigmoids(a, genes=np.ones(N, bool), spliced_key="Ms", **SIGMOID_KWARGS)
    sch.pp.compute_sigmoid(a, spliced_key="Ms")
    return a, N


def fit_W(a, w_scaffold, reg=1.0, epochs=300):
    sch.inf.fit_interactions(
        a, cluster_key=CK, spliced_key="Ms", velocity_key="velocity",
        w_scaffold=w_scaffold, scaffold_regularization=reg,
        only_TFs=w_scaffold is not None, n_epochs=epochs, device=DEV,
        **FIT_KWARGS)
    return np.asarray(a.varp[f"W_{a.obs[CK].cat.categories[0]}"])


def trajectory(a):
    sch.tl.compute_energies(a, spliced_key="Ms", cluster_key=CK)
    sch.tl.compute_jacobians(a, spliced_key="Ms", cluster_key=CK, device="cpu")
    t = a.obs["sim_time"].values.astype(float)
    E = a.obs["energy_total"].values.astype(float)
    lead = np.array([np.max(np.real(e)) for e in a.obsm["jacobian_eigenvalues"]], dtype=float)
    return np.vstack([t, E, lead])


def main():
    out = {}
    print(f"device={DEV}", flush=True)

    # ---- linear: recovered W under 3 scaffolds (panel d) ----
    a, N = prep("linear")
    tf = np.load(f"{ROOT}/linear/tf_mask.npy")
    Wt = np.load(f"{ROOT}/linear/W_true.npy")
    tf_scaf = np.tile(tf.astype(float), (N, 1))
    true_scaf = (np.abs(Wt) > 0).astype(float)
    print("[linear] no-scaffold ...", flush=True); out["lin_W_none"] = fit_W(a, None)
    print("[linear] true-scaffold ...", flush=True); out["lin_W_true"] = fit_W(a, true_scaf, reg=1.0)
    print("[linear] tf-scaffold ...", flush=True); out["lin_W_tf"] = fit_W(a, tf_scaf, reg=1.0)
    out["lin_scaffold"] = tf_scaf
    # linear already holds the tf-scaffold fit -> reuse for its trajectory
    print("[linear] trajectory ...", flush=True); out["traj_linear"] = trajectory(a)

    # ---- bifurcating, cycle: tf-scaffold fit + trajectory (panels f, g) ----
    for b in ["bifurcating", "cycle"]:
        a, N = prep(b)
        tf = np.load(f"{ROOT}/{b}/tf_mask.npy")
        print(f"[{b}] tf-scaffold fit ...", flush=True)
        fit_W(a, np.tile(tf.astype(float), (N, 1)), reg=1.0)
        print(f"[{b}] trajectory ...", flush=True); out[f"traj_{b}"] = trajectory(a)

    path = f"{ROOT}/fig_fits.npz"
    np.savez(path, **out)
    print("WROTE", path, "keys:", list(out.keys()), flush=True)


if __name__ == "__main__":
    main()
