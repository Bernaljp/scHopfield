"""Compute + cache the dyngen fits the composite figure needs (not in the committed JSONs).

Produces reproducibility/data/dyngen/fig_fits.npz with:
  lin_W_none, lin_W_true, lin_W_tf  : linear recovered W under the 3 scaffolds (panel d)
  lin_scaffold                      : the TF scaffold used (tile(tf_mask))  (panel d inset)
  traj_<backbone>                   : 3xN array [sim_time; energy; leading Re eig]  (panels f,g)

Fit recipe matches the benchmark and energy-trajectory scripts of the analysis pipeline,
which are not part of this repository.
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


def prep(b):
    a = ad.read_h5ad(f"{ROOT}/{b}/adata.h5ad")
    N = a.n_vars
    sch.pp.fit_all_sigmoids(a, genes=np.ones(N, bool), spliced_key="Ms")
    sch.pp.compute_sigmoid(a, spliced_key="Ms")
    return a, N


def fit_W(a, w_scaffold, reg=1.0, epochs=300):
    sch.inf.fit_interactions(
        a, cluster_key=CK, spliced_key="Ms", velocity_key="velocity",
        w_scaffold=w_scaffold, scaffold_regularization=reg,
        reconstruction_regularization=100.0, bias_regularization=1.0,
        bias_penalty="l1", only_TFs=w_scaffold is not None, n_epochs=epochs,
        refit_gamma=True, device=DEV, seed=0)
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
