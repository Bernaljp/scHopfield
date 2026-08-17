"""Report-only perturbation measures, kept beside the package rather than in it.

The reported projection-free readouts are part of the scHopfield API and are called from there:
``sch.tl.jacobian_response``, ``sch.tl.jacobian_commitment_push``,
``sch.tl.regulatory_out_strength`` and ``sch.tl.commitment_time``.

What remains in this file is the material the per-dataset report renders but the paper does not
report, so it has not earned a place in the public API: two embedding projections of a knockout,
the energetic consequence of one, and the change in attractor stability under one. The last two
were measured and found insensitive to a single-gene knockout, and are kept for the record rather
than because a figure reads them.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

# The reproducibility tree is flat one level up; this directory holds the compute
# helpers. Both are anchored to this file rather than to the working directory, so a
# script runs the same from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
import scHopfield as sch                                         # noqa: E402


def velocity_projections(adata, ck, genes, basis, dev="cuda", spliced_key="Ms"):
    """Two embedding projections of each knockout, for the report: the INSTANTANEOUS model velocity
    after the KO (the field dx/dt = W sigma(x) - gamma x + I evaluated with the gene held at 0, NO
    propagation), and the PROPAGATED displacement after integrating the ODE with the gene held at 0.
    Both projected to the embedding with the same correlation kernel used for the WT velocity. Shows
    how the immediate (one-step) effect differs from the settled (many-step) effect.
    Returns {gene: (v_instant (n,2), dx_propagated (n,2))}."""
    g_used = sch.tl.get_genes_used(adata)
    proj = sch.tl.build_correlation_projector(adata, basis=basis)
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    out = {}
    for g in genes:
        _, Vko, _ = sch.tl.model_velocity(adata, ck, g_used, ko_gene=g,
                                          spliced_key=spliced_key)
        v_inst = np.asarray(proj(Vko))[:, :2]
        pert = sch.dyn.simulate_shift_ode(adata, {g: 0.0}, cluster_key=ck, n_steps=100,
                                          method="euler", device=dev)
        sch.tl.calculate_flow(pert, source="delta", basis=basis, method="correlation",
                              cluster_key=ck, store_key=f"perturbation_flow_{basis}", verbose=False)
        v_prop = np.asarray(pert.obsm[f"perturbation_flow_{basis}"])[:, :2]
        out[g] = (v_inst, v_prop)
    return out


def energy_reshaping(adata, ck, genes, spliced_key="Ms", dev="cuda", n_steps=100):
    """Energetic consequence of each knockout: the change in the fitted Hopfield energy (Lyapunov
    function E = -0.5 s'Ws + gamma*int(sigma^-1) - I's) between the KNOCKOUT STEADY STATE (the state
    the cells settle into when the ODE is integrated with the gene held at 0) and the wild-type steady
    state, averaged per cluster. Negative dE = the knockout drives that cell type to LOWER-energy (more
    stable) states, positive = higher-energy (less stable). This captures where cells settle
    energetically, unlike silencing a single coordinate (which barely moves the total energy).
    Returns {gene: pd.Series(mean dE over clusters)}."""
    ecols = ("energy_total", "energy_interaction", "energy_degradation", "energy_bias")
    clusters = adata.obs[ck].astype(str).values
    X0 = np.asarray(adata.layers[spliced_key]).astype(float)
    scratch = adata.copy()                                     # one reused copy (avoid repeated large copies)

    def energy_of(dX):                                         # energy of the settled state x0 + delta_X
        scratch.layers[spliced_key] = X0 + np.asarray(dX)
        for c in ecols:
            if c in scratch.obs:
                del scratch.obs[c]
        sch.tl.compute_energies(scratch, spliced_key=spliced_key, cluster_key=ck)
        return np.asarray(scratch.obs["energy_total"].values, float)

    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    wt = sch.dyn.simulate_shift_ode(adata, {}, cluster_key=ck, n_steps=n_steps, method="euler", device=dev)
    E_wt = energy_of(wt.layers["delta_X"])
    out = {}
    for g in genes:
        ko = sch.dyn.simulate_shift_ode(adata, {g: 0.0}, cluster_key=ck, n_steps=n_steps,
                                        method="euler", device=dev)
        out[g] = pd.Series(energy_of(ko.layers["delta_X"]) - E_wt,
                           index=adata.obs_names).groupby(clusters).mean()
    return out


def attractor_stability(adata, ck, lps, genes, spliced_key="Ms", frac=0.3, eps=1e-3):
    """Change in the stability of each fate attractor under each knockout: the largest real part of the
    Jacobian eigenvalues (J = W diag(sigma'(x)) - diag(gamma)) at the field's stable cells of each
    terminal cell type, WT vs the knockout (gene's row and column removed). Positive Delta = the
    knockout pushes an eigenvalue toward zero, i.e. DESTABILIZES that fate (a reprogramming mechanism).
    Returns {gene: pd.Series(Delta max-Re-eigenvalue over terminal cell types)}."""
    sc = sch.tl.fate_scaffold(adata, ck, lps, frac=frac)
    names = sc["names"]; X = sc["X"]; genes = [g for g in dict.fromkeys(genes) if g in names]
    out = {g: {} for g in genes}
    for cl, idx in sc["term"].items():
        try:
            solver = sch.dyn.create_solver(adata, cl, spliced_key=spliced_key)
        except Exception:
            continue
        xbar = np.maximum(X[idx].mean(0), 0.0)                 # expression is non-negative
        spd = (sch.sigmoid(np.maximum(xbar + eps, 0.0), solver.threshold, solver.exponent)
               - sch.sigmoid(np.maximum(xbar - eps, 0.0), solver.threshold,
                             solver.exponent)) / (2 * eps)
        J = np.asarray(solver.W) * spd[None, :] - np.diag(np.asarray(solver.gamma).ravel())
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        maxre_wt = float(np.real(np.linalg.eigvals(J)).max())
        for g in genes:
            keep = np.ones(len(names), bool); keep[names.index(g)] = False
            Jk = J[np.ix_(keep, keep)]
            out[g][cl] = float(np.real(np.linalg.eigvals(Jk)).max()) - maxre_wt
    return {g: pd.Series(out[g]) for g in genes}


if __name__ == "__main__":                                    # quick sanity check on one dataset
    import argparse
    import anndata as ad
    from config import DATASETS
    from _perturb_dynamics_compute import TFS_BY_DATASET
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="pancreas")
    a_ = ap.parse_args()
    from sections import _lineage_pairs
    ds = a_.dataset; cfg = DATASETS[ds]; ck = cfg["cluster_key"]
    adata = ad.read_h5ad(f"{paths.REPORTS}/{ds}/data/adata_analyzed.h5ad")
    lps = [(list(A), list(B), An, Bn) for A, B, An, Bn in _lineage_pairs(adata, ds, ck, cfg)]
    genes = [g for g in TFS_BY_DATASET.get(ds, []) if g in adata.var_names]
    reg = sch.tl.regulatory_out_strength(adata, ck)
    print("=== 1. Jacobian response (regulator targets only) ===")
    resp = sch.tl.jacobian_response(adata, ck, genes)
    for g in genes:
        s = resp[g][reg.reindex(resp[g].index).fillna(0) > 0]
        up = ", ".join(f"{k}(+{v:.2f})" for k, v in s.nlargest(4).items())
        dn = ", ".join(f"{k}({v:.2f})" for k, v in s.nsmallest(4).items())
        print(f"  {g} KO  UP: {up}   DOWN: {dn}")
    print("\n=== 2. Commitment (first-passage) time change (chain steps; + delays) ===")
    for g, d in sch.tl.commitment_time(adata, ck, lps, genes).items():
        print(f"  {g:9s} {d:+.3f}")
    print("\n=== 3. Energetic consequence (KO steady-state dE per cluster; - = more stable) ===")
    for g, s in energy_reshaping(adata, ck, genes).items():
        print(f"  {g:9s} dE range [{s.min():+.1f}, {s.max():+.1f}]  most-lowered: {s.idxmin()} ({s.min():+.1f})")
