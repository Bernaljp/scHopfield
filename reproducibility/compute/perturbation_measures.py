"""Projection-free, model-intrinsic perturbation measures for the per-dataset reports (section E).

These complement the fate-probability lineage effect with readouts that do NOT depend on the
two-dimensional embedding projection (the weakness of the CellOracle-style flow-alignment score and
the old projected lineage bias), and that lean on scHopfield's own structure:

  - ``jacobian_response`` : the predicted gene-level dysregulation of a knockout, from the fitted
    Jacobian (Dynamo / scTenifoldKnk style linear response, r_i = -J_ig x_g). Answers WHICH genes a
    knockout changes and in which direction, entirely in gene space -> a testable prediction.
  - ``commitment_time``   : the change in expected time-to-commitment (mean first-passage time to any
    terminal fate) under the knockout. Answers whether the knockout DELAYS or ACCELERATES
    differentiation.

Energy-landscape reshaping and attractor stability are added as further subsections (see the report
wiring). Column ``g`` of the Jacobian is obtained by finite-differencing the solver's own dynamics,
so it is exact for the fitted field without re-deriving the Hill derivative.
"""
from __future__ import annotations
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
from scHopfield.tools.flow import get_genes_used              # noqa: E402
from scHopfield.dynamics.solver import create_solver          # noqa: E402
import _fate_probability as F                                 # noqa: E402


def jacobian_response(adata, ck, genes, spliced_key="Ms", eps=1e-2):
    """First-order knockout response r_i = -J_ig x_g, averaged over cells (each under its own
    cluster-specific field). J_ig = d v_i / d x_g is obtained by central finite differences of the
    solver's dynamics. r_i > 0 means gene i's production rate RISES when the gene is knocked out
    (loss of repression); r_i < 0 means it FALLS (loss of activation). Returns
    {gene: pd.Series(signed mean response over target genes)}."""
    g_used = get_genes_used(adata)
    names = list(np.asarray(adata.var_names)[g_used])
    X = np.asarray(adata.layers[spliced_key])[:, g_used].astype(float)
    clusters = adata.obs[ck].astype(str).values
    genes = [g for g in dict.fromkeys(genes) if g in names]
    acc = {g: np.zeros(len(names)) for g in genes}
    n = 0
    for c in pd.unique(clusters):
        sel = np.where(clusters == c)[0]
        try:
            solver = create_solver(adata, c, spliced_key=spliced_key)
        except Exception:
            continue
        Xc = X[sel]
        for g in genes:
            gi = names.index(g)
            Xp = Xc.copy(); Xp[:, gi] += eps
            Xm = Xc.copy(); Xm[:, gi] -= eps
            jcol = (solver.dynamics_batch(Xp, 0.0) - solver.dynamics_batch(Xm, 0.0)) / (2 * eps)
            acc[g] += (-jcol * Xc[:, gi][:, None]).sum(0)      # r_i = -J_ig x_g, summed over cells
        n += len(sel)
    return {g: pd.Series(acc[g] / max(n, 1), index=names).drop(labels=[g], errors="ignore")
            for g in genes}


def jacobian_commitment_push(adata, ck, lps, groups, spliced_key="Ms", eps=1e-2):
    """Embed the Jacobian knockout response on the UMAP as a per-cell 'commitment push': for each cell,
    project its first-order response r(cell) = -J[:,g](cell) x_g(cell) onto the gene-space fate axis of
    the gene's own lineage decision, d = normalize(centroid_A - centroid_B). push(cell) = r(cell) . d,
    so + = the knockout's immediate molecular response nudges that cell toward lineage A, - toward B.
    This ties the Jacobian response ('which genes change') to a direction/commitment readout that can be
    painted on the embedding, complementing the eventual (propagated) fate map. J column g is the
    central finite difference of the solver dynamics. Returns {gene: (An, Bn, push array over cells)}."""
    g_used = get_genes_used(adata)
    names = list(np.asarray(adata.var_names)[g_used])
    X = np.asarray(adata.layers[spliced_key])[:, g_used].astype(float)
    clusters = adata.obs[ck].astype(str).values
    gnames = list(groups)
    gene_pair = {}; axis = {}
    for k, (A, B, An, Bn) in enumerate(lps):
        if k >= len(gnames):
            break
        Am = np.isin(clusters, [str(c) for c in A]); Bm = np.isin(clusters, [str(c) for c in B])
        if Am.sum() == 0 or Bm.sum() == 0:
            continue
        d = X[Am].mean(0) - X[Bm].mean(0)
        axis[k] = (d / (np.linalg.norm(d) + 1e-12), An, Bn)
        for gene in groups[gnames[k]]:
            gene_pair[gene] = k
    genes = [g for g in gene_pair if g in names and gene_pair[g] in axis]
    push = {g: np.zeros(len(clusters)) for g in genes}
    for c in pd.unique(clusters):
        sel = np.where(clusters == c)[0]
        try:
            solver = create_solver(adata, c, spliced_key=spliced_key)
        except Exception:
            continue
        Xc = X[sel]
        for gene in genes:
            gi = names.index(gene)
            Xp = Xc.copy(); Xp[:, gi] += eps; Xm = Xc.copy(); Xm[:, gi] -= eps
            jcol = (solver.dynamics_batch(Xp, 0.0) - solver.dynamics_batch(Xm, 0.0)) / (2 * eps)
            r = -jcol * Xc[:, gi][:, None]                     # per-cell first-order response
            push[gene][sel] = r @ axis[gene_pair[gene]][0]     # project onto the fate axis
    return {g: (axis[gene_pair[g]][1], axis[gene_pair[g]][2], push[g]) for g in genes}


def velocity_projections(adata, ck, genes, basis, dev="cuda", spliced_key="Ms"):
    """Two embedding projections of each knockout, for the report: the INSTANTANEOUS model velocity
    after the KO (the field dx/dt = W sigma(x) - gamma x + I evaluated with the gene held at 0, NO
    propagation), and the PROPAGATED displacement after integrating the ODE with the gene held at 0.
    Both projected to the embedding with the same correlation kernel used for the WT velocity. Shows
    how the immediate (one-step) effect differs from the settled (many-step) effect.
    Returns {gene: (v_instant (n,2), dx_propagated (n,2))}."""
    import scHopfield as sch
    from scHopfield.tools.embedding import build_correlation_projector
    g_used = get_genes_used(adata)
    proj = build_correlation_projector(adata, basis=basis)
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    out = {}
    for g in genes:
        _, Vko, _ = F.model_velocity(adata, ck, g_used, ko_gene=g, spliced_key=spliced_key)
        v_inst = np.asarray(proj(Vko))[:, :2]
        pert = sch.dyn.simulate_shift_ode(adata, {g: 0.0}, cluster_key=ck, n_steps=100,
                                          method="euler", device=dev)
        sch.tl.calculate_flow(pert, source="delta", basis=basis, method="correlation",
                              cluster_key=ck, store_key=f"perturbation_flow_{basis}", verbose=False)
        v_prop = np.asarray(pert.obsm[f"perturbation_flow_{basis}"])[:, :2]
        out[g] = (v_inst, v_prop)
    return out


def out_strength(adata, ck):
    """Per-gene regulatory out-strength (column sum of |W| over clusters); >0 marks a regulator."""
    genes = np.asarray(adata.var_names.values)
    outs = np.zeros(adata.n_vars)
    for c in adata.obs[ck].astype(str).unique():
        key = f"W_{c}"
        if key in adata.varp:
            outs += np.abs(np.asarray(adata.varp[key])).sum(0)
    return pd.Series(outs, index=genes)


def commitment_time(adata, ck, lps, genes, basis=None, n_neighbors=30, sigma=0.05, frac=0.3):
    """Change in expected time-to-commitment under each knockout: the mean first-passage time of the
    absorbing Markov chain (expected transition steps to reach ANY terminal fate), over the still-
    transient cells, WT vs KO. Positive = the knockout DELAYS commitment, negative = ACCELERATES it.
    Returns {gene: float median delay (chain steps)}."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    sc = F._fate_scaffold(adata, ck, lps, basis=basis, n_neighbors=n_neighbors, sigma=sigma, frac=frac)
    isterm = np.zeros(len(sc["clusters"]), bool)
    isterm[np.concatenate([v for v in sc["term"].values()])] = True
    trans = np.where(~isterm)[0]

    def fpt(T):
        Q = T.tocsr()[trans][:, trans].tocsc()
        return spsolve(sp.eye(len(trans), format="csc") - Q, np.ones(len(trans)))

    t_wt = fpt(sc["T_wt"])
    names = sc["names"]; genes = [g for g in dict.fromkeys(genes) if g in names]
    out = {}
    for g in genes:
        gi = names.index(g)
        _, Vko, _ = F.model_velocity(adata, ck, sc["g"], ko_gene=g)
        Vk = Vko.copy(); Vk[:, gi] = sc["V_wt"][:, gi]
        Tk = F.transition_matrix(sc["X"], Vk, sc["knn"], sc["sigma"])
        out[g] = float(np.nanmedian(fpt(Tk) - t_wt))
    return out


def energy_reshaping(adata, ck, genes, spliced_key="Ms", dev="cuda", n_steps=100):
    """Energetic consequence of each knockout: the change in the fitted Hopfield energy (Lyapunov
    function E = -0.5 s'Ws + gamma*int(sigma^-1) - I's) between the KNOCKOUT STEADY STATE (the state
    the cells settle into when the ODE is integrated with the gene held at 0) and the wild-type steady
    state, averaged per cluster. Negative dE = the knockout drives that cell type to LOWER-energy (more
    stable) states, positive = higher-energy (less stable). This captures where cells settle
    energetically, unlike silencing a single coordinate (which barely moves the total energy).
    Returns {gene: pd.Series(mean dE over clusters)}."""
    import scHopfield as sch
    from scHopfield.tools.energy import compute_energies
    ecols = ("energy_total", "energy_interaction", "energy_degradation", "energy_bias")
    clusters = adata.obs[ck].astype(str).values
    X0 = np.asarray(adata.layers[spliced_key]).astype(float)
    scratch = adata.copy()                                     # one reused copy (avoid repeated large copies)

    def energy_of(dX):                                         # energy of the settled state x0 + delta_X
        scratch.layers[spliced_key] = X0 + np.asarray(dX)
        for c in ecols:
            if c in scratch.obs:
                del scratch.obs[c]
        compute_energies(scratch, spliced_key=spliced_key, cluster_key=ck)
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
    from scHopfield._utils.math import sigmoid
    sc = F._fate_scaffold(adata, ck, lps, frac=frac)
    names = sc["names"]; X = sc["X"]; genes = [g for g in dict.fromkeys(genes) if g in names]
    out = {g: {} for g in genes}
    for cl, idx in sc["term"].items():
        try:
            solver = create_solver(adata, cl, spliced_key=spliced_key)
        except Exception:
            continue
        xbar = np.maximum(X[idx].mean(0), 0.0)                 # expression is non-negative
        spd = (sigmoid(np.maximum(xbar + eps, 0.0), solver.threshold, solver.exponent)
               - sigmoid(np.maximum(xbar - eps, 0.0), solver.threshold, solver.exponent)) / (2 * eps)
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
    reg = out_strength(adata, ck)
    print("=== 1. Jacobian response (regulator targets only) ===")
    resp = jacobian_response(adata, ck, genes)
    for g in genes:
        s = resp[g][reg.reindex(resp[g].index).fillna(0) > 0]
        up = ", ".join(f"{k}(+{v:.2f})" for k, v in s.nlargest(4).items())
        dn = ", ".join(f"{k}({v:.2f})" for k, v in s.nsmallest(4).items())
        print(f"  {g} KO  UP: {up}   DOWN: {dn}")
    print("\n=== 2. Commitment (first-passage) time change (chain steps; + delays) ===")
    for g, d in commitment_time(adata, ck, lps, genes).items():
        print(f"  {g:9s} {d:+.3f}")
    print("\n=== 3. Energetic consequence (KO steady-state dE per cluster; - = more stable) ===")
    for g, s in energy_reshaping(adata, ck, genes).items():
        print(f"  {g:9s} dE range [{s.min():+.1f}, {s.max():+.1f}]  most-lowered: {s.idxmin()} ({s.min():+.1f})")
