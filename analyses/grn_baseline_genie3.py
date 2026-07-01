"""External GRN-inference baseline: scHopfield vs GENIE3 on synthetic ground truth.

Fills the "benchmarked against X and Y" placeholder (Figure 2). We generate random
sparse signed Hopfield networks with known ground-truth W, simulate expression and
velocity, and compare edge recovery (off-diagonal, |edge| as score) for:

  scHopfield : unconstrained Hill fit (Methods 3.1 pseudoinverse with Hill sigma),
               uses expression AND RNA velocity.
  GENIE3     : tree-ensemble regulatory importance (arboreto), uses expression only.

Metric: AUROC and AUPRC for detecting nonzero ground-truth edges. Reported as mean
+/- s.d. over several random networks. GENIE3 is a widely used, velocity-agnostic
baseline; the comparison isolates the value of the velocity-based dynamical model.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")


def make_network(n_genes, density, seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (n_genes, n_genes))
    mask = rng.random((n_genes, n_genes)) < density
    np.fill_diagonal(mask, False)
    W = W * mask
    gamma = rng.uniform(0.5, 1.5, n_genes)
    I = rng.uniform(0.0, 1.0, n_genes)
    k, n = 1.0, 4.0
    return W, I, gamma, k, n


def simulate(W, I, gamma, k, n, n_traj, pts, seed):
    rng = np.random.default_rng(seed + 1)
    N = W.shape[0]

    def sigma(x):
        xp = np.maximum(x, 0.0) ** n
        return xp / (k ** n + xp)

    def rhs(t, x):
        return W @ sigma(x) - gamma * x + I

    X, V = [], []
    for _ in range(n_traj):
        x0 = rng.uniform(0, 3, N)
        sol = solve_ivp(rhs, (0, 25), x0, t_eval=np.linspace(2, 25, pts), method="LSODA",
                        rtol=1e-7, atol=1e-9)
        if not sol.success:
            continue
        xs = sol.y.T
        X.append(xs)
        V.append(np.stack([rhs(0, xx) for xx in xs]))
    X = np.vstack(X)
    V = np.vstack(V)
    return X.astype(np.float64), V.astype(np.float64), sigma


def schopfield_recover(X, V, gamma, sigma):
    """Unconstrained Hill fit: solve v + gamma x = W sigma(x) + I by least squares."""
    sig = sigma(X)
    A = np.hstack([sig, np.ones((sig.shape[0], 1))])
    WI = np.linalg.lstsq(A, V + gamma[None, :] * X, rcond=None)[0]
    return WI[:-1].T  # W_hat


def genie3_recover(X, gene_names):
    """Canonical GENIE3 (Huynh-Thu et al. 2010): for each target gene, a tree
    ensemble regresses it on all other genes; feature importances are edge weights.
    Implemented directly with sklearn ExtraTrees to avoid arboreto's dask stack."""
    from sklearn.ensemble import ExtraTreesRegressor

    n = X.shape[1]
    # standardize expression (GENIE3 convention)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    M = np.zeros((n, n))  # M[target, regulator]
    for j in range(n):
        others = [i for i in range(n) if i != j]
        et = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=0, n_jobs=-1)
        et.fit(Xs[:, others], Xs[:, j])
        imp = et.feature_importances_
        for k, i in enumerate(others):
            M[j, i] = imp[k]
    return M


def edge_scores(W_score, W_true):
    n = W_true.shape[0]
    off = ~np.eye(n, dtype=bool)
    y = (np.abs(W_true[off]) > 1e-9).astype(int)
    s = np.abs(W_score[off])
    if y.min() == y.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def main():
    n_genes, density, n_nets = 40, 0.08, 4
    rows = []
    for net_seed in range(n_nets):
        W, I, gamma, k, n = make_network(n_genes, density, net_seed)
        X, V, sigma = simulate(W, I, gamma, k, n, n_traj=60, pts=25, seed=net_seed)
        gene_names = [f"g{i}" for i in range(n_genes)]

        Wh = schopfield_recover(X, V, gamma, sigma)
        sch_auroc, sch_auprc = edge_scores(Wh, W)

        try:
            Mg = genie3_recover(X, gene_names)
            g_auroc, g_auprc = edge_scores(Mg, W)
        except Exception as e:
            print(f"  net {net_seed}: GENIE3 failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            g_auroc, g_auprc = float("nan"), float("nan")

        rows.append({"net": net_seed, "sch_auroc": sch_auroc, "sch_auprc": sch_auprc,
                     "genie3_auroc": g_auroc, "genie3_auprc": g_auprc,
                     "n_edges": int((np.abs(W) > 1e-9).sum())})
        print(f"net {net_seed}: scHopfield AUROC={sch_auroc:.3f} AUPRC={sch_auprc:.3f} | "
              f"GENIE3 AUROC={g_auroc:.3f} AUPRC={g_auprc:.3f}", flush=True)

    df = pd.DataFrame(rows)
    summary = {"n_genes": n_genes, "density": density, "n_networks": n_nets,
               "schopfield_auroc_mean": float(df["sch_auroc"].mean()),
               "schopfield_auroc_sd": float(df["sch_auroc"].std()),
               "genie3_auroc_mean": float(df["genie3_auroc"].mean()),
               "genie3_auroc_sd": float(df["genie3_auroc"].std()),
               "schopfield_auprc_mean": float(df["sch_auprc"].mean()),
               "genie3_auprc_mean": float(df["genie3_auprc"].mean()),
               "rows": rows}
    os.makedirs("benchmark_results/grn_baseline", exist_ok=True)
    json.dump(summary, open("benchmark_results/grn_baseline/genie3_vs_schopfield.json", "w"), indent=2)
    print(f"\n=== SUMMARY ({n_nets} nets, {n_genes} genes) ===", flush=True)
    print(f"scHopfield AUROC {summary['schopfield_auroc_mean']:.3f} +/- {summary['schopfield_auroc_sd']:.3f}, "
          f"AUPRC {summary['schopfield_auprc_mean']:.3f}", flush=True)
    print(f"GENIE3     AUROC {summary['genie3_auroc_mean']:.3f} +/- {summary['genie3_auroc_sd']:.3f}, "
          f"AUPRC {summary['genie3_auprc_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
