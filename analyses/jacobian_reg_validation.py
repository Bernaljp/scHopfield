"""Validate the Jacobian-consistency regularizer on the biophysical circuits.

Root cause of poor effective-GRN recovery on these non-Hopfield systems is
identifiability (M10): trajectory-confined data underdetermines W. The regularizer
pulls the model's local sensitivity W*sigma'(x) toward a finite-difference velocity
Jacobian estimated from each cell's neighbors, injecting the missing identifying
information. Before/after over a lambda sweep, scored against the true effective GRN
(sign of the average off-diagonal Jacobian).
"""
import json
import os

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from scHopfield.validation.circuits import Novak1997CellCycle, Adlung2021JakStat
from scHopfield._utils.math import fit_sigmoid, sigmoid
from scHopfield.inference.optimizer import ScaffoldOptimizer


class Shim:
    """Yield ((s,x), v) batches from a plain (s,x,v) DataLoader."""
    def __init__(self, dl): self.dl = dl
    def __iter__(self):
        for s, x, v in self.dl:
            yield (s, x), v
    def __len__(self): return len(self.dl)


def simulate_multi(circ, n_ic, pts, noise, seed):
    rng = np.random.default_rng(seed)
    t0, Y0 = circ.simulate(t_end=200.0, n_samples=800)
    lo, hi = Y0.min(0), Y0.max(0)
    N = Y0.shape[1]
    X, V = [], []
    for _ in range(n_ic):
        x0 = rng.uniform(lo, hi + 1e-3)
        sol = solve_ivp(lambda t, x: circ.rhs(x), (0, 120), x0,
                        t_eval=np.linspace(5, 120, pts), method="LSODA", rtol=1e-7, atol=1e-9)
        if not sol.success:
            continue
        xs = sol.y.T
        X.append(xs); V.append(np.stack([circ.rhs(x) for x in xs]))
    X = np.vstack(X); V = np.vstack(V)
    Xobs = X + rng.normal(0, noise * (hi - lo + 1e-3), X.shape)  # measurement noise
    return Xobs.astype(np.float64), V.astype(np.float64), lo, hi


def true_avg_jac(circ, X, h=1e-5):
    N = X.shape[1]; J = np.zeros((N, N))
    for x in X[::40]:
        f0 = circ.rhs(x)
        for j in range(N):
            xp = x.copy(); xp[j] += h; J[:, j] += (circ.rhs(xp) - f0) / h
    return J / len(X[::40])


def neighbor_jacobian(X, V, k, alpha=1e-2):
    """Per-cell finite-difference velocity Jacobian from k nearest neighbors (ridge)."""
    from sklearn.neighbors import NearestNeighbors
    n, N = X.shape
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(X)
    _, idx = nn.kneighbors(X)
    J = np.zeros((n, N, N))
    for i in range(n):
        nb = idx[i, 1:]
        dX = X[nb] - X[i]; dV = V[nb] - V[i]
        # solve dV ~ J dX : J = dV^T dX (dX^T dX + alpha I)^-1
        G = dX.T @ dX + alpha * np.eye(N)
        J[i] = dV.T @ dX @ np.linalg.inv(G)
    return J


def fit_W(X, V, sig, expo, gamma, jac_data, lam, seed, device, epochs=1200):
    torch.manual_seed(seed); np.random.seed(seed)
    N = X.shape[1]
    scaffold = np.ones((N, N), dtype=np.float32)
    opt = ScaffoldOptimizer(gamma.astype(np.float32), scaffold, torch.device(device),
                            refit_gamma=False, scaffold_regularization=0.0,
                            reconstruction_regularization=1.0, bias_regularization=1e-3,
                            normalize_regularization=True)
    if lam > 0:
        opt.configure_jacobian_consistency(X, sig, expo, jac_data, n_sub=512)
    ds = TensorDataset(torch.tensor(sig, dtype=torch.float32),
                       torch.tensor(X, dtype=torch.float32),
                       torch.tensor(V, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)
    opt.to(torch.device(device))
    opt.train_model(Shim(dl), epochs=epochs, learning_rate=5e-2, criterion="MSE",
                    verbose=False, jacobian_lambda=lam)
    return opt.W.weight.detach().cpu().numpy().astype(np.float64)


def score(W, Jt):
    N = W.shape[0]; off = ~np.eye(N, dtype=bool)
    thr = np.percentile(np.abs(Jt[off]), 50); mask = off & (np.abs(Jt) > thr)
    sa = float((np.sign(W[mask]) == np.sign(Jt[mask])).mean())
    y = (np.abs(Jt[off]) > thr).astype(int); s = np.abs(W[off])
    au = float(roc_auc_score(y, s)) if y.min() != y.max() else float("nan")
    return sa, au


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lams = [0.0, 0.1, 1.0, 10.0, 100.0]
    out = {}
    for name, C in [("Novak_cellcycle", Novak1997CellCycle), ("Adlung_jakstat", Adlung2021JakStat)]:
        circ = C()
        X, V, lo, hi = simulate_multi(circ, n_ic=8, pts=140, noise=0.03, seed=0)
        N = X.shape[1]
        Jt = true_avg_jac(circ, X)
        thr = np.array([fit_sigmoid(X[:, j])[0] for j in range(N)])
        expo = np.array([fit_sigmoid(X[:, j])[1] for j in range(N)])
        sig = np.nan_to_num(sigmoid(X, thr[None, :], expo[None, :]))
        gamma = np.full(N, 0.1)
        Jdat = neighbor_jacobian(X, V, k=max(2 * N, 30))
        rows = []
        for lam in lams:
            W = fit_W(X, V, sig, expo, gamma, Jdat, lam, seed=0, device=dev)
            sa, au = score(W, Jt)
            rows.append({"lambda": lam, "sign_acc": round(sa, 3), "auroc": round(au, 3)})
            print(f"{name} lambda={lam:6.1f}: sign_acc={sa:.3f} AUROC={au:.3f}", flush=True)
        out[name] = rows
    os.makedirs("benchmark_results/jacobian_reg", exist_ok=True)
    json.dump(out, open("benchmark_results/jacobian_reg/validation.json", "w"), indent=2)
    print("wrote benchmark_results/jacobian_reg/validation.json", flush=True)


if __name__ == "__main__":
    main()
