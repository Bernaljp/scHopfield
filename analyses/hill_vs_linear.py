"""Ablation: Hill activation vs a linear model (does the nonlinearity buy anything?).

On the synthetic circuits (toggle switch, repressilator) the true dynamics are
dx/dt = W phi(x) + I - gamma x with a Hill phi, so ground truth is known. We compare:

  Hill  : fit W, I with the circuit's Hill activation (scHopfield's model).
  Linear: fit W, I with phi(x) = x (a linear dynamical system, the natural
          no-nonlinearity baseline), by least squares on the same (x, v).

Metrics: velocity reconstruction R^2 (train), and the number of stable fixed
points the fitted system supports. A linear autonomous system dx/dt = A x + b has
at most ONE fixed point, so it structurally cannot represent the toggle switch's
bistability; the Hill model can. This is the clean argument for the nonlinearity.
"""
import json
import os

import numpy as np
from scipy.optimize import fsolve

from scHopfield.validation.circuits import ToggleCircuit, OscillatorCircuit
from scHopfield.validation.simulate import simulate_circuit


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean(0)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def count_stable_fixed_points(f, jac, n_genes, n_starts=400, xmax=4.0, seed=0):
    """Count distinct stable fixed points of dx/dt=f(x) by multi-start root finding."""
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(n_starts):
        x0 = rng.uniform(0, xmax, n_genes)
        try:
            root, info, ier, _ = fsolve(f, x0, full_output=True)
        except Exception:
            continue
        if ier != 1 or np.any(root < -1e-3):
            continue
        # stable if all Re(eig(J)) < 0
        ev = np.linalg.eigvals(jac(root))
        if np.all(ev.real < -1e-6):
            if not any(np.allclose(root, r, atol=1e-2) for r in found):
                found.append(root)
    return len(found)


def fit_hill(adata):
    """Recover W, I via lstsq using the circuit's Hill sigma (as fit_circuit does)."""
    x = adata.layers["Ms"]
    v = adata.layers["velocity_S"]
    g = adata.var["gamma"].values
    k = adata.uns["ground_truth"].get("k", 1.0)
    n = adata.uns["ground_truth"].get("n", 4)
    sig = (np.maximum(x, 0.0) ** n) / (k ** n + np.maximum(x, 0.0) ** n)
    A = np.hstack([sig, np.ones((sig.shape[0], 1))])
    WI = np.linalg.lstsq(A, v + g[None, :] * x, rcond=None)[0]
    W, I = WI[:-1].T, WI[-1]
    vhat = sig @ W.T + I - g[None, :] * x
    return W, I, g, k, n, r2(v, vhat)


def fit_linear(adata):
    """Fit a linear model dx/dt = W x + I - gamma x by lstsq (phi = identity)."""
    x = adata.layers["Ms"]
    v = adata.layers["velocity_S"]
    g = adata.var["gamma"].values
    A = np.hstack([x, np.ones((x.shape[0], 1))])
    WI = np.linalg.lstsq(A, v + g[None, :] * x, rcond=None)[0]
    W, I = WI[:-1].T, WI[-1]
    vhat = x @ W.T + I - g[None, :] * x
    return W, I, g, r2(v, vhat)


def main():
    out = {}
    for name, circ in [("toggle_bistable", ToggleCircuit(b=4.0)), ("repressilator", OscillatorCircuit())]:
        adata = simulate_circuit(circ, n_trajectories=60, points_per_trajectory=40, noise_sigma=0.0, seed=0)
        n_genes = adata.n_vars

        Wh, Ih, g, k, nh, r2_h = fit_hill(adata)
        f_h = lambda x: Wh @ ((np.maximum(x, 0.0) ** nh) / (k ** nh + np.maximum(x, 0.0) ** nh)) + Ih - g * x
        def jac_h(x):
            xs = np.maximum(x, 1e-9)
            sp = nh * xs ** (nh - 1) * k ** nh / (k ** nh + xs ** nh) ** 2
            return Wh * sp[None, :] - np.diag(g)
        fp_h = count_stable_fixed_points(f_h, jac_h, n_genes)

        Wl, Il, g, r2_l = fit_linear(adata)
        f_l = lambda x: Wl @ x + Il - g * x
        jac_l = lambda x: Wl - np.diag(g)
        fp_l = count_stable_fixed_points(f_l, jac_l, n_genes)

        true_fp = count_stable_fixed_points(lambda x: circ.rhs(x), circ.jacobian, n_genes)
        out[name] = {"hill_recon_r2": round(r2_h, 4), "linear_recon_r2": round(r2_l, 4),
                     "hill_stable_fixedpoints": fp_h, "linear_stable_fixedpoints": fp_l,
                     "true_stable_fixedpoints": true_fp}
        print(f"{name}: Hill R2={r2_h:.4f} (fp={fp_h})  Linear R2={r2_l:.4f} (fp={fp_l})  "
              f"true fp={true_fp}", flush=True)

    os.makedirs("benchmark_results/ablations", exist_ok=True)
    json.dump(out, open("benchmark_results/ablations/hill_vs_linear.json", "w"), indent=2)
    print("wrote benchmark_results/ablations/hill_vs_linear.json", flush=True)


if __name__ == "__main__":
    main()
