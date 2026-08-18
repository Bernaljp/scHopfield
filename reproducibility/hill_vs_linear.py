"""Ablation: what the Hill nonlinearity buys over a linear field.

Writes the JSON that Extended Data Fig. 1 is drawn from:

    <reproducibility>/data/ablations/hill_vs_linear.json

On the synthetic circuits the true dynamics are dx/dt = W phi(x) + I - gamma x with a
Hill phi, so the ground truth is known and the comparison is exact rather than inferred:

  Hill    fit W, I with the circuit's own Hill activation, which is scHopfield's model
  linear  fit W, I with phi(x) = x, the natural no-nonlinearity baseline, by least
          squares on the same (x, v)

Two metrics, which are the two claims the figure carries. Velocity reconstruction R^2
on the training data, and the number of stable fixed points the fitted system supports.
A linear autonomous system dx/dt = A x + b has at most ONE fixed point, so it
structurally cannot represent the toggle switch's bistability, while the Hill model can.
That is the qualitative half of the argument, and it is why the fixed-point count is
reported next to the R^2 rather than instead of it.

This is the one figure input in this repository that needs no dataset at all. The
circuits are simulated here from ``scHopfield.validation``, so a clean clone reproduces
the shipped JSON with nothing fetched and nothing prepared.

The recipe is written out below rather than inherited from the library defaults,
including the parameters that happen to agree with those defaults today. A benchmark
that reads a default is a benchmark that changes silently the next time a default is
revised, which has already happened once in this project to the sigmoid fit that
``identifiability_multi.py`` documents.

Verified, not assumed: run from an unrelated working directory against the public
package, this reproduces the shipped ``hill_vs_linear.json`` exactly, every value in
every field, in about seven seconds on one CPU.

Run:

    python reproducibility/hill_vs_linear.py
    python reproducibility/hill_vs_linear.py --out other.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import fsolve

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402

from scHopfield.validation.circuits import ToggleCircuit, OscillatorCircuit   # noqa: E402
from scHopfield.validation.simulate import simulate_circuit                   # noqa: E402

# ----------------------------------------------------------------------------- #
# The recipe. Every argument is stated, including the ones that agree with the
# library defaults today, so that a later change to a default cannot move the
# published numbers without this file changing too.
#
# b = 4.0 on the toggle is the one value that is not a default: it is the mutual
# repression strength that puts the circuit in its bistable regime, which is the
# whole point of the toggle panel.
# ----------------------------------------------------------------------------- #
CIRCUITS = [
    ("toggle_bistable", ToggleCircuit(a=5.0, b=4.0, k=1.0, n=4, gamma=3.0)),
    ("repressilator", OscillatorCircuit(alpha=10.0, k=1.0, n=4, gamma=1.0)),
]
SIM = dict(n_trajectories=60, points_per_trajectory=40, t_end=30.0,
           transient_fraction=0.1, noise_sigma=0.0, x_max_init=None,
           cluster_label="synthetic", seed=0)

# Multi-start root finding for the fixed-point count. n_starts is large enough that the
# toggle's three fixed points are all found from uniform starts in [0, xmax]^n_genes.
N_STARTS, XMAX, FP_SEED = 400, 4.0, 0
ATOL_DISTINCT = 1e-2        # two roots closer than this are the same fixed point
TOL_NEGATIVE = -1e-3        # a root with a concentration below this is discarded
TOL_STABLE = -1e-6          # stable when every eigenvalue real part is below this

OUT = os.path.join(paths.ABLATIONS, "hill_vs_linear.json")


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean(0)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def count_stable_fixed_points(f, jac, n_genes, n_starts=N_STARTS, xmax=XMAX, seed=FP_SEED):
    """Count distinct stable fixed points of dx/dt = f(x) by multi-start root finding."""
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(n_starts):
        x0 = rng.uniform(0, xmax, n_genes)
        try:
            root, info, ier, _ = fsolve(f, x0, full_output=True)
        except Exception:
            continue
        if ier != 1 or np.any(root < TOL_NEGATIVE):
            continue
        ev = np.linalg.eigvals(jac(root))
        if np.all(ev.real < TOL_STABLE):
            if not any(np.allclose(root, r, atol=ATOL_DISTINCT) for r in found):
                found.append(root)
    return len(found)


def fit_hill(adata):
    """Recover W, I by least squares using the circuit's own Hill sigma."""
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
    """Fit a linear model dx/dt = W x + I - gamma x by least squares (phi = identity)."""
    x = adata.layers["Ms"]
    v = adata.layers["velocity_S"]
    g = adata.var["gamma"].values
    A = np.hstack([x, np.ones((x.shape[0], 1))])
    WI = np.linalg.lstsq(A, v + g[None, :] * x, rcond=None)[0]
    W, I = WI[:-1].T, WI[-1]
    vhat = x @ W.T + I - g[None, :] * x
    return W, I, g, r2(v, vhat)


def main(out: str = OUT) -> None:
    result = {}
    for name, circ in CIRCUITS:
        adata = simulate_circuit(circ, **SIM)
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

        # The truth, counted the same way, so the comparison in panel b is like for like.
        true_fp = count_stable_fixed_points(lambda x: circ.rhs(x), circ.jacobian, n_genes)

        result[name] = {"hill_recon_r2": round(r2_h, 4), "linear_recon_r2": round(r2_l, 4),
                        "hill_stable_fixedpoints": fp_h, "linear_stable_fixedpoints": fp_l,
                        "true_stable_fixedpoints": true_fp}
        print(f"{name}: Hill R2={r2_h:.4f} (fp={fp_h})  Linear R2={r2_l:.4f} (fp={fp_l})  "
              f"true fp={true_fp}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT, help="the JSON Extended Data Fig. 1 reads")
    main(out=ap.parse_args().out)
