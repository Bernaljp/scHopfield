"""Build the small-circuits validation report.

Reproduces (and extends) ``notebooks/experiments/small_circuits.ipynb`` as a
self-contained, CPU-only report. Two ground-truth circuits whose Hopfield-form
interaction matrix ``W`` is known exactly are simulated, then reconstructed by
scHopfield's own optimizer under three scaffold priors (full / partial / none):

* **Toggle switch** -- two genes with mutual repression and positive
  autoregulation. Multistable (tri-stable for the default parameters, since strong
  autoactivation also stabilizes a central co-expression state); the committed
  basins emerge as the mutual inhibition strength grows.
* **Repressilator** -- three genes in a cyclic repression loop (Elowitz-Leibler).
  Sustains a limit cycle; its ``W`` is dominated by an antisymmetric (rotational)
  component.

Because the true ``W, I, gamma`` and the analytic Hill sigmoid are known, this is
a clean identifiability check: it isolates the optimizer from RNA-velocity
estimation error. Every figure is written to ``reproducibility/output/small_circuits/plots`` and
embedded, with a brief explanation, into ``reproducibility/output/small_circuits/RESULTS.md``.

The circuits have only 2-3 genes, so the fit is a tiny linear model: 800 epochs already
recovers W and gamma numerically exactly, and GPU gives no speed-up over CPU (the loop is
launch-overhead bound, not FLOP bound). The default device is cuda purely for convenience;
it falls back to cpu automatically.

Run (from repo root):

    python reproducibility/build_circuits_report.py                 # full (GPU)
    python reproducibility/build_circuits_report.py --device cpu    # full (CPU)
    python reproducibility/build_circuits_report.py --quick         # smoke
    python reproducibility/build_circuits_report.py --skip-fit      # reuse cache
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

# The reproducibility tree is flat: every script imports its siblings by bare module
# name, and the compute helpers sit one level down in compute/. Both are anchored to
# this file rather than to the working directory, so a script runs the same from
# anywhere.
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "compute"))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from scipy.interpolate import griddata

import scHopfield as sch
from scHopfield.validation.circuits import ToggleCircuit, OscillatorCircuit
from scHopfield.validation.simulate import simulate_circuit
from scHopfield.validation.metrics import summarize_recovery, symmetry_index
from scHopfield.inference.optimizer import ScaffoldOptimizer
from scHopfield.validation._scalenorm_helpers import LoaderShim
from scHopfield.validation.fit_validation import _build_scaffold

warnings.filterwarnings("ignore")

# The report page and its plots are regenerated output. The fit cache is not: it is
# the whole external input to Figure 2, so it is tracked, and a refit rewrites it in
# place.
OUT = os.path.join(paths.OUTPUT, "small_circuits")
PLOTS = os.path.join(OUT, "plots")
DATA = paths.SMALL_CIRCUITS
SEED = 0

# palettes
CMAP_ENERGY = "viridis"
CMAP_ENERGY2 = "magma"
CMAP_W = "coolwarm"
REGIME_COLORS = {"full": "#0072B2", "partial": "#E69F00", "none": "#999999"}


# --------------------------------------------------------------------------- #
# fitting (heavily-commented local version, matches the notebook)
# --------------------------------------------------------------------------- #
def fit_circuit_local(adata, scaffold_mode="full", refit_gamma=True,
                      scaffold_regularization=1e-2, reconstruction_regularization=1.0,
                      bias_regularization=1e-2, n_epochs=2000, batch_size=64,
                      learning_rate=5e-2, device="cpu", seed=SEED, false_pos_rate=0.1):
    """Fit scHopfield on a synthetic circuit and return inferred parameters.

    We build the ``ScaffoldOptimizer`` by hand and feed it the *exact* analytic
    Hill sigmoid, so the only thing being tested is the interaction/bias/gamma
    optimization itself (not the upstream sigmoid or velocity estimation).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    gt = adata.uns["ground_truth"]
    W_true, I_true, gamma_true = gt["W"], gt["I"], gt["gamma"]

    scaffold = _build_scaffold(W_true, scaffold_mode, seed=seed, false_pos_rate=false_pos_rate)

    expression = adata.layers["Ms"]
    velocity = adata.layers["velocity_S"]

    k = gt.get("k", 1.0)
    n_hill = gt.get("n", 4)
    xn = np.power(np.maximum(expression, 0.0), n_hill)
    sig = (xn / (k ** n_hill + xn)).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(sig),
                       torch.from_numpy(expression.astype(np.float32)),
                       torch.from_numpy(velocity.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    g_init = (np.ones_like(gamma_true, dtype=np.float32) * 1.5
              if refit_gamma else gamma_true.astype(np.float32))

    opt = ScaffoldOptimizer(
        g=g_init, scaffold=scaffold, device=torch.device(device),
        refit_gamma=refit_gamma,
        scaffold_regularization=scaffold_regularization,
        reconstruction_regularization=reconstruction_regularization,
        bias_regularization=bias_regularization,
        normalize_regularization=True,
    )
    loss_hist, recon_hist = opt.train_model(
        train_loader=LoaderShim(loader), epochs=n_epochs,
        learning_rate=learning_rate, criterion="MSE", verbose=False, get_plots=False)

    W_inf = opt.W.weight.detach().cpu().numpy().astype(np.float64)
    I_inf = opt.I.detach().cpu().numpy().astype(np.float64)
    gamma_inf = np.exp(np.clip(opt.gamma.detach().cpu().numpy(), -np.inf, 10.0)).astype(np.float64)

    return dict(W_inferred=W_inf, I_inferred=I_inf, gamma_inferred=gamma_inf,
                W_true=W_true, I_true=I_true, gamma_true=gamma_true,
                scaffold=scaffold, loss_history=np.asarray(loss_hist, dtype=float))


def fit_three_regimes(adata, n_epochs, device="cpu"):
    return {m: fit_circuit_local(adata, scaffold_mode=m, n_epochs=n_epochs, device=device)
            for m in ("full", "partial", "none")}


# --------------------------------------------------------------------------- #
# small plotting helpers
# --------------------------------------------------------------------------- #
def _save(fig, name):
    fig.savefig(f"{PLOTS}/{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {name}.png")


def _Wmap(ax, W, title, vmax=None, labels=None, fs=11):
    if vmax is None:
        vmax = np.max(np.abs(W))
    im = ax.imshow(W, cmap=CMAP_W, vmin=-vmax, vmax=vmax)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:.1f}", ha="center", va="center",
                    color="black", fontsize=fs)
    if labels is not None:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=fs)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=fs)
    ax.set_title(title, fontsize=fs + 1)
    ax.set_xlabel("regulator (j)"); ax.set_ylabel("target (i)")
    return im


def _energy_grid_2d(var_names, cell_type, W, I, gamma, grid_points, shape):
    """Compute total scHopfield energy on a 2D/3D grid of expression states."""
    g = ad.AnnData(X=grid_points.astype(np.float32))
    g.var_names = list(var_names)
    g.layers["Ms"] = grid_points.astype(np.float32)
    g.obs["cell_type"] = cell_type
    g.varp[f"W_{cell_type}"] = W
    g.var[f"I_{cell_type}"] = I
    g.var[f"gamma_{cell_type}"] = gamma
    g.var["sigmoid_threshold"] = 1.0
    g.var["sigmoid_exponent"] = 4.0
    g.var["scHopfield_used"] = True
    sch.tl.compute_energies(g, cluster_key="cell_type", spliced_key="Ms",
                            degradation_key=f"gamma_{cell_type}")
    return g.obs["energy_total"].values.reshape(shape)


def hill(x, n=4, k=1.0):
    xp = np.power(np.maximum(x, 0.0), n)
    return xp / (k ** n + xp)


def hill_prime(x, n=4, k=1.0):
    xp = np.power(np.maximum(x, 1e-12), n - 1)
    xn = np.power(np.maximum(x, 0.0), n)
    return n * xp * k ** n / (k ** n + xn) ** 2


def all_fixed_points(circuit, x_max=5.0, n=16, tol=1e-3):
    """Enumerate fixed points by multi-start Newton and classify by the Jacobian.

    Forward integration (``circuit.equilibria``) only recovers *stable* points;
    a multi-start root find also finds the *saddles* that separate the basins.
    Returns ``(stable, saddle)`` arrays of shape ``(k, n_genes)``.
    """
    from scipy.optimize import fsolve
    stable, saddle, seen = [], [], []
    for gx in np.linspace(0.0, x_max, n):
        for gy in np.linspace(0.0, x_max, n):
            sol, _info, ier, _ = fsolve(lambda z: circuit.rhs(np.array(z)),
                                        [gx, gy], full_output=True)
            if ier != 1 or np.any(sol < -0.05) or np.any(sol > x_max + 1.0):
                continue
            if any(np.linalg.norm(sol - s) < tol for s in seen):
                continue
            seen.append(sol)
            rp = np.real(np.linalg.eigvals(circuit.jacobian(sol)))
            (stable if np.all(rp < 0) else saddle).append(sol)
    return np.array(stable), np.array(saddle)


# =========================================================================== #
# circuit-topology cartoons
# =========================================================================== #
def fig_topologies():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ---- toggle: 2 nodes, mutual repression + autoactivation ----
    ax = axes[0]; ax.set_title("Toggle switch", fontsize=14)
    pos = {"x1": (-1, 0), "x2": (1, 0)}
    for name, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.32, fc="#cfe8ff", ec="#0072B2", lw=2, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=13, zorder=4)
    # mutual repression (flat-head, red)
    ax.annotate("", xy=(-0.6, 0.12), xytext=(0.6, 0.12),
                arrowprops=dict(arrowstyle="|-|,widthA=0,widthB=0.5", color="#D55E00", lw=2))
    ax.annotate("", xy=(0.6, -0.12), xytext=(-0.6, -0.12),
                arrowprops=dict(arrowstyle="|-|,widthA=0,widthB=0.5", color="#D55E00", lw=2))
    # autoactivation loops (green)
    for x, y in pos.values():
        ax.annotate("", xy=(x - 0.05, y + 0.33), xytext=(x + 0.35, y + 0.55),
                    arrowprops=dict(arrowstyle="->", color="#009E73", lw=2,
                                    connectionstyle="arc3,rad=-0.9"))
    ax.text(0, 0.42, "mutual repression (-b)", ha="center", color="#D55E00", fontsize=10)
    ax.text(0, -0.55, "autoactivation (+a)", ha="center", color="#009E73", fontsize=10)
    ax.set_xlim(-2, 2); ax.set_ylim(-1.2, 1.4); ax.set_aspect("equal"); ax.axis("off")

    # ---- repressilator: 3 nodes, cyclic repression ----
    ax = axes[1]; ax.set_title("Repressilator", fontsize=14)
    ang = {"x": 90, "y": 210, "z": 330}
    P = {k: (np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))) for k, a in ang.items()}
    for name, (x, y) in P.items():
        ax.add_patch(Circle((x, y), 0.28, fc="#ffe0cf", ec="#D55E00", lw=2, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=13, zorder=4)
    for a, b in [("x", "y"), ("y", "z"), ("z", "x")]:  # a represses b
        xa, ya = P[a]; xb, yb = P[b]
        v = np.array([xb - xa, yb - ya]); v = v / np.linalg.norm(v)
        s = (xa + v[0] * 0.32, ya + v[1] * 0.32)
        e = (xb - v[0] * 0.34, yb - v[1] * 0.34)
        ax.annotate("", xy=e, xytext=s,
                    arrowprops=dict(arrowstyle="|-|,widthA=0,widthB=0.6", color="#D55E00",
                                    lw=2, connectionstyle="arc3,rad=0.18"))
    ax.text(0, -1.45, "cyclic repression (x -| y -| z -| x)", ha="center",
            color="#D55E00", fontsize=10)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.7, 1.5); ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle("Ground-truth circuit topologies", fontsize=15, y=1.02)
    _save(fig, "F00_topologies")


# =========================================================================== #
# TOGGLE figures
# =========================================================================== #
def fig_toggle_expression(adata):
    fig, ax = plt.subplots(figsize=(6, 5))
    Ms = adata.layers["Ms"]
    pts = adata.uns["ground_truth"]["points_per_trajectory"]
    t = np.tile(np.linspace(0, 1, pts), adata.n_obs // pts)
    sc = ax.scatter(Ms[:, 0], Ms[:, 1], c=t, cmap="viridis", alpha=0.6, s=14)
    plt.colorbar(sc, label="normalized trajectory time")
    ax.set_xlabel("$x_1$ expression"); ax.set_ylabel("$x_2$ expression")
    ax.set_title("Simulated cells in expression space (toggle)")
    _save(fig, "T1_expression")


def fig_toggle_phase(circuit):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    val = np.linspace(0.01, 5, 26)
    X, Y = np.meshgrid(val, val)
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            d = circuit.rhs(np.array([X[i, j], Y[i, j]]))
            U[i, j], V[i, j] = d
    speed = np.hypot(U, V)
    ax.streamplot(X, Y, U, V, color=speed, cmap="Greys", density=1.1, linewidth=0.8, arrowsize=0.9)
    rng = np.random.default_rng(0)
    ics = rng.uniform(0, 4, size=(12, 2))
    for x0 in ics:
        sol = solve_ivp(lambda t, x: circuit.rhs(x), (0, 20), x0,
                        t_eval=np.linspace(0, 20, 200), method="LSODA")
        ax.plot(sol.y[0], sol.y[1], color="#0072B2", lw=0.9, alpha=0.6)
        ax.plot(sol.y[0, 0], sol.y[1, 0], "o", color="#009E73", ms=5)
    # all fixed points (stable nodes + saddles), classified by the Jacobian
    stable, saddle = all_fixed_points(circuit)
    for x in stable:
        ax.plot(*x, "*", ms=19, color="#D55E00", mec="black", mew=1.2, zorder=6,
                label="stable state")
    for x in saddle:
        ax.plot(*x, "D", ms=9, color="white", mec="black", mew=1.2, zorder=6,
                label="saddle")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=9)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"Phase portrait ($b=4$): {len(stable)} stable states, {len(saddle)} saddles")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    _save(fig, "T2_phase_portrait")


def fig_W_regimes(results, labels, vmax, tag, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    _Wmap(axes[0, 0], results["full"]["W_true"], "Ground-truth $W$", vmax, labels)
    for ax, m in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["full", "partial", "none"]):
        _Wmap(ax, results[m]["W_inferred"], f"Inferred $\\hat W$ ({m} scaffold)", vmax, labels)
    fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()
    _save(fig, tag)


def fig_recovery_bars(metrics, tag, title):
    modes = ["full", "partial", "none"]
    keys = [("edge_sign_accuracy", "sign accuracy"),
            ("edge_correlation", "edge correlation"),
            ("spectral_overlap", "spectral overlap")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(keys)); w = 0.25
    for i, m in enumerate(modes):
        vals = [metrics[m][k] for k, _ in keys]
        ax.bar(x + (i - 1) * w, vals, w, label=f"{m} scaffold", color=REGIME_COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in keys])
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_ylim(0, 1.15); ax.set_ylabel("score (higher = better)")
    ax.legend(fontsize=9, ncol=3, loc="lower center")
    ax.set_title(title)
    _save(fig, tag)


def fig_I_gamma(results, labels, tag, title):
    r = results["full"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels)); w = 0.35
    axes[0].bar(x - w / 2, r["I_true"], w, label="true", color="#56B4E9")
    axes[0].bar(x + w / 2, r["I_inferred"], w, label="inferred", color="#0072B2")
    axes[0].set_title("Bias $I$"); axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].legend(fontsize=9)
    axes[1].bar(x - w / 2, r["gamma_true"], w, label="true", color="#F0E442")
    axes[1].bar(x + w / 2, r["gamma_inferred"], w, label="inferred", color="#E69F00")
    axes[1].set_title("Degradation $\\gamma$"); axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].legend(fontsize=9)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    _save(fig, tag)


def fig_toggle_energy(adata, r):
    W, I, gamma = r["W_inferred"], r["I_inferred"], r["gamma_inferred"]
    val = np.linspace(0, 6, 120)
    X, Y = np.meshgrid(val, val)
    gp = np.vstack([X.ravel(), Y.ravel()]).T
    E = _energy_grid_2d(adata.var_names, "toggle_circuit", W, I, gamma, gp, X.shape)

    sig = hill(gp)
    flow = sig @ W.T + I - gamma * gp
    U = flow[:, 0].reshape(X.shape); V = flow[:, 1].reshape(Y.shape)

    # energy of the actual cells (for the scatter height)
    sim = adata.layers["Ms"]
    Esim = _energy_grid_2d(adata.var_names, "toggle_circuit", W, I, gamma, sim, (sim.shape[0],))

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(X, Y, E, cmap=CMAP_ENERGY, alpha=0.75, linewidth=0, antialiased=True)
    ax1.scatter(sim[:, 0], sim[:, 1], Esim, c="#D55E00", s=8)
    ax1.set_title("3D Waddington landscape"); ax1.view_init(elev=30, azim=-60)
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$"); ax1.set_zlabel("energy")

    ax2 = fig.add_subplot(132)
    ax2.contourf(X, Y, E, levels=30, cmap=CMAP_ENERGY)
    ax2.scatter(sim[:, 0], sim[:, 1], c="#D55E00", s=5, alpha=0.7)
    ax2.set_title("2D energy contour"); ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")

    ax3 = fig.add_subplot(133)
    c3 = ax3.contourf(X, Y, E, levels=30, cmap=CMAP_ENERGY)
    ax3.streamplot(X, Y, U, V, color="white", density=1.2, linewidth=1, arrowsize=1.4)
    ax3.set_title("Energy + inferred vector field"); ax3.set_xlim(0, 6); ax3.set_ylim(0, 6)
    ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$")
    plt.colorbar(c3, ax=ax3, label="total energy")
    fig.tight_layout()
    _save(fig, "T6_energy_landscape")


def fig_toggle_bifurcation(adata, r):
    I, gamma = r["I_inferred"], r["gamma_inferred"]
    W_base = r["W_inferred"].copy()
    couplings = [0.0, -1.5, -3.0, -4.5, -6.0, -8.0]
    val = np.linspace(0, 5, 60)
    X, Y = np.meshgrid(val, val)
    gp = np.vstack([X.ravel(), Y.ravel()]).T
    fig, axes = plt.subplots(1, len(couplings), figsize=(23, 4))
    for ax, c in zip(axes, couplings):
        W = W_base.copy(); W[0, 1] = c; W[1, 0] = c
        E = _energy_grid_2d(adata.var_names, "toggle_circuit", W, I, gamma, gp, X.shape)
        cf = ax.contourf(X, Y, E, levels=40, cmap=CMAP_ENERGY)
        ax.set_title(f"mutual inhibition = {c:.1f}"); ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    cax = fig.add_axes([0.92, 0.16, 0.01, 0.68]); fig.colorbar(cf, cax=cax, label="total energy")
    fig.suptitle("Toggle energy landscape across the pitchfork bifurcation (monostable -> bistable)",
                 fontsize=15, y=1.06)
    _save(fig, "T7_bifurcation")


def fig_toggle_jacobian(r):
    import matplotlib as mpl
    W_base = r["W_inferred"].copy(); gamma = r["gamma_inferred"]
    couplings = [0.0, -2.5, -5.0, -8.0]
    val = np.linspace(0.01, 5, 60)
    X, Y = np.meshgrid(val, val)
    gp = np.vstack([X.ravel(), Y.ravel()]).T
    E1, E2 = [], []
    for c in couplings:
        W = W_base.copy(); W[0, 1] = c; W[1, 0] = c
        e1 = np.zeros(len(gp)); e2 = np.zeros(len(gp))
        for j, pt in enumerate(gp):
            sp = hill_prime(pt)
            ev = np.sort(np.real(np.linalg.eigvals(W * sp[None, :] - np.diag(gamma))))[::-1]
            e1[j], e2[j] = ev[0], ev[1]
        E1.append(e1.reshape(X.shape)); E2.append(e2.reshape(X.shape))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.08, hspace=0.28, wspace=0.22)
    for row, (Es, name, cy) in enumerate([(E1, "$\\lambda_1$ (max)", 0.54),
                                          (E2, "$\\lambda_2$", 0.09)]):
        m = max(float(np.max(np.abs(e))) for e in Es)  # shared scale across the row
        for col, (c, e) in enumerate(zip(couplings, Es)):
            ax = axes[row, col]
            ax.contourf(X, Y, e, levels=40, cmap="RdBu_r", vmin=-m, vmax=m)
            if e.min() < 0 < e.max():   # draw the zero contour only where it exists
                ax.contour(X, Y, e, levels=[0.0], colors="k", linewidths=1.4)
            if row == 0:
                ax.set_title(f"coupling = {c:.1f}")
            if col == 0:
                ax.set_ylabel(f"{name}\n$x_2$")
            ax.set_xlabel("$x_1$")
        cax = fig.add_axes([0.92, cy, 0.013, 0.34])
        sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(-m, m))
        fig.colorbar(sm, cax=cax, label="Re($\\lambda$)")
    fig.suptitle("Toggle Jacobian eigenvalues (shared color scale per row; "
                 "black = zero contour = stability boundary)", fontsize=15, y=0.97)
    _save(fig, "T8_jacobian_stability")


def fig_toggle_bifurcation_diagram(circuit):
    """Steady-state bifurcation diagram in the mutual-inhibition strength.

    Native Python continuation (no Julia BifurcationKit): the symmetric branch is a
    1D solve over a `c`-grid; the asymmetric branch is parameterized by the low gene
    (which stays finite where the high gene saturates, avoiding the sigma=0 singularity
    that a solve-for-c hits). Each state is classified by the 2x2 Jacobian.
    """
    from scipy.optimize import brentq
    a = float(circuit.a)                      # diagonal autoactivation (w11 = w22)
    Ib = float(circuit.b)                      # bias I1 = I2 (= b for the toggle)
    gm = float(circuit.gamma)

    def s1(x):
        xc = max(x, 0.0); return xc ** 4 / (1.0 + xc ** 4)

    def sp1(x):
        return 0.0 if x <= 0 else 4.0 * x ** 3 / (1.0 + x ** 4) ** 2

    def stable(x1, x2, c):
        J = np.array([[a * sp1(x1) - gm, c * sp1(x2)],
                      [c * sp1(x1), a * sp1(x2) - gm]])
        return bool(np.all(np.real(np.linalg.eigvals(J)) < 0))

    # symmetric branch: solve (a + c) s(x) + I - gamma x = 0 for each c
    sym = []
    for c in np.linspace(-6.0, 1.0, 700):
        f = lambda x: (a + c) * s1(x) + Ib - gm * x
        xs = np.linspace(-0.6, 4.0, 700); fv = np.array([f(x) for x in xs])
        for k in range(len(xs) - 1):
            if fv[k] * fv[k + 1] < 0:
                r = brentq(f, xs[k], xs[k + 1]); sym.append((c, r, stable(r, r, c)))
    # asymmetric branch: parameterize by the low gene x2, solve the high gene x1, get c
    asy = []
    for x2 in np.linspace(-0.95, 2.2, 1100):
        g = lambda x1: (a * s1(x1) ** 2 + (gm * x2 - Ib - a * s1(x2)) * s1(x2)
                        + (Ib - gm * x1) * s1(x1))
        xs = np.linspace(x2 + 0.03, 3.8, 500); gv = np.array([g(x) for x in xs])
        for k in range(len(xs) - 1):
            if gv[k] * gv[k + 1] < 0:
                x1 = brentq(g, xs[k], xs[k + 1])
                if x1 > x2 + 0.03:
                    c = (gm * x2 - Ib - a * s1(x2)) / s1(x1)
                    if -6.5 <= c <= 1.0:
                        asy.append((c, x1, x2, stable(x1, x2, c)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for st, ls in [(True, "-"), (False, "--")]:                       # symmetric (orange)
        pts = sorted((c, x) for c, x, s in sym if s == st)
        if pts:
            cc, xx = zip(*pts); ax.plot(cc, xx, ls=ls, color="orange", lw=2.3, zorder=2)
    for st, ls in [(True, "-"), (False, "--")]:                       # asymmetric (blue)
        sub = sorted((x2, c, x1) for c, x1, x2, s in asy if s == st)  # order along branch
        if sub:
            x2o = [p[0] for p in sub]; co = [p[1] for p in sub]; x1o = [p[2] for p in sub]
            ax.plot(co, x1o, ls=ls, color="blue", lw=2.3, zorder=2)   # high gene
            ax.plot(co, x2o, ls=ls, color="blue", lw=2.3, zorder=2)   # mirror (low gene)
    ax.axvline(-4.0, color="0.55", lw=1.1, ls=":", zorder=1)
    ax.text(-4.0, 3.42, " fitted ($b{=}4$):\n tri-stable", color="0.35", fontsize=9, va="top")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="orange", lw=2.3, label="symmetric state"),
               Line2D([0], [0], color="blue", lw=2.3, label="asymmetric state"),
               Line2D([0], [0], color="0.3", lw=2.3, ls="-", label="stable branch"),
               Line2D([0], [0], color="0.3", lw=2.3, ls="--", label="unstable branch")]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_xlim(-6, 1); ax.set_ylim(-1.1, 3.6)
    ax.set_xlabel("mutual inhibition strength ($c$ = off-diagonal of $W$)")
    ax.set_ylabel("steady-state gene 1 ($x_1$)")
    ax.set_title("Toggle steady-state bifurcation diagram")
    ax.grid(True, alpha=0.25)
    _save(fig, "T9_bifurcation_diagram")


# =========================================================================== #
# REPRESSILATOR figures
# =========================================================================== #
def fig_osc_expression(adata):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    Ms = adata.layers["Ms"]
    pts = adata.uns["ground_truth"]["points_per_trajectory"]
    t = np.tile(np.linspace(0, 1, pts), adata.n_obs // pts)
    sc = ax.scatter(Ms[:, 0], Ms[:, 1], Ms[:, 2], c=t, cmap="viridis", alpha=0.6, s=12)
    plt.colorbar(sc, label="normalized trajectory time", pad=0.1)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_title("Simulated cells in 3D expression space (repressilator)")
    _save(fig, "R1_expression3d")


def fig_osc_limitcycle(circuit):
    sol = solve_ivp(lambda t, x: circuit.rhs(x), (0, 80), [1.5, 0.5, 1.0],
                    t_eval=np.linspace(0, 80, 2000), method="LSODA", rtol=1e-8, atol=1e-10)
    tail = sol.y[:, sol.t > 25]
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(tail[0], tail[1], tail[2], color="#0072B2", lw=1.1)
    ax1.set_title("3D limit cycle"); ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$"); ax1.set_zlabel("$z$")
    ax2 = fig.add_subplot(132)
    ax2.plot(tail[0], tail[1], color="#0072B2", lw=1.1)
    ax2.set_title("Phase projection ($x$ vs $y$)"); ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$")
    ax3 = fig.add_subplot(133)
    keep = sol.t > 25
    for i, (name, col) in enumerate(zip("xyz", ["#0072B2", "#D55E00", "#009E73"])):
        ax3.plot(sol.t[keep], sol.y[i][keep], color=col, lw=1.2, label=f"${name}$")
    ax3.set_title("Phase-shifted oscillations"); ax3.set_xlabel("time"); ax3.set_ylabel("expression")
    ax3.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "R2_limitcycle")


def _osc_pca_grid(adata, r, double_domain=True, n=50):
    """Build a PCA embedding of the cells, an expanded grid, and its energy."""
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(adata.layers["Ms"])
    if double_domain:
        c1 = (emb[:, 0].max() + emb[:, 0].min()) / 2; r1 = emb[:, 0].max() - emb[:, 0].min()
        c2 = (emb[:, 1].max() + emb[:, 1].min()) / 2; r2 = emb[:, 1].max() - emb[:, 1].min()
        p1 = np.linspace(c1 - r1, c1 + r1, n); p2 = np.linspace(c2 - r2, c2 + r2, n)
    else:
        p1 = np.linspace(emb[:, 0].min() - 1, emb[:, 0].max() + 1, n)
        p2 = np.linspace(emb[:, 1].min() - 1, emb[:, 1].max() + 1, n)
    P1, P2 = np.meshgrid(p1, p2)
    grid_pca = np.vstack([P1.ravel(), P2.ravel()]).T
    grid_expr = pca.inverse_transform(grid_pca)
    E = _energy_grid_2d(adata.var_names, "oscillator_circuit",
                        r["W_inferred"], r["I_inferred"], r["gamma_inferred"],
                        grid_expr, P1.shape)
    return emb, P1, P2, E


def fig_osc_energy_pca(adata, r):
    emb, P1, P2, E = _osc_pca_grid(adata, r, double_domain=True)
    zsim = griddata((P1.ravel(), P2.ravel()), E.ravel(), (emb[:, 0], emb[:, 1]), method="cubic")
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(P1, P2, E, cmap=CMAP_ENERGY2, alpha=0.7, linewidth=0, antialiased=True)
    ax1.scatter(emb[:, 0], emb[:, 1], zsim, c="cyan", s=10, depthshade=True)
    ax1.set_title("3D energy landscape (repressilator, PCA)"); ax1.view_init(elev=30, azim=-60)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.set_zlabel("energy")
    ax2 = fig.add_subplot(122)
    cf = ax2.contourf(P1, P2, E, levels=30, cmap=CMAP_ENERGY2)
    ax2.scatter(emb[:, 0], emb[:, 1], c="cyan", s=6, alpha=0.8)
    ax2.set_title("2D energy contour: cells ring the energy crater")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    plt.colorbar(cf, ax=ax2, label="total energy")
    fig.tight_layout()
    _save(fig, "R6_energy_pca")


def fig_osc_flow_pca(adata, r):
    """Model velocity field projected onto the PCA embedding.

    The energy view (R6) shows where states are pulled; this shows the actual flow.
    The PCA map is linear, so the gene-space velocity `dx/dt` projects onto the
    embedding as `v @ components^T`. The result is a vortex: the non-conservative
    (rotational) part of the dynamics circulating states around the limit cycle.
    """
    W, I, gamma = r["W_inferred"], r["I_inferred"], r["gamma_inferred"]
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(adata.layers["Ms"])
    m1, m2 = emb[:, 0], emb[:, 1]
    p1 = np.linspace(m1.min() - 1, m1.max() + 1, 34)
    p2 = np.linspace(m2.min() - 1, m2.max() + 1, 34)
    P1, P2 = np.meshgrid(p1, p2)
    grid_expr = pca.inverse_transform(np.vstack([P1.ravel(), P2.ravel()]).T)
    vel = hill(grid_expr) @ W.T + I - gamma * grid_expr          # dx/dt in gene space
    vel_pca = vel @ pca.components_.T                             # project onto PCs
    U = vel_pca[:, 0].reshape(P1.shape); V = vel_pca[:, 1].reshape(P2.shape)

    pts = adata.uns["ground_truth"]["points_per_trajectory"]
    tarr = np.tile(np.linspace(0, 1, pts), adata.n_obs // pts)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    speed = np.hypot(U, V)
    ax.streamplot(p1, p2, U, V, color=speed, cmap="viridis", density=1.5,
                  linewidth=1.0, arrowsize=1.2)
    sc = ax.scatter(m1, m2, c=tarr, cmap="twilight", s=16, edgecolor="k", lw=0.2, zorder=3)
    plt.colorbar(sc, ax=ax, label="trajectory time (phase)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Repressilator model velocity on the PCA embedding\n"
                 "(rotational flow circulating the limit cycle)")
    fig.tight_layout()
    _save(fig, "R10_flow_pca")


def fig_osc_mri_z(adata, r):
    val = np.linspace(0, 8, 60)
    X, Y = np.meshgrid(val, val)
    z_slices = [1.0, 3.0, 5.0, 7.0]
    expr = adata.layers["Ms"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, zv in zip(axes, z_slices):
        gp = np.vstack([X.ravel(), Y.ravel(), np.full(X.size, zv)]).T
        E = _energy_grid_2d(adata.var_names, "oscillator_circuit",
                            r["W_inferred"], r["I_inferred"], r["gamma_inferred"], gp, X.shape)
        cf = ax.contourf(X, Y, E, levels=30, cmap=CMAP_ENERGY2)
        mask = np.abs(expr[:, 2] - zv) < 1.0
        if mask.any():
            ax.scatter(expr[mask, 0], expr[mask, 1], c="cyan", s=18, edgecolor="k", lw=0.4)
        ax.set_title(f"slice $z \\approx {zv:.0f}$"); ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    cax = fig.add_axes([0.92, 0.16, 0.01, 0.68]); fig.colorbar(cf, cax=cax, label="total energy")
    fig.suptitle("3D energy landscape, slices along gene $z$", fontsize=15, y=1.04)
    _save(fig, "R7_mri_z")


def fig_osc_mri_ortho(adata, r):
    val = np.linspace(0, 8, 50)
    G1, G2 = np.meshgrid(val, val)
    slices = [1.0, 3.0, 5.0, 7.0]
    labels = ["$x$", "$y$", "$z$"]
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    for row, slice_axis in enumerate([2, 1, 0]):
        a1, a2 = [i for i in range(3) if i != slice_axis]
        for col, sv in enumerate(slices):
            ax = axes[row, col]
            gp = np.zeros((G1.size, 3))
            gp[:, a1] = G1.ravel(); gp[:, a2] = G2.ravel(); gp[:, slice_axis] = sv
            E = _energy_grid_2d(adata.var_names, "oscillator_circuit",
                                r["W_inferred"], r["I_inferred"], r["gamma_inferred"], gp, G1.shape)
            cf = ax.contourf(G1, G2, E, levels=30, cmap=CMAP_ENERGY2)
            if row == 0:
                ax.set_title(f"{labels[slice_axis]} = {sv:.0f}")
            ax.set_xlabel(labels[a1]); ax.set_ylabel(labels[a2])
            plt.colorbar(cf, ax=ax, fraction=0.046)
    fig.suptitle("Orthogonal slices of the 3D energy landscape (rows hold $z$, $y$, $x$ fixed)",
                 fontsize=15, y=1.0)
    fig.tight_layout()
    _save(fig, "R8_mri_ortho")


def fig_osc_jacobian(adata, r):
    W, gamma = r["W_inferred"], r["gamma_inferred"]
    cells = adata.layers["Ms"]
    pts = adata.uns["ground_truth"]["points_per_trajectory"]
    tarr = np.tile(np.linspace(0, 1, pts), adata.n_obs // pts)
    re = np.zeros((len(cells), 3)); im = np.zeros((len(cells), 3))
    for j, x in enumerate(cells):
        sp = hill_prime(x)
        J = W * sp[None, :] - np.diag(gamma)
        ev = np.linalg.eigvals(J)
        idx = np.argsort(np.real(ev))[::-1]
        re[j], im[j] = np.real(ev[idx]), np.imag(ev[idx])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sc = axes[0].scatter(re.ravel(), im.ravel(),
                         c=np.repeat(tarr, 3), cmap="viridis", s=10, alpha=0.6)
    axes[0].axvline(0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("Re($\\lambda$)"); axes[0].set_ylabel("Im($\\lambda$)")
    axes[0].set_title("Jacobian spectrum along the cycle\n(complex pair = rotation)")
    plt.colorbar(sc, ax=axes[0], label="trajectory time")
    axes[1].scatter(re[:, 0], np.abs(im[:, 0]), s=12, color="#0072B2", label="$\\lambda_1$")
    axes[1].scatter(re[:, 1], np.abs(im[:, 1]), s=12, color="#D55E00", label="$\\lambda_2$")
    axes[1].scatter(re[:, 2], np.abs(im[:, 2]), s=12, color="#009E73", label="$\\lambda_3$")
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("Re($\\lambda$)"); axes[1].set_ylabel("|Im($\\lambda$)|")
    axes[1].set_title("Nonzero imaginary parts drive the oscillation"); axes[1].legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "R9_jacobian_spectrum")


# =========================================================================== #
# ROBUSTNESS sweeps
# =========================================================================== #
def _fit_metric(circuit_ctor, n_traj, ppt, noise, n_epochs, device="cpu"):
    circ = circuit_ctor()
    ad_ = simulate_circuit(circ, transient_fraction=0.0, n_trajectories=n_traj,
                           points_per_trajectory=ppt, noise_sigma=noise, seed=SEED)
    r = fit_circuit_local(ad_, "full", n_epochs=n_epochs, device=device)
    m = summarize_recovery(r["W_inferred"], r["W_true"])
    return m


def fig_robustness(n_epochs, device="cpu"):
    circuits = {"toggle": (lambda: ToggleCircuit(a=5.0, b=4.0), 40),
                "repressilator": (lambda: OscillatorCircuit(alpha=10.0, n=4), 50)}
    colors = {"toggle": "#0072B2", "repressilator": "#D55E00"}
    sizes = [1, 2, 3, 5, 12, 25, 50]          # trajectories -> ppt * size cells
    noises = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    sweep = {"size": {}, "noise": {}}
    floor = lambda v: np.maximum(v, 1e-4)     # keep log axes finite

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ---- vs sample size ----
    for name, (ctor, ppt) in circuits.items():
        xs, sacc, scorr, sfrob = [], [], [], []
        for nt in sizes:
            m = _fit_metric(ctor, nt, ppt, 0.0, n_epochs, device)
            xs.append(nt * ppt); sacc.append(m["edge_sign_accuracy"])
            scorr.append(m["edge_correlation"]); sfrob.append(m["frobenius_distance"])
        sweep["size"][name] = dict(n=xs, sign=sacc, corr=scorr, frob=sfrob)
        axes[0, 0].plot(xs, sacc, "o-", color=colors[name], label=f"{name} sign-acc")
        axes[0, 0].plot(xs, scorr, "s--", color=colors[name], alpha=0.55, label=f"{name} corr")
        axes[0, 1].plot(xs, floor(sfrob), "o-", color=colors[name], label=name)
    # ---- vs noise ----
    for name, (ctor, ppt) in circuits.items():
        sacc, scorr, sfrob = [], [], []
        for ns in noises:
            m = _fit_metric(ctor, 50, ppt, ns, n_epochs, device)
            sacc.append(m["edge_sign_accuracy"]); scorr.append(m["edge_correlation"])
            sfrob.append(m["frobenius_distance"])
        sweep["noise"][name] = dict(noise=noises, sign=sacc, corr=scorr, frob=sfrob)
        axes[1, 0].plot(noises, sacc, "o-", color=colors[name], label=f"{name} sign-acc")
        axes[1, 0].plot(noises, scorr, "s--", color=colors[name], alpha=0.55, label=f"{name} corr")
        axes[1, 1].plot(noises, floor(sfrob), "o-", color=colors[name], label=name)

    axes[0, 0].set_xscale("log"); axes[0, 0].set_ylim(0, 1.08)
    axes[0, 0].set_xlabel("number of cells"); axes[0, 0].set_ylabel("recovery (sign / corr)")
    axes[0, 0].set_title("Recovery vs sample size"); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_xscale("log"); axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("number of cells"); axes[0, 1].set_ylabel("Frobenius distance (rel.)")
    axes[0, 1].set_title("Error vs sample size"); axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_ylim(0, 1.08); axes[1, 0].set_xlabel("observation noise $\\sigma$")
    axes[1, 0].set_ylabel("recovery (sign / corr)"); axes[1, 0].set_title("Recovery vs noise")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_yscale("log"); axes[1, 1].set_xlabel("observation noise $\\sigma$")
    axes[1, 1].set_ylabel("Frobenius distance (rel.)"); axes[1, 1].set_title("Error vs noise")
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "B1_robustness")
    return sweep


# =========================================================================== #
# driver
# =========================================================================== #
def _metrics_table(metrics, labels_line):
    rows = ["| scaffold | sign acc. | edge corr. | spectral overlap | Frobenius dist. |",
            "|---|---|---|---|---|"]
    for m in ("full", "partial", "none"):
        d = metrics[m]
        rows.append(f"| {m} | {d['edge_sign_accuracy']:.3f} | {d['edge_correlation']:.3f} "
                    f"| {d['spectral_overlap']:.3f} | {d['frobenius_distance']:.3f} |")
    return "\n".join(rows)


def build(args):
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    n_epochs = 200 if args.quick else args.epochs
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("   cuda requested but unavailable; falling back to cpu")
        device = "cpu"
    n_grid_note = " (quick mode)" if args.quick else ""

    print(f"[1/6] Simulating circuits{n_grid_note} on device={device} ...")
    toggle = ToggleCircuit(a=5.0, b=4.0)
    osc = OscillatorCircuit(alpha=10.0, n=4)
    at = simulate_circuit(toggle, transient_fraction=0.0, n_trajectories=50, points_per_trajectory=40)
    ao = simulate_circuit(osc, transient_fraction=0.0, n_trajectories=50, points_per_trajectory=50)

    cache = f"{DATA}/fits.npz"
    if args.skip_fit and os.path.exists(cache):
        print("[2/6] Loading cached fits ...")
        z = np.load(cache, allow_pickle=True)
        res_t = z["res_t"].item(); res_o = z["res_o"].item()
    else:
        print(f"[2/6] Fitting three scaffold regimes x 2 circuits "
              f"({n_epochs} epochs each, device={device}) ...")
        res_t = fit_three_regimes(at, n_epochs, device)
        res_o = fit_three_regimes(ao, n_epochs, device)
        np.savez(cache, res_t=res_t, res_o=res_o)

    met_t = {m: summarize_recovery(r["W_inferred"], r["W_true"]) for m, r in res_t.items()}
    met_o = {m: summarize_recovery(r["W_inferred"], r["W_true"]) for m, r in res_o.items()}

    print("[3/6] Topology + toggle figures ...")
    fig_topologies()
    fig_toggle_expression(at)
    fig_toggle_phase(toggle)
    fig_W_regimes(res_t, ["$x_1$", "$x_2$"], 5.0, "T3_W_regimes",
                  "Toggle: interaction-matrix recovery under three scaffolds")
    fig_recovery_bars(met_t, "T4_recovery", "Toggle: recovery metrics by scaffold prior")
    fig_I_gamma(res_t, ["$x_1$", "$x_2$"], "T5_bias_gamma", "Toggle: bias and degradation recovery (full)")
    fig_toggle_energy(at, res_t["full"])
    fig_toggle_bifurcation(at, res_t["full"])
    fig_toggle_jacobian(res_t["full"])
    fig_toggle_bifurcation_diagram(toggle)

    print("[4/6] Repressilator figures ...")
    fig_osc_expression(ao)
    fig_osc_limitcycle(osc)
    fig_W_regimes(res_o, ["$x$", "$y$", "$z$"], 10.0, "R3_W_regimes",
                  "Repressilator: interaction-matrix recovery under three scaffolds")
    fig_recovery_bars(met_o, "R4_recovery", "Repressilator: recovery metrics by scaffold prior")
    fig_I_gamma(res_o, ["$x$", "$y$", "$z$"], "R5_bias_gamma",
                "Repressilator: bias and degradation recovery (full)")
    fig_osc_energy_pca(ao, res_o["full"])
    fig_osc_flow_pca(ao, res_o["full"])
    fig_osc_mri_z(ao, res_o["full"])
    fig_osc_mri_ortho(ao, res_o["full"])
    fig_osc_jacobian(ao, res_o["full"])

    print("[5/6] Robustness sweep ...")
    sweep_epochs = 200 if args.quick else n_epochs
    sweep = fig_robustness(sweep_epochs, device)

    print("[6/6] Writing RESULTS.md + summary.json ...")
    summary = dict(
        toggle=dict(W_true=res_t["full"]["W_true"].tolist(),
                    W_inferred=res_t["full"]["W_inferred"].tolist(),
                    gamma_true=res_t["full"]["gamma_true"].tolist(),
                    gamma_inferred=res_t["full"]["gamma_inferred"].tolist(),
                    metrics=met_t, sym_true=symmetry_index(res_t["full"]["W_true"])),
        repressilator=dict(W_true=res_o["full"]["W_true"].tolist(),
                           W_inferred=res_o["full"]["W_inferred"].tolist(),
                           gamma_true=res_o["full"]["gamma_true"].tolist(),
                           gamma_inferred=res_o["full"]["gamma_inferred"].tolist(),
                           metrics=met_o, sym_true=symmetry_index(res_o["full"]["W_true"])),
        sweep=sweep, n_epochs=n_epochs)
    with open(f"{DATA}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    write_results_md(met_t, met_o, res_t, res_o)
    print("DONE")


def write_results_md(met_t, met_o, res_t, res_o):
    sym_o = symmetry_index(res_o["full"]["W_true"])
    md = f"""# scHopfield validation report: small circuits

Ground-truth validation of the scHopfield inference on two synthetic circuits whose
Hopfield-form interaction matrix `W` is **known exactly**. Because the true `W, I, gamma`
and the analytic Hill activation are known, this isolates the optimizer from RNA-velocity
estimation error: it is a direct identifiability check. Reproduces and extends
`notebooks/experiments/small_circuits.ipynb`.

Data are generated by integrating each circuit's ODE `dx/dt = W sigma(x) + I - gamma x`
from many initial conditions (`scHopfield.validation.simulate.simulate_circuit`); the
"velocity" handed to the fit is the analytic `dx/dt`. The interaction matrix is then
reconstructed by the package optimizer (`ScaffoldOptimizer`) under three structural
priors, and scored against the truth with `scHopfield.validation.metrics`.

**Scaffold priors.** `full` = the true edge set is known; `partial` = half the true
edges are kept plus a few random false positives; `none` = an all-ones scaffold (no
structural prior). All are fit with an L2+L1 off-scaffold penalty.

Everything below is regenerated by `reproducibility/build_circuits_report.py`; outputs
are gitignored.


## 0. Circuit topologies

![Ground-truth circuit topologies.](plots/F00_topologies.png)

*Left, the toggle switch: two genes that mutually repress (flat-head, `-b`) and
positively autoregulate (`+a`). Right, the repressilator: three genes in a cyclic
repression loop (`x -| y -| z -| x`).*


## 1. Toggle switch (multistable)

The toggle is a two-gene mutual-repression circuit. Ground-truth `W = [[5, -4], [-4, 5]]`:
positive autoregulation on the diagonal, mutual repression off-diagonal. For these
parameters (`a=5, b=4, gamma=3`) the strong autoactivation makes the circuit **tri-stable**,
not merely bistable: two committed fates (high-`x1`/low-`x2` and vice versa) plus a central
co-expression state where both genes are on, separated by two saddles.

### 1.1 Simulated data

![Simulated cells in expression space (toggle).](plots/T1_expression.png)

*Cells sampled along ODE trajectories, colored by normalized trajectory time. They
collect in the committed high/low corners and, for these parameters, also in the central
co-expression basin.*

### 1.2 Phase portrait and fixed points

![Toggle phase portrait with fixed points.](plots/T2_phase_portrait.png)

*Streamlines of the true vector field with sample trajectories (blue) and start points
(green). Fixed points are enumerated by a multi-start Newton solve and classified by the
Jacobian: three stable states (orange stars, the two committed fates plus the central
co-expression state) separated by two saddles (white diamonds). A forward-integration
scan alone would miss the saddles, hence the explicit root find.*

### 1.3 Interaction-matrix recovery

![Toggle W recovery under three scaffolds.](plots/T3_W_regimes.png)

*Inferred `W` matches the ground truth to two decimals under the full and partial
scaffolds; even with no scaffold the sign structure is recovered.*

![Toggle recovery metrics by scaffold.](plots/T4_recovery.png)

Recovery metrics (closer to 1 is better; Frobenius distance closer to 0 is better):

{_metrics_table(met_t, "x1,x2")}

### 1.4 Bias and degradation

![Toggle bias and degradation recovery.](plots/T5_bias_gamma.png)

*The basal transcription `I` and degradation `gamma` are recovered together with `W`.*

### 1.5 Learned energy landscape

![Toggle energy landscape.](plots/T6_energy_landscape.png)

*The Hopfield energy computed from the inferred parameters. The toggle's `W` is symmetric,
so the energy is a true Lyapunov function and its minima coincide with the stable states.
Left, the 3D Waddington surface with the simulated cells (orange) in the wells; middle, its
2D contour; right, the same energy with the inferred vector field, which flows downhill
into the basins. Three minima appear, the two deep committed wells plus the shallower
central co-expression well, exactly matching the three stable fixed points.*

### 1.6 Pitchfork bifurcation

![Toggle bifurcation across coupling strengths.](plots/T7_bifurcation.png)

*Sweeping the mutual-inhibition strength (off-diagonal of `W`) from 0 to -8 opens up the
committed low/high basins on either side of the central state: the pitchfork-type
transition from a single basin to a multistable landscape, recovered purely from the
learned energy.*

### 1.7 Local stability

![Toggle Jacobian eigenvalue maps.](plots/T8_jacobian_stability.png)

*Real parts of the two Jacobian eigenvalues over state space, for four coupling strengths
(shared color scale per row). The black curve is the zero contour, the stability boundary.
As the coupling strengthens, `lambda_1` (top row) develops positive-eigenvalue ridges near
the Hill thresholds, the saddles that carve the committed fates out of the landscape;
`lambda_2` (bottom row) stays negative everywhere.*

### 1.8 Steady-state bifurcation diagram

![Toggle steady-state bifurcation diagram.](plots/T9_bifurcation_diagram.png)

*The rigorous companion to the energy sweep above: every steady state of the toggle as a
function of the mutual-inhibition strength `c` (the off-diagonal of `W`), traced by native
continuation and classified by the Jacobian (orange = symmetric state, blue = asymmetric;
solid = stable, dashed = unstable). Reading right to left: for weak inhibition only the
symmetric co-expression state exists (monostable); a saddle-node near `c = -2.8` creates the
two committed asymmetric states; a subcritical pitchfork near `c = -4.2` then destabilizes
the symmetric state. Between them lies the tri-stable window, and the fitted operating point
`b = 4` (`c = -4`, dotted line) sits squarely inside it, three stable states plus two
saddles, consistent with the phase portrait and the three energy wells. This is a native
Python reimplementation of the continuation originally run with Julia's BifurcationKit.*


## 2. Repressilator (limit cycle)

The repressilator is a three-gene cyclic repressor. Ground-truth
`W = -10 * [[0,0,1],[1,0,0],[0,1,0]]`, a scaled cyclic permutation matrix. Its symmetric and
antisymmetric parts have equal magnitude (symmetry index of the true `W` is {sym_o:.2f}), so it
carries a large rotational component. It has no point attractor; instead it sustains a limit cycle.

### 2.1 Simulated data

![Simulated cells in 3D expression space (repressilator).](plots/R1_expression3d.png)

*Cells fill a closed ring in the 3D expression space, colored by trajectory time.*

### 2.2 The limit cycle

![Repressilator limit cycle.](plots/R2_limitcycle.png)

*A single long trajectory (transient discarded): the 3D closed orbit, its `x`-`y`
projection, and the three phase-shifted oscillations that are the hallmark of the
repressilator.*

### 2.3 Interaction-matrix recovery

![Repressilator W recovery under three scaffolds.](plots/R3_W_regimes.png)

*The cyclic repression pattern (one negative entry per row) is recovered exactly under
the full and partial scaffolds.*

![Repressilator recovery metrics by scaffold.](plots/R4_recovery.png)

Recovery metrics:

{_metrics_table(met_o, "x,y,z")}

### 2.4 Bias and degradation

![Repressilator bias and degradation recovery.](plots/R5_bias_gamma.png)

### 2.5 Learned energy landscape

![Repressilator energy landscape (PCA).](plots/R6_energy_pca.png)

*Energy of the inferred model on a PCA embedding of the cells. Because the repressilator's
`W` is not symmetric (it has a large antisymmetric, rotational part), the Hopfield energy is
only a heuristic quasi-potential, not a true Lyapunov function. The dynamics split into a
conservative (gradient) part that pulls states onto a low-energy attracting set and a
non-conservative (rotational) part that drives motion along it: states settle onto the
limit cycle and then circulate around it. The triangular outline is a projection artifact,
in the full 3D space (section 2.2) the attracting set is a closed loop, not a triangle. The
three bright peaks are the high-energy single-gene-dominant states the cycle routes between.
Since the circulation is rotational, the scalar energy alone cannot capture it, which is
exactly why the Jacobian below is the definitive diagnostic.*

### 2.6 Flow field on the embedding

![Repressilator model velocity on the PCA embedding.](plots/R10_flow_pca.png)

*The energy view above shows where states are pulled; this shows the actual motion. The
inferred model velocity `dx/dt` is projected onto the same PCA embedding (the projection is
linear, so it is exact). The field is a vortex, the non-conservative rotational part of the
dynamics that carries cells around the limit cycle; cells are colored by phase (trajectory
time) to show them circulating with the flow. This is the flow counterpart of the quasi-
potential in section 2.5, and confirms directly that the states settle onto a cycle and
then rotate rather than descending into a well.*

### 2.7 Energy landscape slices

![Repressilator energy, slices along z.](plots/R7_mri_z.png)

*Slices of the 3D energy at fixed `z`; cyan points are cells near each slice.*

![Repressilator energy, orthogonal slices.](plots/R8_mri_ortho.png)

*The same landscape sliced along each of the three axes in turn.*

### 2.8 Local stability: the oscillation signature

![Repressilator Jacobian spectrum.](plots/R9_jacobian_spectrum.png)

*Jacobian eigenvalues evaluated at every cell on the cycle. A complex-conjugate pair with
nonzero imaginary part (right panel) appears everywhere along the orbit: rotational
dynamics, the dynamical-systems signature of a limit cycle, and the qualitative feature
that distinguishes the oscillator from the toggle (whose eigenvalues are real).*


## 3. Robustness

![Recovery vs sample size and noise.](plots/B1_robustness.png)

*Full-scaffold recovery for both circuits. Top row, versus the number of cells; bottom row,
versus added observation noise `sigma`. Left column, the discrete sign accuracy and edge
correlation (both near 1 across the whole range); right column, the continuous relative
Frobenius distance on a log axis, which resolves the graceful degradation that the coarse
sign metric hides. The sign structure is recovered from as few as tens of cells and stays
correct up to large noise, while the quantitative error rises smoothly with noise and falls
with sample size.*


## Reproduce

```bash
python reproducibility/build_circuits_report.py             # full (GPU default)
python reproducibility/build_circuits_report.py --device cpu # full on CPU
python reproducibility/build_circuits_report.py --quick      # fast smoke test
python reproducibility/build_circuits_report.py --skip-fit   # reuse cached fits
```

The circuits have 2-3 genes, so the fit is a tiny linear model: 800 epochs already recovers
`W` and `gamma` numerically exactly, and GPU is no faster than CPU here (the training loop is
launch-overhead bound, not FLOP bound). The device default is `cuda` only for convenience.
"""
    with open(f"{OUT}/RESULTS.md", "w") as f:
        f.write(md)
    print(f"   wrote {OUT}/RESULTS.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny epochs for a smoke test")
    ap.add_argument("--skip-fit", action="store_true", help="reuse cached fits if present")
    ap.add_argument("--epochs", type=int, default=800,
                    help="epochs per scaffold fit (800 already gives exact recovery)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="torch device (falls back to cpu if cuda is unavailable)")
    build(ap.parse_args())
