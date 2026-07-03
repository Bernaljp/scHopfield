"""Figure pack 1: small circuits (dynamics, recovery, bifurcations, Hill, Jacobians).

Self-contained, CPU-only. Uses the ground-truth circuits in
``scHopfield.validation.circuits`` (Toggle, Oscillator) whose interaction matrix W is
known exactly, plus the recovery metrics in ``scHopfield.validation.metrics`` and the
package's own Hill fit (``scHopfield._utils.math.fit_sigmoid``, the improved bounded
version). Generates every informative figure from these circuits into

    figure_packs/pack1_small_circuits/{plots,data}/  +  FIGURE_GUIDE.md

Run:  .venv/bin/python analyses/figure_packs/pack1_small_circuits.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scHopfield.validation.circuits import ToggleCircuit, OscillatorCircuit
from scHopfield.validation.metrics import summarize_recovery
from scHopfield._utils.math import fit_sigmoid, sigmoid

OUT = "figure_packs/pack1_small_circuits"
PLOTS = f"{OUT}/plots"
DATA = f"{OUT}/data"
SEED = 0


# --------------------------------------------------------------------------- #
# data generation + recovery (uses the package's estimator math)
# --------------------------------------------------------------------------- #
def sample_states(circuit, mode="box", n=2000, x_max=6.0, noise=0.0, seed=SEED):
    """Return (X, V): states and their velocities dx/dt = rhs(X) (+ optional noise).

    mode='box'        uniform box sampling (dense vector-field coverage);
    mode='trajectory' points collected along ODE trajectories from random ICs
                      (realistic, trajectory-confined coverage).
    """
    rng = np.random.default_rng(seed)
    g = circuit.n_genes
    if mode == "box":
        X = rng.uniform(0.0, x_max, size=(n, g))
    else:
        ics = circuit.sample_initial_conditions(n=max(8, n // 250), seed=seed)
        chunks = []
        for x0 in ics:
            _, y = circuit.simulate(t_end=60.0, n_samples=250, initial_state=x0)
            chunks.append(y)
        X = np.vstack(chunks)
        if X.shape[0] > n:
            X = X[rng.choice(X.shape[0], n, replace=False)]
    V = np.array([circuit.rhs(x) for x in X])
    if noise > 0:
        V = V + rng.normal(0.0, noise * V.std(axis=0, keepdims=True), size=V.shape)
    return X, V


def recover_W(circuit, X, V, use_true_sigma=True):
    """Recover (W, I) by the scHopfield pseudoinverse estimator: solve
    ``[sigma(X), 1] @ [W^T; I] = V + gamma * X`` in least squares.
    """
    g = circuit.n_genes
    if use_true_sigma:
        Sig = circuit.sigma(X)
    else:
        Sig = np.zeros_like(X)
        for j in range(g):
            k, n, _off, _mse = fit_sigmoid(X[:, j])
            Sig[:, j] = sigmoid(X[:, j], k, n)
    gamma = circuit.gamma_vec()
    A = np.hstack([Sig, np.ones((X.shape[0], 1))])
    target = V + gamma[None, :] * X
    sol, *_ = np.linalg.lstsq(A, target, rcond=None)
    W_inf = sol[:-1, :].T
    I_inf = sol[-1, :]
    return W_inf, I_inf


def stable_equilibria(circuit, n_starts=80, t_end=80.0, x_max=8.0, tol=1e-3, seed=SEED):
    """Integrate from random ICs, dedup the attractors, classify stability."""
    from scipy.integrate import solve_ivp
    rng = np.random.default_rng(seed)
    g = circuit.n_genes
    starts = rng.uniform(0, x_max, size=(n_starts, g))
    finals = []
    for x0 in starts:
        sol = solve_ivp(lambda t, x: circuit.rhs(x), (0, t_end), x0,
                        t_eval=[t_end], method="LSODA", rtol=1e-8, atol=1e-10)
        if sol.success:
            finals.append(sol.y[:, -1])
    uniq = []
    for p in finals:
        if not any(np.linalg.norm(p - q) < tol * 50 for q in uniq):
            uniq.append(p)
    stable = []
    for x in uniq:
        ev = np.linalg.eigvals(circuit.jacobian(x))
        if np.all(np.real(ev) < 1e-6):
            stable.append(x)
    return np.array(uniq), np.array(stable)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_toggle_phase(tog):
    """1: toggle-switch phase portrait (vector field + trajectories + attractors)."""
    gx, gy = np.meshgrid(np.linspace(0, 6, 22), np.linspace(0, 6, 22))
    U = np.zeros_like(gx); Vv = np.zeros_like(gy)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            d = tog.rhs(np.array([gx[i, j], gy[i, j]]))
            U[i, j], Vv[i, j] = d
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.streamplot(gx, gy, U, Vv, density=1.1, color="#9bb7d4", linewidth=0.7, arrowsize=0.8)
    rng = np.random.default_rng(SEED)
    for x0 in rng.uniform(0.1, 6, size=(10, 2)):
        _, y = tog.simulate(t_end=40, n_samples=400, initial_state=x0)
        ax.plot(y[:, 0], y[:, 1], color="#2a4d69", lw=0.8, alpha=0.7)
    _, stable = stable_equilibria(tog)
    if stable.size:
        ax.scatter(stable[:, 0], stable[:, 1], s=140, c="#c0392b", zorder=5,
                   edgecolor="k", label="stable equilibria")
    ax.set(xlabel="gene A", ylabel="gene B",
           title=f"Toggle switch phase portrait (b={tog.b}, bistable)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/01_toggle_phase_portrait.png", dpi=140)
    plt.close(fig)


def fig_toggle_timeseries(tog):
    """2: toggle trajectories converging to the two attractors."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rng = np.random.default_rng(1)
    for x0 in rng.uniform(0.1, 6, size=(8, 2)):
        t, y = tog.simulate(t_end=15, n_samples=400, initial_state=x0)
        axes[0].plot(t, y[:, 0], color="#2a6f97", lw=0.9, alpha=0.7)
        axes[1].plot(t, y[:, 1], color="#bc4749", lw=0.9, alpha=0.7)
    axes[0].set(xlabel="time", ylabel="gene A", title="Gene A converges to one of two states")
    axes[1].set(xlabel="time", ylabel="gene B", title="Gene B (anti-correlated)")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/02_toggle_timeseries.png", dpi=140)
    plt.close(fig)


def fig_oscillator(osc):
    """3+4: repressilator time series and limit-cycle phase projection."""
    t, y = osc.simulate(t_end=60, n_samples=3000, initial_state=np.array([1.0, 2.0, 3.0]))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, c in enumerate(["#2a6f97", "#e09f3e", "#9e2a2b"]):
        ax.plot(t, y[:, i], color=c, lw=1.1, label=f"gene {osc.state_names[i]}")
    ax.set(xlabel="time", ylabel="expression", title=f"Repressilator sustained oscillation (alpha={osc.alpha}, n={osc.n})")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/03_oscillator_timeseries.png", dpi=140)
    plt.close(fig)

    tail = y[y.shape[0] // 3:]
    fig = plt.figure(figsize=(5.4, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tail[:, 0], tail[:, 1], tail[:, 2], color="#5f0f40", lw=1.0)
    ax.set(xlabel="A", ylabel="B", zlabel="C", title="Limit cycle in state space")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/04_oscillator_limitcycle.png", dpi=140)
    plt.close(fig)


def fig_recovery_curves(circuits):
    """5+6: recovery metrics vs sample size and vs velocity noise."""
    sizes = [50, 100, 250, 500, 1000, 2000, 4000]
    noises = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
    metrics = ["edge_sign_accuracy", "edge_correlation", "frobenius_distance"]
    curves = {name: {"size": {m: [] for m in metrics}, "noise": {m: [] for m in metrics}}
              for name, _ in circuits}

    for name, circ in circuits:
        Wt = circ.W()
        for ns in sizes:
            X, V = sample_states(circ, mode="box", n=ns, noise=0.0)
            r = summarize_recovery(recover_W(circ, X, V)[0], Wt)
            for m in metrics:
                curves[name]["size"][m].append(r[m])
        for nz in noises:
            X, V = sample_states(circ, mode="box", n=2000, noise=nz)
            r = summarize_recovery(recover_W(circ, X, V)[0], Wt)
            for m in metrics:
                curves[name]["noise"][m].append(r[m])

    for tag, xs, xlabel in [("size", sizes, "# samples"), ("noise", noises, "velocity noise (rel.)")]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, m in zip(axes, metrics):
            for name, _ in circuits:
                ax.plot(xs, curves[name][tag][m], "-o", ms=4, label=name)
            ax.set(xlabel=xlabel, ylabel=m, title=m)
            if tag == "size":
                ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"GRN recovery vs {xlabel}", fontweight="bold", y=1.02)
        fig.tight_layout()
        idx = "05" if tag == "size" else "06"
        fig.savefig(f"{PLOTS}/{idx}_recovery_vs_{tag}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    return curves


def fig_true_vs_fitted_sigma(circuits):
    """7: recovery with the true Hill activation vs the fitted (bounded) Hill."""
    metrics = ["edge_sign_accuracy", "edge_correlation", "frobenius_distance"]
    rows = []
    for name, circ in circuits:
        Wt = circ.W()
        X, V = sample_states(circ, mode="trajectory", n=3000, noise=0.02)
        for lbl, ts in [("true sigma", True), ("fitted sigma", False)]:
            r = summarize_recovery(recover_W(circ, X, V, use_true_sigma=ts)[0], Wt)
            rows.append((name, lbl, r))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    names = [n for n, _ in circuits]
    width = 0.35
    for ax, m in zip(axes, metrics):
        for k, lbl in enumerate(["true sigma", "fitted sigma"]):
            vals = [dict((nm, r) for nm, l, r in rows if l == lbl)[nm][m] for nm in names]
            ax.bar(np.arange(len(names)) + k * width, vals, width, label=lbl)
        ax.set_xticks(np.arange(len(names)) + width / 2)
        ax.set_xticklabels(names); ax.set_title(m)
    axes[0].legend(fontsize=8)
    fig.suptitle("Recovery with true vs fitted Hill activation (trajectory-sampled)", fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/07_true_vs_fitted_sigma.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return [(nm, l, r) for nm, l, r in rows]


def fig_toggle_bifurcation():
    """8: pitchfork bifurcation of the toggle switch as mutual repression b varies."""
    bs = np.linspace(0.0, 6.0, 25)
    branches = []
    n_stable = []
    for b in bs:
        tog = ToggleCircuit(b=float(b))
        _, stable = stable_equilibria(tog, n_starts=60, seed=SEED)
        n_stable.append(len(stable))
        for eq in stable:
            branches.append((b, eq[0]))
    branches = np.array(branches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.scatter(branches[:, 0], branches[:, 1], s=18, c="#c0392b")
    ax1.set(xlabel="mutual repression b", ylabel="stable equilibrium (gene A)",
            title="Toggle pitchfork: one state splits into two")
    ax1.grid(alpha=0.3)
    ax2.plot(bs, n_stable, "-o", ms=4, color="#2a6f97")
    ax2.set(xlabel="mutual repression b", ylabel="# stable equilibria",
            title="Monostable -> bistable transition")
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/08_toggle_bifurcation.png", dpi=140)
    plt.close(fig)
    return {"b": bs.tolist(), "n_stable": n_stable}


def fig_oscillator_bifurcation():
    """9: repressilator oscillation amplitude vs alpha and vs Hill n (onset)."""
    def amplitude(alpha, n):
        osc = OscillatorCircuit(alpha=float(alpha), n=int(n))
        _, y = osc.simulate(t_end=200, n_samples=4000, initial_state=np.array([1.0, 2.0, 3.0]))
        tail = y[int(0.6 * y.shape[0]):]
        return float((tail.max(0) - tail.min(0)).mean())

    alphas = np.linspace(1, 20, 20)
    amp_alpha = [amplitude(a, 4) for a in alphas]
    ns = [1, 2, 3, 4, 6, 8]
    amp_n = [amplitude(10, n) for n in ns]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(alphas, amp_alpha, "-o", ms=4, color="#5f0f40")
    a1.set(xlabel="transcription rate alpha", ylabel="oscillation amplitude",
           title="Amplitude grows with alpha (n=4)")
    a1.grid(alpha=0.3)
    a2.plot(ns, amp_n, "-o", ms=4, color="#0f4c5c")
    a2.axvspan(0.5, 1.5, color="grey", alpha=0.15)
    a2.set(xlabel="Hill coefficient n", ylabel="oscillation amplitude",
           title="Cooperativity n>=2 required for oscillation (alpha=10)")
    a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/09_oscillator_bifurcation.png", dpi=140)
    plt.close(fig)
    return {"alphas": alphas.tolist(), "amp_alpha": amp_alpha, "ns": ns, "amp_n": amp_n}


def fig_hill_sensitivity_heatmap():
    """10: bistability region of the toggle over (b, Hill n)."""
    bs = np.linspace(0.5, 5.0, 10)
    ns = [1, 2, 3, 4, 6, 8]
    grid = np.zeros((len(ns), len(bs)))
    for i, n in enumerate(ns):
        for j, b in enumerate(bs):
            tog = ToggleCircuit(b=float(b), n=int(n))
            _, stable = stable_equilibria(tog, n_starts=40, seed=SEED)
            grid[i, j] = len(stable)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   extent=[bs[0], bs[-1], 0, len(ns)])
    ax.set_yticks(np.arange(len(ns)) + 0.5); ax.set_yticklabels(ns)
    ax.set(xlabel="mutual repression b", ylabel="Hill coefficient n",
           title="Bistability region (# stable equilibria)")
    fig.colorbar(im, ax=ax, label="# stable equilibria")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/10_hill_bistability_heatmap.png", dpi=140)
    plt.close(fig)
    return {"b": bs.tolist(), "n": ns, "n_stable_grid": grid.tolist()}


def fig_hill_recovery(circuits):
    """11: how the Hill coefficient n affects GRN recoverability."""
    ns = [1, 2, 3, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    out = {}
    for name, ctor in circuits:
        accs = []
        for n in ns:
            circ = ctor(n=n)
            X, V = sample_states(circ, mode="box", n=2000, noise=0.05)
            r = summarize_recovery(recover_W(circ, X, V)[0], circ.W())
            accs.append(r["edge_correlation"])
        out[name] = accs
        ax.plot(ns, accs, "-o", ms=4, label=name)
    ax.set(xlabel="Hill coefficient n", ylabel="edge correlation (recovered vs true W)",
           title="Steeper Hill nonlinearity aids identifiability")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/11_hill_vs_recovery.png", dpi=140)
    plt.close(fig)
    return out


def fig_jacobian_spectra(tog, osc):
    """12: Jacobian eigenvalues at the circuits' equilibria (stability geometry)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    _, stable = stable_equilibria(tog)
    for eq in stable:
        ev = np.linalg.eigvals(tog.jacobian(eq))
        a1.scatter(ev.real, ev.imag, s=80, c="#c0392b", edgecolor="k", zorder=3)
    a1.axvline(0, color="k", lw=0.8, ls="--")
    a1.set(xlabel="Re", ylabel="Im", title="Toggle: eigenvalues at stable states (Re<0)")
    a1.grid(alpha=0.3)
    # oscillator unstable fixed point (approx via mean of limit cycle)
    _, y = osc.simulate(t_end=200, n_samples=4000, initial_state=np.array([1.0, 2.0, 3.0]))
    fp = y[int(0.6 * y.shape[0]):].mean(0)
    ev = np.linalg.eigvals(osc.jacobian(fp))
    a2.scatter(ev.real, ev.imag, s=90, c="#5f0f40", edgecolor="k", zorder=3)
    a2.axvline(0, color="k", lw=0.8, ls="--")
    a2.set(xlabel="Re", ylabel="Im", title="Repressilator: unstable focus (Re>0, Im!=0)")
    a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/12_jacobian_spectra.png", dpi=140)
    plt.close(fig)


def fig_W_recovery_heatmaps(circuits):
    """13: recovered vs true interaction matrices, side by side."""
    fig, axes = plt.subplots(len(circuits), 2, figsize=(7, 3.4 * len(circuits)))
    if len(circuits) == 1:
        axes = axes[None, :]
    for row, (name, circ) in enumerate(circuits):
        Wt = circ.W()
        X, V = sample_states(circ, mode="box", n=2000, noise=0.05)
        Wi = recover_W(circ, X, V)[0]
        vmax = np.abs(Wt).max()
        for col, (W, ttl) in enumerate([(Wt, "true"), (Wi, "recovered")]):
            im = axes[row, col].imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, col].set_title(f"{name}: W ({ttl})", fontsize=9)
            fig.colorbar(im, ax=axes[row, col], fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/13_W_recovery_heatmaps.png", dpi=140)
    plt.close(fig)


GUIDE = """# Figure pack 1: small circuits

Ground-truth circuits from `scHopfield.validation.circuits` (interaction matrix W known
exactly), scored with `scHopfield.validation.metrics` and the package's improved bounded
Hill fit. Everything here is CPU-only and regenerated by
`analyses/figure_packs/pack1_small_circuits.py`. Targets paper sections R2 (GRN recovery)
and Fig 3 (the Hill nonlinearity).

## Dynamics
![](plots/01_toggle_phase_portrait.png)
Toggle-switch phase portrait: the vector field and sample trajectories fall into two
stable attractors (mutual repression + autoregulation), the canonical bistable memory.

![](plots/02_toggle_timeseries.png)
Time series: each gene commits to a high or low state depending on the basin.

![](plots/03_oscillator_timeseries.png)
![](plots/04_oscillator_limitcycle.png)
Repressilator: sustained oscillation and its closed limit cycle in state space.

## Recovery rate (inferred vs true W)
![](plots/05_recovery_vs_size.png)
Recovery improves with sample size and saturates once the vector field is well covered.

![](plots/06_recovery_vs_noise.png)
Degradation of recovery as velocity noise increases (robustness envelope).

![](plots/07_true_vs_fitted_sigma.png)
Recovery using the true Hill activation vs the **fitted** (bounded) Hill from
`fit_sigmoid`: the improved preprocessing recovers W nearly as well as an oracle sigma.

![](plots/13_W_recovery_heatmaps.png)
Recovered vs true interaction matrices side by side (sign and magnitude structure).

## Bifurcations
![](plots/08_toggle_bifurcation.png)
Pitchfork: increasing mutual repression `b` splits one stable state into two (monostable
-> bistable).

![](plots/09_oscillator_bifurcation.png)
Repressilator amplitude grows with transcription rate `alpha`; sustained oscillation
requires Hill cooperativity `n >= 2`.

## Sensitivity to the Hill parameters
![](plots/10_hill_bistability_heatmap.png)
Bistability region over `(b, n)`: higher cooperativity enlarges the bistable regime.

![](plots/11_hill_vs_recovery.png)
Steeper Hill nonlinearity (larger `n`) makes the GRN more identifiable from velocity.

## Jacobians / stability
![](plots/12_jacobian_spectra.png)
Jacobian eigenvalues: toggle attractors have all `Re<0` (stable); the repressilator's
fixed point is an unstable focus (`Re>0`, nonzero `Im`) that gives rise to the limit cycle.
"""


def main():
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    np.random.seed(SEED)

    tog = ToggleCircuit()          # bistable defaults (b=4)
    osc = OscillatorCircuit()      # repressilator defaults
    circuits = [("toggle", tog), ("oscillator", osc)]
    circuit_ctors = [("toggle", ToggleCircuit), ("oscillator", OscillatorCircuit)]

    print("dynamics...", flush=True)
    fig_toggle_phase(tog)
    fig_toggle_timeseries(tog)
    fig_oscillator(osc)

    print("recovery curves...", flush=True)
    curves = fig_recovery_curves(circuits)
    tvf = fig_true_vs_fitted_sigma(circuits)

    print("bifurcations...", flush=True)
    bif_t = fig_toggle_bifurcation()
    bif_o = fig_oscillator_bifurcation()

    print("hill sensitivity...", flush=True)
    heat = fig_hill_sensitivity_heatmap()
    hill_rec = fig_hill_recovery(circuit_ctors)

    print("jacobians + W heatmaps...", flush=True)
    fig_jacobian_spectra(tog, osc)
    fig_W_recovery_heatmaps(circuits)

    summary = {
        "recovery_curves": {k: {t: {m: list(map(float, v[t][m])) for m in v[t]}
                                 for t in v} for k, v in curves.items()},
        "true_vs_fitted_sigma": [(nm, lbl, {k: float(x) for k, x in r.items()})
                                 for nm, lbl, r in tvf],
        "toggle_bifurcation": bif_t,
        "oscillator_bifurcation": bif_o,
        "hill_bistability": heat,
        "hill_vs_recovery": {k: list(map(float, v)) for k, v in hill_rec.items()},
    }
    json.dump(summary, open(f"{DATA}/pack1_summary.json", "w"), indent=2)
    open(f"{OUT}/FIGURE_GUIDE.md", "w").write(GUIDE)
    print(f"wrote {OUT}/ (13 figures + data + guide)", flush=True)


if __name__ == "__main__":
    main()
