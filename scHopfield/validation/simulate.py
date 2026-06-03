"""Generic ODE simulator for validation circuits.

Each circuit defines a ground-truth ``rhs(x)`` returning :math:`dx/dt` and
optional Hopfield-form ``W``, ``I``, ``gamma``. This driver integrates a
collection of initial conditions, optionally adds observation noise, and
returns an ``AnnData`` whose ``X`` is expression and ``layers['dxdt']`` is the
analytic derivative evaluated along each trajectory. ``adata.uns['ground_truth']``
carries the true ``W, I, gamma``.

The output structure is what ``fit_circuit`` expects: scHopfield consumes
spliced expression in ``adata.X`` and velocity in ``layers[velocity_key]``.
For these synthetic circuits, the "velocity" is the analytic dx/dt, not an
estimate from RNA velocity tools, which is the whole point per Jesper's J1.5.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import anndata as ad
from scipy.integrate import solve_ivp


def simulate_circuit(
    circuit,
    n_trajectories: int = 50,
    points_per_trajectory: int = 40,
    t_end: float = 30.0,
    transient_fraction: float = 0.1,
    noise_sigma: float = 0.0,
    x_max_init: Optional[float] = None,
    cluster_label: str = "synthetic",
    seed: int = 0,
) -> ad.AnnData:
    """Integrate ``circuit.rhs`` from many random initial conditions and
    return an AnnData with expression and analytic dx/dt.

    Parameters
    ----------
    circuit
        A circuit object exposing ``rhs(x)``, ``W()``, ``I_vec()``,
        ``gamma_vec()``, ``n_genes``, ``gene_names``,
        ``sample_initial_conditions(n, ...)``.
    n_trajectories
        Number of distinct initial conditions to integrate. Each produces
        ``points_per_trajectory`` samples after the transient.
    points_per_trajectory
        Number of time samples per trajectory after dropping the transient.
    t_end
        Final integration time. Total samples per cell will be ``points_per_trajectory``.
    transient_fraction
        Fraction of the trajectory at the start to discard as transient.
    noise_sigma
        Standard deviation of Gaussian observation noise added to both
        expression and dx/dt. Set to 0 for clean data.
    x_max_init
        Upper bound for uniform initial-condition sampling. Defaults to the
        circuit's own ``sample_initial_conditions`` default.
    cluster_label
        Value stored in ``obs['cluster']`` for every cell. scHopfield's fit
        infrastructure expects per-cluster grouping; for synthetic circuits
        we use a single cluster.
    seed
        RNG seed for initial conditions and noise.

    Returns
    -------
    adata : AnnData
        ``adata.X`` is expression (n_cells, n_genes).
        ``adata.layers['dxdt']`` is the analytic derivative.
        ``adata.layers['Ms']`` mirrors ``X`` (scHopfield convenience).
        ``adata.layers['velocity_S']`` mirrors ``dxdt``.
        ``adata.obs['cluster']`` is the cluster label.
        ``adata.var['gamma']`` is the ground-truth per-gene degradation.
        ``adata.uns['ground_truth']`` carries the W matrix and other params.
    """
    rng = np.random.default_rng(seed)
    if x_max_init is None:
        ics = circuit.sample_initial_conditions(n_trajectories, seed=seed)
    else:
        ics = rng.uniform(0, x_max_init, size=(n_trajectories, circuit.n_genes))

    t_start_idx = int(transient_fraction * 1000)
    t_eval = np.linspace(0.0, t_end, 1000)

    all_x, all_dx = [], []
    for x0 in ics:
        sol = solve_ivp(
            lambda t, x: circuit.rhs(x),
            t_span=(0.0, t_end), y0=x0,
            t_eval=t_eval, method="LSODA",
            rtol=1e-8, atol=1e-10,
        )
        if not sol.success:
            continue
        # Subsample after the transient
        idxs = np.linspace(t_start_idx, sol.y.shape[1] - 1,
                           points_per_trajectory, dtype=int)
        x_samples = sol.y[:, idxs].T          # (points, n_genes)
        dx_samples = np.stack([circuit.rhs(x) for x in x_samples])
        all_x.append(x_samples)
        all_dx.append(dx_samples)

    X = np.vstack(all_x)
    DX = np.vstack(all_dx)

    if noise_sigma > 0:
        X = X + rng.normal(0.0, noise_sigma, size=X.shape)
        DX = DX + rng.normal(0.0, noise_sigma, size=DX.shape)

    n_cells, n_genes = X.shape
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = list(circuit.gene_names)
    adata.layers["Ms"] = X.astype(np.float32)
    adata.layers["dxdt"] = DX.astype(np.float32)
    adata.layers["velocity_S"] = DX.astype(np.float32)
    adata.obs["cluster"] = cluster_label
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    adata.var["gamma"] = circuit.gamma_vec().astype(np.float32)
    adata.var["scHopfield_used"] = True

    adata.uns["ground_truth"] = {
        "W": np.asarray(circuit.W(), dtype=np.float64),
        "I": np.asarray(circuit.I_vec(), dtype=np.float64),
        "gamma": np.asarray(circuit.gamma_vec(), dtype=np.float64),
        "circuit_repr": repr(circuit),
        "n_trajectories": n_trajectories,
        "points_per_trajectory": points_per_trajectory,
        "t_end": t_end,
        "noise_sigma": noise_sigma,
        "seed": seed,
    }
    return adata
