"""Simulation utilities for gene regulatory network dynamics."""

import numpy as np
from typing import Optional, List, Dict, Union
from anndata import AnnData
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .solver import create_solver, ODESolver
from .._utils.io import get_matrix, to_numpy, get_genes_used
from ._utils import _parse_perturb_genes, _update_scHopfield_uns

# torchdiffeq method names that can be used on the GPU path
_TORCHDIFFEQ_METHODS = frozenset([
    'euler', 'rk4', 'midpoint', 'dopri5', 'dopri8',
    'bosh3', 'adaptive_heun', 'fehlberg2',
])


def _run_jobs(func, items, n_jobs):
    """Run func(item) for each item. Sequential when n_jobs=1, threaded otherwise.
    Falls back to sequential if parallel execution raises any exception."""
    if n_jobs == 1:
        return [func(item) for item in items]
    try:
        return Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(func)(item) for item in items
        )
    except Exception:
        return [func(item) for item in items]


def simulate_trajectory(
    adata: AnnData,
    cluster: str,
    cell_idx: Union[int, List[int]],
    t_span: np.ndarray,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    method: str = 'euler',
    x_max_percentile: float = 99.0,
    n_jobs: int = 1,
    verbose: bool = False
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Simulate trajectory from one or more cells' initial states.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    cluster : str
        Cluster name
    cell_idx : int or list of int
        Index or list of indices of cells to use as initial conditions.
        Returns a single trajectory array for a scalar, or a list for a list.
    t_span : np.ndarray
        Time points for simulation
    spliced_key : str, optional
        Key for expression data
    degradation_key : str, optional
        Key for degradation rates
    method : str, optional (default: 'euler')
        Integration method:
        - 'euler': Simple Euler method with clipping (stable, recommended)
        - 'odeint': scipy.integrate.odeint
        - 'RK45', 'RK23', etc.: scipy.integrate.solve_ivp methods
    x_max_percentile : float, optional (default: 99.0)
        Percentile of expression to use as upper bound. Prevents divergence.
        Set to None to disable upper bound.
    n_jobs : int, optional (default: 1)
        Number of parallel jobs when cell_idx is a list. 1 = sequential,
        -1 = all cores. Uses threads; no effect for a single cell.
    verbose : bool, optional (default: False)
        Print simulation info

    Returns
    -------
    np.ndarray or list of np.ndarray
        Trajectory (len(t_span) × n_genes) for a scalar cell_idx,
        or a list of trajectories for a list input.
    """
    single = isinstance(cell_idx, (int, np.integer))
    indices = [int(cell_idx)] if single else list(cell_idx)

    genes = get_genes_used(adata)
    X_all = to_numpy(get_matrix(adata, spliced_key, genes=genes))

    solver = create_solver(
        adata, cluster, degradation_key,
        spliced_key=spliced_key,
        x_max_percentile=x_max_percentile
    )

    if verbose:
        print(f"Simulating {len(indices)} trajectory/ies in cluster '{cluster}'")
        print(f"  Method: {method}")
        print(f"  Time span: {t_span[0]:.2f} to {t_span[-1]:.2f} ({len(t_span)} points)")
        if solver.x_max is not None:
            print(f"  Upper bound: {x_max_percentile}th percentile × 2")

    def _run_one(idx):
        x0 = np.maximum(X_all[idx].copy(), 0)
        return solver.solve(x0, t_span, method=method, clip_each_step=True)

    results = _run_jobs(_run_one, indices, n_jobs)

    if verbose:
        print(f"  Final values range: [{results[-1][-1].min():.3f}, {results[-1][-1].max():.3f}]")

    return results[0] if single else results


def simulate_perturbation_ode(
    adata: AnnData,
    cluster: str,
    cell_idx: Union[int, List[int]],
    gene_perturbations: dict,
    t_span: np.ndarray,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    method: str = 'euler',
    x_max_percentile: float = 99.0,
    residual_gene_dynamics: bool = False,
    n_jobs: int = 1,
    verbose: bool = False
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Simulate trajectory with gene perturbations using ODE integration.

    By default, perturbed genes (KO/OE) are held fixed at their perturbed values
    throughout the entire simulation. Set residual_gene_dynamics=True to allow
    perturbed genes to evolve according to the ODE dynamics after initial perturbation.

    For CellOracle-style perturbation simulation, use sch.dyn.simulate_perturbation instead.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    cluster : str
        Cluster name
    cell_idx : int or list of int
        Cell index or list of indices for initial conditions.
        Returns a single trajectory for a scalar, or a list for a list.
    gene_perturbations : dict
        Dictionary mapping gene names to perturbation values.
        - Knockout: {"Gata1": 0.0} sets Gata1 to 0
        - Overexpression: {"Gata1": 5.0} sets Gata1 to 5.0
    t_span : np.ndarray
        Time points
    spliced_key : str, optional
        Expression data key
    degradation_key : str, optional
        Degradation rates key
    method : str, optional (default: 'euler')
        Integration method ('euler', 'odeint', 'RK45', etc.)
    x_max_percentile : float, optional (default: 99.0)
        Percentile for upper bound. Set to None to disable.
    residual_gene_dynamics : bool, optional (default: False)
        If False, perturbed genes are held fixed at their perturbed values.
        If True, perturbed genes can evolve according to ODE dynamics after
        the initial perturbation is applied.
    n_jobs : int, optional (default: 1)
        Number of parallel jobs when cell_idx is a list. 1 = sequential,
        -1 = all cores. Uses threads; no effect for a single cell.
    verbose : bool, optional (default: False)
        Print simulation info

    Returns
    -------
    np.ndarray or list of np.ndarray
        Trajectory with perturbations for a scalar cell_idx,
        or a list of trajectories for a list input.
    """
    single = isinstance(cell_idx, (int, np.integer))
    indices = [int(cell_idx)] if single else list(cell_idx)

    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]
    X_all = to_numpy(get_matrix(adata, spliced_key, genes=genes))

    # Parse perturbations once for all cells
    all_indices, all_values = _parse_perturb_genes(
        gene_names, gene_perturbations, validate_non_negative=True
    )
    fixed_indices = all_indices if (not residual_gene_dynamics and len(all_indices) > 0) else None
    fixed_values = all_values if fixed_indices is not None else None

    solver = create_solver(
        adata, cluster, degradation_key,
        spliced_key=spliced_key,
        x_max_percentile=x_max_percentile
    )
    solver.set_fixed_genes(fixed_indices, fixed_values)

    if verbose:
        print(f"Simulating perturbation for {len(indices)} cell(s) in cluster '{cluster}'")
        print(f"  Perturbations: {gene_perturbations}")
        print(f"  Perturbed genes: {'can evolve' if residual_gene_dynamics else 'held constant'}")
        print(f"  Method: {method}")

    def _run_one(idx):
        x0 = np.maximum(X_all[idx].copy(), 0)
        if len(all_indices) > 0:
            x0[all_indices] = all_values
        return solver.solve(x0, t_span, method=method, clip_each_step=True)

    results = _run_jobs(_run_one, indices, n_jobs)
    return results[0] if single else results

def _simulate_cluster_gpu(
    X_cluster: np.ndarray,
    solver: 'ODESolver',
    t_span: np.ndarray,
    method: str,
    device: 'torch.device',
) -> np.ndarray:
    """
    Integrate all cells in a cluster simultaneously on the GPU.

    Implements the same Hopfield ODE as ODESolver but processes the entire
    cluster as a single batched tensor operation instead of a cell-by-cell
    Python loop. Uses torchdiffeq.odeint when available; falls back to a
    native torch Euler loop otherwise (still GPU-batched).

    Parameters
    ----------
    X_cluster : np.ndarray
        Initial expression states, shape (n_cells, n_genes).
    solver : ODESolver
        Configured solver carrying W, I, gamma, threshold, exponent,
        x_min, x_max, and fixed-gene info.
    t_span : np.ndarray
        Time points, shape (n_steps,).
    method : str
        Integration method name recognised by torchdiffeq
        ('euler', 'rk4', 'dopri5', …). If torchdiffeq is unavailable,
        'euler' is handled by a native torch loop; other methods raise a
        warning and also fall back to the torch Euler loop.
    device : torch.device
        GPU (or CPU) device to run the computation on.

    Returns
    -------
    np.ndarray
        Final expression states, shape (n_cells, n_genes), on CPU.
    """
    import torch

    dtype   = torch.float64
    x_min_v = float(solver.x_min)

    # ── Parameters → tensors ─────────────────────────────────────────────────
    W_t         = torch.tensor(solver.W,         dtype=dtype, device=device)
    I_t         = torch.tensor(solver.I,         dtype=dtype, device=device)
    gamma_t     = torch.tensor(solver.gamma,     dtype=dtype, device=device)
    threshold_t = torch.tensor(solver.threshold, dtype=dtype, device=device)
    exponent_t  = torch.tensor(solver.exponent,  dtype=dtype, device=device)

    x_max_t = (
        torch.tensor(solver.x_max, dtype=dtype, device=device)
        if solver.x_max is not None else None
    )

    # ── Initial states ────────────────────────────────────────────────────────
    X0 = torch.tensor(np.maximum(X_cluster, x_min_v), dtype=dtype, device=device)

    fixed_indices   = solver.fixed_indices
    fixed_values    = solver.fixed_values
    fixed_values_t  = None
    fixed_mask_t    = None

    if fixed_indices is not None and len(fixed_indices) > 0:
        fixed_values_t = torch.tensor(fixed_values, dtype=dtype, device=device)
        X0[:, fixed_indices] = fixed_values_t
        fixed_mask_t = torch.zeros(solver.W.shape[0], dtype=torch.bool, device=device)
        fixed_mask_t[fixed_indices] = True

    # ── Batched Hill-sigmoid ODE ──────────────────────────────────────────────
    # Hill sigmoid: x^n / (x^n + s^n)  — matches sigmoid() in _utils/math.py
    def dynamics(t, x):
        # x: (n_cells, n_genes)
        x_c = x.clamp(min=x_min_v)
        if x_max_t is not None:
            x_c = x_c.clamp(max=x_max_t)

        x_pos = x_c.clamp(min=1e-12)          # avoid 0^n for fractional n
        xn    = x_pos ** exponent_t            # (n_cells, n_genes)
        sn    = threshold_t ** exponent_t      # (n_genes,) — broadcast
        sig   = xn / (xn + sn)                # Hill sigmoid

        dxdt = sig @ W_t.T - gamma_t * x_c + I_t  # (n_cells, n_genes)

        # Soft lower boundary: don't push below x_min
        dxdt = torch.where(x <= x_min_v, dxdt.clamp(min=0.0), dxdt)

        # Soft upper boundary
        if x_max_t is not None:
            dxdt = torch.where(x >= x_max_t, dxdt.clamp(max=0.0), dxdt)

        # Fixed genes: zero derivative so they stay constant
        if fixed_mask_t is not None:
            dxdt = dxdt.clone()
            dxdt[:, fixed_mask_t] = 0.0

        return dxdt

    # ── Integration ───────────────────────────────────────────────────────────
    with torch.no_grad():
        try:
            import torchdiffeq
            _have_tde = True
        except ImportError:
            _have_tde = False

        if _have_tde and method in _TORCHDIFFEQ_METHODS:
            t_tensor   = torch.tensor(t_span, dtype=dtype, device=device)
            trajectory = torchdiffeq.odeint(dynamics, X0, t_tensor, method=method)
            # trajectory: (n_steps, n_cells, n_genes) → take last time point
            X_final = trajectory[-1]

        else:
            # Native torch Euler loop — no torchdiffeq dependency required.
            if method != 'euler':
                import warnings
                warnings.warn(
                    f"torchdiffeq not available or method '{method}' is not in "
                    f"_TORCHDIFFEQ_METHODS; falling back to torch Euler on {device}.",
                    UserWarning,
                )
            x = X0.clone()
            for i in range(1, len(t_span)):
                dt_step = float(t_span[i] - t_span[i - 1])
                x = x + dt_step * dynamics(None, x)
                x = x.clamp(min=x_min_v)
                if x_max_t is not None:
                    x = x.clamp(max=x_max_t)
                if fixed_indices is not None and len(fixed_indices) > 0:
                    x[:, fixed_indices] = fixed_values_t
            X_final = x

    return X_final.cpu().numpy()


def simulate_shift_ode(
    adata: 'AnnData',
    perturb_condition: Dict[str, float],
    cluster_key: str,
    dt: float = 5.0,
    n_steps: int = 100,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    method: str = 'euler',
    use_cluster_specific_GRN: bool = True,
    x_max_percentile: float = 99.0,
    residual_gene_dynamics: bool = False,
    n_jobs: int = -1,
    device: Optional[str] = None,
    verbose: bool = False
) -> 'AnnData':
    """
    Simulate dataset-wide trajectory shifts with gene perturbations using ODE integration.

    This function mimics the propagation-based `simulate_shift` but uses continuous
    ODE integration. It calculates the final state for every cell after a time `dt`
    under the perturbed conditions, and stores the resulting shift (delta_X).

    When a CUDA GPU is available (and `method` is GPU-compatible), all cells in
    each cluster are integrated simultaneously as a single batched tensor operation
    via `_simulate_cluster_gpu`, which uses `torchdiffeq.odeint` when installed and
    falls back to a native torch Euler loop otherwise.  The result is always moved
    back to CPU before being stored in the returned AnnData.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    perturb_condition : dict
        Dictionary mapping gene names to perturbation values (e.g., {"Gata1": 0.0}).
    cluster_key : str
        Key in adata.obs containing cluster assignments.
    dt : float, optional (default: 5.0)
        Total time duration to simulate the ODEs.
    n_steps : int, optional (default: 100)
        Number of time steps for the ODE solver.
    spliced_key : str, optional (default: 'Ms')
        Key for expression data.
    degradation_key : str, optional (default: 'gamma')
        Key for degradation rates.
    method : str, optional (default: 'euler')
        Integration method.
        GPU-compatible (via torchdiffeq or native torch):
          'euler', 'rk4', 'midpoint', 'dopri5', 'dopri8', 'bosh3',
          'adaptive_heun', 'fehlberg2'
        CPU-only (scipy):
          'odeint', 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
    use_cluster_specific_GRN : bool, optional (default: True)
        If True, uses cluster-specific solvers. If False, uses a global solver.
    x_max_percentile : float, optional (default: 99.0)
        Percentile for upper bound. Prevents divergence.
    residual_gene_dynamics : bool, optional (default: False)
        If False, perturbed genes are held fixed. If True, they evolve.
    n_jobs : int, optional (default: -1)
        Number of parallel jobs for the CPU fallback cell loop.
        Ignored when the GPU path is active.
    device : str or None, optional (default: None)
        Target device for GPU-batched integration.
        None  → auto-detect: use 'cuda' if available and method is GPU-compatible,
                otherwise 'cpu'.
        'cuda' → force GPU (raises if CUDA unavailable).
        'cpu'  → always use the CPU path (scipy/joblib, as before).
    verbose : bool, optional (default: False)
        Print simulation progress.

    Returns
    -------
    AnnData
        A copy of the input AnnData with 'simulated_count' and 'delta_X' added to layers.
        All arrays are numpy (CPU) regardless of where the integration ran.
    """
    import torch

    # ── Resolve target device ─────────────────────────────────────────────────
    if device == 'cpu':
        use_gpu = False
        torch_device = torch.device('cpu')
    elif device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")
        use_gpu = True
        torch_device = torch.device('cuda')
    else:  # None → auto-detect
        use_gpu = torch.cuda.is_available() and (method in _TORCHDIFFEQ_METHODS)
        torch_device = torch.device('cuda' if use_gpu else 'cpu')

    if verbose:
        backend = f"GPU ({torch_device})" if use_gpu else "CPU"
        print(f"simulate_shift_ode: backend={backend}, method={method}")

    adata_out = adata.copy()

    # Identify used genes
    genes_mask = get_genes_used(adata_out)
    gene_names = adata_out.var_names[genes_mask]

    # Get initial states
    X_orig = to_numpy(get_matrix(adata_out, spliced_key, genes=genes_mask))
    X_sim = np.zeros_like(X_orig)
    V_sim = np.zeros_like(X_orig)

    # Time span for the ODE simulation
    t_span = np.linspace(0, dt, n_steps)

    clusters = adata_out.obs[cluster_key].unique() if use_cluster_specific_GRN else [None]

    for cluster in clusters:
        if verbose:
            print(f"Processing cluster: {cluster if cluster else 'Global'}")

        if cluster is not None:
            cell_indices = np.where(adata_out.obs[cluster_key] == cluster)[0]
        else:
            cell_indices = np.arange(adata_out.n_obs)

        if len(cell_indices) == 0:
            continue

        # Create solver for this cluster
        solver = create_solver(
            adata_out, cluster, degradation_key,
            spliced_key=spliced_key,
            x_max_percentile=x_max_percentile
        )

        # Configure fixed genes based on perturbations
        all_indices, all_values = _parse_perturb_genes(
            gene_names, perturb_condition, validate_non_negative=True
        )
        if not residual_gene_dynamics and len(all_indices) > 0:
            solver.set_fixed_genes(all_indices, all_values)
        else:
            solver.set_fixed_genes(None, None)

        if use_gpu:
            # ── GPU path: integrate all cells in the cluster as one batch ────
            X_cluster = X_orig[cell_indices]
            try:
                X_sim[cell_indices] = _simulate_cluster_gpu(
                    X_cluster, solver, t_span, method, torch_device
                )
            except torch.cuda.OutOfMemoryError:
                import warnings
                warnings.warn(
                    f"GPU OOM on cluster '{cluster}'; falling back to CPU for this cluster.",
                    RuntimeWarning,
                )
                use_gpu = False  # disable GPU for remaining clusters too
                # fall through to CPU path below

        if not use_gpu:
            # ── CPU path: cell-by-cell with joblib threads ───────────────────
            def _simulate_cell(x0_row):
                x0 = np.maximum(x0_row, 0)
                if len(all_indices) > 0:
                    x0[all_indices] = all_values
                return solver.solve(x0, t_span, method=method, clip_each_step=True)[-1]

            desc   = f"Cells in {cluster if cluster else 'global'}"
            x0_list = [X_orig[idx].copy()
                       for idx in (tqdm(cell_indices, desc=desc) if verbose else cell_indices)]
            results = _run_jobs(_simulate_cell, x0_list, n_jobs)
            X_sim[cell_indices] = np.array(results)

        # Velocity at final state (CPU numpy, vectorised over cells)
        V_sim[cell_indices] = solver.dynamics_batch(X_sim[cell_indices], 0.0)


    # Calculate shift (delta_X)
    delta_X = X_sim - X_orig
    
    # Map back to full gene array shape
    n_cells, n_all_genes = adata_out.shape
    delta_X_full = np.zeros((n_cells, n_all_genes))
    X_sim_full = np.zeros((n_cells, n_all_genes))
    V_sim_full = np.zeros((n_cells, n_all_genes))  # Placeholder for velocity
    
    # Assuming genes_mask is a boolean mask or index array
    delta_X_full[:, genes_mask] = delta_X
    X_sim_full[:, genes_mask] = X_sim
    V_sim_full[:, genes_mask] = V_sim  # Store velocity 
    
    # Save to layers
    adata_out.layers['simulated_count'] = X_sim_full
    adata_out.layers['delta_X'] = delta_X_full
    adata_out.layers['simulated_velocity'] = V_sim_full  # Store velocity in layers
    
    # Update scHopfield metadata dict
    _update_scHopfield_uns(adata_out, perturb_condition=perturb_condition,
                           simulation_method='ODE', ode_dt=dt)

    return adata_out


def calculate_trajectory_flow(
    adata: AnnData,
    wt_trajectories: Dict[str, np.ndarray],
    perturbed_trajectories: Dict[str, np.ndarray],
    cluster_key: str = 'cell_type',
    basis: str = 'umap',
    time_point: int = -1,
    method: str = 'hopfield',
    n_neighbors: int = 30,
    n_jobs: int = 4,
    verbose: bool = True,
) -> np.ndarray:
    """
    Calculate perturbation flow from ODE trajectory simulation results.

    Takes the final (or specified) time point from ODE trajectories and
    computes the flow in embedding space.

    Parameters
    ----------
    adata : AnnData
        Annotated data with cell information
    wt_trajectories : dict
        Dictionary mapping cluster -> WT trajectory (n_time, n_genes)
    perturbed_trajectories : dict
        Dictionary mapping cluster -> perturbed trajectory (n_time, n_genes)
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    basis : str, optional (default: 'umap')
        Embedding basis
    time_point : int, optional (default: -1)
        Which time point to use (-1 for final)
    method : str, optional (default: 'hopfield')
        Flow calculation method:
        - 'hopfield': Use Hopfield model velocity directly
        - 'difference': Simple difference in gene space projected to embedding
    n_neighbors : int, optional (default: 30)
        Number of neighbors for projection
    n_jobs : int, optional (default: 4)
        Number of parallel jobs
    verbose : bool, optional (default: True)
        Print progress

    Returns
    -------
    np.ndarray
        Perturbation flow in embedding space (n_cells, 2)
    """
    # Import here to avoid circular imports
    from ..tools.velocity import compute_velocity
    from ..tools.embedding import project_to_embedding

    genes = get_genes_used(adata)
    n_cells = adata.n_obs
    n_genes = len(genes)

    # Initialize arrays
    delta_X = np.zeros((n_cells, n_genes))
    X_wt_final = np.zeros((n_cells, n_genes))
    X_pert_final = np.zeros((n_cells, n_genes))

    # Get final states from trajectories for each cluster
    for cluster in wt_trajectories.keys():
        if cluster not in perturbed_trajectories:
            continue

        mask = adata.obs[cluster_key] == cluster
        if not mask.any():
            continue

        # Get final time point
        wt_final = wt_trajectories[cluster][time_point]
        pert_final = perturbed_trajectories[cluster][time_point]

        # Assign to all cells in this cluster
        cell_indices = np.where(mask)[0]
        for idx in cell_indices:
            X_wt_final[idx] = wt_final
            X_pert_final[idx] = pert_final
            delta_X[idx] = pert_final - wt_final

    # Store delta_X
    adata.layers['delta_X_ode'] = delta_X

    if method == 'hopfield':
        if verbose:
            print("Computing Hopfield velocities...")

        delta_velocity = np.zeros((n_cells, n_genes))

        for cluster in wt_trajectories.keys():
            mask = adata.obs[cluster_key] == cluster
            if not mask.any():
                continue

            cell_indices = np.where(mask)[0]

            # Compute velocity at WT and perturbed states
            v_wt = compute_velocity(adata, X=X_wt_final[mask], cluster=cluster)
            v_pert = compute_velocity(adata, X=X_pert_final[mask], cluster=cluster)

            delta_velocity[cell_indices] = v_pert - v_wt

        if verbose:
            print("Projecting to embedding...")
        embedding_flow = project_to_embedding(
            adata, delta_velocity, basis=basis,
            n_neighbors=n_neighbors, n_jobs=n_jobs
        )

    else:  # difference method
        if verbose:
            print("Projecting expression difference to embedding...")
        embedding_flow = project_to_embedding(
            adata, delta_X, basis=basis,
            n_neighbors=n_neighbors, n_jobs=n_jobs
        )

    # Store results
    adata.obsm[f'ode_perturbation_flow_{basis}'] = embedding_flow
    adata.uns['ode_perturbation_flow_params'] = {
        'basis': basis,
        'method': method,
        'time_point': time_point,
        'clusters': list(wt_trajectories.keys())
    }

    if verbose:
        print(f"ODE perturbation flow stored in adata.obsm['ode_perturbation_flow_{basis}']")

    return embedding_flow