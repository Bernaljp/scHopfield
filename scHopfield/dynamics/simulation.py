"""Simulation utilities for gene regulatory network dynamics."""

import numpy as np
from typing import Optional, List, Dict
from anndata import AnnData
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .solver import create_solver
from .._utils.io import get_matrix, to_numpy, get_genes_used
from ._utils import _parse_perturb_genes, _update_scHopfield_uns


def simulate_trajectory(
    adata: AnnData,
    cluster: str,
    cell_idx: int,
    t_span: np.ndarray,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    method: str = 'euler',
    x_max_percentile: float = 99.0,
    verbose: bool = False
) -> np.ndarray:
    """
    Simulate trajectory from a cell's initial state.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    cluster : str
        Cluster name
    cell_idx : int
        Index of cell to use as initial condition
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
    verbose : bool, optional (default: False)
        Print simulation info

    Returns
    -------
    np.ndarray
        Simulated trajectory (len(t_span) × n_genes)
    """
    genes = get_genes_used(adata)
    x0 = to_numpy(get_matrix(adata, spliced_key, genes=genes)[cell_idx])

    # Ensure non-negative initial condition
    x0 = np.maximum(x0, 0)

    solver = create_solver(
        adata, cluster, degradation_key,
        spliced_key=spliced_key,
        x_max_percentile=x_max_percentile
    )

    if verbose:
        print(f"Simulating trajectory for cell {cell_idx} in cluster '{cluster}'")
        print(f"  Method: {method}")
        print(f"  Time span: {t_span[0]:.2f} to {t_span[-1]:.2f} ({len(t_span)} points)")
        if solver.x_max is not None:
            print(f"  Upper bound: {x_max_percentile}th percentile × 2")

    trajectory = solver.solve(x0, t_span, method=method, clip_each_step=True)

    if verbose:
        print(f"  Final values range: [{trajectory[-1].min():.3f}, {trajectory[-1].max():.3f}]")

    return trajectory


def simulate_perturbation_ode(
    adata: AnnData,
    cluster: str,
    cell_idx: int,
    gene_perturbations: dict,
    t_span: np.ndarray,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    method: str = 'euler',
    x_max_percentile: float = 99.0,
    residual_gene_dynamics: bool = False,
    verbose: bool = False
) -> np.ndarray:
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
    cell_idx : int
        Cell index for initial condition
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
    verbose : bool, optional (default: False)
        Print simulation info

    Returns
    -------
    np.ndarray
        Simulated trajectory with perturbations
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    x0 = to_numpy(get_matrix(adata, spliced_key, genes=genes)[cell_idx])
    x0 = np.maximum(x0, 0)  # Ensure non-negative

    # Collect perturbation indices and values
    all_indices, all_values = _parse_perturb_genes(
        gene_names, gene_perturbations, validate_non_negative=True
    )
    if len(all_indices) > 0:
        x0[all_indices] = all_values
    if not residual_gene_dynamics and len(all_indices) > 0:
        fixed_indices, fixed_values = all_indices, all_values
    else:
        fixed_indices, fixed_values = None, None

    solver = create_solver(
        adata, cluster, degradation_key,
        spliced_key=spliced_key,
        x_max_percentile=x_max_percentile
    )

    # Set fixed genes (they won't change during simulation) unless residual dynamics allowed
    if not residual_gene_dynamics:
        solver.set_fixed_genes(fixed_indices, fixed_values)

    if verbose:
        print(f"Simulating perturbation for cell {cell_idx} in cluster '{cluster}'")
        print(f"  Perturbations: {gene_perturbations}")
        if residual_gene_dynamics:
            print(f"  Perturbed genes: can evolve (residual_gene_dynamics=True)")
        else:
            print(f"  Perturbed genes: held constant")
        print(f"  Method: {method}")

    trajectory = solver.solve(x0, t_span, method=method, clip_each_step=True)

    return trajectory

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
    verbose: bool = False
) -> 'AnnData':
    """
    Simulate dataset-wide trajectory shifts with gene perturbations using ODE integration.

    This function mimics the propagation-based `simulate_shift` but uses continuous
    ODE integration. It calculates the final state for every cell after a time `dt`
    under the perturbed conditions, and stores the resulting shift (delta_X).

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
        Integration method ('euler', 'odeint', 'RK45', etc.)
    use_cluster_specific_GRN : bool, optional (default: True)
        If True, uses cluster-specific solvers. If False, uses a global solver.
    x_max_percentile : float, optional (default: 99.0)
        Percentile for upper bound. Prevents divergence.
    residual_gene_dynamics : bool, optional (default: False)
        If False, perturbed genes are held fixed. If True, they evolve.
    n_jobs : int, optional (default: -1)
        Number of parallel jobs for cell simulation. -1 uses all available cores.
        Uses threads (not processes) so there is no pickling overhead.
    verbose : bool, optional (default: False)
        Print simulation progress.

    Returns
    -------
    AnnData
        A copy of the input AnnData with 'simulated_count' and 'delta_X' added to layers.
    """
    adata_out = adata.copy()
    
    # Identify used genes
    genes_mask = get_genes_used(adata_out)
    gene_names = adata_out.var_names[genes_mask]
    
    # Get initial states
    X_orig = to_numpy(get_matrix(adata_out, spliced_key, genes=genes_mask))
    X_sim = np.zeros_like(X_orig)
    V_sim = np.zeros_like(X_orig)  # Placeholder for velocity
    
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

        def _simulate_cell(x0_row):
            x0 = np.maximum(x0_row, 0)
            if len(all_indices) > 0:
                x0[all_indices] = all_values
            trajectory = solver.solve(x0, t_span, method=method, clip_each_step=True)
            x_final = trajectory[-1]
            return x_final, solver.dynamics(x_final, 0.0)

        iterator = tqdm(cell_indices, desc=f"Cells in {cluster}") if verbose else cell_indices

        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_simulate_cell)(X_orig[idx].copy()) for idx in iterator
        )

        for i, idx in enumerate(cell_indices):
            X_sim[idx], V_sim[idx] = results[i]


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