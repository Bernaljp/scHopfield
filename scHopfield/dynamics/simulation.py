"""Simulation utilities for gene regulatory network dynamics."""

import numpy as np
from typing import Optional, List, Dict
from anndata import AnnData
from tqdm.auto import tqdm

from .solver import create_solver
from .._utils.io import get_matrix, to_numpy, get_genes_used


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
    fixed_indices = []
    fixed_values = []

    for gene_name, value in gene_perturbations.items():
        if gene_name in gene_names:
            gene_idx = np.where(gene_names == gene_name)[0][0]
            if value < 0:
                raise ValueError(f"Perturbation value must be non-negative, got {value} for {gene_name}")
            # Set initial condition
            x0[gene_idx] = value
            # Mark as fixed (only if not allowing residual dynamics)
            if not residual_gene_dynamics:
                fixed_indices.append(gene_idx)
                fixed_values.append(value)

    fixed_indices = np.array(fixed_indices) if fixed_indices else None
    fixed_values = np.array(fixed_values) if fixed_values else None

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
    verbose : bool, optional (default: False)
        Print simulation progress.

    Returns
    -------
    AnnData
        A copy of the input AnnData with 'simulated_X' and 'delta_X' added to layers.
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
        fixed_indices = []
        fixed_values = []
        
        for gene_name, value in perturb_condition.items():
            if gene_name in gene_names:
                gene_idx = np.where(gene_names == gene_name)[0][0]
                if value < 0:
                    raise ValueError(f"Perturbation value must be non-negative, got {value} for {gene_name}")
                if not residual_gene_dynamics:
                    fixed_indices.append(gene_idx)
                    fixed_values.append(value)
                    
        fixed_indices = np.array(fixed_indices) if fixed_indices else None
        fixed_values = np.array(fixed_values) if fixed_values else None

        if not residual_gene_dynamics:
            solver.set_fixed_genes(fixed_indices, fixed_values)

        # Optional progress bar
        iterator = tqdm(cell_indices, desc=f"Cells in {cluster}") if verbose else cell_indices

        # Simulate for each cell in the cluster
        for idx in iterator:
            x0 = X_orig[idx].copy()
            x0 = np.maximum(x0, 0)
            
            # Apply initial perturbation state
            for gene_name, value in perturb_condition.items():
                if gene_name in gene_names:
                    gene_idx = np.where(gene_names == gene_name)[0][0]
                    x0[gene_idx] = value
                    
            # Integrate ODE
            trajectory = solver.solve(x0, t_span, method=method, clip_each_step=True)
            
            # Store final state
            X_sim[idx] = trajectory[-1]
            V_sim[idx] = solver.dynamics(trajectory[-1])  # Final velocity


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
    if 'scHopfield' not in adata_out.uns:
        adata_out.uns['scHopfield'] = {}
        
    adata_out.uns['scHopfield']['perturb_condition'] = perturb_condition
    adata_out.uns['scHopfield']['simulation_method'] = 'ODE'
    adata_out.uns['scHopfield']['ode_dt'] = dt

    return adata_out

# Keep old name for backwards compatibility
def simulate_perturbation(
    adata: AnnData,
    cluster: str,
    cell_idx: int,
    gene_perturbations: dict,
    t_span: np.ndarray,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma'
) -> np.ndarray:
    """
    Simulate trajectory with gene perturbations (legacy function).

    DEPRECATED: Use simulate_perturbation_ode for more control,
    or sch.dyn.simulate_perturbation for CellOracle-style simulation.
    """
    return simulate_perturbation_ode(
        adata, cluster, cell_idx, gene_perturbations, t_span,
        spliced_key=spliced_key, degradation_key=degradation_key,
        method='euler', x_max_percentile=99.0
    )