"""Simulation utilities for gene regulatory network dynamics."""

import numpy as np
from typing import Optional, List
from anndata import AnnData

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
    verbose: bool = False
) -> np.ndarray:
    """
    Simulate trajectory with gene perturbations using ODE integration.

    This simulates the full ODE dynamics with perturbed initial conditions.
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
        Can be:
        - Fold changes: {"Gata1": 0.0} for knockout, {"Gata1": 2.0} for 2x overexpression
        - Absolute values: {"Gata1": 5.0} sets expression to 5.0
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

    # Apply perturbations
    for gene_name, value in gene_perturbations.items():
        if gene_name in gene_names:
            gene_idx = np.where(gene_names == gene_name)[0][0]
            if value == 0:
                # Knockout: set to 0
                x0[gene_idx] = 0
            elif value < 0:
                raise ValueError(f"Perturbation value must be non-negative, got {value} for {gene_name}")
            else:
                # Could be fold change or absolute value
                # If value < 10, treat as fold change; otherwise absolute
                # (This is a heuristic - user should be explicit)
                x0[gene_idx] = value

    solver = create_solver(
        adata, cluster, degradation_key,
        spliced_key=spliced_key,
        x_max_percentile=x_max_percentile
    )

    if verbose:
        print(f"Simulating perturbation for cell {cell_idx} in cluster '{cluster}'")
        print(f"  Perturbations: {gene_perturbations}")
        print(f"  Method: {method}")

    trajectory = solver.solve(x0, t_span, method=method, clip_each_step=True)

    return trajectory


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