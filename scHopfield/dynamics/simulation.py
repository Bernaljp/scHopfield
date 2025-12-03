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
    degradation_key: str = 'gamma'
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

    Returns
    -------
    np.ndarray
        Simulated trajectory (len(t_span) × n_genes)
    """
    genes = get_genes_used(adata)
    x0 = to_numpy(get_matrix(adata, spliced_key, genes=genes)[cell_idx])

    solver = create_solver(adata, cluster, degradation_key)
    trajectory = solver.solve(x0, t_span)

    return trajectory


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
    Simulate trajectory with gene perturbations.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    cluster : str
        Cluster name
    cell_idx : int
        Cell index for initial condition
    gene_perturbations : dict
        Dictionary mapping gene names to perturbation values (fold changes)
    t_span : np.ndarray
        Time points
    spliced_key : str, optional
        Expression data key
    degradation_key : str, optional
        Degradation rates key

    Returns
    -------
    np.ndarray
        Simulated trajectory with perturbations
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    x0 = to_numpy(get_matrix(adata, spliced_key, genes=genes)[cell_idx])

    # Apply perturbations
    for gene_name, fold_change in gene_perturbations.items():
        if gene_name in gene_names:
            gene_idx = np.where(gene_names == gene_name)[0][0]
            x0[gene_idx] *= fold_change

    solver = create_solver(adata, cluster, degradation_key)
    trajectory = solver.solve(x0, t_span)

    return trajectory
