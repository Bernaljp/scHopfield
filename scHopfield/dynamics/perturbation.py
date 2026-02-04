"""
Perturbation simulation using GRN signal propagation.

This module implements CellOracle-style perturbation simulations using
the scHopfield GRN framework. It simulates how gene expression changes
propagate through the inferred gene regulatory network.

References
----------
Logic for the transition vector field is inspired by the perturbation
simulation workflow in CellOracle:
Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
from anndata import AnnData
from tqdm.auto import tqdm

from .._utils.math import sigmoid
from .._utils.io import get_matrix, to_numpy, get_genes_used


def simulate_perturbation(
    adata: AnnData,
    perturb_condition: Dict[str, float],
    cluster_key: str = 'cell_type',
    n_propagation: int = 3,
    dt: float = 0.1,
    use_cluster_specific_GRN: bool = True,
    clip_delta_X: bool = True,
    store_intermediate: bool = False,
    verbose: bool = True
) -> AnnData:
    """
    Simulate gene expression changes after perturbation using GRN signal propagation.

    This function implements a CellOracle-style simulation where:
    1. Initial perturbation is applied to specific genes
    2. Signal propagates through the GRN iteratively
    3. Final simulated expression and delta_X are computed

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions (W matrices)
    perturb_condition : dict
        Perturbation conditions as {gene_name: value}.
        Examples:
        - Knockout: {"Gata1": 0.0}
        - Overexpression: {"Gata1": 5.0}
        - Multiple: {"Gata1": 0.0, "Tal1": 2.0}
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    n_propagation : int, optional (default: 3)
        Number of signal propagation steps through the GRN.
        Higher values capture more indirect effects but may add noise.
    dt : float, optional (default: 0.1)
        Time step for Euler integration in signal propagation.
        Smaller values give more accurate but slower simulations.
    use_cluster_specific_GRN : bool, optional (default: True)
        If True, uses cluster-specific W matrices.
        If False, uses the 'all' W matrix for all cells.
    clip_delta_X : bool, optional (default: True)
        If True, clips simulated values to the observed expression range
        to avoid out-of-distribution predictions.
    store_intermediate : bool, optional (default: False)
        If True, stores intermediate propagation steps.
    verbose : bool, optional (default: True)
        Whether to show progress information.

    Returns
    -------
    AnnData
        Modified adata with added layers:
        - 'simulated_count': Simulated gene expression after perturbation
        - 'delta_X': Difference between simulated and original expression
        - 'perturbation_input': Expression values used as simulation input

        And added to adata.uns['scHopfield']:
        - 'perturb_condition': The perturbation conditions used
        - 'n_propagation': Number of propagation steps

    References
    ----------
    Logic for the transition vector field is inspired by the perturbation
    simulation workflow in CellOracle:
    Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9

    Examples
    --------
    >>> import scHopfield as sch
    >>> # Knockout simulation
    >>> sch.dyn.simulate_perturbation(adata, {"Gata1": 0.0})
    >>> # Overexpression
    >>> sch.dyn.simulate_perturbation(adata, {"Gata1": 5.0})
    >>> # Check results
    >>> delta = adata.layers['delta_X']
    """
    # Validate perturbation conditions
    _validate_perturb_condition(adata, perturb_condition, verbose=verbose)

    # Get gene indices used in scHopfield
    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values
    n_genes = len(genes)

    # Get base expression from spliced layer
    spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')
    base_expression = to_numpy(get_matrix(adata, spliced_key, genes=genes))

    # Initialize simulation input with base expression
    simulation_input = base_expression.copy()

    # Apply perturbation to initial conditions
    for gene, value in perturb_condition.items():
        if gene in gene_names:
            gene_idx = np.where(gene_names == gene)[0][0]
            simulation_input[:, gene_idx] = value

    # Store perturbation input
    _store_layer(adata, simulation_input, 'perturbation_input', genes)

    # Get clusters
    clusters = adata.obs[cluster_key].unique()

    # Initialize output array
    simulated = np.zeros_like(base_expression)

    # Store intermediate results if requested
    if store_intermediate:
        intermediates = [simulation_input.copy()]

    # Run simulation for each cluster
    if verbose:
        cluster_iter = tqdm(clusters, desc="Simulating perturbation")
    else:
        cluster_iter = clusters

    for cluster in cluster_iter:
        # Get cells in this cluster
        cluster_mask = (adata.obs[cluster_key] == cluster).values
        n_cells_cluster = cluster_mask.sum()

        if n_cells_cluster == 0:
            continue

        # Get cluster-specific or global W matrix
        if use_cluster_specific_GRN and f'W_{cluster}' in adata.varp:
            W = adata.varp[f'W_{cluster}']
            I = adata.var[f'I_{cluster}'].values[genes] if f'I_{cluster}' in adata.var else np.zeros(n_genes)
        elif 'W_all' in adata.varp:
            W = adata.varp['W_all']
            I = adata.var['I_all'].values[genes] if 'I_all' in adata.var else np.zeros(n_genes)
        else:
            raise ValueError(f"No W matrix found for cluster '{cluster}'. Run fit_interactions first.")

        # Get sigmoid parameters
        threshold = adata.var['sigmoid_threshold'].values[genes]
        exponent = adata.var['sigmoid_exponent'].values[genes]

        # Get gamma (decay rate)
        if 'gamma' in adata.var.columns:
            gamma = adata.var['gamma'].values[genes]
        else:
            # Default gamma = 1 if not fitted
            gamma = np.ones(n_genes)

        # Get expression for this cluster
        X_cluster = simulation_input[cluster_mask, :]
        X_original = base_expression[cluster_mask, :]

        # Run signal propagation
        X_simulated = _propagate_signal(
            X=X_cluster,
            W=W,
            I=I,
            gamma=gamma,
            threshold=threshold,
            exponent=exponent,
            n_propagation=n_propagation,
            perturb_condition=perturb_condition,
            gene_names=gene_names,
            dt=dt
        )

        # Store results
        simulated[cluster_mask, :] = X_simulated

    # Clip to observed range if requested
    if clip_delta_X:
        min_vals = base_expression.min(axis=0)
        max_vals = base_expression.max(axis=0)
        simulated = np.clip(simulated, min_vals, max_vals)

    # Compute delta_X
    delta_X = simulated - base_expression

    # Store results
    _store_layer(adata, simulated, 'simulated_count', genes)
    _store_layer(adata, delta_X, 'delta_X', genes)

    # Store metadata
    if 'scHopfield' not in adata.uns:
        adata.uns['scHopfield'] = {}
    adata.uns['scHopfield']['perturb_condition'] = perturb_condition
    adata.uns['scHopfield']['n_propagation'] = n_propagation

    if verbose:
        print(f"✓ Perturbation simulation complete")
        print(f"  Genes perturbed: {list(perturb_condition.keys())}")
        print(f"  Propagation steps: {n_propagation}")
        print(f"  Results stored in adata.layers['simulated_count'] and adata.layers['delta_X']")

    return adata


def _propagate_signal(
    X: np.ndarray,
    W: np.ndarray,
    I: np.ndarray,
    gamma: np.ndarray,
    threshold: np.ndarray,
    exponent: np.ndarray,
    n_propagation: int,
    perturb_condition: Dict[str, float],
    gene_names: np.ndarray,
    dt: float = 0.1
) -> np.ndarray:
    """
    Propagate signal through GRN using iterative updates.

    The update rule is based on the scHopfield dynamics:
    dx/dt = W @ sigmoid(x) - gamma * x + I

    Applied iteratively with Euler integration to simulate signal propagation.

    Parameters
    ----------
    X : np.ndarray
        Initial expression matrix (n_cells, n_genes)
    W : np.ndarray
        Interaction matrix (n_genes, n_genes)
    I : np.ndarray
        Bias vector (n_genes,)
    gamma : np.ndarray
        Decay rate vector (n_genes,)
    threshold : np.ndarray
        Sigmoid threshold parameters
    exponent : np.ndarray
        Sigmoid exponent parameters
    n_propagation : int
        Number of propagation steps
    perturb_condition : dict
        Perturbation conditions
    gene_names : np.ndarray
        Gene names for indexing
    dt : float, optional (default: 0.1)
        Time step for Euler integration

    Returns
    -------
    np.ndarray
        Simulated expression after propagation
    """
    X_current = X.copy()

    # Get indices of perturbed genes
    perturb_indices = {}
    for gene, value in perturb_condition.items():
        if gene in gene_names:
            idx = np.where(gene_names == gene)[0][0]
            perturb_indices[idx] = value

    for step in range(n_propagation):
        # Compute sigmoid activation
        sig = sigmoid(X_current, threshold[None, :], exponent[None, :])
        sig = np.nan_to_num(sig)

        # Compute dx/dt = W @ sigmoid(x) - gamma * x + I
        # Note: Using sig @ W.T because W[i,j] represents effect of gene j on gene i
        dxdt = sig @ W.T - gamma[None, :] * X_current + I[None, :]

        # Euler update: X_new = X + dt * dx/dt
        X_new = X_current + dt * dxdt

        # Enforce perturbation conditions (keep perturbed genes fixed)
        for idx, value in perturb_indices.items():
            X_new[:, idx] = value

        # Ensure non-negative
        X_current = np.maximum(X_new, 0)

    return X_current


def _validate_perturb_condition(
    adata: AnnData,
    perturb_condition: Dict[str, float],
    verbose: bool = True
) -> None:
    """Validate perturbation conditions."""

    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values

    for gene, value in perturb_condition.items():
        # Check gene exists
        if gene not in adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.var_names")

        # Check gene is in scHopfield analysis
        if gene not in gene_names:
            raise ValueError(f"Gene '{gene}' was not included in scHopfield analysis. "
                           f"Check adata.var['scHopfield_used']")

        # Check value is non-negative
        if value < 0:
            raise ValueError(f"Perturbation value must be non-negative. Got {value} for '{gene}'")

        # Warn if value is far from observed range
        gene_idx = np.where(gene_names == gene)[0][0]
        spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')
        expr = to_numpy(get_matrix(adata, spliced_key, genes=[genes[gene_idx]])).flatten()

        min_val, max_val = expr.min(), expr.max()
        if value < min_val * 0.5 or value > max_val * 2:
            if verbose:
                print(f"  Warning: Perturbation value {value} for '{gene}' is outside "
                      f"typical range [{min_val:.2f}, {max_val:.2f}]")


def _store_layer(
    adata: AnnData,
    data: np.ndarray,
    layer_name: str,
    gene_indices: np.ndarray
) -> None:
    """Store data as a layer, expanding to full gene space."""
    full_data = np.zeros((adata.n_obs, adata.n_vars), dtype=data.dtype)
    full_data[:, gene_indices] = data
    adata.layers[layer_name] = full_data


def calculate_perturbation_effect_scores(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    method: str = 'mean'
) -> pd.DataFrame:
    """
    Calculate perturbation effect scores per cluster.

    Summarizes the delta_X values by cluster to quantify the overall
    effect of the perturbation on each cell population.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with simulation results (delta_X layer)
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    method : str, optional (default: 'mean')
        How to summarize effects: 'mean', 'median', 'max', or 'norm'
        - 'mean': Mean absolute delta_X
        - 'median': Median absolute delta_X
        - 'max': Maximum absolute delta_X
        - 'norm': L2 norm of delta_X vector

    Returns
    -------
    pd.DataFrame
        DataFrame with clusters as index and genes as columns,
        containing the summarized perturbation effects.
    """
    if 'delta_X' not in adata.layers:
        raise ValueError("No simulation results found. Run simulate_perturbation first.")

    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values

    delta_X = adata.layers['delta_X'][:, genes]

    clusters = adata.obs[cluster_key].unique()
    results = {}

    for cluster in clusters:
        cluster_mask = (adata.obs[cluster_key] == cluster).values
        delta_cluster = delta_X[cluster_mask, :]

        if method == 'mean':
            score = np.abs(delta_cluster).mean(axis=0)
        elif method == 'median':
            score = np.median(np.abs(delta_cluster), axis=0)
        elif method == 'max':
            score = np.abs(delta_cluster).max(axis=0)
        elif method == 'norm':
            score = np.linalg.norm(delta_cluster, axis=0) / delta_cluster.shape[0]
        else:
            raise ValueError(f"Unknown method: {method}")

        results[cluster] = score

    return pd.DataFrame(results, index=gene_names).T


def calculate_cell_transition_scores(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    basis: str = 'umap'
) -> pd.DataFrame:
    """
    Calculate cell transition scores based on delta_X magnitude.

    This measures how much each cell's state changes due to perturbation,
    which can indicate cells most affected by the perturbation.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with simulation results
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    basis : str, optional (default: 'umap')
        Embedding basis for potential vector field visualization

    Returns
    -------
    pd.DataFrame
        DataFrame with cell-level transition scores
    """
    if 'delta_X' not in adata.layers:
        raise ValueError("No simulation results found. Run simulate_perturbation first.")

    genes = get_genes_used(adata)
    delta_X = adata.layers['delta_X'][:, genes]

    # Compute magnitude of change for each cell
    magnitude = np.linalg.norm(delta_X, axis=1)

    # Store in obs
    adata.obs['perturbation_magnitude'] = magnitude

    # Summarize by cluster
    summary = pd.DataFrame({
        'cluster': adata.obs[cluster_key].values,
        'magnitude': magnitude
    })

    cluster_summary = summary.groupby('cluster').agg(['mean', 'std', 'max']).round(4)
    cluster_summary.columns = ['mean_magnitude', 'std_magnitude', 'max_magnitude']

    return cluster_summary


def get_top_affected_genes(
    adata: AnnData,
    n_genes: int = 20,
    cluster: Optional[str] = None,
    cluster_key: str = 'cell_type',
    exclude_perturbed: bool = True
) -> pd.DataFrame:
    """
    Get the top genes most affected by the perturbation.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with simulation results
    n_genes : int, optional (default: 20)
        Number of top genes to return
    cluster : str, optional
        If specified, analyze only cells in this cluster
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    exclude_perturbed : bool, optional (default: True)
        If True, exclude the perturbed genes from the results

    Returns
    -------
    pd.DataFrame
        DataFrame with top affected genes and their mean delta_X values
    """
    if 'delta_X' not in adata.layers:
        raise ValueError("No simulation results found. Run simulate_perturbation first.")

    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values

    # Exclude perturbed genes if requested
    if exclude_perturbed and 'scHopfield' in adata.uns and 'perturb_condition' in adata.uns['scHopfield']:
        perturbed_genes = list(adata.uns['scHopfield']['perturb_condition'].keys())
        gene_mask = ~np.isin(gene_names, perturbed_genes)
        gene_names = gene_names[gene_mask]
        genes_filtered = genes[gene_mask]
    else:
        genes_filtered = genes

    if cluster is not None:
        mask = (adata.obs[cluster_key] == cluster).values
        delta_X = adata.layers['delta_X'][mask, :][:, genes_filtered]
    else:
        delta_X = adata.layers['delta_X'][:, genes_filtered]

    # Mean change per gene
    mean_delta = delta_X.mean(axis=0)
    abs_mean_delta = np.abs(mean_delta)

    # Get top genes
    top_idx = np.argsort(abs_mean_delta)[-n_genes:][::-1]

    df = pd.DataFrame({
        'gene': gene_names[top_idx],
        'mean_delta_X': mean_delta[top_idx],
        'abs_mean_delta_X': abs_mean_delta[top_idx],
        'direction': ['up' if d > 0 else 'down' for d in mean_delta[top_idx]]
    })

    return df


def compare_perturbations(
    adata: AnnData,
    perturbations: Union[Dict[str, Dict[str, float]], List[Dict[str, float]]],
    labels: Optional[List[str]] = None,
    cluster_key: str = 'cell_type',
    n_propagation: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple perturbation conditions.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    perturbations : dict or list
        Either:
        - Dict mapping labels to perturbation conditions: {"KO": {"Gata1": 0.0}, "OE": {"Gata1": 1.0}}
        - List of perturbation conditions (requires labels parameter)
    labels : list of str, optional
        Labels for each perturbation. Required if perturbations is a list.
        Ignored if perturbations is a dict (keys are used as labels).
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    n_propagation : int, optional (default: 3)
        Number of propagation steps
    verbose : bool, optional (default: True)
        Whether to show progress

    Returns
    -------
    pd.DataFrame
        DataFrame with genes as index and mean |delta_X| for each perturbation condition

    References
    ----------
    Logic for the transition vector field is inspired by the perturbation
    simulation workflow in CellOracle:
    Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
    """
    # Handle dict input: {label: perturbation_condition}
    if isinstance(perturbations, dict):
        labels = list(perturbations.keys())
        perturbations_list = list(perturbations.values())
    else:
        perturbations_list = perturbations
        if labels is None:
            labels = [f"perturb_{i+1}" for i in range(len(perturbations_list))]

    assert len(labels) == len(perturbations_list), "Number of labels must match perturbations"

    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values
    all_deltas = {}

    for label, perturb in zip(labels, perturbations_list):
        if verbose:
            print(f"\nRunning simulation for: {label}")
            print(f"  Condition: {perturb}")

        # Run simulation
        simulate_perturbation(
            adata, perturb,
            cluster_key=cluster_key,
            n_propagation=n_propagation,
            verbose=False
        )

        # Get mean |delta_X| per gene
        delta_X = adata.layers['delta_X'][:, genes]
        mean_abs_delta = np.abs(delta_X).mean(axis=0)
        all_deltas[label] = mean_abs_delta

    # Combine into DataFrame with genes as index
    result = pd.DataFrame(all_deltas, index=gene_names)

    # Sort by total effect across conditions
    result['_total'] = result.sum(axis=1)
    result = result.sort_values('_total', ascending=False)
    result = result.drop('_total', axis=1)

    return result
