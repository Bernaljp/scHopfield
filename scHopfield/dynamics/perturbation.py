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


def _propagate_signal(
    X_current: np.ndarray,
    X_original: np.ndarray,
    W: np.ndarray,
    source_indices: np.ndarray,
    threshold: np.ndarray,
    exponent: np.ndarray,
    dt: float = 1.0,
    x_min: float = 0.0,
    x_max: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Propagate signal through the GRN for one step.

    Computes the effect of source genes on all other genes using:

    x_i^new = x_i^current + dt * sum_k W_ik * (sigmoid_k(x_k^current) - sigmoid_k(x_k^original))

    Where k iterates over the source genes (source_indices).

    Parameters
    ----------
    X_current : np.ndarray
        Current expression matrix (n_cells, n_genes)
    X_original : np.ndarray
        Original expression matrix (n_cells, n_genes)
    W : np.ndarray
        Interaction matrix (n_genes, n_genes), W[i,k] = effect of gene k on gene i
    source_indices : np.ndarray
        Indices of source genes to propagate from
    threshold : np.ndarray
        Sigmoid threshold parameters for all genes
    exponent : np.ndarray
        Sigmoid exponent parameters for all genes
    dt : float, optional (default: 1.0)
        Scaling factor for the propagation step
    x_min : float, optional (default: 0.0)
        Minimum expression value (non-negative constraint)
    x_max : np.ndarray, optional
        Maximum expression values per gene. If None, no upper bound.

    Returns
    -------
    np.ndarray
        Updated expression matrix after one propagation step
    """
    # Compute sigmoid of current expression for source genes
    sig_current = sigmoid(
        X_current[:, source_indices],
        threshold[source_indices],
        exponent[source_indices]
    )

    # Compute sigmoid of original expression for source genes
    sig_original = sigmoid(
        X_original[:, source_indices],
        threshold[source_indices],
        exponent[source_indices]
    )

    # Compute delta sigmoid: sigmoid(x^current) - sigmoid(x^original)
    delta_sig = sig_current - sig_original  # (n_cells, n_source)

    # Get W columns for source genes: W[:, source_indices]
    W_source = W[:, source_indices]  # (n_genes, n_source)

    # Compute delta_X for this step:
    # delta_X_i = dt * sum_k W[i,k] * delta_sig[k]
    delta_X_step = dt * (delta_sig @ W_source.T)  # (n_cells, n_genes)

    # Update expression
    X_new = X_current + delta_X_step

    # Clip to valid range (prevents divergence)
    X_new = np.maximum(X_new, x_min)
    if x_max is not None:
        X_new = np.minimum(X_new, x_max)

    return X_new


def _get_tf_indices(W: np.ndarray) -> np.ndarray:
    """
    Get indices of transcription factors (genes with outgoing edges in GRN).

    Parameters
    ----------
    W : np.ndarray
        Interaction matrix (n_genes, n_genes), W[i,k] = effect of gene k on gene i

    Returns
    -------
    np.ndarray
        Indices of genes that have at least one non-zero outgoing edge
    """
    # TFs are genes that regulate at least one other gene (non-zero column sum)
    has_targets = np.abs(W).sum(axis=0) > 0
    return np.where(has_targets)[0]


def simulate_perturbation(
    adata: AnnData,
    perturb_condition: Dict[str, float],
    cluster_key: str = 'cell_type',
    target_clusters: Optional[List[str]] = None,
    n_propagation: int = 3,
    dt: float = 1.0,
    use_cluster_specific_GRN: bool = True,
    clip_delta_X: bool = True,
    x_max_percentile: float = 99.0,
    verbose: bool = True
) -> AnnData:
    """
    Simulate gene expression changes after perturbation using direct GRN effects.

    Computes the effect of perturbed genes on all other genes using iterative
    signal propagation:

    x_i^new = x_i^current + dt * sum_k W_ik * (sigmoid_k(x_k^current) - sigmoid_k(x_k^original))

    The propagation works as follows:
    - Step 1: Only the manually perturbed genes propagate their effects
    - Steps 2+: All TFs (genes with outgoing edges in the GRN) that have changed
      from their original state propagate their effects

    This captures the cascade where perturbed genes affect other TFs, which
    then also contribute to further propagation through the network.

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
    target_clusters : list of str, optional
        List of cluster names to simulate perturbation in.
        If None, simulates in all clusters.
        Cells not in target clusters will have delta_X = 0.
    n_propagation : int, optional (default: 3)
        Number of signal propagation steps through the GRN.
        Higher values capture more indirect effects.
    dt : float, optional (default: 1.0)
        Scaling factor for each propagation step.
    use_cluster_specific_GRN : bool, optional (default: True)
        If True, uses cluster-specific W matrices.
        If False, uses the 'all' W matrix for all cells.
    clip_delta_X : bool, optional (default: True)
        If True, clips final simulated values to the observed expression range
        to avoid out-of-distribution predictions.
    x_max_percentile : float, optional (default: 99.0)
        Percentile of expression to use as upper bound during propagation.
        This prevents divergence by clipping values at each step.
        Set to None to disable step-wise upper bound clipping.
    verbose : bool, optional (default: True)
        Whether to show progress information.

    Returns
    -------
    AnnData
        Modified adata with added layers:
        - 'simulated_count': Simulated gene expression after perturbation
        - 'delta_X': Difference between simulated and original expression

        And added to adata.uns['scHopfield']:
        - 'perturb_condition': The perturbation conditions used
        - 'n_propagation': Number of propagation steps
        - 'dt': Scaling factor used

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

    # Get sigmoid parameters
    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]

    # Compute expression bounds for stability
    x_min = 0.0
    if x_max_percentile is not None:
        # Use percentile to avoid outliers setting unreasonable bounds
        x_max = np.percentile(base_expression, x_max_percentile, axis=0)
        # Add margin to allow some growth beyond observed values
        x_max = x_max * 2.0
    else:
        x_max = None

    # Get indices and values of perturbed genes
    perturb_indices = []
    perturb_values = []
    for gene, value in perturb_condition.items():
        if gene in gene_names:
            idx = np.where(gene_names == gene)[0][0]
            perturb_indices.append(idx)
            perturb_values.append(value)

    perturb_indices = np.array(perturb_indices)
    perturb_values = np.array(perturb_values)

    # Get clusters to simulate
    all_clusters = adata.obs[cluster_key].unique()
    if target_clusters is not None:
        # Validate target clusters
        invalid_clusters = set(target_clusters) - set(all_clusters)
        if invalid_clusters:
            raise ValueError(f"Target clusters not found in data: {invalid_clusters}")
        clusters = [c for c in target_clusters if c in all_clusters]
        if verbose:
            print(f"Simulating perturbation in {len(clusters)} target clusters: {clusters}")
    else:
        clusters = all_clusters

    # Initialize simulated array with base expression
    simulated = base_expression.copy()

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
        elif 'W_all' in adata.varp:
            W = adata.varp['W_all']
        else:
            raise ValueError(f"No W matrix found for cluster '{cluster}'. Run fit_interactions first.")

        # Get TF indices for this cluster (genes with outgoing edges)
        tf_indices = _get_tf_indices(W)

        # Get expression for this cluster
        X_current = simulated[cluster_mask, :].copy()
        X_original = base_expression[cluster_mask, :].copy()

        # Apply initial perturbation: set perturbed genes to their target values
        X_current[:, perturb_indices] = perturb_values[None, :]

        # Iterative propagation
        for step in range(n_propagation):
            if step == 0:
                # First step: only propagate from manually perturbed genes
                source_indices = perturb_indices
            else:
                # Subsequent steps: propagate from all TFs
                source_indices = tf_indices

            # Propagate signal
            X_current = _propagate_signal(
                X_current=X_current,
                X_original=X_original,
                W=W,
                source_indices=source_indices,
                threshold=threshold,
                exponent=exponent,
                dt=dt,
                x_min=x_min,
                x_max=x_max
            )

            # Keep perturbed genes fixed at their perturbed values
            X_current[:, perturb_indices] = perturb_values[None, :]

        # Store results
        simulated[cluster_mask, :] = X_current

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
    adata.uns['scHopfield']['dt'] = dt

    if verbose:
        print(f"Perturbation simulation complete")
        print(f"  Genes perturbed: {list(perturb_condition.keys())}")
        print(f"  Propagation steps: {n_propagation}")
        print(f"  dt (scaling): {dt}")
        print(f"  Results stored in adata.layers['simulated_count'] and adata.layers['delta_X']")

    return adata


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
    target_clusters: Optional[List[str]] = None,
    n_propagation: int = 3,
    dt: float = 1.0,
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
    target_clusters : list of str, optional
        List of cluster names to simulate perturbation in.
        If None, simulates in all clusters.
    n_propagation : int, optional (default: 3)
        Number of propagation steps
    dt : float, optional (default: 1.0)
        Scaling factor for each propagation step
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
            target_clusters=target_clusters,
            n_propagation=n_propagation,
            dt=dt,
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
