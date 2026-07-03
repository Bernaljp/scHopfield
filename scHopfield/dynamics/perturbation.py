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
from ._utils import _parse_perturb_genes, _get_W_matrix, _compute_x_bounds, _update_scHopfield_uns
from ..tools.perturbation_analysis import compute_perturbation_flow_bias, compute_cluster_effects


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
    residual_gene_dynamics: bool = False,
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
    residual_gene_dynamics : bool, optional (default: False)
        If False, perturbed genes are held fixed at their perturbed values
        throughout all propagation steps.
        If True, perturbed genes can change according to the GRN dynamics
        after the initial perturbation is applied.
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

    # Get base expression from spliced layer
    spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')
    base_expression = to_numpy(get_matrix(adata, spliced_key, genes=genes))

    # Get sigmoid parameters
    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]

    # Compute expression bounds for stability
    x_min, x_max = _compute_x_bounds(base_expression, x_max_percentile, multiplier=2.0)

    # Get indices and values of perturbed genes
    perturb_indices, perturb_values = _parse_perturb_genes(gene_names, perturb_condition)

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
        W = _get_W_matrix(adata, cluster, use_cluster_specific=use_cluster_specific_GRN)

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

            # Keep perturbed genes fixed at their perturbed values (unless residual dynamics allowed)
            if not residual_gene_dynamics:
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
    _update_scHopfield_uns(adata, perturb_condition=perturb_condition,
                           n_propagation=n_propagation, dt=dt)

    if verbose:
        print("Perturbation simulation complete")
        print(f"  Genes perturbed: {list(perturb_condition.keys())}")
        print(f"  Propagation steps: {n_propagation}")
        print(f"  dt (scaling): {dt}")
        if residual_gene_dynamics:
            print("  Perturbed genes: can evolve (residual_gene_dynamics=True)")
        else:
            print("  Perturbed genes: held constant")
        print("  Results stored in adata.layers['simulated_count'] and adata.layers['delta_X']")

    return adata


def _validate_perturb_condition(
    adata: AnnData,
    perturb_condition: Dict[str, float],
    verbose: bool = True
) -> None:
    """Validate perturbation conditions."""

    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}

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
        gene_idx = gene_to_idx[gene]
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
        DataFrame with genes as index and mean ``|delta_X|`` for each perturbation condition

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

        # Get mean ``|delta_X|`` per gene
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


# ---------------------------------------------------------------------------
# High-level KO screen helpers
# ---------------------------------------------------------------------------


def run_ko_screen(
    adata: AnnData,
    genes: List[str],
    lineage_A_clusters: List[str],
    lineage_B_clusters: List[str],
    basis: str,
    wt_flow_key: str,
    cluster_key: str = 'cell_type',
    cluster_order: Optional[List[str]] = None,
    simulate_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, pd.Series]]:
    """
    Run a single-gene KO screen and compute lineage bias + cluster effects.

    For each gene in ``genes``, performs an ODE-based KO simulation
    (``simulate_shift_ode``), then computes:

    - **lineage bias** via :func:`~scHopfield.tools.compute_lineage_bias`
    - **cluster effects** via :func:`~scHopfield.tools.compute_cluster_effects`

    Parameters
    ----------
    adata : AnnData
        Base (WT) AnnData with fitted model. Each gene is simulated on a
        copy so the original object is not modified.
    genes : list of str
        Gene names to screen. Genes absent from ``adata.var_names`` are skipped.
    lineage_A_clusters : list of str
        Cluster names for lineage A (e.g. erythroid).
    lineage_B_clusters : list of str
        Cluster names for lineage B (e.g. myeloid).
    basis : str
        Embedding basis for flow projection (e.g. ``'draw_graph_fa'``).
    wt_flow_key : str
        Key in ``adata.obsm`` for the pre-computed WT Hopfield velocity.
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels.
    cluster_order : list of str, optional
        Ordered cluster names for ``compute_cluster_effects``.
        If None, uses ``adata.obs[cluster_key].unique()``.
    simulate_kwargs : dict, optional
        Extra keyword arguments forwarded to ``simulate_shift_ode``.
        Defaults: ``dt=5.0, n_steps=100, use_cluster_specific_GRN=True, n_jobs=-1``.
    verbose : bool, optional (default: True)
        Print progress for each gene.

    Returns
    -------
    bias_dict : dict[str, dict]
        ``{gene: {'score_A', 'score_B', 'lineage_bias'}}``
    effects_dict : dict[str, pd.Series]
        ``{gene: pd.Series(mean ``|delta_X|`` per cluster)}``

    Examples
    --------
    >>> bias, effects = sch.dyn.run_ko_screen(
    ...     adata, CANDIDATES, ERYTHROID, MYELOID,
    ...     basis='draw_graph_fa', wt_flow_key='original_velocity_flow_draw_graph_fa',
    ...     cluster_key='paul15_clusters', cluster_order=CLUSTER_ORDER,
    ... )
    >>> bias_df = pd.DataFrame(bias).T.sort_values('lineage_bias', ascending=False)
    """
    from .simulation import simulate_shift_ode

    if simulate_kwargs is None:
        simulate_kwargs = {}
    sim_kw = dict(
        cluster_key=cluster_key,
        dt=5.0,
        n_steps=100,
        use_cluster_specific_GRN=True,
        n_jobs=-1,
        verbose=False,
    )
    sim_kw.update(simulate_kwargs)

    if cluster_order is None:
        cluster_order = list(adata.obs[cluster_key].unique())

    bias_dict    = {}
    effects_dict = {}

    for gene in genes:
        if gene not in adata.var_names:
            if verbose:
                print(f"  Skip {gene}: not in adata")
            continue
        if verbose:
            print(f"  KO: {gene}...")
        adata_ko = simulate_shift_ode(
            adata.copy(),
            perturb_condition={gene: 0.0},
            **sim_kw,
        )
        bias_dict[gene]    = compute_perturbation_flow_bias(
            adata_ko, adata,
            lineage_A_clusters, lineage_B_clusters,
            basis, wt_flow_key,
            cluster_key=cluster_key,
        )
        effects_dict[gene] = compute_cluster_effects(
            adata_ko, cluster_order, cluster_key=cluster_key
        )

    if verbose:
        print(f"\nCompleted {len(bias_dict)} single KOs.")

    return bias_dict, effects_dict


def score_ko_panel(
    adata: AnnData,
    panel: Dict[str, int],
    lineage_A_clusters: List[str],
    lineage_B_clusters: List[str],
    basis: str,
    wt_flow_key: str,
    cluster_key: str = 'cell_type',
    simulate_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """Score a known-driver KO panel by the *direction* of the predicted shift.

    Ground-truth-anchored validation used across the paper's KO analyses. Given a
    panel of literature-established regulators annotated with the expected sign of
    their KO's lineage bias (``+1`` = KO should bias toward lineage A, ``-1`` =
    toward lineage B), this runs :func:`run_ko_screen` and reports whether each
    predicted ``lineage_bias`` sign matches expectation, plus the overall
    directional accuracy. This replaces the per-analysis scoring loops that were
    duplicated across the hematopoiesis and neural-crest KO panels.

    Parameters
    ----------
    adata
        Base (WT) AnnData with a fitted model and a precomputed WT flow at
        ``adata.obsm[wt_flow_key]`` (see :func:`~scHopfield.tools.calculate_flow`).
    panel
        Mapping ``{gene: expected_sign}`` with ``expected_sign in {+1, -1}``.
        ``lineage_bias = score_A - score_B``; a driver of lineage A that is knocked
        out removes lineage-A drive, so its expected sign is ``-1``.
    lineage_A_clusters, lineage_B_clusters
        Cluster names defining the two competing lineages.
    basis
        Embedding basis for flow projection.
    wt_flow_key
        Key in ``adata.obsm`` for the WT Hopfield velocity.
    cluster_key
        Key in ``adata.obs`` for cluster labels.
    simulate_kwargs
        Extra kwargs forwarded to ``simulate_shift_ode`` via :func:`run_ko_screen`.
    verbose
        Print per-gene OK/MISS lines.

    Returns
    -------
    table : :class:`pandas.DataFrame`
        One row per scored gene with columns ``gene``, ``lineage_bias``,
        ``expected_sign``, ``pred_sign``, ``correct``.
    accuracy : float
        Fraction of the panel whose KO-bias sign matched expectation
        (``nan`` if no panel gene is present in ``adata``).
    """
    genes = [g for g in panel if g in adata.var_names]
    if verbose:
        missing = [g for g in panel if g not in adata.var_names]
        if missing:
            print(f"Panel genes absent from adata (skipped): {missing}")

    bias_dict, _ = run_ko_screen(
        adata, genes=genes,
        lineage_A_clusters=lineage_A_clusters,
        lineage_B_clusters=lineage_B_clusters,
        basis=basis, wt_flow_key=wt_flow_key,
        cluster_key=cluster_key, simulate_kwargs=simulate_kwargs, verbose=verbose,
    )

    rows = []
    for g in genes:
        bias = float(bias_dict[g]['lineage_bias'])
        expected = int(panel[g])
        pred = int(np.sign(bias)) if bias != 0 else 0
        correct = pred == expected
        rows.append({'gene': g, 'lineage_bias': round(bias, 4),
                     'expected_sign': expected, 'pred_sign': pred, 'correct': bool(correct)})
        if verbose:
            print(f"  {g:8s} bias={bias:+.4f} expect={expected:+d} -> "
                  f"{'OK' if correct else 'MISS'}")

    table = pd.DataFrame(rows)
    accuracy = float(table['correct'].mean()) if len(table) else float('nan')
    if verbose:
        n_ok = int(table['correct'].sum()) if len(table) else 0
        print(f"\nDirectional accuracy: {accuracy:.2f} ({n_ok}/{len(table)})")
    return table, accuracy


def run_pairwise_ko_screen(
    adata: AnnData,
    pairs: List[Tuple[str, str]],
    lineage_A_clusters: List[str],
    lineage_B_clusters: List[str],
    basis: str,
    wt_flow_key: str,
    cluster_key: str = 'cell_type',
    cluster_order: Optional[List[str]] = None,
    simulate_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[Tuple[str, str], pd.Series]]:
    """
    Run a pairwise KO screen and compute lineage bias + cluster effects.

    Parameters
    ----------
    adata : AnnData
        Base (WT) AnnData with fitted model.
    pairs : list of (str, str)
        Gene-name tuples to screen. Pairs where either gene is absent are skipped.
    lineage_A_clusters : list of str
        Cluster names for lineage A.
    lineage_B_clusters : list of str
        Cluster names for lineage B.
    basis : str
        Embedding basis for flow projection.
    wt_flow_key : str
        Key in ``adata.obsm`` for the pre-computed WT Hopfield velocity.
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels.
    cluster_order : list of str, optional
        Ordered cluster names for ``compute_cluster_effects``.
        If None, uses ``adata.obs[cluster_key].unique()``.
    simulate_kwargs : dict, optional
        Extra keyword arguments forwarded to ``simulate_shift_ode``.
        Defaults: ``dt=5.0, n_steps=100, use_cluster_specific_GRN=True, n_jobs=-1``.
    verbose : bool, optional (default: True)
        Print progress for each pair.

    Returns
    -------
    bias_dict : dict[(str, str), dict]
        ``{(geneA, geneB): {'score_A', 'score_B', 'lineage_bias'}}``
    effects_dict : dict[(str, str), pd.Series]
        ``{(geneA, geneB): pd.Series(mean ``|delta_X|`` per cluster)}``

    Examples
    --------
    >>> import itertools
    >>> cross_pairs = list(itertools.product(top5_ery, top5_mye))
    >>> bias, effects = sch.dyn.run_pairwise_ko_screen(
    ...     adata, cross_pairs, ERYTHROID, MYELOID,
    ...     basis='draw_graph_fa', wt_flow_key='original_velocity_flow_draw_graph_fa',
    ...     cluster_key='paul15_clusters', cluster_order=CLUSTER_ORDER,
    ... )
    """
    from .simulation import simulate_shift_ode

    if simulate_kwargs is None:
        simulate_kwargs = {}
    sim_kw = dict(
        cluster_key=cluster_key,
        dt=5.0,
        n_steps=100,
        use_cluster_specific_GRN=True,
        n_jobs=-1,
        verbose=False,
    )
    sim_kw.update(simulate_kwargs)

    if cluster_order is None:
        cluster_order = list(adata.obs[cluster_key].unique())

    bias_dict    = {}
    effects_dict = {}

    for geneA, geneB in pairs:
        if geneA == geneB:
            continue
        if geneA not in adata.var_names or geneB not in adata.var_names:
            if verbose:
                print(f"  Skip ({geneA}, {geneB}): gene not in adata")
            continue
        if verbose:
            print(f"  KO pair: ({geneA}, {geneB})...")
        adata_pair = simulate_shift_ode(
            adata.copy(),
            perturb_condition={geneA: 0.0, geneB: 0.0},
            **sim_kw,
        )
        bias_dict[(geneA, geneB)]    = compute_perturbation_flow_bias(
            adata_pair, adata,
            lineage_A_clusters, lineage_B_clusters,
            basis, wt_flow_key,
            cluster_key=cluster_key,
        )
        effects_dict[(geneA, geneB)] = compute_cluster_effects(
            adata_pair, cluster_order, cluster_key=cluster_key
        )

    if verbose:
        print(f"\nCompleted {len(bias_dict)} pairwise KOs.")

    return bias_dict, effects_dict




def compute_epistasis(
    pair_ko_bias: Dict,
    single_ko_bias,
    lineage_A_genes: Optional[List[str]] = None,
    lineage_B_genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute epistasis metrics for all pairwise KO results.

    For each gene pair (A, B) computes:

    - **cancellation_error**: ``actual_bias - (bias_A + bias_B)`` — deviation
      from the additive expectation (Bliss independence on lineage bias).
    - **synergy_score**: Directionally corrected cancellation error.
      Positive means synergistic (amplifies bias in the same direction).

    Parameters
    ----------
    pair_ko_bias : dict
        ``{(geneA, geneB): {'score_A', 'score_B', 'lineage_bias'}}``
        from ``run_pairwise_ko_screen``.
    single_ko_bias : dict or pd.DataFrame
        ``{gene: {'score_A', 'score_B', 'lineage_bias'}}`` for all single KOs.
        A DataFrame indexed by gene name with these columns is also accepted.
    lineage_A_genes : list of str, optional
        Genes in lineage A; used only to classify pair type (``'ery-ery'``,
        ``'cross'``, etc.). If None, all pairs are labelled ``'unknown'``.
    lineage_B_genes : list of str, optional
        Genes in lineage B; used only for pair type classification.

    Returns
    -------
    pd.DataFrame
        Indexed by ``'geneA+geneB'`` pair string, sorted by ``lineage_bias``
        descending.  Columns: ``geneA``, ``geneB``, ``score_A``, ``score_B``,
        ``lineage_bias``, ``expected_bias``, ``cancellation_error``,
        ``synergy_score``, ``pair_type``.

    Examples
    --------
    >>> pair_df = sch.dyn.compute_epistasis(
    ...     pair_ko_bias, single_ko_bias,
    ...     lineage_A_genes=top5_ery, lineage_B_genes=top5_mye,
    ... )
    """

    def _get(sko, gene, key):
        if isinstance(sko, pd.DataFrame):
            return float(sko.loc[gene, key]) if gene in sko.index else 0.0
        return float(sko.get(gene, {}).get(key, 0.0))

    ery_genes = list(lineage_A_genes) if lineage_A_genes is not None else []
    mye_genes = list(lineage_B_genes) if lineage_B_genes is not None else []

    records = []
    for (gA, gB), bias in pair_ko_bias.items():
        bias_A = _get(single_ko_bias, gA, 'lineage_bias')
        bias_B = _get(single_ko_bias, gB, 'lineage_bias')

        score_A_pair = bias.get('score_A', 0.0)
        score_B_pair = bias.get('score_B', 0.0)
        actual_bias  = bias.get('lineage_bias', np.nan)

        expected_bias       = bias_A + bias_B
        cancellation_error  = actual_bias - expected_bias
        bias_sign = 1 if bias_A > 0 else -1
        synergy_score = cancellation_error * bias_sign

        if ery_genes and mye_genes:
            in_ery_A = gA in ery_genes
            in_ery_B = gB in ery_genes
            in_mye_A = gA in mye_genes
            in_mye_B = gB in mye_genes
            if (in_ery_A and in_mye_B) or (in_mye_A and in_ery_B):
                pair_type = 'cross'
            elif in_ery_A and in_ery_B:
                pair_type = 'ery-ery'
            elif in_mye_A and in_mye_B:
                pair_type = 'mye-mye'
            else:
                pair_type = 'other'
        else:
            pair_type = 'unknown'

        records.append({
            'geneA':              gA,
            'geneB':              gB,
            'pair':               f'{gA}+{gB}',
            'score_A':            score_A_pair,
            'score_B':            score_B_pair,
            'lineage_bias':       actual_bias,
            'expected_bias':      expected_bias,
            'cancellation_error': cancellation_error,
            'synergy_score':      synergy_score,
            'pair_type':          pair_type,
        })

    return (
        pd.DataFrame(records)
        .set_index('pair')
        .sort_values('lineage_bias', ascending=False)
    )


def dose_levels_from_fractions(
    adata: AnnData,
    gene: str,
    fractions,
    spliced_key: str = 'Ms',
    percentile: float = 99.0,
    natural_max: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Convert fractions of a gene's natural expression maximum to absolute levels.

    ``fractions`` are multiples of the gene's natural maximum, so the dose is expressed
    in interpretable units rather than raw expression: ``0`` = full knockout, ``0.5`` =
    knock down to 50%, ``1.0`` = natural level, ``2.0`` = strong overexpression.

    Parameters
    ----------
    adata : AnnData
        AnnData with the gene's expression in ``adata.layers[spliced_key]``.
    gene : str
        Gene to dose.
    fractions : array-like
        Fractions of the natural max to convert to absolute levels.
    spliced_key : str, optional (default: 'Ms')
        Layer used to estimate the natural maximum.
    percentile : float, optional (default: 99.0)
        Percentile of the gene's expression taken as the natural maximum (robust to
        outliers vs the raw max).
    natural_max : float, optional
        If given, used directly instead of estimating from the data.

    Returns
    -------
    (levels, natural_max) : (np.ndarray, float)
        Absolute expression levels and the natural maximum used.
    """
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")
    if natural_max is None:
        idx = int(adata.var_names.get_loc(gene))
        expr = to_numpy(get_matrix(adata, spliced_key, genes=[idx])).ravel()
        natural_max = float(np.percentile(expr, percentile))
    fractions = np.asarray(fractions, dtype=float)
    return fractions * natural_max, float(natural_max)


def run_dose_response(
    adata: AnnData,
    gene: str,
    levels=None,
    lineage_A_clusters: List[str] = None,
    lineage_B_clusters: List[str] = None,
    basis: str = 'umap',
    wt_flow_key: str = None,
    natural_max: Optional[float] = None,
    fractions=None,
    spliced_key: str = 'Ms',
    percentile: float = 99.0,
    cluster_key: str = 'cell_type',
    simulate_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run ODE perturbation at multiple expression levels for dose-response analysis.

    Sweeps from 0 (complete KO) through natural expression to 2x natural max
    (strong OE).  Returns lineage bias at each level, revealing whether the
    erythroid/myeloid switch is graded or threshold-like.

    The dose can be given either as absolute ``levels`` or as ``fractions`` of the
    gene's natural maximum (0 = KO, 0.5 = 50% knockdown, 1 = natural, 2 = strong OE);
    when neither is given, a default fraction sweep is used.

    Parameters
    ----------
    adata : AnnData
        Base (WT) AnnData with fitted model.
    gene : str
        Gene name to perturb.
    levels : array-like, optional
        Absolute expression levels to test (e.g. ``np.linspace(0, max*2, 10)``). If
        None, computed from ``fractions`` via :func:`dose_levels_from_fractions`.
    fractions : array-like, optional
        Fractions of the gene's natural maximum to test (used when ``levels`` is None).
        Defaults to ``[0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]``.
    spliced_key : str, optional (default: 'Ms')
        Layer used to estimate the natural maximum when converting fractions.
    percentile : float, optional (default: 99.0)
        Percentile taken as the gene's natural maximum for the fraction conversion.
    lineage_A_clusters : list of str
        Cluster names for lineage A (e.g. erythroid).
    lineage_B_clusters : list of str
        Cluster names for lineage B (e.g. myeloid).
    basis : str
        Embedding basis for flow projection.
    wt_flow_key : str
        Key in ``adata.obsm`` for the pre-computed WT Hopfield velocity.
    natural_max : float, optional
        Natural expression maximum for the gene (e.g. 99th percentile).
        When provided, adds a ``level_frac`` column (``level / natural_max``).
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels.
    simulate_kwargs : dict, optional
        Extra keyword arguments forwarded to ``simulate_shift_ode``.
    verbose : bool, optional (default: True)
        Show tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        Columns: ``['gene', 'level', 'level_frac', 'score_A', 'score_B', 'lineage_bias']``.
        One row per level.

    Examples
    --------
    >>> gata1_max = float(np.percentile(adata.layers['spliced'][:, idx], 99))
    >>> levels = np.linspace(0, gata1_max * 2, 10)
    >>> dr = sch.dyn.run_dose_response(
    ...     adata, 'Gata1', levels, ERYTHROID, MYELOID,
    ...     basis='draw_graph_fa', wt_flow_key='original_velocity_flow_draw_graph_fa',
    ...     natural_max=gata1_max, cluster_key='paul15_clusters',
    ... )
    """
    from .simulation import simulate_shift_ode
    from ..tools.flow import calculate_flow

    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")

    if simulate_kwargs is None:
        simulate_kwargs = {}
    sim_kw = dict(
        cluster_key=cluster_key,
        dt=5.0,
        n_steps=100,
        use_cluster_specific_GRN=True,
        n_jobs=-1,
        verbose=False,
    )
    sim_kw.update(simulate_kwargs)

    # Resolve dose levels: explicit absolute levels, else fractions of the natural max.
    if levels is None:
        if fractions is None:
            fractions = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        levels, natural_max = dose_levels_from_fractions(
            adata, gene, fractions, spliced_key=spliced_key,
            percentile=percentile, natural_max=natural_max,
        )
    levels = np.asarray(levels, dtype=float)
    records = []

    iter_levels = (
        tqdm(levels, desc=f'{gene} dose-response') if verbose else levels
    )
    for level in iter_levels:
        adata_t = simulate_shift_ode(
            adata.copy(),
            perturb_condition={gene: float(level)},
            **sim_kw,
        )
        calculate_flow(
            adata_t, source='delta', basis=basis, method='celloracle',
            cluster_key=cluster_key,
            store_key=f'perturbation_flow_{basis}',
            verbose=False,
        )
        bias = compute_perturbation_flow_bias(
            adata_t, adata,
            lineage_A_clusters, lineage_B_clusters,
            basis, wt_flow_key,
            cluster_key=cluster_key,
        )
        level_frac = float(level) / natural_max if natural_max is not None else np.nan
        records.append({
            'gene':         gene,
            'level':        float(level),
            'level_frac':   level_frac,
            'score_A':      bias.get('score_A', np.nan),
            'score_B':      bias.get('score_B', np.nan),
            'lineage_bias': bias.get('lineage_bias', np.nan),
        })

    return pd.DataFrame(records)
