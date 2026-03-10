"""Perturbation analysis tools for lineage driver discovery.

Functions for scoring TF drivers from GRN structure, computing lineage bias
from KO perturbation flow, and CellOracle-compatible perturbation scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from anndata import AnnData

from .._utils.io import get_genes_used


def score_driver_tfs(
    adata: AnnData,
    lineage_A_clusters: List[str],
    lineage_B_clusters: List[str],
    cluster_key: str = 'cell_type',
    n_top: Optional[int] = None,
) -> pd.DataFrame:
    """
    Score transcription factors as lineage drivers from GRN structure.

    Combines three signals averaged over the specified lineage clusters:
    - W-matrix row L2-norm (interaction strength)
    - Out-degree centrality (regulatory influence)
    - |Energy-gene correlation| (association with energy landscape)

    Each signal is ranked across genes and summed to a composite score.
    The lineage bias = score_A - score_B; positive values indicate an
    erythroid-biased gene (if A is erythroid), negative values indicate
    a myeloid-biased gene.

    Parameters
    ----------
    adata : AnnData
        Annotated data with fitted interactions and computed centrality /
        energy-gene correlation (run `compute_network_centrality` and
        `energy_gene_correlation` first).
    lineage_A_clusters : list of str
        Cluster names defining lineage A (e.g. erythroid clusters).
    lineage_B_clusters : list of str
        Cluster names defining lineage B (e.g. myeloid clusters).
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels.
    n_top : int, optional
        If provided, return only the top n_top genes by max(score_A, score_B).
        Useful for pre-filtering before a KO screen.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by gene name with columns:
        - ``score_A``: composite rank-sum score for lineage A
        - ``score_B``: composite rank-sum score for lineage B
        - ``lineage_bias``: score_A - score_B
        - ``rank_A``: rank by score_A (1 = highest)
        - ``rank_B``: rank by score_B (1 = highest)
        - ``wnorm_A``, ``wnorm_B``: mean W-matrix row L2-norm per lineage
        - ``deg_A``, ``deg_B``: mean out-degree centrality per lineage
        - ``ecorr_A``, ``ecorr_B``: mean |energy-gene correlation| per lineage

    Examples
    --------
    >>> import scHopfield as sch
    >>> ERYTHROID = ['1Ery', '2Ery', '3Ery']
    >>> MYELOID   = ['9GMP', '10GMP', '11DC']
    >>> tf_df = sch.tl.score_driver_tfs(adata, ERYTHROID, MYELOID, cluster_key='paul15_clusters')
    >>> top_ery = tf_df.nlargest(10, 'score_A')
    """
    def _rank(s: pd.Series) -> pd.Series:
        return s.fillna(0).rank(method='average')

    def _mean_var_col(col_prefix: str, cluster_list: List[str]) -> pd.Series:
        cols = [
            f'{col_prefix}_{cl}'
            for cl in cluster_list
            if f'{col_prefix}_{cl}' in adata.var.columns
        ]
        if not cols:
            return pd.Series(0.0, index=adata.var_names)
        return adata.var[cols].mean(axis=1)

    def _mean_wnorm(cluster_list: List[str]) -> pd.Series:
        result = pd.Series(0.0, index=adata.var_names)
        count = 0
        for cl in cluster_list:
            key = f'W_{cl}'
            if key in adata.varp:
                W = adata.varp[key]
                result += pd.Series(np.linalg.norm(W, axis=1), index=adata.var_names)
                count += 1
        if count > 0:
            result /= count
        return result

    # Compute per-lineage signals
    wnorm_A  = _mean_wnorm(lineage_A_clusters)
    wnorm_B  = _mean_wnorm(lineage_B_clusters)
    deg_A    = _mean_var_col('degree_centrality_out', lineage_A_clusters)
    deg_B    = _mean_var_col('degree_centrality_out', lineage_B_clusters)
    ecorr_A  = _mean_var_col('correlation_total', lineage_A_clusters).abs()
    ecorr_B  = _mean_var_col('correlation_total', lineage_B_clusters).abs()

    # Composite rank-sum scores
    score_A = _rank(wnorm_A) + _rank(deg_A) + _rank(ecorr_A)
    score_B = _rank(wnorm_B) + _rank(deg_B) + _rank(ecorr_B)
    lineage_bias = score_A - score_B

    df = pd.DataFrame({
        'score_A':      score_A.values,
        'score_B':      score_B.values,
        'lineage_bias': lineage_bias.values,
        'wnorm_A':      wnorm_A.values,
        'wnorm_B':      wnorm_B.values,
        'deg_A':        deg_A.values,
        'deg_B':        deg_B.values,
        'ecorr_A':      ecorr_A.values,
        'ecorr_B':      ecorr_B.values,
    }, index=adata.var_names)

    # Add integer ranks (1 = best)
    df['rank_A'] = df['score_A'].rank(method='min', ascending=False).astype(int)
    df['rank_B'] = df['score_B'].rank(method='min', ascending=False).astype(int)

    if n_top is not None:
        max_score = df[['score_A', 'score_B']].max(axis=1)
        df = df.loc[max_score.nlargest(n_top).index].copy()

    return df


def compute_lineage_bias(
    adata_ko: AnnData,
    adata_wt: AnnData,
    lineage_A_clusters: List[str],
    lineage_B_clusters: List[str],
    basis: str,
    wt_flow_key: str,
    cluster_key: str = 'cell_type',
    n_neighbors: int = 30,
) -> Dict[str, float]:
    """
    Compute lineage bias for a KO simulation using flow alignment.

    Projects the perturbation delta_X to the embedding space (dot-product
    KNN projection) and computes the mean cosine similarity with the
    pre-computed WT Hopfield velocity for cells in each lineage.

    Positive score  → KO aligns with that lineage's differentiation direction.
    Negative score  → KO opposes that lineage's direction (blocks/redirects).
    lineage_bias = score_A − score_B: positive = lineage-A-biasing.

    Parameters
    ----------
    adata_ko : AnnData
        AnnData with perturbation results (``delta_X`` layer required).
    adata_wt : AnnData
        Wild-type AnnData containing the precomputed WT flow in obsm.
    lineage_A_clusters : list of str
        Cluster names for lineage A (used to mask cells).
    lineage_B_clusters : list of str
        Cluster names for lineage B (used to mask cells).
    basis : str
        Embedding basis (e.g. ``'draw_graph_fa'`` or ``'umap'``).
    wt_flow_key : str
        Key in ``adata_wt.obsm`` for the WT Hopfield velocity field.
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels.
    n_neighbors : int, optional (default: 30)
        Number of neighbors for dot-product embedding projection.

    Returns
    -------
    dict
        ``{'score_A': float, 'score_B': float, 'lineage_bias': float}``

    Examples
    --------
    >>> bias = sch.tl.compute_lineage_bias(
    ...     adata_ko, adata, ERYTHROID, MYELOID,
    ...     basis='draw_graph_fa', wt_flow_key='original_velocity_flow_draw_graph_fa',
    ... )
    """
    from .embedding import project_to_embedding

    if 'delta_X' not in adata_ko.layers:
        return {'score_A': np.nan, 'score_B': np.nan, 'lineage_bias': np.nan}

    genes_mask = get_genes_used(adata_ko)
    delta_X_used = np.asarray(adata_ko.layers['delta_X'])[:, genes_mask]

    perturb_embed = project_to_embedding(
        adata_ko, delta_X_used, basis=basis, method='dot_product', n_neighbors=n_neighbors
    )

    wt_embed = adata_wt.obsm[wt_flow_key]

    n1 = np.linalg.norm(wt_embed,      axis=1, keepdims=True) + 1e-10
    n2 = np.linalg.norm(perturb_embed, axis=1, keepdims=True) + 1e-10
    cosine_sim = np.sum((wt_embed / n1) * (perturb_embed / n2), axis=1)

    obs_cl   = adata_ko.obs[cluster_key]
    mask_A   = obs_cl.isin(lineage_A_clusters).values
    mask_B   = obs_cl.isin(lineage_B_clusters).values

    score_A = float(cosine_sim[mask_A].mean()) if mask_A.sum() > 0 else np.nan
    score_B = float(cosine_sim[mask_B].mean()) if mask_B.sum() > 0 else np.nan
    lineage_bias = (
        score_A - score_B
        if not (np.isnan(score_A) or np.isnan(score_B))
        else np.nan
    )

    return {'score_A': score_A, 'score_B': score_B, 'lineage_bias': lineage_bias}


def compute_cluster_effects(
    adata_ko: AnnData,
    cluster_order: List[str],
    cluster_key: str = 'cell_type',
) -> pd.Series:
    """
    Compute mean |delta_X| magnitude per cluster after a KO simulation.

    Parameters
    ----------
    adata_ko : AnnData
        AnnData with perturbation results (``delta_X`` layer required).
    cluster_order : list of str
        Ordered list of cluster names to include in the output.
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels.

    Returns
    -------
    pd.Series
        Mean absolute delta_X per cluster, indexed by cluster name.
        Only clusters with ≥1 cell are included.

    Examples
    --------
    >>> effects = sch.tl.compute_cluster_effects(adata_ko, CLUSTER_ORDER, 'paul15_clusters')
    >>> effects.plot.bar()
    """
    if 'delta_X' not in adata_ko.layers:
        return pd.Series(dtype=float)

    delta_X = np.asarray(adata_ko.layers['delta_X'])
    obs_cl  = adata_ko.obs[cluster_key]
    result  = {}
    for cl in cluster_order:
        mask = (obs_cl == cl).values
        if mask.sum() > 0:
            result[cl] = float(np.abs(delta_X[mask]).mean())
    return pd.Series(result)


def compute_perturbation_score(
    adata_ko: AnnData,
    adata_wt: AnnData,
    pseudotime_key: str,
    basis: str,
    cluster_key: str = 'cell_type',
    n_grid: int = 40,
    flow_key: Optional[str] = None,
    n_neighbors: int = 200,
    min_mass: float = 1.0,
) -> Dict:
    """
    Compute CellOracle-style perturbation score (PS) using pseudotime gradient.

    Builds a 2D pseudotime gradient field on a regular grid (polynomial degree-3
    regression + ``np.gradient``), then projects the KO perturbation flow from
    cells onto grid points via Gaussian-weighted KNN averaging.  The perturbation
    score at each grid point is the cosine similarity between the projected flow
    and the local pseudotime gradient direction.

    This matches CellOracle's ``Oracle_development_module.calculate_inner_product``
    logic: flow vectors are projected **cells → grid** (not grid → cells), and
    the inner product is evaluated at grid points where cell density exceeds
    ``min_mass``.  Negative PS = perturbation opposes differentiation.
    Ranking metric = ``ps_negative_sum`` (sum of negative PS at non-empty grid
    points).

    Parameters
    ----------
    adata_ko : AnnData
        AnnData with KO perturbation flow already computed
        (``adata_ko.obsm[flow_key]`` required).
    adata_wt : AnnData
        Wild-type AnnData containing pseudotime values.
    pseudotime_key : str
        Key in ``adata_wt.obs`` for pseudotime values.
    basis : str
        Embedding basis (e.g. ``'umap'`` or ``'draw_graph_fa'``).
    cluster_key : str, optional (default: 'cell_type')
        Reserved; unused.
    n_grid : int, optional (default: 40)
        Number of grid points per embedding dimension.
    flow_key : str, optional
        Key in ``adata_ko.obsm`` for the perturbation flow vectors.
        Defaults to ``f'perturbation_flow_{basis}'``.
    n_neighbors : int, optional (default: 200)
        Number of KNN neighbors (cells) for flow interpolation to grid.
    min_mass : float, optional (default: 1.0)
        Minimum cell density at a grid point (as % of maximum) below which
        the grid point is masked out.  Matches CellOracle's default.

    Returns
    -------
    dict
        ``{
            'ps_per_grid':     np.ndarray (n_grid^2,)  — NaN at masked points,
            'grid_coords':     np.ndarray (n_grid^2, 2),
            'mass':            np.ndarray (n_grid^2,)  — raw density,
            'mass_filter':     np.ndarray (n_grid^2,)  bool — True = masked out,
            'ps_mean':         float,
            'ps_negative_sum': float,   # CellOracle's ranking metric
            'ps_per_cell':     np.ndarray (n_cells,),  # nearest valid grid PS
        }``

    References
    ----------
    Kamimoto et al. (2023). Dissecting cell identity via network inference
    and in silico gene perturbation. *Nature* 614, 742–751.
    https://doi.org/10.1038/s41586-022-05688-9

    Examples
    --------
    >>> ps = sch.tl.compute_perturbation_score(
    ...     adata_ko, adata,
    ...     pseudotime_key='Pseudotime',
    ...     basis='draw_graph_fa',
    ... )
    >>> print(ps['ps_negative_sum'])
    """
    from sklearn.neighbors import NearestNeighbors

    if flow_key is None:
        flow_key = f'perturbation_flow_{basis}'

    embedding_key = f'X_{basis}'
    if embedding_key not in adata_ko.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata_ko.obsm")
    if flow_key not in adata_ko.obsm:
        raise ValueError(
            f"Flow key '{flow_key}' not found in adata_ko.obsm. "
            f"Run sch.tl.calculate_flow(adata_ko, source='delta', basis='{basis}') first."
        )
    if pseudotime_key not in adata_wt.obs.columns:
        raise ValueError(f"Pseudotime key '{pseudotime_key}' not found in adata_wt.obs")

    embedding  = adata_ko.obsm[embedding_key]
    ko_flow    = adata_ko.obsm[flow_key]
    pseudotime = adata_wt.obs[pseudotime_key].values.astype(float)

    # --- Build pseudotime surface on regular grid via polynomial regression ---
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    gx = np.linspace(x_min, x_max, n_grid)
    gy = np.linspace(y_min, y_max, n_grid)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (n_grid^2, 2)

    def _poly_features(xy, degree=3):
        x, y = xy[:, 0], xy[:, 1]
        feats = []
        for d in range(degree + 1):
            for i in range(d + 1):
                feats.append((x ** (d - i)) * (y ** i))
        return np.column_stack(feats)

    X_feat = _poly_features(embedding)
    G_feat = _poly_features(grid_coords)

    coef, *_ = np.linalg.lstsq(X_feat, pseudotime, rcond=None)
    pt_grid  = G_feat @ coef                  # pseudotime on grid (n_grid^2,)
    pt_mat   = pt_grid.reshape(n_grid, n_grid)

    dpt_dy, dpt_dx = np.gradient(pt_mat, gy, gx)   # (n_grid, n_grid) each
    grad_vectors = np.column_stack([dpt_dx.ravel(), dpt_dy.ravel()])  # (n_grid^2, 2)

    # --- Project KO flow: cells → grid (CellOracle direction) ---
    # Fit KNN on cells; query from each grid point to find its nearest cells
    k = min(n_neighbors, len(embedding))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(embedding)
    distances, indices = nn.kneighbors(grid_coords)  # (n_grid^2, k)

    sigma = np.median(distances) + 1e-10
    raw_weights = np.exp(-distances ** 2 / (2 * sigma ** 2))  # (n_grid^2, k)

    # Cell-density proxy at each grid point
    mass = raw_weights.sum(axis=1)  # (n_grid^2,)

    # Normalise weights for flow interpolation
    weights = raw_weights / (raw_weights.sum(axis=1, keepdims=True) + 1e-10)

    # Gaussian-weighted average KO flow at each grid point
    grid_flow = np.einsum('ij,ijk->ik', weights, ko_flow[indices])  # (n_grid^2, 2)

    # --- Density-based mask ---
    mass_norm   = mass / (mass.max() + 1e-10)
    mass_filter = mass_norm < (min_mass / 100.0)   # True = low density, masked out

    # --- Inner product at grid (cosine similarity) ---
    flow_norm = np.linalg.norm(grid_flow,    axis=1, keepdims=True) + 1e-10
    grad_norm = np.linalg.norm(grad_vectors, axis=1, keepdims=True) + 1e-10
    ps_grid_raw = np.sum((grid_flow / flow_norm) * (grad_vectors / grad_norm), axis=1)

    ps_per_grid = ps_grid_raw.astype(float)
    ps_per_grid[mass_filter] = np.nan

    # Summary stats over non-masked grid points
    ps_valid       = ps_grid_raw[~mass_filter]
    ps_mean        = float(ps_valid.mean())        if len(ps_valid) > 0 else np.nan
    ps_negative_sum = float(ps_valid[ps_valid < 0].sum()) if len(ps_valid) > 0 else 0.0

    # --- ps_per_cell: assign each cell its nearest non-masked grid PS ---
    valid_mask = ~mass_filter
    if valid_mask.sum() > 0:
        nn_cell = NearestNeighbors(n_neighbors=1)
        nn_cell.fit(grid_coords[valid_mask])
        _, cell_nn_idx = nn_cell.kneighbors(embedding)
        ps_per_cell = ps_grid_raw[valid_mask][cell_nn_idx[:, 0]]
    else:
        ps_per_cell = np.full(len(embedding), np.nan)

    return {
        'ps_per_grid':     ps_per_grid,
        'grid_coords':     grid_coords,
        'mass':            mass,
        'mass_filter':     mass_filter,
        'ps_mean':         ps_mean,
        'ps_negative_sum': ps_negative_sum,
        'ps_per_cell':     ps_per_cell,
    }
