"""Perturbation analysis tools for lineage driver discovery.

Functions for scoring TF drivers from GRN structure, computing lineage bias
from KO perturbation flow, and CellOracle-compatible perturbation scores.
"""

import warnings

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
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
    - Energy-gene correlation (association with energy landscape)

    Each signal is standardized (z-scored across genes) and averaged into a
    composite score on a comparable scale. The lineage bias = score_A - score_B;
    positive values indicate an erythroid-biased gene (if A is erythroid), negative
    values indicate a myeloid-biased gene. This is a *structural, pre-perturbation*
    prior from the fitted GRN; for the effect of an actual perturbation on lineage
    choice use ``compute_perturbation_commitment_change``.

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
        - ``score_A``: standardized composite score for lineage A
        - ``score_B``: standardized composite score for lineage B
        - ``lineage_bias``: score_A - score_B
        - ``rank_A``: rank by score_A (1 = highest)
        - ``rank_B``: rank by score_B (1 = highest)
        - ``wnorm_A``, ``wnorm_B``: mean W-matrix row L2-norm per lineage
        - ``deg_A``, ``deg_B``: mean out-degree centrality per lineage
        - ``ecorr_A``, ``ecorr_B``: mean absolute energy-gene correlation per lineage

    Examples
    --------
    >>> import scHopfield as sch
    >>> ERYTHROID = ['1Ery', '2Ery', '3Ery']
    >>> MYELOID   = ['9GMP', '10GMP', '11DC']
    >>> tf_df = sch.tl.score_driver_tfs(adata, ERYTHROID, MYELOID, cluster_key='paul15_clusters')
    >>> top_ery = tf_df.nlargest(10, 'score_A')
    """
    def _z(s: pd.Series) -> pd.Series:
        s = s.fillna(0.0).astype(float)
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 0 else s * 0.0

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

    # Standardized composite: mean of z-scored signals so the three heterogeneous
    # signals (W-norm, out-degree, energy correlation) share a comparable scale and
    # contribute equally, instead of an unnormalized rank-sum.
    score_A = (_z(wnorm_A) + _z(deg_A) + _z(ecorr_A)) / 3.0
    score_B = (_z(wnorm_B) + _z(deg_B) + _z(ecorr_B)) / 3.0
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


def compute_perturbation_flow_bias(
    adata_ko: AnnData,
    adata_wt: AnnData,
    lineage_A_clusters: List[str],
    lineage_B_clusters: List[str],
    basis: str,
    wt_flow_key: str,
    cluster_key: str = 'cell_type',
    n_neighbors: int = 30,
    method: str = 'correlation',
    projector=None,
) -> Dict[str, float]:
    """
    Post-perturbation flow-alignment bias for a KO simulation.

    Projects the perturbation delta_X to the embedding space (dot-product
    KNN projection) and computes the mean cosine similarity with the
    pre-computed WT Hopfield velocity for cells in each lineage.

    Positive score  → KO aligns with that lineage's differentiation direction.
    Negative score  → KO opposes that lineage's direction (blocks/redirects).
    lineage_bias = score_A − score_B: positive = lineage-A-biasing.

    Notes
    -----
    This measures how the *perturbation-induced* flow aligns with the unperturbed
    developmental flow; it is relative to the WT flow but is not a matched pre/post
    difference. For an explicit, model-independent pre-vs-post comparison use
    ``compute_perturbation_commitment_change``. (Formerly ``compute_lineage_bias``.)

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

    # Project the perturbation displacement to the embedding. Default is the CellOracle-style
    # correlation (transition-probability) scheme, matching the reference velocity and the flow
    # visualizations; a precomputed ``projector`` (build_correlation_projector) makes screens
    # affordable. Pass method='dot_product' for the older gene-space KNN projection.
    if projector is not None:
        perturb_embed = projector(delta_X_used)
    else:
        perturb_embed = project_to_embedding(
            adata_ko, delta_X_used, basis=basis, method=method, n_neighbors=n_neighbors
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


def compute_lineage_bias(*args, **kwargs) -> Dict[str, float]:
    """Deprecated alias for :func:`compute_perturbation_flow_bias`.

    Renamed to disambiguate it from :func:`score_driver_tfs` (a structural,
    pre-perturbation score that also returned ``score_A/score_B/lineage_bias``).
    For an explicit pre/post comparison use
    :func:`compute_perturbation_commitment_change`.
    """
    warnings.warn(
        "compute_lineage_bias is deprecated; use compute_perturbation_flow_bias, or "
        "compute_perturbation_commitment_change for a matched pre/post comparison.",
        DeprecationWarning, stacklevel=2,
    )
    return compute_perturbation_flow_bias(*args, **kwargs)


def lineage_axis_from_embedding(
    adata: AnnData,
    basis: str,
    lineage_A_clusters: List[str],
    lineage_B_clusters: List[str],
    cluster_key: str = 'cell_type',
) -> np.ndarray:
    """
    Unit vector pointing from lineage B toward lineage A in the embedding.

    Defined as ``centroid(A) - centroid(B)`` over the cells of the given lineage
    clusters in ``adata.obsm['X_{basis}']``, normalized to unit length. The axis is
    derived purely from the data embedding (model-independent), giving a fair
    reference direction for scoring how a flow field commits toward lineage A vs B.
    """
    emb = np.asarray(adata.obsm[f'X_{basis}'])[:, :2]
    obs = adata.obs[cluster_key]
    mask_A = obs.isin(lineage_A_clusters).values
    mask_B = obs.isin(lineage_B_clusters).values
    if mask_A.sum() == 0 or mask_B.sum() == 0:
        raise ValueError("Both lineage_A_clusters and lineage_B_clusters must match cells.")
    axis = emb[mask_A].mean(0) - emb[mask_B].mean(0)
    nrm = np.linalg.norm(axis)
    return axis / nrm if nrm > 0 else axis


def compute_lineage_commitment(
    adata: AnnData,
    flow_key: str,
    lineage_axis: np.ndarray,
) -> np.ndarray:
    """
    Per-cell lineage commitment: cosine of the flow direction with ``lineage_axis``.

    Returns a value in [-1, 1] per cell: +1 = flow points fully toward lineage A,
    -1 = fully toward lineage B. Cells with (near-)zero flow return ~0.
    """
    F = np.asarray(adata.obsm[flow_key])[:, :2].astype(float)
    nrm = np.linalg.norm(F, axis=1, keepdims=True)
    Fu = np.divide(F, nrm, out=np.zeros_like(F), where=nrm > 0)
    axis = np.asarray(lineage_axis, dtype=float)[:2]
    return Fu @ axis


def compute_perturbation_commitment_change(
    adata_ko: AnnData,
    adata_wt: AnnData,
    lineage_A_clusters: List[str],
    lineage_B_clusters: List[str],
    basis: str,
    wt_flow_key: Optional[str] = None,
    ko_flow_key: Optional[str] = None,
    cluster_key: str = 'cell_type',
) -> Dict[str, object]:
    """
    Change in lineage commitment caused by a perturbation (pre vs post).

    A well-defined, model-independent pre/post score. Builds a data-derived lineage
    axis (``centroid(A) - centroid(B)`` in the embedding) and measures the per-cell
    cosine alignment of the developmental flow with that axis, for both the wild-type
    flow (pre) and the perturbed flow (post)::

        C_pre[c]  = cos( WT_flow[c], axis )
        C_post[c] = cos( KO_flow[c], axis )
        delta[c]  = C_post[c] - C_pre[c]

    A positive ``delta`` means the perturbation steers cells *toward* lineage A;
    negative means toward lineage B. Because both terms are cosines against the same
    fixed axis, the comparison is normalized and directly interpretable, unlike the
    ``score_A/score_B`` fields that ``score_driver_tfs`` and
    ``compute_perturbation_flow_bias`` share but define differently.

    Parameters
    ----------
    adata_ko, adata_wt : AnnData
        Perturbed and wild-type objects, each with its embedding flow already computed
        (see ``sch.tl.calculate_flow``).
    lineage_A_clusters, lineage_B_clusters : list of str
        Clusters defining the two competing lineages.
    basis : str
        Embedding basis (e.g. ``'draw_graph_fa'`` or ``'umap'``).
    wt_flow_key : str, optional
        Key in ``adata_wt.obsm`` for the WT velocity flow. Defaults to
        ``f'original_velocity_flow_{basis}'``.
    ko_flow_key : str, optional
        Key in ``adata_ko.obsm`` for the perturbed velocity flow. Defaults to
        ``f'perturbed_velocity_flow_{basis}'``.
    cluster_key : str, optional (default: 'cell_type')
        Key in ``adata.obs`` for cluster labels.

    Returns
    -------
    dict
        ``axis``; per-cell ``commitment_pre``, ``commitment_post``,
        ``delta_commitment``; per-lineage means ``pre_A/pre_B/post_A/post_B`` and
        ``delta_A/delta_B``; and the overall ``delta_mean``.
    """
    if wt_flow_key is None:
        wt_flow_key = f'original_velocity_flow_{basis}'
    if ko_flow_key is None:
        ko_flow_key = f'perturbed_velocity_flow_{basis}'
    for a, k, nm in [(adata_wt, wt_flow_key, 'wt'), (adata_ko, ko_flow_key, 'ko')]:
        if k not in a.obsm:
            raise ValueError(
                f"Flow key '{k}' not found in adata_{nm}.obsm. Compute it with "
                f"sch.tl.calculate_flow(adata_{nm}, source=..., basis='{basis}')."
            )

    axis = lineage_axis_from_embedding(
        adata_wt, basis, lineage_A_clusters, lineage_B_clusters, cluster_key
    )
    c_pre = compute_lineage_commitment(adata_wt, wt_flow_key, axis)
    c_post = compute_lineage_commitment(adata_ko, ko_flow_key, axis)
    n = min(c_pre.shape[0], c_post.shape[0])
    c_pre, c_post = c_pre[:n], c_post[:n]
    delta = c_post - c_pre

    obs = adata_wt.obs[cluster_key].iloc[:n]
    mA = obs.isin(lineage_A_clusters).values
    mB = obs.isin(lineage_B_clusters).values

    def _m(x, m):
        return float(np.mean(x[m])) if m.sum() > 0 else float('nan')

    return {
        'axis': axis,
        'commitment_pre': c_pre,
        'commitment_post': c_post,
        'delta_commitment': delta,
        'pre_A': _m(c_pre, mA), 'pre_B': _m(c_pre, mB),
        'post_A': _m(c_post, mA), 'post_B': _m(c_post, mB),
        'delta_A': _m(delta, mA), 'delta_B': _m(delta, mB),
        'delta_mean': float(np.mean(delta)) if n > 0 else float('nan'),
    }


def compute_cluster_effects(
    adata_ko: AnnData,
    cluster_order: List[str],
    cluster_key: str = 'cell_type',
) -> pd.Series:
    """
    Compute mean ``|delta_X|`` magnitude per cluster after a KO simulation.

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
        return pd.Series(dtype=np.float32)

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
    pseudotime = adata_wt.obs[pseudotime_key].values.astype(np.float32)

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

    ps_per_grid = ps_grid_raw.astype(np.float32)
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


def lineage_de(
    adata: AnnData,
    lineage_A_clusters: List[str],
    lineage_B_clusters: List[str],
    cluster_key: str = 'cell_type',
    spliced_key: str = 'spliced',
    gene_names: Optional[List[str]] = None,
) -> 'pd.DataFrame':
    """
    Differential expression between two lineages via Mann-Whitney U test.

    Ranks every gene by the absolute log2 fold change between the mean
    expression in lineage B and lineage A.  Positive log2FC = lineage-B-high.

    Parameters
    ----------
    adata : AnnData
        Annotated data with expression in ``adata.layers[spliced_key]``.
    lineage_A_clusters : list of str
        Cluster names for lineage A (reference; e.g. erythroid).
    lineage_B_clusters : list of str
        Cluster names for lineage B (query; e.g. myeloid).
    cluster_key : str, optional (default: 'cell_type')
        Key in ``adata.obs`` for cluster labels.
    spliced_key : str, optional (default: 'spliced')
        Key in ``adata.layers`` for the expression matrix.
    gene_names : list of str, optional
        Restrict analysis to these gene names.  If None, uses all genes.

    Returns
    -------
    pd.DataFrame
        Indexed by gene name, sorted by ``abs_log2fc`` descending, with
        columns:
        - ``log2fc``: log2(mean_B / mean_A)
        - ``pval``: two-sided Mann-Whitney U p-value
        - ``abs_log2fc``: ``|log2fc|``
        - ``rank``: 1 = most differentially expressed

    Examples
    --------
    >>> de = sch.tl.lineage_de(adata, ERYTHROID, MYELOID, cluster_key='paul15_clusters')
    >>> stat3_rank = de.loc['Stat3', 'rank']
    """
    from scipy import stats as sp_stats
    from tqdm.auto import tqdm

    mask_A = adata.obs[cluster_key].isin(lineage_A_clusters).values
    mask_B = adata.obs[cluster_key].isin(lineage_B_clusters).values

    expr      = adata.layers[spliced_key]
    all_genes = list(adata.var_names)

    if gene_names is None:
        gene_names = all_genes

    gene_indices    = [all_genes.index(g) for g in gene_names if g in all_genes]
    gene_names_used = [all_genes[i] for i in gene_indices]

    records = []
    for gi, gene in tqdm(
        zip(gene_indices, gene_names_used),
        total=len(gene_indices),
        desc='DE analysis',
    ):
        col    = np.asarray(expr[:, gi]).flatten()
        A_vals = col[mask_A]
        B_vals = col[mask_B]

        A_mean = A_vals.mean() + 1e-6
        B_mean = B_vals.mean() + 1e-6
        log2fc = np.log2(B_mean / A_mean)

        _, pval = sp_stats.mannwhitneyu(B_vals, A_vals, alternative='two-sided')
        records.append({'gene': gene, 'log2fc': log2fc, 'pval': pval})

    df = pd.DataFrame(records).set_index('gene')
    df['abs_log2fc'] = df['log2fc'].abs()
    df = df.sort_values('abs_log2fc', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    return df


def grn_partner_weights(
    adata: AnnData,
    anchor: str,
    cluster_keys: Optional[List[str]] = None,
) -> 'pd.DataFrame':
    """
    Extract bidirectional W^c regulatory weights for an anchor gene.

    For each cluster matrix ``W^c`` in ``adata.varp``, computes:

    ``w_combined_i = W^c[anchor_idx, i] + W^c[i, anchor_idx]``

    capturing the total bidirectional regulatory weight between the anchor
    and every other gene.  The resulting DataFrame is the input for
    partner-selection strategies such as the 4+4+4+4 diversified scheme.

    Parameters
    ----------
    adata : AnnData
        Annotated data with cluster-specific GRN matrices in ``adata.varp``
        under keys ``'W_{cluster_name}'``.
    anchor : str
        Gene name to use as the regulatory anchor.
    cluster_keys : list of str, optional
        Which ``W^c`` keys to include.  If None, all keys matching ``'W_*'``
        in ``adata.varp`` are used.

    Returns
    -------
    pd.DataFrame
        Indexed by gene name (anchor excluded) with columns:
        - ``w_{cluster}``: per-cluster w_combined value
        - ``w_abs_all``: mean ``|w_combined|`` across all specified clusters

    Examples
    --------
    >>> wdf = sch.tl.grn_partner_weights(adata, anchor='Gata1')
    >>> top10 = wdf.nlargest(10, 'w_abs_all')
    """
    gene_names = list(adata.var_names)
    n_genes    = len(gene_names)

    if anchor not in gene_names:
        raise ValueError(f"Anchor gene '{anchor}' not found in adata.var_names")

    a_idx = gene_names.index(anchor)

    if cluster_keys is None:
        cluster_keys = sorted([k for k in adata.varp.keys() if k.startswith('W_')])

    short_names = [k.replace('W_', '') for k in cluster_keys]
    W_mat = np.zeros((len(cluster_keys), n_genes))

    for ci, key in enumerate(cluster_keys):
        W_c = np.asarray(adata.varp[key])
        W_mat[ci, :] = W_c[a_idx, :] + W_c[:, a_idx]

    df = pd.DataFrame(
        W_mat.T,
        index=gene_names,
        columns=[f'w_{s}' for s in short_names],
    )
    df.index.name = 'gene'
    df['w_abs_all'] = np.abs(W_mat).mean(axis=0)
    df = df.drop(index=anchor, errors='ignore')
    return df


def compute_perturbation_alignment(
    adata_ko: AnnData,
    basis: str,
    flow_key: Optional[str] = None,
    ip_key: str = 'ko_vs_wt_inner_product',
    n_grid: int = 30,
    min_density_pct: float = 30.0,
) -> Dict:
    """
    Interpolate perturbation flow onto a regular grid for arrow visualization.

    Takes per-cell perturbation flow vectors and a per-cell flow-alignment
    inner product (computed via ``calculate_flow`` + ``calculate_inner_product``)
    and projects them onto a regular 2D grid using linear interpolation and
    KDE-based density masking.  Returns all arrays needed for a quiver plot.

    Parameters
    ----------
    adata_ko : AnnData
        AnnData after KO simulation with:
        - ``adata_ko.obs[ip_key]``: per-cell inner product (float series)
        - ``adata_ko.obsm[flow_key]``: per-cell 2D perturbation flow
        - ``adata_ko.obsm['X_{basis}']``: 2D embedding coordinates
    basis : str
        Embedding basis key (e.g. ``'draw_graph_fa'``).
    flow_key : str, optional
        Key in ``adata_ko.obsm`` for perturbation flow.
        Defaults to ``f'perturbation_flow_{basis}'``.
    ip_key : str, optional (default: 'ko_vs_wt_inner_product')
        Key in ``adata_ko.obs`` for per-cell inner product values.
    n_grid : int, optional (default: 30)
        Number of grid points per embedding dimension.
    min_density_pct : float, optional (default: 30.0)
        KDE density percentile below which grid arrows are masked out.

    Returns
    -------
    dict
        Keys: ``'coords'`` (n_cells, 2), ``'ip'`` (n_cells,),
        ``'GX'``, ``'GY'``, ``'GU'``, ``'GV'`` (n_grid, n_grid each).
        ``GU``/``GV`` are NaN at low-density grid points.

    Examples
    --------
    >>> aln = sch.tl.compute_perturbation_alignment(adata_ko, basis='draw_graph_fa')
    >>> ax.scatter(aln['coords'][:,0], aln['coords'][:,1], c=aln['ip'])
    >>> ax.quiver(aln['GX'], aln['GY'], aln['GU'], aln['GV'])
    """
    from scipy.interpolate import griddata
    from scipy.stats import gaussian_kde

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
    if ip_key not in adata_ko.obs.columns:
        raise ValueError(
            f"Inner product key '{ip_key}' not found in adata_ko.obs. "
            "Run sch.tl.calculate_inner_product(adata_ko, ...) first."
        )

    coords  = adata_ko.obsm[embedding_key]
    flow_ko = adata_ko.obsm[flow_key]
    ip_ko   = adata_ko.obs[ip_key].values

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    gx = np.linspace(x_min, x_max, n_grid)
    gy = np.linspace(y_min, y_max, n_grid)
    GX, GY    = np.meshgrid(gx, gy)
    grid_pts  = np.column_stack([GX.ravel(), GY.ravel()])

    GU = griddata(coords, flow_ko[:, 0], grid_pts, method='linear').reshape(n_grid, n_grid)
    GV = griddata(coords, flow_ko[:, 1], grid_pts, method='linear').reshape(n_grid, n_grid)

    kde     = gaussian_kde(coords.T, bw_method=0.15)
    density = kde(grid_pts.T).reshape(n_grid, n_grid)
    valid   = ~np.isnan(GU)
    if valid.any():
        min_density = np.percentile(density[valid], min_density_pct)
        low_density = density < min_density
        GU[low_density] = np.nan
        GV[low_density] = np.nan

    return {'coords': coords, 'ip': ip_ko, 'GX': GX, 'GY': GY, 'GU': GU, 'GV': GV}


# --------------------------------------------------------------------------------------------- #
# First-order (Jacobian) knockout readouts
#
# All three readouts below are different summaries of the same finite-difference pass, so they
# share one core rather than each re-differencing the solver. Column g of the Jacobian is the
# central difference of the fitted solver's own dynamics, which is exact for the fitted field and
# does not require re-deriving the Hill derivative.
# --------------------------------------------------------------------------------------------- #
def jacobian_knockout_response(
    adata: AnnData,
    cluster_key: str,
    genes: Optional[List[str]] = None,
    lineage_pairs: Optional[List[tuple]] = None,
    groups: Optional[Dict[str, List[str]]] = None,
    spliced_key: str = 'Ms',
    eps: float = 1e-2,
) -> Dict[str, Dict]:
    """First-order knockout response from the fitted Jacobian, in three summaries.

    Knocking gene ``g`` out removes its drive, so to first order every other gene's production
    rate moves by :math:`r_i = -J_{ig} x_g`. Positive means gene ``i`` rises when ``g`` is knocked
    out, that is, a repression was lifted; negative means an activation was lost. This is a
    gene-space prediction, with no embedding projection anywhere in it.

    Parameters
    ----------
    adata : AnnData
        Fitted object.
    cluster_key : str
        ``adata.obs`` column holding the cell-type assignment.
    genes : list of str, optional
        Genes to knock out. Defaults to the genes named in ``groups`` that resolve to a decision
        axis, which is the set the commitment push is defined for.
    lineage_pairs : list of tuple, optional
        Decisions as ``(A_clusters, B_clusters, A_name, B_name)``. Required for the commitment
        push, which needs an axis to project onto. Decision ``k`` is tied to the ``k`` th entry of
        ``groups``.
    groups : dict, optional
        ``{group name: [gene, ...]}`` tying each gene to the decision it belongs to, in the same
        order as ``lineage_pairs``.
    spliced_key : str, default 'Ms'
        Layer holding the expression state.
    eps : float, default 1e-2
        Central-difference step used for the Jacobian column.

    Returns
    -------
    dict
        ``response``
            ``{gene: Series over target genes}``, the cell-averaged response, with the knocked-out
            gene dropped from its own Series.
        ``response_by_celltype``
            ``{gene: DataFrame of targets by cell type}``, the same response resolved per cluster.
        ``commitment_push``
            ``{gene: (A_name, B_name, per-cell push)}``, the response projected on that gene's own
            decision axis :math:`d = \\mathrm{normalize}(\\bar{x}_A - \\bar{x}_B)`. Positive means
            the immediate molecular response nudges that cell toward arm A. Empty when no
            ``lineage_pairs`` or ``groups`` are given.

    Notes
    -----
    This is the *immediate* response, before any propagation through the network. It answers which
    genes a knockout changes, not where cells end up; the fate readouts in
    :mod:`scHopfield.tools.fate` answer the second question.
    """
    from ..dynamics.solver import create_solver          # local: dynamics imports tools

    genes_used = get_genes_used(adata)
    names = list(np.asarray(adata.var_names)[genes_used])
    X = np.asarray(adata.layers[spliced_key])[:, genes_used].astype(float)
    clusters = adata.obs[cluster_key].astype(str).values

    # tie each named gene to the decision axis it is a driver of
    group_names = list(groups or {})
    gene_pair: Dict[str, int] = {}
    axis: Dict[int, tuple] = {}
    for k, (A, B, An, Bn) in enumerate(lineage_pairs or []):
        if k >= len(group_names):
            break
        in_a = np.isin(clusters, [str(c) for c in A])
        in_b = np.isin(clusters, [str(c) for c in B])
        if in_a.sum() == 0 or in_b.sum() == 0:
            continue
        d = X[in_a].mean(0) - X[in_b].mean(0)
        axis[k] = (d / (np.linalg.norm(d) + 1e-12), An, Bn)
        for gene in (groups or {}).get(group_names[k], []):
            gene_pair[gene] = k

    if genes is None:
        targets = [g for g in gene_pair if g in names and gene_pair[g] in axis]
    else:
        targets = [g for g in dict.fromkeys(genes) if g in names]

    resp_sum = {g: np.zeros(len(names)) for g in targets}
    resp_ct: Dict[str, Dict[str, np.ndarray]] = {g: {} for g in targets}
    push = {g: np.zeros(len(clusters)) for g in targets}
    n_cells = 0
    for cluster in pd.unique(clusters):
        sel = np.where(clusters == str(cluster))[0]
        try:
            solver = create_solver(adata, str(cluster), spliced_key=spliced_key)
        except Exception:
            continue
        Xc = X[sel]
        for g in targets:
            gi = names.index(g)
            x_plus = Xc.copy()
            x_plus[:, gi] += eps
            x_minus = Xc.copy()
            x_minus[:, gi] -= eps
            jcol = (solver.dynamics_batch(x_plus, 0.0)
                    - solver.dynamics_batch(x_minus, 0.0)) / (2 * eps)
            r = -jcol * Xc[:, gi][:, None]              # (n_cells_in_cluster, n_targets)
            resp_sum[g] += r.sum(0)
            resp_ct[g][str(cluster)] = r.mean(0)
            k = gene_pair.get(g)
            if k in axis:
                push[g][sel] = r @ axis[k][0]
        n_cells += len(sel)

    response = {g: pd.Series(resp_sum[g] / max(n_cells, 1), index=names)
                .drop(labels=[g], errors='ignore') for g in targets}
    response_ct = {g: pd.DataFrame(resp_ct[g], index=names) for g in targets}
    commitment_push = {g: (axis[gene_pair[g]][1], axis[gene_pair[g]][2], push[g])
                       for g in targets if gene_pair.get(g) in axis}
    return {'response': response, 'response_by_celltype': response_ct,
            'commitment_push': commitment_push}


def jacobian_response(
    adata: AnnData,
    cluster_key: str,
    genes: List[str],
    spliced_key: str = 'Ms',
    eps: float = 1e-2,
) -> Dict[str, pd.Series]:
    """Cell-averaged first-order knockout response, :math:`r_i = -J_{ig} x_g`, per gene.

    The ``response`` summary of :func:`jacobian_knockout_response`, which see for the definition.

    Returns
    -------
    dict
        ``{gene: Series of signed mean response over target genes}``, the knocked-out gene dropped
        from its own Series.
    """
    return jacobian_knockout_response(adata, cluster_key, genes=genes,
                                      spliced_key=spliced_key, eps=eps)['response']


def jacobian_commitment_push(
    adata: AnnData,
    cluster_key: str,
    lineage_pairs: List[tuple],
    groups: Dict[str, List[str]],
    spliced_key: str = 'Ms',
    eps: float = 1e-2,
) -> Dict[str, tuple]:
    """Per-cell first-order response projected onto each gene's own lineage-decision axis.

    The ``commitment_push`` summary of :func:`jacobian_knockout_response`, which see. This is the
    readout that can be painted on the embedding: it ties "which genes change" to a direction on
    the decision the gene belongs to, complementing the propagated fate map.

    Returns
    -------
    dict
        ``{gene: (A_name, B_name, per-cell push)}``. Positive pushes that cell toward arm A.
    """
    return jacobian_knockout_response(adata, cluster_key, genes=None,
                                      lineage_pairs=lineage_pairs, groups=groups,
                                      spliced_key=spliced_key, eps=eps)['commitment_push']


# --------------------------------------------------------------------------------------------- #
# Combinatorial perturbation
# --------------------------------------------------------------------------------------------- #
def double_knockout_matrix(
    adata: AnnData,
    cluster_key: str,
    scaffold: Dict,
    genes: List[str],
    axes: Dict[tuple, Dict],
    workers: int = 1,
) -> tuple:
    """Fate shift for every single and every pair drawn from ``genes``, and their synergy.

    The synergy is what makes the pair worth measuring:

    .. math::
        s_{gh} = \\Delta_{gh} - (\\Delta_g + \\Delta_h)

    so a pair whose joint effect is exactly the sum of its parts scores zero. Positive means the
    two knockouts reinforce each other beyond additivity, negative that they cancel. Every shift
    is the decider-cell mean change in the A-versus-B split fraction, the same readout the single
    knockout screen reports, so singles and doubles are directly comparable.

    Parameters
    ----------
    adata : AnnData
        Fitted object.
    cluster_key : str
        ``adata.obs`` column holding the cell-type assignment.
    scaffold : dict
        From :func:`scHopfield.tools.fate_scaffold`.
    genes : list of str
        Genes forming the matrix. Every unordered pair is evaluated.
    axes : dict
        From :func:`scHopfield.tools.lineage_pair_axes`, one entry per decision to score.
    workers : int, default 1
        CPU processes for the fate evaluations.

    Returns
    -------
    blocks, fate_single, fate_double : dict, dict, dict
        ``blocks`` maps each decision to ``genes``, ``matrix`` (off-diagonal double-knockout
        shift, diagonal single), ``synergy_matrix``, ``single`` (Series) and ``synergy``
        (``{(g, h): value}``). The two fate dictionaries are returned so a caller can reuse the
        per-cell probabilities without recomputing them.
    """
    import itertools

    from .fate import perturbed_fates, split_fraction

    gene_list = list(dict.fromkeys(genes))
    pairs = list(itertools.combinations(gene_list, 2))
    tasks = [[g] for g in gene_list] + [list(p) for p in pairs]
    evaluated = perturbed_fates(adata, cluster_key, scaffold, tasks, workers)
    fate_single = {g: evaluated[i] for i, g in enumerate(gene_list)}
    fate_double = {p: evaluated[len(gene_list) + i] for i, p in enumerate(pairs)}

    def decider_shift(fate, ax):
        return float((split_fraction(fate, ax['Ac'], ax['Bc'])
                      - ax['split_wt'])[ax['focus']].mean())

    n = len(gene_list)
    idx = {g: i for i, g in enumerate(gene_list)}
    blocks = {}
    for pair_key, ax in axes.items():
        single = pd.Series({g: decider_shift(fate_single[g], ax) for g in gene_list})
        shift_m = np.full((n, n), np.nan)
        syn_m = np.full((n, n), np.nan)
        for g in gene_list:
            shift_m[idx[g], idx[g]] = single[g]
        synergy = {}
        for (g1, g2), fate in fate_double.items():
            d = decider_shift(fate, ax)
            s = d - (single[g1] + single[g2])
            shift_m[idx[g1], idx[g2]] = shift_m[idx[g2], idx[g1]] = d
            syn_m[idx[g1], idx[g2]] = syn_m[idx[g2], idx[g1]] = s
            synergy[(g1, g2)] = s
        blocks[pair_key] = dict(genes=gene_list, matrix=shift_m, synergy_matrix=syn_m,
                                single=single, synergy=synergy)
    return blocks, fate_single, fate_double


# --------------------------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------------------------- #
def fate_bias_candidates(
    adata: AnnData,
    cluster_key: str,
    lineage_pairs: List[tuple],
    genes: List[str],
    n_per_arm: int = 6,
) -> List[str]:
    """Curated genes plus, per decision, the strongest structural driver of each arm.

    A screen reported over curated regulators alone cannot show whether the curated set is
    actually the strongest candidate the data offers. Adding the top structural drivers of both
    arms puts the curated genes in that context without letting the selection depend on the
    perturbation result being reported.

    Returns
    -------
    list of str
        Deduplicated, order-preserving, restricted to measured genes.
    """
    candidates = list(genes)
    for A, B, An, Bn in lineage_pairs:
        scored = score_driver_tfs(adata, A, B, cluster_key=cluster_key)
        candidates += list(scored[scored.lineage_bias > 0]
                           .sort_values('score_A', ascending=False).head(n_per_arm).index)
        candidates += list(scored[scored.lineage_bias <= 0]
                           .sort_values('score_B', ascending=False).head(n_per_arm).index)
    return [g for g in dict.fromkeys(candidates) if g in adata.var_names]


def select_specificity_wings(
    driver_scores: pd.DataFrame,
    out_strength: pd.Series,
    exclude: Optional[List[str]] = None,
    q: float = 95.0,
    pool_per_wing: int = 6,
) -> Dict[str, List[str]]:
    """Lineage-specific candidate regulators: the specificity wings, not the generalist corner.

    A gene scoring above threshold on *both* lineage axes is a generalist, and the Pareto corner
    is full of them. The genes that discriminate a decision are the ones above threshold on one
    axis and below it on the other, which is what this keeps. Pure sinks are dropped first, since
    a gene with no outgoing edges cannot propagate a knockout however well it scores.

    Parameters
    ----------
    driver_scores : pandas.DataFrame
        Indexed by gene, with columns ``score_A`` and ``score_B``.
    out_strength : pandas.Series
        Per-gene regulatory out-strength; see
        :func:`scHopfield.tools.regulatory_out_strength`. Only strictly positive entries survive.
    exclude : list of str, optional
        Genes already spoken for, such as the known regulators or picks from an earlier decision.
    q : float, default 95.0
        Percentile of each score defining that axis's threshold.
    pool_per_wing : int, default 6
        Genes kept per wing, ranked by specificity ``score_A - score_B`` on the A wing and the
        negation on the B wing.

    Returns
    -------
    dict
        ``{"A": [gene, ...], "B": [gene, ...]}``.
    """
    thr_a = np.percentile(driver_scores.score_A, q)
    thr_b = np.percentile(driver_scores.score_B, q)
    is_regulator = out_strength.reindex(driver_scores.index).fillna(0).values > 0
    eligible = driver_scores[is_regulator & ~driver_scores.index.isin(exclude or [])].copy()

    wing_a = eligible[(eligible.score_A > thr_a) & (eligible.score_B <= thr_b)].copy()
    wing_b = eligible[(eligible.score_B > thr_b) & (eligible.score_A <= thr_a)].copy()
    wing_a['spec'] = wing_a.score_A - wing_a.score_B
    wing_b['spec'] = wing_b.score_B - wing_b.score_A
    return {
        'A': list(wing_a.sort_values('spec', ascending=False).index[:pool_per_wing]),
        'B': list(wing_b.sort_values('spec', ascending=False).index[:pool_per_wing]),
    }


def rank_by_fate_effect(
    lineage_pairs: List[tuple],
    pools: Dict[int, Dict[str, List[str]]],
    fate_bias: Dict[tuple, Dict[str, pd.Series]],
    per_pair: int = 3,
    alpha: float = 0.05,
) -> tuple:
    """Keep, per decision, the probed candidates with the strongest measured fate effect.

    Selection is by the perturbation result rather than by structural driver score, so a gene that
    scores well structurally but does nothing when knocked out does not survive. Significant
    candidates are preferred; when too few clear ``alpha`` the ranking falls back to the largest
    absolute effect, so a decision always yields a selection. No directional balance is imposed:
    an axis whose strong drivers all point one way should show them that way.

    Parameters
    ----------
    lineage_pairs : list of tuple
        Decisions, in the same order as the keys of ``pools``.
    pools : dict
        ``{decision index: {"A": [gene, ...], "B": [gene, ...]}}``, from
        :func:`select_specificity_wings`.
    fate_bias : dict
        From :func:`scHopfield.tools.pairwise_fate_bias`.
    per_pair : int, default 3
        Genes kept per decision.
    alpha : float, default 0.05
        Significance threshold on the screen's Wilcoxon p-value.

    Returns
    -------
    genes, groups : list of str, dict
        The selection in decision order, and ``{"A_name vs B_name": [gene, ...]}``. A gene is
        never selected twice across decisions.
    """
    selected: List[str] = []
    groups: Dict[str, List[str]] = {}
    used: set = set()
    for k, (A, B, An, Bn) in enumerate(lineage_pairs):
        pool = pools.get(k, {}).get('A', []) + pools.get(k, {}).get('B', [])
        bias = fate_bias.get((An, Bn), {}).get('bias', pd.Series(dtype=float))
        pvals = fate_bias.get((An, Bn), {}).get('pvals', pd.Series(dtype=float))
        candidates = [g for g in pool if g in bias.index and g not in used]
        significant = [g for g in candidates if float(pvals.get(g, 1.0)) < alpha]
        ranked = sorted(significant or candidates,
                        key=lambda g: abs(float(bias.get(g, 0.0))), reverse=True)
        picks = ranked[:per_pair]
        groups[f"{An} vs {Bn}"] = picks
        used.update(picks)
        selected += picks
    return selected, groups
