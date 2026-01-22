"""Network comparison and analysis."""

import numpy as np
import pandas as pd
import itertools
from typing import Optional, Dict, Union, List
from anndata import AnnData

from .._utils.io import get_genes_used


def network_correlations(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute various similarity metrics between cluster interaction networks.

    Adapted from Landscape.network_correlations.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interaction matrices
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.uns['scHopfield']['network_correlations']:
        - 'jaccard': Jaccard index (binary overlap)
        - 'hamming': Hamming distance (binary difference)
        - 'euclidean': Euclidean distance (continuous)
        - 'pearson': Pearson correlation (continuous)
        - 'pearson_bin': Pearson correlation (binary)
        - 'mean_col_corr': Mean column-wise correlation
        - 'singular': Singular value distance
    """
    adata = adata.copy() if copy else adata

    keys = adata.obs[cluster_key].unique()

    # Initialize result DataFrames
    jaccard = pd.DataFrame(index=keys, columns=keys, data=1.0)
    hamming = pd.DataFrame(index=keys, columns=keys, data=0.0)
    pearson = pd.DataFrame(index=keys, columns=keys, data=1.0)
    pearson_bin = pd.DataFrame(index=keys, columns=keys, data=1.0)
    euclidean = pd.DataFrame(index=keys, columns=keys, data=0.0)
    mean_col = pd.DataFrame(index=keys, columns=keys, data=1.0)
    singular = pd.DataFrame(index=keys, columns=keys, data=0.0)

    # Compute singular values for each network
    svs = {k: np.linalg.svd(adata.varp[f'W_{k}'], compute_uv=False) for k in keys}

    # Compute pairwise metrics
    for k1, k2 in itertools.combinations(keys, 2):
        w1 = adata.varp[f'W_{k1}']
        w2 = adata.varp[f'W_{k2}']
        bw1 = np.sign(w1)
        bw2 = np.sign(w2)

        # Pearson correlation
        pearson.loc[k1, k2] = pearson.loc[k2, k1] = np.corrcoef(w1.ravel(), w2.ravel())[0, 1]

        # Pearson correlation (binary)
        pearson_bin.loc[k1, k2] = pearson_bin.loc[k2, k1] = np.corrcoef(bw1.ravel(), bw2.ravel())[0, 1]

        # Euclidean distance
        euclidean.loc[k1, k2] = euclidean.loc[k2, k1] = np.linalg.norm(w1 - w2)

        # Hamming distance
        hamming.loc[k1, k2] = hamming.loc[k2, k1] = np.count_nonzero(bw1 != bw2)

        # Jaccard index
        intersection = np.logical_and(bw1, bw2)
        union = np.logical_or(bw1, bw2)
        jaccard.loc[k1, k2] = jaccard.loc[k2, k1] = intersection.sum() / union.sum()

        # Mean column-wise correlation
        mean_col_corr = np.mean(np.diag(np.corrcoef(w1, w2, rowvar=False)[:w1.shape[0], :w1.shape[0]]))
        mean_col.loc[k1, k2] = mean_col.loc[k2, k1] = mean_col_corr

        # Singular value distance
        singular.loc[k1, k2] = singular.loc[k2, k1] = np.linalg.norm(svs[k1] - svs[k2])

    # Store all metrics
    if 'network_correlations' not in adata.uns['scHopfield']:
        adata.uns['scHopfield']['network_correlations'] = {}

    adata.uns['scHopfield']['network_correlations']['jaccard'] = jaccard
    adata.uns['scHopfield']['network_correlations']['hamming'] = hamming
    adata.uns['scHopfield']['network_correlations']['euclidean'] = euclidean
    adata.uns['scHopfield']['network_correlations']['pearson'] = pearson
    adata.uns['scHopfield']['network_correlations']['pearson_bin'] = pearson_bin
    adata.uns['scHopfield']['network_correlations']['mean_col_corr'] = mean_col
    adata.uns['scHopfield']['network_correlations']['singular'] = singular

    return adata if copy else None


def get_network_links(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    return_format: str = 'dict'
) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Extract network links (edges) from interaction matrices.

    Converts interaction matrices (W) into edge list format suitable for
    network analysis. Each edge represents a gene-gene interaction with
    its coefficient value.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interaction matrices
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    return_format : str, optional (default: 'dict')
        Return format: 'dict' returns dict of DataFrames per cluster,
        'combined' returns single DataFrame with cluster column

    Returns
    -------
    dict or pd.DataFrame
        If 'dict': Dictionary mapping cluster names to DataFrames with columns:
            - source: source gene
            - target: target gene
            - coef_mean: interaction coefficient
            - coef_abs: absolute value of coefficient
        If 'combined': Single DataFrame with additional 'cluster' column
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]
    clusters = adata.obs[cluster_key].unique()

    links = {}
    for k in clusters:
        if k == 'all':
            continue

        # Get interaction matrix
        W = adata.varp[f'W_{k}']

        # Convert to DataFrame and melt to long format
        df = pd.DataFrame(W.T, index=gene_names, columns=gene_names).reset_index()
        df = df.melt(
            id_vars='index',
            value_vars=df.columns[1:],
            var_name='target',
            value_name='coef_mean'
        )

        # Filter out zero interactions
        df = df[df['coef_mean'] != 0].copy()

        # Rename and add columns
        df.rename(columns={'index': 'source'}, inplace=True)
        df['coef_abs'] = np.abs(df['coef_mean'])
        df['cluster'] = k

        links[k] = df

    if return_format == 'combined':
        return pd.concat(links.values(), ignore_index=True)
    else:
        return links


def compute_network_centrality(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    metrics: Optional[list] = None,
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute network centrality metrics for genes in each cluster.

    Computes various centrality measures using NetworkX on the
    interaction networks. Results are stored in adata.var with
    cluster-specific columns.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interaction matrices
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    metrics : list, optional
        List of centrality metrics to compute. Options:
        - 'degree_centrality'
        - 'in_degree_centrality'
        - 'out_degree_centrality'
        - 'betweenness_centrality'
        - 'closeness_centrality'
        - 'eigenvector_centrality'
        - 'pagerank'
        If None, computes all metrics
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.var for each cluster and metric:
        - '{metric}_{cluster}': centrality values for each gene
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for centrality computation. "
            "Install it with: pip install networkx"
        )

    adata = adata.copy() if copy else adata

    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]
    clusters = adata.obs[cluster_key].unique()

    if metrics is None:
        metrics = [
            'degree_centrality',
            'in_degree_centrality',
            'out_degree_centrality',
            'betweenness_centrality',
            'eigenvector_centrality',
            'pagerank'
        ]

    for cluster in clusters:
        if cluster == 'all':
            continue

        # Get interaction matrix
        W = adata.varp[f'W_{cluster}']

        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from(gene_names)

        # Add edges with non-zero weights
        for i, gene_i in enumerate(gene_names):
            for j, gene_j in enumerate(gene_names):
                if W[i, j] != 0:
                    G.add_edge(gene_i, gene_j, weight=abs(W[i, j]))

        # Compute centrality metrics
        for metric in metrics:
            try:
                if metric == 'degree_centrality':
                    cent = nx.degree_centrality(G)
                elif metric == 'in_degree_centrality':
                    cent = nx.in_degree_centrality(G)
                elif metric == 'out_degree_centrality':
                    cent = nx.out_degree_centrality(G)
                elif metric == 'betweenness_centrality':
                    cent = nx.betweenness_centrality(G, weight='weight')
                elif metric == 'closeness_centrality':
                    cent = nx.closeness_centrality(G, distance='weight')
                elif metric == 'eigenvector_centrality':
                    cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                elif metric == 'pagerank':
                    cent = nx.pagerank(G, weight='weight')
                else:
                    print(f"Warning: Unknown metric '{metric}', skipping...")
                    continue

                # Store in adata.var
                col_name = f'{metric}_{cluster}'
                if col_name not in adata.var.columns:
                    adata.var[col_name] = 0.0
                adata.var.loc[gene_names, col_name] = [cent[g] for g in gene_names]

            except Exception as e:
                print(f"Warning: Could not compute {metric} for {cluster}: {e}")

    return adata if copy else None


def get_top_genes_table(
    adata: AnnData,
    metric: str,
    cluster_key: str = 'cell_type',
    n_genes: int = 20,
    order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create formatted table with top genes per cluster for a given metric.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed centrality metrics
    metric : str
        Centrality metric to use
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    n_genes : int, optional (default: 20)
        Number of top genes to include per cluster
    order : list, optional
        Order of clusters in table. If None, uses all clusters

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns (cluster, ['Gene', metric_name])
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    clusters = adata.obs[cluster_key].unique().tolist()
    if order is not None:
        clusters = [c for c in order if c in clusters]

    # Create result DataFrame
    metric_display = metric.replace('_', ' ').title()
    result = pd.DataFrame(
        index=range(n_genes),
        columns=pd.MultiIndex.from_product([clusters, ['Gene', metric_display]])
    )

    for cluster in clusters:
        col_name = f'{metric}_{cluster}'
        if col_name not in adata.var.columns:
            print(f"Warning: No {metric} data for {cluster}, skipping...")
            continue

        # Get scores and sort
        scores = adata.var[col_name].values[genes]
        sorted_idx = np.argsort(scores)[::-1][:n_genes]
        top_genes = gene_names[sorted_idx]
        top_scores = scores[sorted_idx]

        # Fill table
        result[(cluster, 'Gene')] = top_genes.values
        result[(cluster, metric_display)] = top_scores

    return result
