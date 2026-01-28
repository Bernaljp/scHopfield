"""Network comparison and analysis."""

import numpy as np
import pandas as pd
import itertools
from typing import Optional, Dict, Union, List, Tuple
from anndata import AnnData

from .._utils.io import get_genes_used


def network_correlations(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute various similarity metrics between cluster interaction networks.

    Computes pairwise similarity metrics between cluster-specific interaction
    matrices, allowing comparison of gene regulatory network structures across
    different cell types.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interaction matrices
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
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
    threshold_number: Optional[int] = 2000,
    weight: str = 'coef_abs',
    use_igraph: bool = True,
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute network centrality metrics for genes in each cluster.

    Uses igraph (fast C implementation) by default, with NetworkX as fallback.
    Filters network edges before computing centrality for better performance.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interaction matrices
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    threshold_number : int, optional (default: 2000)
        Maximum number of edges to keep per cluster (top edges by weight).
        Set to None to use all edges (slower).
    weight : str, optional (default: 'coef_abs')
        Weight column to use for filtering: 'coef_abs' or 'coef_mean'
    use_igraph : bool, optional (default: True)
        Use igraph library if available (much faster than NetworkX).
        Falls back to NetworkX if igraph is not installed.
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.var for each cluster:
        - 'degree_all_{cluster}': total degree
        - 'degree_centrality_all_{cluster}': normalized total degree
        - 'degree_in_{cluster}': in-degree
        - 'degree_centrality_in_{cluster}': normalized in-degree
        - 'degree_out_{cluster}': out-degree
        - 'degree_centrality_out_{cluster}': normalized out-degree
        - 'betweenness_centrality_{cluster}': betweenness centrality
        - 'eigenvector_centrality_{cluster}': eigenvector centrality

    Notes
    -----
    This implementation is based on CellOracle's approach for efficiency.
    Edge filtering significantly speeds up computation for large networks.
    """
    adata = adata.copy() if copy else adata

    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]
    clusters = adata.obs[cluster_key].unique()

    # Try to use igraph, fall back to NetworkX
    if use_igraph:
        try:
            from igraph import Graph
            use_igraph_lib = True
        except ImportError:
            print("Warning: igraph not found, falling back to NetworkX (slower).")
            print("For better performance, install igraph: pip install python-igraph")
            use_igraph_lib = False
            try:
                import networkx as nx
            except ImportError:
                raise ImportError(
                    "Neither igraph nor NetworkX is installed. "
                    "Install one of them: pip install python-igraph"
                )
    else:
        use_igraph_lib = False
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required. Install it with: pip install networkx"
            )

    for cluster in clusters:
        if cluster == 'all':
            continue

        # Get filtered network links efficiently
        W = adata.varp[f'W_{cluster}']

        # Convert to DataFrame efficiently
        df = pd.DataFrame(W.T, index=gene_names, columns=gene_names).reset_index()
        df = df.melt(
            id_vars='index',
            value_vars=df.columns[1:],
            var_name='target',
            value_name='coef_mean'
        )

        # Filter non-zero edges
        df = df[df['coef_mean'] != 0].copy()
        df.rename(columns={'index': 'source'}, inplace=True)
        df['coef_abs'] = np.abs(df['coef_mean'])

        # Apply threshold filtering
        if threshold_number is not None and len(df) > threshold_number:
            df = df.sort_values(weight, ascending=False).head(threshold_number)

        if use_igraph_lib:
            # Use igraph (fast)
            # First create graph with all genes to ensure all vertices exist
            g = Graph(directed=True)
            g.add_vertices(list(gene_names))

            # Add edges from filtered DataFrame
            edges = [(row['source'], row['target']) for _, row in df.iterrows()]
            g.add_edges(edges)
            g.es["weight"] = df["coef_abs"].values

            # Calculate centrality scores (returns list in vertex order)
            n_vertices = len(gene_names)
            result_df = pd.DataFrame(index=gene_names)

            for mode in ["all", "in", "out"]:
                degrees = g.degree(mode=mode)
                result_df[f"degree_{mode}"] = degrees
                result_df[f"degree_centrality_{mode}"] = np.array(degrees) / (n_vertices - 1)

            result_df["betweenness_centrality"] = g.betweenness(directed=True, weights="weight")
            result_df["eigenvector_centrality"] = g.eigen_centrality(directed=False, weights="weight")

        else:
            # Use NetworkX (fallback)
            G = nx.from_pandas_edgelist(
                df,
                source='source',
                target='target',
                edge_attr='coef_abs',
                create_using=nx.DiGraph()
            )

            # Ensure all genes are in the graph
            G.add_nodes_from(gene_names)

            # Rename edge attribute to 'weight'
            nx.set_edge_attributes(G, {(u, v): d['coef_abs'] for u, v, d in G.edges(data=True)}, 'weight')

            # Calculate centrality scores
            result_df = pd.DataFrame(index=gene_names)

            degree_all = dict(G.degree())
            degree_in = dict(G.in_degree())
            degree_out = dict(G.out_degree())
            n_nodes = G.number_of_nodes()

            result_df['degree_all'] = [degree_all.get(g, 0) for g in gene_names]
            result_df['degree_in'] = [degree_in.get(g, 0) for g in gene_names]
            result_df['degree_out'] = [degree_out.get(g, 0) for g in gene_names]
            result_df['degree_centrality_all'] = result_df['degree_all'] / (n_nodes - 1)
            result_df['degree_centrality_in'] = result_df['degree_in'] / (n_nodes - 1)
            result_df['degree_centrality_out'] = result_df['degree_out'] / (n_nodes - 1)

            betweenness = nx.betweenness_centrality(G, weight='weight')
            eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

            result_df['betweenness_centrality'] = [betweenness.get(g, 0) for g in gene_names]
            result_df['eigenvector_centrality'] = [eigenvector.get(g, 0) for g in gene_names]

        # Store results in adata.var
        for col in result_df.columns:
            col_name = f'{col}_{cluster}'
            if col_name not in adata.var.columns:
                adata.var[col_name] = 0.0
            adata.var.loc[gene_names, col_name] = result_df[col].values

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


def compute_eigenanalysis(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute eigenvalue decomposition for each cluster's interaction matrix.

    Performs eigendecomposition (eigenvalues and eigenvectors) on the
    interaction matrices W for each cluster. Results are stored in
    adata.uns['scHopfield']['eigenanalysis'].

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interaction matrices
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.uns['scHopfield']['eigenanalysis'] for each cluster:
        - 'eigenvalues_{cluster}': complex eigenvalues array
        - 'eigenvectors_{cluster}': complex eigenvectors matrix

    Notes
    -----
    Eigenvalues with large positive real parts indicate unstable directions.
    Eigenvalues with large negative real parts indicate fast decay directions.
    The eigenvectors show which gene combinations are associated with these dynamics.
    """
    adata = adata.copy() if copy else adata

    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]
    clusters = adata.obs[cluster_key].unique()

    if 'eigenanalysis' not in adata.uns['scHopfield']:
        adata.uns['scHopfield']['eigenanalysis'] = {}

    for cluster in clusters:
        if cluster == 'all':
            continue

        # Get interaction matrix
        W = adata.varp[f'W_{cluster}']

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(W)

        # Store results
        adata.uns['scHopfield']['eigenanalysis'][f'eigenvalues_{cluster}'] = eigenvalues
        adata.uns['scHopfield']['eigenanalysis'][f'eigenvectors_{cluster}'] = eigenvectors
        adata.uns['scHopfield']['eigenanalysis'][f'gene_names'] = gene_names.values

    return adata if copy else None


def get_top_eigenvector_genes(
    adata: AnnData,
    cluster: str,
    which: str = 'max',
    n_genes: int = 20,
    part: str = 'real',
    cluster_key: str = 'cell_type'
) -> pd.DataFrame:
    """
    Get top genes from eigenvector corresponding to extreme eigenvalue.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed eigenanalysis
    cluster : str
        Cluster name
    which : str, optional (default: 'max')
        Which eigenvalue to use: 'max' for largest real part,
        'min' for smallest (most negative) real part
    n_genes : int, optional (default: 20)
        Number of top genes to return
    part : str, optional (default: 'real')
        Which part of eigenvector to use: 'real', 'imag', or 'abs'
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels (for validation)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'gene' and 'component_value'

    Raises
    ------
    ValueError
        If eigenanalysis has not been computed yet
    """
    if 'eigenanalysis' not in adata.uns['scHopfield']:
        raise ValueError(
            "Eigenanalysis not found. Please run sch.tl.compute_eigenanalysis() first."
        )

    # Get eigenvalues and eigenvectors
    eigenvalues = adata.uns['scHopfield']['eigenanalysis'][f'eigenvalues_{cluster}']
    eigenvectors = adata.uns['scHopfield']['eigenanalysis'][f'eigenvectors_{cluster}']
    gene_names = adata.uns['scHopfield']['eigenanalysis']['gene_names']

    # Select eigenvalue
    if which == 'max':
        idx = np.argmax(eigenvalues.real)
    elif which == 'min':
        idx = np.argmin(eigenvalues.real)
    else:
        raise ValueError("which must be 'max' or 'min'")

    eigenvalue = eigenvalues[idx]
    eigenvector = eigenvectors[:, idx]

    # Select component part
    if part == 'real':
        components = eigenvector.real
    elif part == 'imag':
        components = eigenvector.imag
    elif part == 'abs':
        components = np.abs(eigenvector)
    else:
        raise ValueError("part must be 'real', 'imag', or 'abs'")

    # Get top genes by absolute value
    sorted_idx = np.argsort(np.abs(components))[::-1][:n_genes]

    result = pd.DataFrame({
        'gene': gene_names[sorted_idx],
        'component_value': components[sorted_idx],
        'eigenvalue': eigenvalue
    })

    return result


def get_eigenanalysis_table(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    n_genes: int = 20,
    order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create formatted table with top genes from extreme eigenvectors.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed eigenanalysis
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    n_genes : int, optional (default: 20)
        Number of top genes to include per cluster
    order : list, optional
        Order of clusters in table. If None, uses all clusters

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns (cluster, ['+EV gene', '+EV value', '-EV gene', '-EV value'])
    """
    if 'eigenanalysis' not in adata.uns['scHopfield']:
        raise ValueError(
            "Eigenanalysis not found. Please run sch.tl.compute_eigenanalysis() first."
        )

    clusters = adata.obs[cluster_key].unique().tolist()
    if order is not None:
        clusters = [c for c in order if c in clusters]

    # Create result DataFrame
    result = pd.DataFrame(
        index=range(n_genes),
        columns=pd.MultiIndex.from_product([
            clusters,
            ['+EV gene', '+EV value', '-EV gene', '-EV value']
        ])
    )

    for cluster in clusters:
        # Get top genes from max eigenvalue eigenvector
        df_max = get_top_eigenvector_genes(adata, cluster, which='max', n_genes=n_genes)
        result[(cluster, '+EV gene')] = df_max['gene'].values
        result[(cluster, '+EV value')] = [f"{v:.3f}" for v in df_max['component_value'].values]

        # Get top genes from min eigenvalue eigenvector
        df_min = get_top_eigenvector_genes(adata, cluster, which='min', n_genes=n_genes)
        result[(cluster, '-EV gene')] = df_min['gene'].values
        result[(cluster, '-EV value')] = [f"{v:.3f}" for v in df_min['component_value'].values]

    return result
