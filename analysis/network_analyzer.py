"""
Network analysis functionality for scHopfield package.

This module provides network-based analysis methods for gene regulatory networks
including centrality measures, motif analysis, and network structure analysis.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any

from ..core.base_models import BaseAnalyzer


class NetworkAnalyzer(BaseAnalyzer):
    """
    Analyzer for network-based analysis of gene regulatory networks.

    This class provides methods for computing network centrality measures,
    analyzing network structure, and visualizing network properties.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the network analyzer.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def get_links_dict(self) -> Dict[str, pd.DataFrame]:
        """
        Convert interaction matrices to network links format.

        Returns:
            Dictionary mapping cluster names to links DataFrames
        """
        links = {}
        for k in self.analyzer.W.keys():
            w = self.analyzer.W[k]
            links[k] = pd.DataFrame(w.T, index=self.analyzer.gene_names,
                                  columns=self.analyzer.gene_names).reset_index()
            links[k] = links[k].melt(id_vars='index', value_vars=links[k].columns,
                                   var_name='target', value_name='coef_mean')
            links[k] = links[k][links[k]['coef_mean'] != 0]
            links[k].rename(columns={'index': 'source'}, inplace=True)
            links[k]['coef_abs'] = np.abs(links[k]['coef_mean'])
            links[k]['p'] = 0
            links[k]['-logp'] = np.nan
            links[k]['cluster'] = k
        return links

    def compute_network_centralities(self) -> Dict[str, pd.DataFrame]:
        """
        Compute network centrality measures for each cluster.

        Returns:
            Dictionary mapping cluster names to centrality DataFrames
        """
        links_dict = self.get_links_dict()
        centrality_results = {}

        for cluster, links_df in links_dict.items():
            # Create directed graph
            G = nx.from_pandas_edgelist(links_df, source='source', target='target',
                                       edge_attr=['coef_mean', 'coef_abs'], create_using=nx.DiGraph())

            # Compute centrality measures
            try:
                degree_centrality = nx.degree_centrality(G)
                in_degree_centrality = nx.in_degree_centrality(G)
                out_degree_centrality = nx.out_degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

                # Create centrality dataframe
                genes = list(G.nodes())
                centrality_df = pd.DataFrame({
                    'gene': genes,
                    'degree_centrality_all': [degree_centrality.get(g, 0) for g in genes],
                    'degree_centrality_in': [in_degree_centrality.get(g, 0) for g in genes],
                    'degree_centrality_out': [out_degree_centrality.get(g, 0) for g in genes],
                    'betweenness_centrality': [betweenness_centrality.get(g, 0) for g in genes],
                    'eigenvector_centrality': [eigenvector_centrality.get(g, 0) for g in genes],
                    'cluster': cluster
                })

                centrality_results[cluster] = centrality_df

            except Exception as e:
                print(f"Error computing centralities for {cluster}: {e}")
                centrality_results[cluster] = pd.DataFrame()

        return centrality_results

    def plot_centrality_rankings(self, centrality_results: Dict[str, pd.DataFrame],
                                order: List[str], colors: Dict[str, Any],
                                score: str = 'eigenvector_centrality', n_genes: int = 10,
                                figsize: Tuple[int, int] = None) -> None:
        """
        Plot top genes by centrality score for each cluster.

        Args:
            centrality_results: Dictionary of centrality results
            order: Order of clusters to plot
            colors: Colors for each cluster
            score: Centrality score to plot
            n_genes: Number of top genes to show
            figsize: Figure size
        """
        if figsize is None:
            figsize = (5*len(order), 6)

        fig, axes = plt.subplots(1, len(order), figsize=figsize)
        if len(order) == 1:
            axes = [axes]

        for i, cluster in enumerate(order):
            if cluster in centrality_results and not centrality_results[cluster].empty:
                df = centrality_results[cluster].copy()
                df_sorted = df.nlargest(n_genes, score)

                y_pos = range(len(df_sorted))
                axes[i].barh(y_pos, df_sorted[score], color=colors[cluster], alpha=0.7)
                axes[i].set_yticks(y_pos)
                axes[i].set_yticklabels(df_sorted['gene'], fontsize=10)
                axes[i].set_xlabel(score.replace('_', ' ').title())
                axes[i].set_title(f'{cluster}\nTop {n_genes} Genes')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=axes[i].transAxes)
                axes[i].set_title(f'{cluster}\nNo Data')

        plt.tight_layout()
        plt.show()

    def get_centrality_table(self, centrality_results: Dict[str, pd.DataFrame],
                            order: List[str], score: str = 'eigenvector_centrality',
                            n_genes: int = 10) -> pd.DataFrame:
        """
        Create a table of top genes by centrality score.

        Args:
            centrality_results: Dictionary of centrality results
            order: Order of clusters
            score: Centrality score to use
            n_genes: Number of top genes

        Returns:
            DataFrame with top genes by centrality
        """
        df_centrality = pd.DataFrame(
            index=range(1, n_genes + 1),
            columns=pd.MultiIndex.from_product([order, ['Gene', 'Score']])
        )

        for cluster in order:
            if cluster in centrality_results and not centrality_results[cluster].empty:
                df = centrality_results[cluster].copy()
                df_sorted = df.nlargest(n_genes, score)

                df_centrality[cluster, 'Gene'] = df_sorted['gene'].values[:n_genes]
                df_centrality[cluster, 'Score'] = [f"{v:.4f}" for v in df_sorted[score].values[:n_genes]]

        return df_centrality

    def plot_networks_with_centrality(self, ls, centrality_results: Dict[str, pd.DataFrame],
                                    order: List[str], colors: Dict[str, Any],
                                    score: str = 'eigenvector_centrality', threshold: float = 0.01,
                                    max_nodes: int = 15, figsize: Tuple[int, int] = None) -> None:
        """
        Plot network graphs with node sizes based on centrality scores.

        Args:
            ls: LandscapeAnalyzer instance
            centrality_results: Dictionary of centrality results
            order: Order of clusters
            colors: Colors for clusters
            score: Centrality score for node sizing
            threshold: Threshold for edge display
            max_nodes: Maximum number of nodes to display
            figsize: Figure size
        """
        if figsize is None:
            figsize = (8 * ((len(order) + 1) // 2), 16)

        fig, axes = plt.subplots(2, (len(order) + 1) // 2, figsize=figsize)
        if len(order) <= 2:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()

        for i, cluster in enumerate(order):
            if i < len(axes) and cluster in centrality_results:
                self._plot_single_network_with_centrality(
                    ls, cluster, centrality_results[cluster], colors[cluster],
                    score, threshold, max_nodes, axes[i]
                )

        # Hide unused subplots
        for i in range(len(order), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def _plot_single_network_with_centrality(self, ls, cluster: str, centrality_df: pd.DataFrame,
                                           color: Any, score: str, threshold: float,
                                           max_nodes: int, ax: plt.Axes) -> plt.Axes:
        """
        Plot a single network graph with centrality-based node sizing.

        Args:
            ls: LandscapeAnalyzer instance
            cluster: Cluster name
            centrality_df: Centrality data for the cluster
            color: Color for the cluster
            score: Centrality score for node sizing
            threshold: Threshold for edge display
            max_nodes: Maximum number of nodes
            ax: Matplotlib axis

        Returns:
            Updated matplotlib axis
        """
        # Get interaction matrix
        W = ls.W[cluster]

        # Create graph
        G = nx.DiGraph()

        # Add nodes with centrality scores
        for i, gene in enumerate(ls.gene_names):
            if gene in centrality_df['gene'].values:
                centrality_score = centrality_df[centrality_df['gene'] == gene][score].iloc[0]
                G.add_node(gene, centrality=centrality_score)

        # Add edges above threshold
        for i in range(len(ls.gene_names)):
            for j in range(len(ls.gene_names)):
                if abs(W[i, j]) > threshold:
                    source = ls.gene_names[j]
                    target = ls.gene_names[i]
                    if source in G.nodes() and target in G.nodes():
                        G.add_edge(source, target, weight=W[i, j])

        # Keep only top nodes by centrality
        if len(G.nodes()) > max_nodes:
            node_centralities = [(node, G.nodes[node].get('centrality', 0)) for node in G.nodes()]
            node_centralities.sort(key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in node_centralities[:max_nodes]]
            G = G.subgraph(top_nodes)

        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, f'No network data for {cluster}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{cluster} - Network')
            return ax

        # Layout
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

        # Node sizes based on centrality
        node_sizes = [G.nodes[node].get('centrality', 0) * 3000 + 100 for node in G.nodes()]

        # Draw network
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color,
                              node_size=node_sizes, alpha=0.8)

        # Draw edges with weights
        edges = G.edges()
        if edges:
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=weights,
                                  edge_cmap=plt.cm.RdBu_r, arrows=True,
                                  arrowsize=20, width=2, alpha=0.7)

        # Draw labels for top nodes
        if len(G.nodes()) > 0:
            top_5_nodes = dict(sorted([(node, G.nodes[node].get('centrality', 0)) for node in G.nodes()],
                                     key=lambda x: x[1], reverse=True)[:5])
            labels = {node: node for node in top_5_nodes.keys()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold')

        ax.set_title(f'{cluster} - Top {len(G.nodes())} Genes by {score.replace("_", " ").title()}')
        ax.axis('off')

        return ax

    def analyze_expression_centrality_correlation(self, ls, centrality_results: Dict[str, pd.DataFrame],
                                                 spliced_key: str, cluster_key: str,
                                                 order: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Analyze correlation between gene expression and network centrality.

        Args:
            ls: LandscapeAnalyzer instance
            centrality_results: Dictionary of centrality results
            spliced_key: Key for spliced matrix
            cluster_key: Key for cluster information
            order: Order of clusters

        Returns:
            Dictionary of correlation results per cluster
        """
        correlation_results = {}

        for cluster in order:
            if cluster in centrality_results and not centrality_results[cluster].empty:
                # Get cluster cells
                cluster_mask = ls.adata.obs[cluster_key] == cluster
                cluster_expression = ls.get_matrix(spliced_key)[cluster_mask][:, ls.genes]

                # Calculate mean expression for each gene
                mean_expression = np.mean(cluster_expression, axis=0)

                # Get centrality scores
                centrality_df = centrality_results[cluster].set_index('gene')

                # Find common genes
                common_genes = set(ls.gene_names).intersection(set(centrality_df.index))

                if len(common_genes) > 5:  # Need enough genes for meaningful correlation
                    gene_indices = [i for i, gene in enumerate(ls.gene_names) if gene in common_genes]

                    correlations = {}
                    for score in ['degree_centrality_all', 'betweenness_centrality', 'eigenvector_centrality']:
                        if score in centrality_df.columns:
                            centrality_values = [centrality_df.loc[ls.gene_names[i], score]
                                               for i in gene_indices]
                            expression_values = mean_expression[gene_indices]

                            # Compute correlation
                            corr = np.corrcoef(expression_values, centrality_values)[0, 1]
                            correlations[score] = corr

                    correlation_results[cluster] = correlations

        return correlation_results

    def plot_expression_centrality_correlations(self, expression_centrality_corr: Dict[str, Dict[str, float]],
                                              order: List[str], figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot expression-centrality correlations as heatmap.

        Args:
            expression_centrality_corr: Correlation results
            order: Order of clusters
            figsize: Figure size
        """
        if not expression_centrality_corr:
            print("No expression-centrality correlation data computed")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data for plotting
        plot_data = []
        for cluster in order:
            if cluster in expression_centrality_corr:
                for score, corr in expression_centrality_corr[cluster].items():
                    plot_data.append({
                        'Cluster': cluster,
                        'Centrality': score.replace('_', ' ').title(),
                        'Correlation': corr
                    })

        if plot_data:
            df_corr = pd.DataFrame(plot_data)

            # Create pivot table for heatmap
            pivot_df = df_corr.pivot(index='Cluster', columns='Centrality', values='Correlation')

            # Plot heatmap
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title('Gene Expression vs Network Centrality Correlations')
            ax.set_xlabel('Centrality Measure')
            ax.set_ylabel('Cell Type')

            plt.tight_layout()
            plt.show()

            print("Expression-Centrality Correlation Summary:")
            for cluster in order:
                if cluster in expression_centrality_corr:
                    print(f"\n{cluster}:")
                    for score, corr in expression_centrality_corr[cluster].items():
                        print(f"  {score.replace('_', ' ').title()}: {corr:.3f}")
        else:
            print("No correlation data available for plotting")

    def analyze_network_motifs(self, motif_size: int = 3) -> Dict[str, Dict[str, Any]]:
        """
        Analyze network motifs in gene regulatory networks.

        Args:
            motif_size: Size of motifs to analyze

        Returns:
            Dictionary of motif analysis results per cluster
        """
        links_dict = self.get_links_dict()
        motif_results = {}

        for cluster, links_df in links_dict.items():
            if links_df.empty:
                continue

            # Create directed graph
            G = nx.from_pandas_edgelist(links_df, source='source', target='target',
                                       edge_attr='coef_mean', create_using=nx.DiGraph())

            # Basic network statistics
            stats = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'weakly_connected_components': nx.number_weakly_connected_components(G),
                'strongly_connected_components': nx.number_strongly_connected_components(G)
            }

            # Find cycles of different lengths
            try:
                simple_cycles = list(nx.simple_cycles(G, length_bound=5))
                stats['cycles_total'] = len(simple_cycles)
                stats['cycles_length_2'] = len([c for c in simple_cycles if len(c) == 2])
                stats['cycles_length_3'] = len([c for c in simple_cycles if len(c) == 3])
                stats['cycles_length_4'] = len([c for c in simple_cycles if len(c) == 4])
            except:
                stats['cycles_total'] = 0
                stats['cycles_length_2'] = 0
                stats['cycles_length_3'] = 0
                stats['cycles_length_4'] = 0

            # Clustering coefficient
            try:
                stats['clustering_coefficient'] = nx.average_clustering(G)
            except:
                stats['clustering_coefficient'] = 0

            motif_results[cluster] = stats

        return motif_results

    def plot_network_statistics(self, motif_analysis: Dict[str, Dict[str, Any]],
                               order: List[str], colors: Dict[str, Any],
                               figsize: Tuple[int, int] = (18, 12)) -> None:
        """
        Plot network structure statistics.

        Args:
            motif_analysis: Network motif analysis results
            order: Order of clusters
            colors: Colors for clusters
            figsize: Figure size
        """
        if not motif_analysis:
            print("No motif analysis data available")
            return

        motif_df = pd.DataFrame(motif_analysis).T
        motif_df = motif_df.reindex(order)

        print("\nNetwork Structure Analysis:")
        print(motif_df.round(4))

        # Plot network statistics
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        metrics = ['nodes', 'edges', 'density', 'cycles_total', 'clustering_coefficient', 'strongly_connected_components']
        titles = ['Number of Nodes', 'Number of Edges', 'Network Density', 'Total Cycles', 'Clustering Coefficient', 'Strong Components']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if i < len(axes) and metric in motif_df.columns:
                values = motif_df[metric].values
                axes[i].bar(range(len(order)), values, color=[colors[c] for c in order], alpha=0.7)
                axes[i].set_xticks(range(len(order)))
                axes[i].set_xticklabels(order, rotation=45, ha='right')
                axes[i].set_ylabel(title)
                axes[i].set_title(title)
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()