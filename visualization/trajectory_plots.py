"""
Trajectory visualization functionality for scHopfield package.
Contains plotting functions for trajectories, dynamics, and flow fields.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Union, Any, Tuple
from mpl_toolkits.mplot3d import Axes3D

from ..utils.utilities import to_numpy, sigmoid


class TrajectoryPlotter:
    """
    Plotter for trajectory and dynamics visualizations.

    This class provides methods for plotting cell trajectories, flow fields,
    and dynamical system behavior.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the trajectory plotter.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def plot_trajectory_2d(self, trajectory: np.ndarray, embedding_basis: str = 'umap',
                          ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot a 2D trajectory in embedding space.

        Args:
            trajectory: Trajectory data (n_timepoints, n_genes)
            embedding_basis: Embedding basis to use for projection
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Transform trajectory to embedding space if embedding is available
        if hasattr(self.analyzer, 'embedding'):
            trajectory_2d = self.analyzer.embedding.transform(np.clip(trajectory, 0, None))
        else:
            # Use first two genes as fallback
            trajectory_2d = trajectory[:, :2]

        # Plot trajectory
        ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], **kwargs)
        ax.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], c='green', s=100, label='Start')
        ax.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], c='red', s=100, label='End')

        ax.set_xlabel(f'{embedding_basis.upper()} 1')
        ax.set_ylabel(f'{embedding_basis.upper()} 2')
        ax.set_title('Cell Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_trajectory_3d(self, trajectory: np.ndarray, gene_indices: Tuple[int, int, int] = (0, 1, 2),
                          ax: Optional[Axes3D] = None, **kwargs) -> Axes3D:
        """
        Plot a 3D trajectory in gene expression space.

        Args:
            trajectory: Trajectory data (n_timepoints, n_genes)
            gene_indices: Indices of genes to use for 3D visualization
            ax: Matplotlib 3D axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib 3D axes object
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Extract gene trajectories
        x = trajectory[:, gene_indices[0]]
        y = trajectory[:, gene_indices[1]]
        z = trajectory[:, gene_indices[2]]

        # Plot trajectory
        ax.plot(x, y, z, **kwargs)
        ax.scatter(x[0], y[0], z[0], c='green', s=100, label='Start')
        ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, label='End')

        gene_names = self.analyzer.gene_names[list(gene_indices)]
        ax.set_xlabel(gene_names[0])
        ax.set_ylabel(gene_names[1])
        ax.set_zlabel(gene_names[2])
        ax.set_title('3D Gene Expression Trajectory')
        ax.legend()

        return ax

    def plot_phase_portrait(self, gene_indices: Tuple[int, int] = (0, 1), cluster: str = 'all',
                           resolution: int = 20, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot a phase portrait showing the flow field for two genes.

        Args:
            gene_indices: Indices of genes to use for the phase portrait
            cluster: Cluster to analyze
            resolution: Resolution of the flow field grid
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Get expression range for the selected genes
        expression_data = self.analyzer.get_matrix(self.analyzer.spliced_matrix_key, genes=self.analyzer.genes)

        # Validate gene indices
        n_genes = expression_data.shape[1]
        if gene_indices[0] >= n_genes:
            gene_indices = (0, min(1, n_genes-1))
        if gene_indices[1] >= n_genes:
            gene_indices = (gene_indices[0], min(gene_indices[0]+1, n_genes-1))

        # Handle case where there's only one gene
        if n_genes == 1:
            ax.text(0.5, 0.5, f'Phase portrait requires at least 2 genes\nOnly {n_genes} gene available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Phase Portrait - {cluster} (Insufficient genes)')
            return ax

        gene1_range = [np.min(expression_data[:, gene_indices[0]]), np.max(expression_data[:, gene_indices[0]])]
        gene2_range = [np.min(expression_data[:, gene_indices[1]]), np.max(expression_data[:, gene_indices[1]])]

        # Create grid
        x = np.linspace(gene1_range[0], gene1_range[1], resolution)
        y = np.linspace(gene2_range[0], gene2_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Initialize flow field arrays
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Get system parameters
        W = self.analyzer.W[cluster]
        gamma = (self.analyzer.adata.var[self.analyzer.gamma_key][self.analyzer.genes].values
                if not self.analyzer.refit_gamma else self.analyzer.gamma[cluster])
        I = self.analyzer.I[cluster]

        # Compute flow at each grid point
        for i in range(resolution):
            for j in range(resolution):
                # Create state vector (use mean values for other genes)
                state = np.mean(expression_data, axis=0)
                state[gene_indices[0]] = X[i, j]
                state[gene_indices[1]] = Y[i, j]

                # Compute derivatives
                sig = sigmoid(state, self.analyzer.threshold, self.analyzer.exponent)
                dydt = W @ sig - gamma * state + I

                # Store flow components for selected genes
                U[i, j] = dydt[gene_indices[0]]
                V[i, j] = dydt[gene_indices[1]]

        # Plot flow field
        ax.quiver(X, Y, U, V, alpha=0.7, **kwargs)

        # Overlay cell data
        cells = expression_data[:, gene_indices]
        cluster_mask = self.analyzer.adata.obs[self.analyzer.cluster_key] == cluster if cluster != 'all' else slice(None)
        ax.scatter(cells[cluster_mask, 0], cells[cluster_mask, 1], c='red', alpha=0.5, s=10)

        gene_names = self.analyzer.gene_names[list(gene_indices)]
        ax.set_xlabel(gene_names[0])
        ax.set_ylabel(gene_names[1])
        ax.set_title(f'Phase Portrait - {cluster}')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_multiple_trajectories(self, trajectories: List[np.ndarray], embedding_basis: str = 'umap',
                                 ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot multiple trajectories on the same plot.

        Args:
            trajectories: List of trajectory arrays
            embedding_basis: Embedding basis to use for projection
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

        for i, trajectory in enumerate(trajectories):
            # Transform trajectory to embedding space if embedding is available
            if hasattr(self.analyzer, 'embedding'):
                trajectory_2d = self.analyzer.embedding.transform(np.clip(trajectory, 0, None))
            else:
                # Use first two genes as fallback
                trajectory_2d = trajectory[:, :2]

            # Plot trajectory
            ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], color=colors[i],
                   label=f'Trajectory {i+1}', **kwargs)
            ax.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], c=colors[i], s=50, marker='o')
            ax.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], c=colors[i], s=50, marker='s')

        ax.set_xlabel(f'{embedding_basis.upper()} 1')
        ax.set_ylabel(f'{embedding_basis.upper()} 2')
        ax.set_title('Multiple Cell Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_gene_dynamics(self, trajectory: np.ndarray, time_points: np.ndarray,
                          gene_indices: Optional[List[int]] = None,
                          ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot gene expression dynamics over time.

        Args:
            trajectory: Trajectory data (n_timepoints, n_genes)
            time_points: Time points
            gene_indices: Indices of genes to plot (if None, plot all)
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if gene_indices is None:
            gene_indices = range(min(10, trajectory.shape[1]))  # Plot first 10 genes max

        for i in gene_indices:
            gene_name = self.analyzer.gene_names[i] if i < len(self.analyzer.gene_names) else f'Gene {i}'
            ax.plot(time_points, trajectory[:, i], label=gene_name, **kwargs)

        ax.set_xlabel('Time')
        ax.set_ylabel('Gene Expression')
        ax.set_title('Gene Expression Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class NetworkPlotter:
    """
    Plotter for network and interaction visualizations.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the network plotter.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def plot_interaction_matrix(self, cluster: str = 'all', ax: Optional[plt.Axes] = None,
                              **kwargs) -> plt.Axes:
        """
        Plot the interaction matrix as a heatmap.

        Args:
            cluster: Cluster to visualize
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        W = self.analyzer.W[cluster]

        # Create heatmap
        im = ax.imshow(W, cmap=kwargs.get('cmap', 'RdBu_r'),
                      vmin=kwargs.get('vmin', -np.max(np.abs(W))),
                      vmax=kwargs.get('vmax', np.max(np.abs(W))))

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Interaction Strength')

        # Set labels
        gene_names = self.analyzer.gene_names
        ax.set_xticks(range(len(gene_names)))
        ax.set_yticks(range(len(gene_names)))
        ax.set_xticklabels(gene_names, rotation=45, ha='right')
        ax.set_yticklabels(gene_names)

        ax.set_title(f'Interaction Matrix - {cluster}')
        ax.set_xlabel('Target Genes')
        ax.set_ylabel('Source Genes')

        return ax

    def plot_network_graph(self, cluster: str = 'all', threshold: float = 0.1,
                          ax: Optional[plt.Axes] = None, **kwargs):
        """
        Plot the gene regulatory network as a graph.

        Args:
            cluster: Cluster to visualize
            threshold: Threshold for showing interactions
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object or network graph (depending on available libraries)
        """
        try:
            import networkx as nx
            from matplotlib.patches import FancyArrowPatch
        except ImportError:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'NetworkX required for network visualization',
                   ha='center', va='center', transform=ax.transAxes)
            return ax

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        W = self.analyzer.W[cluster]
        gene_names = self.analyzer.gene_names

        # Create networkx graph
        G = nx.DiGraph()

        # Add nodes
        for i, gene in enumerate(gene_names):
            G.add_node(gene)

        # Add edges above threshold
        for i in range(len(gene_names)):
            for j in range(len(gene_names)):
                if abs(W[i, j]) > threshold:
                    G.add_edge(gene_names[j], gene_names[i], weight=W[i, j])

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                              node_size=kwargs.get('node_size', 500))

        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=kwargs.get('font_size', 8))

        # Draw edges with colors based on weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=weights,
                              edge_cmap=plt.cm.RdBu_r, arrows=True,
                              arrowsize=kwargs.get('arrowsize', 20))

        ax.set_title(f'Gene Regulatory Network - {cluster}')
        ax.axis('off')

        return ax

    def plot_gene_data_scatter(self, axis1: str, axis2: str, energy_type: str,
                             energy_type_2: Optional[str] = None, top_n_axis1: int = 1,
                             top_n_axis2: int = 1, order: Optional[List] = None,
                             interaction_direction: Optional[str] = None,
                             exclude_genes: List = [], show: bool = False,
                             **fig_kws) -> Tuple[plt.Figure, np.ndarray]:
        """
        Generalized plotting method for gene data scatter plots.

        Args:
            axis1: Data to plot on x-axis ('rate', 'expression', 'energy')
            axis2: Data to plot on y-axis ('rate', 'expression', 'energy')
            energy_type: Type of energy ('Total', 'Interaction', 'Degradation', 'Bias')
            energy_type_2: Type of energy for second axis (if different)
            top_n_axis1: Number of top genes to highlight on axis1
            top_n_axis2: Number of top genes to highlight on axis2
            order: List of cluster orders
            interaction_direction: Direction for interaction energy ('in' or 'out')
            exclude_genes: List of genes to exclude
            show: Whether to show the plot
            **fig_kws: Additional figure arguments

        Returns:
            Tuple of (figure, axes)
        """
        from collections import defaultdict

        order = self.analyzer.adata.obs[self.analyzer.cluster_key].unique() if order is None else order
        n_plots = len(order) + 1
        interaction_direction = 'in' if interaction_direction is None else interaction_direction
        energy_type_2 = energy_type if energy_type_2 is None else energy_type_2

        figsize = fig_kws.pop('figsize', (10, 6))
        fig, axs = plt.subplots(2, n_plots//2, figsize=figsize, tight_layout=True, sharex=True)

        # Prepare data based on axis1 and axis2
        plot_data = {'rate': {}, 'expression': {}, 'energy': {}}
        plot_labels = {'rate': {}, 'expression': {}, 'energy': {}}

        genes_in = np.where(~np.isin(self.analyzer.gene_names, exclude_genes))[0]

        # Implementation would continue with data preparation and plotting
        # This is a simplified version showing the structure

        if show:
            plt.show()
        else:
            return fig, axs