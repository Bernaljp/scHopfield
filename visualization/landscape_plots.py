"""
Landscape visualization functionality for scHopfield package.
Contains plotting functions for energy landscapes and related analyses.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Any
from mpl_toolkits.mplot3d import Axes3D

from ..utils.utilities import to_numpy, sigmoid, fit_sigmoid


class LandscapePlotter:
    """
    Plotter for energy landscape visualizations.

    This class provides methods for plotting energy landscapes, fitted functions,
    and landscape-related analyses.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the landscape plotter.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def plot_fit(self, gene: str, color_clusters: bool = False,
                ax: Optional[plt.Axes] = None, **fig_kws) -> plt.Axes:
        """
        Plot the fit of a sigmoid function to the expression data of a specified gene.

        Args:
            gene: The gene to plot
            color_clusters: If True, color points by cluster
            ax: The axes on which to draw the plot
            **fig_kws: Additional keyword arguments for plot customization

        Returns:
            The axes containing the plot
        """
        # Retrieve gene index and sort the expression data
        gene_index = self.analyzer.gene_names.get_loc(gene)
        adata_index = self.analyzer.genes[gene_index]
        gexp = to_numpy(self.analyzer.get_matrix(self.analyzer.spliced_matrix_key, genes=[adata_index])).flatten()
        expression_data = np.sort(gexp)
        empirical_cdf = np.linspace(0, 1, len(expression_data))

        # Initialize plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, **fig_kws)
        ax.set_title(gene)

        # Plot original gene expression data
        if color_clusters:
            for cluster in self.analyzer.adata.obs[self.analyzer.cluster_key].unique():
                cluster_indices = self.analyzer.adata.obs[self.analyzer.cluster_key] == cluster
                ax.scatter(expression_data[cluster_indices], empirical_cdf[cluster_indices],
                          label=f'Expression: {cluster}')
        else:
            ax.plot(expression_data, empirical_cdf, '.-', label='Gene expression',
                   color=fig_kws.get('c1', 'k'))

        # Plot fitted curve
        threshold = self.analyzer.threshold[gene_index]
        exponent = self.analyzer.exponent[gene_index]
        offset = self.analyzer.offset[gene_index]
        fitted_curve = sigmoid(expression_data, threshold, exponent) * (1 - offset) + offset

        ax.plot(expression_data, fitted_curve, '.-', label='Fit',
               color=fig_kws.get('c2', 'r'))

        # Add sigmoid formula text
        sigmoid_formula = r"$\frac{{x^{{{:.2f}}}}}{{x^{{{:.2f}}} + {:.2f}^{{{:.2f}}}}}$".format(
            exponent, exponent, threshold, exponent)
        ax.text(0.8, 0.4, sigmoid_formula, transform=ax.transAxes, fontsize=14)

        ax.legend(loc='lower right')
        return ax

    def plot_landscape_embedding(self, which: str = 'UMAP', resolution: int = 50,
                                **kwargs) -> None:
        """
        Compute and visualize the energy embedding for the dataset.

        Args:
            which: The embedding method used
            resolution: The resolution of the grid for energy computation
            **kwargs: Additional keyword arguments for the embedding method
        """
        # This would call the analyzer's energy_embedding method
        # and create visualizations of the resulting landscape
        if hasattr(self.analyzer, 'energy_embedding'):
            self.analyzer.energy_embedding(which, resolution, **kwargs)
        else:
            print("Energy embedding not available. Please compute embedding first.")

    def plot_correlation_landscape(self, gene: str, energy: str = 'total',
                                 clusters: Union[str, List] = 'all',
                                 basis: str = 'umap', **kwargs) -> plt.Figure:
        """
        Plot gene correlation with energy landscape.

        Args:
            gene: Gene name to plot
            energy: Type of energy correlation
            clusters: Clusters to include
            basis: Embedding basis
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib figure object
        """
        # This would implement gene correlation plotting
        # using the analyzer's correlation methods
        fig, ax = plt.subplots(figsize=(8, 6))

        # Implementation would depend on available correlation data
        ax.set_title(f'{gene} - {energy} Energy Correlation')
        ax.set_xlabel(f'{basis.upper()} 1')
        ax.set_ylabel(f'{basis.upper()} 2')

        return fig

    def plot_energy_decomposition(self, cluster: str = 'all', n_genes: int = 5,
                                ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot energy decomposition for top contributing genes.

        Args:
            cluster: Cluster to analyze
            n_genes: Number of top genes to show
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Get energy contributions from different sources
        if hasattr(self.analyzer, 'degradation_energy_decomposed'):
            deg_energy = self.analyzer.degradation_energy_decomposed(cluster)
            bias_energy = self.analyzer.bias_energy_decomposed(cluster)
            int_energy = self.analyzer.interaction_energy_decomposed(cluster)

            # Calculate mean contributions
            mean_deg = np.mean(np.abs(deg_energy), axis=0)
            mean_bias = np.mean(np.abs(bias_energy), axis=0)
            mean_int = np.mean(np.abs(int_energy), axis=0)

            # Get top contributing genes
            total_contribution = mean_deg + mean_bias + mean_int
            top_indices = np.argsort(total_contribution)[-n_genes:][::-1]
            top_genes = self.analyzer.gene_names[top_indices]

            # Create stacked bar plot
            x = range(len(top_genes))
            ax.bar(x, mean_deg[top_indices], label='Degradation', alpha=0.8)
            ax.bar(x, mean_bias[top_indices], bottom=mean_deg[top_indices],
                  label='Bias', alpha=0.8)
            ax.bar(x, mean_int[top_indices],
                  bottom=mean_deg[top_indices] + mean_bias[top_indices],
                  label='Interaction', alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(top_genes, rotation=45, ha='right')
            ax.set_ylabel('Energy Contribution')
            ax.set_title(f'Energy Decomposition - {cluster}')
            ax.legend()

        return ax

    def plot_landscape_comparison(self, clusters: List[str], energy: str = 'total',
                                basis: str = 'umap', **kwargs) -> plt.Figure:
        """
        Compare energy landscapes between different clusters.

        Args:
            clusters: List of clusters to compare
            energy: Type of energy to compare
            basis: Embedding basis
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib figure object
        """
        n_clusters = len(clusters)
        fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5))
        if n_clusters == 1:
            axes = [axes]

        for i, cluster in enumerate(clusters):
            ax = axes[i]

            # Plot energy landscape for this cluster
            if hasattr(self.analyzer, 'grid_energy'):
                energy_dict = {
                    'total': self.analyzer.grid_energy,
                    'interaction': self.analyzer.grid_energy_interaction,
                    'degradation': self.analyzer.grid_energy_degradation,
                    'bias': self.analyzer.grid_energy_bias
                }

                if cluster in energy_dict[energy]:
                    gX = self.analyzer.grid_X[cluster]
                    gY = self.analyzer.grid_Y[cluster]
                    E = energy_dict[energy][cluster]

                    contour = ax.contourf(gX, gY, E, levels=20, cmap='viridis')
                    plt.colorbar(contour, ax=ax)

            ax.set_title(f'{cluster} - {energy.capitalize()} Energy')
            ax.set_xlabel(f'{basis.upper()} 1')
            ax.set_ylabel(f'{basis.upper()} 2')

        plt.tight_layout()
        return fig

    def plot_parameter_distribution(self, parameter: str = 'threshold',
                                  ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot distribution of fitted parameters.

        Args:
            parameter: Parameter to plot ('threshold', 'exponent', 'offset')
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Get parameter values
        if parameter == 'threshold' and hasattr(self.analyzer, 'threshold'):
            values = self.analyzer.threshold
        elif parameter == 'exponent' and hasattr(self.analyzer, 'exponent'):
            values = self.analyzer.exponent
        elif parameter == 'offset' and hasattr(self.analyzer, 'offset'):
            values = self.analyzer.offset
        else:
            ax.text(0.5, 0.5, f'Parameter {parameter} not available',
                   ha='center', va='center', transform=ax.transAxes)
            return ax

        # Create histogram
        ax.hist(values, bins=kwargs.get('bins', 30), alpha=0.7,
               color=kwargs.get('color', 'blue'))

        ax.set_xlabel(parameter.capitalize())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {parameter.capitalize()} Parameters')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_parameter_correlation(self, param1: str = 'threshold', param2: str = 'exponent',
                                 ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot correlation between fitted parameters.

        Args:
            param1: First parameter
            param2: Second parameter
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Get parameter values
        params = {}
        for param in [param1, param2]:
            if param == 'threshold' and hasattr(self.analyzer, 'threshold'):
                params[param] = self.analyzer.threshold
            elif param == 'exponent' and hasattr(self.analyzer, 'exponent'):
                params[param] = self.analyzer.exponent
            elif param == 'offset' and hasattr(self.analyzer, 'offset'):
                params[param] = self.analyzer.offset
            else:
                ax.text(0.5, 0.5, f'Parameter {param} not available',
                       ha='center', va='center', transform=ax.transAxes)
                return ax

        # Create scatter plot
        ax.scatter(params[param1], params[param2], alpha=0.6, **kwargs)

        # Calculate and display correlation
        correlation = np.corrcoef(params[param1], params[param2])[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white'))

        ax.set_xlabel(param1.capitalize())
        ax.set_ylabel(param2.capitalize())
        ax.set_title(f'{param1.capitalize()} vs {param2.capitalize()}')
        ax.grid(True, alpha=0.3)

        return ax


class JacobianPlotter:
    """
    Plotter for Jacobian and stability analysis visualizations.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the Jacobian plotter.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def plot_jacobian_summary(self, fig_size: tuple = (15, 5), part: str = 'real',
                            show: bool = False) -> None:
        """
        Plot summary of Jacobian eigenvalue analysis.

        Args:
            fig_size: Figure size
            part: Part of eigenvalues to analyze ('real' or 'imag')
            show: Whether to show the plot
        """
        if not hasattr(self.analyzer, 'jacobian_eigenvalues'):
            print("Jacobian eigenvalues not computed. Please run compute_jacobians first.")
            return

        # Store temporary variables in adata.obs
        self.analyzer.adata.obs['eval_positive_tmp'] = np.sum(
            np.real(self.analyzer.jacobian_eigenvalues) > 0, axis=1)
        self.analyzer.adata.obs['eval_negative_tmp'] = np.sum(
            np.real(self.analyzer.jacobian_eigenvalues) < 0, axis=1)

        if part == 'real':
            self.analyzer.adata.obs['eval_mean_tmp'] = np.mean(
                np.real(self.analyzer.jacobian_eigenvalues), axis=1)
        else:
            self.analyzer.adata.obs['eval_mean_tmp'] = np.mean(
                np.imag(self.analyzer.jacobian_eigenvalues[:, ::2]), axis=1)

        fig, axs = plt.subplots(1, 3, figsize=fig_size)

        # Create scatter plots (simplified version without dynamo dependency)
        basis = 'umap' if f'X_umap' in self.analyzer.adata.obsm else 'X_pca'
        if basis in self.analyzer.adata.obsm:
            coords = self.analyzer.adata.obsm[basis][:, :2]

            scatter1 = axs[0].scatter(coords[:, 0], coords[:, 1],
                                    c=self.analyzer.adata.obs['eval_positive_tmp'],
                                    cmap='viridis')
            axs[0].set_title('Number of positive eigenvalues\nJacobian')
            plt.colorbar(scatter1, ax=axs[0])

            scatter2 = axs[1].scatter(coords[:, 0], coords[:, 1],
                                    c=self.analyzer.adata.obs['eval_negative_tmp'],
                                    cmap='viridis')
            axs[1].set_title('Number of negative eigenvalues\nJacobian')
            plt.colorbar(scatter2, ax=axs[1])

            scatter3 = axs[2].scatter(coords[:, 0], coords[:, 1],
                                    c=self.analyzer.adata.obs['eval_mean_tmp'],
                                    cmap='RdBu_r')
            axs[2].set_title(f'Mean of {part} part of eigenvalues\nJacobian')
            plt.colorbar(scatter3, ax=axs[2])

        if show:
            plt.show()

        # Clean up temporary variables
        del self.analyzer.adata.obs['eval_positive_tmp']
        del self.analyzer.adata.obs['eval_negative_tmp']
        del self.analyzer.adata.obs['eval_mean_tmp']

    def plot_eigenvalue_distribution(self, cluster: str = 'all',
                                   ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot distribution of eigenvalues for a cluster.

        Args:
            cluster: Cluster to analyze
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if not hasattr(self.analyzer, 'jacobian_eigenvalues'):
            ax.text(0.5, 0.5, 'Jacobian eigenvalues not computed',
                   ha='center', va='center', transform=ax.transAxes)
            return ax

        # Get eigenvalues for the cluster
        if cluster == 'all':
            eigenvals = self.analyzer.jacobian_eigenvalues
        else:
            cluster_mask = self.analyzer.adata.obs[self.analyzer.cluster_key] == cluster
            eigenvals = self.analyzer.jacobian_eigenvalues[cluster_mask]

        # Plot real vs imaginary parts
        real_parts = np.real(eigenvals).flatten()
        imag_parts = np.imag(eigenvals).flatten()

        ax.scatter(real_parts, imag_parts, alpha=0.5, **kwargs)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)

        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title(f'Eigenvalue Distribution - {cluster}')
        ax.grid(True, alpha=0.3)

        return ax