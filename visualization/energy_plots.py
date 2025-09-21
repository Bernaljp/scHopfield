"""
Energy visualization functionality for scHopfield package.
Contains plotting functions for energy landscapes and energy-related analyses.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List, Dict, Union, Any
from mpl_toolkits.mplot3d import Axes3D

try:
    from ..utils.utilities import soften
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.utilities import soften


class EnergyPlotter:
    """
    Plotter for energy-related visualizations.

    This class provides methods for plotting energy landscapes, energy distributions,
    and energy-related correlations.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the energy plotter.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def plot_energy_boxplots(self, order: Optional[List] = None, plot_energy: str = 'all',
                           colors: Optional[Union[List, Dict]] = None, **fig_kws) -> np.ndarray:
        """
        Plot the energy distributions for different clusters using boxplots.

        Args:
            order: Order of clusters to display in the boxplots
            plot_energy: Type of energy to plot ('all', 'interaction', 'degradation', 'bias')
            colors: Colors for the clusters
            **fig_kws: Additional keyword arguments for plot customization

        Returns:
            Array of matplotlib axes objects
        """
        if plot_energy == 'all':
            fig, axs = plt.subplots(2, 2, **fig_kws)
            axs[0, 0].set_title('Total Energy')
            axs[0, 1].set_title('Interaction Energy')
            axs[1, 0].set_title('Degradation Energy')
            axs[1, 1].set_title('Bias Energy')
            axs = axs.flatten()

            es = [self.analyzer.E, self.analyzer.E_interaction,
                  self.analyzer.E_degradation, self.analyzer.E_bias]
        else:
            fig, axs = plt.subplots(1, 1, **fig_kws)
            axs = np.array([axs])
            es = [getattr(self.analyzer, f'E_{plot_energy.lower()}')]

        if order is None:
            order = self.analyzer.adata.obs[self.analyzer.cluster_key].unique()
            if colors is not None:
                assert isinstance(colors, list) and len(colors) >= len(order), \
                    "Colors should be a list of length at least equal to the number of clusters."
                plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
        else:
            if colors is not None:
                assert (isinstance(colors, list) or isinstance(colors, dict)) and len(colors) >= len(order), \
                    "Colors should be a list of length at least equal to the number of clusters."
            colors = colors if isinstance(colors, dict) else {k: colors[i] for i, k in enumerate(order)}
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[colors[i] for i in order])

        for energy, ax in zip(es, axs):
            df = pd.DataFrame.from_dict(energy, orient='index').transpose().melt(
                var_name='Cluster', value_name='Energy').dropna()
            sns.boxplot(data=df, x='Cluster', y='Energy', order=order, ax=ax)

        plt.tight_layout()
        return axs

    def plot_energy_scatters(self, basis: str = 'umap', order: Optional[List] = None,
                           plot_energy: str = 'all', show_legend: bool = False, **fig_kws) -> None:
        """
        Plot the energy landscapes for different clusters using 3D scatter plots.

        Args:
            basis: The basis used for embedding, default is 'umap'
            order: Order of clusters to display
            plot_energy: Type of energy to plot ('all', 'interaction', 'degradation', 'bias')
            show_legend: Whether to show legend
            **fig_kws: Additional keyword arguments for plot customization
        """
        order = self.analyzer.adata.obs[self.analyzer.cluster_key].unique() if order is None else order

        if plot_energy == 'all':
            fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, **fig_kws)
            # Set titles
            axs[0, 0].set_title('Total Energy')
            axs[0, 1].set_title('Interaction Energy')
            axs[1, 0].set_title('Degradation Energy')
            axs[1, 1].set_title('Bias Energy')

            axs = axs.flatten()
            es = [self.analyzer.E, self.analyzer.E_interaction,
                  self.analyzer.E_degradation, self.analyzer.E_bias]
        else:
            fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, **fig_kws)

            axs = np.array([axs])
            es = [getattr(self.analyzer, f'E_{plot_energy.lower()}')]

        for k in order:
            cells = self.analyzer.adata.obsm[f'X_{basis}'][self.analyzer.adata.obs[self.analyzer.cluster_key] == k][:, :2]
            for ax, energy_type in zip(axs, es):
                energies = energy_type[k]
                ax.scatter(*cells.T, energies, label=k)

        if show_legend:
            plt.legend()

        plt.tight_layout()

    def plot_energy_surface_2d(self, clusters: Union[str, List] = 'all', energy: str = 'total',
                             basis: str = 'UMAP', plot_cells: bool = True,
                             ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """
        Plot a 2D contour plot of the energy surface, optionally overlaying cell locations.

        Args:
            clusters: Cluster(s) to be plotted. Use 'all' for all clusters
            energy: Type of energy to plot ('total', 'interaction', 'degradation', 'bias')
            basis: Embedding basis to use for cell locations
            plot_cells: Whether to overlay cell locations on the plot
            ax: Matplotlib axis object for plotting. If None, a new figure is created
            **kwargs: Additional keyword arguments for contour plotting and cell overlay

        Returns:
            Matplotlib axes object containing the plot
        """
        # Energy selection based on the 'energy' argument
        energy_dict = {'total': self.analyzer.grid_energy,
                      'interaction': self.analyzer.grid_energy_interaction,
                      'degradation': self.analyzer.grid_energy_degradation,
                      'bias': self.analyzer.grid_energy_bias}
        energy_data = energy_dict.get(energy, self.analyzer.grid_energy)

        if clusters == 'all':
            clusters = self.analyzer.clusters

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the energy surface for each cluster
        for cluster in clusters:
            if cluster in energy_data:
                gX, gY = self.analyzer.grid_X[cluster], self.analyzer.grid_Y[cluster]
                E = energy_data[cluster]
                ax.contourf(gX, gY, E, levels=kwargs.get('levels', 20),
                           cmap=kwargs.get('cmap', 'viridis'))

        # Optionally overlay cell locations
        if plot_cells:
            for cluster in clusters:
                if cluster != 'all':
                    cluster_idx = self.analyzer.adata.obs[self.analyzer.cluster_key] == cluster
                    cells2d = self.analyzer.adata.obsm[f'X_{basis}'][cluster_idx]
                    ax.scatter(cells2d[:, 0], cells2d[:, 1], label=cluster,
                              edgecolor='k', linewidth=0.5, s=10)

        ax.set_title(f'{energy.capitalize()} Energy Surface')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend(title='Cluster')

        plt.colorbar(ax.collections[-1], ax=ax, label='Energy')
        return ax

    def plot_energy_surface_3d(self, clusters: Union[str, List] = 'all', energy: str = 'total',
                             basis: str = 'UMAP', plot_cells: bool = True,
                             ax: Optional[Axes3D] = None, **kwargs) -> Axes3D:
        """
        Plot a 3D surface plot of the energy landscape, with the option to overlay cell locations.

        Args:
            clusters: Cluster(s) to be plotted. Use 'all' for all clusters
            energy: Type of energy to plot ('total', 'interaction', 'degradation', 'bias')
            basis: Embedding basis to use for cell locations
            plot_cells: Whether to overlay cell locations on the plot
            ax: Matplotlib 3D axis object for plotting. If None, a new figure is created
            **kwargs: Additional keyword arguments for surface plotting and cell overlay

        Returns:
            Matplotlib 3D axes object containing the plot
        """
        # Energy selection based on the 'energy' argument
        energy_dict = {'total': self.analyzer.grid_energy,
                      'interaction': self.analyzer.grid_energy_interaction,
                      'degradation': self.analyzer.grid_energy_degradation,
                      'bias': self.analyzer.grid_energy_bias}
        energy_data = energy_dict.get(energy, self.analyzer.grid_energy)

        if clusters == 'all':
            clusters = self.analyzer.clusters

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Plot the energy surface for each cluster
        for cluster in clusters:
            if cluster in energy_data:
                gX, gY = self.analyzer.grid_X[cluster], self.analyzer.grid_Y[cluster]
                E = energy_data[cluster]
                ax.plot_surface(gX, gY, E, cmap=kwargs.get('cmap', 'viridis'),
                               edgecolor='none', alpha=0.7)

        # Optionally overlay cell locations
        if plot_cells:
            for cluster in clusters:
                if cluster != 'all':
                    cluster_idx = self.analyzer.adata.obs[self.analyzer.cluster_key] == cluster
                    cells2d = self.analyzer.adata.obsm[f'X_{basis}'][cluster_idx]
                    # Would need to implement rezet function for proper 3D positioning
                    ax.scatter(cells2d[:, 0], cells2d[:, 1], 0, label=cluster,
                              edgecolor='k', s=5)

        ax.set_title(f'{energy.capitalize()} Energy Landscape')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Energy')
        ax.legend(title='Cluster')

        return ax

    def plot_energy_trajectory(self, trajectory: np.ndarray, energies: np.ndarray,
                             time_points: np.ndarray, ax: Optional[plt.Axes] = None,
                             **kwargs) -> plt.Axes:
        """
        Plot energy changes along a trajectory.

        Args:
            trajectory: System trajectory
            energies: Energy values along trajectory
            time_points: Time points
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(time_points, energies, **kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Evolution Along Trajectory')
        ax.grid(True)

        return ax


class EnergyCorrelationPlotter:
    """
    Plotter for energy-gene correlation visualizations.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the correlation plotter.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def plot_gene_correlation_scatter(self, clus1: str, clus2: str, annotate: Optional[int] = None,
                                    clus1_low: float = -0.5, clus1_high: float = 0.5,
                                    clus2_low: float = -0.5, clus2_high: float = 0.5,
                                    energy: str = 'total', ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Creates a scatter plot comparing the gene correlations with energy landscapes between two clusters.

        Args:
            clus1: The name of the first cluster for comparison
            clus2: The name of the second cluster for comparison
            annotate: Number of genes to annotate
            clus1_low, clus1_high: Threshold values for first cluster
            clus2_low, clus2_high: Threshold values for second cluster
            energy: Specifies the type of energy landscape
            ax: A matplotlib axes object to plot on

        Returns:
            The axes object with the scatter plot
        """
        # Retrieve the correlation data for each cluster based on the specified energy type
        corr1 = getattr(self.analyzer, f'correlation_{energy}')[clus1] if energy.lower() != 'total' else self.analyzer.correlation[clus1]
        corr2 = getattr(self.analyzer, f'correlation_{energy}')[clus2] if energy.lower() != 'total' else self.analyzer.correlation[clus2]

        # Create a new figure and axes if none are provided
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)

        # Set the limits for the axes
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))

        positions_corners = np.logical_or(
            np.logical_and(corr1 >= clus1_high, corr2 <= clus2_low),
            np.logical_and(corr1 <= clus1_low, corr2 >= clus2_high))

        # Identify correlations that are in opposite corners of the plot
        corr_corners = np.where(positions_corners)[0]

        # Identify correlations that are not in the opposite corners
        corr_center = np.where(~positions_corners)[0]

        # Plot the correlations using different colors for clarity
        ax.scatter(corr1[corr_corners], corr2[corr_corners], c='k', s=0.6, label='Divergent Correlations')
        ax.scatter(corr1[corr_center], corr2[corr_center], c='lightgray', s=0.5, label='Other Correlations')

        if annotate is not None:
            nn = annotate
            # Get top genes with the highest absolute correlation values
            cor_indices = np.argsort((corr1[corr_corners])**2 + (corr2[corr_corners])**2)[-nn:]
            # Get the names of the top genes with the highest absolute correlation values
            gois = self.analyzer.gene_names[corr_corners][cor_indices]
            # Adding labels for the top genes with the highest absolute correlation values
            arrow_dict = {"width": 0.5, "headwidth": 0.5, "headlength": 1, "color": "gray"}
            for gg, xx, yy in zip(gois, corr1[corr_corners][cor_indices], corr2[corr_corners][cor_indices]):
                rand_shift_1 = np.random.uniform(-0.08, 0.08)
                rand_shift_2 = np.random.uniform(-0.08, 0.08)
                ax.annotate(gg, xy=(xx, yy), xytext=(xx+rand_shift_1, yy+rand_shift_2), arrowprops=arrow_dict)

        # Add reference lines and labels
        ax.vlines([clus1_low, clus1_high], ymin=-1, ymax=1, linestyles='dashed', color='r')
        ax.hlines([clus2_low, clus2_high], xmin=-1, xmax=1, linestyles='dashed', color='r')
        ax.set_xlabel(clus1)
        ax.set_ylabel(clus2)

        return ax

    def plot_correlations_grid(self, colors: Optional[Dict] = None, order: Optional[List] = None,
                             energy: str = 'total', x_low: float = -0.5, x_high: float = 0.5,
                             y_low: float = -0.5, y_high: float = 0.5, **kwargs) -> None:
        """
        Plots a matrix where the diagonal shows cell types and the off-diagonal
        plots show gene correlation scatter plots between cell types.

        Args:
            colors: Dictionary mapping cell types to colors
            order: Order of cell types to display
            energy: Type of energy correlation to plot
            x_low, x_high: Threshold values for x-axis
            y_low, y_high: Threshold values for y-axis
            **kwargs: Additional plotting arguments
        """
        cell_types = self.analyzer.adata.obs[self.analyzer.cluster_key].unique() if order is None else order
        n = len(cell_types)
        figsize = kwargs.get('figsize', (15, 15))
        tight_layout = kwargs.get('tight_layout', True)
        fig, axs = plt.subplots(n, n, figsize=figsize, tight_layout=tight_layout)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    for spine in axs[i, j].spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(2)
                        if colors is not None:
                            spine.set_color(colors[cell_types[i]])
                    # Remove ticks
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    # Add text in the middle
                    text = cell_types[i]
                    text = text.replace(' ', '\n', 1)
                    text = text.replace('-', '-\n')
                    axs[i, j].text(0.5, 0.5, text, ha='center', va='center', fontsize=18,
                                  fontweight='bold', fontname='serif', transform=axs[i, j].transAxes)

                    if colors is not None:
                        c = list(colors[cell_types[i]])
                        c[-1] = 0.2  # Assuming 'colors' values are RGBA
                        axs[i, j].set_facecolor(c)
                    continue

                axs[i, j].axis('off')
                self.plot_gene_correlation_scatter(clus1=cell_types[i], clus2=cell_types[j],
                                                 energy=energy, ax=axs[j, i],
                                                 clus1_low=x_low, clus1_high=x_high,
                                                 clus2_low=y_low, clus2_high=y_high)
                # Remove ticks
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
                axs[j, i].set_xlabel('')
                axs[j, i].set_ylabel('')
                # Adjust ticks for the first column and last row
                if i == 0:
                    axs[j, i].set_yticks([-1, -0.5, 0, 0.5, 1])
                if j == n - 1:
                    axs[j, i].set_xticks([-1, -0.5, 0, 0.5, 1])