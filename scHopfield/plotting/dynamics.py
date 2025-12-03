"""Plotting functions for dynamics and trajectories."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_trajectory(
    trajectory: np.ndarray,
    t_span: np.ndarray,
    gene_names: Optional[list] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot gene expression trajectories over time.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory array (n_timepoints × n_genes)
    t_span : np.ndarray
        Time points
    gene_names : list, optional
        Gene names to label
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(trajectory.shape[1]):
        label = gene_names[i] if gene_names and i < len(gene_names) else f'Gene {i}'
        ax.plot(t_span, trajectory[:, i], label=label, alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Expression')
    ax.set_title('Gene Expression Trajectory')
    if trajectory.shape[1] <= 10:
        ax.legend()

    return ax
