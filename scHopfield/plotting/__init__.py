"""Plotting module for visualization functions."""

from .energy import plot_energy_landscape, plot_energy_boxplots, plot_energy_scatters
from .genes import plot_sigmoid_fit
from .networks import (
    plot_interaction_matrix,
    plot_centrality_scatter,
    plot_eigenvalue_spectrum,
    plot_eigenvector_components,
    plot_eigenanalysis_grid,
    plot_grn_network,
)
from .dynamics import plot_trajectory
from .correlation import (
    plot_gene_correlation_scatter,
    plot_correlations_grid
)
from .jacobian import (
    plot_jacobian_eigenvalue_spectrum,
    plot_jacobian_stats_boxplots,
    plot_jacobian_element_grid,
)
from .flow import plot_flow, plot_inner_product
from .tikz import (
    tikz_available,
    render_tikz,
    grn_preamble,
    grn_tikz_body,
    draw_grn,
    draw_grn_mpl,
    GRN_PREAMBLE,
)

__all__ = [
    # Energy visualization
    'plot_energy_landscape',
    'plot_energy_boxplots',
    'plot_energy_scatters',
    # Gene visualization
    'plot_sigmoid_fit',
    # Network visualization
    'plot_interaction_matrix',
    'plot_centrality_scatter',
    'plot_eigenvalue_spectrum',
    'plot_eigenvector_components',
    'plot_eigenanalysis_grid',
    'plot_grn_network',
    'plot_trajectory',
    # Correlation visualization
    'plot_gene_correlation_scatter',
    'plot_correlations_grid',
    # Jacobian visualization
    'plot_jacobian_eigenvalue_spectrum',
    'plot_jacobian_stats_boxplots',
    'plot_jacobian_element_grid',
    # Perturbation visualization
    'plot_flow',
    'plot_inner_product',
    'tikz_available',
    'render_tikz',
    'grn_preamble',
    'grn_tikz_body',
    'draw_grn',
    'draw_grn_mpl',
    'GRN_PREAMBLE',
]
