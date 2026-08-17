"""Plotting module for visualization functions."""

from .energy import (
    plot_energy_landscape,
    plot_energy_components,
    plot_energy_boxplots,
    plot_energy_scatters
)
from .genes import plot_sigmoid_fit
from .networks import (
    plot_interaction_matrix,
    plot_network_centrality_rank,
    plot_centrality_comparison,
    plot_gene_centrality,
    plot_centrality_scatter,
    plot_eigenvalue_spectrum,
    plot_eigenvector_components,
    plot_eigenanalysis_grid,
    plot_grn_network,
    plot_grn_subset
)
from .dynamics import plot_trajectory
from .correlation import (
    plot_gene_correlation_scatter,
    plot_correlations_grid
)
from .jacobian import (
    plot_jacobian_eigenvalue_spectrum,
    plot_jacobian_eigenvalue_boxplots,
    plot_jacobian_stats_boxplots,
    plot_jacobian_element_grid
)
from .perturbation import (
    plot_perturbation_effect_heatmap,
    plot_perturbation_magnitude,
    plot_gene_response,
    plot_top_affected_genes_bar,
    plot_simulation_comparison
)
from .flow import (
    plot_flow,
    plot_inner_product,
    visualize_flow_comparison,
    visualize_perturbation_flow,
    plot_reference_flow,
    plot_ode_perturbation_flow,
    visualize_ode_perturbation,
)

__all__ = [
    # Energy visualization
    'plot_energy_landscape',
    'plot_energy_components',
    'plot_energy_boxplots',
    'plot_energy_scatters',
    # Gene visualization
    'plot_sigmoid_fit',
    # Network visualization
    'plot_interaction_matrix',
    'plot_network_centrality_rank',
    'plot_centrality_comparison',
    'plot_gene_centrality',
    'plot_centrality_scatter',
    'plot_eigenvalue_spectrum',
    'plot_eigenvector_components',
    'plot_eigenanalysis_grid',
    'plot_grn_network',
    'plot_grn_subset',
    # Dynamics visualization
    'plot_trajectory',
    # Correlation visualization
    'plot_gene_correlation_scatter',
    'plot_correlations_grid',
    # Jacobian visualization
    'plot_jacobian_eigenvalue_spectrum',
    'plot_jacobian_eigenvalue_boxplots',
    'plot_jacobian_stats_boxplots',
    'plot_jacobian_element_grid',
    # Perturbation visualization
    'plot_perturbation_effect_heatmap',
    'plot_perturbation_magnitude',
    'plot_gene_response',
    'plot_top_affected_genes_bar',
    'plot_simulation_comparison',
    # Flow visualization
    'plot_flow',
    'plot_inner_product',
    'visualize_flow_comparison',
    'visualize_perturbation_flow',
    'plot_reference_flow',
    'plot_ode_perturbation_flow',
    'visualize_ode_perturbation',
]
