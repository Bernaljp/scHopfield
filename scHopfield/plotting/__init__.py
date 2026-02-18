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
    # Main plotting functions (new unified API)
    plot_flow,
    plot_inner_product,
    visualize_flow_comparison,
    visualize_perturbation_flow,
    plot_reference_flow,
    plot_ode_perturbation_flow,
    visualize_ode_perturbation,
    # Deprecated aliases (for backward compatibility)
    compute_hopfield_velocity,
    compute_hopfield_velocity_delta,
    compute_hopfield_velocity_at_state,
    project_velocity_to_embedding,
    calculate_perturbation_flow_hopfield,
    calculate_perturbed_velocity_flow,
    calculate_original_velocity_flow,
    calculate_perturbation_flow,
    calculate_grid_flow_knn,
    calculate_ode_trajectory_flow,
    calculate_ode_trajectory_inner_product,
    # Plotting aliases (thin wrappers around plot_flow)
    plot_perturbation_flow,
    plot_perturbed_velocity_flow,
    plot_flow_on_grid,
    plot_simulation_flow_on_grid,
    plot_inner_product_on_embedding,
    plot_inner_product_by_cluster,
    visualize_velocity_comparison,
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
    # Flow visualization (new unified API)
    'plot_flow',
    'plot_inner_product',
    'visualize_flow_comparison',
    'visualize_perturbation_flow',
    'plot_reference_flow',
    'plot_ode_perturbation_flow',
    'visualize_ode_perturbation',
    # Deprecated - kept for backward compatibility
    'compute_hopfield_velocity',
    'compute_hopfield_velocity_delta',
    'compute_hopfield_velocity_at_state',
    'project_velocity_to_embedding',
    'calculate_perturbation_flow_hopfield',
    'calculate_perturbed_velocity_flow',
    'calculate_original_velocity_flow',
    'calculate_perturbation_flow',
    'calculate_grid_flow_knn',
    'calculate_ode_trajectory_flow',
    'calculate_ode_trajectory_inner_product',
    'plot_perturbation_flow',
    'plot_perturbed_velocity_flow',
    'plot_flow_on_grid',
    'plot_simulation_flow_on_grid',
    'plot_inner_product_on_embedding',
    'plot_inner_product_by_cluster',
    'visualize_velocity_comparison',
]
