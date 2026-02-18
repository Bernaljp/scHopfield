"""Tools module for analysis functions."""

from .energy import (
    compute_energies,
    decompose_degradation_energy,
    decompose_bias_energy,
    decompose_interaction_energy
)
from .correlation import (
    energy_gene_correlation,
    celltype_correlation,
    future_celltype_correlation,
    get_correlation_table
)
from .embedding import (
    compute_umap,
    energy_embedding,
    save_embedding,
    load_embedding,
    project_to_embedding
)
from .jacobian import (
    compute_jacobians,
    save_jacobians,
    load_jacobians,
    compute_jacobian_stats,
    compute_jacobian_elements,
    compute_rotational_part
)
from .networks import (
    network_correlations,
    get_network_links,
    compute_network_centrality,
    get_top_genes_table,
    compute_eigenanalysis,
    get_top_eigenvector_genes,
    get_eigenanalysis_table
)
from .velocity import (
    compute_reconstructed_velocity,
    validate_velocity,
    compute_velocity,
    compute_velocity_delta
)
from .flow import (
    calculate_flow,
    calculate_grid_flow,
    calculate_inner_product
)

__all__ = [
    # Energy analysis
    'compute_energies',
    'decompose_degradation_energy',
    'decompose_bias_energy',
    'decompose_interaction_energy',
    # Correlation analysis
    'energy_gene_correlation',
    'celltype_correlation',
    'future_celltype_correlation',
    'get_correlation_table',
    # Embedding
    'compute_umap',
    'energy_embedding',
    'save_embedding',
    'load_embedding',
    'project_to_embedding',
    # Jacobian analysis
    'compute_jacobians',
    'save_jacobians',
    'load_jacobians',
    'compute_jacobian_stats',
    'compute_jacobian_elements',
    'compute_rotational_part',
    # Network analysis
    'network_correlations',
    'get_network_links',
    'compute_network_centrality',
    'get_top_genes_table',
    'compute_eigenanalysis',
    'get_top_eigenvector_genes',
    'get_eigenanalysis_table',
    # Velocity computation
    'compute_reconstructed_velocity',
    'validate_velocity',
    'compute_velocity',
    'compute_velocity_delta',
    # Flow computation
    'calculate_flow',
    'calculate_grid_flow',
    'calculate_inner_product',
]
