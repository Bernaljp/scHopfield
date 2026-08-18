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
    project_to_embedding,
    build_correlation_projector,
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
    get_eigenanalysis_table,
    regulatory_out_strength,
    regulatory_coupling,
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
    calculate_inner_product,
    reference_flow,
)
from .io import (
    save_model,
    load_model,
    save_checkpoint,
    load_checkpoint,
)
from .perturbation_analysis import (
    score_driver_tfs,
    compute_perturbation_flow_bias,
    compute_lineage_bias,
    lineage_axis_from_embedding,
    compute_lineage_commitment,
    compute_perturbation_commitment_change,
    compute_cluster_effects,
    compute_perturbation_score,
    lineage_de,
    grn_partner_weights,
    compute_perturbation_alignment,
    jacobian_knockout_response,
    jacobian_response,
    jacobian_commitment_push,
    double_knockout_matrix,
    fate_bias_candidates,
    select_specificity_wings,
    rank_by_fate_effect,
)
from .fate import (
    model_velocity,
    fate_transition_matrix,
    terminal_states,
    fate_probabilities,
    split_fraction,
    decider_mask,
    fate_shift,
    permutation_null_floor,
    fate_scaffold,
    lineage_pair_axes,
    perturbed_fate,
    perturbed_fates,
    pairwise_fate_bias,
    per_cell_fate_shift,
    dose_fate_bias,
    fate_embedding_flow,
    terminal_fate_shift,
    commitment_time,
)
from .._utils.io import get_genes_used
from .character import (
    velocity_speed,
    attractor_index,
    settling_score,
    oscillation_score,
    celltype_character,
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
    'build_correlation_projector',
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
    'regulatory_out_strength',
    'regulatory_coupling',
    # Velocity computation
    'compute_reconstructed_velocity',
    'validate_velocity',
    'compute_velocity',
    'compute_velocity_delta',
    # Flow computation
    'calculate_flow',
    'calculate_grid_flow',
    'calculate_inner_product',
    'reference_flow',
    # Fitted-gene accessor
    'get_genes_used',
    # Model I/O
    'save_model',
    'load_model',
    'save_checkpoint',
    'load_checkpoint',
    # Perturbation analysis
    'score_driver_tfs',
    'compute_perturbation_flow_bias',
    'compute_lineage_bias',
    'lineage_axis_from_embedding',
    'compute_lineage_commitment',
    'compute_perturbation_commitment_change',
    'compute_cluster_effects',
    'compute_perturbation_score',
    'lineage_de',
    'grn_partner_weights',
    'compute_perturbation_alignment',
    # First-order (Jacobian) knockout response
    'jacobian_knockout_response',
    'jacobian_response',
    'jacobian_commitment_push',
    # Combinatorial perturbation and candidate selection
    'double_knockout_matrix',
    'fate_bias_candidates',
    'select_specificity_wings',
    'rank_by_fate_effect',
    # Fate probability (projection-free knockout readout) and its permutation null
    'model_velocity',
    'fate_transition_matrix',
    'terminal_states',
    'fate_probabilities',
    'split_fraction',
    'decider_mask',
    'fate_shift',
    'permutation_null_floor',
    'fate_scaffold',
    'lineage_pair_axes',
    'perturbed_fate',
    'perturbed_fates',
    'pairwise_fate_bias',
    'per_cell_fate_shift',
    'dose_fate_bias',
    'fate_embedding_flow',
    'terminal_fate_shift',
    'commitment_time',
    # Dynamical character
    'velocity_speed',
    'attractor_index',
    'settling_score',
    'oscillation_score',
    'celltype_character',
]
