Tools (sch.tl)
==============

The tools module provides analysis functions for energy, networks, correlations, embeddings, Jacobians, and velocity.

Energy Analysis
---------------

.. currentmodule:: scHopfield.tools

.. autosummary::
   :toctree: generated/

   compute_energies
   decompose_degradation_energy
   decompose_bias_energy
   decompose_interaction_energy

Network Analysis
----------------

.. autosummary::
   :toctree: generated/

   network_correlations
   get_network_links
   compute_network_centrality
   get_top_genes_table
   compute_eigenanalysis
   get_top_eigenvector_genes
   get_eigenanalysis_table
   regulatory_out_strength
   regulatory_coupling

Correlation Analysis
--------------------

.. autosummary::
   :toctree: generated/

   energy_gene_correlation
   celltype_correlation
   future_celltype_correlation
   get_correlation_table

Embedding
---------

.. autosummary::
   :toctree: generated/

   compute_umap
   energy_embedding
   save_embedding
   load_embedding
   project_to_embedding
   build_correlation_projector

Jacobian & Stability Analysis
------------------------------

.. autosummary::
   :toctree: generated/

   compute_jacobians
   save_jacobians
   load_jacobians
   compute_jacobian_stats
   compute_jacobian_elements
   compute_rotational_part

Velocity
--------

.. autosummary::
   :toctree: generated/

   compute_reconstructed_velocity
   validate_velocity
   compute_velocity
   compute_velocity_delta

Flow
----

.. autosummary::
   :toctree: generated/

   calculate_flow
   calculate_grid_flow
   calculate_inner_product
   reference_flow

Perturbation Scoring
--------------------

.. autosummary::
   :toctree: generated/

   score_driver_tfs
   compute_lineage_bias
   compute_perturbation_flow_bias
   lineage_axis_from_embedding
   compute_lineage_commitment
   compute_perturbation_commitment_change
   compute_cluster_effects
   compute_perturbation_score
   compute_perturbation_alignment
   lineage_de
   grn_partner_weights

First-Order (Jacobian) Knockout Response
----------------------------------------

The immediate response to a knockout, before any propagation: which genes it changes, and which
way it pushes a cell along a lineage decision. All three share one finite-difference pass.

.. autosummary::
   :toctree: generated/

   jacobian_knockout_response
   jacobian_response
   jacobian_commitment_push

Fate-Probability Knockout Readouts
-----------------------------------

The projection-free lineage-effect readouts: a knockout is scored by the change it induces in
terminal-state absorption probabilities, so a gene that propagates to nothing scores exactly zero.
Build the wild-type scaffold once with :func:`fate_scaffold` and pass it to the rest.

.. autosummary::
   :toctree: generated/

   fate_scaffold
   lineage_pair_axes
   model_velocity
   fate_transition_matrix
   terminal_states
   fate_probabilities
   split_fraction
   decider_mask
   perturbed_fate
   perturbed_fates
   fate_shift
   pairwise_fate_bias
   per_cell_fate_shift
   dose_fate_bias
   fate_embedding_flow
   terminal_fate_shift
   commitment_time
   permutation_null_floor

Combinatorial Perturbation and Candidate Selection
---------------------------------------------------

.. autosummary::
   :toctree: generated/

   double_knockout_matrix
   fate_bias_candidates
   select_specificity_wings
   rank_by_fate_effect

Dynamical Character
-------------------

Per-cell-type summaries of how the fitted field behaves locally: how fast it moves, whether it is
settling toward an attractor or circulating, and how far along it is.

.. autosummary::
   :toctree: generated/

   velocity_speed
   attractor_index
   settling_score
   oscillation_score
   celltype_character

Fitted-Gene Accessor
--------------------

.. autosummary::
   :toctree: generated/

   get_genes_used

Model I/O
---------

.. autosummary::
   :toctree: generated/

   save_model
   load_model
   save_checkpoint
   load_checkpoint
