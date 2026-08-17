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

Perturbation Scoring
--------------------

.. autosummary::
   :toctree: generated/

   score_driver_tfs
   compute_lineage_bias
   compute_cluster_effects
   compute_perturbation_score
   compute_perturbation_alignment
   lineage_de
   grn_partner_weights

Model I/O
---------

.. autosummary::
   :toctree: generated/

   save_model
   load_model
   save_checkpoint
   load_checkpoint
