Plotting (sch.pl)
=================

The plotting module provides visualization functions for all analysis results.

Energy Plots
------------

.. currentmodule:: scHopfield.plotting

.. autosummary::
   :toctree: generated/

   plot_energy_landscape
   plot_energy_components
   plot_energy_boxplots
   plot_energy_scatters

Network Plots
-------------

.. autosummary::
   :toctree: generated/

   plot_interaction_matrix
   plot_network_centrality_rank
   plot_centrality_comparison
   plot_gene_centrality
   plot_centrality_scatter
   plot_eigenvalue_spectrum
   plot_eigenvector_components
   plot_eigenanalysis_grid
   plot_grn_network
   plot_grn_subset

Jacobian Plots
--------------

.. autosummary::
   :toctree: generated/

   plot_jacobian_eigenvalue_spectrum
   plot_jacobian_eigenvalue_boxplots
   plot_jacobian_stats_boxplots
   plot_jacobian_element_grid

Correlation Plots
-----------------

.. autosummary::
   :toctree: generated/

   plot_gene_correlation_scatter
   plot_correlations_grid

Perturbation Plots
------------------

.. autosummary::
   :toctree: generated/

   plot_perturbation_effect_heatmap
   plot_perturbation_magnitude
   plot_gene_response
   plot_top_affected_genes_bar
   plot_simulation_comparison

Flow Plots (CellOracle-style)
-----------------------------

.. autosummary::
   :toctree: generated/

   calculate_perturbation_flow
   calculate_grid_flow
   calculate_grid_flow_knn
   calculate_inner_product
   plot_reference_flow
   plot_perturbation_flow
   plot_flow_on_grid
   plot_simulation_flow_on_grid
   plot_inner_product_on_embedding
   plot_inner_product_by_cluster
   visualize_perturbation_flow

Other Plots
-----------

.. autosummary::
   :toctree: generated/

   plot_sigmoid_fit
   plot_trajectory

Detailed API
------------

Energy Plots
~~~~~~~~~~~~

plot_energy_landscape
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_energy_landscape

plot_energy_components
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_energy_components

plot_energy_boxplots
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_energy_boxplots

plot_energy_scatters
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_energy_scatters

Network Plots
~~~~~~~~~~~~~

plot_interaction_matrix
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_interaction_matrix

plot_network_centrality_rank
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_network_centrality_rank

plot_centrality_comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_centrality_comparison

plot_gene_centrality
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_gene_centrality

plot_centrality_scatter
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_centrality_scatter

plot_eigenvalue_spectrum
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_eigenvalue_spectrum

plot_eigenvector_components
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_eigenvector_components

plot_eigenanalysis_grid
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_eigenanalysis_grid

plot_grn_network
^^^^^^^^^^^^^^^^

.. autofunction:: plot_grn_network

plot_grn_subset
^^^^^^^^^^^^^^^

.. autofunction:: plot_grn_subset

Jacobian Plots
~~~~~~~~~~~~~~

plot_jacobian_eigenvalue_spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_jacobian_eigenvalue_spectrum

plot_jacobian_eigenvalue_boxplots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_jacobian_eigenvalue_boxplots

plot_jacobian_stats_boxplots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_jacobian_stats_boxplots

plot_jacobian_element_grid
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_jacobian_element_grid

Correlation Plots
~~~~~~~~~~~~~~~~~

plot_gene_correlation_scatter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_gene_correlation_scatter

plot_correlations_grid
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_correlations_grid

Other Plots
~~~~~~~~~~~

plot_sigmoid_fit
^^^^^^^^^^^^^^^^

.. autofunction:: plot_sigmoid_fit

plot_trajectory
^^^^^^^^^^^^^^^

.. autofunction:: plot_trajectory

Perturbation Plots
~~~~~~~~~~~~~~~~~~

plot_perturbation_effect_heatmap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_perturbation_effect_heatmap

plot_perturbation_magnitude
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_perturbation_magnitude

plot_gene_response
^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_gene_response

plot_top_affected_genes_bar
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_top_affected_genes_bar

plot_simulation_comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_simulation_comparison

Flow Plots (CellOracle-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

calculate_perturbation_flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: calculate_perturbation_flow

calculate_grid_flow
^^^^^^^^^^^^^^^^^^^

.. autofunction:: calculate_grid_flow

calculate_grid_flow_knn
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: calculate_grid_flow_knn

calculate_inner_product
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: calculate_inner_product

plot_reference_flow
^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_reference_flow

plot_perturbation_flow
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_perturbation_flow

plot_flow_on_grid
^^^^^^^^^^^^^^^^^

.. autofunction:: plot_flow_on_grid

plot_simulation_flow_on_grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_simulation_flow_on_grid

plot_inner_product_on_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_inner_product_on_embedding

plot_inner_product_by_cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: plot_inner_product_by_cluster

visualize_perturbation_flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: visualize_perturbation_flow
