Plotting (sch.pl)
=================

The plotting module provides visualization functions for all analysis results.

Energy Plots
------------

.. currentmodule:: scHopfield.plotting

.. autosummary::
   :toctree: generated/

   plot_energy_landscape
   plot_energy_boxplots
   plot_energy_scatters

Network Plots
-------------

.. autosummary::
   :toctree: generated/

   plot_interaction_matrix
   plot_centrality_scatter
   plot_eigenvalue_spectrum
   plot_eigenvector_components
   plot_eigenanalysis_grid
   plot_grn_network

Jacobian Plots
--------------

.. autosummary::
   :toctree: generated/

   plot_jacobian_eigenvalue_spectrum
   plot_jacobian_stats_boxplots
   plot_jacobian_element_grid

Correlation Plots
-----------------

.. autosummary::
   :toctree: generated/

   plot_gene_correlation_scatter
   plot_correlations_grid

Flow Plots
----------

.. autosummary::
   :toctree: generated/

   plot_flow
   plot_inner_product

Other Plots
-----------

.. autosummary::
   :toctree: generated/

   plot_sigmoid_fit
   plot_trajectory

Circuit diagrams
----------------

TikZ rendering of a gene regulatory network, with a matplotlib fallback that draws
when no LaTeX toolchain is available.

.. currentmodule:: scHopfield.plotting.tikz

.. autosummary::
   :toctree: generated/

   draw_grn
   draw_grn_mpl
   render_tikz
   tikz_available
   grn_tikz_body
   grn_preamble
   GRN_PREAMBLE
