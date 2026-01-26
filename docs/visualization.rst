Visualization
=============

scHopfield provides comprehensive plotting functions for all analysis results.

Energy Plots
------------

See :doc:`energy_analysis` for details.

.. code-block:: python

   import scHopfield as sch

   # 2D energy landscape
   sch.pl.plot_energy_landscape(adata, cluster='HSC')

   # All components
   sch.pl.plot_energy_components(adata, cluster='HSC')

   # Distributions
   sch.pl.plot_energy_boxplots(adata, cluster_key='cell_type')
   sch.pl.plot_energy_scatters(adata, cluster_key='cell_type')

Network Plots
-------------

See :doc:`network_analysis` for details.

.. code-block:: python

   # Interaction matrix heatmap
   sch.pl.plot_interaction_matrix(adata, cluster='HSC', top_n=30)

   # Centrality rankings
   sch.pl.plot_network_centrality_rank(
       adata, cluster_key='cell_type', metric='degree_centrality_all'
   )

   # GRN network graph
   sch.pl.plot_grn_network(adata, cluster='HSC', topn=50)

   # Eigenvalue spectrum
   sch.pl.plot_eigenvalue_spectrum(adata, cluster='HSC')

Jacobian Plots
--------------

See :doc:`stability_analysis` for details.

.. code-block:: python

   # Eigenvalue spectra
   sch.pl.plot_jacobian_eigenvalue_spectrum(
       adata, cluster_key='cell_type'
   )

   # Stability metrics
   sch.pl.plot_jacobian_stats_boxplots(
       adata, cluster_key='cell_type'
   )

   # Partial derivatives on UMAP
   sch.pl.plot_jacobian_element_grid(
       adata,
       gene_pairs=[('GATA1', 'GATA2'), ('SPI1', 'GATA1')]
   )

Dynamics Plots
--------------

See :doc:`dynamics` for details.

.. code-block:: python

   # Trajectory visualization
   trajectory = sch.dyn.simulate_trajectory(
       adata, cluster='HSC', cell_idx=0,
       t_span=np.linspace(0, 10, 100)
   )
   sch.pl.plot_trajectory(trajectory, np.linspace(0, 10, 100))

Customization
-------------

All plotting functions return matplotlib figure/axes objects for customization:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Get figure object
   fig = sch.pl.plot_energy_landscape(adata, cluster='HSC')

   # Customize
   fig.suptitle('HSC Energy Landscape', fontsize=16)
   plt.tight_layout()
   plt.savefig('hsc_energy.pdf', dpi=300)

Common Parameters
-----------------

Most plotting functions support:

**figsize** : tuple
   Figure size in inches

**cmap** : str or Colormap
   Colormap for heatmaps/continuous values

**colors** : dict
   Custom colors for categorical variables

**order** : list
   Order of categories to display

**top_n** : int
   Number of top items to show

See Also
--------

- :doc:`api/plotting` - Complete plotting API
- :doc:`quickstart` - Quick start with examples
- :doc:`tutorial` - Detailed tutorial
