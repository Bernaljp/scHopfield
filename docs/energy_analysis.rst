Energy Analysis
===============

Energy landscapes provide a quantitative measure of cellular state stability in the Hopfield network framework.

Energy Function
---------------

The total energy is decomposed into three components:

.. math::

   E_{total} = E_{interaction} + E_{degradation} + E_{bias}

Where:

.. math::

   E_{interaction} &= -\frac{1}{2} s^T W s \\
   E_{degradation} &= \sum_i \gamma_i \int \sigma^{-1}(s_i) ds_i \\
   E_{bias} &= -I^T s

Computing Energies
------------------

.. code-block:: python

   import scHopfield as sch

   # Compute energy for all cells
   sch.tl.compute_energies(adata, cluster_key='cell_type')

This adds to ``adata.obs``:

- ``energy_total``
- ``energy_interaction``
- ``energy_degradation``
- ``energy_bias``

Energy Decomposition
--------------------

Analyze gene-wise energy contributions:

.. code-block:: python

   # Degradation energy per gene
   deg_energy = sch.tl.decompose_degradation_energy(
       adata, cluster='HSC'
   )

   # Bias energy per gene
   bias_energy = sch.tl.decompose_bias_energy(
       adata, cluster='HSC'
   )

   # Interaction energy per gene (incoming or outgoing)
   int_energy = sch.tl.decompose_interaction_energy(
       adata, cluster='HSC', side='in'
   )

Energy-Gene Correlations
------------------------

Identify genes driving energy landscapes:

.. code-block:: python

   # Correlate energy with gene expression
   sch.tl.energy_gene_correlation(adata, cluster_key='cell_type')

This stores correlations in ``adata.var``:

- ``correlation_energy_total_{cluster}``
- ``correlation_energy_interaction_{cluster}``
- ``correlation_energy_degradation_{cluster}``
- ``correlation_energy_bias_{cluster}``

Visualization
-------------

Energy Landscapes
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 2D energy landscape on UMAP
   sch.pl.plot_energy_landscape(adata, cluster='HSC')

   # All energy components
   sch.pl.plot_energy_components(adata, cluster='HSC')

Energy Distributions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Boxplots across cell types
   sch.pl.plot_energy_boxplots(
       adata,
       cluster_key='cell_type',
       order=['HSC', 'MPP', 'CMP', 'GMP', 'MEP']
   )

   # Scatter plots
   sch.pl.plot_energy_scatters(adata, cluster_key='cell_type')

Gene-Energy Correlations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Scatter plot for specific genes
   sch.pl.plot_gene_correlation_scatter(
       adata,
       genes=['GATA1', 'SPI1', 'FLI1'],
       cluster='HSC'
   )

Interpretation
--------------

**Low Energy States**
   Stable cellular states (attractors)

**High Energy States**
   Unstable or transitional states

**Energy Gradients**
   Direction of cell state transitions

**Energy Barriers**
   Resistance to state transitions

Use Cases
---------

1. **Identify Attractors**: Find stable cell states as local energy minima
2. **Transition Analysis**: Map differentiation paths via energy gradients
3. **Driver Genes**: Find genes with high energy-expression correlation
4. **State Stability**: Compare energy distributions across cell types

See Also
--------

- :doc:`api/tools` - API reference
- :doc:`stability_analysis` - Jacobian analysis
- :doc:`visualization` - Plotting guide
