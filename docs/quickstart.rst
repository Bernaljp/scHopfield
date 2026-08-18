Quick Start
===========

This guide will walk you through a basic scHopfield analysis workflow.

.. tip::

   In a hurry? :func:`scHopfield.run_pipeline` runs the whole sequence below in
   one call. See :doc:`pipeline`. This page shows the individual steps for when
   you want finer control.

Basic Workflow
--------------

Import Libraries
~~~~~~~~~~~~~~~~

.. code-block:: python

   import scHopfield as sch
   import scanpy as sc
   import numpy as np

Load Data
~~~~~~~~~

Load your single-cell data with RNA velocity computed:

.. code-block:: python

   # Load your data
   adata = sc.read_h5ad('data.h5ad')

   # Required layers and annotations:
   # - adata.layers['Ms'] - spliced counts
   # - adata.layers['velocity_S'] - RNA velocity
   # - adata.var['gamma'] - degradation rates
   # - adata.obs['cell_type'] - cell type annotations

1. Preprocessing
~~~~~~~~~~~~~~~~

Fit sigmoid activation functions to gene expression:

.. code-block:: python

   # Select the genes to model. prepare_dataset defaults to the top 2,000, which is
   # what the published fits use; a few hundred is enough to explore on a laptop.
   highly_variable_genes = adata.var['highly_variable'].values

   # Fit sigmoid functions
   sch.pp.fit_all_sigmoids(adata, genes=highly_variable_genes)

   # Compute sigmoid-transformed expression
   sch.pp.compute_sigmoid(adata)

2. Network Inference
~~~~~~~~~~~~~~~~~~~~

Build the prior-knowledge scaffold, then learn interaction matrices from RNA velocity:

.. code-block:: python

   # A base GRN is a wide table: one column of target gene names, one binary column
   # per transcription factor. build_scaffold restricts it to the genes being modeled.
   # scHopfield does not distribute one; fetch_base_grn downloads a CellOracle table
   # from a pinned commit, verifies its checksum and caches it locally. That table
   # carries CellOracle's license and not scHopfield's: see DATA_SOURCES.md.
   base_grn = sch.fetch_base_grn('mouse')     # or 'human'
   scaffold = sch.inf.build_scaffold(adata, base_grn)

   sch.inf.fit_interactions(
       adata,
       cluster_key='cell_type',
       w_scaffold=scaffold.values.T,   # fit_interactions indexes W as [target, regulator]
       device='cuda'                   # or 'cpu'
   )

This infers cluster-specific gene regulatory networks stored in ``adata.varp['W_{cluster}']``.

.. important::

   Pass the scaffold. The remaining defaults are already the configuration used
   throughout the paper, so no other argument needs tuning to reproduce the published
   method, but the scaffold is the one input the package cannot supply for you.
   ``fit_interactions`` without ``w_scaffold`` fits an unconstrained network, which the
   ablations in the paper find measurably worse. Treat it as a baseline, not a result.

3. Energy Analysis
~~~~~~~~~~~~~~~~~~

Compute energy landscapes:

.. code-block:: python

   # Compute total energy and components
   sch.tl.compute_energies(adata, cluster_key='cell_type')

   # Correlate energies with gene expression
   sch.tl.energy_gene_correlation(adata, cluster_key='cell_type')

4. Network Analysis
~~~~~~~~~~~~~~~~~~~

Analyze network topology:

.. code-block:: python

   # Compute centrality metrics
   sch.tl.compute_network_centrality(adata, cluster_key='cell_type')

   # Eigenvalue decomposition
   sch.tl.compute_eigenanalysis(adata, cluster_key='cell_type')

   # Compare networks across cell types
   sch.tl.network_correlations(adata, cluster_key='cell_type')

5. Stability Analysis
~~~~~~~~~~~~~~~~~~~~~

Compute Jacobian matrices and stability metrics:

.. code-block:: python

   # Compute Jacobian eigenvalues
   sch.tl.compute_jacobians(
       adata,
       cluster_key='cell_type',
       device='cuda'
   )

   # Compute summary statistics
   sch.tl.compute_jacobian_stats(adata)

   # Save to disk (optional, to save memory)
   sch.tl.save_jacobians(adata, 'jacobians.h5')

6. Visualization
~~~~~~~~~~~~~~~~

Generate plots:

.. code-block:: python

   # Energy landscape
   sch.pl.plot_energy_landscape(adata, cluster='HSC')

   # Interaction matrix
   sch.pl.plot_interaction_matrix(adata, cluster='HSC', top_n=30)

   # GRN network graph
   sch.pl.plot_grn_network(adata, cluster='HSC', topn=50)

   # Jacobian eigenvalue spectra
   sch.pl.plot_jacobian_eigenvalue_spectrum(
       adata,
       cluster_key='cell_type'
   )

   # Energy distributions
   sch.pl.plot_energy_boxplots(
       adata,
       cluster_key='cell_type'
   )

7. Dynamics Simulation
~~~~~~~~~~~~~~~~~~~~~~

Simulate gene expression trajectories:

.. code-block:: python

   # Simulate from a cell's initial state
   trajectory = sch.dyn.simulate_trajectory(
       adata,
       cluster='HSC',
       cell_idx=0,
       t_span=np.linspace(0, 10, 100)
   )

   # Plot trajectory
   sch.pl.plot_trajectory(trajectory, np.linspace(0, 10, 100))

Advanced: Perturbation Experiments
-----------------------------------

Simulate a gene knockout across all cells with CellOracle-style GRN propagation
(``perturb_condition`` maps a gene to its clamped value; ``0`` is a knockout):

.. code-block:: python

   # Simulate a Gata1 knockout (propagation-based, all cells)
   ko = sch.dyn.simulate_perturbation(
       adata,
       perturb_condition={'Gata1': 0.0},
       cluster_key='cell_type',
       n_propagation=3,
   )
   # ko.layers['delta_X'] holds the predicted expression shift per cell

Score a panel of known regulators by the *direction* of the predicted lineage
shift (a ground-truth-anchored validation):

.. code-block:: python

   # First compute the wild-type Hopfield velocity flow in an embedding
   sch.tl.calculate_flow(adata, source='original', basis='umap',
                         method='hopfield', cluster_key='cell_type',
                         store_key='wt_flow_umap')

   # +1 = KO should bias toward lineage A; -1 = toward lineage B
   panel = {'Gata1': -1, 'Klf1': -1, 'Spi1': +1, 'Cebpa': +1}
   table, accuracy = sch.dyn.score_ko_panel(
       adata, panel=panel,
       lineage_A_clusters=['Ery'], lineage_B_clusters=['Mono', 'Neu'],
       basis='umap', wt_flow_key='wt_flow_umap', cluster_key='cell_type',
   )

For continuous ODE-based trajectories under a perturbation, use
:func:`~scHopfield.dynamics.simulate_perturbation_ode` (single cluster/cells) or
:func:`~scHopfield.dynamics.simulate_shift_ode` (dataset-wide).

Typical Workflow Summary
------------------------

A complete analysis typically follows this sequence:

1. **Preprocessing** → Fit sigmoid activation functions
2. **Network Inference** → Learn cluster-specific interaction matrices
3. **Energy Analysis** → Compute landscapes and identify driver genes
4. **Network Analysis** → Analyze topology via centrality and eigenanalysis
5. **Stability Analysis** → Compute Jacobians for local stability
6. **Visualization** → Generate publication-ready plots
7. **Dynamics** → Simulate trajectories and test perturbations

Each step builds on the previous, with all results stored in the AnnData object for seamless integration.

Next Steps
----------

- Work through the :doc:`tutorials <tutorial>`, six executed notebooks that run
  this pipeline on real data and then read the fit
- Run the whole sequence in one call with :doc:`pipeline`
- Check the :doc:`api/index` reference for every public function
