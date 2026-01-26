Quick Start
===========

This guide will walk you through a basic scHopfield analysis workflow.

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

   # Select highly variable genes (50-200 recommended)
   highly_variable_genes = adata.var['highly_variable'].values

   # Fit sigmoid functions
   sch.pp.fit_all_sigmoids(adata, genes=highly_variable_genes)

   # Compute sigmoid-transformed expression
   sch.pp.compute_sigmoid(adata)

2. Network Inference
~~~~~~~~~~~~~~~~~~~~

Learn interaction matrices from RNA velocity:

.. code-block:: python

   sch.inf.fit_interactions(
       adata,
       cluster_key='cell_type',
       n_epochs=1000,
       device='cuda'  # or 'cpu'
   )

This infers cluster-specific gene regulatory networks stored in ``adata.varp['W_{cluster}']``.

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

Simulate gene knockouts or overexpression:

.. code-block:: python

   # Simulate GATA1 knockout
   perturbed = sch.dyn.simulate_perturbation(
       adata,
       cluster='HSC',
       cell_idx=0,
       gene_perturbations={'GATA1': 0.0},  # knockout
       t_span=np.linspace(0, 20, 200)
   )

   # Simulate GATA1 overexpression
   overexpressed = sch.dyn.simulate_perturbation(
       adata,
       cluster='HSC',
       cell_idx=0,
       gene_perturbations={'GATA1': 10.0},  # overexpression
       t_span=np.linspace(0, 20, 200)
   )

   # Compare with wild-type
   wt = sch.dyn.simulate_trajectory(
       adata,
       cluster='HSC',
       cell_idx=0,
       t_span=np.linspace(0, 20, 200)
   )

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

- Read the :doc:`tutorial` for a detailed walkthrough with real data
- Explore specific analyses in the User Guide
- Check the :doc:`api/tools` reference for all available functions
- See :doc:`examples` for Jupyter notebooks with complete analyses
