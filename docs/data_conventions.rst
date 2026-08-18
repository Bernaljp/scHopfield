Data Storage Conventions
========================

scHopfield follows standard AnnData conventions and stores results systematically throughout the analysis workflow.

Overview
--------

Results are stored in different AnnData slots depending on their dimensionality:

- ``adata.var`` - Gene-level data (per gene)
- ``adata.obs`` - Cell-level data (per cell)
- ``adata.varp`` - Gene-gene matrices
- ``adata.obsm`` - Cell embeddings and high-dimensional cell data
- ``adata.varm`` - Gene-dimensional arrays
- ``adata.layers`` - Gene expression matrices
- ``adata.uns`` - Unstructured metadata and results

adata.var (Gene-level Data)
----------------------------

Sigmoid Parameters
~~~~~~~~~~~~~~~~~~

Added by ``sch.pp.fit_all_sigmoids()`` and ``sch.pp.compute_sigmoid()``:

- ``sigmoid_threshold`` - Fitted threshold parameter
- ``sigmoid_exponent`` - Fitted exponent parameter
- ``sigmoid_offset`` - Fitted offset parameter
- ``sigmoid_mse`` - Mean squared error of fit
- ``scHopfield_used`` - Boolean mask of genes used in analysis

``fit_all_sigmoids`` fits a two-component activation by default (``bimodal=True``),
so it writes three further columns describing the second component:

- ``sigmoid_threshold2`` - Threshold of the second Hill component
- ``sigmoid_exponent2`` - Exponent of the second Hill component
- ``sigmoid_mix`` - Weight on the first component, in [0, 1]

All three are present whichever way the fit ran. A single-component fit sets the
second component equal to the first and ``sigmoid_mix`` to 1, which is exactly a
single Hill curve, so code that ignores these columns is unaffected. Reading them
is how you tell a genuine single-component fit from an activation that was never
fitted: ``load_model`` restores the network but not the activation, and
``compute_sigmoid`` on the defaults left behind by a load will quietly evaluate a
single Hill curve.

Network Parameters (per cluster)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added by ``sch.inf.fit_interactions()``:

- ``I_{cluster}`` - Bias vector for each cluster
- ``gamma_{cluster}`` - Cluster-specific degradation rates (if ``refit_gamma=True``)

Network Centrality (per cluster)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added by ``sch.tl.compute_network_centrality()``:

- ``degree_all_{cluster}`` - Total degree (in + out)
- ``degree_centrality_all_{cluster}`` - Normalized total degree
- ``degree_in_{cluster}`` - In-degree
- ``degree_out_{cluster}`` - Out-degree
- ``degree_centrality_in_{cluster}`` - Normalized in-degree
- ``degree_centrality_out_{cluster}`` - Normalized out-degree
- ``betweenness_centrality_{cluster}`` - Betweenness centrality
- ``eigenvector_centrality_{cluster}`` - Eigenvector centrality

Correlations (per cluster)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added by ``sch.tl.energy_gene_correlation()``:

- ``correlation_total_{cluster}`` - Total energy vs gene expression
- ``correlation_interaction_{cluster}`` - Interaction energy correlation
- ``correlation_degradation_{cluster}`` - Degradation energy correlation
- ``correlation_bias_{cluster}`` - Bias energy correlation

The same four columns are also written with ``all`` in place of the cluster name,
over every cell at once.

adata.obs (Cell-level Data)
----------------------------

Energy Values
~~~~~~~~~~~~~

Added by ``sch.tl.compute_energies()`` (shared across all cells):

- ``energy_total`` - Total Hopfield energy
- ``energy_interaction`` - Interaction component
- ``energy_degradation`` - Degradation component
- ``energy_bias`` - Bias component

Jacobian Statistics
~~~~~~~~~~~~~~~~~~~~

Added by ``sch.tl.compute_jacobian_stats()``:

- ``jacobian_eig1_real`` - Real part of leading eigenvalue
- ``jacobian_eig1_imag`` - Imaginary part of leading eigenvalue
- ``jacobian_positive_evals`` - Count of positive real eigenvalues
- ``jacobian_negative_evals`` - Count of negative real eigenvalues
- ``jacobian_trace`` - Trace of Jacobian matrix

Added by ``sch.tl.compute_rotational_part()``:

- ``jacobian_rotational`` - Frobenius norm of antisymmetric part

Added by ``sch.tl.compute_jacobian_elements()``:

- ``jacobian_df_{gene_i}_dx_{gene_j}`` - Specific partial derivatives

adata.varp (Gene-Gene Matrices)
--------------------------------

Interaction Matrices (per cluster)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added by ``sch.inf.fit_interactions()``:

- ``W_{cluster}`` - Interaction matrix for each cluster (n_genes × n_genes, sparse)

.. code-block:: python

   # Access interaction matrix
   W_hsc = adata.varp['W_HSC']
   print(W_hsc.shape)  # (n_genes, n_genes)

adata.obsm (Cell Embeddings)
-----------------------------

Dimensionality Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~

Added by ``sch.tl.compute_umap()``:

- ``X_umap`` - UMAP coordinates (n_obs × 2)

Jacobian Eigenvalues
~~~~~~~~~~~~~~~~~~~~~

Added by ``sch.tl.compute_jacobians()``:

- ``jacobian_eigenvalues`` - Jacobian eigenvalues for all cells (n_obs × n_genes, complex)

.. code-block:: python

   # Access Jacobian eigenvalues
   evals = adata.obsm['jacobian_eigenvalues']
   print(evals.shape)  # (n_cells, n_genes)
   print(evals.dtype)  # complex128

adata.varm (Gene-Dimensional Arrays)
-------------------------------------

Energy Embedding
~~~~~~~~~~~~~~~~

Added by ``sch.tl.energy_embedding()``:

- ``highD_grid`` - High-dimensional grid points for energy landscape projection

adata.layers (Expression Matrices)
-----------------------------------

Transformed Expression
~~~~~~~~~~~~~~~~~~~~~~~

Added by ``sch.pp.compute_sigmoid()``:

- ``sigmoid`` - Sigmoid-transformed expression (n_obs × n_genes)

Velocity
~~~~~~~~

Added by ``sch.tl.compute_reconstructed_velocity()``, but only when it is given a
layer to write to. Its ``layer_key`` argument defaults to ``None``, in which case
the model-predicted velocity is returned as an array and nothing is stored:

.. code-block:: python

   # returned, not stored
   v = sch.tl.compute_reconstructed_velocity(adata, cluster_key='cell_type')

   # stored in adata.layers['velocity_reconstructed']
   sch.tl.compute_reconstructed_velocity(
       adata, cluster_key='cell_type', layer_key='velocity_reconstructed'
   )

Passing a single ``cluster`` writes to ``f'{layer_key}_{cluster}'`` rather than to
``layer_key`` itself.

adata.uns['scHopfield'] (Metadata)
----------------------------------

Configuration
~~~~~~~~~~~~~

- ``cluster_key`` - Name of cluster key used
- ``genes_used`` - Indices of genes used in analysis

Models & Embeddings
~~~~~~~~~~~~~~~~~~~

Added by ``sch.inf.fit_interactions()``:

- ``models`` - Trained optimizer models (dictionary by cluster)

Added by ``sch.tl.compute_umap()``:

- ``embedding`` - UMAP model object

Energy Grids (per cluster)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added by ``sch.tl.energy_embedding()``:

- ``grid_X_{cluster}`` - X coordinates of 2D grid
- ``grid_Y_{cluster}`` - Y coordinates of 2D grid
- ``grid_energy_{cluster}`` - Total energy on grid
- ``grid_energy_interaction_{cluster}`` - Interaction energy on grid
- ``grid_energy_degradation_{cluster}`` - Degradation energy on grid
- ``grid_energy_bias_{cluster}`` - Bias energy on grid

Network Analysis
~~~~~~~~~~~~~~~~

Added by ``sch.tl.network_correlations()``:

- ``network_correlations`` - Dictionary of similarity metrics between clusters

  - ``'jaccard'`` - Jaccard index
  - ``'hamming'`` - Hamming distance
  - ``'euclidean'`` - Euclidean distance
  - ``'pearson'`` - Pearson correlation
  - ``'pearson_bin'`` - Pearson correlation (binary)
  - ``'mean_col_corr'`` - Mean column-wise correlation
  - ``'singular'`` - Singular value distance

Added by ``sch.tl.celltype_correlation()``:

- ``celltype_correlation`` - RV coefficient matrix

Added by ``sch.tl.future_celltype_correlation()``:

- ``future_celltype_correlation`` - Future state correlation matrix

Eigenanalysis (per cluster)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added by ``sch.tl.compute_eigenanalysis()``:

- ``eigenanalysis[f'eigenvalues_{cluster}']`` - Eigenvalues of W
- ``eigenanalysis[f'eigenvectors_{cluster}']`` - Eigenvectors of W

Example: Accessing Results
---------------------------

.. code-block:: python

   import scHopfield as sch

   # After running full analysis...

   # Access energy values
   total_energy = adata.obs['energy_total']

   # Access interaction matrix for HSC
   W_hsc = adata.varp['W_HSC']

   # Access centrality for a specific cluster
   degree_hsc = adata.var['degree_centrality_all_HSC']

   # Access Jacobian eigenvalues
   jacobian_evals = adata.obsm['jacobian_eigenvalues']

   # Access network correlations
   correlations = adata.uns['scHopfield']['network_correlations']
   jaccard = correlations['jaccard']

   # Get top genes by centrality
   top_genes = sch.tl.get_top_genes_table(
       adata,
       cluster='HSC',
       metric='degree_centrality_all',
       n=20
   )

See Also
--------

- :doc:`api/tools` - All functions that modify adata
- :doc:`quickstart` - Basic workflow
- :doc:`tutorial` - Detailed tutorial
