Network Analysis
================

Analyze gene regulatory network topology using centrality metrics, eigenanalysis, and network comparison.

Network Centrality
------------------

Compute centrality metrics for all genes:

.. code-block:: python

   import scHopfield as sch

   sch.tl.compute_network_centrality(
       adata,
       cluster_key='cell_type',
       threshold_number=2000,  # Edge filtering for speed
       use_igraph=True  # Use igraph for 10-100× speedup
   )

Available Metrics
~~~~~~~~~~~~~~~~~

Stored in ``adata.var`` with ``_{cluster}`` suffix:

- ``degree_all`` - Total degree (in + out)
- ``degree_in`` - In-degree (regulated by)
- ``degree_out`` - Out-degree (regulates)
- ``degree_centrality_all`` - Normalized total degree
- ``betweenness_centrality`` - Betweenness
- ``eigenvector_centrality`` - Eigenvector centrality

Top Genes
~~~~~~~~~

Extract top hub genes:

.. code-block:: python

   # Get top 20 genes by degree centrality
   top_genes = sch.tl.get_top_genes_table(
       adata,
       cluster='HSC',
       metric='degree_centrality_all',
       n=20
   )
   print(top_genes)

Eigenanalysis
-------------

Decompose interaction matrices:

.. code-block:: python

   # Compute eigenvalues and eigenvectors
   sch.tl.compute_eigenanalysis(adata, cluster_key='cell_type')

Access Results
~~~~~~~~~~~~~~

.. code-block:: python

   # Eigenvalues for HSC
   evals = adata.uns['scHopfield']['eigenanalysis']['eigenvalues_HSC']
   evecs = adata.uns['scHopfield']['eigenanalysis']['eigenvectors_HSC']

Top Eigenvector Genes
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get genes with highest loadings in 1st eigenvector
   top_genes = sch.tl.get_top_eigenvector_genes(
       adata,
       cluster='HSC',
       k=0,  # First eigenvector
       n=10  # Top 10 genes
   )

   # Get formatted table
   table = sch.tl.get_eigenanalysis_table(
       adata,
       cluster='HSC',
       k=0,
       n=10
   )
   print(table)

Network Comparison
------------------

Compare GRN structure across cell types:

.. code-block:: python

   # Compute similarity metrics
   sch.tl.network_correlations(adata, cluster_key='cell_type')

   # Access results
   correlations = adata.uns['scHopfield']['network_correlations']

   # Jaccard index
   jaccard = correlations['jaccard']
   print(jaccard)

Available Metrics
~~~~~~~~~~~~~~~~~

- ``jaccard`` - Binary overlap
- ``hamming`` - Binary difference
- ``euclidean`` - Continuous distance
- ``pearson`` - Continuous correlation
- ``pearson_bin`` - Binary correlation
- ``mean_col_corr`` - Column-wise correlation
- ``singular`` - Singular value distance

Visualization
-------------

Centrality Rankings
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Ranked centrality plot
   sch.pl.plot_network_centrality_rank(
       adata,
       cluster_key='cell_type',
       metric='degree_centrality_all',
       top_n=30
   )

   # Compare across clusters
   sch.pl.plot_centrality_comparison(
       adata,
       clusters=['HSC', 'MEP'],
       metric='degree_centrality_all',
       top_n=20
   )

   # Multi-panel for specific genes
   sch.pl.plot_gene_centrality(
       adata,
       genes=['GATA1', 'SPI1', 'FLI1'],
       cluster_key='cell_type'
   )

Interaction Matrix
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Heatmap of W matrix
   sch.pl.plot_interaction_matrix(
       adata,
       cluster='HSC',
       top_n=30,
       cmap='RdBu_r'
   )

GRN Network Graph
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Full network
   sch.pl.plot_grn_network(
       adata,
       cluster='HSC',
       topn=50,
       score_size='degree_centrality_out'
   )

   # Focused subnetwork
   sch.pl.plot_grn_subset(
       adata,
       cluster='HSC',
       selected_genes=['GATA1', 'GATA2', 'KLF1', 'TAL1']
   )

Eigenvalue Spectra
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Eigenvalues in complex plane
   sch.pl.plot_eigenvalue_spectrum(adata, cluster='HSC')

   # Eigenvector components
   sch.pl.plot_eigenvector_components(
       adata,
       cluster='HSC',
       k=0,  # First eigenvector
       top_n=20
   )

   # Comprehensive grid
   sch.pl.plot_eigenanalysis_grid(adata, cluster_key='cell_type')

Applications
------------

1. **Hub Gene Identification**: Find master regulators via centrality
2. **Network Motifs**: Identify regulatory modules
3. **Network Rewiring**: Compare GRN changes across cell types
4. **Dimensionality Reduction**: Use eigenvectors for network-based embedding

See Also
--------

- :doc:`api/tools` - Network analysis functions
- :doc:`api/plotting` - Visualization functions
- :doc:`stability_analysis` - Jacobian analysis
