Stability Analysis
==================

Analyze local stability of cellular states using Jacobian matrices and eigenvalue decomposition.

Overview
--------

The Jacobian matrix captures local dynamics at each cell state:

.. math::

   J = W \cdot \text{diag}\left(\frac{d\sigma}{dx}\right) - \text{diag}(\gamma)

Eigenvalues of J indicate:

- **Positive real part**: Unstable directions (repulsion)
- **Negative real part**: Stable directions (attraction)
- **Imaginary part**: Oscillatory dynamics

Computing Jacobians
-------------------

.. code-block:: python

   import scHopfield as sch

   # Compute Jacobian eigenvalues for all cells
   sch.tl.compute_jacobians(
       adata,
       cluster_key='cell_type',
       compute_eigenvectors=False,  # Set True if needed
       device='cuda'  # GPU acceleration
   )

This stores eigenvalues in ``adata.obsm['jacobian_eigenvalues']`` (n_cells × n_genes, complex).

Saving to Disk
~~~~~~~~~~~~~~

For large datasets, save to HDF5:

.. code-block:: python

   # Save to file
   sch.tl.save_jacobians(adata, 'jacobians.h5')

   # Load later
   sch.tl.load_jacobians(adata, 'jacobians.h5')

Summary Statistics
------------------

Compute stability metrics:

.. code-block:: python

   sch.tl.compute_jacobian_stats(adata)

This adds to ``adata.obs``:

- ``jacobian_eig1_real`` - Real part of leading eigenvalue
- ``jacobian_eig1_imag`` - Imaginary part of leading eigenvalue
- ``jacobian_positive_evals`` - Count of positive real eigenvalues
- ``jacobian_negative_evals`` - Count of negative real eigenvalues
- ``jacobian_trace`` - Trace of Jacobian

Rotational Dynamics
-------------------

Compute antisymmetric component:

.. code-block:: python

   sch.tl.compute_rotational_part(adata, device='cuda')

Stores ``jacobian_rotational`` in ``adata.obs`` (Frobenius norm of antisymmetric part).

Specific Partial Derivatives
-----------------------------

Compute Jacobian elements for gene pairs:

.. code-block:: python

   # Compute df_i/dx_j for specific pairs
   sch.tl.compute_jacobian_elements(
       adata,
       gene_pairs=[
           ('GATA1', 'GATA2'),
           ('FLI1', 'KLF1'),
           ('SPI1', 'GATA1')
       ],
       device='cuda'
   )

Stores as ``jacobian_df_{gene_i}_dx_{gene_j}`` in ``adata.obs``.

Visualization
-------------

Eigenvalue Spectra
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Eigenvalues in complex plane per cluster
   sch.pl.plot_jacobian_eigenvalue_spectrum(
       adata,
       cluster_key='cell_type',
       colors=cluster_colors
   )

   # Boxplots of eigenvalue components
   sch.pl.plot_jacobian_eigenvalue_boxplots(
       adata,
       cluster_key='cell_type'
   )

Stability Metrics
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Boxplots of summary statistics
   sch.pl.plot_jacobian_stats_boxplots(
       adata,
       cluster_key='cell_type',
       order=['HSC', 'MPP', 'CMP', 'GMP', 'MEP']
   )

Partial Derivatives on UMAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot specific Jacobian elements
   sch.pl.plot_jacobian_element_grid(
       adata,
       gene_pairs=[
           ('GATA1', 'GATA2'),
           ('SPI1', 'GATA1'),
           ('FLI1', 'KLF1')
       ],
       ncols=2
   )

Interpretation
--------------

**Unstable Cells** (many positive eigenvalues)
   Transitional or primed states ready to differentiate

**Stable Cells** (mostly negative eigenvalues)
   Terminal or quiescent states

**High Rotational Part**
   Cells with oscillatory dynamics or cyclic behavior

**Trace Analysis**
   - Negative trace: Overall attractive dynamics
   - Positive trace: Overall repulsive dynamics

Applications
------------

1. **Identify Transition States**: Cells with positive eigenvalues
2. **Bifurcation Analysis**: Detect decision points in differentiation
3. **Oscillatory Behavior**: Find cells with imaginary eigenvalues
4. **Regulatory Strength**: Analyze specific gene-gene interactions

Performance Tips
----------------

- Use ``device='cuda'`` for GPU acceleration (10-100× faster)
- Set ``compute_eigenvectors=False`` unless needed (saves memory)
- Save large Jacobian data to HDF5 files
- Compute only needed partial derivatives, not all elements

See Also
--------

- :doc:`api/tools` - Jacobian functions
- :doc:`api/plotting` - Visualization functions
- :doc:`energy_analysis` - Energy landscapes
