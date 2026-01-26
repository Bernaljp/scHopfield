FAQ
===

Frequently Asked Questions
---------------------------

Installation
~~~~~~~~~~~~

**Q: Which Python version should I use?**

A: Python 3.8 or higher is required. We recommend Python 3.10 for best compatibility.

**Q: Do I need a GPU?**

A: No, but GPU acceleration (CUDA) can speed up network inference and Jacobian computation by 10-100×.

**Q: How do I install igraph?**

A: See :doc:`installation` for platform-specific instructions. The package works fine without it, but igraph provides significant speedup for large networks.

Data Requirements
~~~~~~~~~~~~~~~~~

**Q: What data do I need?**

A: You need:
   - Single-cell RNA-seq data in AnnData format
   - RNA velocity computed (e.g., with scVelo)
   - Cell type annotations
   - Highly variable genes selected (50-200 recommended)

**Q: Can I use data without RNA velocity?**

A: No, RNA velocity is required for network inference. Use scVelo or velocyto to compute it first.

**Q: How many genes should I use?**

A: We recommend 50-200 highly variable genes for computational efficiency and interpretability.

Analysis
~~~~~~~~

**Q: How long does analysis take?**

A: On a dataset with 10,000 cells and 100 genes:
   - Network inference: 5-30 minutes (CPU) or 1-5 minutes (GPU)
   - Energy computation: <1 minute
   - Jacobian analysis: 10-60 minutes (CPU) or 2-10 minutes (GPU)

**Q: My network inference is slow. How can I speed it up?**

A: Try:
   - Use ``device='cuda'`` for GPU acceleration
   - Reduce ``n_epochs`` (1000 is usually sufficient)
   - Use fewer genes
   - Increase ``learning_rate`` slightly

**Q: How do I interpret energy values?**

A: Lower energy indicates more stable states (attractors). High energy indicates unstable or transitional states.

**Q: What does a positive Jacobian eigenvalue mean?**

A: It indicates instability in that direction - the cell state wants to move away from its current position.

Memory Issues
~~~~~~~~~~~~~

**Q: I'm running out of memory when computing Jacobians. What should I do?**

A: Use HDF5 storage:

.. code-block:: python

   sch.tl.compute_jacobians(adata, device='cuda')
   sch.tl.save_jacobians(adata, 'jacobians.h5')
   # This removes Jacobians from memory

**Q: Can I process clusters separately?**

A: Yes, process one cluster at a time to save memory.

Results
~~~~~~~

**Q: How do I save my results?**

A: All results are stored in the AnnData object. Save it with:

.. code-block:: python

   adata.write_h5ad('results.h5ad')

**Q: Where are the interaction matrices stored?**

A: In ``adata.varp['W_{cluster}']`` for each cluster.

**Q: How do I export networks for Cytoscape?**

A: Use ``get_network_links()``:

.. code-block:: python

   links = sch.tl.get_network_links(
       adata, cluster_key='cell_type', return_format='dataframe'
   )
   links.to_csv('network_edges.csv', index=False)

Troubleshooting
~~~~~~~~~~~~~~~

**Q: Import error: "No module named 'scHopfield'"**

A: Make sure scHopfield is installed:

.. code-block:: bash

   pip install -e .

**Q: "RuntimeError: CUDA out of memory"**

A: Reduce batch size, use fewer genes, or switch to CPU with ``device='cpu'``.

**Q: Network inference gives NaN values**

A: Try:
   - Reduce ``learning_rate``
   - Check that velocity values are not all zero
   - Ensure proper normalization of input data

Contributing
~~~~~~~~~~~~

**Q: How can I contribute to scHopfield?**

A: See :doc:`contributing` for guidelines.

**Q: I found a bug. Where do I report it?**

A: Open an issue on GitHub: https://github.com/Bernaljp/scHopfield/issues

Still Have Questions?
---------------------

- Check the :doc:`tutorial` for detailed examples
- See the :doc:`api/tools` for function documentation
- Ask on GitHub Issues: https://github.com/Bernaljp/scHopfield/issues
