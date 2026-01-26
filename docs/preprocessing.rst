Preprocessing
=============

Sigmoid Function Fitting
-------------------------

scHopfield uses sigmoid activation functions to model gene expression:

.. math::

   \sigma(x; \theta, \alpha, \beta) = \frac{\alpha}{1 + e^{-\beta(x - \theta)}} + \text{offset}

Where:

- :math:`\theta` is the threshold parameter
- :math:`\alpha` is the amplitude (typically 1)
- :math:`\beta` is the exponent (steepness)
- offset is a baseline value

Fitting Process
~~~~~~~~~~~~~~~

The ``fit_all_sigmoids()`` function fits these parameters to each gene's expression distribution:

.. code-block:: python

   import scHopfield as sch

   # Fit sigmoid functions to selected genes
   sch.pp.fit_all_sigmoids(
       adata,
       genes=highly_variable_genes,
       spliced_key='Ms'
   )

This stores the fitted parameters in ``adata.var``:

- ``sigmoid_threshold``
- ``sigmoid_exponent``
- ``sigmoid_offset``
- ``sigmoid_mse`` (fit quality)

Computing Sigmoid Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After fitting, compute the sigmoid-transformed expression:

.. code-block:: python

   sch.pp.compute_sigmoid(adata)

This creates ``adata.layers['sigmoid']`` with transformed values.

Quality Control
---------------

Check sigmoid fit quality:

.. code-block:: python

   # Plot fit for specific genes
   sch.pl.plot_sigmoid_fit(adata, gene='GATA1')

   # Check MSE distribution
   import matplotlib.pyplot as plt
   plt.hist(adata.var['sigmoid_mse'].dropna(), bins=50)
   plt.xlabel('MSE')
   plt.ylabel('Count')
   plt.title('Sigmoid Fit Quality')
   plt.show()

Best Practices
--------------

1. **Gene Selection**: Use 50-200 highly variable genes for computational efficiency
2. **Quality Control**: Remove genes with poor sigmoid fits (high MSE)
3. **Expression Range**: Ensure genes have sufficient dynamic range
4. **Normalization**: Use log-normalized expression as input

See Also
--------

- :doc:`api/preprocessing` - API reference
- :doc:`quickstart` - Quick start guide
