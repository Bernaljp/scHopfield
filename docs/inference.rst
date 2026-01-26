Network Inference
=================

scHopfield infers gene regulatory networks (GRNs) from RNA velocity data using gradient descent optimization.

Theory
------

The Hopfield network dynamics are:

.. math::

   \frac{dx}{dt} = W \cdot \sigma(x) - \gamma \cdot x + I

Where RNA velocity approximates :math:`dx/dt`. By fitting W (interactions) and I (biases) to minimize the difference between observed and predicted velocity, we infer the GRN structure.

Basic Usage
-----------

.. code-block:: python

   import scHopfield as sch

   # Infer networks for all cell types
   sch.inf.fit_interactions(
       adata,
       cluster_key='cell_type',
       n_epochs=1000,
       device='cuda'  # or 'cpu'
   )

This learns cluster-specific interaction matrices stored in ``adata.varp['W_{cluster}']``.

Parameters
----------

Key parameters to tune:

**n_epochs** (int, default=1000)
   Number of training epochs. Increase for better convergence.

**learning_rate** (float, default=0.01)
   Learning rate for gradient descent. Decrease if training is unstable.

**device** (str, default='cpu')
   Use 'cuda' for GPU acceleration (10-100× faster).

**refit_gamma** (bool, default=False)
   Whether to refit degradation rates per cluster.

**l1_reg** (float, default=0.0)
   L1 regularization for sparsity.

**l2_reg** (float, default=0.0)
   L2 regularization to prevent overfitting.

Advanced Options
----------------

Regularization
~~~~~~~~~~~~~~

Add sparsity constraints:

.. code-block:: python

   sch.inf.fit_interactions(
       adata,
       cluster_key='cell_type',
       n_epochs=2000,
       l1_reg=0.01,  # L1 for sparsity
       l2_reg=0.001,  # L2 for regularization
       device='cuda'
   )

Custom Loss Function
~~~~~~~~~~~~~~~~~~~~

Coming soon - support for custom loss functions.

Monitoring Training
-------------------

Check convergence by examining loss values:

.. code-block:: python

   # Access training history (if stored)
   models = adata.uns['scHopfield']['models']
   for cluster, model in models.items():
       print(f"{cluster}: final loss = {model.loss_history[-1]}")

Validation
----------

Validate inferred networks:

.. code-block:: python

   # Compute reconstructed velocity
   sch.tl.compute_reconstructed_velocity(adata)

   # Validate against observed velocity
   sch.tl.validate_velocity(adata)

See Also
--------

- :doc:`api/inference` - API reference
- :doc:`network_analysis` - Analyzing inferred networks
- :doc:`energy_analysis` - Energy landscape analysis
