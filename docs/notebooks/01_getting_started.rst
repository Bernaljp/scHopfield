Getting Started
===============

.. note::
   Convert ``notebooks/01_getting_started.py`` to
   ``docs/notebooks/01_getting_started.ipynb`` using Jupytext or
   ``jupyter nbconvert``, then this page will render the full notebook.

This notebook covers:

- Data loading and quality control (removing NaN-velocity genes)
- Sigmoid function fitting to gene expression distributions
- Network inference without scaffold constraints
- Scaffold-constrained inference using a CellOracle mouse scATAC-seq GRN
- Saving and loading the fitted model to/from an HDF5 file

**Key parameters**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``N_EPOCHS``
     - 1000
     - Training epochs for gradient-descent inference
   * - ``BATCH_SIZE``
     - 128
     - Mini-batch size
   * - ``SCAFFOLD_REGULARIZATION``
     - 1e-2
     - L1/L2 penalty weight for scaffold-guided sparsity
   * - ``W_THRESHOLD``
     - 1e-12
     - Absolute value threshold below which W entries are zeroed
