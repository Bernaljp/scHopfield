Installation
============

Prerequisites
-------------

Before using scHopfield, you need:

1. **Single-cell RNA-seq data** in AnnData format
2. **RNA velocity** computed (e.g., using `scVelo <https://scvelo.readthedocs.io/>`_)

   - ``adata.layers['Ms']`` - spliced counts
   - ``adata.layers['velocity_S']`` - RNA velocity
   - ``adata.var['gamma']`` - degradation rates

3. **Cell type annotations** (e.g., ``adata.obs['cell_type']``)
4. **Highly variable genes** selected (recommended: 50-200 genes)

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/Bernaljp/scHopfield.git
   cd scHopfield
   pip install -e .

The ``-e`` flag installs in "editable" mode, meaning changes to the source code are immediately reflected.

Standard Installation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/Bernaljp/scHopfield.git
   cd scHopfield
   pip install .

With Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For enhanced functionality (seaborn, igraph, dynamo):

.. code-block:: bash

   pip install -e ".[optional]"

For development tools:

.. code-block:: bash

   pip install -e ".[dev]"

For all features:

.. code-block:: bash

   pip install -e ".[all,dev,docs]"

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

These are automatically installed:

- **Numerical computing**: numpy >= 1.20.0, scipy >= 1.7.0, pandas >= 1.3.0
- **Visualization**: matplotlib >= 3.4.0
- **Single-cell analysis**: anndata >= 0.8.0, scanpy >= 1.9.0
- **Deep learning**: torch >= 1.9.0
- **Network analysis**: networkx >= 2.6.0
- **Dimensionality reduction**: umap-learn >= 0.5.0, scikit-learn >= 1.0.0
- **Utilities**: tqdm >= 4.62.0, h5py >= 3.0.0, hoggorm >= 0.13.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For enhanced performance and features:

**seaborn** (>= 0.11.0)
   For boxplot visualizations

**python-igraph** (>= 0.9.0)
   For 10-100× faster network centrality computation on large networks

   .. code-block:: bash

      pip install python-igraph

**dynamo-release** (>= 1.0.0)
   For RNA velocity integration

System Requirements
-------------------

- **Python**: >= 3.8
- **OS**: Linux, macOS, Windows
- **Memory**: Recommended 16GB+ RAM for large datasets
- **GPU**: Optional (CUDA-compatible GPU for faster training with ``device='cuda'``)

Verify Installation
-------------------

After installation, verify it works:

.. code-block:: python

   import scHopfield as sch
   print(sch.__version__)  # Should print: 0.1.0

   # Check available modules
   print(dir(sch))  # Should show: pp, inf, tl, pl, dyn, etc.

Troubleshooting
---------------

PyTorch Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install PyTorch separately first:

.. code-block:: bash

   # CPU version
   pip install torch --index-url https://download.pytorch.org/whl/cpu

   # CUDA version (replace cu118 with your CUDA version)
   pip install torch --index-url https://download.pytorch.org/whl/cu118

Then install scHopfield:

.. code-block:: bash

   pip install -e .

igraph Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

python-igraph requires C libraries:

**On macOS:**

.. code-block:: bash

   brew install igraph
   pip install python-igraph

**On Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get install libigraph0-dev
   pip install python-igraph

**On Windows:**

.. code-block:: bash

   pip install python-igraph
   # If that fails, download pre-built wheel from:
   # https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph

.. note::
   The package will work fine without igraph, it will just use networkx (slower for large networks)

Using Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~

If using conda, create a clean environment:

.. code-block:: bash

   conda create -n schopfield python=3.10
   conda activate schopfield
   pip install -e .

Next Steps
----------

Once installed, proceed to the :doc:`quickstart` guide to begin analyzing your data.
