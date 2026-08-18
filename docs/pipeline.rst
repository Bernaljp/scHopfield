End-to-end pipeline
===================

scHopfield ships a single high-level entry point, :func:`scHopfield.run_pipeline`,
that runs the canonical analysis in the same order every time:

.. code-block:: text

   prepare -> (gene subset) -> (scaffold) -> fit GRN -> energies
           -> Jacobians -> Jacobian stats -> network centrality

Every step is an ordinary ``sch.*`` call, so nothing here is a black box; the
wrapper just wires them together with sensible defaults and records what it did in
``adata.uns['scHopfield_pipeline']``.

.. tip::

   For fully worked, executed examples, see the :doc:`tutorials <tutorial>`. They
   run this pipeline on pancreatic endocrinogenesis and then read the fit: energy,
   stability, network structure, single-gene knockouts and double knockouts.

One call
--------

.. code-block:: python

   import scHopfield as sch

   adata = sch.run_pipeline(
       adata,
       cluster_key="cell_type",
       prepare=True,        # run velocity + sigmoid preprocessing first
       n_top_genes=2000,    # the gene count the published fits use
       device="cuda",
       seed=0,
   )

   # energies, stability, and the GRN are now in adata:
   adata.obs["energy_total"]           # Lyapunov energy per cell
   adata.obs["jacobian_eig1_real"]     # leading Jacobian eigenvalue (stability)
   adata.varp["W_<cluster>"]           # cell-type-specific interaction matrix

With a prior-knowledge scaffold
-------------------------------

Pass a CellOracle-style base GRN through :func:`scHopfield.build_scaffold` to
guide the fit (regularizing the free interactions toward known edges):

.. code-block:: python

   import pandas as pd

   base = pd.read_parquet("base_GRN.parquet")
   scaffold = sch.inf.build_scaffold(adata, base)

   adata = sch.run_pipeline(
       adata,
       cluster_key="cell_type",
       scaffold=scaffold.values.T,        # fit_interactions expects W[target, regulator]
       device="cuda",
   )

Where the paper's runs live
---------------------------

The code that produces the figures in the paper is in ``reproducibility/`` in this
repository, together with the dataset preparation steps. It is separate from the
package on purpose: the package is the method, and ``reproducibility/`` is one set
of runs of it.

The scaffold is the one input the package cannot supply. ``fetch_base_grn``
downloads a CellOracle base network from a pinned commit and caches it locally.
That table carries CellOracle's license, which restricts use to non-commercial
academic purposes, and not scHopfield's MIT license; ``DATA_SOURCES.md`` gives the
restriction in full and the works to cite.

Individual steps
----------------

If you want full control, run the steps yourself; see :doc:`quickstart` for the
long form and the :doc:`api/index` for every function.

.. seealso::

   :func:`scHopfield.run_pipeline`,
   :func:`scHopfield.build_scaffold`,
   :func:`scHopfield.prepare_dataset`,
   :func:`scHopfield.dynamics.score_ko_panel`
