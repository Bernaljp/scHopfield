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

   For a fully worked, executed example, from one ``run_pipeline`` call to every
   downstream analysis (energy, stability, drivers, eigenmodes, in-silico
   knockouts, and a 100% known-driver KO validation), see the
   :doc:`End-to-End Pipeline notebook <notebooks/08_end_to_end_pipeline>`.

One call
--------

.. code-block:: python

   import scHopfield as sch

   adata = sch.run_pipeline(
       adata,
       cluster_key="cell_type",
       prepare=True,        # run velocity + sigmoid preprocessing first
       n_top_genes=250,     # subset to the most dynamic genes
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
       fit_kwargs=dict(scaffold_regularization=0.1, only_TFs=True),
       device="cuda",
   )

Reproducible across datasets
----------------------------

``analyses/run_full_pipeline.py`` runs the identical pipeline over several
datasets and stores the fitted data and figures in a uniform layout:

.. code-block:: bash

   # all datasets
   python analyses/run_full_pipeline.py --device cuda --n-genes 250

   # a single dataset
   python analyses/run_full_pipeline.py --only pancreas

Each dataset produces, under ``benchmark_results/pipeline/<dataset>/``:

``adata_fitted.h5ad``
    Fitted GRN, energies, and Jacobian eigenvalues.
``summary.json``
    Per-cluster energy / stability medians, top GRN drivers, and the in-silico
    knockout impact of the top driver.
``energy_stability.png``
    Per-cluster energy depth, leading Jacobian eigenvalue, and instability count.
``top_drivers.png``
    Strongest regulators by GRN out-strength.
``perturbation_impact.png``
    Predicted per-cluster impact of knocking out the top driver.

The run demonstrates the method on both mouse and human developmental systems,
with either an unconstrained (pseudoinverse) fit or a scaffold-guided fit:

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 12 30

   * - Dataset
     - Species
     - Cells
     - Fit
     - Example top drivers
   * - Hematopoiesis (Paul 2015)
     - mouse
     - 2,671
     - scaffold
     - Myc, Nfe2, Myb
   * - Pancreas endocrinogenesis
     - mouse
     - 3,696
     - pseudoinverse
     - Rfx6, Pam, Ptprn2
   * - Murine neural crest
     - mouse
     - 6,788
     - pseudoinverse
     - Wwtr1, Rtn1, Pcdh1
   * - Human limb
     - human
     - 12,207
     - pseudoinverse
     - MEST, SHD, COBL
   * - Schwann cell development
     - mouse
     - 8,821
     - pseudoinverse
     - (see summary.json)

Each recovered driver set is dominated by regulators of the system in question
(e.g. Rfx6 for pancreatic endocrine fate, Nfe2 for the erythroid branch),
providing a face-validity check on the inferred networks.

Individual steps
----------------

If you want full control, run the steps yourself; see :doc:`quickstart` for the
long form and the :doc:`api/index` for every function.

.. seealso::

   :func:`scHopfield.run_pipeline`,
   :func:`scHopfield.build_scaffold`,
   :func:`scHopfield.prepare_dataset`,
   :func:`scHopfield.dynamics.score_ko_panel`
