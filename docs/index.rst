scHopfield
==========

.. rst-class:: lead

   Energy landscapes, stability, and in-silico perturbation for single-cell gene
   regulatory dynamics.

scHopfield models a gene regulatory network as a continuous Hopfield system whose
state evolves as

.. math::

   \frac{dx}{dt} = W \, \sigma(x) - \gamma \, x + I

and turns that model, fit per cell type from RNA velocity, into a Lyapunov **energy
landscape**, **Jacobian stability** analysis, and CellOracle-style **in-silico
knockouts**, all stored back into your :class:`~anndata.AnnData`.

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 4 4 0 0

   .. grid-item-card:: :octicon:`download;1.5em;sd-mr-1` Installation
      :link: installation
      :link-type: doc
      :class-card: sd-border-1

      Install scHopfield with pip and set up an environment.

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1` Quick Start
      :link: quickstart
      :link-type: doc
      :class-card: sd-border-1

      Go from an AnnData object to an energy landscape in a dozen lines.

   .. grid-item-card:: :octicon:`book;1.5em;sd-mr-1` Tutorials
      :link: tutorial
      :link-type: doc
      :class-card: sd-border-1

      Six executed notebooks, from a first fit to combinatorial knockouts.

   .. grid-item-card:: :octicon:`code-square;1.5em;sd-mr-1` API Reference
      :link: api/index
      :link-type: doc
      :class-card: sd-border-1

      Every public function, grouped by preprocessing, inference, tools,
      dynamics, and plotting.

The whole pipeline, one call
----------------------------

.. code-block:: python

   import scHopfield as sch

   adata = sch.run_pipeline(
       adata,
       cluster_key="cell_type",
       prepare=True,        # velocity + sigmoid preprocessing
       n_top_genes=2000,    # the gene count the published fits use
       device="cuda",
       seed=0,
   )
   # -> fitted GRN, energies, Jacobian stability, and drivers,
   #    all written back into adata.

Prefer full control? Every step is an ordinary call you can run on its own, see
:doc:`quickstart`.

What you can do
---------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: :octicon:`git-branch;1.2em` Cell-type GRNs

      Infer cluster-specific interaction matrices from RNA velocity, with an
      optional prior-knowledge scaffold (e.g. a CellOracle base GRN).

   .. grid-item-card:: :octicon:`graph;1.2em` Energy landscapes

      Compute a Lyapunov energy and decompose it into interaction,
      degradation, and bias components to quantify state stability.

   .. grid-item-card:: :octicon:`pulse;1.2em` Jacobian stability

      Eigenvalue spectra at every cell state: identify unstable progenitors
      and stable terminal attractors.

   .. grid-item-card:: :octicon:`beaker;1.2em` In-silico perturbation

      Knock out or overexpress genes and score the predicted lineage shift
      against known biology.

   .. grid-item-card:: :octicon:`hubot;1.2em` Network drivers

      Rank driver transcription factors by centrality and GRN out-strength,
      per cell type.

   .. grid-item-card:: :octicon:`verified;1.2em` Reproducible

      One seed threads through every stochastic step; the same pipeline runs
      identically across datasets.

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   quickstart
   tutorial
   pipeline

.. toctree::
   :hidden:
   :caption: API Reference

   api/index

.. toctree::
   :hidden:
   :caption: About

   data_conventions
   data_sources
   faq
   changelog
   contributing
