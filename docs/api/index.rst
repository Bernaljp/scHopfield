API Reference
=============

scHopfield is organized into submodules, imported together as ``sch``:

.. code-block:: python

   import scHopfield as sch

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Preprocessing ``sch.pp``
      :link: preprocessing
      :link-type: doc

      Velocity preparation and sigmoid activation fitting.

   .. grid-item-card:: Inference ``sch.inf``
      :link: inference
      :link-type: doc

      Cell-type GRN inference and prior-knowledge scaffolds.

   .. grid-item-card:: Tools ``sch.tl``
      :link: tools
      :link-type: doc

      Energy, Jacobians, networks, embeddings, flow, perturbation scoring.

   .. grid-item-card:: Dynamics ``sch.dyn``
      :link: dynamics
      :link-type: doc

      ODE simulation and in-silico perturbation.

   .. grid-item-card:: Plotting ``sch.pl``
      :link: plotting
      :link-type: doc

      Publication-ready figures for every analysis.

   .. grid-item-card:: Workflows ``sch``
      :link: workflows
      :link-type: doc

      The high-level, reproducible end-to-end pipeline.

Top-level functions
-------------------

.. currentmodule:: scHopfield

.. autosummary::
   :toctree: generated/
   :nosignatures:

   run_pipeline
   set_seed
   build_scaffold
   prepare_dataset

.. toctree::
   :hidden:

   workflows
   preprocessing
   inference
   tools
   dynamics
   plotting
