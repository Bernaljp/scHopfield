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

   .. grid-item-card:: Validation ``sch.validation``
      :link: validation
      :link-type: doc

      Synthetic circuits with known ground truth, and recovery metrics.

Top-level functions
-------------------

Everything ``import scHopfield as sch`` puts on ``sch`` directly, besides the
``sch.pp`` / ``sch.inf`` / ``sch.tl`` / ``sch.dyn`` / ``sch.pl`` / ``sch.validation``
module aliases above. Each one is re-exported from the submodule it belongs to, and
is documented in full on that submodule's page.

.. currentmodule:: scHopfield

.. autosummary::
   :toctree: generated/
   :nosignatures:

   run_pipeline
   prepare_dataset
   fit_all_sigmoids
   compute_sigmoid
   fetch_base_grn
   build_scaffold
   fit_interactions
   compute_energies
   energy_embedding
   compute_umap
   simulate_trajectory
   sigmoid
   set_seed

``sch.ODESolver`` is re-exported too; it is documented as
:class:`~scHopfield.dynamics.ODESolver` on the :doc:`dynamics <dynamics>` page.

.. toctree::
   :hidden:

   workflows
   preprocessing
   inference
   tools
   dynamics
   plotting
   validation
