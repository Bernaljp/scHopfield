Inference (sch.inf)
===================

The inference module learns cell-type-specific gene regulatory network
parameters from RNA velocity, optionally guided by a prior-knowledge scaffold.

.. currentmodule:: scHopfield.inference

GRN inference
-------------

.. autosummary::
   :toctree: generated/

   fit_interactions

Optimizer
---------

Objects behind :func:`fit_interactions`, public because a caller driving its own
training loop needs them directly.

.. autosummary::
   :toctree: generated/

   ScaffoldOptimizer
   MaskedLinearLayer
   CustomDataset

Prior-knowledge scaffolds
-------------------------

.. autosummary::
   :toctree: generated/

   fetch_base_grn
   build_scaffold

scHopfield does not distribute a base gene regulatory network. ``fetch_base_grn``
downloads one from the CellOracle repository at a pinned commit, verifies its
checksum and caches it locally. That table is distributed under CellOracle's own
license, which restricts use to non-commercial academic purposes, and not under
scHopfield's MIT license. See ``DATA_SOURCES.md`` in the repository root for the
restriction in full and for the works to cite.
