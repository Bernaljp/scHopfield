Stability Analysis
==================

.. note::
   Convert ``notebooks/04_stability_analysis.py`` to
   ``docs/notebooks/04_stability_analysis.ipynb`` using Jupytext or
   ``jupyter nbconvert``, then this page will render the full notebook.

This notebook covers:

- Computing per-cell Jacobian matrices and eigenvalue spectra
- Saving and loading Jacobians to/from HDF5 files for large datasets
- Summary statistics (trace, leading eigenvalue, positive eigenvalue count) on UMAP
- Rotational dynamics: antisymmetric part of J and oscillatory tendency
- Element-wise Jacobian analysis for specific gene regulatory pairs
  (e.g. GATA1 ↔ FLI1, CEBPA ↔ RUNX1) visualised on UMAP
